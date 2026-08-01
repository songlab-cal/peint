"""PEINT transformer models for protein evolutionary modeling.

This module provides encoder-decoder transformer architectures built on top of
frozen ESM2 models. The models predict target sequences conditioned on source
sequences and evolutionary time.

Classes:
    _PeintTransformerBase: Abstract base class with shared initialization and methods
    PeintTransformer: Flash Attention variant for training
    PeintGenerator: Flash Attention with KV caching for autoregressive generation
    PeintEvaluator: Flash Attention with encoder caching for likelihood evaluation
    PeintTransformerVanilla: Standard attention for interpretability (returns attention weights)
    PeintLightningModule: PyTorch Lightning wrapper for training

Attention Mask Convention:
    All public methods expect masks where True=padding (positions to ignore).
    This follows PyTorch's convention. Flash Attention variants flip masks internally
    (True=attend), documented at each flip location.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from esm.modules import RobertaLMHead

from protevo.models._transformer_modules import (
    GeometricTimeEmbedder,
    FlashMHADecoderBlock,
    FlashMHAEncoderBlock,
    ESMDecoderBlock,
    ESMEncoderBlock,
    KV_CachedFlashMHADecoderBlock,
    EncoderCachedFlashMHADecoderBlock,
    FFN_EXPANSION_FACTOR,
)
from protevo.models._config import PeintConfig
from protevo.inference._batching import (
    DEFAULT_MAX_TOKENS,
    fixed_size_batches,
    largest_batch_first,
    token_budget_batches,
)
from protevo.inference._tokenize import build_token_lut, encode_batch, pad_encoded

from protevo.utils import amino_acids

# Model constants
DEFAULT_MAX_SEQ_LEN = 1022  # ESM2's max sequence length minus special tokens
STANDARD_STATES = list(amino_acids) + ['<eos>']

# How often the generation loop checks whether every sequence has emitted <eos>.
# The check reads a CUDA bool tensor into Python, which is a hard device sync;
# amortizing it over this many steps keeps the decode loop asynchronous. Any
# tokens produced after the true stopping point are past <eos> and get truncated
# by decode_sequences, so the generated strings do not depend on this value.
EOS_CHECK_INTERVAL = 16


def _config_from_kwargs(
    embed_dim: int,
    num_heads: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    max_len: int,
    **kwargs
) -> PeintConfig:
    """Create PeintConfig from legacy kwargs for backward compatibility."""
    return PeintConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_seq_len=max_len,
        dropout_p=kwargs.get('dropout_p', 0.0),
        use_attention_bias=kwargs.get('use_attention_bias', True),
        label_smoothing=kwargs.get('label_smoothing', 0.0),
        max_encoder_seq_len=kwargs.get('max_encoder_seq_len', 1024),
        max_decoder_seq_len=kwargs.get('max_decoder_seq_len', 1024),
        weight_decay=kwargs.get('weight_decay', 0.0),
        # Ablation axes (default to published behavior when absent from kwargs,
        # e.g. loading an older checkpoint whose hparams predate these fields).
        mlm_weight=kwargs.get('mlm_weight', 1.0),
        encoder_backbone=kwargs.get('encoder_backbone', 'ESM2-150M'),
        esm_finetune_mode=kwargs.get('esm_finetune_mode', 'frozen'),
        lora_rank=kwargs.get('lora_rank', None),
        lora_alpha=kwargs.get('lora_alpha', None),
        architecture=kwargs.get('architecture', 'encoder_decoder'),
    )


def _is_esmc_backbone(module) -> bool:
    """True if ``module`` is an ESM-C model (esm3 package).

    Detected by class identity rather than ``isinstance`` so we never import the
    optional esm3 package just to check an ESM2 backbone (keeps ESM2 the default
    and ESM2-only environments/tests fast).
    """
    cls = type(module)
    return cls.__name__ == "ESMC" and cls.__module__.split(".")[0] in ("esm", "esm3")


###################################
# Abstract Base Class             #
###################################

class _PeintTransformerBase(nn.Module, ABC):
    """Abstract base class for all PEINT transformer variants.

    Provides shared initialization logic and helper methods. Subclasses must
    implement layer factory methods to create appropriate encoder/decoder layers.

    Attributes:
        _flip_attention_masks: If True, flip mask convention for Flash Attention
            (public API uses True=padding, Flash uses True=attend)
    """

    _flip_attention_masks: bool = False

    def __init__(
        self,
        esm_model,
        esm_vocab,
        embed_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_len: int = DEFAULT_MAX_SEQ_LEN,
        **kwargs
    ):
        super().__init__()

        # Create typed config from parameters (validates internally)
        self.config = _config_from_kwargs(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_len=max_len,
            **kwargs
        )

        # Store frequently accessed config values as instance attributes
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.num_encoder_layers = self.config.num_encoder_layers
        self.num_decoder_layers = self.config.num_decoder_layers
        self.max_len = self.config.max_seq_len
        self.dropout_p = self.config.dropout_p
        self.use_bias = self.config.use_attention_bias

        # Pretrained backbone. Behaviour by esm_finetune_mode:
        #   "frozen" (default, published): backbone fully frozen — bit-identical to
        #            the released model; NO import of the LoRA module.
        #   "lora":   base frozen, trainable low-rank adapters injected (A5 ablation);
        #            the adapters start as zero so the model matches "frozen" at init.
        #   "full":   backbone left fully trainable.
        self.esm = esm_model
        self.vocab = esm_vocab
        self.esm.eval()

        # Inference-time lookups, built lazily and then reused across calls.
        # Plain attributes rather than buffers/submodules so state_dict is
        # unchanged and existing checkpoints keep loading.
        self._inv_vocab = None       # token id -> token string, for decode_sequences
        self._zero_idx = None        # non-amino-acid token ids banned during sampling
        self._token_lut_cache = None  # char -> token id table, see _token_lut

        # PEINT reads only the backbone's hidden representations (see
        # _compute_language_model_representations), never its LM-head logits, and
        # it holds its own copy of the head for the encoder-side MLM logits. Tell
        # our ESM2 wrappers to skip that dead computation. Guarded by hasattr so a
        # stock fair-esm backbone is left untouched.
        if hasattr(self.esm, 'emb_layer_norm_after') and hasattr(self.esm, 'lm_head'):
            self.esm.return_logits = False
        mode = self.config.esm_finetune_mode
        if mode in ("frozen", "lora"):
            self.esm.requires_grad_(False)
        if mode == "lora":
            # Imported lazily so the default path has no LoRA dependency.
            from protevo.models._lora import apply_lora_to_backbone
            apply_lora_to_backbone(
                self.esm,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )

        # Loss functions
        self.y_criterion = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=self.vocab.padding_idx,
            label_smoothing=self.config.label_smoothing
        )
        self.x_criterion = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=self.vocab.padding_idx
        )

        # Which backbone family are we wrapping? (ESM2 vs ESM-C have different
        # embedding / LM-head / representation APIs.)
        self._is_esmc = _is_esmc_backbone(self.esm)

        # Token embeddings (initialized from the backbone, frozen). ESM2 exposes
        # `embed_tokens`; ESM-C exposes `embed`.
        if self._is_esmc:
            assert embed_dim == 960, (
                "ESM-C support currently requires embed_dim=960 (esmc_300m)."
            )
            self.embedding = nn.Embedding(self.esm.embed.weight.shape[0], embed_dim)
            self.embedding.load_state_dict(self.esm.embed.state_dict())
        else:
            self.embedding = nn.Embedding(len(self.vocab), embed_dim)
            self.embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        self.embedding.requires_grad_(False)

        # Time embedding
        self.time_embedding = GeometricTimeEmbedder(frequency_embedding_size=embed_dim)

        # Create encoder and decoder layers via factory methods
        self.enc_layers = self._create_encoder_layers()
        self.dec_layers = self._create_decoder_layers()

        # Language model head (initialized from the backbone, frozen). ESM2 uses a
        # RobertaLMHead tied to the token embedding; ESM-C uses its `sequence_head`
        # (a RegressionHead). NOTE: ESM-C's sequence_head is 64-wide (reserved slots
        # beyond the 33 real tokens in the vocab), so we match that width — the real
        # token ids (0..len(vocab)-1) are a subset of those 64 outputs.
        if self._is_esmc:
            # RegressionHead is the upstream ESM-C head factory from the esm3 package
            # (esm3.layers.regression_head: Linear->GELU->LayerNorm->Linear), not code
            # we wrote. Rebuilding it here and loading esm.sequence_head's weights
            # mirrors the ESM-C integration in the research repo (protein-evolution
            # protevo/models/_transformer.py). esmc_300m's sequence_head is 64-wide
            # (guarded by embed_dim==960 above).
            try:
                from esm.layers.regression_head import RegressionHead
            except ImportError:
                from esm3.layers.regression_head import RegressionHead
            ESMC_300M_SEQUENCE_HEAD_DIM = 64
            self.lm_head = RegressionHead(self.embed_dim, ESMC_300M_SEQUENCE_HEAD_DIM)
            self.lm_head.load_state_dict(self.esm.sequence_head.state_dict())
        else:
            self.lm_head = RobertaLMHead(
                embed_dim=self.embed_dim,
                output_dim=len(self.vocab),
                weight=self.embedding.weight
            )
            self.lm_head.load_state_dict(self.esm.lm_head.state_dict())
        self.lm_head.requires_grad_(False)

    @abstractmethod
    def _create_encoder_layers(self) -> nn.ModuleList:
        """Create encoder layers. Must return nn.ModuleList."""
        pass

    @abstractmethod
    def _create_decoder_layers(self) -> nn.ModuleList:
        """Create decoder layers. Must return nn.ModuleList."""
        pass

    def _compute_language_model_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Compute backbone representations for the source sequence.

        Args:
            x: Source tokens [B, L] with CLS and EOS from dataloader

        Returns:
            Final hidden state from the backbone [B, L, D]
        """
        if self._is_esmc:
            # ESM-C takes token ids directly and returns an object with `.embeddings`
            # (the final hidden state); it handles padding internally.
            return self.esm(x).embeddings
        res = self.esm(
            x,
            repr_layers=[self.esm.num_layers],
            need_head_weights=False
        )
        return res['representations'][self.esm.num_layers]

    def _prepare_decoder_input(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Prepare decoder input by adding time embedding.

        Args:
            y: Target tokens [B, L]
            t: Evolutionary time [B, 1]

        Returns:
            Decoder hidden states [B, L, D] with time added
        """
        h_y = self.embedding(y)
        ht = self.time_embedding(t)
        ht = ht.expand_as(h_y)
        return h_y + ht

    def _prepare_attention_masks(
        self,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> tuple:
        """Prepare attention masks, flipping if needed for Flash Attention.

        Args:
            x_attn_mask: Source mask, True=padding [B, L_src]
            y_attn_mask: Target mask, True=padding [B, L_tgt]

        Returns:
            Tuple of (x_attn_mask, y_attn_mask), flipped if _flip_attention_masks
        """
        if self._flip_attention_masks:
            # Flip: input True=padding -> Flash Attention True=attend
            return ~x_attn_mask, ~y_attn_mask
        return x_attn_mask, y_attn_mask

    def decode_sequences(self, decoded: torch.Tensor) -> List[str]:
        """Convert decoded token tensors to amino acid strings.

        Args:
            decoded: [B, L] tensor of decoded token indices

        Returns:
            List of amino acid sequence strings (without CLS, truncated at EOS)
        """
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in self.vocab.to_dict().items()}
        inv_vocab = self._inv_vocab

        # One device->host transfer for the whole batch. The original did a
        # `.item()` per token on a CUDA tensor, i.e. B x L individual syncs
        # (~38k for a batch of 64 at length 600) after every generate() call.
        decoded_rows = decoded[:, 1:].tolist()  # drop cls

        output_sequences = []
        for row in decoded_rows:
            decoded_str = ''.join([inv_vocab.get(p) for p in row])
            eos_idx = decoded_str.find('<eos>')
            if eos_idx != -1:
                decoded_str = decoded_str[:eos_idx]
            output_sequences.append(decoded_str)
        return output_sequences

    def _prepare_generation(self, x: torch.Tensor, device: torch.device) -> tuple:
        """Prepare tensors for generation loop.

        Args:
            x: Source tokens [B, L]
            device: Target device

        Returns:
            Tuple of (batch_size, x_attn_mask, y_decoded, eos_reached, zero_idx)
        """
        batch_size = x.size(0)
        x_attn_mask = x.eq(self.vocab.padding_idx)
        y_decoded = torch.tensor([self.vocab.cls_idx]).unsqueeze(0).repeat(batch_size, 1).to(device)
        eos_reached = torch.zeros(batch_size, dtype=torch.bool).to(device)
        # Built once and kept on the device. It used to be rebuilt per call and
        # left on the CPU, so `logits[..., zero_idx] = -inf` copied it host->device
        # on every generated token.
        if self._zero_idx is None:
            self._zero_idx = torch.tensor([
                self.vocab.get_idx(tok)
                for tok in self.vocab.all_toks
                if tok not in STANDARD_STATES
            ])
        zero_idx = self._zero_idx.to(device)
        self._zero_idx = zero_idx
        return batch_size, x_attn_mask, y_decoded, eos_reached, zero_idx

    def evaluate_transition_logits(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the model to get y logits for likelihood computation.

        Args:
            x: [B, L] source sequence tokens
            y: [B, L] target sequence tokens
            t: [B, 1] evolutionary time
            x_attn_mask: [B, L] mask for x, True=padding
            y_attn_mask: [B, L] mask for y, True=padding

        Returns:
            y_logits tensor for computing loss/likelihood
        """
        with torch.no_grad():
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                result = self(x, y, t, x_attn_mask, y_attn_mask)
                # Handle both 2-value (Flash) and 5-value (Vanilla) returns
                y_logits = result[1] if isinstance(result, tuple) else result
        return y_logits

    def encode_sequences(self, sequences: List[str], targets: bool = False) -> tuple:
        """Encode sequences into padded token tensors.

        Args:
            sequences: List of amino acid strings
            targets: If True, also return target tokens (input shifted for
                teacher-forced likelihood evaluation)

        Returns:
            Tuple of (padded_inputs, padded_targets or None)
        """
        return encode_batch(
            sequences, self.vocab, lut=self._token_lut, targets=targets
        )

    @property
    def _token_lut(self):
        """256-entry char -> token-id table, built once per model.

        fair-esm's ``Alphabet.encode`` costs ~2 ms per sequence in pure Python;
        homology search calls ``encode_sequences`` on the whole query set once per
        reference, so that is O(N^2) tokenization on the critical path of an
        all-vs-all run. The table reproduces ``encode`` exactly - see
        ``protevo.inference._tokenize``.
        """
        if self._token_lut_cache is None:
            self._token_lut_cache = build_token_lut(self.vocab)
        return self._token_lut_cache

    def _likelihood_logits(
        self,
        batch_idx: int,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Target logits for one likelihood batch.

        Default (uncached) path used by every model type. Precision follows
        ``evaluate_transition_logits``, which is overridden per model type
        (Flash variants use bfloat16, Vanilla uses fp32). ``PeintEvaluator``
        overrides this to add encoder caching.
        """
        return self.evaluate_transition_logits(x, y, t, x_attn_mask, y_attn_mask)

    def _reset_likelihood_cache(self) -> None:
        """Hook to clear per-call caches after likelihood evaluation (no-op by default)."""
        pass

    @torch.no_grad()
    def evaluate_likelihood(
        self,
        x: str,
        y: List[str],
        t: List[float],
        device: torch.device,
        batch_size: int = 128,
        y_tokens: Optional[Sequence[np.ndarray]] = None,
        pack_by_length: bool = False,
        max_tokens: Optional[int] = None,
    ) -> np.ndarray:
        """Evaluate likelihood of target sequences given a source sequence.

        Works for all model variants regardless of Flash Attention: Flash models
        run in bfloat16, Vanilla runs in fp32, and ``PeintEvaluator`` additionally
        caches the encoder across batches for speed.

        Args:
            x: Single source sequence string
            y: List of target sequence strings
            t: List of evolutionary times (same length as y)
            device: Target device
            batch_size: Batch size for evaluation
            y_tokens: Optional pre-tokenized targets (one array per entry in ``y``,
                as returned by ``protevo.inference.encode_all``). Lets a caller that
                scores the same corpus against many references — homology
                all-vs-all — pay tokenization once instead of once per reference.
            pack_by_length: Opt-in. Group targets of similar length instead of
                consuming them in input order, so batches carry far less padding
                and short sequences pack more rows per batch. Results are still
                returned in the order of ``y``.
                **Off by default, and worth measuring before you turn it on**: it
                gained 3-5% on likelihood and nothing on homology, because
                flash-attention already unpads internally. It was expected to
                perturb scores but measured bit-identical; see
                ``protevo.inference._batching``.
                Note ``batch_size`` is ignored when this is set — ``max_tokens``
                becomes the thing that bounds a batch.
            max_tokens: Padded-position budget per batch when ``pack_by_length`` is
                set. Defaults to ``protevo.inference._batching.DEFAULT_MAX_TOKENS``.

        Returns:
            Array of mean per-residue negative log-likelihoods (one per target)
        """
        assert len(t) == len(y), "Time and sequences must be the same length"
        if y_tokens is not None:
            assert len(y_tokens) == len(y), "y_tokens must align with y"

        n = len(y)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        if pack_by_length:
            lengths = ([len(tok) for tok in y_tokens] if y_tokens is not None
                       else [len(seq) for seq in y])
            # max_tokens alone governs the batch, so short sequences get *more*
            # rows rather than the same 32. Capping rows at batch_size here would
            # leave the batch count unchanged and reduce only intra-batch padding,
            # which measured as no speedup at all: with flash-attention already
            # unpadding internally, this workload is launch-bound rather than
            # padded-FLOP-bound, so the win has to come from fewer, fuller batches.
            batches = largest_batch_first(token_budget_batches(
                lengths,
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
                max_batch_size=None,
            ))
        else:
            batches = fixed_size_batches(n, batch_size)

        encoded_x, _ = self.encode_sequences([x])
        # Uploaded once and then broadcast, rather than materializing B copies on
        # the host and re-sending them for every batch.
        encoded_x = encoded_x.to(device)

        # Scattered back into corpus order, so the caller sees the order of ``y``
        # whichever batching produced it.
        out = None

        for ordinal, idxs in enumerate(tqdm(batches)):
            if y_tokens is not None:
                y_encoded, y_targets = pad_encoded(
                    [y_tokens[j] for j in idxs], self.vocab, targets=True
                )
            else:
                y_batch = [y[j] for j in idxs]
                y_encoded, y_targets = self.encode_sequences(y_batch, targets=True)
            y_encoded = y_encoded.to(device)
            y_targets = y_targets.to(device)
            y_attn_mask = y_encoded.eq(self.vocab.padding_idx)

            x_encoded = encoded_x.expand(y_encoded.size(0), -1)
            x_attn_mask = x_encoded.eq(self.vocab.padding_idx)

            # Indexing rather than slicing, since batches need not be contiguous.
            # Dtype is unaffected: torch.tensor infers float32 from Python floats
            # and float64 from numpy scalars either way, so a list `t` and an
            # ndarray `t` both land on the dtype the old slicing produced.
            times = torch.tensor([t[j] for j in idxs]).unsqueeze(-1).to(device)

            # ordinal, not the slice offset: _likelihood_logits only tests
            # `> 0` to decide whether to reuse the cached encoder, and with
            # variable-width batches there is no slice offset to pass.
            logits = self._likelihood_logits(
                ordinal, x_encoded, y_encoded, times, x_attn_mask, y_attn_mask
            )

            ll = nn.functional.cross_entropy(
                logits.float().transpose(1, 2),
                y_targets,
                ignore_index=self.vocab.padding_idx,
                reduction='none'
            )

            non_pad = y_targets.ne(self.vocab.padding_idx)
            ll = (ll * non_pad).sum(dim=-1) / non_pad.sum(dim=-1)

            ll_np = ll.cpu().numpy()
            if out is None:
                out = np.empty(n, dtype=ll_np.dtype)
            out[idxs] = ll_np

        self._reset_likelihood_cache()
        return out.squeeze()


######################################
# Flash Attention Transformer Models #
######################################

class PeintTransformer(_PeintTransformerBase):
    """PEINT encoder-decoder transformer with Flash Attention.

    The ESM2 model encodes the source sequence, and its final hidden representation
    feeds into an encoder/decoder stack. The decoder autoregressively predicts
    target sequences conditioned on source sequence and evolutionary time.

    Architecture:
        - Frozen ESM2 encoder for pretrained protein representations
        - Trainable encoder/decoder transformer layers with Flash Attention
        - Time encoded via geometric embeddings
        - Rotary position embeddings (RoPE) in attention layers
    """

    _flip_attention_masks = True

    def _create_encoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            FlashMHAEncoderBlock(
                embed_dim=self.embed_dim,
                ffn_embed_dim=FFN_EXPANSION_FACTOR * self.embed_dim,
                attention_heads=self.num_heads,
                add_bias_kv=False,
                dropout_p=self.dropout_p,
                use_bias=self.use_bias,
                layer_idx=l
            ) for l in range(self.num_encoder_layers)
        ])

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            FlashMHADecoderBlock(
                embed_dim=self.embed_dim,
                ffn_embed_dim=FFN_EXPANSION_FACTOR * self.embed_dim,
                attention_heads=self.num_heads,
                add_bias_kv=False,
                dropout_p=self.dropout_p,
                use_bias=self.use_bias,
                layer_idx=l
            ) for l in range(self.num_decoder_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> tuple:
        """Forward pass through encoder-decoder transformer.

        Args:
            x: Source sequence tokens [B, L_src]
            y: Target sequence tokens [B, L_tgt]
            t: Evolutionary time [B, 1]
            x_attn_mask: Attention mask for source, True=padding [B, L_src]
            y_attn_mask: Attention mask for target, True=padding [B, L_tgt]

        Returns:
            Tuple of (x_logits, y_logits):
                x_logits: Logits for source sequence reconstruction [B, L_src, vocab_size]
                y_logits: Logits for target sequence prediction [B, L_tgt, vocab_size]
        """
        h_y = self._prepare_decoder_input(y, t)
        h_x = self._compute_language_model_representations(x)
        x_attn_mask, y_attn_mask = self._prepare_attention_masks(x_attn_mask, y_attn_mask)

        if self.num_encoder_layers == 0:
            # Ablation "remove extra encoder layers": the frozen backbone output is
            # used directly as fixed decoder memory; every decoder layer cross-
            # attends to it (there are no encoder layers to refine h_x).
            for dec_layer in self.dec_layers:
                h_y = dec_layer(
                    x=h_y,
                    y=h_x,
                    x_padding_mask=y_attn_mask,
                    y_padding_mask=x_attn_mask
                )
        else:
            for i, enc_layer in enumerate(self.enc_layers):
                h_x = enc_layer(x=h_x, x_padding_mask=x_attn_mask)

                if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                    idx = self.num_decoder_layers - self.num_encoder_layers + i
                    dec_layer = self.dec_layers[idx]
                    h_y = dec_layer(
                        x=h_y,
                        y=h_x,
                        x_padding_mask=y_attn_mask,
                        y_padding_mask=x_attn_mask
                    )

        x_logits = self.lm_head(h_x)
        y_logits = self.lm_head(h_y)

        return x_logits, y_logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        max_decode_steps: int,
        device: torch.device,
        temperature: float = 1.0,
        p: float = 1.0
    ) -> List[str]:
        """Generate sequences using nucleus sampling.

        Args:
            x: Source tokens [B, L]
            t: Evolutionary time [B, 1]
            max_decode_steps: Maximum generation length
            device: Target device
            temperature: Sampling temperature (default 1.0)
            p: Nucleus sampling parameter (default 1.0, meaning full sampling)

        Returns:
            List of generated amino acid sequences
        """
        _, x_attn_mask, y_decoded, eos_reached, zero_idx = self._prepare_generation(x, device)

        for _ in range(max_decode_steps):
            y_attn_mask = y_decoded.eq(self.vocab.padding_idx)
            _, logits = self(x, y_decoded, t, x_attn_mask, y_attn_mask)

            logits = logits[:, -1, :] / temperature
            logits[..., zero_idx] = -np.inf

            next_tok = sampling_function(logits, p=p)
            y_decoded = torch.cat([y_decoded, next_tok], dim=1)

            eos_reached |= (next_tok.squeeze(-1) == self.vocab.eos_idx)
            if eos_reached.all():
                break

        return self.decode_sequences(y_decoded)


class PeintGenerator(PeintTransformer):
    """PEINT transformer with KV caching for efficient autoregressive generation.

    Inherits from PeintTransformer but uses cached decoder layers that store
    key-value pairs across generation steps, avoiding redundant computation.
    """

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            KV_CachedFlashMHADecoderBlock(
                embed_dim=self.embed_dim,
                ffn_embed_dim=FFN_EXPANSION_FACTOR * self.embed_dim,
                attention_heads=self.num_heads,
                add_bias_kv=False,
                dropout_p=self.dropout_p,
                use_bias=self.use_bias,
                layer_idx=l,
                max_encoder_seq_len=self.config.max_encoder_seq_len,
                max_decoder_seq_len=self.config.max_decoder_seq_len
            ) for l in range(self.num_decoder_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with optional KV caching for efficient generation.

        Args:
            x_attn_mask: True=padding, flipped internally for Flash Attention
            y_attn_mask: True=padding, flipped internally for Flash Attention
            use_cache: If True, use cached encoder outputs (call with full sequence first)

        Returns:
            y_logits: Target sequence logits [B, L_tgt, vocab_size]
        """
        h_y = self._prepare_decoder_input(y, t)
        x_attn_mask, y_attn_mask = self._prepare_attention_masks(x_attn_mask, y_attn_mask)

        if not use_cache:
            h_x = self._compute_language_model_representations(x)

            if self.num_encoder_layers == 0:
                # No extra encoder: every decoder layer cross-attends to (and
                # caches) the frozen backbone representation as fixed memory.
                for dec_layer in self.dec_layers:
                    h_y = dec_layer(
                        x=h_y,
                        y=h_x,
                        x_padding_mask=y_attn_mask,
                        y_padding_mask=x_attn_mask
                    )
            else:
                for i, enc_layer in enumerate(self.enc_layers):
                    h_x = enc_layer(x=h_x, x_padding_mask=x_attn_mask)

                    if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                        idx = self.num_decoder_layers - self.num_encoder_layers + i
                        dec_layer = self.dec_layers[idx]
                        h_y = dec_layer(
                            x=h_y,
                            y=h_x,
                            x_padding_mask=y_attn_mask,
                            y_padding_mask=x_attn_mask
                        )
        else:
            for dec_layer in self.dec_layers:
                h_y = dec_layer(
                    x=h_y,
                    y=None,
                    x_padding_mask=y_attn_mask,
                    y_padding_mask=x_attn_mask
                )

        return self.lm_head(h_y)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        max_decode_steps: int,
        device: torch.device,
        temperature: float = 1.0,
        p: float = 1.0
    ) -> List[str]:
        """Generate sequences using KV-cached decoding with nucleus sampling.

        Args:
            x: Source tokens [B, L]
            t: Evolutionary time [B, 1]
            max_decode_steps: Maximum generation length
            device: Target device
            temperature: Sampling temperature (default 1.0)
            p: Nucleus sampling parameter (default 1.0)

        Returns:
            List of generated amino acid sequences
        """
        batch_size, x_attn_mask, y_decoded, eos_reached, zero_idx = self._prepare_generation(x, device)

        # Initialize KV caches
        for dec_layer in self.dec_layers:
            dec_layer.self_attn.init_kv_cache(batch_size, max_decode_steps)

        # First forward pass to fill KV cache
        y_attn_mask = y_decoded.eq(self.vocab.padding_idx)
        logits = self.forward(x, y_decoded, t, x_attn_mask, y_attn_mask, use_cache=False)

        last_step = max_decode_steps - 2
        for step in range(max_decode_steps - 1):
            logits = logits[:, -1, :] / temperature
            logits[..., zero_idx] = -np.inf

            next_token = sampling_function(logits, p=p)

            y_decoded = torch.cat([y_decoded, next_token], dim=1)
            eos_reached |= (next_token.squeeze(-1) == self.vocab.eos_idx)

            # The original loop forwarded after sampling on every iteration,
            # including the last one, whose logits were then discarded. Break
            # before that final wasted decoder pass.
            if step == last_step:
                break

            # `eos_reached.all()` forces a GPU->CPU sync, and running it on every
            # token serializes the whole decode loop against the CUDA queue.
            # Checking periodically costs at most EOS_CHECK_INTERVAL - 1 extra
            # steps, and those tokens are all past <eos>, which decode_sequences
            # truncates - so the returned strings are unchanged.
            if (step + 1) % EOS_CHECK_INTERVAL == 0 and eos_reached.all():
                break

            y_attn_mask = next_token.eq(self.vocab.padding_idx)
            logits = self.forward(x, next_token, t, x_attn_mask, y_attn_mask, use_cache=True)

        self._reset_kv_cache()
        return self.decode_sequences(y_decoded)

    def _reset_kv_cache(self):
        """Reset KV caches in all decoder layers."""
        for dec_layer in self.dec_layers:
            dec_layer.cross_attn.kv_cache = None
            dec_layer.self_attn.reset_kv_cache()


class PeintEvaluator(PeintTransformer):
    """PEINT transformer with encoder caching for efficient likelihood evaluation.

    Caches encoder KV pairs so that the same source sequence can be evaluated
    against many target sequences efficiently (e.g., for beam search or
    likelihood computation over candidate sequences).
    """

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            EncoderCachedFlashMHADecoderBlock(
                embed_dim=self.embed_dim,
                ffn_embed_dim=FFN_EXPANSION_FACTOR * self.embed_dim,
                attention_heads=self.num_heads,
                add_bias_kv=False,
                dropout_p=self.dropout_p,
                use_bias=self.use_bias,
                layer_idx=l
            ) for l in range(self.num_decoder_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        """Forward pass with encoder caching for efficient likelihood evaluation.

        Args:
            x_attn_mask: True=padding, flipped internally for Flash Attention
            y_attn_mask: True=padding, flipped internally for Flash Attention
            use_cache: If True, reuse cached encoder KV (for evaluating many y sequences)

        Returns:
            y_logits: Target sequence logits [B, L_tgt, vocab_size]
        """
        h_y = self._prepare_decoder_input(y, t)
        x_attn_mask, y_attn_mask = self._prepare_attention_masks(x_attn_mask, y_attn_mask)

        if not use_cache:
            h_x = self._compute_language_model_representations(x)

            if self.num_encoder_layers == 0:
                # No extra encoder: every decoder layer cross-attends to (and
                # caches) the frozen backbone representation as fixed memory.
                for dec_layer in self.dec_layers:
                    h_y = dec_layer(
                        x=h_y,
                        y=h_x,
                        x_padding_mask=y_attn_mask,
                        y_padding_mask=x_attn_mask
                    )
            else:
                for i, enc_layer in enumerate(self.enc_layers):
                    h_x = enc_layer(x=h_x, x_padding_mask=x_attn_mask)

                    if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                        idx = self.num_decoder_layers - self.num_encoder_layers + i
                        dec_layer = self.dec_layers[idx]
                        h_y = dec_layer(
                            x=h_y,
                            y=h_x,
                            x_padding_mask=y_attn_mask,
                            y_padding_mask=x_attn_mask
                        )
        else:
            for dec_layer in self.dec_layers:
                h_y = dec_layer(
                    x=h_y,
                    y=None,
                    x_padding_mask=y_attn_mask,
                    y_padding_mask=x_attn_mask
                )

        return self.lm_head(h_y)

    def _likelihood_logits(
        self,
        batch_idx: int,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encoder-cached logits: encode the source once, reuse for later batches."""
        with torch.no_grad():
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                return self(
                    x, y, t, x_attn_mask, y_attn_mask, use_cache=batch_idx > 0
                )

    def _reset_likelihood_cache(self) -> None:
        self.reset_kv_cache()

    def reset_kv_cache(self):
        """Reset encoder KV caches in all decoder layers."""
        for dec_layer in self.dec_layers:
            dec_layer.cross_attn.kv_cache = None


##########################################
# Non-Flash Attention Transformer Models #
##########################################

class PeintTransformerVanilla(_PeintTransformerBase):
    """PEINT transformer without Flash Attention.

    Same architecture as PeintTransformer but uses standard PyTorch attention.
    Returns additional outputs (representations, attention weights) useful for
    analysis and interpretability.

    Unlike Flash variants, this class does NOT flip attention masks since
    standard attention uses the same convention as the public API (True=padding).
    """

    _flip_attention_masks = False

    def _create_encoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ESMEncoderBlock(
                embed_dim=self.embed_dim,
                ffn_embed_dim=FFN_EXPANSION_FACTOR * self.embed_dim,
                attention_heads=self.num_heads,
                add_bias_kv=False,
                dropout_p=self.dropout_p,
                use_bias=self.use_bias,
                layer_idx=l
            ) for l in range(self.num_encoder_layers)
        ])

    def _create_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([
            ESMDecoderBlock(
                embed_dim=self.embed_dim,
                ffn_embed_dim=FFN_EXPANSION_FACTOR * self.embed_dim,
                attention_heads=self.num_heads,
                add_bias_kv=False,
                dropout_p=self.dropout_p,
                use_bias=self.use_bias,
                layer_idx=l
            ) for l in range(self.num_decoder_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> tuple:
        """Forward pass returning logits and attention weights.

        Args:
            x: Source sequence tokens [B, L_src]
            y: Target sequence tokens [B, L_tgt]
            t: Evolutionary time [B, 1]
            x_attn_mask: Attention mask for source, True=padding [B, L_src]
            y_attn_mask: Attention mask for target, True=padding [B, L_tgt]

        Returns:
            Tuple of (x_logits, y_logits, representations, self_attentions, cross_attentions):
                x_logits: Source logits [B, L_src, vocab_size]
                y_logits: Target logits [B, L_tgt, vocab_size]
                representations: Dict of layer hidden states
                self_attentions: Dict of self-attention weights
                cross_attentions: Dict of cross-attention weights
        """
        h_y = self._prepare_decoder_input(y, t)
        h_x = self._compute_language_model_representations(x)

        representations = {}
        self_attentions = {}
        cross_attentions = {}

        if self.num_encoder_layers == 0:
            # No extra encoder: decoder cross-attends directly to the frozen
            # backbone representation (used as fixed memory) at every layer.
            for j, dec_layer in enumerate(self.dec_layers):
                h_y, (self_att, cross_att) = dec_layer(
                    x=h_y,
                    y=h_x,
                    x_attn_mask=y_attn_mask,
                    y_attn_mask=x_attn_mask
                )
                representations[f'decoder_{j}'] = h_y
                self_attentions[f'decoder_{j}'] = self_att
                cross_attentions[f'decoder_{j}'] = cross_att
        else:
            for i, enc_layer in enumerate(self.enc_layers):
                h_x, attn = enc_layer(x=h_x, attn_mask=x_attn_mask)
                representations[f'encoder_{i}'] = h_x
                self_attentions[f'encoder_{i}'] = attn

                if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                    idx = self.num_decoder_layers - self.num_encoder_layers + i
                    dec_layer = self.dec_layers[idx]
                    h_y, (self_att, cross_att) = dec_layer(
                        x=h_y,
                        y=h_x,
                        x_attn_mask=y_attn_mask,
                        y_attn_mask=x_attn_mask
                    )
                    representations[f'decoder_{i}'] = h_y
                    self_attentions[f'decoder_{i}'] = self_att
                    cross_attentions[f'decoder_{i}'] = cross_att

        x_logits = self.lm_head(h_x)
        y_logits = self.lm_head(h_y)

        return x_logits, y_logits, representations, self_attentions, cross_attentions

    def evaluate_transition_logits(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        x_attn_mask: torch.Tensor,
        y_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the model to get y logits (discards attention outputs)."""
        with torch.no_grad():
            _, y_logits, _, _, _ = self(x, y, t, x_attn_mask, y_attn_mask)
        return y_logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        max_decode_steps: int,
        device: torch.device,
        temperature: float = 1.0,
        p: float = 1.0
    ) -> List[str]:
        """Generate sequences using nucleus sampling."""
        _, x_attn_mask, y_decoded, eos_reached, zero_idx = self._prepare_generation(x, device)

        for _ in range(max_decode_steps):
            y_attn_mask = y_decoded.eq(self.vocab.padding_idx)
            _, y_logits, _, _, _ = self(x, y_decoded, t, x_attn_mask, y_attn_mask)

            logits = y_logits[:, -1, :] / temperature
            logits[..., zero_idx] = -np.inf

            next_tok = sampling_function(logits, p=p)
            y_decoded = torch.cat([y_decoded, next_tok], dim=1)

            eos_reached |= (next_tok.squeeze(-1) == self.vocab.eos_idx)
            if eos_reached.all():
                break

        return self.decode_sequences(y_decoded)


#####################
# Sampling Function #
#####################

def sampling_function(logits: torch.Tensor, p: float = 0.9, argmax_sample: bool = False) -> torch.Tensor:
    """Perform top-p (nucleus) sampling on logits.

    Args:
        logits: Logits of shape [batch_size, vocab_size]
        p: Nucleus sampling parameter (default 0.9). Use 1.0 for full sampling,
           0.0 for argmax.
        argmax_sample: If True, always use argmax regardless of p

    Returns:
        Sampled token indices of shape [batch_size, 1]
    """
    probs = nn.functional.softmax(logits, dim=-1)

    if argmax_sample or p == 0.0:
        return probs.argmax(-1, keepdim=True)

    if p >= 1.0:
        return torch.multinomial(probs, 1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cumulative_probs < p

    nucleus_mask = nucleus.clone()
    nucleus_mask[:, 1:] = nucleus[:, :-1]
    nucleus_mask[:, 0] = True
    sorted_probs = sorted_probs.masked_fill(~nucleus_mask, 0)

    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    sampled_indices = torch.multinomial(sorted_probs, 1)
    next_tok = torch.gather(sorted_indices, 1, sampled_indices)

    return next_tok
