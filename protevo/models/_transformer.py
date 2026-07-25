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
from typing import List

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

from protevo.utils import amino_acids

# Model constants
DEFAULT_MAX_SEQ_LEN = 1022  # ESM2's max sequence length minus special tokens
STANDARD_STATES = list(amino_acids) + ['<eos>']


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
        use_time_conditioning=kwargs.get('use_time_conditioning', True),
        encoder_backbone=kwargs.get('encoder_backbone', 'ESM2-150M'),
        esm_finetune_mode=kwargs.get('esm_finetune_mode', 'frozen'),
        lora_rank=kwargs.get('lora_rank', None),
        architecture=kwargs.get('architecture', 'encoder_decoder'),
    )


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
        self.use_time_conditioning = self.config.use_time_conditioning

        # Pretrained backbone. Frozen by default (published PEINT); the "lora"/"full"
        # fine-tuning modes keep it trainable and are wired up in the LoRA ablation.
        self.esm = esm_model
        self.vocab = esm_vocab
        self.esm.eval()
        if self.config.esm_finetune_mode == "frozen":
            self.esm.requires_grad_(False)

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

        # Token embeddings (initialized from ESM, frozen)
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        self.embedding.requires_grad_(False)

        # Time embedding
        self.time_embedding = GeometricTimeEmbedder(frequency_embedding_size=embed_dim)

        # Create encoder and decoder layers via factory methods
        self.enc_layers = self._create_encoder_layers()
        self.dec_layers = self._create_decoder_layers()

        # Language model head (initialized from ESM, frozen)
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
        """Compute ESM2 representations for source sequence.

        Args:
            x: Source tokens [B, L] with CLS and EOS from dataloader

        Returns:
            Final hidden state from ESM [B, L, D]
        """
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
        if not self.use_time_conditioning:
            # Ablate evolutionary-time conditioning: decoder input is the token
            # embedding alone, with no additive time signal.
            return h_y
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
        inv_vocab = {v: k for k, v in self.vocab.to_dict().items()}
        output_sequences = []
        for seq in decoded:
            decoded_str = ''.join([inv_vocab.get(p.item()) for p in seq[1:]])  # remove cls
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
        zero_idx = torch.tensor([
            self.vocab.get_idx(tok)
            for tok in self.vocab.all_toks
            if tok not in STANDARD_STATES
        ])
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
        encoded_inputs = []
        encoded_targets = []
        for seq in sequences:
            encoded_core = self.vocab.encode(seq)
            if targets:
                encoded_inputs.append(torch.tensor([self.vocab.cls_idx] + encoded_core))
                encoded_targets.append(torch.tensor(encoded_core + [self.vocab.eos_idx]))
            else:
                encoded_inputs.append(
                    torch.tensor([self.vocab.cls_idx] + encoded_core + [self.vocab.eos_idx])
                )

        padded_inputs = nn.utils.rnn.pad_sequence(
            encoded_inputs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        if targets:
            padded_targets = nn.utils.rnn.pad_sequence(
                encoded_targets, batch_first=True, padding_value=self.vocab.padding_idx
            )
            return padded_inputs, padded_targets

        return padded_inputs, None

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
        batch_size: int = 128
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

        Returns:
            Array of mean per-residue negative log-likelihoods (one per target)
        """
        assert len(t) == len(y), "Time and sequences must be the same length"

        likelihoods = []
        encoded_x, _ = self.encode_sequences([x])

        for i in tqdm(range(0, len(y), batch_size)):
            y_batch = y[i:i + batch_size]
            y_encoded, y_targets = self.encode_sequences(y_batch, targets=True)
            y_encoded = y_encoded.to(device)
            y_targets = y_targets.to(device)
            y_attn_mask = y_encoded.eq(self.vocab.padding_idx)

            x_encoded = encoded_x.repeat(y_encoded.size(0), 1).to(device)
            x_attn_mask = x_encoded.eq(self.vocab.padding_idx)

            times = torch.tensor(t[i:i + batch_size]).unsqueeze(-1).to(device)

            logits = self._likelihood_logits(
                i, x_encoded, y_encoded, times, x_attn_mask, y_attn_mask
            )

            ll = nn.functional.cross_entropy(
                logits.float().transpose(1, 2),
                y_targets,
                ignore_index=self.vocab.padding_idx,
                reduction='none'
            )

            non_pad = y_targets.ne(self.vocab.padding_idx)
            ll = (ll * non_pad).sum(dim=-1) / non_pad.sum(dim=-1)
            likelihoods.append(ll.cpu().numpy())

        self._reset_likelihood_cache()
        return np.vstack([ll[:, None] for ll in likelihoods]).squeeze()


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

        for _ in range(max_decode_steps - 1):
            logits = logits[:, -1, :] / temperature
            logits[..., zero_idx] = -np.inf

            next_token = sampling_function(logits, p=p)

            y_new = next_token
            y_attn_mask = y_new.eq(self.vocab.padding_idx)
            logits = self.forward(x, y_new, t, x_attn_mask, y_attn_mask, use_cache=True)

            y_decoded = torch.cat([y_decoded, next_token], dim=1)

            eos_reached |= (next_token.squeeze(-1) == self.vocab.eos_idx)
            if eos_reached.all():
                break

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
