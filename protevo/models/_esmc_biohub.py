"""Biohub ESM-C (transformers) backbone for PEINT — a self-contained base-LM model.

ESMC-300M is served through HuggingFace transformers (Biohub's transformers fork
registers the ``esmc`` model type). Unlike ESM2 — which PEINT re-implements with Flash
Attention (``ESM2Flash``) because stock ESM2 is not flash — ESM-C is already flash-native
(loaded with ``attn_implementation="flash_attention_2"``), so PEINT uses it frozen as-is
and just stacks its own encoder/decoder layers on top. Only the embedding / head /
representation access differs from ESM2 (handled in ``_transformer._is_esmc_biohub_backbone``).

Needs only the Biohub transformers fork installed (NOT the ``esm`` package), so it coexists
with fair-esm. See the split-backbone note in the README.
"""

from typing import List, Sequence

# name -> (hf repo id, embed_dim). ESMC-300M hidden size is 960; head is 64-wide.
ESMC_BIOHUB_REGISTRY = {
    "esmc-biohub":      ("biohub/ESMC-300M", 960),
    "biohub/ESMC-300M": ("biohub/ESMC-300M", 960),
}

_ESMC_BIOHUB_REPO = "biohub/ESMC-300M"


class _ESMCBiohubVocab:
    """Alphabet adapter over Biohub ESM-C's HF ``ESMCTokenizer``.

    Exposes the interface PEINT expects from an ESM2 ``Alphabet`` (``cls_idx``/``eos_idx``/
    ``pad_idx``/``mask_idx``/``unk_idx``, ``encode``/``decode``, ``__len__``, ``to_dict``,
    ``all_toks``, ``get_idx``), sourced from the transformers tokenizer so token ids match
    the ESM-C checkpoint. ``len`` is 33 real tokens; the model head is 64-wide.
    """

    def __init__(self, repo: str = _ESMC_BIOHUB_REPO):
        from transformers import AutoTokenizer  # optional; only when biohub ESM-C is used

        tok = AutoTokenizer.from_pretrained(repo)
        self._tokenizer = tok
        vocab = tok.get_vocab()  # token -> id
        self.token_to_index = dict(vocab)
        self.index_to_token = {i: t for t, i in vocab.items()}
        self.all_tokens = [self.index_to_token[i] for i in range(len(self.index_to_token))]
        self.all_toks = self.all_tokens
        self.idx_to_tok = self.index_to_token
        self.tok_to_idx = self.token_to_index
        self.INDEX2AA = self.index_to_token
        self.AA2INDEX = self.token_to_index

        def _require(idx, name):
            if idx is None:
                raise ValueError(f"ESMCTokenizer has no {name} token id")
            return idx

        self.cls_idx = _require(tok.cls_token_id, "cls")
        self.eos_idx = _require(tok.eos_token_id, "eos")
        self.pad_idx = _require(tok.pad_token_id, "pad")
        self.mask_idx = _require(tok.mask_token_id, "mask")
        self.unk_idx = tok.unk_token_id
        if self.unk_idx is None:
            self.unk_idx = self.token_to_index.get("<unk>", self.pad_idx)
        self.chain_break_idx = self.token_to_index.get("|", None)
        self.padding_idx = self.pad_idx

    def get_idx(self, token: str) -> int:
        return self.token_to_index.get(token, self.unk_idx)

    def get_tok(self, index: int) -> str:
        return self.index_to_token.get(index, "<unk>")

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_index.get(tok, self.unk_idx) for tok in tokens]

    def decode(self, indices: Sequence[int]) -> List[str]:
        return [self.index_to_token.get(i, "<unk>") for i in indices]

    def to_dict(self) -> dict:
        return dict(self.token_to_index)

    def __len__(self) -> int:
        return len(self.all_tokens)


_ESMC_BIOHUB_VOCAB_SINGLETON = None


def get_esmc_biohub_vocab() -> _ESMCBiohubVocab:
    """Return the (lazily constructed) singleton Biohub ESM-C vocabulary."""
    global _ESMC_BIOHUB_VOCAB_SINGLETON
    if _ESMC_BIOHUB_VOCAB_SINGLETON is None:
        _ESMC_BIOHUB_VOCAB_SINGLETON = _ESMCBiohubVocab()
    return _ESMC_BIOHUB_VOCAB_SINGLETON


def build_esmc_biohub_backbone(name: str = "esmc-biohub", use_flash: bool = True):
    """Build the frozen biohub ESM-C backbone + vocab.

    Returns ``(ESMCForMaskedLM, vocab, embed_dim)``. ``use_flash=True`` loads under
    FlashAttention-2 (bf16, GPU); ``use_flash=False`` uses the default attention (fp32,
    CPU-capable). PEINT detects the module via ``_is_esmc_biohub_backbone`` and reads the
    token embedding from ``model.esmc.embed``, the 64-wide head from ``model.lm_head``, and
    representations from the final hidden state (``output_hidden_states``).
    """
    import torch
    from transformers import AutoModelForMaskedLM

    repo, embed_dim = ESMC_BIOHUB_REGISTRY[name]
    if use_flash:
        kwargs = {"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
    else:
        kwargs = {}  # default attention, fp32 — CPU-capable
    module = AutoModelForMaskedLM.from_pretrained(repo, **kwargs)
    return module, get_esmc_biohub_vocab(), embed_dim
