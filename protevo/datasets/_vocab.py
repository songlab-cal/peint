"""Vocabularies for non-ESM2 backbones.

Currently provides the ESM-C vocabulary (:func:`get_esmc_vocab`), which wraps
ESM-C's own sequence tokenizer so that ``len(vocab)`` matches the model's
``sequence_head`` output. The ``esm3`` package is imported lazily, so importing
``protevo`` does not require esm3 unless an ESM-C backbone is actually used.
"""

from typing import List, Sequence

try:  # esm3 provides ESM-C's tokenizer/vocab; optional dependency.
    from esm3.utils.constants import esm3 as _ESM3Constants
    _ESM3_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without esm3
    _ESM3Constants = None
    _ESM3_AVAILABLE = False


class _ESMCVocab:
    """Vocabulary for the ESM-C backbone, based on ESM-3's sequence token list.

    Exposes the same interface the PEINT model/collator expect from an ESM2
    ``Alphabet`` (``cls_idx``/``eos_idx``/``padding_idx``/``mask_idx``, ``encode``,
    ``__len__``, ``to_dict``, ``all_toks``, ``get_idx``), so ESM-C slots in wherever
    an ESM2 alphabet is used. ``encode`` maps residues to ESM-C token ids, and the
    decoder predicts in that same 64-token space (matching ``sequence_head``).
    """

    def __init__(self):
        if not _ESM3_AVAILABLE:
            raise ImportError(
                "ESM-C vocabulary requires the 'esm3' package (pip install esm)."
            )
        tokens = _ESM3Constants.SEQUENCE_VOCAB
        self.all_tokens = tokens
        self.all_toks = tokens  # synonym
        self.index_to_token = {idx: tok for idx, tok in enumerate(tokens)}
        self.token_to_index = {tok: idx for idx, tok in enumerate(tokens)}
        self.idx_to_tok = self.index_to_token  # synonyms
        self.tok_to_idx = self.token_to_index
        self.INDEX2AA = self.index_to_token
        self.AA2INDEX = self.token_to_index

        self.unk_idx = self.token_to_index['<unk>']
        self.cls_idx = self.token_to_index['<cls>']
        self.eos_idx = self.token_to_index['<eos>']
        self.pad_idx = self.token_to_index['<pad>']
        self.mask_idx = self.token_to_index['<mask>']
        self.chain_break_idx = self.token_to_index.get('|', None)
        self.padding_idx = self.pad_idx  # synonym

    def get_idx(self, token: str) -> int:
        return self.token_to_index.get(token, self.unk_idx)

    def get_tok(self, index: int) -> str:
        return self.index_to_token.get(index, '<unk>')

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_index.get(tok, self.unk_idx) for tok in tokens]

    def decode(self, indices: Sequence[int]) -> List[str]:
        return [self.index_to_token.get(i, '<unk>') for i in indices]

    def to_dict(self) -> dict:
        return self.token_to_index.copy()

    def __len__(self) -> int:
        return len(self.all_tokens)


_ESMC_VOCAB_SINGLETON = None


def get_esmc_vocab() -> _ESMCVocab:
    """Return the (lazily constructed) singleton ESM-C vocabulary."""
    global _ESMC_VOCAB_SINGLETON
    if _ESMC_VOCAB_SINGLETON is None:
        _ESMC_VOCAB_SINGLETON = _ESMCVocab()
    return _ESMC_VOCAB_SINGLETON
