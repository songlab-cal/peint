"""Vectorized sequence tokenization shared by the dataset and the inference paths.

``PeintDataset`` already replaced fair-esm's ``Alphabet.encode`` with a 256-entry
char -> token-id lookup table, because ``encode`` costs ~2 ms per sequence in pure
Python. The inference paths never got that treatment, so
``_PeintTransformerBase.encode_sequences`` still calls ``vocab.encode`` once per
sequence — and homology search calls ``encode_sequences`` on the *entire* query set
once per reference, making tokenization an O(N^2) pure-Python cost on the critical
path of an all-vs-all run.

This module holds the LUT construction in one place so both callers share it.

Equivalence: ``Alphabet.encode`` maps each residue independently to
``tok_to_idx[residue]``, which is exactly a table lookup, so the LUT reproduces it
byte-for-byte on every input the original accepts. The table is strictly more
permissive in two documented cases — ``J`` maps to ``I`` (following the dataset
loader's original ``.replace``) and unknown characters map to the vocab's unknown
token — where ``vocab.encode`` would raise ``KeyError``.
"""

from __future__ import annotations

import string
from typing import List, Optional, Sequence

import numpy as np
import torch

# Characters worth probing when building the table: the 26 uppercase letters plus
# the gap/insertion symbols used in MSAs.
_LUT_ALPHABET = string.ascii_uppercase + "-."


def build_token_lut(vocab, dtype=np.int16) -> np.ndarray:
    """Build the 256-entry char -> token-id table for ``vocab``.

    Args:
        vocab: A fair-esm ``Alphabet`` (or the ESM-C vocab shim) exposing
            ``encode``, and ideally ``unk_idx`` / ``padding_idx``.
        dtype: Integer dtype for the table. ``PeintDataset`` uses ``int8`` to keep
            its flat token buffer small; the inference path uses ``int16`` since it
            converts to ``long`` immediately anyway.

    Returns:
        ``np.ndarray`` of shape (256,) indexable by ``ord(char)``.
    """
    # Default entry = the vocab's unknown token (matches encode() on chars the
    # vocab doesn't know); fall back to padding for minimal vocabs without one.
    default = getattr(vocab, "unk_idx", getattr(vocab, "padding_idx", 0))
    lut = np.full(256, default, dtype=dtype)
    for char in _LUT_ALPHABET:
        try:
            lut[ord(char)] = vocab.encode(char)[0]
        except Exception:
            pass
    # 'J' is not in the ESM vocab; the dataset loader historically rewrote it to 'I'.
    lut[ord("J")] = lut[ord("I")]
    return lut


def encode_one(sequence: str, lut: np.ndarray) -> np.ndarray:
    """Tokenize a single sequence with a prebuilt table."""
    return lut[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def encode_batch(
    sequences: Sequence[str],
    vocab,
    lut: Optional[np.ndarray] = None,
    targets: bool = False,
    device: Optional[torch.device] = None,
):
    """Tokenize and pad a batch of sequences.

    Drop-in replacement for ``_PeintTransformerBase.encode_sequences``, producing
    identical tensors:

    * ``targets=False``: inputs are ``[cls] + seq + [eos]``.
    * ``targets=True``: inputs are ``[cls] + seq``, targets are ``seq + [eos]``
      (teacher forcing).

    Padding is right-padding to the batch maximum with ``vocab.padding_idx``,
    matching ``nn.utils.rnn.pad_sequence``.

    Returns:
        ``(padded_inputs, padded_targets_or_None)`` as int64 tensors.
    """
    if lut is None:
        lut = build_token_lut(vocab)

    cls_idx = vocab.cls_idx
    eos_idx = vocab.eos_idx
    pad_idx = vocab.padding_idx

    encoded = [encode_one(seq, lut) for seq in sequences]
    lengths = [len(e) for e in encoded]
    n = len(encoded)

    # Inputs carry one extra leading token, plus a trailing eos when not teacher
    # forcing. Targets are the same length as the (cls-prefixed) inputs.
    in_extra = 1 if targets else 2
    max_in = (max(lengths) + in_extra) if n else in_extra

    inputs = np.full((n, max_in), pad_idx, dtype=np.int64)
    for i, (toks, length) in enumerate(zip(encoded, lengths)):
        inputs[i, 0] = cls_idx
        inputs[i, 1:length + 1] = toks
        if not targets:
            inputs[i, length + 1] = eos_idx

    input_tensor = torch.from_numpy(inputs)
    if device is not None:
        input_tensor = input_tensor.to(device)

    if not targets:
        return input_tensor, None

    max_tgt = (max(lengths) + 1) if n else 1
    tgts = np.full((n, max_tgt), pad_idx, dtype=np.int64)
    for i, (toks, length) in enumerate(zip(encoded, lengths)):
        tgts[i, :length] = toks
        tgts[i, length] = eos_idx

    target_tensor = torch.from_numpy(tgts)
    if device is not None:
        target_tensor = target_tensor.to(device)
    return input_tensor, target_tensor


def encode_all(sequences: Sequence[str], vocab, lut: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Tokenize every sequence once, without padding.

    Useful for the homology paths, which want to pay tokenization once for the
    whole corpus and then re-slice it per reference.
    """
    if lut is None:
        lut = build_token_lut(vocab)
    return [encode_one(seq, lut) for seq in sequences]
