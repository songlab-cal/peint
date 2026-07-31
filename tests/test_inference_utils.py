"""Tests for protevo.inference: fast tokenization and work sharding.

These are CPU-only and need no checkpoint — the point is to pin the two claims
the optimized inference paths rest on:

1. the LUT tokenizer reproduces ``Alphabet.encode`` exactly, and the padded
   tensors match what ``encode_sequences`` used to build;
2. sharding covers every work item exactly once, whatever the GPU count.
"""

import numpy as np
import pytest
import torch
from esm.data import Alphabet

from protevo.inference import (
    build_token_lut,
    encode_all,
    encode_batch,
    encode_one,
    item_seed,
    pad_encoded,
    shard_indices,
)

SEQS = [
    "MALIDNKTELFIIESCKQSHGVINSELVNQLILQLECDIESLQQALLPIAATFAQAPISS",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "ACDEFGHIKLMNPQRSTVWY",
    "XBUZO",          # unusual but in-vocab residues
    "MKV",            # short, exercises ragged padding
]


@pytest.fixture(scope="module")
def vocab():
    return Alphabet.from_architecture("ESM-1b")


class TestTokenizerEquivalence:
    """The LUT must agree with fair-esm's own encoder."""

    def test_lut_matches_alphabet_encode(self, vocab):
        lut = build_token_lut(vocab)
        for seq in SEQS:
            assert encode_one(seq, lut).tolist() == vocab.encode(seq), seq

    def test_lut_covers_every_single_vocab_residue(self, vocab):
        lut = build_token_lut(vocab)
        for char in "ACDEFGHIKLMNPQRSTVWYXBUZO-.":
            assert lut[ord(char)] == vocab.encode(char)[0], char

    def test_j_maps_to_i(self, vocab):
        """'J' is absent from the ESM vocab; the dataset loader rewrote it to 'I'."""
        lut = build_token_lut(vocab)
        assert lut[ord("J")] == lut[ord("I")]

    def test_encode_all_matches_encode_one(self, vocab):
        lut = build_token_lut(vocab)
        for got, seq in zip(encode_all(SEQS, vocab), SEQS):
            assert got.tolist() == encode_one(seq, lut).tolist()


def _reference_encode_sequences(sequences, vocab, targets=False):
    """The original pure-Python implementation, kept here as the oracle."""
    encoded_inputs, encoded_targets = [], []
    for seq in sequences:
        core = vocab.encode(seq)
        if targets:
            encoded_inputs.append(torch.tensor([vocab.cls_idx] + core))
            encoded_targets.append(torch.tensor(core + [vocab.eos_idx]))
        else:
            encoded_inputs.append(torch.tensor([vocab.cls_idx] + core + [vocab.eos_idx]))

    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        encoded_inputs, batch_first=True, padding_value=vocab.padding_idx
    )
    if targets:
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            encoded_targets, batch_first=True, padding_value=vocab.padding_idx
        )
        return padded_inputs, padded_targets
    return padded_inputs, None


class TestPaddedTensorsMatchOriginal:
    """encode_batch must be a drop-in for the old encode_sequences."""

    @pytest.mark.parametrize("targets", [False, True])
    def test_matches_reference(self, vocab, targets):
        got_in, got_tgt = encode_batch(SEQS, vocab, targets=targets)
        want_in, want_tgt = _reference_encode_sequences(SEQS, vocab, targets=targets)

        assert got_in.dtype == want_in.dtype == torch.int64
        assert torch.equal(got_in, want_in)
        if targets:
            assert torch.equal(got_tgt, want_tgt)
        else:
            assert got_tgt is None and want_tgt is None

    def test_pad_encoded_matches_encode_batch(self, vocab):
        """Pre-tokenizing then padding equals tokenizing and padding in one go."""
        direct_in, direct_tgt = encode_batch(SEQS, vocab, targets=True)
        staged_in, staged_tgt = pad_encoded(encode_all(SEQS, vocab), vocab, targets=True)
        assert torch.equal(direct_in, staged_in)
        assert torch.equal(direct_tgt, staged_tgt)

    def test_single_sequence(self, vocab):
        got, _ = encode_batch([SEQS[0]], vocab)
        want, _ = _reference_encode_sequences([SEQS[0]], vocab)
        assert torch.equal(got, want)


class TestSharding:
    """Work assignment must be a partition, and independent of shard count."""

    @pytest.mark.parametrize("n_items", [1, 5, 17, 100])
    @pytest.mark.parametrize("n_shards", [1, 2, 3, 8])
    def test_shards_partition_the_work(self, n_items, n_shards):
        shards = [shard_indices(n_items, n_shards, r) for r in range(n_shards)]
        flat = [i for s in shards for i in s]
        assert sorted(flat) == list(range(n_items)), "every item exactly once"

    def test_shards_are_balanced(self):
        sizes = [len(shard_indices(100, 8, r)) for r in range(8)]
        assert max(sizes) - min(sizes) <= 1

    def test_item_seed_depends_on_index_not_rank(self):
        """Seeding by item index is what makes 8-GPU output match 1-GPU output."""
        assert item_seed(0, 7) == item_seed(0, 7)
        assert item_seed(0, 7) != item_seed(0, 8)
        assert 0 <= item_seed(0, 10**9) < 2**31 - 1

    def test_run_sharded_single_process_preserves_order(self):
        """num_gpus=1 runs in-process; results still come back in item order."""
        from protevo.inference import run_sharded

        items = list(range(10))
        results = run_sharded(items, _square_worker, num_gpus=1)
        assert results == [i * i for i in items]

    def test_run_sharded_empty(self):
        from protevo.inference import run_sharded

        assert run_sharded([], _square_worker, num_gpus=1) == []


def _square_worker(shard, device, rank):
    """Module-level so it stays picklable under spawn."""
    return [(idx, value * value) for idx, value in shard]
