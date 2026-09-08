"""Guard that the memory-efficient loader is byte-identical to the original.

PeintDataset now stores tokens as int16 and time as a scalar (instead of int64
tokens + a per-residue float32 vector), cutting per-rank RAM ~5x so the dataset
fits on shared nodes. This test reimplements the ORIGINAL int64 / length-vector
dataset + collator inline and asserts the collated batch (dtypes and values) is
identical, so training is provably unchanged.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.data import Alphabet

from peint.datasets._torch_datasets import PeintCollator, PeintDataset

VOCAB = Alphabet.from_architecture("ESM-1b")  # no download

ROWS = [
    ("ACDEFGHIK", "ACDEYGHIK", "0.1"),
    ("MKTAYIAKQRQISFVK", "MKTAYIAKQRQISFVR", "0.5"),
    ("WWWWYY", "WWWWYF", "0.02"),
    ("GG", "GA", "0.003"),   # below MIN_TIME_THRESHOLD -> clamped to 5e-3
    ("PLKJHTRESAQ", "PLKIHTRESAQ", "1.7"),
]


def _write(path, rows):
    with open(path, "w") as f:
        f.write("n transitions\n")
        for x, y, t in rows:
            f.write(f"{x} {y} {t}\n")


# --- reference: the ORIGINAL int64 tokens + per-residue float32 time pipeline ---
class _OldDataset:
    def __init__(self, data_path, vocab, families, max_len=1024):
        data = []
        for fam in families:
            with open(os.path.join(data_path, fam + ".txt")) as fh:
                fh.readline()
                for line in fh:
                    data.append(line.rstrip("\n").split())
        data = [d for d in data if len(d[0]) <= max_len and len(d[1]) <= max_len]
        self.x = [torch.tensor(vocab.encode(d[0].replace("J", "I"))) for d in data]
        self.y = [torch.tensor(vocab.encode(d[1].replace("J", "I"))) for d in data]
        times = [float(d[2]) for d in data]
        self.lengths = [len(d[0]) for d in data]
        self.t = [max(t, 5e-3) * torch.ones(self.lengths[i]) for i, t in enumerate(times)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.t[i], self.lengths[i]


def _old_collate(batch, vocab, mask_prob):
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    ts = [b[2] for b in batch]
    masks = [torch.rand_like(s, dtype=torch.float) < mask_prob for s in xs]
    xin = [s.masked_fill(m, vocab.mask_idx) for s, m in zip(xs, masks)]
    xtg = [s.masked_fill(~m, vocab.padding_idx) for s, m in zip(xs, masks)]
    xin = [F.pad(x, (0, 1), value=vocab.eos_idx) for x in xin]
    xtg = [F.pad(x, (0, 1), value=vocab.eos_idx) for x in xtg]
    xin = [F.pad(x, (1, 0), value=vocab.cls_idx) for x in xin]
    xtg = [F.pad(x, (1, 0), value=vocab.cls_idx) for x in xtg]
    xin = nn.utils.rnn.pad_sequence(xin, batch_first=True, padding_value=vocab.padding_idx)
    xtg = nn.utils.rnn.pad_sequence(xtg, batch_first=True, padding_value=vocab.padding_idx)
    ys = [F.pad(y, (0, 1), value=vocab.eos_idx) for y in ys]
    yin = [F.pad(y[:-1], (1, 0), value=vocab.cls_idx) for y in ys]
    yin = nn.utils.rnn.pad_sequence(yin, batch_first=True, padding_value=vocab.padding_idx)
    ytg = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=vocab.padding_idx)
    ts = torch.stack([torch.tensor([b[0] for b in ts])], dim=-1)
    return xin, xtg, yin, ytg, ts, xin == vocab.padding_idx, yin == vocab.padding_idx


NAMES = ["x_inputs", "x_targets", "y_inputs", "y_targets", "ts", "x_pad_mask", "y_pad_mask"]


def test_collated_batch_is_byte_identical(tmp_path):
    _write(tmp_path / "fam.txt", ROWS)
    new_ds = PeintDataset(str(tmp_path), vocab=VOCAB, families=["fam"], max_len=1024)
    old_ds = _OldDataset(str(tmp_path), vocab=VOCAB, families=["fam"], max_len=1024)

    new_batch = [new_ds[i] for i in range(len(new_ds))]
    old_batch = [old_ds[i] for i in range(len(old_ds))]

    collator = PeintCollator(VOCAB, mask_prob=0.15)
    torch.manual_seed(123)
    out_new = collator(new_batch)
    torch.manual_seed(123)          # same RNG state -> same MLM mask
    out_old = _old_collate(old_batch, VOCAB, 0.15)

    for name, a, b in zip(NAMES, out_new, out_old):
        assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{name}: shape {a.shape} != {b.shape}"
        assert torch.equal(a, b), f"{name}: values differ"


def test_storage_is_compact():
    """Tokens stored as int16 and time as a python scalar (the memory win)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "fam.txt"), ROWS)
        ds = PeintDataset(d, vocab=VOCAB, families=["fam"], max_len=1024)
    # Tokens live in ONE flat buffer (not millions of per-item tensors). For the
    # standard ESM vocab (33 tokens) that buffer is int8 -> this is the path the
    # bit-identity test above exercises.
    assert len(VOCAB) <= 128
    assert ds.x_buf.dtype == torch.int8
    assert ds.y_buf.dtype == torch.int8
    x0, y0, t0, _ = ds[0]
    assert x0.dtype == torch.int8 and y0.dtype == torch.int8
    assert isinstance(t0, float)  # scalar, not a per-residue tensor
    # Lossless: the compact store round-trips to the exact original indices.
    import torch as _t
    assert _t.equal(x0.long(), _t.tensor(VOCAB.encode(ROWS[0][0].replace("J", "I"))))


def test_vectorized_tokenization_matches_vocab_encode_every_char():
    """C4: the LUT-based tokenizer equals vocab.encode(...) for every possible
    residue char (all A-Z + gap chars), including the J->I fold. This is the exact
    per-residue guarantee that makes the fast loader byte-identical to the slow one.
    """
    import string
    import tempfile

    chars = string.ascii_uppercase + "-."           # every char the LUT populates
    seq = chars + "J"                                 # include J (must fold to I)
    # x must be <= max_len and pair with a y; reuse seq for both.
    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "fam.txt"), [(seq, seq, "0.1")])
        ds = PeintDataset(d, vocab=VOCAB, families=["fam"], max_len=1024)
    x0, _, _, _ = ds[0]
    expected = VOCAB.encode(seq.replace("J", "I"))
    assert torch.equal(x0.long(), torch.tensor(expected))
    # explicit J->I check
    assert x0.long()[-1].item() == VOCAB.encode("I")[0]


def test_returned_tokens_are_long_for_the_model():
    """Collator output tokens are int64 (embedding / cross-entropy require long)."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "fam.txt"), ROWS)
        ds = PeintDataset(d, vocab=VOCAB, families=["fam"], max_len=1024)
        out = PeintCollator(VOCAB)([ds[i] for i in range(len(ds))])
    x_inputs, x_targets, y_inputs, y_targets, ts, *_ = out
    for tok in (x_inputs, x_targets, y_inputs, y_targets):
        assert tok.dtype == torch.long
    assert ts.dtype == torch.float32 and ts.shape[-1] == 1
