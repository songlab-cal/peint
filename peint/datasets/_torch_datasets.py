"""PyTorch Dataset and Collator for PEINT models.

This module contains data loading utilities that do not require Lightning.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Constants
MIN_TIME_THRESHOLD = 5e-3  # Minimum evolutionary time to avoid numerical instability


class PeintDataset(Dataset):
    '''PyTorch Dataset for PEINT models.

    Loads protein sequence pairs with evolutionary times from text files.

    Args:
        data_path: Path to directory containing family data files
        vocab: Tokenizer with encode() method (e.g., ESM vocabulary)
        families: List of family names to load. If empty, loads all .txt files
        max_len: Maximum sequence length (longer sequences are filtered out)

    Returns:
        Tuple of (x, y, t, length) where x and y are tokenized sequences,
        t is evolutionary time, and length is sequence length.
    '''

    def __init__(self, data_path, vocab, families=[], max_len=1024):
        self.data_path = data_path
        self.families = families
        self.vocab = vocab

        infiles = [family + '.txt' for family in families] if len(families) != 0 \
              else list(filter(lambda x: x.endswith('.txt'), os.listdir(data_path)))

        # Tokens are held in ONE flat contiguous buffer with per-transition offsets,
        # not one small tensor per transition. Millions of tiny tensor objects carry
        # huge Python/torch overhead (measured ~100 GB/rank for the full dataset); a
        # flat int8 buffer (the ESM vocab is < 128 tokens) plus offsets holds the exact
        # same indices in ~10 GB/rank. Since every DDP rank loads the whole dataset,
        # this is what lets it fit on shared nodes. Values are cast back to long in the
        # collator, so the batch fed to the model is unchanged.
        #
        # Two passes over the files: pass 1 records lengths/times only (no sequences
        # retained, so peak memory stays tiny); pass 2 fills a preallocated buffer. The
        # file/line iteration order and the max_len filter are identical across passes,
        # so transition indexing matches the original loader exactly.
        token_dtype = np.int8 if len(vocab) <= 128 else np.int16

        # Vectorized tokenization. Calling vocab.encode() per sequence is ~2 ms/call
        # for fair-esm's Alphabet (~18 h over the full dataset). Instead build a
        # 256-entry char->token-id lookup table once and map residues with numpy,
        # which is ~1000x faster and byte-identical: encoding is purely per-residue.
        # 'J' is not in the ESM vocab, so it maps to 'I' (the original .replace).
        import string as _string
        # Default entry = the vocab's unknown token (matches encode() on chars the
        # vocab doesn't know); fall back to padding for minimal vocabs without one.
        _lut_default = getattr(vocab, "unk_idx", getattr(vocab, "padding_idx", 0))
        token_lut = np.full(256, _lut_default, dtype=token_dtype)
        for _c in _string.ascii_uppercase + "-.":
            try:
                token_lut[ord(_c)] = vocab.encode(_c)[0]
            except Exception:
                pass
        token_lut[ord("J")] = token_lut[ord("I")]

        x_lengths, y_lengths = [], []
        self.lengths, self.t = [], []
        for filename in infiles:
            with open(os.path.join(data_path, filename), 'r') as file_handle:
                file_handle.readline()  # skip header
                for line in file_handle:
                    d = line.rstrip('\n').split()
                    if len(d[0]) > max_len or len(d[1]) > max_len:
                        continue
                    # encode() maps one residue -> one token, so encoded len == string len
                    x_lengths.append(len(d[0]))
                    y_lengths.append(len(d[1]))
                    self.lengths.append(len(d[0]))
                    # Time is constant across a sequence and the collator only uses one
                    # value per transition, so store a single clamped scalar.
                    self.t.append(max(float(d[2]), MIN_TIME_THRESHOLD))

        n = len(self.lengths)
        self.x_off = np.zeros(n + 1, dtype=np.int64)
        self.y_off = np.zeros(n + 1, dtype=np.int64)
        if n:
            self.x_off[1:] = np.cumsum(x_lengths)
            self.y_off[1:] = np.cumsum(y_lengths)
        x_buf = np.empty(int(self.x_off[-1]), dtype=token_dtype)
        y_buf = np.empty(int(self.y_off[-1]), dtype=token_dtype)

        i = 0
        for filename in infiles:
            with open(os.path.join(data_path, filename), 'r') as file_handle:
                file_handle.readline()  # skip header
                for line in file_handle:
                    d = line.rstrip('\n').split()
                    if len(d[0]) > max_len or len(d[1]) > max_len:
                        continue
                    x_buf[self.x_off[i]:self.x_off[i + 1]] = \
                        token_lut[np.frombuffer(d[0].encode("ascii"), dtype=np.uint8)]
                    y_buf[self.y_off[i]:self.y_off[i + 1]] = \
                        token_lut[np.frombuffer(d[1].encode("ascii"), dtype=np.uint8)]
                    i += 1

        # Guard that the compact store is lossless: any token index that exceeded the
        # dtype range would have silently wrapped to a negative value on assignment.
        # (Indices are 0..len(vocab)-1 by construction, so this never triggers for
        # ESM2/ESM-C; it hard-fails instead of corrupting data if a vocab ever grows.)
        if n:
            assert x_buf.min() >= 0 and x_buf.max() < len(vocab), "token store overflow"
            assert y_buf.min() >= 0 and y_buf.max() < len(vocab), "token store overflow"

        self.x_buf = torch.from_numpy(x_buf)
        self.y_buf = torch.from_numpy(y_buf)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        x = self.x_buf[int(self.x_off[idx]):int(self.x_off[idx + 1])]
        y = self.y_buf[int(self.y_off[idx]):int(self.y_off[idx + 1])]
        return x, y, self.t[idx], self.lengths[idx]


class PeintCollator:
    '''Batch collator for PEINT models.

    Handles padding and masking for MLM.
    '''
    def __init__(self, vocab, mask_prob=0.15):
        self.vocab = vocab
        self.num_classes = len(vocab)
        self.mask_prob = mask_prob

    def __call__(self, batch):
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        ts = [b[2] for b in batch]

        # Apply MLM masking and add BOS/EOS tokens
        x_inputs, x_targets = self.mask_for_mlm(xs)
        x_inputs = [F.pad(x, (0, 1), value=self.vocab.eos_idx) for x in x_inputs]
        x_targets = [F.pad(x, (0, 1), value=self.vocab.eos_idx) for x in x_targets]
        x_inputs = [F.pad(x, (1, 0), value=self.vocab.cls_idx) for x in x_inputs]
        x_targets = [F.pad(x, (1, 0), value=self.vocab.cls_idx) for x in x_targets]

        # Cast tokens back to long after padding (int16 storage -> int64 for the
        # embedding / cross-entropy); the resulting indices are identical to the
        # original int64 pipeline.
        x_inputs = nn.utils.rnn.pad_sequence(x_inputs, batch_first=True, padding_value=self.vocab.padding_idx).long()
        x_targets = nn.utils.rnn.pad_sequence(x_targets, batch_first=True, padding_value=self.vocab.padding_idx).long()

        # Add EOS/BOS to y sequences
        ys = [F.pad(y, (0, 1), value=self.vocab.eos_idx) for y in ys]
        y_inputs = [F.pad(y[:-1], (1, 0), value=self.vocab.cls_idx) for y in ys]
        y_inputs = nn.utils.rnn.pad_sequence(y_inputs, batch_first=True, padding_value=self.vocab.padding_idx).long()
        y_targets = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=self.vocab.padding_idx).long()

        # One time value per transition -> shape (B, 1). Robust to both the compact
        # scalar-time store (PeintDataset) and the legacy per-residue time vector
        # (max(t, MIN)*ones(len)) still produced by some datasets (e.g. the evaluation
        # AlignedTransitions dataset): the collator only ever uses a single time per
        # transition, so take element 0 of a tensor time or the scalar directly.
        ts = torch.tensor(
            [float(v.flatten()[0]) if torch.is_tensor(v) else float(v) for v in ts],
            dtype=torch.float32,
        ).unsqueeze(-1)

        x_pad_mask = x_inputs == self.vocab.padding_idx
        y_pad_mask = y_inputs == self.vocab.padding_idx

        return x_inputs, x_targets, y_inputs, y_targets, ts, x_pad_mask, y_pad_mask

    def mask_for_mlm(self, seqs):
        '''Mask tokens for MLM on X. Must be done prior to padding.'''
        x_mlm_masks = [
            torch.rand_like(seq, dtype=torch.float) < self.mask_prob for seq in seqs
        ]

        x_mlm_inputs = [
            seq.masked_fill(mask, self.vocab.mask_idx) for seq, mask in zip(seqs, x_mlm_masks)
        ]

        x_mlm_targets = [
            seq.masked_fill(~mask, self.vocab.padding_idx) for seq, mask in zip(seqs, x_mlm_masks)
        ]

        return x_mlm_inputs, x_mlm_targets
