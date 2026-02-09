"""PyTorch Dataset and Collator for PEINT models.

This module contains data loading utilities that do not require Lightning.
"""

import os

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

        data = []

        infiles = [family + '.txt' for family in families] if len(families) != 0 \
              else list(filter(lambda x: x.endswith('.txt'), os.listdir(data_path)))

        for filename in infiles:
            with open(os.path.join(data_path, filename), 'r') as file_handle:
                file_handle.readline()  # skip header
                for line in file_handle:
                    data.append(line.rstrip('\n').split())

        # filter out sequences longer than max_len
        data = list(filter(lambda x: len(x[0]) <= max_len and len(x[1]) <= max_len, data))

        # ESM's vocab has all ambiguous aa codes other than J, so we replace it here
        self.x = [torch.tensor(vocab.encode(d[0].replace("J", "I"))) for d in data]
        self.y = [torch.tensor(vocab.encode(d[1].replace("J", "I"))) for d in data]

        times = [float(d[2]) for d in data]
        self.lengths = [len(d[0]) for d in data]

        self.t = [max(t, MIN_TIME_THRESHOLD) * torch.ones(self.lengths[i]) for i, t in enumerate(times)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.t[idx], self.lengths[idx]


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

        x_inputs = nn.utils.rnn.pad_sequence(x_inputs, batch_first=True, padding_value=self.vocab.padding_idx)
        x_targets = nn.utils.rnn.pad_sequence(x_targets, batch_first=True, padding_value=self.vocab.padding_idx)

        # Add EOS/BOS to y sequences
        ys = [F.pad(y, (0, 1), value=self.vocab.eos_idx) for y in ys]
        y_inputs = [F.pad(y[:-1], (1, 0), value=self.vocab.cls_idx) for y in ys]
        y_inputs = nn.utils.rnn.pad_sequence(y_inputs, batch_first=True, padding_value=self.vocab.padding_idx)
        y_targets = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=self.vocab.padding_idx)

        ts = torch.stack([torch.tensor([b[0] for b in ts])], dim=-1)

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
