"""PyTorch Lightning DataModule for PEINT training.

This module contains the Lightning DataModule for training PEINT models.

Requires: lightning
"""

import lightning as pl
from torch.utils.data import DataLoader, random_split

from protevo.datasets._torch_datasets import PeintDataset, PeintCollator


class PeintDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for PEINT training.

    Wraps PeintDataset with train/val splitting and batching.
    """

    def __init__(
        self,
        data_path,
        vocab,
        families=[],
        max_len=1022,
        mask_prob=0.15,
        batch_size=32,
        train_frac=0.85,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_len = max_len
        self.families = families
        self.vocab = vocab
        self.batch_size = batch_size
        self.mask_prob = mask_prob
        self.train_frac = train_frac
        self.val_frac = 1 - train_frac

    def setup(self, stage=None):
        self.dataset = PeintDataset(
            self.data_path,
            vocab=self.vocab,
            families=self.families,
            max_len=self.max_len,
        )
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [self.train_frac, self.val_frac]
        )
        self.collate_fn = PeintCollator(vocab=self.vocab, mask_prob=self.mask_prob)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
