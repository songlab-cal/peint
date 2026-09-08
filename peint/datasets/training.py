"""Training components for PEINT datasets.

This module provides PyTorch Lightning DataModule for training PEINT models.
Import from here when you need training functionality.

Requires: lightning

Example:
    from peint.datasets.training import PeintDataModule
"""

from peint.datasets._training import PeintDataModule

__all__ = [
    "PeintDataModule",
]
