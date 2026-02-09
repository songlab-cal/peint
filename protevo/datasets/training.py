"""Training components for PEINT datasets.

This module provides PyTorch Lightning DataModule for training PEINT models.
Import from here when you need training functionality.

Requires: lightning

Example:
    from protevo.datasets.training import PeintDataModule
"""

from protevo.datasets._training import PeintDataModule

__all__ = [
    "PeintDataModule",
]
