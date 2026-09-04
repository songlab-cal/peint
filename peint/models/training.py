"""Training components for PEINT models.

This module provides PyTorch Lightning integration for training PEINT models.
Import from here when you need training functionality.

Requires: lightning, wandb

Example:
    from peint.models.training import PeintLightningModule, ValidationLikelihoodCallback
"""

from peint.models._training import PeintLightningModule
from peint.models._training_callbacks import (
    ValidationLikelihoodCallback,
    GradNormCallback,
)

__all__ = [
    "PeintLightningModule",
    "ValidationLikelihoodCallback",
    "GradNormCallback",
]
