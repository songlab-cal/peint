"""PEINT: Protein Evolution IN Time

An encoder-decoder transformer model for protein evolutionary modeling.
The encoder uses ESM2 pretrained representations, and the decoder
autoregressively predicts target sequences given source sequences
and evolutionary time.

Core components (no Lightning/wandb dependency):
    - PeintTransformer, PeintGenerator, PeintEvaluator, PeintTransformerVanilla
    - PeintDataset, PeintCollator

Training components (require Lightning/wandb):
    - from peint.models.training import PeintLightningModule
    - from peint.datasets.training import PeintDataModule
"""

from .models import (
    PeintTransformer,
    PeintTransformerVanilla,
    PeintGenerator,
    PeintEvaluator,
)
from .datasets import (
    PeintDataset,
    PeintCollator,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "PeintTransformer",
    "PeintTransformerVanilla",
    "PeintGenerator",
    "PeintEvaluator",
    # Datasets
    "PeintDataset",
    "PeintCollator",
]
