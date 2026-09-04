"""PEINT model components.

Core model classes (no Lightning/wandb dependency):
    - PeintTransformer: Flash Attention variant for training
    - PeintTransformerVanilla: Standard attention for interpretability
    - PeintGenerator: Flash Attention with KV caching for generation
    - PeintEvaluator: Flash Attention with encoder caching for likelihood evaluation

Training components (require Lightning/wandb, import from peint.models.training):
    - PeintLightningModule: Lightning wrapper for training
    - ValidationLikelihoodCallback: Logs validation metrics to wandb
    - GradNormCallback: Logs gradient norms during training
"""

from ._equ import equ_model__cached, equ_rate_matrix
from ._lg import (
    evaluate_lg_model_transitions_log_likelihood__cached,
    train_lg_model__cached,
)
from ._uniform_random_guess import (
    evaluate_uniform_random_guess_model_transitions_log_likelihood__cached,
)
from ._wag import (
    evaluate_wag_model_transitions_log_likelihood__cached,
    train_wag_model__cached,
)
from ._config import PeintConfig
from ._transformer import (
    _PeintTransformerBase,
    PeintTransformer,
    PeintTransformerVanilla,
    PeintGenerator,
    PeintEvaluator,
)

from ._loading import load_model, load_peint_model
from ._esm_registry import ESM2_REGISTRY, get_esm_model

__all__ = [
    # Configuration
    "PeintConfig",
    # Classical models
    "equ_model__cached",
    "equ_rate_matrix",
    "evaluate_lg_model_transitions_log_likelihood__cached",
    "evaluate_uniform_random_guess_model_transitions_log_likelihood__cached",
    "train_lg_model__cached",
    "evaluate_wag_model_transitions_log_likelihood__cached",
    "train_wag_model__cached",
    # PEINT transformer models
    "_PeintTransformerBase",
    "PeintTransformer",
    "PeintTransformerVanilla",
    "PeintGenerator",
    "PeintEvaluator",
    # Loading functions
    "load_model",
    "load_peint_model",
    # ESM registry
    "ESM2_REGISTRY",
    "get_esm_model",
]
