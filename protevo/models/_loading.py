"""
Model loading utilities for PEINT models.

This module provides functions to load PEINT models from checkpoints,
automatically handling both full checkpoints and PEINT-only checkpoints.
"""

import torch
import torch.nn as nn
from esm.data import Alphabet
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

from protevo.models._transformer_modules import FLASH_AVAILABLE

# Backbone construction is centralized in _esm_registry.build_esm_backbone; the
# concrete ESM2Flash/ESM2Model classes are imported there, not here.
if FLASH_AVAILABLE:
    from protevo.models._transformer import PeintTransformer, PeintGenerator, PeintEvaluator

from protevo.models._transformer import PeintTransformerVanilla


logger = logging.getLogger(__name__)


def _is_peint_only_checkpoint(state_dict: Dict[str, Any]) -> bool:
    """
    Determine if a checkpoint contains only PEINT parameters or full model.

    Args:
        state_dict: The state dictionary from the checkpoint

    Returns:
        True if PEINT-only checkpoint, False if full checkpoint
    """
    # Check for ESM parameters
    esm_keys = [k for k in state_dict.keys() if k.startswith('model.esm.')]

    # Check for embedding and lm_head parameters
    embedding_keys = [k for k in state_dict.keys()
                     if 'embedding' in k and 'time_embedding' not in k]
    lm_head_keys = [k for k in state_dict.keys() if 'lm_head' in k]

    has_esm_params = len(esm_keys) > 0 or len(embedding_keys) > 0 or len(lm_head_keys) > 0

    return not has_esm_params


def _load_esm_model(
    backbone_name: str = "ESM2-150M", use_flash: bool = True
) -> nn.Module:
    """
    Build the pretrained backbone module for a given registry name.

    Thin wrapper around :func:`protevo.models._esm_registry.build_esm_backbone`
    (the single source of truth shared with training), kept for backward
    compatibility. Returns only the module; callers that also need the vocab
    should call ``build_esm_backbone`` directly.

    Args:
        backbone_name: Registry key (e.g. ``"ESM2-8M"``/``"ESM2-150M"``).
        use_flash: Whether to use the Flash Attention version (if available).

    Returns:
        Backbone module instance with pretrained weights loaded.
    """
    from protevo.models._esm_registry import build_esm_backbone

    logger.info(f"Building backbone '{backbone_name}' (use_flash={use_flash})")
    module, _vocab, _embed_dim = build_esm_backbone(backbone_name, use_flash=use_flash)
    return module


def load_peint_model(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu'),
    model_type: str = 'generator',
    use_flash: bool = True,
    strict_loading: bool = False,
    map_location: Optional[str] = None
) -> Tuple[nn.Module, Alphabet]:
    """
    Load a PEINT model from checkpoint, automatically detecting checkpoint type.

    This function handles both:
    - Full checkpoints (containing ESM, PEINT, and all parameters)
    - PEINT-only checkpoints (containing only trainable PEINT layers)

    For PEINT-only checkpoints, ESM2 weights are automatically loaded from
    the public pretrained model (esm2_t30_150M_UR50D).

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt)
        device: Device to load the model onto
        model_type: Model variant to use:
            - 'standard': PeintTransformer/PeintTransformerVanilla (no caching)
            - 'generator': PeintGenerator (caches encoder for generation)
            - 'evaluator': PeintEvaluator (caches encoder for likelihood evaluation)
        use_flash: Whether to use Flash Attention (if available). Defaults to True.
        strict_loading: If True, require exact match when loading state dict.
                       If False, allow missing keys (useful for PEINT-only checkpoints).
        map_location: Optional location to map tensors (default: None uses 'cpu')

    Returns:
        Tuple of (model, vocabulary)
        - model: The loaded PEINT model in eval mode
        - vocabulary: ESM alphabet/vocabulary object

    Example:
        >>> # Load for likelihood evaluation
        >>> model, vocab = load_peint_model(
        ...     'checkpoints/peintvep.ckpt',
        ...     device='cuda',
        ...     model_type='evaluator'
        ... )

        >>> # Load for sequence generation
        >>> model, vocab = load_peint_model(
        ...     'checkpoints/peintvep.ckpt',
        ...     device='cuda',
        ...     model_type='generator'
        ... )
    """
    if model_type not in ('standard', 'generator', 'evaluator'):
        raise ValueError(f"model_type must be 'standard', 'generator', or 'evaluator', got '{model_type}'")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    if map_location is None:
        map_location = 'cpu'

    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Validate checkpoint structure
    if 'state_dict' not in ckpt:
        raise ValueError("Checkpoint missing 'state_dict' key")

    if 'hyper_parameters' not in ckpt:
        logger.warning("Checkpoint missing 'hyper_parameters' - using defaults")
        hyper_params = {}
    else:
        hyper_params = ckpt['hyper_parameters']

    state_dict = ckpt['state_dict']

    # Detect checkpoint type
    is_peint_only = _is_peint_only_checkpoint(state_dict)

    if is_peint_only:
        logger.info("Detected PEINT-only checkpoint (ESM parameters not included)")
        logger.info("Loading ESM2 from pretrained model...")
    else:
        logger.info("Detected full checkpoint (includes ESM parameters)")

    # Build the backbone structure for the checkpoint's configured size. The
    # backbone name is read from the saved hyper-parameters (defaulting to the
    # published ESM2-150M for older checkpoints predating this field), so a model
    # trained on a different backbone rebuilds correctly instead of silently
    # defaulting to 150M. For PEINT-only checkpoints the pretrained weights are
    # used as-is; for full checkpoints they are overwritten by the checkpoint's
    # state dict below.
    from protevo.models._esm_registry import build_esm_backbone

    encoder_backbone = hyper_params.get('encoder_backbone', 'ESM2-150M')
    logger.info(f"Backbone from hyper-parameters: {encoder_backbone}")
    esm_model, vocab, _ = build_esm_backbone(
        encoder_backbone, use_flash=use_flash and FLASH_AVAILABLE
    )

    # Select model class based on Flash Attention and model type
    if use_flash and FLASH_AVAILABLE:
        if model_type == 'evaluator':
            logger.info("Using PeintEvaluator (Flash Attention + Encoder Caching)")
            model_class = PeintEvaluator
        elif model_type == 'generator':
            logger.info("Using PeintGenerator (Flash Attention + Cached Decoder)")
            model_class = PeintGenerator
        else:
            logger.info("Using PeintTransformer (Flash Attention)")
            model_class = PeintTransformer
    else:
        if model_type in ('evaluator', 'generator'):
            logger.warning(
                f"Flash Attention not available. Falling back to PeintTransformerVanilla "
                f"(requested model_type='{model_type}' requires Flash Attention for caching)"
            )
        logger.info("Using PeintTransformerVanilla (Standard)")
        model_class = PeintTransformerVanilla

    # Create model instance
    try:
        model = model_class(
            esm_model=esm_model,
            esm_vocab=vocab,
            **hyper_params
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}")

    # Prepare state dict (remove 'model.' prefix if present)
    renamed_state_dict = {
        k.replace('model.', ''): v
        for k, v in state_dict.items()
    }

    # Load state dict
    try:
        incompatible = model.load_state_dict(renamed_state_dict, strict=strict_loading)

        if not strict_loading:
            # Report any issues
            if incompatible.missing_keys:
                missing_esm = [k for k in incompatible.missing_keys
                              if 'esm' in k.lower() or 'embedding.weight' in k or 'lm_head' in k]
                missing_other = [k for k in incompatible.missing_keys
                               if k not in missing_esm]

                if is_peint_only and missing_esm:
                    logger.info(f"Missing {len(missing_esm)} ESM parameters (expected for PEINT-only checkpoint)")

                if missing_other:
                    logger.warning(f"Missing {len(missing_other)} non-ESM parameters:")
                    for key in missing_other[:5]:
                        logger.warning(f"  - {key}")
                    if len(missing_other) > 5:
                        logger.warning(f"  ... and {len(missing_other) - 5} more")

            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected {len(incompatible.unexpected_keys)} keys in checkpoint:")
                for key in incompatible.unexpected_keys[:5]:
                    logger.warning(f"  - {key}")
                if len(incompatible.unexpected_keys) > 5:
                    logger.warning(f"  ... and {len(incompatible.unexpected_keys) - 5} more")

    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")

    # Set to eval mode and move to device
    model = model.eval()

    if device != torch.device('cpu'):
        logger.info(f"Moving model to {device}")
        model = model.to(device)

    logger.info("Model loaded successfully")

    # Log checkpoint metadata if available
    if 'epoch' in ckpt:
        logger.info(f"Checkpoint from epoch {ckpt['epoch']}")
    if 'global_step' in ckpt:
        logger.info(f"Global step: {ckpt['global_step']}")

    return model, vocab


def load_model(
    model_checkpoint_path: str,
    use_cached_model: bool,
    device: torch.device,
    use_flash: bool = FLASH_AVAILABLE
) -> Tuple[nn.Module, Alphabet]:
    """
    Legacy interface for loading PEINT models.

    This function maintains compatibility with existing code but uses the
    new load_peint_model function internally.

    Args:
        model_checkpoint_path: Path to checkpoint file
        use_cached_model: If True, use cached decoder variant (generator)
        device: Device to load model onto

    Returns:
        Tuple of (model, vocabulary)
    """
    return load_peint_model(
        checkpoint_path=model_checkpoint_path,
        device=device,
        model_type='generator' if use_cached_model else 'standard',
        use_flash=use_flash
    )
