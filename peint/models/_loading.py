"""
Model loading utilities for PEINT models.

This module provides functions to load PEINT models from checkpoints,
automatically handling both full checkpoints and PEINT-only checkpoints.
"""

import torch
import torch.nn as nn
import esm
from esm.data import Alphabet
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

from peint.models._transformer_modules import FLASH_AVAILABLE

if FLASH_AVAILABLE:
    from peint.models._flash_esm import ESM2Flash
    from peint.models._transformer import PeintTransformer, PeintGenerator, PeintEvaluator
else:
    from peint.models._flash_esm import ESM2Model

from peint.models._transformer import PeintTransformerVanilla


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


def _load_esm_model(use_flash: bool = True) -> nn.Module:
    """
    Load ESM2 model from pretrained weights.

    Args:
        use_flash: Whether to use Flash Attention version (if available)

    Returns:
        ESM2 model instance with pretrained weights loaded
    """
    # Load standard ESM2 pretrained model first
    logger.info("Loading pretrained ESM2 weights...")
    esm_pretrained, _ = esm.pretrained.esm2_t30_150M_UR50D()

    if use_flash and FLASH_AVAILABLE:
        logger.info("Creating ESM2 model with Flash Attention support")
        # Create Flash ESM model with same architecture
        flash_esm = ESM2Flash(
            num_layers=esm_pretrained.num_layers,
            embed_dim=esm_pretrained.embed_dim,
            attention_heads=esm_pretrained.attention_heads,
            alphabet='ESM-1b',
            token_dropout=True,
            dropout_p=0.0
        )

        # Load pretrained weights
        flash_esm.load_state_dict(esm_pretrained.state_dict(), strict=False)
        del esm_pretrained

        return flash_esm
    else:
        if not FLASH_AVAILABLE:
            logger.info("Flash Attention not available, using standard ESM2 model")
        else:
            logger.info("Loading standard ESM2 model (Flash Attention disabled)")

        from peint.models._flash_esm import ESM2Model

        # Create ESM2Model with same architecture
        esm_model = ESM2Model(
            num_layers=esm_pretrained.num_layers,
            embed_dim=esm_pretrained.embed_dim,
            attention_heads=esm_pretrained.attention_heads,
            alphabet='ESM-1b',
            token_dropout=True,
            dropout_p=0.0
        )

        # Load pretrained weights
        esm_model.load_state_dict(esm_pretrained.state_dict(), strict=False)
        del esm_pretrained

        return esm_model


def load_peint_model(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu'),
    model_type: str = 'generator',
    use_flash: bool = True,
    strict_loading: bool = False,
    map_location: Optional[str] = None,
    max_encoder_seq_len: Optional[int] = None,
    max_decoder_seq_len: Optional[int] = None
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

    # Detect the frozen encoder backbone. Biohub ESM-C checkpoints record encoder_backbone
    # in {"esmc", "esmc-biohub"}; everything else is ESM2.
    # ESM-C is flash-native and brings its own backbone + 64-wide vocab, so it is built here
    # rather than via the ESM2 pretrained/scaffold paths. Its saved encoder/embedding/lm_head
    # weights load with strict=False below (they equal the biohub ESMC-300M weights in bf16),
    # leaving the freshly built backbone in place.
    is_esmc = (
        hyper_params.get("encoder_backbone") in ("esmc", "esmc-biohub")
        or hyper_params.get("which_esm") == "esmc-biohub"
    )

    # Whether the checkpoint omits encoder weights (used later when reporting missing keys).
    # ESM-C checkpoints ship a full backbone under model.esm.*, so they are never PEINT-only.
    is_peint_only = False if is_esmc else _is_peint_only_checkpoint(state_dict)

    if is_esmc:
        logger.info("Detected Biohub ESM-C checkpoint; building the transformers ESM-C backbone")
        from peint.models._esmc_biohub import build_esmc_biohub_backbone
        esm_model, vocab, _ = build_esmc_biohub_backbone(
            "esmc-biohub", use_flash=use_flash and FLASH_AVAILABLE
        )
    else:
        if is_peint_only:
            logger.info("Detected PEINT-only checkpoint (ESM parameters not included)")
            logger.info("Loading ESM2 from pretrained model...")
            esm_model = _load_esm_model(use_flash=use_flash and FLASH_AVAILABLE)
        else:
            # Create ESM model structure (weights will be loaded from checkpoint)
            logger.info("Detected full checkpoint (includes ESM parameters)")
            logger.info("Creating ESM model structure for full checkpoint")

            temp_esm, _ = esm.pretrained.esm2_t30_150M_UR50D()

            if use_flash and FLASH_AVAILABLE:
                logger.info("Using Flash ESM model")
                esm_model = ESM2Flash(
                    num_layers=temp_esm.num_layers,
                    embed_dim=temp_esm.embed_dim,
                    attention_heads=temp_esm.attention_heads,
                    alphabet='ESM-1b',
                    token_dropout=True,
                    dropout_p=0.0
                )
            else:
                logger.info("Using standard ESM model")
                from peint.models._flash_esm import ESM2Model
                esm_model = ESM2Model(
                    num_layers=temp_esm.num_layers,
                    embed_dim=temp_esm.embed_dim,
                    attention_heads=temp_esm.attention_heads,
                    alphabet='ESM-1b',
                    token_dropout=True,
                    dropout_p=0.0
                )

            del temp_esm
            # ESM weights will be loaded from checkpoint via model.load_state_dict() below

        # ESM2 vocabulary (ESM-C set its own vocab above)
        vocab = Alphabet.from_architecture("ESM-1b")

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

    # PeintLightningModule saved max_seq_len + optimizer-only hparams. ESM-C checkpoints come
    # from that trainer, so map/drop those keys for the plain module constructor; ESM2
    # checkpoints keep their existing (working) kwargs untouched.
    if is_esmc:
        model_kwargs = dict(hyper_params)
        if "max_seq_len" in model_kwargs:
            model_kwargs["max_len"] = model_kwargs.pop("max_seq_len")
        for _k in ("lr", "lora_lr", "num_warmup_steps", "num_training_steps", "which_esm"):
            model_kwargs.pop(_k, None)
    else:
        model_kwargs = hyper_params

    # Optionally resize the encoder/decoder positional caches (KV cache + RoPE) so long
    # sequences do not overrun the default 1024-wide buffers during generation. The decoder
    # generates up to 2*root tokens, so simulation passes a decoder length sized to that.
    if max_encoder_seq_len is not None or max_decoder_seq_len is not None:
        model_kwargs = dict(model_kwargs)
        if max_encoder_seq_len is not None:
            model_kwargs["max_encoder_seq_len"] = max_encoder_seq_len
        if max_decoder_seq_len is not None:
            model_kwargs["max_decoder_seq_len"] = max_decoder_seq_len

    # Create model instance
    try:
        model = model_class(
            esm_model=esm_model,
            esm_vocab=vocab,
            **model_kwargs
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
    use_flash: bool = FLASH_AVAILABLE,
    max_encoder_seq_len: Optional[int] = None,
    max_decoder_seq_len: Optional[int] = None
) -> Tuple[nn.Module, Alphabet]:
    """
    Legacy interface for loading PEINT models.

    This function maintains compatibility with existing code but uses the
    new load_peint_model function internally.

    Args:
        model_checkpoint_path: Path to checkpoint file
        use_cached_model: If True, use cached decoder variant (generator)
        device: Device to load model onto
        max_encoder_seq_len / max_decoder_seq_len: Optional overrides for the positional
            cache sizes (see load_peint_model); needed when generating sequences longer than
            the model's default 1024-wide decoder cache.

    Returns:
        Tuple of (model, vocabulary)
    """
    return load_peint_model(
        checkpoint_path=model_checkpoint_path,
        device=device,
        model_type='generator' if use_cached_model else 'standard',
        use_flash=use_flash,
        max_encoder_seq_len=max_encoder_seq_len,
        max_decoder_seq_len=max_decoder_seq_len
    )
