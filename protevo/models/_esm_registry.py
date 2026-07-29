"""Registry of supported pretrained backbones for PEINT.

This is the single source of truth for "given a backbone name, build the frozen
pretrained encoder module + vocabulary". Both training (``train_peint_model.py``)
and inference/loading (``_loading.py``) go through :func:`build_esm_backbone`, so
the backbone size is never hardcoded in one place and inferred in another.

Currently only the ESM2 UR50D family is registered. The ESM-C backbone is added
here in the ESM-C ablation (A3), behind its own optional import, so that
``encoder_backbone="esmc"`` slots in without touching call sites.
"""

import esm

# name -> (pretrained loader, embed_dim)
ESM2_REGISTRY = {
    "ESM2-8M":   (esm.pretrained.esm2_t6_8M_UR50D,    320),
    "ESM2-35M":  (esm.pretrained.esm2_t12_35M_UR50D,  480),
    "ESM2-150M": (esm.pretrained.esm2_t30_150M_UR50D, 640),
}

# ESM-C backbones (EvolutionaryScale `esm3` package). name -> (from_pretrained id,
# embed_dim). Only the 300M model (hidden size 960) is currently supported.
ESMC_REGISTRY = {
    "esmc":      ("esmc_300m", 960),
    "ESM-C":     ("esmc_300m", 960),
    "esmc_300m": ("esmc_300m", 960),
}


def get_esm_model(name):
    """Return ``(loader, embed_dim)`` for a registered ESM2 backbone name."""
    if name not in ESM2_REGISTRY:
        raise ValueError(
            f"Unknown ESM2 model '{name}'. Available: {list(ESM2_REGISTRY)}"
        )
    loader, embed_dim = ESM2_REGISTRY[name]
    return loader, embed_dim


def get_backbone_embed_dim(name):
    """Return the hidden/embed dim for any registered backbone (ESM2 or ESM-C).

    Backbone-agnostic (unlike :func:`get_esm_model`, which only knows the ESM2
    loaders): the ESM-C entries carry a ``from_pretrained`` id, not a callable, so
    the training launcher uses this to resolve ``embed_dim`` before building.
    """
    if name in ESM2_REGISTRY:
        return ESM2_REGISTRY[name][1]
    if name in ESMC_REGISTRY:
        return ESMC_REGISTRY[name][1]
    raise ValueError(
        f"Unknown backbone '{name}'. Available: "
        f"{list(ESM2_REGISTRY) + list(ESMC_REGISTRY)}"
    )


def _build_esmc_backbone(name):
    """Build an ESM-C backbone + its vocab. Requires the optional `esm3` package."""
    try:
        from esm.models.esmc import ESMC  # esm3 package exposes this as esm.models
    except ImportError:
        from esm3.models.esmc import ESMC  # older layout
    from protevo.datasets._vocab import get_esmc_vocab

    version, embed_dim = ESMC_REGISTRY[name]
    # ESM-C uses flash-attention internally, which only supports fp16/bf16, so the
    # backbone MUST stay bf16 (unlike the float32 ESM2 backbones). It is frozen, so it
    # needs no float32 master weights. The dtype boundary to the float32 PEINT layers
    # is handled where the backbone's outputs are consumed (see _transformer.py).
    module = ESMC.from_pretrained(version)
    return module, get_esmc_vocab(), embed_dim


def build_esm_backbone(name: str = "ESM2-150M", use_flash: bool = True):
    """Build the frozen pretrained backbone module + vocab for a registry name.

    Mirrors the construction used at training time so a checkpoint always rebuilds
    on the exact backbone it was trained with.

    Args:
        name: Registry key (e.g. ``"ESM2-8M"``/``"ESM2-35M"``/``"ESM2-150M"``).
        use_flash: If True and Flash Attention is importable, wrap in ``ESM2Flash``;
            otherwise fall back to the standard-attention ``ESM2Model``.

    Returns:
        Tuple ``(backbone_module, vocab, embed_dim)``. The module holds pretrained
        weights; PEINT freezes it (unless a fine-tuning mode is requested).
    """
    # ESM-C backbones use a different loader (esm3 package) and vocab.
    if name in ESMC_REGISTRY:
        return _build_esmc_backbone(name)

    # Lazy import to avoid any import-time coupling with the transformer modules.
    from esm.data import Alphabet

    from protevo.models._transformer_modules import FLASH_AVAILABLE

    loader, embed_dim = get_esm_model(name)
    esm_pretrained, loader_vocab = loader()

    # All ESM2 UR50D sizes share the ESM-1b alphabet. We (a) build the wrapper module
    # with alphabet="ESM-1b" and (b) return that same ESM-1b vocab, so every size uses
    # one consistent vocabulary and the token indices line up with the pretrained
    # embed_tokens / lm_head weights loaded below. Guard that the loader agrees, so a
    # future backbone with a different alphabet fails loudly instead of misaligning.
    vocab = Alphabet.from_architecture("ESM-1b")
    assert loader_vocab.to_dict() == vocab.to_dict(), (
        f"{name} alphabet differs from ESM-1b; token indices would misalign."
    )

    if use_flash and FLASH_AVAILABLE:
        from protevo.models._flash_esm import ESM2Flash

        module = ESM2Flash(
            num_layers=esm_pretrained.num_layers,
            embed_dim=esm_pretrained.embed_dim,
            attention_heads=esm_pretrained.attention_heads,
            alphabet="ESM-1b",
            token_dropout=True,
            dropout_p=0.0,  # ESM2 does not use dropout
        )
    else:
        from protevo.models._flash_esm import ESM2Model

        module = ESM2Model(
            num_layers=esm_pretrained.num_layers,
            embed_dim=esm_pretrained.embed_dim,
            attention_heads=esm_pretrained.attention_heads,
            alphabet="ESM-1b",
            token_dropout=True,
            dropout_p=0.0,
        )

    # The rotary-embedding buffers differ, so some keys are intentionally missing.
    module.load_state_dict(esm_pretrained.state_dict(), strict=False)
    del esm_pretrained

    return module, vocab, embed_dim
