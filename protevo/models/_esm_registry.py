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


def get_esm_model(name):
    """Return ``(loader, embed_dim)`` for a registered ESM2 backbone name."""
    if name not in ESM2_REGISTRY:
        raise ValueError(
            f"Unknown ESM2 model '{name}'. Available: {list(ESM2_REGISTRY)}"
        )
    loader, embed_dim = ESM2_REGISTRY[name]
    return loader, embed_dim


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
    # Lazy import to avoid any import-time coupling with the transformer modules.
    from protevo.models._transformer_modules import FLASH_AVAILABLE

    loader, embed_dim = get_esm_model(name)
    esm_pretrained, vocab = loader()

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
