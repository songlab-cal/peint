"""Registry of supported base ESM2 models for PEINT training."""

import esm

ESM2_REGISTRY = {
    "ESM2-8M":   (esm.pretrained.esm2_t6_8M_UR50D,    320),
    "ESM2-35M":  (esm.pretrained.esm2_t12_35M_UR50D,  480),
    "ESM2-150M": (esm.pretrained.esm2_t30_150M_UR50D, 640),
}


def get_esm_model(name):
    if name not in ESM2_REGISTRY:
        raise ValueError(
            f"Unknown ESM2 model '{name}'. Available: {list(ESM2_REGISTRY)}"
        )
    loader, embed_dim = ESM2_REGISTRY[name]
    return loader, embed_dim


# Biohub ESM-C (transformers) — a separate base-LM model, registered here additively.
# Its loader/vocab live in `_esmc_biohub` and need the Biohub transformers fork (not the
# `esm` package). Imports there are lazy, so this line pulls no transformers at import time.
from peint.models._esmc_biohub import (  # noqa: E402,F401
    ESMC_BIOHUB_REGISTRY,
    build_esmc_biohub_backbone,
    get_esmc_biohub_vocab,
)
