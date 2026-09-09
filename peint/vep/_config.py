"""Central configuration for the VEP module.

Single source of truth for filesystem paths, CherryML cache hashes, default time
grids, and the encoder registry. Other modules should import from here rather than
hard-coding paths/hashes. ``_vep_utils`` re-exports the path constants for backward
compatibility, so existing ``from peint.vep._vep_utils import VEP_DATA_DIR`` keeps
working.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# Repo root: .../protein-evolution
MAIN_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = MAIN_DIR / "data/local_data/peint_data"        # symlink -> /scratch
VEP_DATA_DIR = DATA_DIR / "vep"
# ProteinGym checkout: only needed to rebuild VEP inputs or compare against ProteinGym's
# own baselines. Not required for scoring, training, or any published panel.
PROTEINGYM_DIR = Path(
    os.environ.get("PEINT_PROTEINGYM_DIR", str(MAIN_DIR / "peint/vep/ProteinGym3"))
)
CHECKPOINTS_DIR = MAIN_DIR / "data/local_data/checkpoints"  # symlink -> /scratch

# Results layout. Scored predictions live under test_lls,
# split into a curated `production/` set and an `archive/` set of superseded runs.
TEST_LLS_DIR = VEP_DATA_DIR / "test_lls"
TEST_LLS_PRODUCTION_DIR = TEST_LLS_DIR / "production"
TEST_LLS_ARCHIVE_DIR = TEST_LLS_DIR / "archive"

# ---------------------------------------------------------------------------
# CherryML cache hashes
# ---------------------------------------------------------------------------
# These are content hashes of CherryML cache entries. They are stable as long as the
# upstream MSA/transition config does not change; if CherryML caching config changes,
# update them here (single place) rather than across files.

# Test transitions for ProteinGym DMS substitutions (wt/mutant pairs fed to PEINT).
TEST_TRANSITIONS_HASH = (
    "3a13efc22507796bfec09150d959917b6e034c3c29639e25c0cadf0f02921d4c"
)
TEST_TRANSITIONS_DIR = (
    PROTEINGYM_DIR
    / "_cache_cherryml/create_test_transition_pairs"
    / TEST_TRANSITIONS_HASH
    / "output_transition_pairs_dir"
)

# Training MSAs for the DMS families (input to transition extraction in _dms_datasets).
TRAINING_MSA_HASHES = {
    "default": "0387663ec7bc0597ab6ca8c3760b44fa24a245aca9eb0cf9c780cc5428e5cccb",
    "hhfilter90": "963b3a1ed2b3b910f893fc86313847297ce8860fd08486b0002321bbbde0b0f6",
}


def training_msa_dir(name: str = "hhfilter90") -> Path:
    """CherryML output MSA dir for a given training-set variant ('default'|'hhfilter90')."""
    return (
        PROTEINGYM_DIR
        / "_cache_cherryml/DMS_substitutions_dataset_create_training_msas"
        / TRAINING_MSA_HASHES[name]
        / "output_msa_dir"
    )


# Canonical training-transition source (the other one, dms_transitions/, is deprecated).
CANONICAL_TRANSITIONS = "hhfilter90"


def training_transitions_dir(name: str = CANONICAL_TRANSITIONS, aligned: bool = False) -> Path:
    """Transitions extracted from DMS-family MSAs (output of _dms_datasets.py)."""
    sub = "aligned" if aligned else "unaligned"
    return VEP_DATA_DIR / "training_data" / name / "transitions" / sub


# Reference files used throughout scoring/plotting.
DMS_SUBSTITUTIONS_REF = PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"
DMS_DATA_FOLDER = (
    PROTEINGYM_DIR
    / "input_data/ProteinGym_v1.3/DMS_ProteinGym_substitutions/DMS_ProteinGym_substitutions"
)

# ---------------------------------------------------------------------------
# Time grids
# ---------------------------------------------------------------------------
# Default sweep used for per-family-vs-time analyses and figures.
DEFAULT_TIME_GRID = [
    0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
]
# Default single time for VEP scoring (Prillo et al.).
DEFAULT_T = 1.0

# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------
# Maps an encoder key to how its frozen pLM encoder is built. Stock ESM2 sizes load
# from fair-esm; `vesm_*` keys load vESM weights (HuggingFace EsmForMaskedLM) and
# convert them into the fair-esm ESM2Flash layout (see encoders.py). vESM_650M is
# architecturally identical to ESM2-650M, so it reuses the 650M scaffold.
ESM_REGISTRY = {
    "150M": {"fair_esm": "esm2_t30_150M_UR50D", "embed_dim": 640},
    "650M": {"fair_esm": "esm2_t33_650M_UR50D", "embed_dim": 1280},
    "3B": {"fair_esm": "esm2_t36_3B_UR50D", "embed_dim": 2560},
    "15B": {"fair_esm": "esm2_t48_15B_UR50D", "embed_dim": 5120},
    "vesm_150M": {
        "base": "150M",
        "embed_dim": 640,
        "hf_repo": "ntranoslab/vesm",
        "hf_file": "VESM_150M.pth",
    },
    "vesm_650M": {
        "base": "650M",
        "embed_dim": 1280,
        "hf_repo": "ntranoslab/vesm",
        "hf_file": "VESM_650M.pth",
    },
    # Biohub ESM-C (transformers) — a separate base-LM. load_esm_model delegates to
    # peint.models._esmc_biohub.build_esmc_biohub_backbone; needs the transformers fork.
    "esmc-biohub": {"backbone": "esmc-biohub", "embed_dim": 960},
}

# Local staging dir for downloaded encoder weights (on shared /scratch, so SLURM compute
# nodes need no internet). encoders._load_vesm prefers a file here over hf_hub_download.
ENCODER_STAGING_DIR = VEP_DATA_DIR / "encoders"

# Convenience: encoder key -> embed_dim, and the reverse map used by load_model when a
# checkpoint only records embed_dim (stock ESM sizes are unambiguous by embed_dim).
EMBED_DIM = {k: v["embed_dim"] for k, v in ESM_REGISTRY.items()}
EMBED_DIM_TO_ESM = {640: "150M", 1280: "650M", 2560: "3B", 5120: "15B", 960: "esmc-biohub"}
