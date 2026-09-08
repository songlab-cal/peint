"""Tests for classical (AliSim / matrix-exponential) evolution simulation.

`evolve_classical` runs pure-Python (no external binary). The AliSim tests
require the iqtree2 submodule to be built (``iqtree2/build/iqtree2``) and are
skipped otherwise. Fixtures are a real family (``1ekj_1_C``): a cherryml-format
tree, its MAFFT-aligned empirical MSA (the correct ``-s`` input), and a root
sequence.
"""

import os

import pytest

from peint.simulation._alisim import find_iqtree2_path
from peint.utils import read_msa

SIM_DIR = os.path.join(os.path.dirname(__file__), "..", "peint", "tests", "alisim_test_dir")
TREE_DIR = os.path.join(SIM_DIR, "tree_dir")
MSA_DIR = os.path.join(SIM_DIR, "msa_dir")
ROOT_DIR = os.path.join(SIM_DIR, "root_sequences_dir")
FAMILY = "1ekj_1_C"

requires_iqtree = pytest.mark.skipif(
    not os.path.exists(find_iqtree2_path()),
    reason="iqtree2 build not found (build the iqtree2 submodule)",
)


@pytest.fixture
def isolated_cache(tmp_path):
    import peint.caching as caching

    caching.set_cache_dir(str(tmp_path / "_cache"))
    yield


def _run_alisim(model, mode, out_dir):
    from peint.simulation import simulate_alisim_evolution

    return simulate_alisim_evolution(
        tree_dir=TREE_DIR,
        msa_dir=MSA_DIR,
        root_seqs_dir=ROOT_DIR,
        families=[FAMILY],
        evolutionary_model=model,
        mode=mode,
        num_processes=1,
        insertion_rate=0,
        deletion_rate=0,
        output_msa_dir=str(out_dir),
    )["output_msa_dir"]


def test_classical_models_available():
    from peint.simulation import CLASSICAL_MODELS

    assert CLASSICAL_MODELS == ["WAG", "LG", "LG4X", "LG+C60", "LG+S256"]


def test_evolve_classical_pure_python():
    """CTMC single-sequence evolution needs no external binary."""
    from cherryml.markov_chain import get_lg_path

    from peint.simulation.classical import evolve_classical

    x = "ACDEFGHIKLMNPQRSTVWY"
    y = evolve_classical(x=x, t=0.5, rate_matrix_path=get_lg_path(), random_seed=0)
    assert len(y) == len(x)
    assert set(y) <= set("ACDEFGHIKLMNPQRSTVWY")


@requires_iqtree
def test_alisim_wag_inference(tmp_path, isolated_cache):
    res = _run_alisim("WAG", "inference", tmp_path / "out_wag")
    msa = read_msa(os.path.join(res, FAMILY + ".txt"))
    assert len(msa) > 0
    assert len({len(s) for s in msa.values()}) == 1  # aligned output
    # non-mixture model does not write a per-site mixture posterior
    assert not os.path.exists(os.path.join(res, FAMILY + ".siteprob"))


@requires_iqtree
@pytest.mark.slow
def test_alisim_s256_inference_writes_siteprob(tmp_path, isolated_cache):
    """LG+S256 must load the UDM profiles and write the .siteprob posterior
    consumed by the back-mutation / markov-cycling analysis."""
    res = _run_alisim("LG+S256", "inference", tmp_path / "out_s256")
    assert os.path.exists(os.path.join(res, FAMILY + ".txt"))
    assert os.path.exists(os.path.join(res, FAMILY + ".siteprob"))
