import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import tqdm
from typing import List, Optional
import subprocess

from Bio import SeqIO
import cherryml
from ete3 import Tree

from protevo import caching as protevo_caching
from protevo.caching import secure_parallel_output
from protevo.utils import read_msa
from protevo.utils import get_process_args

_UDM_NEX_PATH = str(Path(__file__).parent / "assets" / "udm_hogenom_0256_lclr_iqtree.nex")

# Stereotyped iqtree2 definitions for each benchmark model. Each entry specifies
# the exact extra args appended to the alisim command. `site_rate_model` toggles
# `--site-rate MODEL` which makes iqtree2 estimate per-site rates via empirical
# Bayes posterior from the input MSA+tree under the given model.
# `write_mixture_posterior` toggles `-wspm`, which writes the per-site posterior over
# the model's mixture classes (the 256 UDM profiles for S256) to a .siteprob file.
# Only meaningful for profile-mixture models; used to quantify per-site profile entropy.
MODEL_DEFINITIONS = {
    "WAG":     {"extra_args": ["-m", "WAG"],                                       "site_rate_model": False},
    "LG":      {"extra_args": ["-m", "LG+G4"],                                     "site_rate_model": True},
    "LG4X":    {"extra_args": ["-m", "LG4X"],                                      "site_rate_model": True},
    "LG+C60":  {"extra_args": ["-m", "LG+C60"],                                    "site_rate_model": True, "write_mixture_posterior": True},
    "LG+S256": {"extra_args": ["-mdef", _UDM_NEX_PATH, "-m", "LG+UDM0256LCLR"],    "site_rate_model": True, "write_mixture_posterior": True},
}
CLASSICAL_MODELS = list(MODEL_DEFINITIONS.keys())

# Models for which prior_anchored simulation mode is supported. Restricted to
# profile-mixture models — the empirical leak in inference mode is dominated by
# per-site posterior profile assignment, which only applies here. Non-mixture
# models would need explicit rate-heterogeneity values (e.g., +G4 shape) when
# run without -s, which is out of scope.
PRIOR_MODE_SUPPORTED_MODELS = {"LG+C60", "LG+S256"}

ALISIM_MODES = ("inference", "prior_anchored")

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

def find_iqtree2_path():

    iqtree_executable = Path(__file__).parent.parent.parent / "iqtree2" / "build" / "iqtree2"

    return iqtree_executable


def _get_gapless_root_length(msa: dict, root_id: str) -> int:
    """Return the number of non-gap residues in the root sequence."""
    return len(msa[root_id]) - msa[root_id].count('-')


def _overlay_empirical_gaps(
    simulated_msa: dict,
    empirical_msa: dict,
    root_id: str,
) -> dict:
    """
    Map a gapless simulated MSA back into the empirical alignment frame using the
    root's gap pattern. Every leaf gets the same gap pattern as the root: residues
    at the root's residue columns, gaps at the root's gap columns. Substitution-only
    simulation cannot synthesize residues at columns where the root was gapped, so
    per-leaf empirical gap patterns are *not* preserved — each leaf shares the
    root's frame and has exactly `gapless_root_length` residues.
    """
    emp_root = empirical_msa[root_id]
    root_res_cols = [i for i, c in enumerate(emp_root) if c != '-']
    emp_length = len(emp_root)

    output = {}
    for name, sim_seq in simulated_msa.items():
        if name not in empirical_msa:
            continue
        if len(sim_seq) != len(root_res_cols):
            raise ValueError(
                f"Length mismatch for {name}: simulated has {len(sim_seq)} residues, "
                f"empirical root has {len(root_res_cols)} non-gap positions"
            )
        new_seq = ['-'] * emp_length
        for i, col in enumerate(root_res_cols):
            new_seq[col] = sim_seq[i]
        output[name] = ''.join(new_seq)
    return output


def _build_alisim_cmd(
    *,
    mode: str,
    iqtree2_path,
    output_path: str,
    tree_file: str,
    inference_msa_path: Optional[str],
    inference_root_seq_arg: Optional[str],
    prior_root_seq_arg: Optional[str],
    root_id: str,
    gapless_length: int,
    model_def: dict,
    insertion_rate: float,
    deletion_rate: float,
    use_outgroup: bool,
) -> list:
    """Build an iqtree2 --alisim command for the requested mode.

    `inference_root_seq_arg` and `prior_root_seq_arg` are pre-formatted
    "<path>,<id>" strings (None when not used). `use_outgroup` is True for the
    full-tree path (passes `-o root_id`) and False for subtree simulations.
    """
    base = [
        iqtree2_path,
        "--alisim", output_path,
        "-te", tree_file,
        "--out-format", "fasta",
        "-seed", "42",
    ]

    if mode == "inference":
        cmd = base + [
            "-s", inference_msa_path,
            "--root-seq", inference_root_seq_arg,
            "--write-all",
        ]
        if use_outgroup:
            cmd += ["-o", root_id]
        if model_def.get("site_rate_model"):
            cmd += ["--site-rate", "MODEL"]
        if model_def.get("write_mixture_posterior"):
            # Per-site posterior over mixture classes -> .siteprob (needs the -s inference pass).
            cmd += ["-wspm"]

    elif mode == "prior_anchored":
        cmd = base + [
            "--root-seq", prior_root_seq_arg,
            "--length", str(gapless_length),
        ]
        # No -s, no --site-rate (requires -s), no --write-all (no internal nodes
        # are written without -s when indel rate is 0; per alisim.cpp:1501).

    else:
        raise ValueError(f"Unknown mode: {mode}")

    cmd += model_def["extra_args"]

    if insertion_rate + deletion_rate > 0:
        cmd += [
            "--indel",
            f"{insertion_rate}, {deletion_rate}",
            "--no-unaligned",
        ]

    return cmd


def _simulate_alisim(
    tree_dir: str,
    msa_dir: str,
    root_seqs_dir: str,
    family: str,
    evolutionary_model: str,
    mode: str = "inference",
    insertion_rate: Optional[float] = 1e-2,
    deletion_rate: Optional[float] = 1e-2,
    output_msa_dir: str = None
):
    iqtree2_path = find_iqtree2_path()

    tree_path = os.path.join(tree_dir, family + ".txt")
    msa_path_orig = os.path.join(msa_dir, family + ".txt")
    output_path = os.path.join(output_msa_dir, family)

    if os.path.exists(f"{output_path}.success") and os.path.exists(f"{output_path}.txt"):
        return

    ete_tree = Tree(tree_path, format=1)
    if root_seqs_dir:
        root_seq_path = os.path.join(root_seqs_dir, family + ".txt")
        with open(root_seq_path, 'r') as f:
            root_id = f.readline().strip()[1:]  # extract identifier from >{identifier}
    else:
        root_id = ete_tree.name
        if not root_id:
            raise ValueError(
                f"Tree for {family} has an unnamed root; pass root_seqs_dir "
                f"or provide a tree with a named root node."
            )
    ete_tree.set_outgroup(ete_tree & root_id)

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.nwk') as tree_file:
        tree_file.write(ete_tree.write(format=1))
        tree_file = tree_file.name

    # Empirical MSA, with PEINT-added sequences stripped, used for gapless length,
    # gap overlay, and (in inference mode) as the -s input.
    empirical_msa = read_msa(msa_path_orig)
    empirical_msa = {k: v for k, v in empirical_msa.items() if not k.endswith('_added')}
    gapless_length = _get_gapless_root_length(empirical_msa, root_id)

    inference_msa_path = None
    inference_root_seq_arg = None
    prior_root_seq_arg = None
    cleanup_msa_artifacts = False

    if mode == "inference":
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as msa_file:
            for k, v in empirical_msa.items():
                msa_file.write(f">{k}\n{v}\n")
            inference_msa_path = msa_file.name
        inference_root_seq_arg = f"{inference_msa_path},{root_id}"
        cleanup_msa_artifacts = True
    elif mode == "prior_anchored":
        # Prefer the existing root_seqs_dir file if available; else write a temp
        # gapless root from the empirical MSA. Either way the file is gapless-only.
        if root_seqs_dir:
            prior_root_path = os.path.join(root_seqs_dir, family + ".txt")
        else:
            root_residues = empirical_msa[root_id].replace('-', '')
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fa') as f:
                f.write(f">{root_id}\n{root_residues}\n")
                prior_root_path = f.name
        prior_root_seq_arg = f"{prior_root_path},{root_id}"

    model_def = MODEL_DEFINITIONS[evolutionary_model]
    alisim_cmd = _build_alisim_cmd(
        mode=mode,
        iqtree2_path=iqtree2_path,
        output_path=output_path,
        tree_file=tree_file,
        inference_msa_path=inference_msa_path,
        inference_root_seq_arg=inference_root_seq_arg,
        prior_root_seq_arg=prior_root_seq_arg,
        root_id=root_id,
        gapless_length=gapless_length,
        model_def=model_def,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        use_outgroup=True,
    )

    result = subprocess.run(alisim_cmd, capture_output=True, text=True)

    alisim_output = os.path.join(output_msa_dir, family + '.fa')
    family_output = os.path.join(output_msa_dir, family + '.txt')
    if not os.path.exists(alisim_output):
        raise RuntimeError(
            f"AliSim did not produce expected output {alisim_output}.\n"
            f"Command: {' '.join(str(x) for x in alisim_cmd)}\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if mode == "prior_anchored":
        sim_msa = read_msa(alisim_output)
        gapped_msa = _overlay_empirical_gaps(sim_msa, empirical_msa, root_id)
        with open(alisim_output, 'w') as f:
            for name, seq in gapped_msa.items():
                f.write(f">{name}\n{seq}\n")

    os.rename(alisim_output, family_output)

    if cleanup_msa_artifacts:
        # Preserve the per-site mixture posterior (-wspm) into the output dir before
        # removing the other inference side-effects.
        siteprob_src = inference_msa_path + '.siteprob'
        if os.path.exists(siteprob_src):
            shutil.move(siteprob_src, output_path + '.siteprob')
        # IQTree writes side-effect files next to the -s input; clean those up.
        undesired_output_formats = ['.iqtree', '.treefile', '.log', '.ckp.gz']
        for fmt in undesired_output_formats:
            fpath = inference_msa_path + fmt
            if os.path.exists(fpath):
                os.remove(fpath)

    secure_parallel_output(output_msa_dir, family)

def _map_func_simulate_alisim(args):
    tree_dir, msa_dir, root_seqs_dir, families, evolutionary_model, mode, \
        insertion_rate, deletion_rate, output_msa_dir = args

    for family in families:
        _simulate_alisim(
            tree_dir=tree_dir,
            msa_dir=msa_dir,
            family=family,
            evolutionary_model=evolutionary_model,
            mode=mode,
            root_seqs_dir=root_seqs_dir,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            output_msa_dir=output_msa_dir
        )

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    exclude_args_if_default=["root_seqs_dir", "mode"],
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def simulate_alisim_evolution(
    tree_dir: str,
    msa_dir: str,
    root_seqs_dir: str,
    families: List[str],
    evolutionary_model: str,
    mode: str = "inference",
    num_processes: int = 1,
    insertion_rate: Optional[float] = 0,
    deletion_rate: Optional[float] = 0,
    output_msa_dir: Optional[str] = None,
):
    assert evolutionary_model in CLASSICAL_MODELS, f"Evolutionary model {evolutionary_model} isn't supported, please pass in one of {CLASSICAL_MODELS}"
    assert mode in ALISIM_MODES, f"Unknown mode: {mode}. Expected one of {ALISIM_MODES}"
    if mode != "inference":
        assert evolutionary_model in PRIOR_MODE_SUPPORTED_MODELS, (
            f"Prior modes are only supported for {sorted(PRIOR_MODE_SUPPORTED_MODELS)}, "
            f"got {evolutionary_model}. Other models would need explicit parameter values "
            f"(e.g., +G4 shape) to run without -s."
        )
    assert insertion_rate >= 0 and deletion_rate >= 0, "Received negative insertion and/or deletion rates"

    logger = logging.getLogger(__name__)
    logger.info(f"Simulating evolution using AliSim with {evolutionary_model} (mode={mode}) for {len(families)} families")

    if output_msa_dir is not None and not os.path.exists(output_msa_dir):
        os.mkdir(output_msa_dir)

    map_args = [
        [
            tree_dir,
            msa_dir,
            root_seqs_dir,
            get_process_args(process_rank, num_processes, families),
            evolutionary_model,
            mode,
            insertion_rate,
            deletion_rate,
            output_msa_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_simulate_alisim, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_simulate_alisim, map_args),
                total=len(map_args),
            )
        )

def _get_site_rates(
    msa_dir: str,
    tree_dir: str,
    family: str,
    model: str="LG+G4",
    output_site_rates_dir: Optional[str] = None
):
    """Precalculates site rates using IQTree2 for a given MSA and tree.
    The reason this is required for the Historian inputs is that AliSim likes
    to recalculate the tree during site rate estimation which leads to different
    outputs than expected (we want to simulate on the same tree structure).
    
    Instead - first calculate site rates across the alignment sites using IQTree2.
    Then we can pass these site rates into AliSim during simulation without the MSA,
    which causes AliSim to recalculate the tree.

    Finally, we'll just copy the GAP patterns from the original MSA into the simulated MSA
    """
    iqtree2_path = find_iqtree2_path()

    dirs = [(tree_dir, msa_dir, output_site_rates_dir)]

    for (tree_dir, msa_dir, output_dir) in dirs:
        tree_path = os.path.join(tree_dir, family + ".txt")
        msa_path = os.path.join(msa_dir, family + ".txt")
        output_path = os.path.join(output_dir, family)

        if os.path.exists(f"{output_path}.site_rates"):
            continue


        #-t 1a2t_1_A.txt -s 1a2t.fasta --tree-fix --rate -m LG+G4
        iqtree_cmd = [
            iqtree2_path,
            "-s", msa_path,
            "-t", tree_path,
            "--tree-fix",
            "--rate", 
            "-m", model,
            "--prefix", output_path,
            "-seed", "42"
        ]

        result=subprocess.run(iqtree_cmd, capture_output=True, text=True)

    secure_parallel_output(output_site_rates_dir, family)

def _simulate_alisim_subtree(
    msa_dir: str,
    tree_dir: str,
    family: str,
    evolutionary_model: str,
    mode: str = "inference",
    insertion_rate: Optional[float] = 1e-2,
    deletion_rate: Optional[float] = 1e-2,
    site_rates_dir: Optional[str] = None,
    output_msa_dir: str = None
):
    iqtree2_path = find_iqtree2_path()

    tree_path = os.path.join(tree_dir, family + ".txt")
    msa_path = os.path.join(msa_dir, family + ".txt")
    output_path = os.path.join(output_msa_dir, family)

    if os.path.exists(f"{output_path}.success") and os.path.exists(f"{output_path}.txt"):
        return

    msa = read_msa(msa_path)
    tree = Tree(tree_path, format=1)

    # AliSim has issues with the long internal- names; rename internal-N -> iN.
    for n in tree.traverse():
        if n.name.startswith('internal'):
            suffix = n.name.split('-')[-1]
            n.name = 'i' + suffix

    tree_string = tree.write(format=3).replace(';', f"{tree.name};")  # restore the root label

    new_msa = {k.replace('internal-', 'i'): v.upper() for k, v in msa.items()}
    root_id = tree.name
    root_seq_full = new_msa[root_id]
    gapless_length = len(root_seq_full) - root_seq_full.count('-')

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.nwk') as tree_file:
        tree_file.write(tree_string)
        tree_file_path = tree_file.name

    inference_msa_path = None
    inference_root_seq_arg = None
    prior_root_seq_arg = None
    cleanup_target = None

    # The leaves-only file is needed in inference mode (as -s input). The root file
    # is needed in inference and prior_anchored modes (as --root-seq input). For
    # prior_anchored, --root-seq strips gaps server-side, but we hand it a gapless
    # file regardless to remove ambiguity.
    if mode == "inference":
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as msa_file:
            for k, v in new_msa.items():
                if k.startswith('seq'):
                    msa_file.write(f">{k}\n{v}\n")
            inference_msa_path = msa_file.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as root_file:
            root_file.write(f">{root_id}\n{root_seq_full}\n")
            root_file_path = root_file.name
        inference_root_seq_arg = f"{root_file_path},{root_id}"
        cleanup_target = inference_msa_path
    elif mode == "prior_anchored":
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as root_file:
            root_file.write(f">{root_id}\n{root_seq_full.replace('-', '')}\n")
            root_file_path = root_file.name
        prior_root_seq_arg = f"{root_file_path},{root_id}"

    model_def = MODEL_DEFINITIONS[evolutionary_model]
    alisim_cmd = _build_alisim_cmd(
        mode=mode,
        iqtree2_path=iqtree2_path,
        output_path=output_path,
        tree_file=tree_file_path,
        inference_msa_path=inference_msa_path,
        inference_root_seq_arg=inference_root_seq_arg,
        prior_root_seq_arg=prior_root_seq_arg,
        root_id=root_id,
        gapless_length=gapless_length,
        model_def=model_def,
        insertion_rate=insertion_rate,
        deletion_rate=deletion_rate,
        use_outgroup=False,
    )

    result = subprocess.run(alisim_cmd, capture_output=True, text=True)

    alisim_output = os.path.join(output_msa_dir, family + '.fa')
    family_output = os.path.join(output_msa_dir, family + '.txt')
    if not os.path.exists(alisim_output):
        raise RuntimeError(
            f"AliSim did not produce expected output {alisim_output}.\n"
            f"Command: {' '.join(str(x) for x in alisim_cmd)}\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if mode == "prior_anchored":
        sim_msa = read_msa(alisim_output)
        leaf_emp_msa = {k: v for k, v in new_msa.items() if k.startswith('seq')}
        # Add the root so _overlay_empirical_gaps can reference its gap pattern
        leaf_emp_msa[root_id] = root_seq_full
        gapped_msa = _overlay_empirical_gaps(sim_msa, leaf_emp_msa, root_id)
        with open(alisim_output, 'w') as f:
            for name, seq in gapped_msa.items():
                f.write(f">{name}\n{seq}\n")

    os.rename(alisim_output, family_output)

    if cleanup_target is not None:
        # Preserve the per-site mixture posterior (-wspm) into the output dir before
        # removing the other inference side-effects.
        siteprob_src = cleanup_target + '.siteprob'
        if os.path.exists(siteprob_src):
            shutil.move(siteprob_src, output_path + '.siteprob')
        undesired_output_formats = ['.iqtree', '.treefile', '.log', '.ckp.gz']
        for fmt in undesired_output_formats:
            fpath = cleanup_target + fmt
            if os.path.exists(fpath):
                os.remove(fpath)

    secure_parallel_output(output_msa_dir, family)

def _map_func_simulate_alisim_subtree(args):
    tree_dir, msa_dir, families, evolutionary_model, mode, \
        insertion_rate, deletion_rate, output_msa_dir = args

    for family in families:
        _simulate_alisim_subtree(
            tree_dir=tree_dir,
            msa_dir=msa_dir,
            family=family,
            evolutionary_model=evolutionary_model,
            mode=mode,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            output_msa_dir=output_msa_dir
        )

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    exclude_args_if_default=["insertion_rate", "deletion_rate", "mode"],
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def simulate_alisim_evolution_subtree(
    tree_dir: str,
    msa_dir: str,
    families: List[str],
    evolutionary_model: str,
    mode: str = "inference",
    num_processes: int = 1,
    insertion_rate: Optional[float] = 0,
    deletion_rate: Optional[float] = 0,
    output_msa_dir: Optional[str] = None,
):
    assert evolutionary_model in CLASSICAL_MODELS, f"Evolutionary model {evolutionary_model} isn't supported, please pass in one of {CLASSICAL_MODELS}"
    assert mode in ALISIM_MODES, f"Unknown mode: {mode}. Expected one of {ALISIM_MODES}"
    if mode != "inference":
        assert evolutionary_model in PRIOR_MODE_SUPPORTED_MODELS, (
            f"Prior modes are only supported for {sorted(PRIOR_MODE_SUPPORTED_MODELS)}, "
            f"got {evolutionary_model}. Other models would need explicit parameter values "
            f"(e.g., +G4 shape) to run without -s."
        )
    assert insertion_rate >= 0 and deletion_rate >= 0, "Received negative insertion and/or deletion rates"

    logger = logging.getLogger(__name__)
    logger.info(f"Simulating evolution using AliSim with {evolutionary_model} (mode={mode}) for {len(families)} families")

    if output_msa_dir is not None and not os.path.exists(output_msa_dir):
        os.mkdir(output_msa_dir)

    map_args = [
        [
            tree_dir,
            msa_dir,
            get_process_args(process_rank, num_processes, families),
            evolutionary_model,
            mode,
            insertion_rate,
            deletion_rate,
            output_msa_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_simulate_alisim_subtree, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_simulate_alisim_subtree, map_args),
                total=len(map_args),
            )
        )