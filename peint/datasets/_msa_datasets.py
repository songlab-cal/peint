"""
Isolate out the functionalities to build the training dataset on any list of MSAs
"""

import os
from typing import List
from loguru import logger
from pathlib import Path
import cherryml
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from peint import utils
from peint.io import (
    read_msa,
    write_msa,
)
from peint.datasets._datasets import extract_transitions, alphabetize_msa

# Repo root (peint/), used only for the default WAG rate-matrix path in estimate_trees.
MAIN_DIR = Path(__file__).resolve().parents[2]


def _process_sequence(seq: str, return_full_length_unaligned: bool) -> str:
    """
    - If keep_original = True: keep the original sequence
    - If keep_original = False and return_full_length_unaligned = True:
        1) Convert the lowercase letters to uppercase and keep them
        2) Remove the gaps
    - If keep_original = False and return_full_length_unaligned = False:
        Remove the lowercase letters and keep the gaps
        The sequences should be of the same length

    Returns:
    - The processed sequence
    """

    def _make_upper_if_lower_and_clear_gaps(c: str) -> str:
        if c.islower():
            return c.upper()
        elif c.isupper():
            return c
        else:
            assert c == "-"
            return ""

    if return_full_length_unaligned:
        processed_seq = "".join([_make_upper_if_lower_and_clear_gaps(c) for c in seq])
    else:
        # Don't keep insertions wrt the reference and keep gaps
        processed_seq = "".join([c for c in seq if not c.islower()])

    return processed_seq


def _subsample_msa(
    family: str,
    input_msa_dir: str,
    output_msa_dir: str,
    max_num_sequences: int | None = 1024,
    seed: int = 42,
    return_full_length_unaligned: bool = True,
):
    output_msa_path = Path(output_msa_dir) / f"{family}.txt"
    if os.path.exists(output_msa_path):
        logger.info(f"Subsampled MSA exists for {family}, skipping")
        return

    # Assume the msa is in .txt format
    input_msa_path = Path(input_msa_dir) / f"{family}.txt"
    msa = read_msa(input_msa_path)
    nseqs = len(msa)

    # Subsample if needed
    if max_num_sequences is not None:
        rng = np.random.default_rng(seed)

        # Calculate how many sequences to keep
        max_seqs = min(nseqs, max_num_sequences)

        # Always keep query sequence (index 0) and randomly sample the rest
        indices_to_keep = [0] + sorted(
            rng.choice(range(1, nseqs), size=max_seqs - 1, replace=False).tolist()
        )
    else:
        # Keep all sequences
        indices_to_keep = range(nseqs)

    # If return aligned sequences, alphabetize MSA
    # to be consistent with trrosetta processing
    msa = alphabetize_msa(
        msa=msa,
        alphabet=tuple(list(utils.amino_acids) + [utils.gap_character]),
        out_of_alphabet_character=utils.gap_character,
    )

    subsampled_msa = {
        k: _process_sequence(
            v, return_full_length_unaligned=return_full_length_unaligned
        )
        for i, (k, v) in enumerate(msa.items())
        if i in indices_to_keep
    }
    write_msa(subsampled_msa, output_msa_path)


def estimate_trees(
    msa_dir: str,
    output_dir: str,
    num_rate_categories: int = 1,
    rate_matrix_path: str = os.path.join(MAIN_DIR, "data/rate_matrices/wag.txt"),
    num_processes: int = 1,
):
    """
    Estimate trees for all MSAs in a directory
    """
    # Get the file names of all MSAs
    msa_names = [
        file.split(".txt")[0] for file in os.listdir(msa_dir) if file.endswith(".txt")
    ]

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    output_tree_dir = os.path.join(output_dir, "output_tree_dir")
    output_site_rates_dir = os.path.join(output_dir, "output_site_rates_dir")
    output_likelihood_dir = os.path.join(output_dir, "output_likelihood_dir")
    os.makedirs(output_tree_dir, exist_ok=True)
    os.makedirs(output_site_rates_dir, exist_ok=True)
    os.makedirs(output_likelihood_dir, exist_ok=True)

    # FUTURE NOTE: since we wish to examine the intermediate outputs
    # we don't used the cached function
    fast_tree_bin = (
        cherryml.phylogeny_estimation._fast_tree._install_fast_tree_and_return_bin_path()
    )
    Parallel(n_jobs=num_processes)(
        delayed(
            cherryml.phylogeny_estimation._fast_tree.run_fast_tree_with_custom_rate_matrix
        )(
            msa_path=os.path.join(msa_dir, f"{msa_name}.txt"),
            family=msa_name,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
            output_tree_dir=output_tree_dir,
            output_site_rates_dir=output_site_rates_dir,
            output_likelihood_dir=output_likelihood_dir,
            extra_command_line_args="",
            fast_tree_bin=fast_tree_bin,
        )
        for msa_name in tqdm(msa_names)
    )


def construct_dataset_on_msas(
    families: List[str],
    input_msa_dir: str,
    output_msa_dir: str,
    output_tree_dir: str,
    output_transitions_dir: str,
    num_processes: int = 32,
    max_num_sequences: int = 1024,
    return_full_length_unaligned: bool = True,
):
    """ """
    logger.info(f"Construct datasets from {input_msa_dir} on {len(families)} families")

    # Initialize the output subdirectory names
    output_msa_dir = Path(output_msa_dir)
    output_tree_dir = Path(output_tree_dir)
    output_transitions_dir = Path(output_transitions_dir)

    # The subsampled MSAs and transitions are different depending on whether
    # we return the aligned and unaligned outputs
    # The tree directory will be the same
    if return_full_length_unaligned:
        output_msa_dir /= "unaligned"
        output_transitions_dir /= "unaligned"
    else:
        output_msa_dir /= "aligned"
        output_transitions_dir /= "aligned"
    output_msa_dir.mkdir(parents=True, exist_ok=True)
    output_tree_dir.mkdir(parents=True, exist_ok=True)
    output_transitions_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Subsampling MSA to {max_num_sequences} sequences, output written to {output_msa_dir}"
    )
    Parallel(n_jobs=num_processes)(
        delayed(_subsample_msa)(
            family=family,
            input_msa_dir=input_msa_dir,
            output_msa_dir=output_msa_dir,
            max_num_sequences=max_num_sequences,
            return_full_length_unaligned=return_full_length_unaligned,
        )
        for family in tqdm(families)
    )

    if not return_full_length_unaligned:
        # If the MSAs are aligned, estimate the trees
        logger.info(f"Estimating trees, output to {output_tree_dir}...")
        estimate_trees(
            msa_dir=output_msa_dir,
            output_dir=output_tree_dir,
            num_processes=num_processes,
        )

    logger.info(f"Extracting transitions, output to {output_transitions_dir}")
    extract_transitions(
        msa_dir=output_msa_dir,
        tree_dir=output_tree_dir / "output_tree_dir",
        families=families,
        num_processes=num_processes,
        include_gaps=(not return_full_length_unaligned),
        output_transitions_dir=output_transitions_dir,
    )
