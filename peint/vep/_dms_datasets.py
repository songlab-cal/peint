from pathlib import Path
from loguru import logger
import os

from peint import caching
from peint.vep._vep_utils import PROTEINGYM_DIR, VEP_DATA_DIR
from peint.vep.ProteinGym3.proteingym.baselines.SiteRM._datasets import (
    get_dms_substitutions_families,
)
from peint.datasets._msa_datasets import construct_dataset_on_msas


def construct_training_dataset_on_dms_msas(
    output_dir: str,
    families: list[str],
    input_msa_dir: str,
    num_processes: int = 32,
    max_num_sequences: int = 1024,
):
    """
    Args:
        input_msa_dir (str): Assume the MSAs in this directory are processed and
            stored as .txt files
    """
    logger.info(f"{len(families)} families, input_msa_dir={input_msa_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Storing outputs to {output_dir}")
    output_msa_dir = output_dir / "subsampled_msas"
    output_tree_dir = output_dir / "fast_tree"
    output_transitions_dir = output_dir / "transitions"

    logger.info("Get the datasets with aligned sequences")
    construct_dataset_on_msas(
        families=families,
        input_msa_dir=input_msa_dir,
        output_msa_dir=output_msa_dir,
        output_tree_dir=output_tree_dir,
        output_transitions_dir=output_transitions_dir,
        num_processes=num_processes,
        max_num_sequences=max_num_sequences,
        return_full_length_unaligned=False,
    )
    logger.info("Get the datasets with unaligned sequences")
    construct_dataset_on_msas(
        families=families,
        input_msa_dir=input_msa_dir,
        output_msa_dir=output_msa_dir,
        output_tree_dir=output_tree_dir,
        output_transitions_dir=output_transitions_dir,
        num_processes=num_processes,
        max_num_sequences=max_num_sequences,
        return_full_length_unaligned=True,
    )


def main():
    # Enable peint's on-disk caching. It is OFF by default (`_CACHE_DIR` is a
    # module global that nothing sets otherwise), and `extract_transitions` is a
    # cache-managed function: with caching on, the wrapper fills in its
    # alignment-mask / transition-name output dirs (which this builder doesn't
    # pass) and skips already-completed families on re-runs via `.success` tokens.
    cache_dir = VEP_DATA_DIR / "_cache_protevo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    caching.set_cache_dir(str(cache_dir))

    families = get_dms_substitutions_families(
        DMS_reference_file_path=PROTEINGYM_DIR
        / "reference_files"
        / "DMS_substitutions.csv"
    )
    output_dir = VEP_DATA_DIR / "training_data"
    name = "hhfilter90"
    output_dir = output_dir / name

    if name == "default":
        input_msa_dir = (
            PROTEINGYM_DIR
            / "_cache_cherryml/DMS_substitutions_dataset_create_training_msas/0387663ec7bc0597ab6ca8c3760b44fa24a245aca9eb0cf9c780cc5428e5cccb/output_msa_dir"
        )
    elif name == "hhfilter90":
        input_msa_dir = (
            PROTEINGYM_DIR
            / "_cache_cherryml/DMS_substitutions_dataset_create_training_msas/963b3a1ed2b3b910f893fc86313847297ce8860fd08486b0002321bbbde0b0f6/output_msa_dir"
        )
        # After running hhfilter, some family have a much smaller MSAs
        # Only use the family with >= 10 sequences
        min_num_sequences = 10
        families_filtered = []
        for family in families:
            input_msa_path = input_msa_dir / f"{family}.txt"
            if not os.path.exists(input_msa_path):
                continue
            with open(input_msa_path, "r") as file:
                count = sum(1 for line in file if line.startswith(">"))
            if count >= min_num_sequences:
                families_filtered.append(family)
        families = families_filtered

    construct_training_dataset_on_dms_msas(
        output_dir=output_dir, families=families, input_msa_dir=input_msa_dir
    )


if __name__ == "__main__":
    main()
