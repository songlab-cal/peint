import os
import logging
import multiprocessing
from typing import List, Optional

from protevo import caching as protevo_caching
from protevo.caching import secure_parallel_output
from protevo.utils import get_process_args
from protevo import io

import tqdm

from Bio.PDB import PDBParser, DSSP


def _compute_secondary_structure_annotations(
    pdb_dir: str,
    family: str,
    output_dir: str,
) -> None:
    output_path = os.path.join(
        output_dir, family + ".txt"
    )

    # Load the structure from a PDB file
    pdb_parser = PDBParser()

    pdb_path = os.path.join(pdb_dir, family + ".pdb")
    structure = pdb_parser.get_structure(family, pdb_path)

    # Assuming your structure has only one model, select the first model
    model = structure[0]

    # Run DSSP, requires DSSP executable available in your system
    dssp = DSSP(model, pdb_path, dssp='dssp')

    # Initialize an empty string to hold the secondary structure sequence
    secondary_structure = ''

    # Loop through DSSP output to build the secondary structure string
    for res in dssp:
        # 'res' looks like so:
        # (182, 'G', 'H', 0.47619047619047616, -58.6, -52.5, -4, -2.1, 4,
        #  -1.7, -5, -0.2, -2, -0.2)
        # First 3 entries are: index (1-based), amino-acid, structure.
        # DSSP codes: H = alpha helix, B = beta bridge, E = extended strand
        # (part of beta sheet), etc.
        dssp_key = res[2]
        secondary_structure += dssp_key

    io.write_secondary_structure(secondary_structure, output_path)
    secure_parallel_output(output_dir, family)


def _map_func_compute_secondary_structure_annotations(args: List) -> None:
    pdb_dir = args[0]
    families = args[1]
    output_dir = args[2]

    for family in families:
        _compute_secondary_structure_annotations(
            pdb_dir=pdb_dir,
            family=family,
            output_dir=output_dir,
        )


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=["output_dir"],
)
def compute_secondary_structure_annotations(
    pdb_dir: str,
    families: List[str],
    num_processes: int,
    output_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Going to compute secondary structures for {len(families)} families")

    if not os.path.exists(pdb_dir):
        raise ValueError(f"Could not find pdb_dir {pdb_dir}")

    map_args = [
        [
            pdb_dir,
            get_process_args(process_rank, num_processes, families),
            output_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_compute_secondary_structure_annotations, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_compute_secondary_structure_annotations, map_args),
                total=len(map_args),
            )
        )
