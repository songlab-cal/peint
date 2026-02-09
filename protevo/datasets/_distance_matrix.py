import os
import logging
import multiprocessing
from typing import List, Optional

import numpy as np
import tqdm

from biotite.structure.io.pdb import PDBFile
from scipy.spatial.distance import pdist, squareform

from protevo import caching as protevo_caching
from cherryml.caching import secure_parallel_output
from cherryml.utils import get_process_args

from protevo.io import write_distance_map

# below code is drafted from CherryML/cherryml/benchmarking/_contact_generation/ContactMatrix.py

def extend(a, b, c, L, A, D) -> float:
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])

class DistanceMatrix:
    """
    Creates a distance matrix from a PDB file.

    Reads the PDB file at f'{pdb_dir}/{protein_family_name}.pdb' and
    computes the distance matrix, which can be written out to a file 
    with the write_to_file method.

    Args:
        pdb_dir: Directory where the pdb structure files (.pdb) are found.
        protein_family_name: Name of the protein family.

    Attributes:
        nsites: Number of sites in the protein.
    """

    def __init__(
        self,
        pdb_dir: str,
        protein_family_name: str,
    ):
        pdb_file = os.path.join(pdb_dir, protein_family_name + ".pdb")
        pdbfile = PDBFile.read(str(pdb_file))
        structure = pdbfile.get_structure()
        N = structure.coord[0, structure.atom_name == "N"]
        C = structure.coord[0, structure.atom_name == "C"]
        CA = structure.coord[0, structure.atom_name == "CA"]
        Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
        self.dist_matrix = squareform(pdist(Cbeta))

    @property
    def nsites(self) -> int:
        r"""
        Number of sites in the sequence
        """
        assert self.dist_matrix.shape[0] == self.dist_matrix.shape[1]
        return self.dist_matrix.shape[0]

    def write_to_file(self, outfile: str) -> None:
        r"""
        Writes the distance matrix to outfile. Spaces are used as separators.
        """
        n = self.nsites
        cm = np.zeros(shape=(n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                cm[i, j] = self.dist_matrix[i, j]
        write_distance_map(cm, outfile)
        
# below code is drafted from CherryML/cherryml/benchmarking/data/pfam_15k.py 

def _compute_distance_map(
    pfam_15k_pdb_dir: str,
    family: str,
    output_distance_map_dir: str,
) -> None:
    output_distance_map_path = os.path.join(
        output_distance_map_dir, family + ".txt"
    )
    distance_matrix = DistanceMatrix(
        pdb_dir=pfam_15k_pdb_dir,
        protein_family_name=family,
    )
    distance_matrix.write_to_file(output_distance_map_path)
    secure_parallel_output(output_distance_map_dir, family)
    
def _map_func_compute_distance_maps(args: List) -> None:
    pfam_15k_pdb_dir = args[0]
    families = args[1]
    output_distance_map_dir = args[2]

    for family in families:
        _compute_distance_map(
            pfam_15k_pdb_dir=pfam_15k_pdb_dir,
            family=family,
            output_distance_map_dir=output_distance_map_dir,
        )
        
@protevo_caching.cached_parallel_computation(
    exclude_args=["num_processes"],
    parallel_arg="families",
    output_dirs=["output_distance_map_dir"],
    write_extra_log_files=True,
)
def compute_distance_maps(
    pfam_15k_pdb_dir: str,
    families: List[str],
    num_processes: int,
    output_distance_map_dir: Optional[str] = None,
):
    logger = logging.getLogger(__name__)
    logger.info(f"Going to compute distance maps for {len(families)} families")

    if not os.path.exists(pfam_15k_pdb_dir):
        raise ValueError(f"Could not find pfam_15k_pdb_dir {pfam_15k_pdb_dir}")

    map_args = [
        [
            pfam_15k_pdb_dir,
            get_process_args(process_rank, num_processes, families),
            output_distance_map_dir,
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_compute_distance_maps, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_compute_distance_maps, map_args),
                total=len(map_args),
            )
        )
