import os
from typing import Optional, List, Dict
from Bio import SeqIO
import numpy as np
import pandas as pd
import cherryml
from protevo.io import read_rate_matrix, read_site_rates, read_tree
from protevo import utils
from protevo.utils import get_process_args, matrix_exponential_reversible, read_msa, write_msa
from protevo import caching as protevo_caching
from protevo.caching import secure_parallel_output
from protevo.datasets import run_mafft
import logging
import multiprocessing 
import tqdm

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

def evolve_classical(
    x: str,
    t: float,
    rate_matrix_path: str,
    random_seed: Optional[int] = None,
    site_rates: Optional[List[float]] = None
) -> str:
    """
    Evolve a protein sequence `x` under the WAG model.

    Args:
        x: The starting sequence.
        t: The amount of time.

        random_seed: If an int is passed in, this function will give reproducible results.
        rate_matrix_path: Path to the WAG rate matrix.

    Returns:
        The sampled sequence `y` obtained from evolving `x` for time `t` under the WAG model.
    """
    if t == 0.0:
        return x
    if not os.path.exists(rate_matrix_path):
        raise FileNotFoundError(
            f"Make sure your rate matrix is located at {rate_matrix_path}. "
            f"You can download the WAG rate matrix from https://github.com/songlab-cal/CherryML/blob/main/data/rate_matrices/wag.txt"
        )
    rate_matrix = read_rate_matrix(rate_matrix_path)
    if site_rates is None:
        site_rates = [1.0] * len(x)

    y_list = []
    rng = np.random.default_rng(seed=random_seed)

    for i, x_i in enumerate(x):
        if x_i == '-':
            y_list.append('-')
            continue 

        matrix_exponentials = matrix_exponential_reversible(
            rate_matrix=rate_matrix.to_numpy(),
            exponents=[t * site_rates[i]],
        )

        matrix_exponential_df = pd.DataFrame(
            matrix_exponentials[0, :, :],
            index=utils.amino_acids,
            columns=utils.amino_acids,
        )
        y_list.append(rng.choice(utils.amino_acids, p=matrix_exponential_df.loc[x_i]))
    return "".join(y_list)


def _simulate_evolution_on_tree(
    rate_matrix_path: str,
    tree_dir: str,
    msa_dir: str,
    root_seqs_dir: str,
    family: str,
    site_rates_dir: Optional[str] = None,
    exclude_internal: bool = True,
    random_seed: Optional[int] = 42,
    output_msa_dir: Optional[str] = None
):
    # Reroot tree
    tree = read_tree(os.path.join(tree_dir, family + ".txt"))
    root_id = tree.root()
    if root_seqs_dir: 
        root_seq_path = os.path.join(root_seqs_dir, family + ".txt")
        with open(root_seq_path, 'r') as f:
            root_id = f.readline().strip()[1:] # extract identifier from >{identifier}
            root_seq = f.readline()
    else:
        assert msa_dir is not None, "Received None for both msa_dir and root_seqs_dir. A root sequence is needed to start simulation, so please pass in at least one of the above."
        msa=read_msa(os.path.join(msa_dir, family + ".txt"))
        root_seq=msa[root_id]
    
    ete_tree = tree.to_ete3()
    ete_tree.set_outgroup(root_id)

    site_rates = read_site_rates(os.path.join(site_rates_dir, family + ".txt")) if site_rates_dir else [1.0] * len(root_seq)

    # Simulate evolution down the tree: BFS
    msa = {}
    msa[root_id] = root_seq
    queue = []
    queue.append(ete_tree)

    while queue:
        node = queue.pop()
        left, right = node.children
        parent = node.up

        if parent:
            output=evolve_classical(
                x=msa[parent.name],
                t=node.dist,
                rate_matrix_path=rate_matrix_path,
                random_seed=random_seed,
                site_rates=site_rates
            )
        else:
            output = msa[node.name]
        
        msa[node.name] = output 

        queue.append(left)
        queue.append(right)
    
    # If exclude internal, then remove these sequences from the MSA
    if exclude_internal:
        msa = {k: v for k, v in msa.items() if 'internal' not in k}
    
    write_msa(msa=msa, output_path=os.path.join(output_msa_dir, family + ".txt"))
    secure_parallel_output(output_dir=output_msa_dir, parallel_arg=family)

def _map_func_simulate_evolution_on_tree(args):
    rate_matrix_path, tree_dir, msa_dir, root_seqs_dir, families, site_rates_dir, exclude_internal, random_seed, output_msa_dir = args 

    for family in families:
        _simulate_evolution_on_tree(
            rate_matrix_path=rate_matrix_path,
            tree_dir=tree_dir,
            msa_dir=msa_dir,
            root_seqs_dir=root_seqs_dir,
            family=family,
            site_rates_dir=site_rates_dir,
            exclude_internal=exclude_internal,
            random_seed=random_seed,
            output_msa_dir=output_msa_dir
        )

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def simulate_classical_evolution_on_tree(
    rate_matrix_path: str,
    tree_dir: str,
    msa_dir: str,
    root_seqs_dir: str,
    families: List[str],
    site_rates_dir: Optional[str] = None,
    exclude_internal: bool = False,
    random_seed: Optional[int] = 42,
    num_processes: int = 1,
    output_msa_dir: Optional[str] = None
):
    logger = logging.getLogger(__name__)
    logger.info(f"Simulating evolution using {rate_matrix_path} for {len(families)} families")

    map_args = [
        [
            rate_matrix_path,
            tree_dir,
            msa_dir,
            root_seqs_dir,
            get_process_args(process_rank, num_processes, families),
            site_rates_dir,
            exclude_internal,
            random_seed,
            output_msa_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_simulate_evolution_on_tree, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_simulate_evolution_on_tree, map_args),
                total=len(map_args),
            )
        )    
