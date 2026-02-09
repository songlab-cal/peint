import logging 
import multiprocessing 
import os
from pathlib import Path
import tempfile
import tqdm 
from typing import List, Optional
import subprocess 

import cherryml 
from protevo import caching as protevo_caching
from protevo.caching import secure_parallel_output
from protevo.utils import read_msa
from protevo.io import read_tree
from protevo.utils import get_process_args

CLASSICAL_MODELS = ["WAG", "LG"]

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

def find_iqtree2_path():

    iqtree_executable = Path(__file__).parent.parent.parent / "iqtree2" / "build" / "iqtree2"

    return iqtree_executable

def _simulate_alisim(
    tree_dir: str,
    msa_dir: str,
    root_seqs_dir: str,
    family: str,
    evolutionary_model: str,
    insertion_rate: Optional[float] = 1e-2,
    deletion_rate: Optional[float] = 1e-2,
    output_msa_dir: str = None
):  
    iqtree2_path = find_iqtree2_path()

    dirs = [(tree_dir, msa_dir, output_msa_dir)]

    for (tree_dir, msa_dir, output_dir) in dirs:
        tree_path = os.path.join(tree_dir, family + ".txt")
        msa_path = os.path.join(msa_dir, family + ".txt")
        output_path = os.path.join(output_dir, family)

        if os.path.exists(f"{output_path}.success") and os.path.exists(f"{output_path}.txt"):
            continue

        tree = read_tree(tree_path)
        root_id = tree.root()
        if root_seqs_dir: 
            root_seq_path = os.path.join(root_seqs_dir, family + ".txt")
            with open(root_seq_path, 'r') as f:
                root_id = f.readline().strip()[1:] # extract identifier from >{identifier}
        ete_tree = tree.to_ete3()
        ete_tree.set_outgroup(ete_tree&root_id)
            
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.nwk') as tree_file:
            tree_file.write(ete_tree.write(format=1))
            tree_file = tree_file.name

        # The input MSA contains the PEINT sequences as well. We only want the empirical sequences
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as msa_file:
            msa = read_msa(msa_path)
            msa = {k: v for k, v in msa.items() if not k.endswith('_added')}
            for k, v in msa.items():
                msa_file.write(f">{k}\n{v}\n")
            msa_path = msa_file.name

        alisim_cmd = [
            iqtree2_path, 
            "--alisim", output_path, 
            "-te", tree_file, 
            "-s", msa_path,
            "--root-seq", f"{msa_path},{root_id}",
            "-o", root_id,
            "--out-format", "fasta",
            "-seed", "42",
            "--write-all"
        ]

        # Include site rates from Gamma distribution with 4 rate categories and shape parameter 0.5 for LG
        if evolutionary_model == 'LG':
            alisim_cmd.append("--site-rate") 
            alisim_cmd.append("MODEL")

            alisim_cmd.append("-m") # 4 rate categories
            alisim_cmd.append("LG+G4")
        else:
            alisim_cmd.append("-m")
            alisim_cmd.append(evolutionary_model)
        
        if insertion_rate + deletion_rate > 0:
            alisim_cmd.append("--indel")
            alisim_cmd.append(f"{insertion_rate}, {deletion_rate}")
            alisim_cmd.append("--no-unaligned")    
        
        result=subprocess.run(alisim_cmd, capture_output=True, text=True)

        # AliSim writes to output_dir/output.fasta
        alisim_output = os.path.join(output_dir, family + '.fa')
        family_output = os.path.join(output_dir, family + '.txt')
        os.rename(alisim_output, family_output)

        # IQTree also likes to write logging files to the MSA directory: remove these.
        undesired_output_formats = ['.iqtree', '.treefile', '.log', '.ckp.gz']
        for fmt in undesired_output_formats:
            fpath = msa_path + fmt 
            os.remove(fpath)
    
    secure_parallel_output(output_msa_dir, family)

def _map_func_simulate_alisim(args):
    tree_dir, msa_dir, root_seqs_dir, families, evolutionary_model, insertion_rate, deletion_rate, output_msa_dir = args 

    for family in families:
        _simulate_alisim(
            tree_dir=tree_dir,
            msa_dir=msa_dir,
            family=family,
            evolutionary_model=evolutionary_model,
            root_seqs_dir=root_seqs_dir,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,
            output_msa_dir=output_msa_dir
        )

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    exclude_args_if_default=["root_seqs_dir"],
    output_dirs=["output_msa_dir"],
    write_extra_log_files=True,
)
def simulate_alisim_evolution(
    tree_dir: str,
    msa_dir: str,
    root_seqs_dir: str,
    families: List[str],
    evolutionary_model: str,
    num_processes: int = 1,
    insertion_rate: Optional[float] = 0,
    deletion_rate: Optional[float] = 0,
    output_msa_dir: Optional[str] = None,
):
    assert evolutionary_model in CLASSICAL_MODELS, f"Evolutionary model {evolutionary_model} isn't supported, please pass in one of {CLASSICAL_MODELS}"
    assert insertion_rate >= 0 and deletion_rate >= 0, "Received negative insertion and/or deletion rates"

    logger = logging.getLogger(__name__)
    logger.info(f"Simulating evolution using AliSim with {evolutionary_model} for {len(families)} families")

    if output_msa_dir is not None and not os.path.exists(output_msa_dir):
        os.mkdir(output_msa_dir)

    map_args = [
        [
            tree_dir,
            msa_dir,
            root_seqs_dir,
            get_process_args(process_rank, num_processes, families),
            evolutionary_model,
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