import os
import random
from functools import partial

from Bio import SeqIO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esm
import logging
import tqdm
from typing import List, Dict, Tuple, Optional
from collections import deque
import multiprocessing
import time
from ete3 import Tree

from protevo import caching as protevo_caching
from protevo.caching import secure_parallel_output
from protevo.utils import get_process_args, write_msa, read_msa
from protevo.models._loading import load_model

from protevo.io import (
    read_tree
)

def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_batch(parent_sequences, branch_lengths, vocab):
    x_toks = encode_and_pad_sequences(parent_sequences, vocab, 'cpu')
    time = torch.tensor(branch_lengths, dtype=torch.float32).unsqueeze(1)
    return x_toks, time

def encode_and_pad_sequences(sequences: List[str], vocab, device):
    encoded_sequences = []
    for seq in sequences:
        encoded = torch.tensor([vocab.cls_idx] + vocab.encode(seq) + [vocab.eos_idx], device=device)
        encoded_sequences.append(encoded)

    padded_sequences = nn.utils.rnn.pad_sequence(encoded_sequences, batch_first=True, padding_value=vocab.padding_idx)
    return padded_sequences

def rejection_filter(x, length, ratio):
    return (1 - ratio) * length < x < (1 + ratio) * length

def load_msa_and_tree(
        msa_dir:str,
        tree_dir:str,
        family_name:str) -> Tuple[Tree, Dict[str, str]]:

    if not family_name.endswith('.txt'):
        family_name = family_name + '.txt'

    msa_file = os.path.join(msa_dir, family_name)
    tree_file = os.path.join(tree_dir, family_name)

    msa = {r.id: str(r.seq) for r in SeqIO.parse(msa_file, "fasta")}
    tree = read_tree(tree_file)

    return tree, msa

def filter_sequences(decoded_sequences, length_criterion, likelihood_fn=None, x_toks=None, times=None):
    """
    Filter sequences based on length and likelihood criteria.

    Args:
        decoded_sequences: List of sequences to filter
        length_criterion: Function that takes sequence length and returns bool
        likelihood_fn: Optional function to calculate sequence likelihoods
        x_toks: Input sequences tensor for likelihood calculation
        times: Time points tensor for likelihood calculation

    Returns:
        str: The chosen sequence
    """
    filtered_indices = [i for i, seq in enumerate(decoded_sequences)
                       if length_criterion(len(seq))]

    if not filtered_indices:
        return None

    filtered_sequences = [decoded_sequences[i] for i in filtered_indices]

    if likelihood_fn and x_toks is not None and times is not None:
        # Select corresponding x_toks and times for filtered sequences
        filtered_x_toks = x_toks[filtered_indices]
        filtered_times = times[filtered_indices]

        likelihoods = likelihood_fn(filtered_sequences, filtered_x_toks, filtered_times)
        chosen_idx = likelihoods.argmax().item()
        chosen_sequence = filtered_sequences[chosen_idx]
    else:
        chosen_sequence = random.choice(filtered_sequences)

    return chosen_sequence

def calculate_sequence_likelihoods(
    model,
    sequences: List[str],
    x_toks: torch.Tensor,
    times: torch.Tensor,
    vocab,
    device: torch.device
) -> torch.Tensor:
    """
    Calculate mean per-token likelihood for each sequence in the batch.

    Args:
        model: The transformer model
        sequences: List of candidate sequences to evaluate
        x_toks: Input sequences tensor [batch_size, seq_len]
        times: Time points tensor [batch_size, 1]
        vocab: The vocabulary object
        device: torch device

    Returns:
        torch.Tensor: Mean per-token likelihood for each sequence
    """
    # Prepare input and target sequences
    y_input = []
    y_target = []

    for seq in sequences:
        y_input.append(torch.tensor([vocab.cls_idx] + vocab.encode(seq), device=device))
        y_target.append(torch.tensor(vocab.encode(seq) + [vocab.eos_idx], device=device))


    y_inputs = nn.utils.rnn.pad_sequence(y_input, batch_first=True, padding_value=vocab.padding_idx)
    y_targets = nn.utils.rnn.pad_sequence(y_target, batch_first=True, padding_value=vocab.padding_idx)

    y_padding_mask = y_inputs.eq(vocab.padding_idx)
    x_padding_mask = x_toks.eq(vocab.padding_idx)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            output = model(x_toks, y_inputs, times, x_padding_mask, y_padding_mask)

    if isinstance(output, tuple):
        y_logits = output[-1]
    else:
        y_logits = output

    loss = F.cross_entropy(
        y_logits.transpose(1, 2),
        y_targets,
        ignore_index=vocab.padding_idx,
        reduction='none'
    )

    non_pad_mask = ~y_targets.eq(vocab.padding_idx)

    mean_loss = (loss * non_pad_mask).sum(dim=1) / non_pad_mask.sum(dim=1)

    return -mean_loss  # Return negative loss as likelihood

def create_likelihood_function(model, vocab, device):
    """
    Creates a likelihood function to be used in sequence filtering.

    Args:
        model: The transformer model
        vocab: The vocabulary object
        device: torch device

    Returns:
        function: A likelihood function that takes sequences and returns their scores
    """
    def likelihood_fn(sequences, x_toks, times):
        return calculate_sequence_likelihoods(
            model=model,
            sequences=sequences,
            x_toks=x_toks,
            times=times,
            vocab=vocab,
            device=device
        )

    return likelihood_fn

def simulate_evolution_with_rejection_sampling_batched(
        model,
        root_sequences: Dict[str, str],
        trees: List[Tree],
        vocab,
        device,
        max_decode_steps,
        max_batch_size,
        rejection_sampling_length_ratio: dict,
        n_sequences=4,
        p_threshold=1.0,
        use_likelihood_filtering: bool =False,
        likelihood_eval_model = None,
        max_retries=3,
        branch_scale_factor = 1.0,
        seed=42):

    _seed_all(seed)

    assert(all([hasattr(t, 'treename') for t in trees])), "All trees must have a 'treename' attribute"

    all_sequences = {tree.treename: {} for tree in trees}
    starting_points = []
    total_nodes_to_decode = 0
    total_decoded_nodes = 0
    for i,tree in enumerate(trees):
        starting_points.append((tree, root_sequences[tree.treename], list(tree.children), 0))
        for n in tree.traverse():
            total_nodes_to_decode += 1

    queue = deque(starting_points)  # (node, sequence, remaining_children, retry_count)

    if use_likelihood_filtering:
        likelihood_fn = create_likelihood_function(
            model if not likelihood_eval_model else likelihood_eval_model,
            vocab,
            device)
    else:
        likelihood_fn = None

    while queue:
        batch_nodes = []
        batch_sequences = []
        batch_branch_lengths = []
        batch_retry_counts = []
        batch_tree_names = []

        effective_branch_count = max_batch_size // n_sequences

        while queue and len(batch_nodes) < max_batch_size:
            node, parent_sequence, remaining_children, retry_count = queue.popleft()
            if remaining_children:
                child = remaining_children.pop(0)
                batch_nodes.extend([child] * n_sequences)
                batch_sequences.extend([parent_sequence] * n_sequences)
                batch_branch_lengths.extend([child.dist * branch_scale_factor] * n_sequences)
                batch_retry_counts.extend([retry_count] * n_sequences)
                batch_tree_names.extend([node.treename] * n_sequences)

                if remaining_children:
                    queue.appendleft((node, parent_sequence, remaining_children, retry_count))

                if len(batch_nodes) // n_sequences >= effective_branch_count:
                    break

        if not batch_nodes:
            continue

        x_toks, time = prepare_batch(batch_sequences, batch_branch_lengths, vocab)
        x_toks, time = x_toks.to(device), time.to(device)

        with torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
            decoded_sequences = model.generate(x = x_toks,
                                            t = time,
                                            device = device,
                                            max_decode_steps = max_decode_steps,
                                            p = p_threshold)

        for i in range(0, len(batch_nodes), n_sequences):
            node = batch_nodes[i]
            node_sequences = decoded_sequences[i:i+n_sequences]
            retry_count = batch_retry_counts[i]
            tree_name = batch_tree_names[i]

            length_criterion = rejection_sampling_length_ratio[tree_name] #indexed by tree_name

            chosen_sequence = filter_sequences(
                node_sequences,
                length_criterion,
                likelihood_fn,
                x_toks=x_toks[i:i+n_sequences],
                times=time[i:i+n_sequences]
            )

            if hasattr(model, '_reset_kv_cache'):
                #if this is the cached model, reset the cache
                #this should happen prior to next round of generation
                model._reset_kv_cache()

            if chosen_sequence is None:
                if retry_count < max_retries:
                    # Add the parent node back to the left of the queue for another try
                    parent_node = node.up
                    queue.appendleft((parent_node, all_sequences.get(parent_node.name, root_sequences[tree_name]), [node], retry_count + 1))
                else:
                    # If max retries reached, use a random sequence from the generated ones
                    chosen_sequence = random.choice(node_sequences)

            if chosen_sequence is not None:
                #picked a sequence
                total_decoded_nodes += 1
                all_sequences[tree_name][node.name] = chosen_sequence
                if not node.is_leaf():
                    queue.append((node, chosen_sequence, list(node.children), 0))  # Reset retry count for child nodes

        if total_decoded_nodes % 250 == 0:
            print(f"Total decoded nodes: {total_decoded_nodes}/{total_nodes_to_decode}")

    return all_sequences

def simulate_leaves_from_root(
        tree: Tree,
        initial_sequence: str,
        initial_label: str,
        model: nn.Module,
        vocab: esm.data.Alphabet,
        device: torch.device,
        n_sequences: int = 4,
        max_batch_size: int = 64,
        p_threshold: float = 1.0,
        ratio_rejection_sampling: float = 0.1,
        use_likelihood_filtering: bool =False,
        likelihood_eval_model = None,
        branch_scale_factor = 1.0,
        seed: int = 42
):

    _seed_all(seed)

    max_decode_steps = 2 * len(initial_sequence) #to be on safe side, ideally it will terminate earlier

    length_criterion = partial(rejection_filter, length = len(initial_sequence), ratio = ratio_rejection_sampling)

    to_process = []
    for leaf in tree.get_leaves():
        if leaf.name != initial_label:
            distance = tree.get_distance(leaf, initial_label) * branch_scale_factor
            to_process.append((leaf, distance, 0))

    queue = deque(to_process)

    if use_likelihood_filtering:
        likelihood_fn = create_likelihood_function(
            model if not likelihood_eval_model else likelihood_eval_model,
            vocab,
            device)
    else:
        likelihood_fn = None

    simulated_sequences = {initial_label: initial_sequence}

    while queue:
        batch_nodes = []
        batch_sequences = []
        batch_branch_lengths = []
        batch_retry_counts = []

        while queue and len(batch_nodes) < max_batch_size:
            leaf, distance, retry_count = queue.popleft()
            batch_nodes.extend([leaf] * n_sequences)
            batch_sequences.extend([initial_sequence] * n_sequences)
            batch_branch_lengths.extend([distance] * n_sequences)
            batch_retry_counts.extend([retry_count] * n_sequences)

            if len(batch_nodes) >= max_batch_size:
                break

        if not batch_nodes:
            continue

        x_toks, time = prepare_batch(batch_sequences, batch_branch_lengths, vocab)
        x_toks, time = x_toks.to(device), time.to(device)

        with torch.autocast(device_type = "cuda", dtype = torch.bfloat16):
            decoded_sequences = model.generate(x = x_toks,
                                            t = time,
                                            device = device,
                                            max_decode_steps = max_decode_steps,
                                            p = p_threshold)

        for i in range(0, len(batch_nodes), n_sequences):
            node = batch_nodes[i]
            node_sequences = decoded_sequences[i:i+n_sequences]
            retry_count = batch_retry_counts[i]

            chosen_sequence = filter_sequences(
                node_sequences,
                length_criterion,
                likelihood_fn,
                x_toks=x_toks[i:i+n_sequences],
                times=time[i:i+n_sequences]
            )

            if hasattr(model, '_reset_kv_cache'):
                #if this is the cached model, reset the cache
                #this should happen prior to next round of generation
                model._reset_kv_cache()

            if chosen_sequence is None:
                if retry_count < 3:
                    # Add the parent node back to the left of the queue for another try
                    queue.appendleft((node, distance, retry_count + 1))
                else:
                    # If max retries reached, use a random sequence from the generated ones
                    chosen_sequence = random.choice(node_sequences)

            if chosen_sequence is not None:
                simulated_sequences[node.name] = chosen_sequence

    return simulated_sequences

def simulate_families_with_rejection_sampling_batched(
        msa_dir: str,
        tree_dir: str,
        root_sequences_dir: str,
        family_names: List[str],
        model: nn.Module,
        vocab: esm.data.Alphabet,
        device: torch.device,
        single_shot: bool = False,
        n_sequences_to_sample: int = 4,
        max_batch_size: int = 64,
        nucleus_sampling_p: float = 1.0,
        branch_scale_factor = 1.0,
        ratio_rejection_sampling: float = 0.1,
        use_likelihood_filtering: bool =False,
        likelihood_eval_model: nn.Module=None
    ):
    simulated = {}
    trees = []
    trees_ete = []
    root_sequences = []
    root_labels = []
    rejection_sample_functions = {}
    for family_name in family_names:

        # Use user-supplied root sequences
        if root_sequences_dir:
            tree_path = os.path.join(tree_dir, family_name + ".txt")
            root_seq_path = os.path.join(root_sequences_dir, family_name + ".txt")
            tree = read_tree(tree_path)

            initial_label, initial_seq = next(iter(read_msa(root_seq_path).items()))
        # Use default median length sequence from MSA
        else:
            tree, msa = load_msa_and_tree(msa_dir=msa_dir, tree_dir=tree_dir, family_name=family_name)

            #find the median sequence length
            sorted_seqs = sorted(msa.items(), key = lambda x: len(x[1]))
            median_idx = len(sorted_seqs) // 2

            initial_label, initial_seq = sorted_seqs[median_idx]

        tree_ete = tree.to_ete3()
        reroot_node = tree_ete&initial_label
        tree_ete.set_outgroup(reroot_node)
        new_root = tree_ete.get_tree_root()

        for n in new_root.traverse():
            n.add_features(treename=family_name)

        trees_ete.append(tree_ete)
        trees.append(new_root)
        root_sequences.append(initial_seq)
        root_labels.append(initial_label)

        rejection_sample_functions[family_name] = partial(rejection_filter, length = len(initial_seq), ratio = ratio_rejection_sampling)

    max_decode_steps = max([len(seq) for seq in root_sequences]) * 2 #to be on safe side, ideally it will terminate earlier

    print(f"Loaded and prepped data for {len(family_names)} families!")
    simulation_scheme = 'single shot' if single_shot else 'progressive'
    print(f"Starting simulation using {simulation_scheme} decoding")
    t0 = time.time()

    if single_shot:
        for family_name, tree, initial_seq, initial_label in zip(family_names, trees_ete, root_sequences, root_labels):
            simulated[family_name] = simulate_leaves_from_root(
                tree = tree,
                initial_sequence = initial_seq,
                initial_label = initial_label,
                model = model,
                vocab = vocab,
                device = device,
                n_sequences = n_sequences_to_sample,
                max_batch_size = max_batch_size,
                p_threshold = nucleus_sampling_p,
                ratio_rejection_sampling = ratio_rejection_sampling,
            )
    else:
        root_sequences = {tree.treename: seq for tree, seq in zip(trees, root_sequences)}
        simulated = simulate_evolution_with_rejection_sampling_batched(
            model = model,
            root_sequences = root_sequences,
            trees = trees,
            vocab = vocab,
            device = device,
            max_decode_steps= max_decode_steps,
            max_batch_size= max_batch_size,
            n_sequences = n_sequences_to_sample,
            p_threshold = nucleus_sampling_p,
            branch_scale_factor = branch_scale_factor,
            rejection_sampling_length_ratio = rejection_sample_functions,
            use_likelihood_filtering = use_likelihood_filtering,
            likelihood_eval_model = likelihood_eval_model
        )
    tf = time.time()
    print(f"Finished simulating {len(family_names)} families in {tf-t0} seconds")

    return simulated, trees, root_sequences, root_labels

def _map_func_simulate_peint_evolution(map_args):
    msa_dir, tree_dir, root_sequences_dir, families, model, vocab, single_shot, device, n_sequences_to_sample, max_batch_size, nucleus_sampling_p, ratio_rejection_sampling, use_likelihood_filtering, random_seed, output_sequences_dir = map_args
    _seed_all(random_seed)

    simulated, _, root_seqs, _ = simulate_families_with_rejection_sampling_batched(
        msa_dir=msa_dir,
        tree_dir=tree_dir,
        root_sequences_dir=root_sequences_dir,
        family_names=families,
        model=model,
        vocab=vocab,
        device=device,
        single_shot=single_shot,
        n_sequences_to_sample=n_sequences_to_sample,
        max_batch_size=max_batch_size,
        nucleus_sampling_p=nucleus_sampling_p,
        ratio_rejection_sampling=ratio_rejection_sampling,
        use_likelihood_filtering=use_likelihood_filtering
    )

    for family in families:
        all_seqs = {'root': root_seqs[family]}
        simulated_seqs = simulated[family]
        all_seqs.update(simulated_seqs)
        out_path = os.path.join(output_sequences_dir, family + ".txt")

        write_msa(all_seqs, out_path)
        secure_parallel_output(output_sequences_dir, family)

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=['max_batch_size', 'num_processes'],
    exclude_args_if_default=["msa_dir"],
    output_dirs=['output_sequences_dir'],
    write_extra_log_files=True
)
def simulate_peint_evolution_down_tree(
    tree_dir: str,
    root_sequences_dir: Optional[str],
    families: List[str],
    model_path: str,
    msa_dir: Optional[str] = None,
    single_shot: bool = False,
    device: torch.device = 'cuda',
    n_sequences_to_sample: int = 4,
    max_batch_size: int = 64,
    nucleus_sampling_p: float = 1.0,
    ratio_rejection_sampling: float = 0.1,
    use_likelihood_filtering: bool = False,
    random_seed: int = 0,
    num_processes: int = 1,
    output_sequences_dir: Optional[str] = None
):
    logger = logging.getLogger(__name__)
    logger.info(f"Simulating evolution using PEINT for {len(families)} families")

    assert msa_dir or root_sequences_dir, "Didn't receive an msa or root sequence directory. There is no way to construct a starting sequence for simulation"
    assert (msa_dir and not root_sequences_dir) or (root_sequences_dir and not msa_dir), "Received both an msa and root sequences directory. Only one may be used to pick a root sequence for simulation."

    model, vocab = load_model(model_path, use_cached_model=True, device=device)
    map_args = [
        [
            msa_dir,
            tree_dir,
            root_sequences_dir,
            get_process_args(process_rank, num_processes, families),
            model,
            vocab,
            single_shot,
            device,
            n_sequences_to_sample,
            max_batch_size,
            nucleus_sampling_p,
            ratio_rejection_sampling,
            use_likelihood_filtering,
            random_seed,
            output_sequences_dir
        ]
        for process_rank in range(num_processes)
    ]

    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(_map_func_simulate_peint_evolution, map_args),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(_map_func_simulate_peint_evolution, map_args),
                total=len(map_args),
            )
        )
