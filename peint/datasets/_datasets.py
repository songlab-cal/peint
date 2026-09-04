import os
import random
import warnings
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import hashlib
import numpy as np
import multiprocessing
import tqdm
import subprocess
from Bio import SeqIO
import tempfile
import re

import cherryml
from cherryml.benchmarking import pfam_15k
from cherryml.utils import get_families
from .secondary_structure import compute_secondary_structure_annotations
from peint.caching import secure_parallel_output

from peint import caching as peint_caching
from peint.caching import secure_parallel_output
from peint.utils import get_process_args
from peint import utils
from peint.io import (
    Tree,
    get_msa_num_residues,
    get_msa_num_sequences,
    get_msa_num_sites,
    read_msa,
    read_pickle,
    read_secondary_structure,
    read_tree,
    write_msa,
    write_pickle,
    write_transitions,
    write_tree,
)
from ._distance_matrix import compute_distance_maps


@peint_caching.cached_computation(
    output_dirs=["output_dir"],
)
def get_msas_number_of_sites__cached(
    msa_dir: str,
    families: List[str],
    output_dir: Optional[str] = None,
):
    """
    Get the total number of sites in the dataset.
    """
    res = 0
    for family in families:
        num_sites = get_msa_num_sites(os.path.join(msa_dir, family + ".txt"))
        res += num_sites
    assert res >= 1
    write_pickle(res, os.path.join(output_dir, "result.txt"))


@peint_caching.cached_computation(
    output_dirs=["output_dir"],
)
def get_msas_number_of_sequences__cached(
    msa_dir: str,
    families: List[str],
    output_dir: Optional[str] = None,
):
    """
    Get the total number of sequences in the dataset.
    """
    res = 0
    for family in families:
        num_sequences = get_msa_num_sequences(
            os.path.join(msa_dir, family + ".txt")
        )
        res += num_sequences
    assert res >= 1
    write_pickle(res, os.path.join(output_dir, "result.txt"))


@peint_caching.cached_computation(
    output_dirs=["output_dir"],
)
def get_msas_number_of_residues__cached(
    msa_dir: str,
    families: List[str],
    exclude_gaps: bool,
    output_dir: Optional[str] = None,
):
    """
    Get the total number of residues in the dataset.
    """
    res = 0
    for family in families:
        res += get_msa_num_residues(
            os.path.join(msa_dir, family + ".txt"), exclude_gaps=exclude_gaps
        )
    assert res >= 1
    write_pickle(res, os.path.join(output_dir, "result.txt"))


@peint_caching.cached()
def report_dataset_statistics_str(
    msa_dir: str, families: Optional[List[str]] = None
) -> str:
    """
    Reports statistics on the training data:
    - Total number of MSAs
    - Number of sequences per MSA.
    - Number of sites per MSA.
    - Total number of residues.
    """
    if families is None:
        families = get_families(msa_dir)
    number_of_sites = read_pickle(
        get_msas_number_of_sites__cached(
            msa_dir=msa_dir,
            families=families,
        )["output_dir"]
        + "/result.txt"
    )
    number_of_sequences = read_pickle(
        get_msas_number_of_sequences__cached(
            msa_dir=msa_dir,
            families=families,
        )["output_dir"]
        + "/result.txt"
    )
    number_of_residues = read_pickle(
        get_msas_number_of_residues__cached(
            msa_dir=msa_dir,
            families=families,
            exclude_gaps=True,
        )["output_dir"]
        + "/result.txt"
    )
    number_of_residues_including_gaps = read_pickle(
        get_msas_number_of_residues__cached(
            msa_dir=msa_dir,
            families=families,
            exclude_gaps=False,
        )["output_dir"]
        + "/result.txt"
    )
    res = f"Number of MSAs = {len(families)}\n"
    res += f"Number of sequences: {number_of_sequences}\n"
    res += f"Number of sites: {number_of_sites}\n"
    res += f"Number of residues: {number_of_residues}\n"
    res += (
        "Number of residues including gaps: "
        f"{number_of_residues_including_gaps}\n"
    )
    return res


def split_tree_on_edge(
    tree: Tree,
    edge: Tuple[str, str],
) -> Tuple[Tree, Tree]:
    """
    Split a tree into two trees by removing the given edge.

    The two resulting trees will be rooted at the endpoints of the edge.

    Args:
        tree: Tree to split.
        edge: Edge to split tree on.
    Returns:
        The two trees obtained by splitting on the edge.
    """

    def dfs(tree, node_outside, node_inside, subtree):
        """
        Invariant: `node_inside` has already been added to the `subtree`
        Adds all inside neighboring nodes and their edges, then calls dfs
        retursively on them.
        """
        neighboring_nodes_inside = [
            node_and_len for node_and_len in tree.children(node_inside)
        ]
        if not tree.is_root(node_inside):
            neighboring_nodes_inside += [tree.parent(node_inside)]
        neighboring_nodes_inside = [
            node_and_len
            for node_and_len in neighboring_nodes_inside
            if node_and_len[0] != node_outside
        ]
        for node_and_len in neighboring_nodes_inside:
            subtree.add_node(node_and_len[0])
            subtree.add_edge(node_inside, node_and_len[0], node_and_len[1])
        for node_and_len in neighboring_nodes_inside:
            dfs(tree, node_inside, node_and_len[0], subtree)

    subtree_edge_0 = Tree()
    subtree_edge_0.add_node(edge[0])
    dfs(
        tree=tree,
        node_outside=edge[1],
        node_inside=edge[0],
        subtree=subtree_edge_0,
    )

    subtree_edge_1 = Tree()
    subtree_edge_1.add_node(edge[1])
    dfs(
        tree=tree,
        node_outside=edge[0],
        node_inside=edge[1],
        subtree=subtree_edge_1,
    )

    return subtree_edge_0, subtree_edge_1


def find_optimal_edge_split(
    tree: Tree,
) -> Tuple[str, str]:
    """
    Find the most balanced edge to split a tree on.

    Args:
        tree: Tree for which to find the optimal edge to split on.
    Returns:
        The edge (u, v) which to split the tree on.
    """

    def _dfs_sizes_down(tree, node, sizes_down) -> None:
        """
        Populate the size below this node (number of leaves).
        """
        if tree.is_leaf(node):
            sizes_down[node] = 1
        else:
            [
                _dfs_sizes_down(tree, child, sizes_down)
                for (child, _) in tree.children(node)
            ]
            sizes_down[node] = sum(
                [sizes_down[child] for (child, _) in tree.children(node)]
            )

    def get_sizes_down(tree) -> Dict[str, int]:
        """
        Number of leaves under a node, for each node in the tree.
        If the node is a leaf, then the size down is 1.
        """
        sizes_down = {}
        _dfs_sizes_down(tree, tree.root(), sizes_down)
        return sizes_down

    sizes_down = get_sizes_down(tree)
    number_of_leaves = len(tree.leaves())

    optimal_edge = None
    optimal_error = number_of_leaves
    for (parent, child, _) in tree.edges():
        current_error = abs(sizes_down[child] - number_of_leaves / 2)
        if current_error < optimal_error:
            optimal_error = current_error
            optimal_edge = (parent, child)

    return optimal_edge


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=[
        "output_train_msa_dir",
        "output_train_tree_dir",
        "output_test_msa_dir",
        "output_test_tree_dir",
    ],
    write_extra_log_files=True,
)
def msa_treewise_train_test_split(
    msa_dir: str,
    tree_dir: str,
    families: List[str],
    num_processes: int,
    output_train_msa_dir: Optional[str] = None,
    output_train_tree_dir: Optional[str] = None,
    output_test_msa_dir: Optional[str] = None,
    output_test_tree_dir: Optional[str] = None,
) -> None:
    """
    Split a MSA into a training and a testing half based on their phylogeny.

    The split is determined by the optimal edge split of the tree, i.e. the
    edge which splits the tree into the most even parts in terms of number of
    leaves.

    NOTE:
    """
    for family in families:
        msa_path = os.path.join(msa_dir, family + ".txt")
        msa = read_msa(msa_path)
        tree_path = os.path.join(tree_dir, family + ".txt")
        tree = read_tree(tree_path)

        optimal_edge = find_optimal_edge_split(tree)
        train_subtree, test_subtree = split_tree_on_edge(tree, optimal_edge)
        assert len(tree.leaves()) == len(train_subtree.leaves()) + len(
            test_subtree.leaves()
        )
        assert sorted(tree.leaves()) == sorted(
            train_subtree.leaves() + test_subtree.leaves()
        )
        assert sorted(tree.leaves()) == sorted(list(msa.keys()))
        train_msa = {leaf: msa[leaf] for leaf in train_subtree.leaves()}
        test_msa = {leaf: msa[leaf] for leaf in test_subtree.leaves()}

        output_train_msa_path = os.path.join(
            output_train_msa_dir, f"{family}.txt"
        )
        output_test_msa_path = os.path.join(
            output_test_msa_dir, f"{family}.txt"
        )
        output_train_tree_path = os.path.join(
            output_train_tree_dir, f"{family}.txt"
        )
        output_test_tree_path = os.path.join(
            output_test_tree_dir, f"{family}.txt"
        )
        write_msa(train_msa, output_train_msa_path)
        write_msa(test_msa, output_test_msa_path)
        write_tree(train_subtree, output_train_tree_path)
        write_tree(test_subtree, output_test_tree_path)


def extract_transitions_from_tree(
    tree: Tree,
    msa: Dict[str, str],
    include_gaps: bool = True,
    alignment_mask: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str, float, Optional[str], Optional[str], str, str]]:
    """
    Extract transitions from the tree.

    The transitions come from all (recursively picked) cherries, and are
    bidirectional, so that if (x, y, t) is a transition, then also is
    (y, x, t).

    Whether gaps (assumed to be "-") are included or not is determined by
    `include_gaps`.

    We also return in the last coordinates the alignment masks of x and y, and the sequence names,
    which can later be used to make more apples-to-apples comparisons. I.e. we return
    (x, y, t, alignment_mask[x], alignment_mask[y], name_1, name_2).
    If alignment_mask is None, the alignment_masks will simply be None.

    The alignment mask simply indicates which position in x and y align to the reference, (after
    out-of-alphabet characters have been converted to gaps. Thus, if the MSA data is:
    (AG-L, A--L, 0.1)
    And the full-length version is:
    (AGLP, AXLPC, 0.1)
    then the alignment masks would be 1110 and 10100
    """
    if alignment_mask is not None and include_gaps:
        raise ValueError(
            f"The alignment mask is only used for the full-length, unaligned "
            f"gapless sequences dataset!"
        )
    total_pairs = []
    transitions = []

    def dfs(node) -> Optional[Tuple[int, float]]:
        """
        Pair up leaves under me.

        Return a single unpaired leaf and its distance, it such exists.
        """
        if tree.is_leaf(node):
            return (node, 0.0)
        unmatched_leaves_under = []
        distances_under = []
        for child, branch_length in tree.children(node):
            maybe_unmatched_leaf, maybe_distance = dfs(child)
            if maybe_unmatched_leaf is not None:
                assert maybe_distance is not None
                unmatched_leaves_under.append(maybe_unmatched_leaf)
                distances_under.append(maybe_distance + branch_length)
        assert len(unmatched_leaves_under) == len(distances_under)
        index = 0

        while index + 1 <= len(unmatched_leaves_under) - 1:
            total_pairs.append(1)
            (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                (unmatched_leaves_under[index], distances_under[index]),
                (
                    unmatched_leaves_under[index + 1],
                    distances_under[index + 1],
                ),
            )
            leaf_seq_1, leaf_seq_2 = msa[leaf_1], msa[leaf_2]
            if alignment_mask is not None:
                if len(alignment_mask[leaf_1]) != len(leaf_seq_1):
                    raise ValueError(
                        f"Alignment mask {alignment_mask[leaf_1]} ({len(alignment_mask[leaf_1])}) should have the same length "
                        f"as {leaf_seq_1} ({len(leaf_seq_1)}), but it does not."
                    )
                if len(alignment_mask[leaf_2]) != len(leaf_seq_2):
                    raise ValueError(
                        f"Alignment mask {alignment_mask[leaf_2]} ({len(alignment_mask[leaf_2])}) should have the same length "
                        f"as {leaf_seq_2} ({len(leaf_seq_2)}), but it does not."
                    )
            transitions.append(
                (
                    leaf_seq_1,
                    leaf_seq_2,
                    branch_length_1 + branch_length_2,
                    alignment_mask[leaf_1] if alignment_mask is not None else None,
                    alignment_mask[leaf_2] if alignment_mask is not None else None,
                    leaf_1,
                    leaf_2,
                )
            )  # Note: gaps will be removed later
            transitions.append(
                (
                    leaf_seq_2,
                    leaf_seq_1,
                    branch_length_1 + branch_length_2,
                    alignment_mask[leaf_2] if alignment_mask is not None else None,
                    alignment_mask[leaf_1] if alignment_mask is not None else None,
                    leaf_2,
                    leaf_1,
                )
            )  # Note: gaps will be removed later
            index += 2
        if len(unmatched_leaves_under) % 2 == 0:
            return (None, None)
        else:
            return (unmatched_leaves_under[-1], distances_under[-1])

    dfs(tree.root())
    if not include_gaps:
        # Remove gaps from all sequences.
        def remove_gaps(seq: str) -> str:
            return ''.join([char for char in seq if char != "-"])

        transitions = [
            (remove_gaps(x), remove_gaps(y), t, amx, amy, l1, l2)
            for (x, y, t, amx, amy, l1, l2) in transitions
        ]
    assert len(total_pairs) == int(len(tree.leaves()) / 2)
    assert 2 * len(total_pairs) == len(transitions)
    return transitions


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    exclude_args_if_default=["include_gaps", "alignment_mask_dir"],
    output_dirs=[
        "output_transitions_dir",
        "output_alignment_mask_dir",
        "output_transition_names_dir",
    ],
    write_extra_log_files=True,
)
def extract_transitions(
    msa_dir: str,
    tree_dir: str,
    families: List[str],
    num_processes: int,
    include_gaps: bool = True,
    alignment_mask_dir: Optional[str] = None,
    output_transitions_dir: Optional[str] = None,
    output_alignment_mask_dir: Optional[str] = None,
    output_transition_names_dir: Optional[str] = None,
    _version: str = "2024_05_11_v4",
) -> None:
    """
    Extract transitions from the trees.

    NOTE: Has not been paralellized; `num_processes` unused.
    """
    for family in families:
        msa_path = os.path.join(msa_dir, family + ".txt")
        msa = read_msa(msa_path)
        tree_path = os.path.join(tree_dir, family + ".txt")
        tree = read_tree(tree_path)
        alignment_mask = None
        if alignment_mask_dir is not None:
            alignment_mask_path = os.path.join(alignment_mask_dir, family + ".txt")
            alignment_mask = read_msa(alignment_mask_path)
        transitions_and_alignment_masks = extract_transitions_from_tree(
            tree=tree,
            msa=msa,
            include_gaps=include_gaps,
            alignment_mask=alignment_mask,
        )
        transitions_and_alignment_masks = sorted(
            transitions_and_alignment_masks,
            key=lambda transition: (
                transition[5],
                transition[6],
                transition[2],
                transition[0],
                transition[1],
            ),
        )
        transitions = [
            (x, y, t) for (x, y, t, amx, amy, l1, l2) in transitions_and_alignment_masks
        ]
        transitions_mask = [
            (amx, amy, t) for (x, y, t, amx, amy, l1, l2) in transitions_and_alignment_masks
        ]
        transitions_names = [
            (l1, l2, t) for (x, y, t, amx, amy, l1, l2) in transitions_and_alignment_masks
        ]
        # Check length again
        if alignment_mask is not None:
            for (x, y, t), (amx, amy, t2) in zip(transitions, transitions_mask):
                assert(len(x) == len(amx))
                assert(len(y) == len(amy))
                assert(abs(t - t2) < 1e-8)

        transitions_path = os.path.join(output_transitions_dir, family + ".txt")
        write_transitions(
            transitions=transitions, transitions_path=transitions_path
        )
        secure_parallel_output(output_transitions_dir, family)

        alignment_masks_path = os.path.join(output_alignment_mask_dir, family + ".txt")
        write_transitions(
            transitions=transitions_mask, transitions_path=alignment_masks_path
        )
        secure_parallel_output(output_alignment_mask_dir, family)

        transition_names_path = os.path.join(output_transition_names_dir, family + ".txt")
        write_transitions(
            transitions=transitions_names, transitions_path=transition_names_path
        )
        secure_parallel_output(output_transition_names_dir, family)


def alphabetize_seq(
    seq: str, alphabet_set: Set[str], out_of_alphabet_character: str
) -> str:
    """
    Alphabetize a sequence.
    """
    res = "".join(
        [
            character
            if character in alphabet_set
            else out_of_alphabet_character
            for character in seq
        ]
    )
    return res


def alphabetize_msa(
    msa: Dict[str, str],
    alphabet: List[str],
    out_of_alphabet_character: str,
) -> Dict[str, str]:
    """
    Alphabetize an MSA.
    """
    alphabet_set = set(alphabet)
    res = {
        seq_name: alphabetize_seq(
            seq=seq,
            alphabet_set=alphabet_set,
            out_of_alphabet_character=out_of_alphabet_character,
        )
        for (seq_name, seq) in msa.items()
    }
    return res


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=[
        "output_msa_dir",
    ],
    write_extra_log_files=True,
)
def alphabetize_msas(
    msa_dir: str,
    alphabet: Tuple[str],
    out_of_alphabet_character: str,
    families: List[str],
    num_processes: int,
    output_msa_dir: Optional[str] = None,
) -> None:
    """
    Alphabetize MSAs.

    NOTE: Not paralellized; `num_processes` is unused.
    """
    for family in families:
        msa_path = os.path.join(msa_dir, family + ".txt")
        msa = read_msa(msa_path)
        aphabetized_msa = alphabetize_msa(
            msa=msa,
            alphabet=alphabet,
            out_of_alphabet_character=out_of_alphabet_character,
        )
        output_msa_path = os.path.join(output_msa_dir, family + ".txt")
        write_msa(
            msa=aphabetized_msa,
            msa_path=output_msa_path,
        )


@peint_caching.cached()
def validate_secondary_structure_annotations(
    families: List[str],
    msa_dir: str,
    secondary_structure_dir: str,
) -> None:
    """
    Raises a ValueError if the secondary structure of some family does not
    match the length of the proteins in the MSA.
    """
    for family in families:
        msa_path = os.path.join(
            msa_dir,
            family + ".txt"
        )
        msa = read_msa(
            msa_path
        )
        secondary_structure_path = os.path.join(
            secondary_structure_dir,
            family + ".txt"
        )
        secondary_structure = read_secondary_structure(
            secondary_structure_path
        )
        secondary_structure_length = len(secondary_structure)
        protein_length = len(list(msa.values())[0])
        if secondary_structure_length != protein_length:
            raise ValueError(
                f"The secondary structure for protein family {family} "
                f"has length {secondary_structure_length} (as determined by "
                f"the file {secondary_structure_path}) whereas the MSA for "
                f"the family (found at {msa_path}) suggests {protein_length}."
            )


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_alignment_mask_dir"],
)
def _get_alignment_masks__cached(
    pfam_15k_msa_dir: str,
    num_sequences: Optional[int],
    families: List[str],
    return_full_length_unaligned_sequences: bool = True,
    output_alignment_mask_dir: Optional[str] = None,
    _version: str = "2024_05_11_v4",
):
    """
    Copy-pasta of cherryml's pfam_15k._subsample_pfam_15k_msa, but returning the
    information necessary to get alignment masks.
    """
    if not return_full_length_unaligned_sequences:
        raise ValueError(
            "This funtion is only meant to be used with full-length, "
            "gapless sequences!"
        )
    for family in families:
        pfam_15k_msa_path = os.path.join(
            pfam_15k_msa_dir, family + ".a3m"
        )
        if not os.path.exists(pfam_15k_msa_path):
            raise FileNotFoundError(f"MSA file {pfam_15k_msa_path} does not exist!")

        # Read MSA
        msa = []  # type: List[Tuple[str, str]]
        alignment_masks = []  # type: List[Tuple[str, str]]
        with open(pfam_15k_msa_path) as file:
            lines = list(file)
            n_lines = len(lines)
            for i in range(0, n_lines, 2):
                if not lines[i][0] == ">":
                    raise Exception("Protein name line should start with '>'")
                protein_name = lines[i][1:].strip()
                protein_seq = lines[i + 1].strip()
                # Lowercase amino acids in the sequence are insertions wrt
                # the reference sequence and should be removed if one
                # desires to obtain sequences which are all of the same
                # length (equal to the length of the reference sequence)
                # as in an MSA.
                if return_full_length_unaligned_sequences:
                    # In this case, we keep insertions wrt the reference
                    # sequence and remove gaps, thus obtaining the
                    # full-length, unaligned protein sequences.
                    # We just need to make all lowercase letters uppercase and
                    # remove gaps.
                    def make_upper_if_lower_and_clear_gaps(c: str) -> str:
                        # Note that c may be a gap, which is why we
                        # don't just do c.upper(); technically it
                        # works to do "-".upper() but I think it's
                        # confusing so I'll just write more code.
                        if c.islower():
                            return c.upper()
                        elif c.isupper():
                            return c
                        else:
                            assert c == "-"
                            return ""

                    ##### This is new
                    alignment_mask_vector = []
                    for c in protein_seq:
                        if c == "-":
                            alignment_mask_vector.append("")  # Gap '-'
                        elif c.islower():  # Is an insertion, i.e. not aligned against the reference sequence
                            alignment_mask_vector.append("0")  # Bad, e.g. 'a'
                        else:
                            assert c.isupper()
                            if c in utils.amino_acids:
                                alignment_mask_vector.append("1")  # Good, e.g. 'A'
                            else:
                                alignment_mask_vector.append("0")  # Weird characters, e.g. 'X', 'J'.
                    alignment_mask = "".join(alignment_mask_vector)

                    protein_seq = "".join(
                        [
                            make_upper_if_lower_and_clear_gaps(c)
                            for c in protein_seq
                        ]
                    )
                    assert len(alignment_mask) == len(protein_seq)
                else:
                    assert(False)  # We should NOT be here since this function is only used for full-length, gapless sequences
                msa.append((protein_name, protein_seq))
                alignment_masks.append((protein_name, alignment_mask))
            # Check that all sequences in the MSA have the same length.
            if not return_full_length_unaligned_sequences:
                for i in range(len(msa) - 1):
                    if len(msa[i][1]) != len(msa[i + 1][1]):
                        raise Exception(
                            f"Sequence\n{msa[i][1]}\nand\n{msa[i + 1][1]}\nin the "
                            f"MSA do not have the same length! ({len(msa[i][1])} vs"
                            f" {len(msa[i + 1][1])})"
                        )

        # Subsample MSA
        family_int_hash = (
            int(
                hashlib.sha512(
                    (family + "-_subsample_pfam_15k_msa").encode("utf-8")
                ).hexdigest(),
                16,
            )
            % 10**8
        )
        rng = np.random.default_rng(family_int_hash)
        nseqs = len(msa)
        if num_sequences is not None:
            max_seqs = min(nseqs, num_sequences)
            seqs_to_keep = [0] + list(
                rng.choice(range(1, nseqs, 1), size=max_seqs - 1, replace=False)
            )
            seqs_to_keep = sorted(seqs_to_keep)
            msa = [msa[i] for i in seqs_to_keep]
            alignment_masks = [alignment_masks[i] for i in seqs_to_keep]
        # msa_dict = dict(msa)  # Unused, since we only write out the alignment masks
        alignment_masks_dict = dict(alignment_masks)
        write_msa(
            msa=alignment_masks_dict, msa_path=os.path.join(output_alignment_mask_dir, family + ".txt")
        )
        secure_parallel_output(output_alignment_mask_dir, family)


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_dir"],
)
def _check_alignment_mask_keys_correct(
    msa_dir: str,
    alignment_mask_dir: str,
    families: List[str],
    output_dir: Optional[str] = None,
    _version: str = "2024_05_11_v4",
):
    """
    Simply checks that all the keys in the MSA and alignment masks are the same
    """
    for family in families:
        msa = read_msa(
            os.path.join(
                msa_dir, family + ".txt"
            )
        )
        alignment_mask = read_msa(
            os.path.join(
                alignment_mask_dir, family + ".txt"
            )
        )
        if sorted(list(msa.keys())) != sorted(list(alignment_mask.keys())):
            raise ValueError(
                f"Keys of:\n"
                f"{msa_dir}\n"
                f"and\n"
                f"{alignment_mask_dir}\n"
                "do not match!"
            )
        with open(
            os.path.join(
                output_dir,
                family + ".txt"
            ), "w"
        ) as output_file:
            output_file.write("OK!")
        secure_parallel_output(output_dir, family)


def get_a3m_families(
    a3m_dir: str,
    num_families: int = -1,
    random_seed: int = 42,
) -> List[str]:
    """
    Get family names from a3m files in a directory.

    Args:
        a3m_dir: Directory containing .a3m files.
        num_families: Number of families to return. If -1, returns all families.
            If greater than available families, returns all with a warning.
        random_seed: Seed for random shuffling when num_families > 0.

    Returns:
        Sorted list of family names (filenames without .a3m extension).
    """
    if not os.path.isdir(a3m_dir):
        raise ValueError(f"Directory does not exist: {a3m_dir}")

    all_families = sorted([
        f[:-4] for f in os.listdir(a3m_dir) if f.endswith(".a3m")
    ])

    if len(all_families) == 0:
        raise ValueError(f"No .a3m files found in {a3m_dir}")

    if num_families == -1:
        return all_families

    if num_families > len(all_families):
        warnings.warn(
            f"Requested {num_families} families but only {len(all_families)} "
            f"available. Returning all families."
        )
        return all_families

    random.Random(random_seed).shuffle(all_families)
    return sorted(all_families[:num_families])


@peint_caching.cached(
    exclude=["num_processes"],
    exclude_if_default=[
        "include_gaps",
        "return_full_length_unaligned_sequences",
        "do_train_test_split",
    ],
)
def a3m_dataset__cached(
    a3m_dir: str,
    num_sequences_per_family: int = 1024,
    tree_estimator_name: str = "FastTree",
    rate_matrix_path: str = "data/rate_matrices/wag.txt",
    num_rate_categories: int = 1,
    alphabet: Tuple[str] = tuple(
        list(utils.amino_acids) + [utils.gap_character]
    ),
    out_of_alphabet_character: str = utils.gap_character,
    num_processes: int = 32,
    num_families: int = -1,
    include_gaps: bool = True,
    return_full_length_unaligned_sequences: bool = False,
    do_train_test_split: bool = True,
    version: str = "2024_05_11_v4",
) -> Dict:
    """
    Create dataset from a directory of a3m files.

    Args:
        a3m_dir: Directory containing .a3m files.
        num_sequences_per_family: Number of sequences to subsample per family.
        tree_estimator_name: Tree estimation method (only "FastTree" supported).
        rate_matrix_path: Path to rate matrix for tree estimation.
        num_rate_categories: Number of rate categories for tree estimation.
        alphabet: Valid amino acid alphabet including gap character.
        out_of_alphabet_character: Character to replace out-of-alphabet chars.
        num_processes: Number of parallel processes.
        num_families: Number of families to use (-1 for all).
        include_gaps: Whether to include gaps in transitions.
        return_full_length_unaligned_sequences: Return full-length unaligned
            sequences instead of aligned sequences.
        do_train_test_split: If True, split tree into train/test halves.
            If False, extract transitions from the full tree.

    Returns:
        Dictionary with paths to transitions, MSAs, trees, and related data.
        If do_train_test_split=True, includes train_* and test_* keys.
        If do_train_test_split=False, includes transitions_dir, msa_dir, tree_dir.
    """
    if return_full_length_unaligned_sequences and include_gaps:
        raise ValueError(
            "return_full_length_unaligned_sequences=True and include_gaps=True "
            "is invalid since full-length unaligned proteins contain no gaps."
        )

    if tree_estimator_name == "FastTree":
        tree_estimator = partial(
            cherryml.phylogeny_estimation.fast_tree,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
        )
    else:
        raise ValueError(f"Unknown tree_estimator_name: '{tree_estimator_name}'")

    # Get families from the a3m directory
    families = get_a3m_families(a3m_dir, num_families)

    # Subsample the MSAs
    msa_dir = pfam_15k.subsample_pfam_15k_msas(
        pfam_15k_msa_dir=a3m_dir,
        num_sequences=num_sequences_per_family,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    dataset_statistics_str = report_dataset_statistics_str(
        msa_dir=msa_dir,
        families=families,
    )
    print(
        f"Dataset (subsampled to {num_sequences_per_family} seqs per family)"
        f" statistics:\n{dataset_statistics_str}"
    )

    # Estimate trees
    tree_dir = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        num_processes=num_processes,
    )["output_tree_dir"]

    # Alphabetize MSAs
    alphabetized_msa_dir = alphabetize_msas(
        msa_dir=msa_dir,
        alphabet=alphabet,
        out_of_alphabet_character=out_of_alphabet_character,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    # Learn site-specific rates from full MSA
    output_site_rates_4cat_dir = cherryml.fast_tree(
        msa_dir=alphabetized_msa_dir,
        families=families,
        rate_matrix_path=rate_matrix_path,
        num_rate_categories=4,
        num_processes=num_processes,
    )["output_site_rates_dir"]

    # Handle full-length unaligned sequences if requested
    alignment_mask_dir = None
    if return_full_length_unaligned_sequences:
        msa_dir = pfam_15k.subsample_pfam_15k_msas(
            pfam_15k_msa_dir=a3m_dir,
            num_sequences=num_sequences_per_family,
            families=families,
            num_processes=num_processes,
            return_full_length_unaligned_sequences=True,
        )["output_msa_dir"]

        alignment_mask_dir = _get_alignment_masks__cached(
            pfam_15k_msa_dir=a3m_dir,
            num_sequences=num_sequences_per_family,
            families=families,
            return_full_length_unaligned_sequences=True,
        )["output_alignment_mask_dir"]

        _check_alignment_mask_keys_correct(
            msa_dir=msa_dir,
            alignment_mask_dir=alignment_mask_dir,
            families=families,
        )

        # Don't alphabetize for full-length sequences
        alphabetized_msa_dir = msa_dir

    if not do_train_test_split:
        # Extract transitions from full tree
        transitions_dict = extract_transitions(
            msa_dir=alphabetized_msa_dir,
            tree_dir=tree_dir,
            families=families,
            num_processes=num_processes,
            include_gaps=include_gaps,
            alignment_mask_dir=alignment_mask_dir,
        )

        return {
            "families": families,
            "transitions_dir": transitions_dict["output_transitions_dir"],
            "msa_dir": alphabetized_msa_dir,
            "tree_dir": tree_dir,
            "site_rates_4cat_dir": output_site_rates_4cat_dir,
            "alignment_mask_dir": transitions_dict["output_alignment_mask_dir"],
            "transition_names_dir": transitions_dict["output_transition_names_dir"],
        }

    # Split MSAs into train/test based on tree structure
    msa_train_test_split_dict = msa_treewise_train_test_split(
        msa_dir=alphabetized_msa_dir,
        tree_dir=tree_dir,
        families=families,
        num_processes=num_processes,
    )

    # Extract transitions from train/test splits
    output_train_transitions_dict = extract_transitions(
        msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
        tree_dir=msa_train_test_split_dict["output_train_tree_dir"],
        families=families,
        num_processes=num_processes,
        include_gaps=include_gaps,
        alignment_mask_dir=alignment_mask_dir,
    )

    output_test_transitions_dict = extract_transitions(
        msa_dir=msa_train_test_split_dict["output_test_msa_dir"],
        tree_dir=msa_train_test_split_dict["output_test_tree_dir"],
        families=families,
        num_processes=num_processes,
        include_gaps=include_gaps,
        alignment_mask_dir=alignment_mask_dir,
    )

    return {
        "families": families,
        "train_transitions_dir": output_train_transitions_dict["output_transitions_dir"],
        "train_msa_dir": msa_train_test_split_dict["output_train_msa_dir"],
        "test_msa_dir": msa_train_test_split_dict["output_test_msa_dir"],
        "train_site_rates_4cat_dir": output_site_rates_4cat_dir,
        "test_transitions_dir": output_test_transitions_dict["output_transitions_dir"],
        "train_alignment_mask_dir": output_train_transitions_dict["output_alignment_mask_dir"],
        "test_alignment_mask_dir": output_test_transitions_dict["output_alignment_mask_dir"],
        "train_transition_names_dir": output_train_transitions_dict["output_transition_names_dir"],
        "test_transition_names_dir": output_test_transitions_dict["output_transition_names_dir"],
        "full_tree_dir": tree_dir,
        "train_tree_dir": msa_train_test_split_dict["output_train_tree_dir"],
        "test_tree_dir": msa_train_test_split_dict["output_test_tree_dir"],
    }


@peint_caching.cached(
    exclude=["num_processes"],
    exclude_if_default=[
        "include_gaps",
        "return_full_length_unaligned_sequences"
    ],
)
def pfam_15k__treewise_train_test_split__cached(
    num_sequences_per_family: int = 1024,
    tree_estimator_name: str = "FastTree",
    rate_matrix_path: str = "data/rate_matrices/wag.txt",
    num_rate_categories: int = 1,
    alphabet: Tuple[str] = tuple(
        list(utils.amino_acids) + [utils.gap_character]
    ),
    out_of_alphabet_character: str = utils.gap_character,
    num_processes: int = 32,
    num_families: int = 15051,  # For prototyping
    include_gaps: bool = True,
    return_full_length_unaligned_sequences: bool = False,
    version: str = "2024_05_11_v4",  # To update as we e.g. add covariates
) -> Dict:
    """
    Creates the training and testing data from the TrRosetta datasets.

    Returns the directories containing the training and testing data.

    Args:
        num_sequences_per_family: Number of sequences to subsample per family
            to limit the size of the MSAs. The size of the training and test
            MSAs will thus be ~num_sequences_per_family/2.
        rate_matrix_path: Path of rate matrix to use in the tree estimator.
        num_rate_categories: How many rate categories to use to create the
            dataset.
        alphabet: Alphabet of valid amino acids, including the gap character if
            it is modelled as its own state.
        out_of_alphabet_character: When character to use to replace
            out-of-alphabet characters.
        num_processes: Number of processes used to parallelize the dataset
            generation.
        num_families: Number of families to use; helpful for testing dataset
            generation on a small number of families.
        include_gaps: Whether to include gaps ("-") in the (x, y, t) train and
            test triples.
        return_full_length_unaligned_sequences: Whether to include insertions
            with respect to the reference sequence and thus return the full
            length, unaligned sequences. Can only be used with
            `include_gaps=False`.
    Returns:
        data_dirs["train_transitions_dir"]/{family}.txt contains the training
            transitions (x, y, t) for the given family. Whether or not gaps
            are included in x and y is determined by `include_gaps`.
            Similarly, whether insertions wrt the reference sequence are
            included (and therefore full-length, unaligned sequences are
            returned) is determined by
            `return_full_length_unaligned_sequences`.
        data_dirs["train_msa_dir"]/{family}.txt contains the training MSA for
            the given family.
        data_dirs["train_contact_map_dir"]/{family}.txt contains the training
            contact map for the given family.
        data_dirs["train_site_rates_4cat_dir"]/{family}.txt contains the site
            rates estimated by FastTree, which can be used as additional
            covariates in the model.
        data_dirs["train_secondary_structure_dir"]/{family}.txt contains the
            secondary structure annotations estimated by DSSP, which can be
            used as additional covariates in the model.
        data_dirs["families"] contains the list of families that should be used
            for training.
        data_dirs["test_transitions_dir"]/{family}.txt contains the TEST
            transitions (x, y, t) for the given family. Whether or not gaps
            are included in x and y is determined by `include_gaps`.
            Similarly, whether insertions wrt the reference sequence are
            included (and therefore full-length, unaligned sequences are
            returned) is determined by
            `return_full_length_unaligned_sequences`.
        data_dirs["test_alignment_mask_dir"] contains the binary alignment mask
            for each transition. For example, if in the original MSA the
            transition is (AG-L, A--L, 0.1) and the full transition is
            (AGLPD, AXLP, 0.1) then the alignment mask will be
            (11100, 1010, 0.1). This way, the mask provides a mapping between
            the sites in the full sequence and the aligned sites in the
            aligned transitions.
    """
    PFAM_15K_MSA_DIR = "/home/akoehl/Projects/protein-evolution/input_data/a3m"
    PFAM_15K_PDB_DIR = "/home/akoehl/Projects/protein-evolution/input_data/pdb"

    if return_full_length_unaligned_sequences and include_gaps:
        raise ValueError(
            "You specified return_full_length_unaligned_sequences = True "
            "and include_gaps = True, which makes no sense since the full "
            "length unaligned proteins contain no gaps."
        )

    if tree_estimator_name == "FastTree":
        tree_estimator = partial(
            cherryml.phylogeny_estimation.fast_tree,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
        )
    else:
        raise ValueError(
            "Unknown or unsupported tree_estimator_name: "
            f"'{tree_estimator_name}'"
        )

    # Get the families.
    families_all = pfam_15k.get_families(
        PFAM_15K_MSA_DIR,
    )
    random.Random(42).shuffle(families_all)
    families = sorted(families_all[:num_families])

    # Subsample the MSAs
    msa_dir = pfam_15k.subsample_pfam_15k_msas(
        pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
        num_sequences=num_sequences_per_family,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]
    dataset_statistics_str = report_dataset_statistics_str(
        msa_dir=msa_dir,
        families=families,
    )
    print(
        f"PFAM 15K (subsampled to {num_sequences_per_family} seqs per familiy)"
        f" statistics:\n{dataset_statistics_str}"
    )

    # Estimate trees for each family
    tree_dir = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        num_processes=num_processes,
    )["output_tree_dir"]

    # Filter out-of-alphabet characters (thus "alphabetizing" them)
    alphabetized_msa_dir = alphabetize_msas(
        msa_dir=msa_dir,
        alphabet=alphabet,
        out_of_alphabet_character=out_of_alphabet_character,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    # Now split each MSA into a training and testing half based on the first
    # split of each tree
    msa_train_test_split_dict = msa_treewise_train_test_split(
        msa_dir=alphabetized_msa_dir,
        tree_dir=tree_dir,
        families=families,
        num_processes=num_processes,
    )

    # Now extract all cherries (x, y, t) from each tree to obtain the training
    # & testing data.
    output_train_transitions_dir = extract_transitions(
        msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
        tree_dir=msa_train_test_split_dict["output_train_tree_dir"],
        families=families,
        num_processes=num_processes,
        include_gaps=include_gaps,
    )["output_transitions_dir"]
    output_test_transitions_dir = extract_transitions(
        msa_dir=msa_train_test_split_dict["output_test_msa_dir"],
        tree_dir=msa_train_test_split_dict["output_test_tree_dir"],
        families=families,
        num_processes=num_processes,
        include_gaps=include_gaps,
    )["output_transitions_dir"]

    output_contact_map_dir = pfam_15k.compute_contact_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families,
        angstrom_cutoff=8.0,
        num_processes=num_processes,
    )["output_contact_map_dir"]

    output_distance_map_dir = compute_distance_maps(
        pfam_15k_pdb_dir=PFAM_15K_PDB_DIR,
        families=families,
        num_processes=num_processes,
    )["output_distance_map_dir"]

    # Learn site-specific rates (useful covariates for e.g. LG model)
    output_site_rates_4cat_dirs = cherryml.fast_tree(
        msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
        families=families,
        rate_matrix_path=rate_matrix_path,
        num_rate_categories=4,
        num_processes=num_processes,
    )["output_site_rates_dir"]

    # Get secondary structure annotations
    output_secondary_structure_dir = ""
    """
    output_secondary_structure_dir = compute_secondary_structure_annotations(
        pdb_dir=PFAM_15K_PDB_DIR,
        families=families,
        num_processes=num_processes,
    )["output_dir"]
    # Check that the secondary structure annotations have the correct length
    validate_secondary_structure_annotations(
        families=families,
        msa_dir=msa_dir,
        secondary_structure_dir=output_secondary_structure_dir,
    )
    """
    
    # If we want to use unaligned full-length proteins, it is time to
    # take care of this... Lots of copy-paste with above but shrug...
    output_train_alignment_mask_dir = None
    output_test_alignment_mask_dir = None
    output_train_transition_names_dir = None
    output_test_transition_names_dir = None
    if return_full_length_unaligned_sequences:
        # Need to compute the correct train_transitions_dir and
        # test_transitions_dir.
        # Subsample the MSAs, but retaining the FULL LENGTH sequences.
        msa_dir = pfam_15k.subsample_pfam_15k_msas(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            num_sequences=num_sequences_per_family,
            families=families,
            num_processes=num_processes,
            return_full_length_unaligned_sequences=True,
        )["output_msa_dir"]

        alignment_mask_dir = _get_alignment_masks__cached(
            pfam_15k_msa_dir=PFAM_15K_MSA_DIR,
            num_sequences=num_sequences_per_family,
            families=families,
            return_full_length_unaligned_sequences=True,
        )["output_alignment_mask_dir"]
        # Check that the MSAs and alignment masks have the same keys.
        _check_alignment_mask_keys_correct(
            msa_dir=msa_dir,
            alignment_mask_dir=alignment_mask_dir,
            families=families,
        )

        # Will NOT alphabetize since I don't want to introduce gaps this way.
        # We will just keep ambiguous amino acids such as X, J, etc.
        alphabetized_msa_dir = msa_dir
        # Now split each MSA into a training and testing half based on the
        # first split of each tree
        msa_train_test_split_dict = msa_treewise_train_test_split(
            msa_dir=alphabetized_msa_dir,
            tree_dir=tree_dir,
            families=families,
            num_processes=num_processes,
        )
        # Now extract all cherries (x, y, t) from each tree to obtain the
        # training & testing data.
        output_train_transitions_dict = extract_transitions(
            msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
            tree_dir=msa_train_test_split_dict["output_train_tree_dir"],
            families=families,
            num_processes=num_processes,
            include_gaps=include_gaps,
            alignment_mask_dir=alignment_mask_dir,
        )
        output_train_transitions_dir = output_train_transitions_dict["output_transitions_dir"]
        output_train_alignment_mask_dir = output_train_transitions_dict["output_alignment_mask_dir"]
        output_train_transition_names_dir = output_train_transitions_dict["output_transition_names_dir"]
        output_test_transitions_dict = extract_transitions(
            msa_dir=msa_train_test_split_dict["output_test_msa_dir"],
            tree_dir=msa_train_test_split_dict["output_test_tree_dir"],
            families=families,
            num_processes=num_processes,
            include_gaps=include_gaps,
            alignment_mask_dir=alignment_mask_dir,
        )
        output_test_transitions_dir = output_test_transitions_dict["output_transitions_dir"]
        output_test_alignment_mask_dir = output_test_transitions_dict["output_alignment_mask_dir"]
        output_test_transition_names_dir = output_test_transitions_dict["output_transition_names_dir"]

    data_dirs = {
        "families": families,
        "train_transitions_dir": output_train_transitions_dir,
        "train_msa_dir": msa_train_test_split_dict["output_train_msa_dir"],
        "train_contact_map_dir": output_contact_map_dir,
        "train_distance_map_dir": output_distance_map_dir,
        "train_site_rates_4cat_dir": output_site_rates_4cat_dirs,
        "train_secondary_structure_dir":  output_secondary_structure_dir,
        "test_transitions_dir": output_test_transitions_dir,
        "train_alignment_mask_dir": output_train_alignment_mask_dir,
        "test_alignment_mask_dir": output_test_alignment_mask_dir,
        "train_transition_names_dir": output_train_transition_names_dir,
        "test_transition_names_dir": output_test_transition_names_dir,

    }
    return data_dirs

@peint_caching.cached(
    exclude=["num_processes"],
    exclude_if_default=[
        "include_gaps",
        "return_full_length_unaligned_sequences"
    ],
)
def casp14__treewise_train_test_split__cached(
    num_sequences_per_family: int = 1024,
    tree_estimator_name: str = "FastTree",
    rate_matrix_path: str = "data/rate_matrices/wag.txt",
    num_rate_categories: int = 1,
    alphabet: Tuple[str] = tuple(
        list(utils.amino_acids) + [utils.gap_character]
    ),
    out_of_alphabet_character: str = utils.gap_character,
    num_processes: int = 32,
    num_families: int = 15051,  # For prototyping
    include_gaps: bool = True,
    return_full_length_unaligned_sequences: bool = False,
    version: str = "2024_05_11_v4",  # To update as we e.g. add covariates
) -> Dict:
    """
    Creates the training and testing data from the TrRosetta datasets.

    Returns the directories containing the training and testing data.

    Args:
        num_sequences_per_family: Number of sequences to subsample per family
            to limit the size of the MSAs. The size of the training and test
            MSAs will thus be ~num_sequences_per_family/2.
        rate_matrix_path: Path of rate matrix to use in the tree estimator.
        num_rate_categories: How many rate categories to use to create the
            dataset.
        alphabet: Alphabet of valid amino acids, including the gap character if
            it is modelled as its own state.
        out_of_alphabet_character: When character to use to replace
            out-of-alphabet characters.
        num_processes: Number of processes used to parallelize the dataset
            generation.
        num_families: Number of families to use; helpful for testing dataset
            generation on a small number of families.
        include_gaps: Whether to include gaps ("-") in the (x, y, t) train and
            test triples.
        return_full_length_unaligned_sequences: Whether to include insertions
            with respect to the reference sequence and thus return the full
            length, unaligned sequences. Can only be used with
            `include_gaps=False`.
    Returns:
        data_dirs["train_transitions_dir"]/{family}.txt contains the training
            transitions (x, y, t) for the given family. Whether or not gaps
            are included in x and y is determined by `include_gaps`.
            Similarly, whether insertions wrt the reference sequence are
            included (and therefore full-length, unaligned sequences are
            returned) is determined by
            `return_full_length_unaligned_sequences`.
        data_dirs["train_msa_dir"]/{family}.txt contains the training MSA for
            the given family.
        data_dirs["train_contact_map_dir"]/{family}.txt contains the training
            contact map for the given family.
        data_dirs["train_site_rates_4cat_dir"]/{family}.txt contains the site
            rates estimated by FastTree, which can be used as additional
            covariates in the model.
        data_dirs["train_secondary_structure_dir"]/{family}.txt contains the
            secondary structure annotations estimated by DSSP, which can be
            used as additional covariates in the model.
        data_dirs["families"] contains the list of families that should be used
            for training.
        data_dirs["test_transitions_dir"]/{family}.txt contains the TEST
            transitions (x, y, t) for the given family. Whether or not gaps
            are included in x and y is determined by `include_gaps`.
            Similarly, whether insertions wrt the reference sequence are
            included (and therefore full-length, unaligned sequences are
            returned) is determined by
            `return_full_length_unaligned_sequences`.
        data_dirs["test_alignment_mask_dir"] contains the binary alignment mask
            for each transition. For example, if in the original MSA the
            transition is (AG-L, A--L, 0.1) and the full transition is
            (AGLPD, AXLP, 0.1) then the alignment mask will be
            (11100, 1010, 0.1). This way, the mask provides a mapping between
            the sites in the full sequence and the aligned sites in the
            aligned transitions.
    """

    CASP14_MSA_DIR = "/scratch/users/akoehl/protein-evolution/input_data/casp14_a3m"

    if return_full_length_unaligned_sequences and include_gaps:
        raise ValueError(
            "You specified return_full_length_unaligned_sequences = True "
            "and include_gaps = True, which makes no sense since the full "
            "length unaligned proteins contain no gaps."
        )

    if tree_estimator_name == "FastTree":
        tree_estimator = partial(
            cherryml.phylogeny_estimation.fast_tree,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
        )
    else:
        raise ValueError(
            "Unknown or unsupported tree_estimator_name: "
            f"'{tree_estimator_name}'"
        )

    # Get the families.
    families_all = pfam_15k.get_families(
        CASP14_MSA_DIR,
    )


    if num_families >= len(families_all):
        families = families_all
    else:
        random.Random(42).shuffle(families_all)
        families = sorted(families_all[:num_families])

    # Subsample the MSAs
    msa_dir = pfam_15k.subsample_pfam_15k_msas(
        pfam_15k_msa_dir=CASP14_MSA_DIR,
        num_sequences=num_sequences_per_family,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]
    dataset_statistics_str = report_dataset_statistics_str(
        msa_dir=msa_dir,
        families=families,
    )
    print(
        f"CASP14 (subsampled to {num_sequences_per_family} seqs per familiy)"
        f" statistics:\n{dataset_statistics_str}"
    )

    # Estimate trees for each family
    tree_dir = tree_estimator(
        msa_dir=msa_dir,
        families=families,
        num_processes=num_processes,
    )["output_tree_dir"]

    # Filter out-of-alphabet characters (thus "alphabetizing" them)
    alphabetized_msa_dir = alphabetize_msas(
        msa_dir=msa_dir,
        alphabet=alphabet,
        out_of_alphabet_character=out_of_alphabet_character,
        families=families,
        num_processes=num_processes,
    )["output_msa_dir"]

    # Now split each MSA into a training and testing half based on the first
    # split of each tree
    msa_train_test_split_dict = msa_treewise_train_test_split(
        msa_dir=alphabetized_msa_dir,
        tree_dir=tree_dir,
        families=families,
        num_processes=num_processes,
    )

    # Now extract all cherries (x, y, t) from each tree to obtain the training
    # & testing data.
    output_train_transitions_dir = extract_transitions(
        msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
        tree_dir=msa_train_test_split_dict["output_train_tree_dir"],
        families=families,
        num_processes=num_processes,
        include_gaps=include_gaps,
    )["output_transitions_dir"]
    output_test_transitions_dir = extract_transitions(
        msa_dir=msa_train_test_split_dict["output_test_msa_dir"],
        tree_dir=msa_train_test_split_dict["output_test_tree_dir"],
        families=families,
        num_processes=num_processes,
        include_gaps=include_gaps,
    )["output_transitions_dir"]

    # Learn site-specific rates (useful covariates for e.g. LG model)
    output_site_rates_4cat_dirs = cherryml.fast_tree(
        msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
        families=families,
        rate_matrix_path=rate_matrix_path,
        num_rate_categories=4,
        num_processes=num_processes,
    )["output_site_rates_dir"]

    
    # If we want to use unaligned full-length proteins, it is time to
    # take care of this... Lots of copy-paste with above but shrug...
    output_train_alignment_mask_dir = None
    output_test_alignment_mask_dir = None
    output_train_transition_names_dir = None
    output_test_transition_names_dir = None
    if return_full_length_unaligned_sequences:
        # Need to compute the correct train_transitions_dir and
        # test_transitions_dir.
        # Subsample the MSAs, but retaining the FULL LENGTH sequences.
        msa_dir = pfam_15k.subsample_pfam_15k_msas(
            pfam_15k_msa_dir=CASP14_MSA_DIR,
            num_sequences=num_sequences_per_family,
            families=families,
            num_processes=num_processes,
            return_full_length_unaligned_sequences=True,
        )["output_msa_dir"]

        alignment_mask_dir = _get_alignment_masks__cached(
            pfam_15k_msa_dir=CASP14_MSA_DIR,
            num_sequences=num_sequences_per_family,
            families=families,
            return_full_length_unaligned_sequences=True,
        )["output_alignment_mask_dir"]
        # Check that the MSAs and alignment masks have the same keys.
        _check_alignment_mask_keys_correct(
            msa_dir=msa_dir,
            alignment_mask_dir=alignment_mask_dir,
            families=families,
        )

        # Will NOT alphabetize since I don't want to introduce gaps this way.
        # We will just keep ambiguous amino acids such as X, J, etc.
        alphabetized_msa_dir = msa_dir
        # Now split each MSA into a training and testing half based on the
        # first split of each tree
        msa_train_test_split_dict = msa_treewise_train_test_split(
            msa_dir=alphabetized_msa_dir,
            tree_dir=tree_dir,
            families=families,
            num_processes=num_processes,
        )
        # Now extract all cherries (x, y, t) from each tree to obtain the
        # training & testing data.
        output_train_transitions_dict = extract_transitions(
            msa_dir=msa_train_test_split_dict["output_train_msa_dir"],
            tree_dir=msa_train_test_split_dict["output_train_tree_dir"],
            families=families,
            num_processes=num_processes,
            include_gaps=include_gaps,
            alignment_mask_dir=alignment_mask_dir,
        )
        output_train_transitions_dir = output_train_transitions_dict["output_transitions_dir"]
        output_train_alignment_mask_dir = output_train_transitions_dict["output_alignment_mask_dir"]
        output_train_transition_names_dir = output_train_transitions_dict["output_transition_names_dir"]
        output_test_transitions_dict = extract_transitions(
            msa_dir=msa_train_test_split_dict["output_test_msa_dir"],
            tree_dir=msa_train_test_split_dict["output_test_tree_dir"],
            families=families,
            num_processes=num_processes,
            include_gaps=include_gaps,
            alignment_mask_dir=alignment_mask_dir,
        )
        output_test_transitions_dir = output_test_transitions_dict["output_transitions_dir"]
        output_test_alignment_mask_dir = output_test_transitions_dict["output_alignment_mask_dir"]
        output_test_transition_names_dir = output_test_transitions_dict["output_transition_names_dir"]

    data_dirs = {
        "families": families,
        "train_transitions_dir": output_train_transitions_dir,
        "train_msa_dir": msa_train_test_split_dict["output_train_msa_dir"],
        "test_msa_dir": msa_train_test_split_dict["output_test_msa_dir"],
        "train_site_rates_4cat_dir": output_site_rates_4cat_dirs,
        "test_transitions_dir": output_test_transitions_dir,
        "train_alignment_mask_dir": output_train_alignment_mask_dir,
        "test_alignment_mask_dir": output_test_alignment_mask_dir,
        "train_transition_names_dir": output_train_transition_names_dir,
        "test_transition_names_dir": output_test_transition_names_dir,
        "full_tree_dir": tree_dir,
        "train_tree_dir": msa_train_test_split_dict["output_train_tree_dir"],
        "test_tree_dir": msa_train_test_split_dict["output_test_tree_dir"],
    }
    return data_dirs


def example_pfam_15k__treewise_train_test_split(
    data_dir_filepath = 'tests/example_data'
): 
    """Returns an small subset of 3 families from pfam_15k__treewise_train_test_split.

    Args:
        data_dir_filepath (str, optional): filepath to directory where example data is stored. Defaults to 'tests/example_data_dirs'.
    """
    with open(os.path.join(data_dir_filepath, 'families.txt')) as file: 
        families = file.read() 
    families = families.split('\n')
    data_dirs = {
        "families": families,
        "train_transitions_dir": os.path.join(data_dir_filepath, 'train_transitions'),
        "train_msa_dir": os.path.join(data_dir_filepath, 'train_msas'),
        "train_contact_map_dir": os.path.join(data_dir_filepath, 'train_contact_maps'),
        "train_distance_map_dir": os.path.join(data_dir_filepath, 'train_distance_maps'),
        "train_site_rates_4cat_dir": os.path.join(data_dir_filepath, 'train_site_rates_4cat'),
        "train_secondary_structure_dir": os.path.join(data_dir_filepath, 'train_secondary_structure'),
        "test_transitions_dir": os.path.join(data_dir_filepath, 'test_transitions'),
    }
    return data_dirs