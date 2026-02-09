"""
Implements the WAG model.
"""
import logging
import multiprocessing
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
from cherryml.estimation_end_to_end import lg_end_to_end_with_cherryml_optimizer

from protevo import caching as protevo_caching
from protevo import io, utils
from protevo.io import Tree
from protevo.models._model_utils import _condition_on_non_gap

MSAType = Dict[str, str]


def _init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)


_init_logger()


def create_dummy_tree_and_msa(
    transitions: List[Tuple[str, str, float]],
) -> Tuple[Tree, MSAType]:
    """
    Given the transitions, create a dummy tree to hold them.
    """
    # First, if (x, y, t) and (y, x, t) are transitions,
    # just keep one of them. This should exactly halve the data.
    deduped_transitions = sorted(
        list(set([(min(x, y), max(x, y), t) for (x, y, t) in transitions]))
    )
    if len(transitions) != 2 * len(deduped_transitions):
        raise ValueError(
            "Was not able to deduplicate transitions: "
            f"len(transitions) = {len(transitions)}, "
            f"len(deduped_transition) = {len(deduped_transitions)}."
        )
    # Now create the dummy MSA
    seqs = sorted(sum([[x, y] for (x, y, t) in deduped_transitions], []))
    if len(seqs) != len(set(seqs)):
        raise ValueError(
            "Each sequence should appear in exactly one (deduped) transition."
        )
    seq_names = {seq: f"seq-{i}" for i, seq in enumerate(seqs)}
    dummy_msa = {seq_names[seq]: seq for seq in seqs}
    # Now create the dummy tree.
    dummy_tree = Tree()
    dummy_tree.add_node("root")
    for i, (x, y, t) in enumerate(deduped_transitions):
        internal_node_name = f"internal-{i}"
        dummy_tree.add_nodes([internal_node_name, seq_names[x], seq_names[y]])
        dummy_tree.add_edges(
            [
                ("root", internal_node_name, 42.0),  # Length doesn't matter
                (internal_node_name, seq_names[x], t / 2.0),
                (internal_node_name, seq_names[y], t / 2.0),
            ]
        )
    return dummy_tree, dummy_msa


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_tree_dir", "output_msa_dir"],
    exclude_args=["num_processes"],
    write_extra_log_files=True,
)
def create_dummy_trees_and_msas(
    transitions_dir: str,
    families: List[str],
    num_processes: int = 1,
    output_tree_dir: Optional[str] = None,
    output_msa_dir: Optional[str] = None,
) -> None:
    """
    Note: not parallelized since this is not a bottleneck.
    """
    for family in families:
        transitions = io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        dummy_tree, dummy_msa = create_dummy_tree_and_msa(
            transitions=transitions,
        )
        io.write_tree(
            dummy_tree, os.path.join(output_tree_dir, family + ".txt")
        )
        io.write_msa(dummy_msa, os.path.join(output_msa_dir, family + ".txt"))


def create_dummy_site_rates_from_transitions(
    transitions: List[Tuple[str, str, float]]
) -> List[float]:
    sequences = list(set(sum([[x, y] for (x, y, t) in transitions], [])))
    if len(set([len(x) for x in sequences])) != 1:
        raise ValueError(
            "All sequences in the provided transitions should have the "
            "same length."
        )
    sequence_length = len(sequences[0])
    site_rates = [1.0] * sequence_length
    return site_rates


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=["output_site_rates_dir"],
    exclude_args=["num_processes"],
    write_extra_log_files=True,
)
def create_dummy_site_rates(
    transitions_dir: str,
    families: List[str],
    num_processes: int = 1,
    output_site_rates_dir: Optional[str] = None,
):
    """
    Create dummy site rates, all equal to 1.
    """
    for family in families:
        transitions = io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        site_rates = create_dummy_site_rates_from_transitions(
            transitions=transitions,
        )
        io.write_site_rates(
            site_rates,
            os.path.join(output_site_rates_dir, family + ".txt"),
        )


@protevo_caching.cached_computation(
    output_dirs=["output_model_dir"],
    exclude_args=["num_processes"],
    exclude_args_if_default=["use_cpp_counting_implementation"],
)
def train_wag_model__cached(
    train_transitions_dir: str,
    families: List[str],
    alphabet: List[str] = list(utils.amino_acids) + [utils.gap_character],
    use_cpp_counting_implementation: bool = True,
    num_processes: int = 1,
    output_model_dir: Optional[str] = None,
):
    """
    Train the WAG model on the given transitions.

    Uses the CherryML package under the hood. To be able to use CherryML,
    we create 'dummy' trees and MSAs, since CherryML was writen to optimize
    the composite likelihood of cherries in the trees. This composite
    likelihood agrees exactly with the log-likelihood of all the training
    transitions (and in fact, this is precisely the motivation of the protein
    evolution project).

    Args:
        training_data_dirs: Directories with the training data.
        families: Families to use for training.

    Returns:
        path where the trained model is stored.
    """
    # First create dummy trees from the transitions. These are essentially
    # start-shaped trees from which all the transitions hang as cherries.
    create_dummy_trees_and_msas_dirs = create_dummy_trees_and_msas(
        transitions_dir=train_transitions_dir,
        families=families,
        num_processes=num_processes,
    )
    dummy_tree_dir, dummy_msa_dir = (
        create_dummy_trees_and_msas_dirs["output_tree_dir"],
        create_dummy_trees_and_msas_dirs["output_msa_dir"],
    )
    # We also need dummy site ratesm since CherryML learn under the more
    # general LG model. The dummy site rates are thus all hardcoded to 1s.
    dummy_site_rates_dir = create_dummy_site_rates(
        transitions_dir=train_transitions_dir,
        families=families,
        num_processes=num_processes,
    )["output_site_rates_dir"]

    # Now just run CherryML on the dummy data.
    learned_rate_matrix_path = lg_end_to_end_with_cherryml_optimizer(
        msa_dir=dummy_msa_dir,
        families=families,
        tree_estimator=None,  # Because we are using GT transitions.
        initial_tree_estimator_rate_matrix_path=None,  # Idem.
        num_iterations=1,  # Idem.
        tree_dir=dummy_tree_dir,  # We pass in the GT trees.
        site_rates_dir=dummy_site_rates_dir,  # We pass in the dummy site rates
        alphabet=alphabet,
        num_processes_tree_estimation=num_processes,
        num_processes_counting=num_processes,
        num_processes_optimization=min(2, num_processes),
        use_cpp_counting_implementation=use_cpp_counting_implementation,
    )["learned_rate_matrix_path"]

    learned_rate_matrix = io.read_rate_matrix(learned_rate_matrix_path)
    io.write_rate_matrix(
        rate_matrix=learned_rate_matrix.to_numpy(),
        states=list(learned_rate_matrix.columns),
        rate_matrix_path=os.path.join(output_model_dir, "result.txt"),
    )


def evaluate_wag_model_transitions_log_likelihood_per_site(
    transitions: List[Tuple[str, str, float]],
    rate_matrix: pd.DataFrame,
    condition_on_non_gap: bool = False,
) -> List[List[float]]:
    """
    Compute the per-site log-likelihood of the given transitions under the WAG model.

    It is assumed that the rate_matrix represents a reversible model.

    The log-likelihood under the WAG model is given by:
    P(y_i | x_i, t) = log( exp(rate_matrix * t)[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        rate_matrix: The rate matrix parameter of the WAG model.
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
    Returns:
        lls: The per-site log-likelihood of each transition.
    """
    matrix_exponentials = utils.matrix_exponential_reversible(
        rate_matrix=rate_matrix.to_numpy(),
        exponents=[t for (x, y, t) in transitions],
    )
    lls = []
    for i, (x, y, t) in enumerate(transitions):
        if len(x) != len(y):
            raise ValueError(
                f"Transition has two sequences of different lengths: {x}, {y}."
            )
        mexp_df = pd.DataFrame(
            matrix_exponentials[i, :, :],
            index=rate_matrix.index,
            columns=rate_matrix.columns,
        )
        if condition_on_non_gap:
            mexp_df = _condition_on_non_gap(mexp_df)
        lls.append(
            [
                np.log(mexp_df.at[x_i, y_i]) for x_i, y_i in zip(x, y)
            ]
        )
    return lls


def evaluate_wag_model_transitions_log_likelihood(
    transitions: List[Tuple[str, str, float]],
    rate_matrix: pd.DataFrame,
) -> List[float]:
    """
    Compute the log-likelihood of the given transitions under the WAG model.

    It is assumed that the rate_matrix represents a reversible model.

    The log-likelihood under the WAG model is given by:
    P(y | x, t) = sum_i log( exp(rate_matrix * t)[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        rate_matrix: The rate matrix parameter of the WAG model.
    Returns:
        lls: The log-likelihood of each transition.
    """
    lls_per_site = evaluate_wag_model_transitions_log_likelihood_per_site(
        transitions=transitions,
        rate_matrix=rate_matrix,
    )
    res = [sum(x) for x in lls_per_site]
    return res


def _evaluate_wag_model_transitions_log_likelihood__cached__map_func(
    args: List,
):
    """
    Auxiliary version of
    "evaluate_wag_model_transitions_log_likelihood__cached"
    used for multiprocessing.
    """
    assert len(args) == 6
    transitions_dir = args[0]
    families = args[1]
    model_dir = args[2]
    output_transitions_log_likelihood_dir = args[3]
    output_transitions_log_likelihood_per_site_dir = args[4]
    condition_on_non_gap = args[5]
    for family in families:
        transitions = io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        rate_matrix = io.read_rate_matrix(os.path.join(model_dir, "result.txt"))
        ##### Now do the per-site LLs
        transitions_log_likelihood_per_site = (
            evaluate_wag_model_transitions_log_likelihood_per_site(
                transitions=transitions,
                rate_matrix=rate_matrix,
                condition_on_non_gap=condition_on_non_gap,
            )
        )
        io.write_transitions_log_likelihood_per_site(
            transitions_log_likelihood_per_site=transitions_log_likelihood_per_site,
            transitions_log_likelihood_per_site_path=os.path.join(
                output_transitions_log_likelihood_per_site_dir, family + ".txt"
            ),
        )
        protevo_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_per_site_dir,
            parallel_arg=family,
        )
        ##### Now add over sites.
        transitions_log_likelihood = [
            sum(x) for x in transitions_log_likelihood_per_site
        ]
        io.write_transitions_log_likelihood(
            transitions_log_likelihood=transitions_log_likelihood,
            transitions_log_likelihood_path=os.path.join(
                output_transitions_log_likelihood_dir, family + ".txt"
            ),
        )
        protevo_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_dir,
            parallel_arg=family,
        )


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_transitions_log_likelihood_dir",
        "output_transitions_log_likelihood_per_site_dir",
    ],
    exclude_args=["num_processes"],
    exclude_args_if_default=["condition_on_non_gap"],
    write_extra_log_files=True,
)
def evaluate_wag_model_transitions_log_likelihood__cached(
    transitions_dir: str,
    families: List[str],
    model_dir: str,
    condition_on_non_gap: bool = False,
    num_processes: int = 1,
    output_transitions_log_likelihood_dir: Optional[str] = None,
    output_transitions_log_likelihood_per_site_dir: Optional[str] = None,
    _version: str = "2024_03_20_v1",
) -> None:
    """
    Compute transitions log-likelihood under the WAG model.

    Rate matrix must be stored in {model_dir}/result.txt

    Args:
        transitions_dir: The directory with the transitions for which to
            compute the log-likelihood. The transitions for family 'family'
            should be '{family}.txt'
        families: List of families for which to compute the log-likelihood.
        model_dir: The directory containing the rate matrix in the file
            'result.txt'
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
        num_processes: How many processes to use to paralellize the likelihood
            evaluation. The parallelization is family-based.
        output_transitions_log_likelihood_dir: Where the log-likelihoods will
            get written. The log-likelihoods for family 'family' will be in
            the file '{family}.txt', with one line per transition.
        output_transitions_log_likelihood_per_site_dir: Where the per-site
            log-likelihoods will get written. The log-likelihoods for family
            'family' will be in the file '{family}.txt', with one line per
            transition.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        f"Going to run on {len(families)} families using {num_processes} "
        "processes"
    )

    map_args = [
        [
            transitions_dir,
            utils.get_process_args(process_rank, num_processes, families),
            model_dir,
            output_transitions_log_likelihood_dir,
            output_transitions_log_likelihood_per_site_dir,
            condition_on_non_gap,
        ]
        for process_rank in range(num_processes)
    ]

    map_func = _evaluate_wag_model_transitions_log_likelihood__cached__map_func
    if num_processes > 1:
        with multiprocessing.Pool(num_processes) as pool:
            list(
                tqdm.tqdm(
                    pool.imap(
                        map_func,
                        map_args,
                    ),
                    total=len(map_args),
                )
            )
    else:
        list(
            tqdm.tqdm(
                map(
                    map_func,
                    map_args,
                ),
                total=len(map_args),
            )
        )