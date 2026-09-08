"""
Implements the LG model.
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

from peint import caching as peint_caching
from peint import io, utils
from peint.models._model_utils import _condition_on_non_gap

from . import _wag

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


@peint_caching.cached_computation(
    output_dirs=["output_model_dir"],
    exclude_args=["num_processes"],
    exclude_args_if_default=["use_cpp_counting_implementation"],
)
def train_lg_model__cached(
    train_transitions_dir: str,
    train_site_rates_dir: str,
    families: List[str],
    alphabet: List[str] = list(utils.amino_acids) + [utils.gap_character],
    use_cpp_counting_implementation: bool = True,
    num_processes: int = 1,
    output_model_dir: Optional[str] = None,
):
    """
    Train the LG model on the given transitions.

    Uses the CherryML package under the hood. To be able to use CherryML,
    we create 'dummy' trees and MSAs, since CherryML was writen to optimize
    the composite likelihood of cherries in the trees. This composite
    likelihood agrees exactly with the log-likelihood of all the training
    transitions (and in fact, this is precisely the motivation of the protein
    evolution project).

    Args:
        training_data_dirs: Directories with the training data.
        train_site_rates_dir: Training site rates.
        families: Families to use for training.

    Returns:
        path where the trained model is stored.
    """
    # First create dummy trees from the transitions. These are essentially
    # start-shaped trees from which all the transitions hang as cherries.
    create_dummy_trees_and_msas_dirs = _wag.create_dummy_trees_and_msas(
        transitions_dir=train_transitions_dir,
        families=families,
        num_processes=num_processes,
    )
    dummy_tree_dir, dummy_msa_dir = (
        create_dummy_trees_and_msas_dirs["output_tree_dir"],
        create_dummy_trees_and_msas_dirs["output_msa_dir"],
    )

    # Now just run CherryML on the dummy data.
    learned_rate_matrix_path = lg_end_to_end_with_cherryml_optimizer(
        msa_dir=dummy_msa_dir,
        families=families,
        tree_estimator=None,  # Because we are using GT transitions.
        initial_tree_estimator_rate_matrix_path=None,  # Idem.
        num_iterations=1,  # Idem.
        tree_dir=dummy_tree_dir,  # We pass in the GT trees.
        site_rates_dir=train_site_rates_dir,  # We pass in training site rates
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


def evaluate_lg_model_transitions_log_likelihood_per_site(
    transitions: List[Tuple[str, str, float]],
    site_rates: List[float],
    rate_matrix: pd.DataFrame,
    condition_on_non_gap: bool = False,
) -> List[List[float]]:
    """
    Compute the per-site log-likelihood of the given transitions under the LG model.

    It is assumed that the rate_matrix represents a reversible model.

    The log-likelihood under the LG model is given by:
    P(y_i | x_i, t, r) = log( exp(rate_matrix * t * r[i])[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        site_rates: The site rates parameter of the LG model.
        rate_matrix: The rate matrix parameter of the LG model.
        condition_on_non_gap: If True, then the per-site probabilities will be
            renormalized after conditioning on the gap status.
    Returns:
        lls: The per-site log-likelihood of each transition.
    """
    rate_categories = sorted(list(set(site_rates)))
    rate_category_to_int = {rate: i for (i, rate) in enumerate(rate_categories)}
    matrix_exponentials = [
        utils.matrix_exponential_reversible(
            rate_matrix=rate_matrix.to_numpy(),
            exponents=[t * rate for (x, y, t) in transitions],
        )
        for rate in rate_categories
    ]
    res = []
    for i, (x, y, t) in enumerate(transitions):
        if len(x) != len(y):
            raise ValueError(
                f"Transition has two sequences of different lengths: {x}, {y}."
            )
        mexp_dfs = [
            pd.DataFrame(
                matrix_exponentials[rate_idx][i, :, :],
                index=rate_matrix.index,
                columns=rate_matrix.columns,
            )
            for rate_idx in range(len(rate_categories))
        ]
        if condition_on_non_gap:
            mexp_dfs = [
                _condition_on_non_gap(
                    mexp_dfs[rate_idx]
                )
                for rate_idx in range(len(rate_categories))
            ]
        assert len(x) == len(y)
        assert len(x) == len(site_rates)
        lls = [
            np.log(
                mexp_dfs[rate_category_to_int[site_rates[i]]].at[x[i], y[i]]
            )
            for i in range(len(x))
        ]
        res.append(lls)
    return res


def evaluate_lg_model_transitions_log_likelihood(
    transitions: List[Tuple[str, str, float]],
    site_rates: List[float],
    rate_matrix: pd.DataFrame,
) -> List[float]:
    """
    Compute the log-likelihood of the given transitions under the LG model.

    It is assumed that the rate_matrix represents a reversible model.

    The log-likelihood under the LG model is given by:
    P(y | x, t, r) = sum_i log( exp(rate_matrix * t * r[i])[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        site_rates: The site rates parameter of the LG model.
        rate_matrix: The rate matrix parameter of the LG model.
    Returns:
        lls: The log-likelihood of each transition.
    """
    lls_per_site = evaluate_lg_model_transitions_log_likelihood_per_site(
        transitions=transitions,
        site_rates=site_rates,
        rate_matrix=rate_matrix,
    )
    lls = [sum(x) for x in lls_per_site]
    return lls


def _evaluate_lg_model_transitions_log_likelihood__cached__map_func(
    args: List,
):
    """
    Auxiliary version of
    "evaluate_lg_model_transitions_log_likelihood__cached"
    used for multiprocessing.
    """
    assert len(args) == 7
    transitions_dir = args[0]
    site_rates_dir = args[1]
    families = args[2]
    model_dir = args[3]
    output_transitions_log_likelihood_dir = args[4]
    output_transitions_log_likelihood_per_site_dir = args[5]
    condition_on_non_gap = args[6]
    for family in families:
        transitions = io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        site_rates = io.read_site_rates(
            os.path.join(site_rates_dir, family + ".txt")
        )
        rate_matrix = io.read_rate_matrix(os.path.join(model_dir, "result.txt"))
        ##### Now do the per-site LLs
        transitions_log_likelihood_per_site = (
            evaluate_lg_model_transitions_log_likelihood_per_site(
                transitions=transitions,
                site_rates=site_rates,
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
        peint_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_per_site_dir,
            parallel_arg=family,
        )
        ##### Now sum over sites
        transitions_log_likelihood = [
            sum(x) for x in transitions_log_likelihood_per_site
        ]
        io.write_transitions_log_likelihood(
            transitions_log_likelihood=transitions_log_likelihood,
            transitions_log_likelihood_path=os.path.join(
                output_transitions_log_likelihood_dir, family + ".txt"
            ),
        )
        peint_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_dir,
            parallel_arg=family,
        )


@peint_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_transitions_log_likelihood_dir",
        "output_transitions_log_likelihood_per_site_dir",
    ],
    exclude_args=["num_processes"],
    exclude_args_if_default=["condition_on_non_gap"],
    write_extra_log_files=True,
)
def evaluate_lg_model_transitions_log_likelihood__cached(
    transitions_dir: str,
    site_rates_dir: str,
    families: List[str],
    model_dir: str,
    condition_on_non_gap: bool = False,
    num_processes: int = 1,
    output_transitions_log_likelihood_dir: Optional[str] = None,
    output_transitions_log_likelihood_per_site_dir: Optional[str] = None,
    _version: str = "2024_03_20_v1",
) -> None:
    """
    Compute transitions log-likelihood under the LG model.

    Rate matrix must be stored in {model_dir}/result.txt

    Args:
        transitions_dir: The directory with the transitions for which to
            compute the log-likelihood. The transitions for family 'family'
            should be in the file '{family}.txt'
        site_rates_dir: The directory with the site rates with which to
            compute the log-likelihood. The site rates for family 'family'
            should be in the file '{family}.txt'
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
            site_rates_dir,
            utils.get_process_args(process_rank, num_processes, families),
            model_dir,
            output_transitions_log_likelihood_dir,
            output_transitions_log_likelihood_per_site_dir,
            condition_on_non_gap,
        ]
        for process_rank in range(num_processes)
    ]

    map_func = _evaluate_lg_model_transitions_log_likelihood__cached__map_func
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