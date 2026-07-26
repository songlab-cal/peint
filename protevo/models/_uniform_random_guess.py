"""
Implements the uniform random guess model.

The model has no parameters and ignores its inputs: every amino acid is equally
likely at every site, whatever the source residue and however much time has
passed. It is the floor for the likelihood comparison, and the only baseline
that cannot benefit from either chemistry or evolutionary time.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import tqdm

from protevo import caching as protevo_caching
from protevo import io, utils

logger = logging.getLogger(__name__)


def evaluate_uniform_random_guess_model_transitions_log_likelihood_per_site(
    transitions: List[Tuple[str, str, float]],
    num_states: int = len(utils.amino_acids),
) -> List[List[float]]:
    """
    Compute the per-site log-likelihood of the given transitions under the
    uniform random guess model.

    Every scored column contributes ``-log(num_states)``. Gap columns contribute
    0, which is the same convention the other models follow: WAG and LG give a
    gap target probability 1 under ``condition_on_non_gap``, and PEINT writes 0
    into the columns its target does not occupy. Every model therefore gets the
    same free columns, and their mean per-site likelihoods stay comparable.

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        num_states: Size of the state space guessed over, the 20 amino acids by
            default.

    Returns:
        lls: The per-site log-likelihood of each transition.
    """
    guess = -np.log(num_states)
    return [
        [0.0 if y_i == utils.gap_character else guess for y_i in y]
        for (_, y, _) in transitions
    ]


@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    output_dirs=[
        "output_transitions_log_likelihood_dir",
        "output_transitions_log_likelihood_per_site_dir",
    ],
    exclude_args_if_default=["num_states"],
    write_extra_log_files=True,
)
def evaluate_uniform_random_guess_model_transitions_log_likelihood__cached(
    transitions_dir: str,
    families: List[str],
    num_states: int = len(utils.amino_acids),
    output_transitions_log_likelihood_dir: Optional[str] = None,
    output_transitions_log_likelihood_per_site_dir: Optional[str] = None,
    _version: str = "2026_07_25_v1",
) -> None:
    """
    Compute transitions log-likelihood under the uniform random guess model.

    There is no model to train and nothing to parallelize: the log-likelihood of
    a column depends only on whether it holds a residue.

    Args:
        transitions_dir: The directory with the transitions for which to
            compute the log-likelihood. The transitions for family 'family'
            should be in the file '{family}.txt'
        families: List of families for which to compute the log-likelihood.
        num_states: Size of the state space guessed over.
        output_transitions_log_likelihood_dir: Where the log-likelihoods will
            get written. The log-likelihoods for family 'family' will be in
            the file '{family}.txt', with one line per transition.
        output_transitions_log_likelihood_per_site_dir: Where the per-site
            log-likelihoods will get written. The log-likelihoods for family
            'family' will be in the file '{family}.txt', with one line per
            transition.
    """
    logger.info(f"Going to run on {len(families)} families")

    for family in tqdm.tqdm(families):
        transitions = io.read_transitions(
            os.path.join(transitions_dir, family + ".txt")
        )
        transitions_log_likelihood_per_site = (
            evaluate_uniform_random_guess_model_transitions_log_likelihood_per_site(
                transitions=transitions,
                num_states=num_states,
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

        io.write_transitions_log_likelihood(
            transitions_log_likelihood=[
                sum(lls) for lls in transitions_log_likelihood_per_site
            ],
            transitions_log_likelihood_path=os.path.join(
                output_transitions_log_likelihood_dir, family + ".txt"
            ),
        )
        protevo_caching.secure_parallel_output(
            output_dir=output_transitions_log_likelihood_dir,
            parallel_arg=family,
        )
