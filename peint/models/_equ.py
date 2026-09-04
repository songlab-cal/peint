"""
Implements the EQU model.

EQU is the null model in which every state exchanges with every other at the
same rate. It knows nothing about amino acid chemistry, only about how much time
has passed, which makes it the floor that any real model has to clear.

There is nothing to learn from the training transitions, so the only thing to do
is write the rate matrix where the evaluators look for a model. EQU is then
scored through ``evaluate_wag_model_transitions_log_likelihood__cached``, which
is simply "one rate matrix, no site rates".
"""

import os
from typing import List, Optional

import numpy as np

from peint import caching as peint_caching
from peint import io, utils


def equ_rate_matrix(alphabet: List[str]) -> np.ndarray:
    """Rate matrix with equal exchangeabilities, one substitution per unit time.

    Matches the convention of ``data/rate_matrices/equ.txt``, but is built for
    whatever alphabet is passed so that the gap state can be included.
    """
    num_states = len(alphabet)
    rate_matrix = np.full((num_states, num_states), 1.0 / (num_states - 1))
    np.fill_diagonal(rate_matrix, -1.0)
    return rate_matrix


@peint_caching.cached_computation(output_dirs=["output_model_dir"])
def equ_model__cached(
    alphabet: List[str] = list(utils.amino_acids) + [utils.gap_character],
    output_model_dir: Optional[str] = None,
):
    """Write the EQU rate matrix in the layout the WAG and LG evaluators read.

    Args:
        alphabet: States of the model. The gap must come last, so that
            ``condition_on_non_gap`` can find it.

    Returns:
        Path where the model is stored.
    """
    io.write_rate_matrix(
        rate_matrix=equ_rate_matrix(alphabet),
        states=alphabet,
        rate_matrix_path=os.path.join(output_model_dir, "result.txt"),
    )
