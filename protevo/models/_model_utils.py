"""Utility functions for PEINT models.

This module contains lightweight utilities that do not require training dependencies.
"""

from typing import List
import json
import pandas as pd

import numpy as np

from protevo.utils import get_quantization_points_from_geometric_grid

VALID_BINS = np.array([float(f) for f in get_quantization_points_from_geometric_grid()])

def get_quantile_idx(quantiles: List[float], t: float) -> int:
    """Returns the quantile index that time t falls in.

    Args:
        quantiles (List[float]): List of len(quantiles)-1 quantiles where each quantile is denoted by [quantiles[i], quantiles[i+1]).
        t (float): time t that we want the quantile index of.

    Returns:
        int quantile_idx between [0, len(quantiles)-2] where t falls between quantiles[quantile_idx] and quantiles[quantile_idx+1]. If t is smaller than quantiles[0], it belongs in the first quantile. If t is greater than quantiles[-1], it belongs in the last quantile .
    """
    if t < quantiles[0]:
        return 0
    elif t > quantiles[-1]:
        return len(quantiles) - 2

    idx_to_insert_t = np.searchsorted(quantiles, t, "right")
    return idx_to_insert_t - 1

def _condition_on_non_gap(conditional_probability_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    NOTE: Assumes that the gap state is the last one in the alphabet!
    """
    if conditional_probability_matrix.columns[-1] != "-":
        raise ValueError(
            "It is assumed that the gap state is the last one! "
            "Last state was instead: "
            f"{conditional_probability_matrix.columns[-1]}"
        )

    data = conditional_probability_matrix.values.copy()
    row_sums = np.sum(data[:, :-1], axis=1, keepdims=True)
    data[:, :-1] /= row_sums
    data[:, -1] = 1.0

    res = pd.DataFrame(
        data,
        index=conditional_probability_matrix.index,
        columns=conditional_probability_matrix.columns
    )
    return res

def read_family_file(family_file):
    with open(family_file, "r") as f:
        famdata = json.load(f)
    return famdata["families"]


def gradient_norm(model):
    """Compute the L2 norm of gradients across all model parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm