"""Evaluation of PEINT against classical site-independent models.

PEINT scores unaligned sequences; LG and WAG score alignment columns. The
functions here run PEINT on unaligned transitions and project the resulting
per-residue log-likelihoods back onto the alignment, writing them in the same
format as ``peint.models.evaluate_lg_model_transitions_log_likelihood__cached``.

    from peint.evaluation import (
        AlignedTransitionsDataset,
        evaluate_transitions_log_likelihood_per_site,
    )
"""

from ._aligned_transitions import (
    DEFAULT_MAX_LENGTH,
    AlignedTransition,
    AlignedTransitionsDataset,
    list_families,
)
from ._likelihood import (
    GAP_LOG_LIKELIHOOD,
    evaluate_peint_model_transitions_log_likelihood__cached,
    evaluate_transitions_log_likelihood_per_site,
    sum_over_sites,
)

__all__ = [
    # Data
    "AlignedTransition",
    "AlignedTransitionsDataset",
    "list_families",
    "DEFAULT_MAX_LENGTH",
    # Evaluation
    "evaluate_transitions_log_likelihood_per_site",
    "evaluate_peint_model_transitions_log_likelihood__cached",
    "sum_over_sites",
    "GAP_LOG_LIKELIHOOD",
]
