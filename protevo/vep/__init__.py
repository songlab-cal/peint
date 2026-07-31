"""Variant Effect Prediction (VEP) with PEINT on the ProteinGym substitution benchmark.

Train a small PEINT head on a frozen ESM2 (or vESM) encoder using transitions extracted
from each ProteinGym family's MSA, then score mutants by the conditional log-likelihood
``log p(x_mut | x_wt, t)`` and correlate against the DMS assay (Spearman).

Pipeline (see README.md):
    MSAs -> transitions (_dms_datasets) -> TRAIN (train_peint_vep) -> SCORE (compute_fitness)

Paths, the encoder registry, and cache hashes live in ``_config``; import from there rather
than hard-coding.
"""
