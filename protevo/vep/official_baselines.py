"""Per-assay Spearman from ProteinGym's officially released zero-shot scores.

This is the **maintained, canonical** source for the model baselines PEINT is
compared against. The ProteinGym release ships one score column per published
model inside each per-assay file under
``ProteinGym3/input_data/ProteinGym_v1.3/zero_shot_substitutions_scores/``
(e.g. ``ESMC-300M``, ``ESM2_150M``, ``ESM2_650M``, ``ESM3``, ``ESM1v_ensemble``).
Reading those directly reproduces the leaderboard numbers exactly and keeps every
baseline on one footing with the PEINT evaluation.

This is the single source of truth for those baselines; running the models
ourselves only reproduces these numbers, so reported comparisons come from here.

Output schema matches ``compute_fitness.py`` (``DMS_id/family, assay_type,
spearman``) so official baselines and PEINT checkpoints aggregate identically.
"""
import argparse

import pandas as pd
import scipy.stats

from protevo.vep._config import PROTEINGYM_DIR

RELEASED_SCORES_DIR = (
    PROTEINGYM_DIR / "input_data" / "ProteinGym_v1.3" / "zero_shot_substitutions_scores"
)
DMS_REFERENCE = PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"

# Sensible default set for PEINT comparisons: the ESM-C / ESM2 backbones plus a
# couple of standard references. Any column present in the release is valid.
DEFAULT_MODELS = [
    "ESMC-300M",
    "ESMC-600M",
    "ESM2_150M",
    "ESM2_650M",
    "ESM3",
    "ESM1v_ensemble",
]


def released_zero_shot_spearman(models, max_len=1022, dms_ids=None):
    """Per-assay Spearman between ``DMS_score`` and each released model's score.

    Args:
        models: released score column names (e.g. ``["ESMC-300M", "ESM2_150M"]``).
        max_len: drop assays whose ``target_seq`` exceeds this length, matching the
            PEINT evaluation which skips sequences > 1022.
        dms_ids: optional restriction to a subset of assays (default: all).

    Returns:
        Long DataFrame ``[DMS_id, assay_type, model, spearman]``.
    """
    ref = pd.read_csv(DMS_REFERENCE).set_index("DMS_id")
    ids = list(ref.index) if dms_ids is None else dms_ids
    models = list(models)
    rows = []
    for dms_id in ids:
        if dms_id not in ref.index:
            continue
        target_seq = ref.loc[dms_id, "target_seq"]
        if not isinstance(target_seq, str) or len(target_seq) > max_len:
            continue
        score_file = RELEASED_SCORES_DIR / f"{dms_id}.csv"
        if not score_file.exists():
            continue
        df = pd.read_csv(
            score_file,
            usecols=lambda c: c == "DMS_score" or c in models,
            low_memory=False,
        )
        assay_type = ref.loc[dms_id, "coarse_selection_type"]
        for model in models:
            if model not in df.columns:
                continue
            pair = df[["DMS_score", model]].dropna()
            if pair["DMS_score"].nunique() < 2 or pair[model].nunique() < 2:
                continue
            rho = scipy.stats.spearmanr(pair["DMS_score"], pair[model]).correlation
            rows.append((dms_id, assay_type, model, float(rho)))
    return pd.DataFrame(rows, columns=["DMS_id", "assay_type", "model", "spearman"])


def class_averaged(per_assay):
    """Mean-of-class-means Spearman over ``assay_type``, indexed by model.

    Per-assay Spearman -> mean within each ProteinGym function class -> mean across
    classes (the ProteinGym aggregation).
    """
    within = per_assay.groupby(["model", "assay_type"])["spearman"].mean()
    return within.groupby("model").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--out", required=True, help="per-assay long CSV to write")
    args = ap.parse_args()

    per_assay = released_zero_shot_spearman(args.models, args.max_len)
    per_assay.to_csv(args.out, index=False)
    ca = class_averaged(per_assay)
    print(f"released zero-shot (max_len={args.max_len}) per-assay -> {args.out}\n")
    for model in args.models:
        sub = per_assay[per_assay.model == model]
        if len(sub):
            print(f"  {model:14s} n={sub.DMS_id.nunique():3d}  class-avg={ca.get(model, float('nan')):.4f}")


if __name__ == "__main__":
    main()
