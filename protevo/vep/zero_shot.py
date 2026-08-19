"""Single-sequence zero-shot VEP scoring (masked-marginals) for fair-esm-family encoders.

Scores each ProteinGym DMS variant by the sum over its point mutations of
``logP(mut | masked context) - logP(wt | masked context)`` -- ProteinGym's official
masked-marginals protocol. Each position of the wild type is masked in turn (L forward
passes, batched), independent of the number of variants, so even mega-scale assays are
cheap.

Intended for encoders that load via ``encoders.load_esm_model`` (stock ESM2, vESM) and
are **not** in the ProteinGym zero-shot release -- e.g. vESM. For released models
(ESM2, ESM-C, VESPA, ...) use ``official_baselines.py`` instead, which reads the shipped
per-variant scores directly.

Output matches ``compute_fitness`` / the figure loaders: per-assay
``[family, assay_type, spearman]`` over DMS assays with ``target_seq`` <= ``max_len``,
so it lands on the same core set as everything else. DMS CSVs are read via
``compute_fitness._dms_subs_folder`` (respects ``VEP_DMS_SUBS_DIR``).
"""
import argparse

import numpy as np
import pandas as pd
import scipy.stats
import torch

from protevo.vep._config import PROTEINGYM_DIR
from protevo.vep.compute_fitness import _dms_subs_folder
from protevo.vep.encoders import load_esm_model

AAS = "ACDEFGHIKLMNPQRSTVWY"


@torch.no_grad()
def wt_marginal_logp(model, vocab, wt, device):
    """Per-position log-probs from ONE forward over the unmasked wild type.

    This is vESM's intended protocol (see the ntranoslab/vesm README `get_llrs`): vESM is
    co-distilled to produce good log-likelihood ratios from the unmasked pass, so masking
    degrades it. Stock ESM/ESM-C prefer masked-marginals instead.
    """
    ids = torch.tensor([[vocab.cls_idx] + vocab.encode(wt) + [vocab.eos_idx]], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = model(ids)["logits"][0].float().cpu()
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def masked_marginal_logp(model, vocab, wt, device, mask_bs=128):
    """Per-position log-probs with each residue masked in turn (batched)."""
    ids = torch.tensor([vocab.cls_idx] + vocab.encode(wt) + [vocab.eos_idx], device=device)
    L = len(wt)
    logp = torch.full((L + 2, len(vocab)), float("nan"))
    for st in range(1, L + 1, mask_bs):
        en = min(st + mask_bs, L + 1)
        batch = ids.unsqueeze(0).repeat(en - st, 1).clone()
        for j, p in enumerate(range(st, en)):
            batch[j, p] = vocab.mask_idx
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(batch)["logits"].float().cpu()
        out = torch.log_softmax(out, dim=-1)
        for j, p in enumerate(range(st, en)):
            logp[p] = out[j, p]
    return logp


def score_variants(logp, wt, mutants, aa_idx):
    """Sum of per-position log-ratios for each ':'-joined variant (NaN if malformed)."""
    scores = []
    for mut in mutants.astype(str):
        total, ok = 0.0, True
        for m in mut.split(":"):
            wa, ma = m[0], m[-1]
            try:
                pos = int(m[1:-1])
            except ValueError:
                ok = False
                break
            if not (1 <= pos <= len(wt)) or wt[pos - 1] != wa or wa not in aa_idx or ma not in aa_idx:
                ok = False
                break
            total += float(logp[pos, aa_idx[ma]] - logp[pos, aa_idx[wa]])
        scores.append(total if ok else np.nan)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which_esm", required=True, help="load_esm_model key, e.g. vesm_150M")
    ap.add_argument("--out", required=True, help="per-assay spearman_results.csv to write")
    ap.add_argument(
        "--marginals",
        choices=["masked", "wt"],
        default="masked",
        help="masked = mask each position (ESM/ProteinGym default); wt = single unmasked "
        "forward LLR (vESM's distilled protocol -- masking degrades vESM).",
    )
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--mask_bs", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="score only first N assays (debug)")
    args = ap.parse_args()

    ref = pd.read_csv(
        PROTEINGYM_DIR / "reference_files" / "DMS_substitutions.csv"
    ).set_index("DMS_id")
    dms_dir = _dms_subs_folder()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_esm_model(args.which_esm, use_flash=(device == "cuda"))
    model = model.to(device).eval()
    aa_idx = {a: vocab.get_idx(a) for a in AAS}

    rows, skipped = [], 0
    for i, dms_id in enumerate(ref.index):
        if args.limit and len(rows) >= args.limit:
            break
        wt = ref.loc[dms_id, "target_seq"]
        if not isinstance(wt, str) or len(wt) > args.max_len:
            skipped += 1
            continue
        path = dms_dir / ref.loc[dms_id, "DMS_filename"]
        if not path.exists():
            skipped += 1
            continue
        df = pd.read_csv(path, low_memory=False)
        if args.marginals == "wt":
            logp = wt_marginal_logp(model, vocab, wt, device)
        else:
            logp = masked_marginal_logp(model, vocab, wt, device, args.mask_bs)
        df = df.assign(score=score_variants(logp, wt, df["mutant"], aa_idx)).dropna(subset=["score"])
        if df["score"].nunique() < 2 or df["DMS_score"].nunique() < 2:
            skipped += 1
            continue
        rho = float(scipy.stats.spearmanr(df["DMS_score"], df["score"]).correlation)
        cst = ref.loc[dms_id, "coarse_selection_type"]
        rows.append((dms_id, cst, rho))
        print(f"[{i + 1}/{len(ref)}] {dms_id} spearman={rho:.4f} n={len(df)} class={cst}", flush=True)

    out = pd.DataFrame(rows, columns=["family", "assay_type", "spearman"])
    out.to_csv(args.out, index=False)
    within = out.groupby("assay_type")["spearman"].mean()
    print(f"\n=== {args.which_esm} zero-shot ({args.marginals}-marginals) ===")
    print("per-class means:\n" + within.to_string())
    print(f"\nflat mean      = {out['spearman'].mean():.4f}")
    print(f"class-averaged = {within.mean():.4f}  ({len(within)} classes)")
    print(f"n_assays = {len(out)} | skipped = {skipped} | -> {args.out}")


if __name__ == "__main__":
    main()
