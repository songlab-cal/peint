"""Assemble the computational-efficiency table for the release checkpoint.

Referee minor comment 2 asks for "computational efficiency, including training
cost, inference throughput, GPU memory usage, generation speed, and parameter
counts relative to both classical simulators and competing neural approaches".

This script covers the PEINT side of that: parameter counts (split into the
frozen backbone that users download and the PEINT layers the checkpoint actually
ships), plus throughput and peak GPU memory for each inference workload, taken
from the JSON reports that ``run_workload.py`` writes.

    python benchmarks/inference/efficiency_table.py \
        --results-dir results --out results/efficiency

Rows are labelled by (workload, tree, GPU count) so a baseline run and an
optimized run of the same workload sit next to each other.

Not covered here, and deliberately not faked:

* **Classical simulators (AliSim WAG/LG).** ``protevo.simulation._alisim`` shells
  out to an ``iqtree2`` binary. The ``iqtree2/`` submodule is not initialized in
  this worktree and no binary is on PATH, so there is nothing to time. To fill
  those rows: ``git submodule update --init --recursive`` then build with cmake,
  and time ``simulation/classical.py`` on the same trees. Note the result is a
  CPU wall-clock number and should be labelled as such — it is not comparable to
  a GPU throughput figure without stating core count.
* **Training cost.** Recoverable from the ablation run logs under
  ``/scratch/users/yufan.cao/protevo_ablations/logs/`` (768 sequences/step x 60k
  steps, at the GPU-hours those jobs recorded), not from anything measured here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_reports(results_dir: str):
    """Read every ``*.json`` report ``run_workload.py --mode bench`` produced."""
    reports = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or "rows" not in payload:
            continue  # parity verdicts and dump files, not bench reports
        for row in payload["rows"]:
            row = dict(row)
            row["_meta"] = payload.get("metadata", {})
            row["_source"] = os.path.relpath(path, results_dir)
            reports.append(row)
    return reports


def _tree_label(meta) -> str:
    repo = meta.get("repo", "")
    if repo.endswith("peint-fast"):
        return "optimized"
    if repo.endswith("peint"):
        return "baseline"
    return os.path.basename(repo) or "?"


def _throughput(row):
    """Pick the natural throughput figure for this workload, with its unit."""
    if "tok_per_s" in row:
        return row["tok_per_s"], "tokens/s"
    if "pairs_per_s" in row:
        return row["pairs_per_s"], "pairs/s"
    if "seq_per_s" in row:
        return row["seq_per_s"], "sequences/s"
    return float("nan"), ""


def build_rows(reports):
    """Flatten reports into table rows, sorted for side-by-side comparison."""
    rows = []
    for r in reports:
        meta = r["_meta"]
        value, unit = _throughput(r)
        rows.append({
            "workload": r.get("workload", "?"),
            "tree": _tree_label(meta),
            "gpus": meta.get("num_gpus", 1),
            "gpu": meta.get("gpu", "?"),
            "batch_size": r.get("batch_size", ""),
            "wall_ms": round(r.get("mean_ms", float("nan")), 1),
            "throughput": round(value, 1),
            "unit": unit,
            "peak_mem_mib": round(r.get("peak_reserved_mib", float("nan")), 0),
        })
    rows.sort(key=lambda d: (d["workload"], d["batch_size"] if d["batch_size"] != "" else 0,
                             d["gpus"], d["tree"]))
    return rows


def add_speedups(rows):
    """Annotate each optimized row with its speedup over the matching baseline."""
    baseline = {
        (r["workload"], r["batch_size"], r["gpus"]): r
        for r in rows if r["tree"] == "baseline"
    }
    for r in rows:
        key = (r["workload"], r["batch_size"], 1)
        base = baseline.get(key)
        if base is None or r["tree"] == "baseline" or not base["wall_ms"]:
            r["speedup"] = ""
            r["mem_ratio"] = ""
        else:
            r["speedup"] = f"{base['wall_ms'] / r['wall_ms']:.2f}x"
            r["mem_ratio"] = (f"{base['peak_mem_mib'] / r['peak_mem_mib']:.2f}x"
                              if r["peak_mem_mib"] else "")
    return rows


def param_counts(checkpoint: str, device: str = "cpu"):
    """Frozen-backbone vs PEINT-layer parameter split for the release model."""
    sys.path.insert(0, _HERE)
    import torch

    import harness
    import workloads

    model, _ = workloads.build_model(
        checkpoint, "standard", torch.device(device), use_flash=(device != "cpu")
    )
    return harness.count_params(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/efficiency")
    ap.add_argument("--checkpoint", default=None,
                    help="If given, also report parameter counts (loads the model)")
    ap.add_argument("--device", default="cpu",
                    help="Device for the parameter count only; cpu is fine and avoids a GPU")
    args = ap.parse_args()

    sys.path.insert(0, _HERE)
    import harness

    reports = _load_reports(args.results_dir)
    if not reports:
        raise SystemExit(f"no bench reports found under {args.results_dir}/ - "
                         f"run benchmarks/inference/bench.sbatch first")

    rows = add_speedups(build_rows(reports))
    columns = ["workload", "tree", "gpus", "gpu", "batch_size", "wall_ms",
               "throughput", "unit", "peak_mem_mib", "speedup", "mem_ratio"]

    metadata = {"n_reports": len(reports)}
    if args.checkpoint:
        params = param_counts(args.checkpoint, args.device)
        metadata["params"] = params
        print("\nParameter counts")
        print(harness.to_markdown([{
            "frozen backbone (ESM2, downloaded)": params["backbone_params"],
            "PEINT layers (shipped in ckpt)": params["peint_params"],
            "total": params["total_params"],
        }]))

    print(harness.to_markdown(rows, columns))
    harness.save_report(rows, args.out, columns=columns, metadata=metadata)


if __name__ == "__main__":
    main()
