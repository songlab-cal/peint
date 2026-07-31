"""Parity gate: does the optimized tree produce the *same numbers* as the baseline?

Every Tier-1 optimization in this branch is supposed to be provably identical —
same kernels, same inputs, same batch composition, same reduction order. This
script is what proves it. It runs ``run_workload.py --mode dump`` twice as
separate subprocesses (once with ``--repo <baseline>``, once with
``--repo <fast>``) and diffs the resulting arrays.

    python benchmarks/inference/parity.py --tier 1

Tier 1 demands bit-exact equality and fails the run otherwise. Tier 2 covers
changes that deliberately alter batch composition (length bucketing, continuous
batching); it only reports the deviation, which should be on the order of the
difference you already get today from changing ``batch_size``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "run_workload.py")

# Workloads that are deterministic enough to compare bit-for-bit.
#   logits      - teacher-forced forward, no sampling
#   likelihood  - teacher-forced NLLs, no sampling
#   homology    - all-vs-all NLLs, no sampling
#   generate    - argmax decoding (p=0.0), so also deterministic
DEFAULT_WORKLOADS = ["logits", "likelihood", "generate"]


def _run_dump(repo: str, workload: str, out_prefix: str, extra: List[str]) -> None:
    cmd = [
        sys.executable, _RUNNER,
        "--repo", repo,
        "--workload", workload,
        "--mode", "dump",
        "--out", out_prefix,
    ] + extra
    print(f"[parity] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"[parity] dump failed for repo={repo} workload={workload}")
    # Surface just the identifying line so the log shows which package ran.
    for line in proc.stdout.splitlines():
        if line.startswith("[run_workload] protevo <-") or line.startswith("[run_workload] wrote"):
            print("   " + line)


def _compare_array(base: str, fast: str) -> Tuple[bool, Dict[str, object]]:
    a = np.load(base)
    b = np.load(fast)
    if a.shape != b.shape:
        return False, {"reason": "shape mismatch", "baseline": a.shape, "fast": b.shape}
    exact = bool(np.array_equal(a, b))
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    finite = diff[np.isfinite(diff)]
    denom = np.abs(a.astype(np.float64))
    rel = finite.max() / max(denom.max(), 1e-30) if finite.size else 0.0
    return exact, {
        "n": int(a.size),
        "max_abs_diff": float(finite.max()) if finite.size else 0.0,
        "mean_abs_diff": float(finite.mean()) if finite.size else 0.0,
        "max_rel_diff": float(rel),
    }


def _compare_sequences(base: str, fast: str) -> Tuple[bool, Dict[str, object]]:
    with open(base) as fh:
        a = json.load(fh)["sequences"]
    with open(fast) as fh:
        b = json.load(fh)["sequences"]
    if len(a) != len(b):
        return False, {"reason": "count mismatch", "baseline": len(a), "fast": len(b)}
    mismatched = [i for i, (s, t) in enumerate(zip(a, b)) if s != t]
    return not mismatched, {
        "n": len(a),
        "n_mismatched": len(mismatched),
        "first_mismatch": mismatched[0] if mismatched else None,
    }


def _compare_hits(base: str, fast: str) -> Tuple[bool, Dict[str, object]]:
    with open(base) as fh:
        a = json.load(fh)["hits"]
    with open(fast) as fh:
        b = json.load(fh)["hits"]
    if len(a) != len(b):
        return False, {"reason": "hit count mismatch", "baseline": len(a), "fast": len(b)}
    keys_match = all(x["query_id"] == y["query_id"] and x["db_id"] == y["db_id"]
                     for x, y in zip(a, b))
    sa = np.array([x["score"] for x in a], dtype=np.float64)
    sb = np.array([y["score"] for y in b], dtype=np.float64)
    diff = np.abs(sa - sb)
    # Ranking is what homology detection actually consumes, so check it too.
    rank_match = bool(np.array_equal(np.argsort(sa, kind="stable"),
                                     np.argsort(sb, kind="stable")))
    return bool(keys_match and np.array_equal(sa, sb)), {
        "n": len(a),
        "keys_match": keys_match,
        "ranking_identical": rank_match,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
    }


_COMPARERS = {
    "logits": ("{}.npy", _compare_array),
    "likelihood": ("{}.npy", _compare_array),
    "generate": ("{}.json", _compare_sequences),
    "homology": ("{}.json", _compare_hits),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-repo", default=os.path.join(_HERE, "..", "..", "..", "peint"),
                    help="Pristine tree to compare against (default ../../../peint)")
    ap.add_argument("--fast-repo", default=os.path.join(_HERE, "..", ".."),
                    help="Optimized tree (default: this worktree)")
    ap.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS),
                    help=f"Comma-separated subset of {sorted(_COMPARERS)}")
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2],
                    help="1 = require bit-exact equality; 2 = report deviation only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-flash", action="store_true")
    ap.add_argument("--n-targets", type=int, default=256)
    ap.add_argument("--n-sequences", type=int, default=32)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--workdir", default=None,
                    help="Where to keep the dumps (default: a temp dir, deleted after)")
    ap.add_argument("--out", default=None, help="Optional JSON path for the verdict")
    args = ap.parse_args()

    baseline = os.path.abspath(args.baseline_repo)
    fast = os.path.abspath(args.fast_repo)
    if baseline == fast:
        raise SystemExit("[parity] baseline and fast repos are the same path")

    extra = ["--device", args.device,
             "--n-targets", str(args.n_targets),
             "--n-sequences", str(args.n_sequences),
             "--decode-steps", str(args.decode_steps)]
    if args.no_flash:
        extra.append("--no-flash")

    workdir_ctx = (tempfile.TemporaryDirectory() if args.workdir is None
                   else None)
    workdir = args.workdir or workdir_ctx.name
    os.makedirs(workdir, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}
    failures: List[str] = []
    try:
        for workload in [w for w in args.workloads.split(",") if w]:
            if workload not in _COMPARERS:
                raise SystemExit(f"[parity] unknown workload {workload!r}")
            pattern, comparer = _COMPARERS[workload]
            base_prefix = os.path.join(workdir, f"baseline_{workload}")
            fast_prefix = os.path.join(workdir, f"fast_{workload}")

            _run_dump(baseline, workload, base_prefix, extra)
            _run_dump(fast, workload, fast_prefix, extra)

            exact, detail = comparer(pattern.format(base_prefix), pattern.format(fast_prefix))
            detail["exact"] = exact
            results[workload] = detail

            status = "EXACT" if exact else ("DEVIATES" if args.tier == 2 else "FAIL")
            print(f"[parity] {workload:<12} {status:<9} {json.dumps(detail)}")
            if not exact and args.tier == 1:
                failures.append(workload)
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    verdict = {
        "tier": args.tier,
        "baseline_repo": baseline,
        "fast_repo": fast,
        "results": results,
        "passed": not failures,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(verdict, fh, indent=2)
        print(f"[parity] wrote {args.out}")

    if failures:
        raise SystemExit(f"[parity] TIER-1 PARITY FAILED for: {', '.join(failures)}")
    print(f"[parity] PASS (tier {args.tier})")


if __name__ == "__main__":
    main()
