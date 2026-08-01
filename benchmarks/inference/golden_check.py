"""Check both trees against the released golden logits fixture.

``tests/test_model_inference.py::test_logits_match_reference`` already compares
the model to ``protevo/tests/y_logits.npy``, but only through the Vanilla
(non-Flash, fp32) path and only at ``np.allclose`` tolerance. Most of the
optimization work on this branch lives in the Flash variants, which that test
never touches.

This closes the gap from the other direction: it dumps teacher-forced logits from
*both* the pristine tree and the optimized tree, in whichever precision path you
ask for, and reports each one's distance from the golden fixture. The claim worth
establishing is not "optimized matches golden exactly" — the Flash path runs bf16
and the fixture is fp32, so it cannot — but "optimized is no further from golden
than pristine is", i.e. the optimization introduced no drift of its own.

    python benchmarks/inference/golden_check.py                 # Flash path
    python benchmarks/inference/golden_check.py --no-flash      # Vanilla path
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_HERE, "run_workload.py")

# Tolerance used by tests/test_model_inference.py::test_logits_match_reference.
TEST_TOL = 1e-4


def _dump_logits(repo: str, out_prefix: str, no_flash: bool) -> np.ndarray:
    cmd = [sys.executable, _RUNNER, "--repo", repo, "--workload", "logits",
           "--mode", "dump", "--out", out_prefix]
    if no_flash:
        cmd.append("--no-flash")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"[golden] dump failed for {repo}")
    return np.load(f"{out_prefix}.npy")


def _distance(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "within_test_tol": bool(np.allclose(a, b, rtol=TEST_TOL, atol=TEST_TOL)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-repo", default=os.path.join(_HERE, "..", "..", "..", "peint"))
    ap.add_argument("--fast-repo", default=os.path.join(_HERE, "..", ".."))
    ap.add_argument("--no-flash", action="store_true",
                    help="Use the Vanilla fp32 path (what the pytest fixture uses)")
    args = ap.parse_args()

    baseline = os.path.abspath(args.baseline_repo)
    fast = os.path.abspath(args.fast_repo)
    golden_path = os.path.join(fast, "protevo", "tests", "y_logits.npy")
    golden = np.load(golden_path)
    print(f"[golden] fixture {golden_path} shape={golden.shape} dtype={golden.dtype}")
    print(f"[golden] path: {'Vanilla fp32' if args.no_flash else 'Flash bf16'}")

    with tempfile.TemporaryDirectory() as work:
        base_logits = _dump_logits(baseline, os.path.join(work, "base"), args.no_flash)
        fast_logits = _dump_logits(fast, os.path.join(work, "fast"), args.no_flash)

    if base_logits.shape != golden.shape:
        raise SystemExit(f"[golden] shape mismatch vs fixture: {base_logits.shape} "
                         f"!= {golden.shape}")

    base_vs_golden = _distance(base_logits, golden)
    fast_vs_golden = _distance(fast_logits, golden)
    fast_vs_base = _distance(fast_logits, base_logits)

    print(f"[golden] pristine  vs fixture : {base_vs_golden}")
    print(f"[golden] optimized vs fixture : {fast_vs_golden}")
    print(f"[golden] optimized vs pristine: {fast_vs_base}")

    # The load-bearing assertion: optimization added no drift of its own.
    if fast_vs_golden["max_abs_diff"] > base_vs_golden["max_abs_diff"]:
        raise SystemExit(
            "[golden] FAIL - optimized tree is further from the fixture than the "
            "pristine tree is; the optimization introduced drift"
        )
    print("[golden] PASS - optimized is no further from the fixture than pristine")

    if args.no_flash and not fast_vs_golden["within_test_tol"]:
        raise SystemExit("[golden] FAIL - Vanilla path outside the pytest tolerance")


if __name__ == "__main__":
    main()
