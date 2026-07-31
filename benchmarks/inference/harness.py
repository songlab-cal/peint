"""Measurement primitives for PEINT inference benchmarking.

Everything else under ``benchmarks/inference/`` imports from here so that the
baseline tree (``../peint``) and the optimized tree (``../peint-fast``) are
measured with byte-identical instrumentation.

Why CUDA events rather than ``time.time()``: CUDA kernel launches are
asynchronous, so wall-clock around a launch measures the launch, not the work.
The only pre-existing timing in the package
(``protevo/simulation/_simulate_on_tree.py``) makes exactly that mistake.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import subprocess
from typing import Callable, Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def cuda_timeit(
    fn: Callable[[], object],
    warmup: int = 2,
    iters: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Time ``fn`` with CUDA events, falling back to a CPU timer off-GPU.

    Args:
        fn: Zero-argument callable to time. Its return value is discarded.
        warmup: Untimed calls run first (allocator warmup, autotuning, lazy
            RoPE/cos-sin table construction).
        iters: Timed repetitions.
        device: Device to synchronize on. Defaults to the current CUDA device.

    Returns:
        Dict with ``mean_ms``, ``std_ms``, ``min_ms``, ``max_ms``, ``iters``.
    """
    use_cuda = torch.cuda.is_available() and (device is None or device.type == "cuda")

    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize(device)

    samples: List[float] = []
    for _ in range(iters):
        if use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize(device)
            samples.append(start.elapsed_time(end))
        else:
            import time

            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)

    return {
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iters": iters,
    }


def peak_memory(fn: Callable[[], object], device: Optional[torch.device] = None) -> Dict[str, float]:
    """Run ``fn`` once and report peak CUDA memory in MiB.

    Returns zeros on CPU. ``allocated`` is what the caller's tensors occupy;
    ``reserved`` is what the caching allocator holds from the driver, which is
    the number that determines whether a job fits on a card.
    """
    if not torch.cuda.is_available():
        fn()
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize(device)
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


# --------------------------------------------------------------------------
# Model statistics
# --------------------------------------------------------------------------

def count_params(model: torch.nn.Module) -> Dict[str, int]:
    """Split parameter counts into frozen backbone vs trainable PEINT layers.

    The referee asks for parameter counts, and the honest number for PEINT is
    two numbers: the frozen ESM2 backbone (downloaded, not shipped) and the
    PEINT layers that the checkpoint actually contains. ``model.esm`` is the
    backbone attribute used throughout ``protevo.models._transformer``.
    """
    backbone = getattr(model, "esm", None)
    backbone_ids = {id(p) for p in backbone.parameters()} if backbone is not None else set()

    backbone_n = 0
    peint_n = 0
    trainable_n = 0
    for p in model.parameters():
        n = p.numel()
        if id(p) in backbone_ids:
            backbone_n += n
        else:
            peint_n += n
        if p.requires_grad:
            trainable_n += n

    return {
        "backbone_params": backbone_n,
        "peint_params": peint_n,
        "total_params": backbone_n + peint_n,
        "trainable_params": trainable_n,
    }


def env_metadata() -> Dict[str, str]:
    """Capture enough environment detail that a number is reproducible later."""
    meta = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = str(torch.cuda.device_count())
        cap = torch.cuda.get_device_capability(0)
        meta["compute_capability"] = f"{cap[0]}.{cap[1]}"
    else:
        meta["gpu"] = "none"
        meta["gpu_count"] = "0"
        meta["compute_capability"] = ""
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        meta["git_commit"] = ""
    return meta


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def to_markdown(rows: Sequence[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> str:
    """Render benchmark rows as a GitHub-flavoured markdown table."""
    if not rows:
        return "_(no rows)_\n"
    cols = list(columns) if columns else list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def save_report(
    rows: Sequence[Dict[str, object]],
    out_prefix: str,
    columns: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    """Write ``<prefix>.csv``, ``<prefix>.md`` and ``<prefix>.json`` side by side."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or ".", exist_ok=True)
    cols = list(columns) if columns else (list(rows[0].keys()) if rows else [])

    with open(f"{out_prefix}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(f"{out_prefix}.md", "w") as fh:
        if metadata:
            fh.write("<!-- " + json.dumps(metadata) + " -->\n\n")
        fh.write(to_markdown(rows, cols))

    with open(f"{out_prefix}.json", "w") as fh:
        json.dump({"metadata": metadata or {}, "rows": list(rows)}, fh, indent=2)

    print(f"[harness] wrote {out_prefix}.{{csv,md,json}}")
