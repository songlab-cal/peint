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


# --------------------------------------------------------------------------
# Model FLOPs Utilization
# --------------------------------------------------------------------------

# Vendor peak dense bf16 tensor-core throughput (TFLOP/s) and HBM bandwidth
# (GB/s). Sparsity-doubled marketing numbers are deliberately NOT used. These are
# a cross-check only; `measure_peak_tflops` below is the number MFU is reported
# against, because it is what this machine actually reaches.
VENDOR_PEAK = {
    "NVIDIA RTX A5000": {"bf16_tflops": 54.2, "hbm_gbs": 768.0},
    "NVIDIA A100-SXM4-80GB": {"bf16_tflops": 312.0, "hbm_gbs": 2039.0},
    "NVIDIA A100-SXM4-40GB": {"bf16_tflops": 312.0, "hbm_gbs": 1555.0},
    "NVIDIA A100 80GB PCIe": {"bf16_tflops": 312.0, "hbm_gbs": 1935.0},
    "NVIDIA A100-PCIE-40GB": {"bf16_tflops": 312.0, "hbm_gbs": 1555.0},
    "NVIDIA H200": {"bf16_tflops": 989.0, "hbm_gbs": 4800.0},
}


def measure_peak_tflops(device=None, dtype=torch.bfloat16, n: int = 8192,
                        iters: int = 20) -> float:
    """Achievable dense bf16 GEMM throughput on this device, in TFLOP/s.

    MFU against a vendor headline number flatters or punishes a kernel for
    reasons that have nothing to do with the model. Calibrating against a large
    square matmul measured on the same card gives a ceiling the workload could
    actually have reached.
    """
    if not torch.cuda.is_available():
        return float("nan")
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        a @ b
    end.record()
    torch.cuda.synchronize(device)
    seconds = start.elapsed_time(end) / 1e3
    # 2 FLOPs per multiply-add.
    return (2.0 * n ** 3 * iters) / seconds / 1e12


def transformer_flops(n_layers: int, d_model: int, n_tokens: int,
                      ctx_len: int = 0, ffn_factor: int = 4,
                      cross_ctx_len: int = 0) -> float:
    """Forward FLOPs for ``n_tokens`` through ``n_layers`` transformer layers.

    Counts the two things that dominate:

    * **projections and FFN** - 2 FLOPs per MAC, per token:
      qkv+out = 4 d^2, FFN = 2 * ffn_factor * d^2, so 2 * (4 + 2f) d^2 per token.
    * **attention scores and values** - 2 * 2 * d * ctx per token (QK^T then PV),
      using ``ctx_len`` as the average context each token attends over. Add
      ``cross_ctx_len`` for a cross-attention sublayer (its own out/kv projections
      are folded into the 4 d^2 term as an approximation).

    LayerNorms, activations and the rotary embedding are omitted: they are O(d)
    per token against O(d^2), well under a percent here. This is an analytic
    count, not a profiler measurement - it answers "how far from the roofline",
    not "where did every cycle go".
    """
    per_token_dense = 2.0 * (4 + 2 * ffn_factor) * d_model ** 2
    per_token_attn = 4.0 * d_model * (ctx_len + cross_ctx_len)
    return n_layers * n_tokens * (per_token_dense + per_token_attn)


def peint_generation_flops(batch: int, src_len: int, steps: int,
                           d_model: int = 640, n_enc_backbone: int = 30,
                           n_enc_peint: int = 5, n_dec: int = 5,
                           vocab: int = 33) -> Dict[str, float]:
    """Analytic forward FLOPs for one batched ``PeintGenerator.generate`` call.

    Split into the prefill (frozen ESM2 backbone + PEINT encoder stack, run once
    over the source) and the decode loop (``steps`` single-token passes through
    the decoder, cross-attending to the source and self-attending to a prefix that
    grows to ``steps``).
    """
    prefill = transformer_flops(n_enc_backbone + n_enc_peint, d_model,
                                n_tokens=batch * src_len, ctx_len=src_len)
    # Self-attention context averages steps/2 over the run.
    decode = transformer_flops(n_dec, d_model, n_tokens=batch * steps,
                               ctx_len=steps / 2.0, cross_ctx_len=src_len)
    lm_head = 2.0 * batch * steps * d_model * vocab
    return {"prefill_flops": prefill, "decode_flops": decode + lm_head,
            "total_flops": prefill + decode + lm_head}


def mfu(total_flops: float, seconds: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization: achieved / peak, as a fraction."""
    if not seconds or not peak_tflops or peak_tflops != peak_tflops:
        return float("nan")
    return (total_flops / seconds / 1e12) / peak_tflops


def measure_peak_bandwidth(device=None, n_bytes: int = 2 * 1024**3,
                           iters: int = 20) -> float:
    """Achievable HBM bandwidth on this device, GB/s, via a large copy.

    The vendor figure is a ceiling no real kernel reaches. Measuring a big
    contiguous copy on the same card gives a denominator the decode loop could
    plausibly have hit, the same way measure_peak_tflops does for compute.
    """
    if not torch.cuda.is_available():
        return float("nan")
    n = n_bytes // 2  # bf16 elements
    src = torch.empty(n, device=device, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize(device)
    seconds = start.elapsed_time(end) / 1e3
    # copy touches each byte twice (read + write)
    return (2 * n_bytes * iters) / seconds / 1e9


def decode_bytes_per_step(batch: int, src_len: int, steps: int,
                          d_model: int = 640, n_heads: int = 20,
                          n_dec: int = 5, bytes_per_elem: int = 2) -> Dict[str, float]:
    """Memory traffic of one decode step, broken down.

    An earlier version of this counted **weights only** and reported ~1% HBM
    utilization, which was wrong by ~20x and led to calling the loop
    "overhead-bound". At realistic batch the KV cache dominates by ~450x:

    * cross-attention K/V - ``batch x src_len`` per layer, constant across the
      whole run but re-read on every single step, because attention must see all
      keys;
    * self-attention K/V - grows with position, averaging ``steps/2``;
    * decoder weights - streamed once per step, and negligible beside the above.

    The frozen ESM2 backbone is excluded: it runs only at prefill.
    """
    head_dim = d_model // n_heads
    kv_elem = 2 * n_heads * head_dim  # K and V
    cross = batch * src_len * kv_elem * bytes_per_elem * n_dec
    self_attn = batch * (steps / 2.0) * kv_elem * bytes_per_elem * n_dec
    weights = n_dec * (4 * d_model**2 + 2 * 4 * d_model**2) * bytes_per_elem
    return {"cross_kv_bytes": cross, "self_kv_bytes": self_attn,
            "weight_bytes": weights, "total_bytes": cross + self_attn + weights}


def decode_bandwidth_utilization(batch: int, src_len: int, steps: int,
                                 seconds: float, peak_gbs: float, **kw) -> float:
    """Fraction of achievable HBM bandwidth the decode loop actually reaches."""
    if not seconds or not peak_gbs or peak_gbs != peak_gbs:
        return float("nan")
    b = decode_bytes_per_step(batch, src_len, steps, **kw)["total_bytes"]
    achieved_gbs = (b * steps / 1e9) / seconds
    return achieved_gbs / peak_gbs
