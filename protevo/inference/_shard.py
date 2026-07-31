"""Data-parallel inference across the GPUs of a single node.

Every PEINT inference workload is embarrassingly parallel over its work list —
generation over root sequences, likelihood over query batches, homology
all-vs-all over reference sequences. None of them needs gradient synchronization,
so this deliberately uses plain process fan-out rather than DDP: no NCCL, no
process group, nothing that can hang on a watchdog.

Why this module exists rather than ``multiprocessing.Pool``: the two places in
the package that already try to parallelize inference
(``simulation/_simulate_on_tree.py`` and ``time_mle/t_mle.py``) put a live CUDA
``nn.Module`` into the argument list handed to ``pool.imap``. Under fork that
shares an already-initialized CUDA context; under spawn it tries to pickle the
model. Neither works, so in practice every inference workload has been
single-GPU. Here each rank loads the checkpoint itself, which is the only thing
that actually works — and it costs one model load per rank, not per item.

Determinism: each item is seeded from its own index, not from the rank, so
results do not depend on how many GPUs happened to be available. Results are
merged back in the original item order for the same reason.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

# A worker receives (shard, device, rank) and returns a list of
# (original_index, result) pairs.
WorkerFn = Callable[[List[Tuple[int, Any]], torch.device, int], List[Tuple[int, Any]]]


def available_gpus() -> int:
    """Number of visible CUDA devices (0 if none)."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def item_seed(base_seed: int, index: int) -> int:
    """Seed for one work item.

    Derived from the item's position in the *original* list so that a run on 8
    GPUs reproduces a run on 1 GPU exactly. Keep this out of any per-rank RNG.
    """
    return (base_seed + index) % (2**31 - 1)


def shard_indices(n_items: int, n_shards: int, rank: int) -> List[int]:
    """Round-robin assignment of item indices to ``rank``.

    Round-robin rather than contiguous blocks because work items are often sorted
    by size (e.g. proteomes by sequence count); contiguous blocks would hand one
    rank all the large ones and leave the rest idle.
    """
    return list(range(rank, n_items, n_shards))


def _worker_entry(rank: int, worker_fn, items, n_shards, out_dir, base_seed, device_ids):
    """Body of one rank. Module-level so it survives spawn pickling."""
    device_id = device_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    idxs = shard_indices(len(items), n_shards, rank)
    shard = [(i, items[i]) for i in idxs]
    logger.info("rank %d/%d on cuda:%d, %d items", rank, n_shards, device_id, len(shard))
    print(f"[shard] rank {rank}/{n_shards} -> cuda:{device_id}, {len(shard)} items", flush=True)

    torch.manual_seed(base_seed + rank)
    results = worker_fn(shard, device, rank)

    with open(os.path.join(out_dir, f"part_{rank}.pkl"), "wb") as fh:
        pickle.dump(results, fh)
    print(f"[shard] rank {rank} wrote {len(results)} results", flush=True)


def run_sharded(
    items: Sequence[Any],
    worker_fn: WorkerFn,
    num_gpus: Optional[int] = None,
    out_dir: Optional[str] = None,
    base_seed: int = 0,
    device_ids: Optional[Sequence[int]] = None,
) -> List[Any]:
    """Run ``worker_fn`` over ``items``, sharded across GPUs, and merge the results.

    Args:
        items: The work list. Must be picklable — these cross a process boundary.
        worker_fn: Module-level function ``(shard, device, rank) -> [(idx, result)]``.
            It is pickled by ``spawn``, so a lambda or a closure will not work; a
            ``functools.partial`` over a module-level function does, and is the
            usual way to bind a checkpoint path or a shared corpus. It should load
            whatever model it needs *inside* the call, once, then loop over
            ``shard``.
        num_gpus: How many ranks to run. Defaults to every visible GPU. ``1`` runs
            in-process, which keeps tracebacks readable when debugging.
        out_dir: Where ranks drop their partial results. A temp dir by default.
        base_seed: Combined with the item index by :func:`item_seed`.
        device_ids: Explicit CUDA device ordinals. Defaults to ``range(num_gpus)``.

    Returns:
        Results in the original order of ``items``.

    Raises:
        RuntimeError: If a rank produced no output file (it died).
    """
    n_items = len(items)
    if n_items == 0:
        return []

    n_gpus = available_gpus()
    if num_gpus is None:
        num_gpus = max(n_gpus, 1)
    num_gpus = max(1, min(num_gpus, n_items))
    if n_gpus and num_gpus > n_gpus:
        logger.warning("requested %d GPUs but only %d visible; using %d",
                       num_gpus, n_gpus, n_gpus)
        num_gpus = n_gpus

    if device_ids is None:
        device_ids = list(range(num_gpus))
    device_ids = list(device_ids)[:num_gpus]

    # Single rank: skip the process machinery entirely.
    if num_gpus == 1:
        device = torch.device("cuda", device_ids[0]) if n_gpus else torch.device("cpu")
        if n_gpus:
            torch.cuda.set_device(device_ids[0])
        torch.manual_seed(base_seed)
        shard = list(enumerate(items))
        results = worker_fn(shard, device, 0)
        return _merge([results], n_items)

    tmp_ctx = tempfile.TemporaryDirectory() if out_dir is None else None
    work_dir = out_dir or tmp_ctx.name
    os.makedirs(work_dir, exist_ok=True)

    try:
        mp.spawn(
            _worker_entry,
            args=(worker_fn, list(items), num_gpus, work_dir, base_seed, device_ids),
            nprocs=num_gpus,
            join=True,
        )

        parts = []
        for rank in range(num_gpus):
            path = os.path.join(work_dir, f"part_{rank}.pkl")
            if not os.path.exists(path):
                raise RuntimeError(
                    f"rank {rank} produced no output ({path} missing) - it most "
                    f"likely crashed; check the job log above for its traceback"
                )
            with open(path, "rb") as fh:
                parts.append(pickle.load(fh))
        return _merge(parts, n_items)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def _merge(parts: Sequence[List[Tuple[int, Any]]], n_items: int) -> List[Any]:
    """Reassemble per-rank ``(index, result)`` pairs into original order."""
    merged: List[Any] = [None] * n_items
    seen = set()
    for part in parts:
        for idx, value in part:
            merged[idx] = value
            seen.add(idx)
    missing = sorted(set(range(n_items)) - seen)
    if missing:
        raise RuntimeError(f"{len(missing)} work items produced no result, e.g. {missing[:5]}")
    return merged
