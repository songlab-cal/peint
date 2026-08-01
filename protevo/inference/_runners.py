"""Multi-GPU drivers for the three PEINT inference workloads.

Each driver splits its work list, hands the shards to :func:`run_sharded`, and
merges the results back in the original order. The worker functions are
module-level (so ``spawn`` can pickle them) and each loads the checkpoint exactly
once per rank, then loops.

**Result stability under sharding.** All three drivers are built so that output
does not depend on how many GPUs were used:

* homology / target scoring — each reference is scored independently, so a shard
  boundary cannot change any number;
* generation — the work list is chunked into fixed batches *before* sharding, so
  a given sequence always lands in the same batch with the same neighbours and
  the same seed, whatever the GPU count.

That last point is the reason generation shards over batches rather than over
individual sequences: batch composition determines both the RNG draw order and
the padded shape, so pinning it is what makes 8-GPU output match 1-GPU output.
"""

from __future__ import annotations

import contextlib
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from protevo.inference._shard import item_seed, run_sharded

logger = logging.getLogger(__name__)


def _autocast(device: torch.device, enabled: bool = True):
    """bf16 autocast on CUDA, no-op elsewhere.

    The Flash kernels reject fp32 outright, and ``PeintGenerator.generate`` has no
    autocast of its own, so callers must supply it — this mirrors what
    ``simulation/_simulate_on_tree.py`` does around ``generate``.
    """
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _chunk(items: Sequence[Any], size: int) -> List[List[Any]]:
    """Split into fixed-size chunks, preserving order."""
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _generate_worker(
    shard: List[Tuple[int, Any]],
    device: torch.device,
    rank: int,
    *,
    checkpoint: str,
    use_flash: bool,
    max_decode_steps: Optional[int],
    temperature: float,
    p: float,
    base_seed: int,
    autocast: bool,
) -> List[Tuple[int, List[str]]]:
    """Generate one batch of sequences per shard item."""
    from protevo.models import load_peint_model

    model, vocab = load_peint_model(
        checkpoint_path=checkpoint, device=device,
        model_type="generator", use_flash=use_flash,
    )
    model = model.eval()

    out: List[Tuple[int, List[str]]] = []
    for chunk_idx, batch in shard:
        sources = [s for s, _ in batch]
        times = [t for _, t in batch]

        # Seeded from the chunk's position in the original list, so the sampled
        # sequences are identical no matter which rank ran this chunk.
        torch.manual_seed(item_seed(base_seed, chunk_idx))

        encoded = [
            torch.tensor([vocab.cls_idx] + vocab.encode(s) + [vocab.eos_idx])
            for s in sources
        ]
        x = torch.nn.utils.rnn.pad_sequence(
            encoded, batch_first=True, padding_value=vocab.padding_idx
        ).to(device)
        t = torch.tensor(times, dtype=torch.float32).unsqueeze(-1).to(device)

        steps = max_decode_steps or 2 * max(len(s) for s in sources)
        with _autocast(device, autocast):
            generated = model.generate(
                x=x, t=t, max_decode_steps=steps, device=device,
                temperature=temperature, p=p,
            )
        out.append((chunk_idx, generated))
    return out


def generate_sharded(
    checkpoint: str,
    sources: Sequence[str],
    times: Sequence[float],
    batch_size: int = 32,
    num_gpus: Optional[int] = None,
    max_decode_steps: Optional[int] = None,
    temperature: float = 1.0,
    p: float = 1.0,
    use_flash: bool = True,
    base_seed: int = 0,
    autocast: bool = True,
) -> List[str]:
    """Generate one evolved sequence per ``(source, time)`` pair, across all GPUs.

    Args:
        checkpoint: Path to the PEINT checkpoint. Each rank loads it itself.
        sources: Source sequences ``x``.
        times: Evolutionary times ``t``, one per source.
        batch_size: Sequences per ``generate`` call. Also fixes the sharding unit,
            so changing it changes the sampled output (as it does today).
        num_gpus: Ranks to run. Defaults to every visible GPU.
        max_decode_steps: Defaults to twice the longest source *in each batch*.

    Returns:
        Generated sequences, in the order of ``sources``.
    """
    assert len(sources) == len(times), "sources and times must be the same length"
    batches = _chunk(list(zip(sources, times)), batch_size)

    worker = partial(
        _generate_worker,
        checkpoint=checkpoint, use_flash=use_flash,
        max_decode_steps=max_decode_steps, temperature=temperature, p=p,
        base_seed=base_seed, autocast=autocast,
    )
    per_batch = run_sharded(batches, worker, num_gpus=num_gpus, base_seed=base_seed)
    return [seq for batch in per_batch for seq in batch]


# --------------------------------------------------------------------------
# Likelihood / VEP scoring
# --------------------------------------------------------------------------

def _score_worker(
    shard: List[Tuple[int, Any]],
    device: torch.device,
    rank: int,
    *,
    checkpoint: str,
    use_flash: bool,
    batch_size: int,
    pack_by_length: bool = False,
    max_tokens: Optional[int] = None,
) -> List[Tuple[int, np.ndarray]]:
    """Score one (reference, targets, times) job per shard item."""
    from protevo.models import load_peint_model

    model, _ = load_peint_model(
        checkpoint_path=checkpoint, device=device,
        model_type="evaluator", use_flash=use_flash,
    )
    model = model.eval()

    out: List[Tuple[int, np.ndarray]] = []
    for job_idx, (ref_seq, targets, times) in shard:
        nlls = model.evaluate_likelihood(
            x=ref_seq, y=list(targets), t=list(times),
            device=device, batch_size=batch_size,
            pack_by_length=pack_by_length, max_tokens=max_tokens,
        )
        out.append((job_idx, np.atleast_1d(np.asarray(nlls, dtype=np.float64))))
    return out


def score_targets_sharded(
    checkpoint: str,
    jobs: Sequence[Tuple[str, Sequence[str], Sequence[float]]],
    batch_size: int = 32,
    num_gpus: Optional[int] = None,
    use_flash: bool = True,
    pack_by_length: bool = False,
    max_tokens: Optional[int] = None,
) -> List[np.ndarray]:
    """Score many ``(reference, targets, times)`` jobs across all GPUs.

    Each job is self-contained — one encoder pass for the reference, reused across
    its target batches — so sharding cannot change any score. This is the right
    entry point for VEP (one job, many variants, so prefer more targets per job)
    and for scoring many references in bulk.

    Returns:
        One NLL array per job, in the order of ``jobs``.
    """
    worker = partial(
        _score_worker, checkpoint=checkpoint, use_flash=use_flash, batch_size=batch_size,
        pack_by_length=pack_by_length, max_tokens=max_tokens,
    )
    return run_sharded(list(jobs), worker, num_gpus=num_gpus)


# --------------------------------------------------------------------------
# Homology detection (all-vs-all)
# --------------------------------------------------------------------------

def _homology_worker(
    shard: List[Tuple[int, Any]],
    device: torch.device,
    rank: int,
    *,
    checkpoint: str,
    sequences: Sequence[Tuple[str, str]],
    time: float,
    batch_size: int,
    use_flash: bool,
    pack_by_length: bool = False,
    max_tokens: Optional[int] = None,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Score every other sequence against each reference in this rank's shard."""
    from protevo.homology_detection._peint import PeintHomologySearcher, PeintSearchConfig

    searcher = PeintHomologySearcher(PeintSearchConfig(
        checkpoint_path=checkpoint,
        device=str(device),
        time=time,
        batch_size=batch_size,
        use_flash=use_flash,
        pack_by_length=pack_by_length,
        max_tokens=max_tokens,
    ))

    seq_ids = [sid for sid, _ in sequences]
    seqs = [seq for _, seq in sequences]

    out: List[Tuple[int, List[Dict[str, Any]]]] = []
    for ref_idx, _ in shard:
        ref_id, ref_seq = sequences[ref_idx]
        other = [j for j in range(len(sequences)) if j != ref_idx]
        if not other:
            out.append((ref_idx, []))
            continue

        nlls = searcher._compare_sequences(
            ref_seq, [seqs[j] for j in other], [time] * len(other)
        )
        # Returned as plain dicts so the results pickle without importing
        # HomologyHit in the parent.
        out.append((ref_idx, [
            {"query_id": seq_ids[j], "db_id": ref_id, "score": float(nll), "time": time}
            for j, nll in zip(other, nlls)
        ]))
    return out


def all_vs_all_sharded(
    checkpoint: str,
    sequences: Sequence[Tuple[str, str]],
    time: float = 1.0,
    batch_size: int = 32,
    num_gpus: Optional[int] = None,
    use_flash: bool = True,
    pack_by_length: bool = False,
    max_tokens: Optional[int] = None,
) -> List:
    """All-vs-all homology search, sharded over reference sequences.

    This is the workload with the most to gain: it is O(N^2) decoder passes and
    every reference is independent, so scaling is close to linear in GPU count.
    Because each reference is scored against the same query list regardless of
    which rank owns it, the hit list is identical to the single-GPU result.

    Returns:
        ``HomologyHit`` objects, ordered by reference then by query — the same
        order ``PeintHomologySearcher.all_vs_all`` produces.
    """
    from protevo.homology_detection._base import HomologyHit

    worker = partial(
        _homology_worker,
        checkpoint=checkpoint, sequences=list(sequences), time=time,
        batch_size=batch_size, use_flash=use_flash,
        pack_by_length=pack_by_length, max_tokens=max_tokens,
    )
    per_ref = run_sharded(list(range(len(sequences))), worker, num_gpus=num_gpus)

    hits = []
    for ref_hits in per_ref:
        for h in ref_hits:
            hits.append(HomologyHit(
                query_id=h["query_id"], db_id=h["db_id"], score=h["score"],
                extra={"time": h["time"]},
            ))
    return hits


def _proteomes_worker(
    shard: List[Tuple[int, Any]],
    device: torch.device,
    rank: int,
    *,
    checkpoint: str,
    proteomes: Dict[str, List[Tuple[str, str]]],
    distances: Optional[Dict[Tuple[str, str], float]],
    default_time: float,
    skip_same_proteome: bool,
    batch_size: int,
    use_flash: bool,
    pack_by_length: bool = False,
    max_tokens: Optional[int] = None,
) -> List[Tuple[int, List[Dict[str, Any]]]]:
    """Score one reference sequence per shard item, across proteomes."""
    from protevo.homology_detection._peint import PeintHomologySearcher, PeintSearchConfig

    searcher = PeintHomologySearcher(PeintSearchConfig(
        checkpoint_path=checkpoint,
        device=str(device),
        time=default_time,
        batch_size=batch_size,
        use_flash=use_flash,
        pack_by_length=pack_by_length,
        max_tokens=max_tokens,
    ))

    tokens_by_proteome = searcher.tokenize_proteomes(proteomes)
    # The query set depends only on the reference's proteome, so build it once per
    # proteome even though this rank's shard interleaves proteomes.
    query_sets: Dict[str, Any] = {}

    out: List[Tuple[int, List[Dict[str, Any]]]] = []
    for item_idx, (ref_proteome, ref_pos) in shard:
        if ref_proteome not in query_sets:
            query_sets[ref_proteome] = searcher.build_query_set(
                proteomes, ref_proteome, distances, skip_same_proteome, tokens_by_proteome
            )
        query_seqs, query_times, query_info, query_tokens = query_sets[ref_proteome]

        ref_id, ref_seq = proteomes[ref_proteome][ref_pos]
        if not query_seqs:
            out.append((item_idx, []))
            continue

        nlls = searcher._compare_sequences(ref_seq, query_seqs, query_times, query_tokens)
        out.append((item_idx, [
            {"query_id": qid, "db_id": ref_id, "score": float(nll),
             "time": tval, "ref_proteome": ref_proteome, "query_proteome": qprot}
            for nll, tval, (qid, qprot) in zip(nlls, query_times, query_info)
        ]))
    return out


def all_vs_all_proteomes_sharded(
    checkpoint: str,
    proteomes: Dict[str, List[Tuple[str, str]]],
    distances: Optional[Dict[Tuple[str, str], float]] = None,
    skip_same_proteome: bool = True,
    default_time: float = 1.0,
    batch_size: int = 32,
    num_gpus: Optional[int] = None,
    use_flash: bool = True,
    pack_by_length: bool = False,
    max_tokens: Optional[int] = None,
) -> List:
    """Cross-proteome all-vs-all, sharded over reference sequences.

    This is what ``python -m protevo.homology_detection all-vs-all --proteome-dir``
    runs, and it is O(total_refs x total_queries) decoder passes. Each reference is
    independent, so the merged hit list matches the single-GPU result exactly.

    Returns:
        ``HomologyHit`` objects in the same order as
        :meth:`PeintHomologySearcher.all_vs_all_proteomes`.
    """
    from protevo.homology_detection._base import HomologyHit

    # Work items in the same order the serial implementation visits them, so the
    # merged output ordering is identical.
    items = [
        (name, pos)
        for name in proteomes
        for pos in range(len(proteomes[name]))
    ]

    worker = partial(
        _proteomes_worker,
        checkpoint=checkpoint, proteomes=dict(proteomes), distances=distances,
        default_time=default_time, skip_same_proteome=skip_same_proteome,
        batch_size=batch_size, use_flash=use_flash,
        pack_by_length=pack_by_length, max_tokens=max_tokens,
    )
    per_ref = run_sharded(items, worker, num_gpus=num_gpus)

    hits = []
    for ref_hits in per_ref:
        for h in ref_hits:
            hits.append(HomologyHit(
                query_id=h["query_id"], db_id=h["db_id"], score=h["score"],
                extra={
                    "time": h["time"],
                    "ref_proteome": h["ref_proteome"],
                    "query_proteome": h["query_proteome"],
                },
            ))
    return hits
