"""Length-aware batching for likelihood scoring — opt-in, off by default.

Every batching path in the package pads to the batch maximum and consumes
sequences in input order. On a proteome, where lengths run from ~50 to ~1000
residues, that means most of a batch can be padding: a batch of 32 holding one
900-residue sequence and 31 short ones costs 32 x 900 padded positions to score
maybe 4000 real ones.

Sorting by length before batching removes most of that. It is deliberately **not**
the default, because it changes which sequences share a batch, and batch
composition changes GEMM tiling and flash-attention accumulation order — so NLLs
move in the last few significant figures, exactly as they already do today if you
change ``batch_size``. That makes it a Tier-2 change under this branch's rules:
available behind a flag, never silently on, and always reported with a measured
deviation against the Tier-1 path.

Ordering constraint worth knowing about: batches are emitted largest-first. The
encoder K/V cache in ``EncoderCachedFlashMHCA`` is allocated on the first batch
and reallocated (losing ``cache_size``, hence the cached encoder) if a later batch
has *more* rows. With fixed-size batches only the final batch can be smaller, so
this never came up; with variable sizes it would silently corrupt every batch
after the largest. See :func:`largest_batch_first`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

# Padded positions per batch. 16384 = 32 sequences of 512, i.e. roughly what a
# batch_size=32 run costs on typical protein lengths, so it is a like-for-like
# default rather than a quiet increase in memory.
DEFAULT_MAX_TOKENS = 16384


def fixed_size_batches(n_items: int, batch_size: int) -> List[List[int]]:
    """Index lists equivalent to the default ``y[i:i + batch_size]`` slicing."""
    return [list(range(i, min(i + batch_size, n_items)))
            for i in range(0, n_items, batch_size)]


def token_budget_batches(
    lengths: Sequence[int],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_batch_size: Optional[int] = None,
) -> List[List[int]]:
    """Group indices into batches of at most ``max_tokens`` *padded* positions.

    Sequences are visited shortest-first, so each batch holds sequences of similar
    length and the padding inside it is small. The cost of a batch is
    ``len(batch) * max(length in batch)`` — the padded rectangle actually sent to
    the GPU, not the sum of real lengths.

    Args:
        lengths: Sequence lengths, indexed the same way as the corpus.
        max_tokens: Padded-position budget per batch.
        max_batch_size: Optional hard cap on sequences per batch, for cases where
            the limit is batch dimension rather than memory.

    Returns:
        Lists of indices into ``lengths``. Every index appears exactly once. A
        sequence longer than ``max_tokens`` gets a batch to itself rather than
        being dropped.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))

    batches: List[List[int]] = []
    current: List[int] = []
    current_max = 0

    for idx in order:
        length = lengths[idx]
        padded_max = max(current_max, length)
        would_exceed = (
            (len(current) + 1) * padded_max > max_tokens
            or (max_batch_size is not None and len(current) + 1 > max_batch_size)
        )
        if current and would_exceed:
            batches.append(current)
            current, current_max = [], 0
            padded_max = length
        current.append(idx)
        current_max = padded_max

    if current:
        batches.append(current)
    return batches


def largest_batch_first(batches: List[List[int]]) -> List[List[int]]:
    """Reorder so the batch with the most rows runs first.

    Required for the encoder-cached evaluator: it sizes its K/V cache from the
    first batch it sees and reallocates — dropping ``cache_size`` to zero, and
    with it the cached encoder — whenever a later batch has more rows. Running the
    widest batch first means every later batch reads a prefix of a cache that is
    already large enough.

    ``sorted`` is stable, so batches of equal width keep their shortest-first
    order and the result stays deterministic.
    """
    return sorted(batches, key=lambda b: -len(b))


def padding_waste(lengths: Sequence[int], batches: Sequence[Sequence[int]]) -> dict:
    """Padded vs real positions for a batching, for reporting the win.

    Returns ``real``, ``padded`` and ``waste_fraction`` — the share of positions
    the GPU processes that carry no residue.
    """
    real = sum(lengths)
    padded = sum(len(b) * max(lengths[i] for i in b) for b in batches if b)
    return {
        "real_positions": real,
        "padded_positions": padded,
        "waste_fraction": (padded - real) / padded if padded else 0.0,
        "n_batches": len(batches),
    }
