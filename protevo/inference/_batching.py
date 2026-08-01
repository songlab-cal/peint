"""Length-aware batching for likelihood scoring — opt-in, off by default.

Every batching path in the package pads to the batch maximum and consumes
sequences in input order. On a proteome, where lengths run from ~50 to ~1000
residues, that means most of a batch can be padding: a batch of 32 holding one
900-residue sequence and 31 short ones costs 32 x 900 padded positions to score
maybe 4000 real ones.

Sorting by length before batching removes most of that — but measured on this
model it is barely worth doing, which is why it is off by default. On an A5000,
cutting padding waste from 43% to 5% and the batch count from 64 to 21 sped up
``evaluate_likelihood`` by 3%, and homology all-vs-all not at all. The attention
path already unpads internally (``unpad_input`` + the varlen kernels), so it costs
what the real tokens cost whatever shape the batch is; only the FFN, LayerNorms
and LM head see padding. See ``benchmarks/inference/README.md`` for the numbers.

It was also expected to perturb scores, since it changes which sequences share a
batch. Measured, it does not: packed and unpacked results are bit-identical over
512 likelihood targets and 1560 homology pairs. That follows from the same
varlen property — a sequence's result does not depend on its batch-mates. Treat
it as an observation on one GPU and one set of shapes rather than a guarantee
(cuBLAS may split reductions differently as the batch dimension changes), which
is why this stays behind a flag with a check attached.

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


def length_sorted_batches(lengths: Sequence[int], batch_size: int) -> List[List[int]]:
    """Fixed-width batches of similar-length items, shortest first.

    Used by generation, where the cost driver is different from likelihood. There
    the waste was padded positions, which flash-attention largely skips; here it is
    *sequential decode steps*, and nothing skips those:

    * ``max_decode_steps`` defaults to twice the longest source in the batch, and
    * the loop runs until every row has emitted ``<eos>``,

    so one 900-residue source makes a batch of 50-residue sources run 1800 steps
    each. Grouping similar lengths cuts that directly. Batch width stays fixed
    because generation memory scales with rows x steps, not with a token budget.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def decode_step_waste(lengths: Sequence[int], batches: Sequence[Sequence[int]],
                      steps_per_residue: int = 2) -> dict:
    """Decode steps actually run vs the minimum each sequence needed.

    A batch runs ``steps_per_residue * max(length in batch)`` steps for every row,
    so a row needing fewer pays the difference. This is the generation analogue of
    :func:`padding_waste`, and unlike padded positions it is real sequential work.
    """
    needed = sum(steps_per_residue * L for L in lengths)
    paid = sum(len(b) * steps_per_residue * max(lengths[i] for i in b)
               for b in batches if b)
    return {
        "steps_needed": needed,
        "steps_paid": paid,
        "waste_fraction": (paid - needed) / paid if paid else 0.0,
        "n_batches": len(batches),
    }


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
