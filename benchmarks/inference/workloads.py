"""The three inference workloads under optimization, defined once.

Imported by both ``run_workload.py`` (which points ``sys.path`` at either the
baseline tree or the optimized tree) and ``efficiency_table.py``. Nothing here
imports ``protevo`` at module scope, so the caller controls which copy of the
package gets used.

The benchmark corpus is **synthetic and seeded**: it is derived from the
repository's own golden fixture (``protevo/tests/example_transition.txt``) by
deterministic point mutation and length jitter. That keeps benchmarking
independent of the shared read-only data caches while still exercising the
realistic sequence-length regime (~200-350 residues) and, with jitter on, the
ragged-batch regime that length bucketing targets.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


# --------------------------------------------------------------------------
# Fixtures and synthetic corpus
# --------------------------------------------------------------------------

def load_fixture(repo_root: str) -> Tuple[str, str, float]:
    """Read the golden ``(x, y, t)`` transition shipped with the package."""
    path = os.path.join(repo_root, "protevo", "tests", "example_transition.txt")
    with open(path) as fh:
        x, y, t = fh.readline().strip().split(" ")
    return x, y, float(t)


def make_corpus(
    base_seq: str,
    n: int,
    seed: int = 0,
    mutation_rate: float = 0.15,
    length_jitter: float = 0.0,
) -> List[Tuple[str, str]]:
    """Build ``n`` deterministic variants of ``base_seq``.

    Args:
        base_seq: Sequence to mutate.
        n: Number of variants to produce.
        seed: RNG seed. Same seed always yields the same corpus, which is what
            makes baseline-vs-optimized parity comparisons meaningful.
        mutation_rate: Per-residue substitution probability.
        length_jitter: If > 0, each variant is truncated to a random fraction in
            ``[1 - length_jitter, 1.0]`` of the base length, producing the ragged
            batches that expose padding waste.

    Returns:
        List of ``(sequence_id, sequence)`` pairs.
    """
    rng = random.Random(seed)
    corpus = []
    for i in range(n):
        seq = list(base_seq)
        for j in range(len(seq)):
            if rng.random() < mutation_rate:
                seq[j] = rng.choice(AMINO_ACIDS)
        if length_jitter > 0.0:
            keep = int(len(seq) * rng.uniform(1.0 - length_jitter, 1.0))
            seq = seq[:max(keep, 16)]
        corpus.append((f"seq{i:05d}", "".join(seq)))
    return corpus


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

def build_model(checkpoint: str, model_type: str, device, use_flash: bool):
    """Load a PEINT model of the requested variant.

    ``model_type`` is one of ``generator`` / ``evaluator`` / ``standard``, matching
    ``protevo.models.load_peint_model``.
    """
    from protevo.models import load_peint_model

    model, vocab = load_peint_model(
        checkpoint_path=checkpoint,
        device=device,
        model_type=model_type,
        use_flash=use_flash,
    )
    return model.eval(), vocab


# --------------------------------------------------------------------------
# Workload 1: autoregressive generation
# --------------------------------------------------------------------------

def generation_callable(model, vocab, x_seq: str, t: float, batch_size: int, device,
                        max_decode_steps: Optional[int] = None, p: float = 1.0,
                        temperature: float = 1.0, autocast: bool = True):
    """Return a zero-argument callable that generates one full batch.

    ``max_decode_steps`` defaults to ``2 * len(x_seq)``, matching what
    ``protevo.simulation._simulate_on_tree`` uses in production.

    ``PeintGenerator.generate`` carries ``@torch.no_grad()`` but no autocast of
    its own, and the Flash kernels reject fp32 outright, so the *caller* is
    responsible for precision. Production does this at
    ``simulation/_simulate_on_tree.py`` (``torch.autocast(bfloat16)`` around
    ``generate``); we replicate it here so the benchmark measures the real path.
    """
    import contextlib

    import torch

    steps = max_decode_steps if max_decode_steps is not None else 2 * len(x_seq)
    x_tokens = [vocab.cls_idx] + vocab.encode(x_seq) + [vocab.eos_idx]
    x_toks = torch.tensor(x_tokens).unsqueeze(0).repeat(batch_size, 1).to(device)
    ts = torch.tensor([t], dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(device)

    def _ctx():
        if autocast and device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _run():
        with _ctx():
            return model.generate(
                x=x_toks, t=ts, max_decode_steps=steps,
                device=device, temperature=temperature, p=p,
            )

    return _run, {"decode_steps": steps, "src_len": len(x_seq), "autocast_bf16": autocast}


# --------------------------------------------------------------------------
# Workload 2: likelihood / VEP scoring
# --------------------------------------------------------------------------

def likelihood_callable(model, x_seq: str, targets: Sequence[str], t: float,
                        batch_size: int, device):
    """Return a zero-argument callable scoring ``targets`` against ``x_seq``."""
    times = [t] * len(targets)
    seqs = list(targets)

    def _run():
        return model.evaluate_likelihood(
            x=x_seq, y=seqs, t=times, device=device, batch_size=batch_size
        )

    return _run, {"n_targets": len(seqs), "batch_size": batch_size}


# --------------------------------------------------------------------------
# Workload 3: homology detection (all-vs-all)
# --------------------------------------------------------------------------

def homology_callable(checkpoint: str, corpus: Sequence[Tuple[str, str]],
                      device: str, batch_size: int, time: float = 0.5,
                      num_gpus: int = 1):
    """Return a zero-argument callable running all-vs-all over ``corpus``.

    With ``num_gpus == 1`` this drives ``PeintHomologySearcher.all_vs_all``
    directly (the searcher owns model loading and the per-reference encoder
    cache). With more, it drives the sharded runner, which loads the checkpoint
    once per rank instead. Both are available on the baseline tree only in the
    former form, so the sharded path is benchmarked against the optimized tree.
    """
    seqs = list(corpus)
    info = {"n_sequences": len(seqs), "batch_size": batch_size, "num_gpus": num_gpus}

    if num_gpus > 1:
        from protevo.inference._runners import all_vs_all_sharded

        def _run():
            return all_vs_all_sharded(
                checkpoint=checkpoint, sequences=seqs, time=time,
                batch_size=batch_size, num_gpus=num_gpus,
            )

        return _run, info

    from protevo.homology_detection._peint import PeintHomologySearcher, PeintSearchConfig

    config = PeintSearchConfig(
        checkpoint_path=checkpoint,
        device=device,
        batch_size=batch_size,
        time=time,
    )
    searcher = PeintHomologySearcher(config)

    def _run():
        return searcher.all_vs_all(seqs)

    return _run, info


def teacher_forced_logits(model, vocab, x_seq: str, y_seq: str, t: float, device):
    """Deterministic ``y_logits`` for one transition — the parity gate's payload.

    Mirrors ``tests/conftest.prepare_model_input`` so the numbers are directly
    comparable to the golden ``protevo/tests/y_logits.npy``.

    Goes through ``evaluate_transition_logits`` rather than calling ``model(...)``
    directly: that method owns the per-variant precision policy (bf16 autocast for
    the Flash variants, fp32 for Vanilla), and the Flash kernels raise
    "FlashAttention only support fp16 and bf16 data type" on a bare fp32 call.
    """
    import torch

    x_tokens = [vocab.cls_idx] + vocab.encode(x_seq) + [vocab.eos_idx]
    y_tokens = [vocab.cls_idx] + vocab.encode(y_seq)

    x_toks = torch.tensor(x_tokens).unsqueeze(0).to(device)
    y_toks = torch.tensor(y_tokens).unsqueeze(0).to(device)
    ts = torch.tensor([t], dtype=torch.float32).unsqueeze(0).to(device)
    x_mask = x_toks.eq(vocab.padding_idx)
    y_mask = y_toks.eq(vocab.padding_idx)

    logits = model.evaluate_transition_logits(x_toks, y_toks, ts, x_mask, y_mask)
    return logits.float().cpu().numpy()


def workload_metadata(name: str, extra: Dict[str, object]) -> Dict[str, object]:
    """Uniform row prefix so every report table has the same leading columns."""
    row: Dict[str, object] = {"workload": name}
    row.update(extra)
    return row


# --------------------------------------------------------------------------
# Workload 4: bulk generation over many (x, t) pairs
# --------------------------------------------------------------------------

def bulk_generation_callable(checkpoint: str, sources: Sequence[str],
                             times: Sequence[float], batch_size: int,
                             num_gpus: int = 1, pack_by_length: bool = False,
                             use_flash: bool = True, p: float = 1.0):
    """Return a callable generating one sequence per ``(source, time)`` pair.

    This is the shape that matters for bulk work and that the single-source
    ``generation_callable`` cannot expose: real corpora have *heterogeneous source
    lengths*, and generation cost is driven by ``2 * max(source length in batch)``
    sequential decode steps with no early exit until every row finishes. Grouping
    similar lengths is therefore attacking sequential work, not padded FLOPs.
    """
    from protevo.inference._batching import (
        decode_step_waste,
        fixed_size_batches,
        length_sorted_batches,
    )
    from protevo.inference._runners import generate_sharded

    srcs, ts = list(sources), list(times)
    lengths = [len(s) for s in srcs]
    groups = (length_sorted_batches(lengths, batch_size) if pack_by_length
              else fixed_size_batches(len(srcs), batch_size))
    waste = decode_step_waste(lengths, groups)

    def _run():
        return generate_sharded(
            checkpoint=checkpoint, sources=srcs, times=ts,
            batch_size=batch_size, num_gpus=num_gpus, p=p,
            use_flash=use_flash, pack_by_length=pack_by_length,
        )

    info = {
        "n_sequences": len(srcs),
        "batch_size": batch_size,
        "num_gpus": num_gpus,
        "pack_by_length": pack_by_length,
        "min_len": min(lengths),
        "max_len": max(lengths),
        "decode_steps_paid": waste["steps_paid"],
        "decode_steps_needed": waste["steps_needed"],
        "decode_step_waste": waste["waste_fraction"],
    }
    return _run, info
