# PEINT inference speedup

This worktree (`peint-fast`, branch `inference-speedup`, branched from
`referee3-ablations @ 4d7354a`) makes the released `peint.ckpt` faster to *run*.
It does not retrain anything and does not change what the model computes.

Two kinds of change live here:

* **Tier 1 — provably identical.** Loop-invariant work hoisted out of the decode
  loop, device syncs removed, caches allocated in the compute dtype, tokenization
  vectorized. Same kernels, same inputs, same reduction order. Gated by
  `benchmarks/inference/parity.py`, which requires `max_abs_diff == 0.0` against
  the pristine `../peint` tree.
* **Multi-GPU sharding.** Data-parallel fan-out over the work list. No DDP, no
  NCCL — these workloads need no gradient sync. Results are merged in the
  original item order and each work item is seeded from its own index, so output
  does not depend on how many GPUs ran it.

## Running things

Everything goes through SLURM; the checkpoint is 219 MB and the model is 150M
parameters, so none of it belongs on a login node.

```bash
# Before/after on one workload
sbatch --job-name=gen_base --export=ALL,REPO=../peint,WORKLOAD=generate,\
OUT=results/base_generate,"BATCH_SIZES=1 8 32 64" benchmarks/inference/bench.sbatch
sbatch --job-name=gen_fast --export=ALL,REPO=.,WORKLOAD=generate,\
OUT=results/fast_generate,"BATCH_SIZES=1 8 32 64" benchmarks/inference/bench.sbatch

# Bit-exactness gate (run after every change)
sbatch benchmarks/inference/parity.sbatch

# Multi-GPU: identical results + scaling
sbatch benchmarks/inference/shard_check.sbatch

# The package's own tests, inside this worktree
sbatch benchmarks/inference/tests.sbatch
```

`--repo` is what selects which copy of `protevo` is measured, so the baseline and
the optimized tree run byte-identical instrumentation.

Note on `sbatch --export`: it splits its own argument on commas, so a
comma-separated value silently truncates to its first element. The job scripts
take space-separated lists (`"BATCH_SIZES=1 8 32 64"`) and convert them.

## Multi-GPU API

```python
from protevo.inference._runners import all_vs_all_sharded

hits = all_vs_all_sharded(
    checkpoint="model_checkpoints/peint.ckpt",
    sequences=[(seq_id, seq), ...],
    time=0.5, batch_size=32, num_gpus=8,
)
```

Also exposed on the CLI:

```bash
python -m protevo.homology_detection all-vs-all \
  --method peint --checkpoint model_checkpoints/peint.ckpt \
  --proteome-dir <dir> --distance-matrix <times.csv> \
  --num-gpus 8 --output results.csv     # --num-gpus 0 = every visible GPU
```

`generate_sharded` and `score_targets_sharded` (VEP / bulk likelihood) live
alongside it in `protevo/inference/_runners.py`.

Each rank loads the checkpoint itself. That is the whole point: the two places
the package already tried to parallelize
(`simulation/_simulate_on_tree.py`, `time_mle/t_mle.py`) hand a live CUDA
`nn.Module` to `multiprocessing.Pool`, which cannot work under fork or spawn — so
every inference workload had in practice been single-GPU.

### Sharding has a fixed cost — check your work list is big enough

Each `run_sharded` call spawns its ranks and each rank builds the model from
scratch (~15 s of ESM2-150M construction per GPU), paid once per *call*. On 4×
A5000 a 120-sequence all-vs-all took **27.7 s on one GPU and 37.8 s on four** —
28 s of compute cannot cover four model loads. Above roughly a minute of
single-GPU work it pays off and scaling approaches linear. Call it once with the
whole work list, never inside a loop.

Peak-memory numbers are reported as `nan` for sharded runs: the allocations
happen inside the spawned ranks, where the parent's allocator cannot see them.
The per-rank figure is the single-GPU number for that rank's shard.

## Measured

One RTX A5000, release `peint.ckpt`, ESM2-150M backbone. Full table:
`results/efficiency.md` (regenerate with `efficiency_table.py`).

| workload | setting | baseline | optimized | speedup | peak mem |
|---|---|---|---|---|---|
| generation | batch 1, 568 decode steps | 3290 ms | 2365 ms | **1.39×** | 1048 → 1012 MiB |
| generation | batch 8 | 3369 ms | 2453 ms | **1.37×** | 1392 → 1156 MiB |
| generation | batch 32 | 3587 ms | 2485 ms | **1.44×** | 2866 → 1848 MiB |
| generation | batch 64 | 3803 ms | 2652 ms | **1.43×** | 5516 → 2828 MiB (1.95×) |
| likelihood / VEP | 2048 targets, batch 32 | 5444 ms | 1840 ms | **2.96×** | 2144 → 1744 MiB |
| likelihood / VEP | 2048 targets, batch 128 | 5435 ms | 1931 ms | **2.81×** | 6066 → 4652 MiB |
| homology all-vs-all | N=200 (39 800 pairs) | 129.5 s | 60.2 s | **2.15×** | 2144 → 1744 MiB |

Generation throughput at batch 64 goes from 9 558 to 13 707 tokens/s while using
roughly half the memory — which is itself a throughput lever, since it leaves
room for a larger batch.

The likelihood and homology gains are mostly tokenization: 2048 sequences at
fair-esm's ~2 ms/sequence is ~4 s of pure Python before any GPU work, which is
almost exactly the 3.6 s that disappeared.

Correctness for every row above: `parity.py --tier 1` reports
`max_abs_diff == 0.0` on logits, likelihood, generation and homology (the last
also confirms identical ranking).

### Multi-GPU scaling

All-vs-all on the optimized tree, 4× A5000, N=400 (159 600 pairs):

| GPUs | wall | throughput | speedup |
|---|---|---|---|
| 1 | 184.6 s | 864 pairs/s | — |
| 4 | 72.9 s | 2 190 pairs/s | **2.53×** |

The shortfall from 4× is startup, not sharding: ideal compute is 184.6/4 ≈ 46 s,
plus roughly 25 s of per-rank model construction, which is 71 s — essentially the
72.9 s measured. The compute itself scales linearly; the fixed cost is what you
are paying for, and it shrinks as a fraction of a longer run.

At N=120 the same comparison is a *slowdown* (27.7 s → 37.8 s). See the fixed-cost
note below before reaching for `--num-gpus`.

Hit lists were identical between the 1-GPU and 4-GPU runs across 14 280 hits.

Note the two single-GPU homology rows are not directly comparable to each other:
throughput rises with N (864 pairs/s at N=400 vs 661 at N=200) because the
per-reference encoder pass amortizes over more queries. A baseline run at N=400
was not made, so the combined Tier-1 + 4-GPU figure against the original code is
an extrapolation, not a measurement.

## What was slow, and why

Measured on the release checkpoint; see the commit messages for the full list.

* **Cross-attention re-unpadded the encoder cache on every generated token.** The
  encoder memory and its padding mask are fixed for a whole `generate()` call,
  yet `unpad_input` re-gathered the entire `[B, L_enc, 2, H, hd]` cache per layer
  per token, each call paying a `.max().item()` device sync. Prefill now caches
  the unpadded K/V and its varlen metadata.
* **KV caches were float32 while compute runs in bf16.** Every read materialized
  a converted copy of the whole cache, per layer, per token.
* **RoPE tables were rebuilt every token in every layer**, because
  `KVCached_MHSA` called `rot_emb` without `max_seqlen` and the offset grew by
  one each step.
* **`decode_sequences` did `B × L` individual `.item()` calls** on a CUDA tensor —
  roughly 38k device syncs after a batch-64, length-600 generate.
* **Tokenization was O(N²) in homology search.** `evaluate_likelihood` called
  fair-esm's ~2 ms/sequence `Alphabet.encode` on the *entire* query set once per
  reference. The dataset loader had already solved this with a char→id lookup
  table; the inference path never got it. It does now, via
  `protevo/inference/_tokenize.py`, shared by both.
* **The ESM2 backbone's LM head ran on every encoder call and was discarded.**
  PEINT reads only `result['representations']` and keeps its own copy of the head.

## A bug the parity gate caught

Worth recording, because it is the kind of thing a speed change hides.

`EncoderCachedFlashMHCA` stores *unrotated* K/V and rotates them on read — and
flash-attn's rotary embedding rotates **in place**. The original float32 cache was
silently providing the required private copy, as a side effect of the
`.to(q.dtype)` conversion to bf16. Allocating the cache in the compute dtype made
that conversion a no-op returning the cache itself, so every likelihood batch
re-rotated the cache and scores drifted further with each batch (mean absolute
error 0.72 by the end of a 256-sequence run). The copy is now explicit.

Nothing about the output looked malformed — the NLLs were finite, ordered, and
plausible. Only a bit-exactness check against the unmodified tree surfaced it.
