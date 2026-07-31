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
