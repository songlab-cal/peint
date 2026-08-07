# PEINT inference optimization — implementation report

Branch `inference-speedup`, worktree `rebuttal/peint-fast/`, branched from
`referee3-ablations @ 4d7354a`. Nine commits. No retraining; the released
`peint.ckpt` is unchanged and every default-path number it produces is unchanged.

---

## 0. Where this stands

Single GPU, sequence generation (284-residue sources, bf16, each card at its own
best batch):

| GPU | batch | seq/s | tok/s | M seq / GPU-hour | M seq / GPU-day |
|---|---|---|---|---|---|
| **H200** | 1024 | **276** | 156 906 | 0.99 | 23.9 |
| A100-40GB | 768 | 151 | 85 831 | 0.54 | 13.1 |
| A100-80GB | 1024 | 128 | 72 815 | 0.46 | 11.1 |

Against the released code, same card, each at its own best batch: **3.3x** on A100
(45.8 -> 151.1 seq/s). Roughly two thirds of that comes from being able to run a
larger batch at all — the optimizations halved activation memory and the baseline
OOMs first.

Other workloads on H200: likelihood/VEP **6.95x** (526 ms vs 3656 ms for 2048
targets), homology all-vs-all **2.98x** (26.3 s vs 78.5 s at N=200).

Everything above is bit-exact against the released checkpoint, verified
independently on A5000, A100 and H200.

**Fleet.** Staging the conda env to node-local NVMe (see
`benchmarks/staging/README.md`) made the two NFSv4.2 nodes usable, taking
`jsteinhardt` from 2 to **4 H200 nodes = 32 GPUs**, or roughly 8 800 seq/s and
**~760 M sequences/day** — if multi-node fan-out scales. Tested to 4 GPUs on one
node; beyond that is unmeasured.

**Best batch is 1024, and run-to-run variance is ~20%.** An earlier cross-run
comparison put the peak at 3072; a same-node, same-run sweep shows 1024 (276.2 seq/s,
31.4 GB) matching 3072 (274.5 seq/s, 113.9 GB). The earlier ranking was noise. Use
1024 — identical throughput, 3.6x less memory, and the remainder becomes headroom for
longer sequences. Never compare batch sizes across runs or nodes here.

**Remaining single-GPU headroom looks like <=2x.** The decode loop is
KV-cache-bandwidth-bound: at batch 3072 a step moves 22.5 GB, of which 22.4 GB is
K/V and 0.05 GB is weights, sustaining **27% of the bandwidth this card
actually delivers** (4275 GB/s measured by on-card copy, 89% of the 4.8 TB/s spec).
So it is not a hardware wall — but cross-attention must re-read every encoder key on
every step and self-attention the whole prefix, at ~1 FLOP per byte. That access
pattern will not saturate HBM no matter how it is written, so the practical ceiling
is ~1.5-2x, via moving fewer bytes (fp8 K/V, not bit-exact) rather than kernel
micro-tuning.

---

## 1. Headline results

Release `peint.ckpt` (ESM2-150M frozen backbone + PEINT layers), measured on
three GPUs. Speedups are optimized vs baseline **on the same card**.

| workload | setting | A5000 | A100 | H200 |
|---|---|---|---|---|
| generation | batch 1 | 1.39× | 1.36× | 1.46× |
| generation | batch 8 | 1.37× | 1.31× | 1.44× |
| generation | batch 32 | 1.44× | 1.37× | 1.48× |
| generation | batch 64 | 1.43× | 1.37× | **1.66×** |
| likelihood / VEP | 2048 targets, batch 32 | 2.96× | 4.18× | 4.03× |
| likelihood / VEP | 2048 targets, batch 128 | 2.81× | 4.53× | **6.95×** |
| homology all-vs-all | N=200, 39 800 pairs | 2.15× | 2.73× | **2.98×** |

Absolute numbers per card:

| workload | setting | GPU | before | after | peak memory |
|---|---|---|---|---|---|
| generation | batch 64 | A5000 | 3803.4 ms | 2652.2 ms | 5516 → 2828 MiB (**1.95×**) |
| generation | batch 64 | A100 | 3670.6 ms | 2684.0 ms | 5516 → 2828 MiB |
| generation | batch 64 | H200 | 3226.7 ms | 1947.7 ms | 5540 → 2852 MiB |
| likelihood | 2048 tgt, bs 128 | A5000 | 5434.6 ms | 1931.1 ms | 6066 → 4652 MiB |
| likelihood | 2048 tgt, bs 128 | A100 | 4988.9 ms | 1101.6 ms | 6066 → 4652 MiB |
| likelihood | 2048 tgt, bs 128 | H200 | 3655.6 ms | 525.8 ms | 6090 → 4676 MiB |
| homology | N=200 | A5000 | 129.5 s | 60.2 s | 2144 → 1744 MiB |
| homology | N=200 | A100 | 111.0 s | 40.7 s | 2144 → 1744 MiB |
| homology | N=200 | H200 | 78.5 s | 26.3 s | 2168 → 1768 MiB |

Generation throughput at batch 64: **9 558 → 13 707 tok/s** (A5000),
**9 904 → 13 544** (A100), **11 266 → 18 664** (H200) — on roughly half the memory.

### The speedup grows with GPU speed

Likelihood at batch 128, same workload on all three cards:

| GPU | baseline | optimized | speedup |
|---|---|---|---|
| A5000 | 5434.6 ms | 1931.1 ms | 2.81× |
| A100 | 4988.9 ms | 1101.6 ms | 4.53× |
| H200 | 3655.6 ms | 525.8 ms | **6.95×** |

The baseline barely moves across hardware (5435 → 3656 ms, only 1.49× from A5000
to H200) while the optimized path scales 3.67×. That is direct confirmation of the
diagnosis in §3: the baseline was **CPU-bound on Python tokenization**, so a faster
GPU could not help it. Removing that bottleneck is what lets the hardware matter —
and it means the win is largest on exactly the cards you would deploy on.

Multi-GPU, all-vs-all at N=400 (159 600 pairs), 4× A5000:

| GPUs | wall | throughput | speedup |
|---|---|---|---|
| 1 | 184.6 s | 864 pairs/s | — |
| 4 | 72.9 s | 2 190 pairs/s | **2.53×** |

**Correctness.** Every number above comes from a path that is bit-exact against
the untouched `../peint` tree: `max_abs_diff == 0.0` on teacher-forced logits,
likelihood NLLs, argmax generation, and homology hit lists (ranking identical).
The 4-GPU hit list is identical to the 1-GPU one across 14 280 hits.

---

## 2. Reproducing any of it

Everything runs under SLURM — the checkpoint is 219 MB and the model is 150M
parameters, so none of it belongs on a login node.

```bash
cd /scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast

# Before/after on one workload. --repo selects which protevo is measured, so
# baseline and optimized run byte-identical instrumentation.
sbatch --job-name=gen_base --export=ALL,REPO=../peint,WORKLOAD=generate,\
OUT=results/base_generate,"BATCH_SIZES=1 8 32 64" benchmarks/inference/bench.sbatch
sbatch --job-name=gen_fast --export=ALL,REPO=.,WORKLOAD=generate,\
OUT=results/fast_generate,"BATCH_SIZES=1 8 32 64" benchmarks/inference/bench.sbatch

sbatch benchmarks/inference/parity.sbatch        # bit-exactness gate
sbatch benchmarks/inference/shard_check.sbatch   # multi-GPU: identical + scaling
sbatch benchmarks/inference/tests.sbatch         # package test suite

python benchmarks/inference/efficiency_table.py --results-dir results \
       --out results/efficiency                  # aggregate into one table
```

Gotcha that cost two reruns: `sbatch --export` splits its own argument on
commas, so `BATCH_SIZES=1,8,32,64` silently truncates to `1`. The job scripts
take space-separated lists and convert them.

---

## 3. Commit-by-commit walkthrough

### `4cccba7` — Inference benchmark + Tier-1 parity harness
*6 files, +957*

Built the measurement apparatus before touching any model code, because the
whole branch rests on being able to prove nothing changed.

- `benchmarks/inference/harness.py` — CUDA-event timing (wall-clock around an
  async kernel launch measures the launch, not the work — the one pre-existing
  timer in the package, in `simulation/_simulate_on_tree.py`, makes exactly that
  mistake), peak-memory capture, and a parameter count split into frozen backbone
  vs shipped PEINT layers.
- `benchmarks/inference/workloads.py` — the three workloads, plus a seeded
  synthetic corpus derived from the repo's own golden fixture
  (`protevo/tests/example_transition.txt`), so benchmarking never touches the
  shared read-only caches.
- `benchmarks/inference/run_workload.py` — one script, `--repo` picks which
  `protevo` gets imported. This is what makes before/after honest.
- `benchmarks/inference/parity.py` — runs `--mode dump` against two trees and
  diffs the arrays.

**Validation:** ran the gate with both sides pointing at identical trees; all
three workloads reported exact. Also surfaced a real fact — a bare `model(...)`
call in fp32 raises *"FlashAttention only support fp16 and bf16 data type"*, so
the harness goes through `evaluate_transition_logits`, which owns the per-variant
precision policy.

### `0ce00ff` — Tier-1: remove per-token overhead from the decode loop
*8 files, +890 / −67*

The substantive generation work. Everything here is provably identical.

In `protevo/models/_transformer_modules.py`:

| change | why |
|---|---|
| `KVCached_MHCA` caches the **unpadded** encoder K/V + `cu_seqlens`/`max_seqlen` at prefill | the encoder memory and its mask are fixed for a whole `generate()` call, yet `unpad_input` was re-gathering the entire `[B, L_enc, 2, H, hd]` cache per layer per token, each call paying a `.max().item()` device sync |
| single-token decode queries build their unpad metadata directly | the query is a freshly sampled token, never `<pad>` (sampling masks every non-amino-acid logit to `-inf`), so `unpad_input` there is an identity gather — one more sync per layer per token removed |
| KV caches allocate in the **compute dtype** | they were fp32 while decoding runs under bf16 autocast, so every read materialized a converted copy of the whole cache. bf16 → fp32 → bf16 is lossless, so values are unchanged |
| `KVCached_MHSA` passes `max_seqlen` to `rot_emb` | without it, flash-attn's `_update_cos_sin_cache` saw `seqlen + offset` grow by one each step and rebuilt the whole table every token in every layer |
| `GeometricTimeEmbedder` registers its frequency grid as a buffer | it ran `np.geomspace` on the CPU and copied H2D on *every* forward — i.e. every generated token |
| `reset_kv_cache` stops `zero_()`-ing the buffer | reads are bounded by `cache_size` and every position is written before read; the memset was redundant |

In `protevo/models/_transformer.py`: `zero_idx` kept on device (it was CPU, so
`logits[..., zero_idx] = -inf` copied H2D every token); `eos_reached.all()` checked
every `EOS_CHECK_INTERVAL = 16` steps instead of every step; the final decoder
pass whose logits were discarded is gone; `decode_sequences` does one
`.cpu().tolist()` instead of `B × L` individual `.item()` calls (~38k syncs after
a batch-64, length-600 generate).

In `protevo/models/_flash_esm.py`: the backbone's LM head is skipped. PEINT reads
only `result['representations']` and keeps its own copy of the head, so a
`[B, L, 640]` LayerNorm plus a 640→640 dense, gelu and tied 640→33 projection ran
on every encoder call and was thrown away.

> **Commit-boundary note.** This commit also carries the initial files of
> `protevo/inference/` (`_shard.py`, `_runners.py`, `_tokenize.py`). They were in
> the working tree but not yet imported by anything — dead code at this commit,
> wired up in `86c43a9`.

**Result:** generation 1.37–1.44× faster across the batch sweep, memory at batch
64 down 1.95×.

**A bug this commit's parity run caught — worth reading.** Allocating the caches
in bf16 broke `EncoderCachedFlashMHCA`, which stores *unrotated* K/V and rotates
on read. Flash-attn's rotary works **in place**, and the fp32 cache had been
silently supplying the required private copy as a side effect of the dtype cast.
With a matching dtype the cast returns the cache itself, so every likelihood batch
re-rotated the cache and scores drifted further with each batch — mean absolute
error 0.72 by the end of a 256-sequence run. Nothing looked malformed: the NLLs
were finite, ordered and plausible. Only bit-exactness against the unmodified tree
surfaced it. The copy is now explicit and commented.

### `86c43a9` — Tier-1 tokenization + multi-GPU sharded inference
*15 files, +724 / −72*

**Tokenization.** This was the real bottleneck for likelihood and homology, and it
was on the CPU. `evaluate_likelihood` called fair-esm's ~2 ms/sequence
`Alphabet.encode` on the entire query set *once per reference*, so an N-sequence
all-vs-all paid O(N²) pure-Python tokenization before touching a GPU. The dataset
loader had already solved this with a 256-entry char→id table;
`protevo/inference/_tokenize.py` lifts it out so both share one implementation.
`evaluate_likelihood` also gained pre-tokenized input and uploads the source once
instead of materializing B host-side copies per batch. The homology searchers
tokenize their corpus once, and `all_vs_all_proteomes` hoists the query-set
construction (which depends only on the reference's *proteome*) out of the
per-reference loop.

At 2048 targets that is ~4 s of Python — almost exactly the 3.6 s that
disappeared. Likelihood 2.96×, homology 2.15×.

**Multi-GPU.** `protevo/inference/_shard.py` fans work out over a node's GPUs with
plain process spawn — no DDP, no NCCL, since none of these workloads needs
gradient sync. Each rank loads the checkpoint itself, which is precisely what the
package's two existing attempts could not do: `simulation/_simulate_on_tree.py`
and `time_mle/t_mle.py` both hand a live CUDA `nn.Module` to
`multiprocessing.Pool`, which works under neither fork nor spawn — so in practice
every inference workload had been single-GPU.

`_runners.py` adds `generate_sharded`, `score_targets_sharded`,
`all_vs_all_sharded`, `all_vs_all_proteomes_sharded`; the homology CLI gains
`--num-gpus` (`0` = every visible GPU).

Output is independent of GPU count **by construction**: results merge in original
item order, each item is seeded from its own index rather than its rank, and
generation shards over *pre-chunked batches* so batch composition — which fixes
both the RNG draw order and the padded shape — cannot shift.

Also: `HomologySearcher.write_results` became a `staticmethod` (it never used
`self`), letting the sharded CLI path serialize results without constructing a
searcher and occupying GPU 0 for nothing.

Adds `tests/test_inference_utils.py` — the LUT reproduces `Alphabet.encode`
residue for residue, `encode_batch` matches the original tensors, shard assignment
is a balanced partition.

### `ac51831` — Document sharding's fixed cost; efficiency table; benchmark fixes
*5 files, +215 / −4*

Sharding is correct but **not free**, and the failure mode looks like "multi-GPU
made it slower". Each `run_sharded` call spawns ranks that each build ESM2-150M
(~15 s per GPU), paid once per *call*. Measured: a 120-sequence all-vs-all takes
**27.7 s on one GPU and 37.8 s on four** — 28 s of compute cannot cover four model
loads. Documented in `_shard.py` and the README with the measured crossover
(roughly a minute of single-GPU work).

Benchmark fixes: sharded runs report peak memory as `nan` rather than `0`, since
the allocations happen in the spawned ranks where the parent's allocator cannot
see them; scaling now uses a work list long enough to amortize startup, separate
from the smaller correctness check; `TQDM_DISABLE` in the job scripts.

`efficiency_table.py` aggregates the JSON reports into the params / memory /
throughput table the referee asked for.

### `9d57343` — Efficiency table: key rows by problem size and commit
*1 file, +43 / −10*

Three ways the aggregator produced misleading numbers, all fixed:

- it grouped homology runs of different N together, dividing an N=120 optimized
  run by an N=200 baseline and reporting a **4.67× speedup that never happened**;
- it globbed its own output back in (that report also has a `rows` key), adding
  `nan` rows on every re-run;
- it could not distinguish two runs of the same workload from different points on
  the branch. Rows now carry the git commit, and reports that measured an
  uncommitted intermediate state move to `results/superseded/`, which the
  aggregator skips.

The superseded run here is the N=200 homology measurement taken between the
decode-loop and tokenization commits; it read 1.00× because the tokenizer work
was not in the tree yet.

### `cdc00dc`, `80b7dd5` — README results
*+52 lines of documentation*

Single-GPU table, then the multi-GPU scaling numbers with the honest caveat that
the 4-GPU shortfall is startup rather than sharding: ideal compute is 184.6/4 ≈
46 s plus ~25 s of per-rank model construction = 71 s, against 72.9 s measured. The
compute scales linearly.

### `16f7c9b` — Tier-2: opt-in length-bucketed batching
*10 files, +375 / −16*

`--pack-by-length` / `--max-tokens` on the CLI, `pack_by_length=` / `max_tokens=`
on `evaluate_likelihood`, `PeintSearchConfig` and the sharded runners. **Off by
default.** Sorts by length and fills batches to a padded-position budget rather
than a row count; results still return in corpus order (the loop now iterates over
index lists and scatters into a preallocated array).

`protevo/inference/_batching.py` emits batches **widest-first**, and that is
load-bearing rather than cosmetic: `EncoderCachedFlashMHCA` sizes its K/V cache
from the first batch and reallocates — dropping `cache_size`, and with it the
cached encoder — if a later batch has more rows. With fixed-size batches only the
final batch can be smaller, so this never arose; with variable widths it would
have silently corrupted every batch after the widest.

The first version capped rows at `batch_size`, which made bucketing tidy each
32-row batch without reducing the batch count, and measured as exactly zero gain
(1752 → 1756 ms; 53.1 → 53.6 s). `max_tokens` alone now governs width.

`parity.py` gained `--baseline-extra` / `--fast-extra` so a flag can be measured
on-vs-off within one tree, and `--length-jitter` (at jitter 0 every sequence is
the same length and bucketing is a no-op).

### `f9fa9ee` — Measure Tier-2; correct two claims it disproved
*4 files, +99 / −17*

See §5.

---

## 4. All measurements in one place

### Generation — `PeintGenerator.generate`, source length 284, 568 decode steps

| GPU | batch | baseline ms | optimized ms | speedup | baseline tok/s | optimized tok/s | baseline MiB | optimized MiB |
|---|---|---|---|---|---|---|---|---|
| A5000 | 1 | 3290.5 | 2365.4 | 1.39× | 173 | 240 | 1048 | 1012 |
| A5000 | 8 | 3369.2 | 2452.8 | 1.37× | 1349 | 1853 | 1392 | 1156 |
| A5000 | 32 | 3587.1 | 2484.6 | 1.44× | 5067 | 7316 | 2866 | 1848 |
| A5000 | 64 | 3803.4 | 2652.2 | 1.43× | 9558 | 13707 | 5516 | 2828 |
| A100 | 1 | 3203.8 | 2352.0 | 1.36× | 177 | 241 | 1048 | 1012 |
| A100 | 8 | 3326.3 | 2534.2 | 1.31× | 1366 | 1793 | 1392 | 1156 |
| A100 | 32 | 3529.0 | 2576.8 | 1.37× | 5151 | 7054 | 2866 | 1848 |
| A100 | 64 | 3670.6 | 2684.0 | 1.37× | 9904 | 13544 | 5516 | 2828 |
| H200 | 1 | 2459.6 | 1688.5 | 1.46× | 231 | 336 | 1060 | 1024 |
| H200 | 8 | 2576.6 | 1793.8 | 1.44× | 1764 | 2533 | 1416 | 1180 |
| H200 | 32 | 2726.5 | 1836.9 | 1.48× | 6666 | 9895 | 2890 | 1872 |
| H200 | 64 | 3226.7 | 1947.7 | 1.66× | 11266 | 18664 | 5540 | 2852 |

### Likelihood / VEP — `evaluate_likelihood`, 2048 targets

| GPU | batch | baseline ms | optimized ms | speedup | baseline seq/s | optimized seq/s | baseline MiB | optimized MiB |
|---|---|---|---|---|---|---|---|---|
| A5000 | 32 | 5443.7 | 1839.7 | 2.96× | 376 | 1113 | 2144 | 1744 |
| A5000 | 128 | 5434.6 | 1931.1 | 2.81× | 377 | 1061 | 6066 | 4652 |
| A100 | 32 | 5296.0 | 1267.6 | 4.18× | 387 | 1616 | 2144 | 1744 |
| A100 | 128 | 4988.9 | 1101.6 | 4.53× | 411 | 1859 | 6066 | 4652 |
| H200 | 32 | 3500.2 | 867.5 | 4.03× | 585 | 2361 | 2168 | 1768 |
| H200 | 128 | 3655.6 | 525.8 | 6.95× | 560 | 3895 | 6090 | 4676 |

### Homology all-vs-all

| N | pairs | config | wall | pairs/s | speedup |
|---|---|---|---|---|---|
| 200 | 39 800 | A5000 baseline, 1 GPU | 129.5 s | 307 | — |
| 200 | 39 800 | A5000 optimized, 1 GPU | 60.2 s | 661 | 2.15× |
| 200 | 39 800 | A100 baseline, 1 GPU | 111.0 s | 358 | — |
| 200 | 39 800 | A100 optimized, 1 GPU | 40.7 s | 977 | 2.73× |
| 200 | 39 800 | H200 baseline, 1 GPU | 78.5 s | 507 | — |
| 200 | 39 800 | H200 optimized, 1 GPU | 26.3 s | 1 514 | 2.98× |
| 400 | 159 600 | A5000 optimized, 1 GPU | 184.6 s | 864 | — |
| 400 | 159 600 | A5000 optimized, 4 GPU | 72.9 s | 2 190 | 2.53× vs 1 GPU |
| 120 | 14 280 | A5000 optimized, 1 GPU | 27.7 s | 515 | — |
| 120 | 14 280 | A5000 optimized, 4 GPU | 37.8 s | 378 | **0.73× — slower** |

Two things not to misread. Single-GPU throughput rises with N (864 pairs/s at
N=400 vs 661 at N=200) because the per-reference encoder pass amortizes over more
queries, so those rows are not comparable to each other. And no baseline run was
made at N=400, so a combined Tier-1 + multi-GPU figure against the original code
would be an extrapolation, not a measurement.


### Length binning for generation: 1.37x on a ragged corpus

`generate_sharded(...)` — **on by default** since it is a provable no-op on uniform-length input. Measured on A100-40GB, 2048 sources
of length 28-283, batch 64, with model load timed separately:

| | generation only | wall | decode steps executed |
|---|---|---|---|
| unbucketed | 93.8 s | 112.5 s | 663 104 |
| bucketed | **68.4 s** | 86.2 s | **465 856** |
| ratio | **1.37x** | 1.31x | 1.42x |

18.2 -> 23.7 seq/s. Steps executed track generation time closely (1.42x vs 1.37x),
confirming sequential decode steps are the cost driver - a batch runs
`2 x max(source length)` steps and does not exit until every row emits `<eos>`, so
mixing a 28-residue source with a 283-residue one makes the short one pay.

Two earlier numbers for this were both wrong. **1.07x** came from a run whose every
timed call carried a ~15 s checkpoint build (45% of wall at n=512); at n=2048 with
load measured separately that confound is gone. **1.7-1.8x** was an analytic
prediction from `decode_step_waste`, which assumes each batch runs the full
`2 x max(len)` - the early exit on `eos_reached.all()` falsifies that and recovered
44% of the gap the metric assumed. Trust the executed-step counter, not the bound.

### Tier-2 length bucketing (opt-in) — 2048 targets

| corpus | lengths | fixed-32 batches / waste | packed batches / waste | unpacked | packed | speedup |
|---|---|---|---|---|---|---|
| jitter 0.5 | 142–283 | 64 / 23.8% | 28 / 1.2% | 1825.6 ms | 1744.2 ms | 1.05× |
| jitter 0.9 | 28–283 | 64 / 43.3% | 21 / 4.7% | 1767.6 ms | 1722.6 ms | 1.03× |

Homology N=200 at jitter 0.5: 53.4 s both ways — no difference.

### Correctness

| check | result |
|---|---|
| Tier-1 parity, teacher-forced logits | `max_abs_diff = 0.0` (9405 values) |
| Tier-1 parity, likelihood NLLs | `max_abs_diff = 0.0` (256 values) |
| Tier-1 parity, argmax generation | 0 / 8 sequences differ |
| Tier-1 parity, homology hits | `max_abs_diff = 0.0`, ranking identical (992 pairs) |
| Tier-2, packed vs unpacked | `max_abs_diff = 0.0` (512 targets, 1560 pairs) |
| 4-GPU vs 1-GPU hit list | identical, 14 280 hits |
| golden fixture `y_logits.npy` | see below |
| package test suite | 129 passed, 1 failed, 3 skipped |

### Fidelity against the released golden fixture

`protevo/tests/y_logits.npy` predates this branch, so it is the one *absolute*
reference available — everything else compares the optimized tree to the pristine
tree. `tests/test_model_inference.py::test_logits_match_reference` **passes**
(confirmed running, not skipped).

That test alone is a narrow check, though: it exercises only the Vanilla
(non-Flash, fp32) path via `loaded_model(use_flash=False)`, at `np.allclose`
tolerance — and almost all of this branch's work is in the Flash variants.
`benchmarks/inference/golden_check.py` covers both paths and both trees:

| path | pristine vs fixture | optimized vs fixture | optimized vs pristine |
|---|---|---|---|
| Vanilla fp32 | max 2.19e-05, mean 1.41e-06 ✓ within 1e-4 | max 2.19e-05, mean 1.41e-06 ✓ | **0.0** |
| Flash bf16 | max 0.296, mean 0.0131 | max 0.296, mean 0.0131 | **0.0** |

The optimized tree's distance from the fixture is bit-identical to the pristine
tree's, on both paths. The optimization introduced no drift of its own — which is
the assertion `golden_check.py` enforces.

Worth stating plainly, because it is easy to over-read the pytest result: **the
Flash path has never been within the fixture's 1e-4 tolerance.** It sits 0.296 max
/ 0.013 mean away, because the fixture is fp32 and the Flash variants run bf16.
That is a pre-existing property of the released code, not something introduced
here — but "reproduces `y_logits.npy`" is a statement about the Vanilla path only,
and should not be quoted unqualified.

```bash
python benchmarks/inference/golden_check.py             # Flash bf16
python benchmarks/inference/golden_check.py --no-flash  # Vanilla fp32
```

The one failure is `tests/test_a5_lora.py::test_lora_config_differs_from_baseline_only_in_finetune_axes`,
**pre-existing on the base commit**: `f7a7d68` set `lora.yaml` to
`accumulate_grad_batches: 24` (single-GPU, same 768 effective batch) without adding
that field to the test's ignore set. Unrelated to inference; inherited, not caused.

---

### Batch size is the dominant throughput lever, and memory is what caps it

Everything above holds batch size fixed, which understates what the optimizations
buy in production. Sweeping batch to each card's ceiling (source length 284, one
timed pass per point):

**A100-PCIE-40GB**

| batch | baseline seq/s | baseline mem | optimized seq/s | optimized mem |
|---|---|---|---|---|
| 64 | 17.9 | 4.0 GB | 22.8 | 2.5 GB |
| 128 | 28.8 | 9.2 GB | 46.5 | 5.0 GB |
| 256 | 45.0 | 19.5 GB | 75.8 | 9.8 GB |
| 512 | 45.8 | 37.7 GB | 105.3 | 19.6 GB |
| 768 | 45.6 | 38.8 GB | **151.1** | 30.0 GB |
| 1024 | **OOM** | — | 132.2 | 38.3 GB |

The baseline plateaus around 46 seq/s from batch 256 and then OOMs: it runs out of
memory before it runs out of headroom. **At each card's best feasible batch,
45.8 -> 151.1 seq/s = 3.30x** — 2.4x more than the 1.37x measured at fixed batch
64. On a memory-constrained card the 1.99x activation-memory reduction (matched at
batch 256: 19 980 -> 10 062 MiB) converts directly into throughput, because batch
size is the lever and memory is the cap.

**H200** saturates at batch **1024**: 276.2 seq/s in 31.4 GB. Batch 3072 gives 274.5 seq/s for 113.9 GB — the same throughput for 3.6x the memory.

### Throughput per unit compute

| GPU | tree | best batch | seq/s | tok/s | memory | M seq / GPU-hour | M seq / GPU-day |
|---|---|---|---|---|---|---|---|
| A100-40GB | baseline | 512 | 45.8 | 26 037 | 37.7 GB | 0.16 | 4.0 |
| A100-40GB | optimized | 768 | 151.1 | 85 831 | 30.0 GB | 0.54 | 13.1 |
| H200 | optimized | 1024 | 276.2 | 156 906 | 31.4 GB | 0.99 | 23.9 |

Scales roughly linearly with sequence length (cost is ~2 x source length decode
steps), so treat these as figures for ~284-residue proteins.

Two practical notes:

* **Do not set batch to the maximum that fits.** Throughput is non-monotonic near
  capacity — A100 drops 151 -> 132 seq/s from batch 768 to 1024, H200 drops
  274 -> 267 from 3072 to 4096. Allocator pressure. Best batch is around 75-80% of
  what fits.
* **H200 is only 1.81x an A100 here despite roughly 3x the compute**, because the
  A100 is capped at batch 768 by memory while the H200 runs 3072. For bulk
  generation on A100s, memory is the binding constraint and anything that shrinks
  the activation footprint buys throughput close to one-for-one.

### On MFU as a target: it is not one

Measured on H200 with a calibrated ceiling (809.5 TFLOP/s achieved on a large bf16
GEMM, 82% of the 989 spec figure):

| batch | ms/step | seq/s | MFU | HBM util |
|---|---|---|---|---|
| 1 | 3.06 | 0.6 | 0.0% | 3% |
| 64 | 3.49 | 32.3 | 0.6% | 2% |
| 1024 | 7.56 | 238 | 4.1% | 1% |

Low MFU here does **not** mean 25x is available. MFU is not monotone with
throughput: deleting the KV cache would raise it sharply by burning FLOPs on
recomputation while making generation slower. Any metric improvable by doing
useless work cannot be an objective. It is a reasonable proxy for training
(compute-bound, FLOPs fixed by model x data); autoregressive decode is inherently
low arithmetic intensity, so even an optimal decoder scores low.

What the table legitimately shows is a *diagnosis* — but note the HBM column above
is **wrong**: it counted weight traffic only and undercounted by ~20x, which led me
to call the loop "overhead-bound". Counting K/V traffic (see
`decode_bytes_per_step`), a batch-3072 step moves 22.5 GB and sustains ~1.1 TB/s,
so the loop is **memory-bound at large batch** and only overhead-bound at small
batch. The `ms/step`
column says the same thing more directly — 1024x the work for 2.5x the time, so
per-step cost is nearly all fixed overhead. That is why batch size dominates, and
why CUDA graphs would mostly help *latency* rather than throughput.

**The objective is sequences per second at fixed output quality.** Both seq/s and
tok/s are reported: seq/s is the goal, tok/s makes runs at different sequence
lengths comparable.

## 5. Predictions the measurements disproved

Recorded because they are the useful part.

**Length bucketing barely helps this model.** I expected a large win on ragged
corpora. Cutting padding waste from 43% to 5% and the batch count from 64 to 21
gained **3%** on likelihood, and homology was **unchanged**. Flash-attention
already unpads internally, so the attention path costs what the *real* tokens
cost whatever shape the batch is; only the FFN, LayerNorms and LM head see
padding, and at B≈32 / L≈284 / D=640 on an A5000 that is not what the clock is
waiting on. General lesson: do not assume a padding optimization helps a
flash-attention model.

**Bucketing is bit-exact, which I said it would not be.** The plan asserted NLLs
would drift in the last few significant figures because batch composition changes.
Measured, they do not move at all. Same varlen property explains it: a sequence's
result does not depend on its batch-mates, and the remaining GEMMs reduce over the
feature dimension rather than the batch. Recorded as an observation on one GPU and
one set of shapes, not a guarantee — cuBLAS can split reductions differently as
the batch dimension changes — which is why the flag and the check remain.

**Length binning for generation: predicted 1.7-1.8x, first measured 1.07x, actually
1.37x.** Both earlier figures were wrong for different reasons. The 1.07x run gave
every timed call a ~15 s checkpoint build, 45% of wall time at n=512. The 1.7-1.8x
prediction came from `decode_step_waste`, which assumes each batch runs the full
`2 x max(len)`; the early exit on `eos_reached.all()` falsifies that and recovered
44% of the assumed gap. With load timed separately at n=2048, generation goes
93.8 s -> 68.4 s (**1.37x**), and executed decode steps 663 104 -> 465 856 (1.42x),
which track each other closely. Lesson: instrument what actually executed rather
than trusting an analytic bound, and never let a fixed per-call cost sit inside the
timed region.

**A dtype hazard that wasn't one.** Replacing `t[i:i+bs]` with `[t[j] for j in
idxs]` looked like it would change `torch.tensor`'s inferred dtype for numpy
inputs. Checked on a compute node: numpy scalars carry their dtype, so both forms
give float64 for an ndarray and float32 for a list. No fix needed; the reasoning
is now a comment so nobody re-derives it.

---

## 6. Not done

- **Classical-simulator rows for the referee's efficiency table.** Blocked:
  `protevo/simulation/_alisim.py` shells out to an `iqtree2` binary, the submodule
  is uninitialized in this worktree and none is on PATH. Needs
  `git submodule update --init --recursive` plus a cmake build, then timing
  `simulation/classical.py` on the same trees. Note the result is CPU wall-clock
  and must be labelled as such — it is not comparable to a GPU throughput figure
  without stating core count.
- **Training cost** for the same table — recoverable from the ablation logs under
  `/scratch/users/yufan.cao/protevo_ablations/logs/` (768 sequences/step × 60k
  steps at the recorded GPU-hours), not from anything measured here.
- **Continuous batching for generation.** The decode loop still runs until *every*
  row emits `<eos>`, so one long sequence pays for the whole batch. Listed in the
  original plan under Tier-2; given how little length bucketing bought, worth
  measuring the padding waste before building it.
- **`time_mle`.** Its 80-step Adam loop re-runs the frozen backbone 160× per batch
  over inputs that never change — structurally the largest remaining redundancy in
  the package, but out of scope here (it is not one of the three shipping
  workloads).
- **Batch-ceiling and throughput-per-compute measured on A100-40GB and H200
  only.** A5000 was swept to batch 64 only, so its ceiling is unknown.
- **Multi-GPU scaling measured on A5000 only.** The single-GPU numbers now cover
  A5000, A100 and H200, but the sharding study (§4) is A5000. The ~15 s per-rank
  model load is largely CPU-side, so the crossover point should be similar; the
  compute half of the trade shrinks on faster cards, which would push the
  crossover to *larger* work lists, not smaller.
