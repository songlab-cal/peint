# Computational efficiency of PEINT — referee summary

Answers the referee request for "training cost, inference throughput, GPU memory
usage, generation speed, and parameter counts relative to both classical
simulators and competing neural approaches."

Everything below is **measured** unless a row says otherwise. Sources are given
per section. Three of the five requested axes are fully covered; §6 lists what is
not, and why, because two of the gaps are not cheap to close.

---

## 0. One-paragraph version

PEINT is a 206 M-parameter model of which only **57.4 M are trained** — the
ESM2-150M encoder is frozen, so the released checkpoint is 219 MB and training
needed just 2 GPUs. The published checkpoint cost **≈ 89 A100-GPU-hours**
(≈ 3.7 GPU-days). At inference a single H200 generates **276 evolved sequences/s**
(157 k tokens/s, ≈ 24 M sequences/GPU-day) in **31 GB**, or **2.6 GB** if batch is
kept at 64, so the model runs on any ≥ 8 GB card and on CPU without Flash
Attention. Likelihood/VEP scoring reaches **3 900 sequences/s** and all-vs-all
homology **1 500 pairs/s** on one H200. Against classical substitution simulators
the honest statement is a trade, not a win: WAG/LG carry **208** fixed parameters
and simulate on CPU essentially for free, whereas PEINT spends ~10⁵× the
parameters and a GPU to buy indels, context-dependence and a learned, non-
site-independent process.

---

## 1. Parameter counts

Counted directly from the released weights (`peint.ckpt`, `vep.ckpt`) and from the
cached `esm2_t30_150M_UR50D.pt`.

| Component | Parameters | Trained? | On disk |
|---|---|---|---|
| ESM2-150M encoder backbone (`esm2_t30_150M_UR50D`) | **148.16 M** | frozen | 593 MB, auto-downloaded by `fair-esm` |
| — of which the LM head PEINT skips at inference | 0.43 M | frozen | — |
| PEINT layers, base model (5 enc × 4.92 M + 5 dec × 6.57 M) | **57.45 M** | trained | **219 MB** (`peint.ckpt`) |
| **Total at inference (base)** | **205.61 M** | 28 % trained | |
| PEINT layers, VEP model (2 enc + 2 dec) | 22.98 M | trained | 88 MB (`vep.ckpt`) |
| **Total at inference (VEP)** | 171.14 M | 13 % trained | |

Two numbers matter to a reader, and they are different questions: the model is
206 M parameters, but the artefact we distribute is 57 M, because the backbone is
public and frozen. `peint.ckpt` is exactly 57.45 M × 4 B — fp32 weights only, no
optimizer state.

Weights in memory: ≈ 0.41 GB in bf16, ≈ 0.82 GB in fp32.

**Relative to classical simulators.** WAG and LG are 20-state reversible
substitution matrices: 190 exchangeabilities + 19 free stationary frequencies =
**208 parameters**, estimated once from a fixed database and thereafter constant.
The profile-mixture variants added in revision (LG4X, LG+C60, LG+S256) reach
O(10²–10⁴). Both approaches additionally consume per-family nuisance parameters —
tree topology, branch lengths, site-rate categories — fitted by IQ-TREE; PEINT
needs the same tree to simulate down, so that cost is common to both and not a
differentiator.

---

## 2. Training cost

The released checkpoint's own SLURM record is outside the retained accounting
window, so the figures below come from the **A0 baseline reproduction run**
(job 3365814, `protevo_ablations/baseline/`), which was verified to reproduce the
release — 0.683 vs 0.682 mean per-site likelihood at t ≈ 0.1 — at the same
effective batch. Its `epoch=2-step=40000.ckpt` matches the released checkpoint's
`epoch=2, global_step=40000` exactly.

| | |
|---|---|
| Hardware | 2 × A100-PCIE-40GB, DDP, bf16 |
| Effective batch | 768 sequences/update (32 × accum 12 × 2 GPUs) ≈ 250 k tokens |
| Steady-state rate | 4 000 optimizer steps per 3 h 56 min = **1 017 steps/h** |
| Cost per 1 000 steps | **1.97 GPU-h** (2.22 GPU-h including validation) |
| Training throughput | 781 k sequences/h = **217 seq/s** across 2 GPUs (≈ 108 seq/s/GPU) |
| **Released checkpoint (step 40 000)** | **≈ 89 GPU-h ≈ 3.7 GPU-days**, ≈ 44 h wall on 2 GPUs |
| Full 300 k-step schedule, if run out | ≈ 665 GPU-h ≈ 28 GPU-days |

Freezing the encoder is what makes this small: gradients and optimizer state exist
for 57.4 M parameters, not 206 M, which is why two 40 GB cards suffice for a
768-sequence effective batch.

> **Flag for the response letter.** The released `peint.ckpt` is at
> **global_step = 40 000**, not the 300 000 in the LR schedule
> (`num_training_steps=300000`), and its `lr` is 3e-4 where `train_peint_model.sh`
> passes 4e-4. Quote 40 k steps / ≈ 89 GPU-h, not the schedule horizon.

---

## 3. Generation speed and inference throughput

Workload: 284-residue source, `max_decode_steps = 568`, bf16 autocast,
`PeintGenerator` (Flash Attention + KV cache), single GPU, synthetic uniform-length
corpus. Raw rows in `peint-fast/results/*.{md,csv,json}`; each JSON carries host,
GPU, torch/CUDA version, SLURM job id and git commit.

**Released code (`peint/`) — what a reader gets today:**

| GPU | best batch | seq/s | tok/s | source |
|---|---|---|---|---|
| H200 | 1024 | 76.2 | 43 285 | `h200_base_generate_large.md` |
| A100-40GB | 256–768 (plateau) | 45.8 | 25 912 | `a100_ceiling_base.md` — **OOMs at 1024** |
| A100-80GB | 1536 | 41.7 | 23 658 | `a100_80gb_base.md` |

**Optimization branch (`inference-speedup`, bit-exact — see §5):**

| GPU | best batch | seq/s | tok/s | M seq/GPU-day | speedup |
|---|---|---|---|---|---|
| **H200** | 1024 | **276.2** | 156 906 | **23.9** | 3.6× |
| A100-40GB | 768 | 151.1 | 85 831 | 13.1 | 3.30× |
| A100-80GB | 1024 | 128.2 | 72 815 | 11.1 | 3.07× |
| A5000 | 64 | 24.1 | 13 707 | 2.1 | 1.43× |

At *matched* batch 64 the speedup is only 1.37–1.66×; roughly two thirds of the
3.3× is being able to run a larger batch at all, since the optimizations halved
activation memory and the baseline OOMs first. Quote the matched-batch figure if a
conservative framing is wanted.

Batch is the dominant lever: H200 generation goes 18.7 k tok/s @64 → 141.0 k
@1024, a **7.5×** span. Throughput is non-monotonic near capacity (276 @1024 vs
246 @2048), so ~75–80 % of what fits is the sweet spot.

**Other workloads, one H200:**

| workload | problem size | released code | optimized | speedup |
|---|---|---|---|---|
| likelihood / VEP | 2 048 targets, bs 128 | 3 656 ms (560 seq/s) | **526 ms (3 895 seq/s)** | 6.95× |
| homology all-vs-all | N=200 (39 800 pairs) | 78.5 s (507 pairs/s) | **26.3 s (1 514 pairs/s)** | 2.98× |

The likelihood speedup *grows* with GPU speed (2.81× on A5000 → 6.95× on H200)
because the released path is CPU-bound on per-sequence Python tokenization, which
a faster card cannot help.

**Multi-GPU.** Data-parallel fan-out over a node's GPUs, no DDP/NCCL. All-vs-all
at N=400: 864 pairs/s on 1 GPU → 2 190 pairs/s on 4 (2.53×). The shortfall is
startup, not sharding — each rank builds ESM2-150M (~15 s/GPU), so a work list
under ~1 min of single-GPU work is *slower* on 4 GPUs (measured: 27.7 s → 37.8 s
at N=120). Compute itself scales linearly.

---

## 4. GPU memory

Peak reserved (`torch.cuda.max_memory_reserved`), optimized path, H200 generation:

| batch | 64 | 128 | 256 | 512 | 1024 | 3072 |
|---|---|---|---|---|---|---|
| peak | **2.6 GB** | 5.2 GB | 10.0 GB | 19.8 GB | 30.7 GB | 111 GB |

Roughly linear in batch. Other workloads: likelihood 2 048 targets bs 128 =
**4.6 GB**; homology N=200 = **1.7 GB**.

Baseline vs optimized at matched batch 256 on A100: 19 980 → 10 062 MiB =
**1.99×**. On a 40 GB card that halving *is* throughput, because it is what allows
batch 768.

**Practical adoption guidance for the reader:**
- **≥ 8 GB card** — batch 64 at 2.6 GB, ~33 seq/s on an H200-class GPU.
- **40 GB A100 or better** — best-throughput config (~31 GB at batch 1024).
- **80 GB buys nothing** for generation: throughput peaks near batch 1024 and
  declines beyond.
- **CPU / pre-Ampere GPU** — `PeintTransformerVanilla` (`use_flash=False`) runs
  anywhere; no throughput number measured for it (§6).

---

## 5. Fidelity cost of the speedups: zero

Every optimized number above is **bit-exact** against the released checkpoint:
`max_abs_diff == 0.0` on teacher-forced logits, likelihood NLLs and argmax
generation, and identical homology hit lists (14 280 hits, ranking identical),
verified independently on A5000, A100 and H200
(`sbatch benchmarks/inference/parity.sbatch`). No `torch.compile`, no TF32, no
quantization, no added autocast — nothing that trades exactness for speed. Output
is also independent of GPU count by construction.

---

## 6. What is NOT measured — do not assert these

1. **Classical simulator wall-clock (WAG/LG via AliSim).** Nothing. *Blocked, not
   skipped:* `protevo/simulation/_alisim.py` shells out to an `iqtree2` binary and
   the git submodule is uninitialized with no `iqtree2` on PATH. To close:
   `git submodule update --init --recursive`, cmake build, then time
   `simulation/classical.py` on the same trees. It must be reported as **CPU**
   wall-clock with core count — putting it in the same column as GPU throughput
   without that label would be misleading, and a referee will notice.
2. **Competing neural approaches.** Nothing measured. The only before/after we
   have is PEINT vs PEINT.
3. **Flash vs vanilla, and KV-cache vs no-cache, head-to-head.** Never timed. We
   know `PeintTransformerVanilla` re-runs the full ESM2 backbone once per generated
   token, but there is no number.
4. **Sequence-length dependence.** Every figure is at source length 284. Cost
   should be ~linear (decode steps = 2 × length) but it is not swept.
5. **Larger backbones.** No ESM2-650M or ESM-C efficiency data of any kind.
6. **Training cost of the original run** — §2 is the verified reproduction's rate
   on 2 × A100-40GB, not the original job's accounting record (which used
   4 GPUs × accum 6 × batch 32, the same effective batch).

## 7. Caveats to carry into the letter

- **Which code are we quoting?** The 3.3×/6.95× figures require the
  `inference-speedup` branch. If the response letter quotes 276 seq/s, that branch
  has to ship in the release; otherwise quote the §3 released-code row.
- Ceiling sweeps used `--iters 1 --warmup 0`; `std_ms = 0.00` there is an artefact
  of n=1. The `{base,fast}_*` sweeps used warmup 2 / iters 5.
- **Run-to-run variance is ~20 % across nodes.** Never compare batch sizes across
  runs or nodes; an earlier cross-run comparison put the peak at batch 3072 and
  that was noise.
- All benchmark nodes were shared (`mix` state); some contention is baked in.
- The corpus is **synthetic** — seeded mutations of `protevo/tests/example_transition.txt`
  — and **uniform-length**, one source replicated across the batch. Real ragged
  corpora are slower unless length binning is on (worth 1.37× for generation, now
  the default; `results/bin_{pack,nopack}.md`).
- **Do not quote MFU or HBM-utilization.** The metric in files older than
  `results/h200_roofline.md` is wrong by ~20×, and even corrected it is a
  diagnostic, not an efficiency claim.
- All of it is ESM2-150M / `peint.ckpt`.

---

*Sources: `docs/INFERENCE_OPTIMIZATION_REPORT.md` §1 and §5; `results/*.json`
(metadata-stamped); `model_checkpoints/{peint,vep}.ckpt`;
`/scratch/users/yufan.cao/torch_cache/hub/checkpoints/esm2_t30_150M_UR50D.pt`;
SLURM job 3365814 and `protevo_ablations/baseline/`.*
