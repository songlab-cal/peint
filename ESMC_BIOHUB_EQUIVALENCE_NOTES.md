# ESM‑C Biohub ↔ esm3 equivalence — verification notes

**Date:** 2026‑08‑10  ·  **Branch:** `vep-esmc-biohub`  ·  **Node:** yss A100 80GB  ·  **Env:** `peint-esmc`

Cross‑check of the **Biohub transformers‑fork ESM‑C** build (Antoine's `esmc-biohub` registry
entry) against **Yufan's esm3‑package ESM‑C** implementation, using Yufan's golden handoff.

- **Checkpoint (A3 ESM‑C, 60k; frozen `esmc_300m` + trained PEINT):**
  `/scratch/users/yufan.cao/protevo_ablations/esmc/20260729-5e5d20h960d-esmc-14498fams-esmc/epoch=4-step=60000.ckpt`
- **Golden reference:** `/scratch/users/yufan.cao/handoff/esmc_biohub/` (`xyt_reference.npz`, `esmc_vocab.json`)

## TL;DR

**The two ESM‑C builds are equivalent.** Same tokenizer, same weights, same predictions. The
only difference is numerical (bf16 activation‑kernel noise between two independent
implementations), and it never flips a prediction.

## 1. Environment / installations (in conda env `peint-esmc`)

- **transformers:** Biohub fork **4.57.6** (has `transformers/models/esmc/`). ⚠️ The `peint` env's
  mainline **transformers 5.14.1 does NOT work** for the Biohub build — use `peint-esmc`.
- **`transformer-engine` 1.13.0 was installed to test the fused‑LayerNorm path, then uninstalled**
  (env kept simple + numerically identical to the already‑trained VEP runs, which had no TE).
  - install had pulled prebuilt `transformer_engine_cu12` + source‑built `transformer_engine_torch`
    (deps `cmake`/`ninja`/`pybind11`; `nvcc` from `/usr/local/cuda-12.8`, `--no-build-isolation`,
    torch pinned `==2.5.0` so it was NOT upgraded).
  - **Result: TE is optional for correctness** (see §3) — it did not change predictions and did not
    close the logit gap, so it was removed. To re‑enable later: `pip install --no-build-isolation
    transformer-engine[pytorch]==1.13.0` with the CUDA toolkit on PATH.
  - `cmake`/`ninja`/`pybind11` were left installed (harmless build tools).

## 2. Equivalence results

**Tokenizer — identical.** Biohub `ESMCTokenizer` maps `x_seq`/`y_seq` to the exact golden
`x_tokens`/`y_tokens`; all AA + special ids match `esmc_vocab.json` (A=5, C=23, D=13, Q=16, N=17,
cls=0, eos=2, pad=1, mask=32).

**Weights — same weights (all 308 backbone tensors).** Compared biohub `ESMC-300M` fp32
safetensors vs the checkpoint's `model.esm.*` under an explicit esm3↔transformers‑fork name map
(fused `layernorm_qkv`, `ffn.1/3` ↔ `fc1/fc2_weight`, TE `_extra_state` placeholders ignored):

| | Result |
|---|---|
| Tensors mapped 1:1 | **308 / 308** (0 missing, 0 shape mismatches, full bijection) |
| `esm3_bf16 == bf16(biohub_fp32)` bit‑for‑bit | **308 / 308** |
| Storage | Biohub ships **fp32**; esm3 checkpoint stored **bf16** |

→ In the **bf16 production path** the backbones are **bitwise identical**. Biohub's fp32 is just the
same weights at higher precision (esm3 is the bf16 truncation of it).

**Logits — predictions identical.** Fed the exact golden token ids through the production VEP
loader → forward, compared `y_logits (2, 34, 64)`:

| Config | max\|Δ\| vs golden | mean\|Δ\| | per‑position argmax |
|---|---|---|---|
| bf16 + flash (no TE) | 0.44 | 0.034 | **exact (100%)** |
| bf16 + flash (**TE**) | 0.53 | 0.032 | **exact (100%)** |
| fp32 + eager (no TE) | 0.36 | 0.040 | **exact (100%)** |
| fp32 + eager (**TE**) | 0.36 | 0.040 | **exact (100%)** |

Per‑position AA argmax matches exactly at every position, both times (t=0.1, 0.5), over all 64
head slots and the 20 AA slots.

## 3. Why the raw logits differ by ~0.4 (and why it doesn't matter)

- The `~0.03` mean / `~0.5` max logit gap is **irreducible bf16 activation numerics** between two
  independent ESM‑C codebases (transformers‑fork `flash_attn` vs the esm3 package's kernels),
  compounded over 30 layers. It is **prediction‑neutral** (argmax exact everywhere).
- **transformer_engine did NOT close it** — in fp32 the gap was unchanged (0.363 → 0.364), proving
  the LayerNorm‑reduction fallback was never the source. The driver is the attention/accumulation
  path, not LayerNorm.
- Weights are the same (§2), so this is purely the forward‑kernel implementation difference.

## 4. Loader fix (applied) — `encoder_backbone` naming gap

`peint/vep/_vep_utils.py::load_model` routed to the biohub backbone **only** when
`hparams["encoder_backbone"] == "esmc-biohub"`. Yufan's checkpoints (trained on
`foundation`/`ablation-a3-esmc`) record **`encoder_backbone="esmc"`** (esm3‑package name), so
`load_model` fell through to the ESM2 arch‑inference path and **raised** on ESM‑C checkpoints.

**Fix:** accept `encoder_backbone in ("esmc", "esmc-biohub")` (and the same for `which_esm`), so
esm3‑trained ESM‑C checkpoints auto‑load onto the biohub backbone with no override.

**Suggestion for Yufan:** consider standardizing the hparam string across branches (e.g., write
`encoder_backbone="esmc-biohub"` at save time), so checkpoints are self‑describing without the
alias.

## Reproduce

Env `peint-esmc`, `HF_HOME=/scratch/users/akoehl/hf_cache`,
`PYTHONPATH=/scratch/users/akoehl/peint`. Verification scripts (tokenizer + all‑weights +
logits, incl. the TE and fp32 variants) were run one‑off; can be folded into `tests/` on request.
