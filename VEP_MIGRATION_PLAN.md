# VEP module migration plan

Migrate the Variant Effect Prediction (VEP) pipeline (junhaobearxiong) from
`songlab-cal/protein-evolution@bear_analysis:peint/vep` into the split repos:

- **peint (this repo)** — the **train / score / eval** suite, as a new `peint/vep/` package.
  Rationale: the encoder is easy to swap, so training a PEINT model on VEP data and scoring it
  is a core library capability.
- **peint-paper** — the **plotting** code (paper + supplementary figures), in a later phase.

Source of truth is `origin/bear_analysis` (the module was restructured in commits
`ae1ebcf` / `42ffc29`; a stale local `bear_analysis` predates that — always read from the remote).

## Scope decisions (confirmed with user)

| Decision | Choice |
|---|---|
| What lands in peint core now | **Train + score + eval only** (no `plotting/`, no `experiments/`) |
| Encoder wiring | **Keep a VEP-local `encoders.py`** (self-contained; imports only `ESM2Flash` from core). peint's minimal `models/_esm_registry.py` (8M/35M/150M) is left untouched. |
| Training-transition builder | **Port it** — bring `_dms_datasets.py` + its `construct_dataset_on_msas` dependency into peint. |
| Plotting | **Migrate to peint-paper** in Phase 2 (not dropped). |
| Quantized time (`--use_quantized_time`) | **Drop it.** peint's transformer hardwires the continuous `GeometricTimeEmbedder`; the flag was already a silent no-op there. Follow-up only if ever needed (would require adding `SinusoidalTimeEncoding` to core). |

## Name / module translation (bear → peint)

Verified against peint's current code.

| bear (`origin/bear_analysis`) | peint (this repo) | Status |
|---|---|---|
| `ESMPretrainedTransformer` (`models._transformer`) | `PeintTransformer` | renamed (peint adds `_PeintTransformerBase` ABC above it) |
| `ProtEvoPretrainedTransformerModule` (`models._transformer`) | `PeintLightningModule` (`models._training`; public `models.training`) | renamed **+ moved module** |
| `PeintGenerator`, `PeintEvaluator` | `PeintGenerator`, `PeintEvaluator` | same names (now subclass `PeintTransformer`) |
| `ESM2Flash` (`models._flash_esm`) | `ESM2Flash` | unchanged |
| `ValidationLikelihoodCallback`, `GradNormCallback` (imported from `models._model_utils`) | same names, **moved** to `models._training_callbacks` (public `models.training`) | fix import path |
| `ProtevoESMEncoderDecoderDataModule` (`datasets._torch_datasets`) | `PeintDataModule` (`datasets._training`) | renamed; near-identical **minus `quantize_time`** |
| `ProtEvoDataset` / `ESMEncoderDecoderTransformerCollator` | `PeintDataset` / `PeintCollator` (`datasets._torch_datasets`) | renamed (used internally by the DataModule) |
| `get_quantile_idx`, `get_quantization_points_from_geometric_grid` (`peint.utils`) | same | unchanged |
| `evaluate_wag_model_transitions_log_likelihood_per_site` (`models._wag`) | same | unchanged |
| `construct_dataset_on_msas` (`datasets._msa_datasets`) | **absent** — port it | see Phase 1 |

### Model-API compatibility notes (verified)
- `PeintLightningModule.__init__` matches bear's `ProtEvoPretrainedTransformerModule.__init__`
  signature (`esm_model, esm_vocab, max_seq_len, num_heads, num_encoder_layers,
  num_decoder_layers, embed_dim, lr, num_warmup_steps, num_training_steps, **kwargs`).
- Extra kwargs the VEP trainer passes (`use_quantized_time`, `which_esm`) are tolerated:
  `_config_from_kwargs` ignores unknown keys, and `which_esm` is still captured by
  `save_hyperparameters()` — so the "rebuild the frozen encoder from `which_esm` at load"
  design is preserved.
- `dropout_p`, `use_attention_bias`, `label_smoothing`, `weight_decay` are all supported by
  `PeintConfig`.
- `PeintDataModule` exposes `.train_dataset / .val_dataset / .collate_fn / .val_dataloader()`,
  which is everything `ProtevoMixedDataModule` and `DMSEvaluationCallback` rely on.
- `_vep_utils.load_model`'s frozen-prefix + `strict=False` logic already matches peint's
  `model.esm.* / model.embedding.* / model.lm_head.*` layout.

## Dependency assessment — porting `construct_dataset_on_msas` (LOW RISK)

Everything `datasets/_msa_datasets.py` needs already exists in peint:

- `peint.io`: `Tree, get_msa_num_residues/sequences/sites, read_msa, read_pickle, read_tree,
  write_msa, write_pickle, write_tree, write_transitions, read_secondary_structure` — all exported ✓
- `peint.datasets._datasets`: `extract_transitions` (L463), `alphabetize_msa` (L557) ✓
- `peint.utils`: `amino_acids`, `gap_character` ✓
- `cherryml` (git dep) ✓, `data/rate_matrices/wag.txt` ✓, `peint.caching` ✓

Edits on port: drop the unused `from peint import caching as peint_caching` import; repoint
`from peint.vep._vep_utils import MAIN_DIR` to `peint.vep._config.MAIN_DIR`. Add `loguru` +
`joblib` to peint deps if not already present.

## Phase 0 — setup

1. Vendor `ProteinGym3` as a git submodule under `peint/vep/`, pinned at `8fd91a0`
   (needed by `_dms_datasets.get_dms_substitutions_families` and the test-transition cache paths).
2. Create `peint/vep/__init__.py`.
3. Gitignored data symlinks (document in README, per bear's setup):
   `data/local_data/peint_data -> /scratch/.../peint_paper`,
   `data/local_data/checkpoints -> /scratch/.../checkpoints/peint`.
4. Add `loguru`, `joblib` to `pyproject.toml` if missing.

## Phase 1 — train / score / eval  → `peint/vep/`

Copy-and-adapt, file by file. Only `_train_utils.py`, `_vep_utils.py`, `train_peint_vep.py`
need edits beyond docstrings.

| File | Action |
|---|---|
| `_config.py` | Copy verbatim. Paths resolve off peint's `MAIN_DIR`. Optional: allow `DATA_ROOT` env override to match peint convention. |
| `encoders.py` | Copy verbatim (self-contained; imports `ESM2Flash` from core). VEP keeps its own `ESM_REGISTRY`. |
| `_scoring.py` | Copy; docstring rename `ESMPretrainedTransformer`→`PeintTransformer` (code is model-agnostic). |
| `_vep_utils.py` | Rewrite `ProtEvoPretrainedTransformerModule`→`PeintLightningModule` (from `peint.models.training`); `PeintGenerator` unchanged. Remove any `use_quantized_time` handling. |
| `_train_utils.py` | Rewrite 3 imports: callbacks `models._model_utils`→`models.training`; datamodule `ProtevoESMEncoderDecoderDataModule`→`PeintDataModule` (`datasets._training`); model `ProtEvoPretrainedTransformerModule`→`PeintLightningModule`. Remove quantized-time plumbing (arg parser flag, `prepare_model_args`, `DMSEvaluationCallback` branch). `ProtevoMixedDataModule` / `DMSEvaluationCallback` / mixing samplers are otherwise unchanged. |
| `train_peint_vep.py` | Import `PeintDataModule`; drop the `quantize_time=` kwarg on both DataModule constructions. |
| `compute_fitness.py` | Copy verbatim (imports resolve after core). Drop the `"quantized" in ckpt_name` branch. |
| `compute_wag_scores.py` | Copy verbatim (`evaluate_wag_model_transitions_log_likelihood_per_site` present). |
| `_dms_datasets.py` | Copy verbatim (needs ProteinGym3 + the ported builder). |
| `datasets/_msa_datasets.py` | **Port into peint** (see dependency assessment). Drop unused caching import; repoint `MAIN_DIR`. |

### Tests (real, no mocks — per CLAUDE.md)
- Pure key-mapping test for `encoders.convert_hf_esm_state_dict_to_fair_esm` on a small
  synthetic state dict (no downloads).
- Import smoke test for the `peint.vep` package.

## Phase 2 — plotting → peint-paper  (after Phase 1 lands)

Move `plotting/{main_figures,analysis_figures,_style,spearman_from_csv}.py` into peint-paper.
Two small deps travel with them and need a home (decide at Phase 2):

- `peint.io._files` (746 B; imported as `from peint.io._files import *`)
- `peint.eda._time_distribution.get_time_distribution`

Options: vendor both into peint-paper, or add thin `io/eda` shims to peint core. Also repoint the
hardcoded figure run-targets to `paper_config`.

## Verification

- `python -c "import peint.vep"` smoke import.
- Run the encoders converter unit test.
- If a small cached test-transitions dir is reachable, a 1-family dry run of
  `python -m peint.vep.compute_fitness ... --family_subset <one>` end-to-end.

## Status (updated 2026-08-01)

**Phases 0 + 1 complete, committed** (peint `vep-migration`): all VEP files ported to
`peint/vep/`, builder `_msa_datasets.py` in place, bear→peint renames applied, quantized-time
removed, ProteinGym3 submodule vendored, tests + README written. Scoring is Lightning/wandb-free
and CPU-capable; `load_model` auto-handles both stripped (distributed `vep.ckpt`) and full checkpoints.

**Phase 2 (plotting) partial**, committed (peint-paper `vep-figure`): `main_figures.py` →
`figures/figure5_vep.py`. Deferred: `analysis_figures.py` + `spearman_from_csv.py` (+ their
`get_time_distribution` / `io._files` deps).

**Phase 3 — GPU validation (DONE, A100 80GB, `peint` env):**
- **Scoring (flash path):** full 298-variant NRAM set, `load_model(vep.ckpt, use_flash=True)` →
  `PeintTransformer`, Spearman **0.4040** (notebook ref ~0.405). Exercises the stripped-ckpt rebuild.
- **Training:** `train_peint_vep` 3-step smoke (150M, 3 families, mix_ratio 1.0) runs end to end and
  writes 830M *full* checkpoints; a fresh checkpoint reloads via the full-ckpt branch
  (`_infer_esm_arch_from_state_dict` = (30,640,20)) and scores. Both `load_model` branches now GPU-verified.
- **SFT mixing:** `--finetune_data_path` (DMS gapless) + `--data_path` (original 1024L transitions
  `/scratch/users/akoehl/old/protein-evolution/local_data/15k_gapless_scale_1024l`) +
  `--finetune_mix_ratio 0.5` → `MixedDatasetSampler` pulls 8 finetune + 8 original per batch of 16.
- Installed `wandb` 0.28.1 into the `peint` env (training requires it — callbacks import wandb at
  module top, entrypoint hardcodes a `WandbLogger`; run offline via `WANDB_MODE=offline`).

**Code fixes this session** (`_train_utils.py`, uncommitted): `--devices` now auto-selects when the
flag is omitted (pass ints for specific ids), and `validate_args` fails fast with a clear
`RuntimeError` if CUDA is unavailable — previously the `None` default crashed Lightning's Trainer.

**Data fix:** `DMS_finetune_transitions_gapless` had sed-corrupted times — bear's regeneration ran
`sed 's/-//g'` (to strip gap chars) which also ate the `-` in float exponents, so `9.99e-09`→`9.99e09`
(1e-8 → 1e10, above the `MIN_TIME_THRESHOLD` floor, so used verbatim). Fixed in place by copying the
correct time token from the sibling aligned dir `DMS_finetune_transitions` (proven row-for-row
correspondent; all 4750 corrupted times were sed-consistent). 154 files / 4750 tokens; aligned dir untouched.

## Out of scope / follow-ups

- Quantized time (would require `SinusoidalTimeEncoding` in core).
- `experiments/` (`per_family.py`, `mle_optimal_t.py`) — bear marks these non-paper.
- The three locally-modified `ProteinGym3` files (machine-specific ProteinGym zero-shot config) —
  not used by the PEINT pipeline; do not vendor as-is.
- `bash_scripts/` — bear notes these are shelterin/local-only and pass some undefined flags; rely
  on the README example commands rather than porting them verbatim.
