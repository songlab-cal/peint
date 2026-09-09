# Variant Effect Prediction (VEP)

Variant effect prediction with **PEINT** on the **ProteinGym** substitution benchmark.

PEINT is an encoder–decoder transformer trained on evolutionary transitions — pairs of
related sequences `(x, y)` with an evolutionary time `t`. A mutant is scored by the
conditional log-likelihood `log p(x_mut | x_wt, t)`, with `x` the wild type, `y` the mutant,
and `t = 1.0` by default (following Prillo et al.). A small trainable PEINT head sits on top
of a **frozen** ESM2 (or vESM) encoder; that head is what these scripts train and score.

## Setup

**Environment.** Install peint into the `peint` conda env: `pip install -e .`.
Optional extras: `.[train]` adds `wandb` (the default training logger); `.[vep]` adds
`huggingface_hub` + `transformers` (only needed for vESM encoders). Training expects a single
A100 (bf16 + FlashAttention).

**ProteinGym data.** Nothing in this module needs a ProteinGym checkout to *score* a
checkpoint or to draw the paper's VEP panels; the scored per-run tables ship with the paper
repository. A checkout is needed only to rebuild the VEP inputs from scratch, or to compare
against ProteinGym's own zero-shot baselines. Two external resources are involved, both from
[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) (MIT):

- `reference_files/DMS_substitutions.csv` — the DMS assay reference table (217 assays).
- `input_data/ProteinGym_v1.3/zero_shot_substitutions_scores/` — the official baseline scores,
  distributed as a download rather than in the repository.

Put a ProteinGym checkout at `peint/vep/ProteinGym3` (the default, gitignored) or point
`PEINT_PROTEINGYM_DIR` at it. The CherryML cache directories named in `_config.py` are produced
by ProteinGym's own preprocessing and live under that same directory.

**Data symlinks.** Large data and checkpoints live on `/scratch` and are reached through two
per-user symlinks under the repo root. They are gitignored (`local_data/`), so create them
yourself:

```bash
ln -s /scratch/users/<you>/.../peint_data     data/local_data/peint_data
ln -s /scratch/users/<you>/.../checkpoints    data/local_data/checkpoints
```

`_config.py` is the single source of truth for paths, CherryML cache hashes, the time grid,
and the encoder registry — import from it rather than hard-coding.

## Module layout

| Path | Purpose |
|---|---|
| `_config.py` | Paths, cache hashes, time grid, `ESM_REGISTRY`. Import paths from here. |
| `encoders.py` | `load_esm_model(which_esm, use_flash)` — the one place the frozen encoder is chosen (stock ESM2 + vESM). |
| `_scoring.py` | `score_transition_pairs`, the shared per-mutant likelihood loop. |
| `_vep_utils.py` | `load_model` (Lightning-free; handles stripped + full checkpoints) + result-loading helpers. |
| `_dms_datasets.py` | Builds training transitions from the DMS-family MSAs (uses `datasets/_msa_datasets.py`). |
| `train_peint_vep.py` | Training entrypoint (all families; the paper config). |
| `_train_utils.py` | Arg parser, model/trainer/callback setup, `DMSEvaluationCallback`. |
| `compute_fitness.py` | Offline scoring + Spearman (GPU or CPU). |
| `compute_wag_scores.py` | WAG baseline scores. |

Plotting/figure code lives in the **peint-paper** repo, not here.

## Pipeline

```
ProteinGym MSAs → training transitions → TRAIN → SCORE
 (upstream ProteinGym)  _dms_datasets.py   train_peint_vep   compute_fitness
```

**1. Upstream ProteinGym preprocessing.** Two artifacts come from ProteinGym rather than from
this package: the reformatted training MSAs, which `training_msa_dir()` points at, and the
wild-type/mutant test transition pairs, which `_config.TEST_TRANSITIONS_DIR` points at. Both are
CherryML-cached directories under whatever `PEINT_PROTEINGYM_DIR` names, and PEINT only reads
them. To regenerate them, run `proteingym/baselines/SiteRM/_datasets.py` in a ProteinGym
checkout.

To only *evaluate* an existing checkpoint against the shipped transitions, stop here.

**2. Build training transitions.** `python -m peint.vep._dms_datasets` (no args; the
production set is hardcoded to `hhfilter90`, dropping families with <10 sequences).

**3. Train** (needs `.[train]` + a GPU):

```bash
python -m peint.vep.train_peint_vep \
  --finetune_data_path     $VEP/training_data/hhfilter90/transitions/unaligned/ \
  --finetune_families_file $SETS/dms_fam_all.json --finetune_mix_ratio 1.0 \
  --which_esm 650M --num_encoder_layers 2 --num_decoder_layers 2 \
  --batch_size 128 --accumulate_grad_batches 4 --lr 1e-5 --max_steps ... \
  --eval_dms --eval_dms_families_file $SETS/dms_fam_30.json \
  --eval_dms_transitions_dir <TEST_TRANSITIONS_DIR> \
  --eval_dms_labels_dir <DMS_DATA_FOLDER> --output_dir $CKPT --name_addon 650M
```

`--which_esm` accepts any key in `_config.ESM_REGISTRY` (`150M`/`650M`/`3B`/`15B`, or
`vesm_650M`); it sets `embed_dim` automatically and is recorded in the checkpoint hparams.
With `--eval_dms`, the best `--save_top_k` checkpoints by `dms/avg/spearman` are kept.

**4. Score** (GPU or CPU — the encoder stack is chosen automatically):

```bash
python -m peint.vep.compute_fitness \
  --checkpoints $CKPT/<run>/<ckpt>.ckpt \
  --output_dir  $VEP/test_lls/production \
  --data_dir    <TEST_TRANSITIONS_DIR> --times 1.0
```

Per `(run, ckpt, t)` it writes `*_preds.txt`, `scores/<family>.csv`, `spearman_results.csv`,
and `spearman_by_mutation_depth.csv`. Options: `--batch_size`, `--family_subset`, `--indels`,
`--output_suffix`, time sweeps, and `--wag_predictions_dir`/`--t_wag` for WAG-corrected scores
(`python -m peint.vep.compute_wag_scores` produces the WAG baseline).

## Model loading & the encoder swap

`_vep_utils.load_model(ckpt, device, use_flash=True)` returns a scoring-ready model with **no
Lightning wrapper** and handles both checkpoint layouts automatically:

- **Full** checkpoint (frozen encoder saved): the ESM2 architecture is inferred from the saved
  weights and the encoder is loaded straight from the checkpoint — no `which_esm`, no download,
  fully offline (works identically for stock ESM2 and vESM).
- **Stripped** ("PEINT-only") checkpoint — what the distributed `vep.ckpt` is — the frozen
  encoder is rebuilt from `which_esm` (falling back to `embed_dim` for stock sizes) and the head
  is loaded with `strict=False`.

`use_flash=True` (default) builds the FlashAttention stack (`ESM2Flash` + `PeintTransformer`,
GPU). Set `use_flash=False` for the standard-PyTorch stack (`ESM2Model` +
`PeintTransformerVanilla`) to run on CPU; `compute_fitness` selects this automatically off-GPU.

## Notes

- **Continuous time only.** The quantized-time option from the original code was dropped
  (peint's transformer uses the continuous `GeometricTimeEmbedder`); `t = 1.0` is the default.
- **Scoring needs neither Lightning nor wandb.** Only training pulls those in.
- **wandb.** The training scripts log to W&B; substitute your own Lightning logger if you don't
  want it (peint's training callbacks are wandb-native, so training as-shipped requires wandb).
- **ProteinGym** is not a dependency of this package. The one helper that used to be imported
  from it (`get_dms_substitutions_families`) is now in `_dms_datasets.py`; its data files stay
  external, as described above.
