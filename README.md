# PEINT: Protein Evolution IN Time

An encoder-decoder transformer for modeling protein sequence evolution. Given a source sequence and evolutionary time, PEINT autoregressively predicts the target sequence.

The paper's figures, benchmarks and data live in the companion
[`peint-paper`](https://github.com/songlab-cal/peint-paper) repository. The data — model
checkpoints included — is deposited on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22151902.svg)](https://doi.org/10.5281/zenodo.22151902)

Cite [`10.5281/zenodo.22151902`](https://doi.org/10.5281/zenodo.22151902) — record `22151902`,
the exact version the paper's results were reproduced from. This repository and `peint-paper`
both carry the matching tag `zenodo-22151902`.

**What this repository needs from the deposit:** `peint_checkpoints.tar.zst` alone — the five
model checkpoints. Add `peint_transitions_aligned.tar.zst` and `peint_transitions_unaligned.tar.zst`
only to rerun the held-out likelihood evaluation. Every other archive is figure data belonging to
`peint-paper`, whose `data/MANIFEST.toml` is the authoritative per-file inventory.

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.5 (install separately to match your CUDA version)
- CUDA-capable GPU recommended (Flash Attention requires Ampere or newer)
- For non-Flash version (`PeintTransformerVanilla`): any GPU or CPU

Note: Flash Attention (`flash-attn`) requires specific CUDA/PyTorch/OS combinations. If installation fails, see `installation.md` for troubleshooting, including how to find the correct prebuilt wheel.
We used `flash-attn==2.7.0.post2`

```bash
# Clone the repository
git clone https://github.com/songlab-cal/peint.git
cd peint

# 1. Install PyTorch (match your CUDA version) and Einops prior to installing Flash Attention
# See https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121  # example for CUDA 12.1
pip install einops==0.8.1

# 2. (Recommended) Install Flash Attention for faster training/inference
# Requires Ampere+ GPU. See installation.md for troubleshooting.
# This can be found at the Flash-Attention Page: https://github.com/Dao-AILab/flash-attention
# We used FlashAttention 2.7.0.post2, which is compatible with PyTorch 2.5.0 and CUDA 12.1

# 3. Install core package
pip install -e .

# For training (adds pytorch-lightning and wandb)
pip install -e ".[train]"

# For development
pip install -e ".[dev]"
```

## Model Checkpoints

The trained checkpoints ship in the [Zenodo deposit](https://doi.org/10.5281/zenodo.22151902)
as `peint_checkpoints.tar.zst`. Its members unpack under `peint/model_checkpoints/`, which would
collide with this repository's `peint/` package directory — so extract it elsewhere and move the
directory into place:

```bash
curl -L -O "https://zenodo.org/records/22151902/files/peint_checkpoints.tar.zst?download=1"
mkdir -p /tmp/peint_ckpt
tar --use-compress-program=unzstd -xf peint_checkpoints.tar.zst -C /tmp/peint_ckpt
mv /tmp/peint_ckpt/peint/model_checkpoints ./model_checkpoints   # creates model_checkpoints/peint.ckpt, vep.ckpt, ...
rm -rf /tmp/peint_ckpt
```

Requires `zstd` on `PATH`. The two checkpoints below are the ones this README uses; the archive
carries five in total.

| Checkpoint | Description |
|------------|-------------|
| `model_checkpoints/peint.ckpt` | Base PEINT model — sequence generation, likelihood evaluation, and time estimation. |
| `model_checkpoints/vep.ckpt` | Model for variant effect prediction (see `vep.ipynb`). |

These are lightweight **PEINT-only** checkpoints: they contain just the trained PEINT layers. The ESM2 backbone (`esm2_t30_150M_UR50D`) is downloaded automatically from the public release the first time a model is loaded.

## Quick Start

### Loading a Pretrained Model

```python
from peint.models import load_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (use_flash=False for compatibility with all GPUs)
model, vocab = load_model(
    'model_checkpoints/peint.ckpt',
    use_cached_model=True,
    device=device,
    use_flash=False
)

# Encode a source sequence
x = "ACDEFGHIKLMNPQRSTVWY"
x_tokens = torch.tensor([vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx]).unsqueeze(0).to(device)
t = torch.tensor([[0.1]]).to(device)  # evolutionary time

# Generate evolved sequence
generated = model.generate(x_tokens, t, max_decode_steps=len(x) + 10, device=device)
print(generated[0])
```

### Evaluating Likelihood

```python
from peint.models import load_model
import torch

model, vocab = load_model('model_checkpoints/peint.ckpt', use_cached_model=False, device='cuda', use_flash=False)

x = "ACDEFGHIK"
y = "ACDEYGHIK"
t = 0.1

# Prepare inputs
x_toks = torch.tensor([vocab.cls_idx] + vocab.encode(x) + [vocab.eos_idx]).unsqueeze(0).cuda()
y_toks = torch.tensor([vocab.cls_idx] + vocab.encode(y)).unsqueeze(0).cuda()
y_targets = torch.tensor(vocab.encode(y) + [vocab.eos_idx]).unsqueeze(0).cuda()
ts = torch.tensor([[t]]).cuda()

x_mask = x_toks.eq(vocab.padding_idx)
y_mask = y_toks.eq(vocab.padding_idx)

# Forward pass
with torch.no_grad():
    x_logits, y_logits, *_ = model(x_toks, y_toks, ts, x_mask, y_mask)

# Compute negative log likelihood
nll = torch.nn.functional.cross_entropy(
    y_logits.transpose(-1, -2), y_targets,
    ignore_index=vocab.padding_idx
)
print(f"NLL: {nll.item():.4f}")
```

### Variant Effect Prediction

Load `vep.ckpt` as an `evaluator` and score many variants against a wild-type sequence with `evaluate_likelihood`. This works with or without Flash Attention (it falls back to the standard-attention model automatically).

```python
from peint.models import load_peint_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, vocab = load_peint_model('model_checkpoints/vep.ckpt', device=device, model_type='evaluator')

wild_type = "ACDEFGHIKLMNPQRSTVWY"
variants = ["ACDEFGHIKLMNPQRSTVWY", "ACDEYGHIKLMNPQRSTVWY"]  # mutant sequences

# Per-residue negative log-likelihood of each variant given the wild type at t=1.
# Higher likelihood (lower NLL) => more wild-type-like / higher predicted fitness.
nlls = model.evaluate_likelihood(x=wild_type, y=variants, t=[1.0] * len(variants), device=device)
print(nlls)
```

See `vep.ipynb` for an end-to-end example on deep mutational scanning data.

### Per-Site Likelihood vs. Classical Models

PEINT reads unaligned sequences, so its per-residue log-likelihoods are indexed by
residue, while LG and WAG score alignment columns. `peint.evaluation` bridges the
two: it runs PEINT on the unaligned transitions, drops the residues that are
insertions relative to the query (using the a3m alignment mask), and places what
remains in its alignment column. Gap columns are scored 0, so summing over sites
ignores them; use `dataset.scored_columns_mask()` to average over the rest.

```python
from peint.evaluation import (
    AlignedTransitionsDataset,
    evaluate_transitions_log_likelihood_per_site,
)
from peint.models import load_peint_model
import torch

device = torch.device('cuda')
model, vocab = load_peint_model('model_checkpoints/peint.ckpt', device=device, model_type='standard')

dataset = AlignedTransitionsDataset(
    transitions_dir='.../unaligned/test_transitions_dir',
    aligned_transitions_dir='.../aligned/test_transitions_dir',
    alignment_mask_dir='.../unaligned/test_alignment_mask_dir',
    family='4djg_1_B',
    vocab=vocab,
)
# [num_transitions, alignment_width]; 0 in gap columns, NaN where nothing was scored.
per_site = evaluate_transitions_log_likelihood_per_site(model, vocab, dataset, device)
mean_ll = per_site[dataset.scored_columns_mask()].mean()
```

The same computation over many families, cached and written in the same layout as
the LG/WAG evaluators:

```bash
python -m peint.evaluation \
    --transitions-dir   local_data/unaligned/test_transitions_dir/output_transitions_dir \
    --aligned-transitions-dir local_data/aligned/test_transitions_dir \
    --alignment-mask-dir local_data/unaligned/test_alignment_mask_dir \
    --checkpoint model_checkpoints/peint.ckpt \
    --output-dir likelihoods --num-families 5 --device cuda
```

`figure2_ll_eval.py` runs the full comparison against the uniform random guess, WAG
and LG baselines and produces the likelihood-vs-time figures.

Note: the Flash Attention path scores in bfloat16, which costs roughly 0.1 nats on
an individual site. Pass `use_flash=False` for fp32 when per-site values matter.

### Homology Detection

There are many options for homology detection, we provide a set of tools to do various types of homology detection.
The most basic is an all-vs-all setup, in which you provide a directory of named proteome files (FASTA format), and a distance matrix specifying the evolutionary time between each pair of proteomes.
You can either use peint, or DIAMOND (Blastp) for comparison.

```bash
python -m peint.homology_detection all-vs-all \
 --method peint \
 --checkpoint model_checkpoints/<model_ckpt>.pt \
 --proteome-dir peint/tests/homology_test_dir \
 --distance-matrix peint/tests/times.csv \
 --output results.csv
```

## Training

The command used for the released checkpoint (also in `train_peint_model.sh`):

```bash
python train_peint_model.py \
    --data_path /path/to/unaligned/train_transitions_dir \
    --families_file families.json \
    --output_dir checkpoints \
    --esm_model ESM2-150M \
    --num_encoder_layers 5 --num_decoder_layers 5 --embed_dim 640 --num_heads 20 \
    --use_attention_bias --dropout_p 0.0 --max_seq_len 1022 \
    --batch_size 32 --accumulate_grad_batches 12 --devices 0 1 --accelerator gpu \
    --lr 3e-4 --weight_decay 0.01 --grad_clip 1.0 --num_warmup_steps 2000 \
    --max_steps 300000 --checkpoint_every 4000 --n_families -1 --seed 0
```

Pass this configuration explicitly: several argparse defaults differ from it. The effective batch
is 768 sequences per update (32 x 12 accumulation x 2 GPUs); keep that product fixed on other GPU
counts. `--data_path` must be a **gapless** transitions directory (see below).

## Dataset Creation

Create transitions from a directory of `.a3m` multiple sequence alignments. There are two frames:

| frame | built with | used for |
|---|---|---|
| **unaligned / gapless** | `include_gaps=False`, `return_full_length_unaligned_sequences=True` | **training and validating PEINT** |
| aligned | `include_gaps=True` (the default) | fitting the classical WAG / LG baselines |

> **Gaps are removed at dataset-creation time, not by the training data loader.** Training on
> aligned transitions silently trains the model on gap tokens.

Configure both the PEINT and CherryML cache directories, and call dataset creation under an
`if __name__ == "__main__":` guard (it uses multiprocessing):

```python
from cherryml import caching as cherryml_caching
from peint import caching as peint_caching
from peint.datasets import a3m_dataset__cached


def main():
    peint_caching.set_cache_dir("/path/to/_cache_peint")
    cherryml_caching.set_cache_dir("/path/to/_cache_cherryml")

    # PEINT training data: gapless, with the within-family train/test tree split
    data_dirs = a3m_dataset__cached(
        a3m_dir="/path/to/a3m_files",
        num_families=-1,                    # -1 = all families
        num_sequences_per_family=2048,
        return_full_length_unaligned_sequences=True,
        include_gaps=False,
        do_train_test_split=True,
        num_processes=16,
    )
    print(data_dirs["train_transitions_dir"])   # --data_path for train_peint_model.py

    # Aligned data, for the classical baselines only
    baseline_dirs = a3m_dataset__cached(
        a3m_dir="/path/to/a3m_files",
        num_families=-1,
        num_sequences_per_family=2048,
        do_train_test_split=True,
        num_processes=16,
    )


if __name__ == "__main__":
    main()
```

## Testing

```bash
# Run unit tests (no checkpoint required)
pytest tests/test_datasets.py

# Run integration tests (requires checkpoint in model_checkpoints/)
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Project Structure

```
peint/
├── models/           # Model architectures
│   ├── _transformer.py          # Main PEINT models
│   ├── _transformer_modules.py  # Attention and layer modules
│   ├── _flash_esm.py           # Flash Attention ESM2
│   └── _loading.py             # Checkpoint loading
├── datasets/         # Data loading
├── simulation/       # Sequence simulation on trees
├── time_mle/        # Time estimation
├── caching/         # Computation caching utilities
└── utils.py
```
