# PEINT: Protein Evolution IN Time

An encoder-decoder transformer for modeling protein sequence evolution. Given a source sequence and evolutionary time, PEINT autoregressively predicts the target sequence.

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

The trained checkpoints are distributed separately as `peint_model_checkpoints.zip`. Unzip it at the repository root to create the `model_checkpoints/` directory:

```bash
unzip peint_model_checkpoints.zip   # creates model_checkpoints/peint.ckpt and model_checkpoints/vep.ckpt
```

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

```bash
python train_peint_model.py \
    --data_path /path/to/transitions \
    --families_file families.json \
    --output_dir checkpoints \
    --batch_size 32 \
    --lr 3e-4 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --embed_dim 640 \
    --num_heads 20
```

## Dataset Creation

Create training data from a directory of `.a3m` multiple sequence alignment files.

A wrapper around many of these functions is the caching decorator.
For more information, you can see the original source [caching-decorator source here](https://github.com/sprillo/caching-decorator).
The idea is that some functions are fairly expensive to run (tree reconstruction, etc), and represent a bottleneck in a pipeline.
You really want to cache these functions.
This decorator wraps a function and caches its results to disk.

```python
from peint.datasets import get_a3m_families, a3m_dataset__cached

# List available families
families = get_a3m_families("/path/to/a3m_files", num_families=100)

# Create dataset from full trees (no train/test split)
data_dirs = a3m_dataset__cached(
    a3m_dir="/path/to/a3m_files",
    num_families=100,              # -1 for all families
    num_sequences_per_family=512,
    num_processes=16,
    do_train_test_split=False,     # Use full trees
)
# Returns: transitions_dir, msa_dir, tree_dir, site_rates_4cat_dir

# Or with train/test split (for generalization evaluation)
data_dirs = a3m_dataset__cached(
    a3m_dir="/path/to/a3m_files",
    num_families=100,
    num_sequences_per_family=512,
    num_processes=16,
    do_train_test_split=True,      # Split trees into train/test halves
)
# Returns: train_transitions_dir, test_transitions_dir, train_msa_dir, etc.
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a3m_dir` | required | Directory containing `.a3m` files |
| `num_families` | -1 | Number of families (-1 = all) |
| `num_sequences_per_family` | 1024 | Max sequences to subsample per family |
| `do_train_test_split` | True | Split tree for train/test, or use full tree |
| `include_gaps` | True | Include gaps in aligned sequences |
| `return_full_length_unaligned_sequences` | False | Use unaligned sequences (requires `include_gaps=False`) |

Results are cached; subsequent calls with identical parameters return immediately.

## Model Variants

| Class | Flash Attention | KV Cache | Use Case |
|-------|-----------------|----------|----------|
| `PeintTransformer` | Yes | No | Training |
| `PeintGenerator` | Yes | Yes | Fast generation |
| `PeintEvaluator` | Yes | Encoder only | Batch likelihood evaluation |
| `PeintTransformerVanilla` | No | No | CPU/old GPU, attention visualization |

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
