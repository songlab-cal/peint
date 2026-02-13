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
## Quick Start

### Loading a Pretrained Model

```python
from protevo.models import load_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model (use_flash=False for compatibility with all GPUs)
model, vocab = load_model(
    'checkpoints/model.ckpt',
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
from protevo.models import load_model
import torch

model, vocab = load_model('checkpoints/model.ckpt', use_cached_model=False, device='cuda', use_flash=False)

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

### Homology Detection

There are many options for homology detection, we provide a set of tools to do various types of homology detection.
The most basic is an all-vs-all setup, in which you provide a directory of named proteome files (FASTA format), and a distance matrix specifying the evolutionary time between each pair of proteomes.
You can either use peint, or DIAMOND (Blastp) for comparison.

```bash
python -m protevo.homology_detection all-vs-all \
 --method peint \
 --checkpoint model_checkpoints/<model_ckpt>.pt \
 --proteome-dir protevo/tests/homology_test_dir \
 --distance-matrix protevo/tests/times.csv \
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
from protevo.datasets import get_a3m_families, a3m_dataset__cached

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
protevo/
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
