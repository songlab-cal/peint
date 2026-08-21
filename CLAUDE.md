# PEINT Repository

PEINT is an encoder-decoder transformer model designed for protein evolutionary modeling.
The encoder is given a source sequence (x) and the decoder is given a target sequence (y) and the tree distance between them (t).
The model is trained to autoregressively decode the target sequence given the source sequence and the tree distance.
Time is added as a sinusoidal embedding to the decoder input.
ESM2 is used as a pretrained model to provide general pretrained features for the encoder input.

## Repository Structure

- `protevo/`: Main PEINT model code, including the encoder-decoder architecture, datasets and utils.
  - `protevo/models/`: Model architecture code (no pytorch-lightning or wandb dependency).
    - `_transformer.py`: Core transformer classes: `PeintTransformer` (Flash Attention), `PeintTransformerVanilla` (standard attention), `PeintGenerator` (KV caching for generation), `PeintEvaluator` (encoder caching for likelihood evaluation).
    - `_transformer_modules.py`: Transformer modules. Some use FLASH attention, some don't. Some have KV caching, some don't.
    - `_flash_esm.py`: Rewrites the base ESM2 modules using FLASH attention for faster training.
    - `_loading.py`: Model loading from checkpoints (works without pytorch-lightning installed).
    - `_lg.py`: LG model.
    - `_wag.py`: WAG model.
    - `_equ.py`: EQU model (equal exchangeabilities; not trained).
    - `_uniform_random_guess.py`: Uniform random guess baseline (no parameters).
    - `_optimization.py`: LR scheduler.
  - `protevo/models/training.py`: Training components (requires pytorch-lightning and wandb).
    - `PeintLightningModule`: PyTorch Lightning wrapper for training.
    - `ValidationLikelihoodCallback`, `GradNormCallback`: Training callbacks.
  - `protevo/datasets/`: Dataset loading and processing code.
    - `_torch_datasets.py`: `PeintDataset` and `PeintCollator` (no lightning dependency).
    - `training.py`: `PeintDataModule` (requires pytorch-lightning).
  - `protevo/evaluation/`: Per-site likelihood evaluation against the classical models.
    - `_aligned_transitions.py`: `AlignedTransitionsDataset`, which maps PEINT's unaligned residues back onto alignment columns using the a3m alignment mask.
    - `_likelihood.py`: Per-site log-likelihoods, cached per family, written in the same format as the LG/WAG evaluators.
  - `protevo/homology.py`: Homology search code.
  - `protevo/simulation/`: Code for simulating protein sequences along phylogenetic trees.
  - `protevo/time_mle/`: Maximum likelihood estimation of evolutionary time.
  - `protevo/caching/`: Caching utilities for expensive computations.
  - `protevo/utils.py`: Utility functions.
- `tests/`: Pytest test suite (integration tests require model checkpoint).

## Dependency Architecture

Training dependencies (pytorch-lightning, wandb) are isolated in dedicated modules. Core model code can be used for inference without these dependencies:

```python
# Core imports - no lightning/wandb required
from protevo.models import PeintTransformer, PeintGenerator, PeintEvaluator, load_model
from protevo.datasets import PeintDataset, PeintCollator

# Training imports - requires lightning/wandb
from protevo.models.training import PeintLightningModule, ValidationLikelihoodCallback
from protevo.datasets.training import PeintDataModule
```

Can you help me to make this ready to share. There are some hard-coded links, and I have changed the name of the checkpoint from `epoch=2-step=40000.ckpt` to `peint.ckpt`. This needs to be updated in the tests.

I have also provided a checkpoint for variant effect prediction `vep.ckpt`.
Let's do the following:
1. Remove hard links.
2. Convert these checkpoints to peint_only checkpoints (much smaller), and ensure the tests still run.
3. Create a simple VEP notebook showing VEP performance. I have the data in this folder (`NRAM_I33A0_Jiang_2016.csv`), and an example notebook (`vep_example.ipynb`) that come from a different time and may not work. Please update the functions to use the `vep.ckpt` and run VEP on that model on that example.


## Design Principles

1. **Lightweight Core**: Core model code has minimal dependencies. Training-specific dependencies (pytorch-lightning, wandb) are isolated in `training.py` modules.
2. **Format Code**: Keep everything well linted and formatted.
3. **Documentation**: Add docstrings to non-trivial functions and classes. Be concise.
4. **Simplify Relentlessly**: Remove complexity aggressively - the simplest design that works is usually best.

## Testing Guidelines

**Real Testing Only** - Do not use mocks or fake implementations.

1. **Use existing fixtures**: Check `tests/conftest.py` for real data fixtures
2. **Use compatibility patterns**: Follow `tests/test_trainer_netam_compatibility.py` for real validation
3. **Use actual models**: Load real models, don't mock them
4. **Ask for help**: If real testing seems difficult, the design may need improvement
