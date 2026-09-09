# PEINT Installation Instructions

PEINT runs on top of a **frozen base protein language model**. Two backbones are supported, and
they need **two different environments** because they pull incompatible versions of
`transformers`:

| Environment | Backbone | Used for |
|---|---|---|
| `peint` | ESM2 (`fair-esm`) | `peint.ckpt`, `vep.ckpt`, training, simulation, time MLE, homology |
| `peint-esmc` | Biohub ESM-C 300M (`transformers` fork) | ESM-C checkpoints (`peint_esmc.ckpt`, `vep_esmc.ckpt`) |

`peint-esmc` is a **clone of `peint`** with one package swapped (see
[ESM-C environment](#esm-c-environment-peint-esmc)). Everything the ESM2 environment does, the
ESM-C environment also does — but not vice versa, so if you only ever touch ESM2 checkpoints you
can stop after the first section.

The core set of dependencies essentially works all the way back to PyTorch 2.2, while the
benchmarking dependencies require PyTorch 2.5.0 or higher.

The work as it is done here makes extensive use of a caching decorator around core functions.
This is described more in the README.

## Core environment (`peint`)

```bash
conda create -n peint python=3.10 -y
conda activate peint

# 1. PyTorch first, matched to your CUDA version. See https://pytorch.org/get-started/locally/
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install einops==0.8.1

# 2. (Recommended) Flash Attention. Requires compute capability 8.0 (Ampere) or higher.
#    See "Installing Flash Attention" below if the default install fails.
pip install flash-attn==2.7.0.post2 --no-build-isolation

# 3. The package itself
pip install -e .            # core
pip install -e ".[train]"   # + wandb, for training
pip install -e ".[dev]"     # + pytest, black, ruff
```

Version pins are declared in `pyproject.toml`. For reference, these are the versions actually
installed in the environment the results were produced with (Python 3.10.19):

| Package | Version | Notes |
|---|---|---|
| `torch` | 2.5.0+cu124 | |
| `flash-attn` | 2.7.0.post2 | requires compute capability ≥ 8.0 |
| `einops` | 0.8.1 | |
| `numpy` | 2.2.6 | |
| `scipy` | 1.15.3 | |
| `pandas` | 2.3.3 | |
| `biopython` | 1.86 | |
| `fair-esm` | 2.0.0 | ESM2 |
| `cherryml` | 0.2.0 | `git+https://github.com/songlab-cal/CherryML`; brings in ete3 etc. |
| `ete3` | 3.1.3 | tree manipulation, via CherryML |
| `loguru` | 0.7.3 | |
| `joblib` | 1.5.3 | |
| `lightning` | 2.5.2 | training only |
| `wandb` | 0.28.1 | training only |

Flash Attention is strongly recommended for training and inference, but is not required: we also
provide non-Flash versions of the models (`PeintTransformerVanilla`, `use_flash=False`) that run
on any GPU or on CPU. These let you score transitions, simulate evolution, or train your own
models either way.

### External dependencies

**IQTree2** is what we use to simulate sequence evolution with the classical models WAG and LG
through [AliSim](https://academic.oup.com/bioinformatics/article/39/9/btad540/7258693). It's
provided as a submodule:

```bash
git submodule update --init iqtree2

cd iqtree2
mkdir build
cd build
cmake ..
make -j
cd ../..  # Return to main package directory

# verify installation
git submodule status
# Should show: a00094e03d1ae984e1497e16738f91514df8c366 iqtree2 (v2.3.4-212-ga00094e0)
```

## ESM-C environment (`peint-esmc`)

ESM-C checkpoints (`encoder_backbone="esmc"` / `"esmc-biohub"`) run the ESM-C 300M backbone through
HuggingFace `transformers` — specifically the **Biohub fork**, which registers the `esmc` model
type that mainline `transformers` does not have. This is the only extra requirement; the rest of
the stack (Python 3.10, torch 2.5.0, flash-attn 2.7.0.post2, fair-esm) is unchanged, so the
environment is just a clone of `peint` with `transformers` swapped out.

```bash
conda create --name peint-esmc --clone peint -y

# The Biohub transformers fork, pinned to a commit. --force-reinstall guarantees the fork
# replaces whatever transformers the clone inherited (pip can treat an equal version string as
# already satisfied and skip it); --no-deps keeps it from touching torch/numpy in the clone.
/path/to/conda/envs/peint-esmc/bin/pip install --force-reinstall --no-deps \
    "transformers @ git+https://github.com/Biohub/transformers.git@ef32577f55da19a4989cd7b22e004dc43a4998cb"

# The fork requires huggingface-hub<1.0 (mainline transformers 5.x pulls in hub 1.x, which the
# clone may have inherited). Quote the '<' so the shell does not read it as a redirect.
/path/to/conda/envs/peint-esmc/bin/pip install 'huggingface-hub<1.0'
```

Resulting versions — identical to `peint` except for these two:

| Package | `peint` | `peint-esmc` |
|---|---|---|
| `transformers` | 5.14.1 (mainline) | **4.57.6 (Biohub fork, `ef32577`)** |
| `huggingface_hub` | 1.27.0 | **0.36.2** (`<1.0`) |

With mainline `transformers`, loading ESM-C fails with:

```
ValueError: The checkpoint you are trying to load has model type `esmc` but Transformers does
not recognize this architecture.
```

### Hugging Face model download and cache

The backbone weights are pulled from the Hugging Face Hub repo **`biohub/ESMC-300M`** (~1.2 GB,
fp32) the first time an ESM-C model is built. Point `HF_HOME` at a directory with room for it —
on a cluster, somewhere on scratch rather than your home quota:

```bash
export HF_HOME=/path/to/hf_cache     # persistent cache for biohub/ESMC-300M
```

Set this in your shell profile or Slurm script; every ESM-C entry point (`load_peint_model`,
`figure2_ll_eval.py`, the VEP scripts) reads through it. After the first download you can run
fully offline with `HF_HUB_OFFLINE=1`. Do not use the deprecated `TRANSFORMERS_CACHE` — the fork
warns about it and it is removed in transformers v5.

If the machine has no outbound network, fetch the repo elsewhere and copy the cache directory
across, or pre-download it with:

```bash
HF_HOME=/path/to/hf_cache python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('biohub/ESMC-300M')"
```

### Why the fork, and not the `esm` package (namespace collision)

ESM-C is also distributed in EvolutionaryScale's `esm` package (esm3 / ESM Cambrian). **Do not
install it into a PEINT environment.** It installs a top-level module named `esm` — the exact
import name `fair-esm` uses — so the two cannot coexist: whichever is installed second wins, and
PEINT's ESM2 path (`import esm; esm.pretrained.esm2_t30_150M_UR50D`) breaks. It also currently
requires Python ≥ 3.12 and `torch>=2.11,<2.12`, which is incompatible with the Python 3.10 /
torch 2.5.0 / flash-attn 2.7.0.post2 stack above.

The Biohub fork sidesteps this entirely: its ESM-C implementation lives under the `transformers`
namespace (`transformers/models/esmc/`) and imports `esm` nowhere, so it coexists with `fair-esm`
and no PEINT code has to change. `peint/models/_esmc_biohub.py` is the only module that touches
it, and its `transformers` imports are lazy — ESM2-only users never load it.

The two builds have been verified equivalent (same tokenizer, bitwise-identical bf16 weights,
identical predictions); see `ESMC_BIOHUB_EQUIVALENCE_NOTES.md`.

### `transformer_engine` (optional)

On load, the fork prints a warning that `transformer_engine` is not installed and that it is
falling back to pure-PyTorch LayerNorm/MLP. **This is expected and safe to ignore.** Attention is
still Flash; only the fused fp32-reduction LayerNorm is unavailable. We tested with and without
`transformer_engine` and it changed no predictions (see `ESMC_BIOHUB_EQUIVALENCE_NOTES.md` §3), so
the environment is kept without it. To enable it anyway, with the CUDA toolkit on `PATH`:

```bash
pip install --no-build-isolation transformer-engine[pytorch]==1.13.0
```

### Verifying the ESM-C environment

```bash
conda activate peint-esmc
export HF_HOME=/path/to/hf_cache

python - <<'EOF'
import torch, transformers, esm                      # fork + fair-esm coexist
print("transformers:", transformers.__version__)     # 4.57.6 (fork)
from transformers import EsmForMaskedLM              # ESM2/vESM path still works on the fork

from peint.models._esmc_biohub import build_esmc_biohub_backbone
model, vocab, dim = build_esmc_biohub_backbone("esmc-biohub", use_flash=False)  # CPU-capable
print("embed_dim:", dim, "| vocab:", len(vocab))     # 960 | 33
ids = torch.tensor([[vocab.cls_idx] + vocab.encode("ACDEFGHIK") + [vocab.eos_idx]])
with torch.no_grad():
    out = model(ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
print(out.hidden_states[-1].shape, out.logits.shape)  # [1, 11, 960] [1, 11, 64]
EOF
```

Then load an actual ESM-C checkpoint — `load_peint_model` detects the backbone from the
checkpoint's hyperparameters and builds the ESM-C encoder automatically (no flag needed):

```python
from peint.models import load_peint_model
model, vocab = load_peint_model("model_checkpoints/peint_esmc.ckpt", device="cuda")
```

Two things to expect here, both harmless: an `Unexpected 308 keys in checkpoint` message (the
checkpoint ships the frozen backbone under `model.esm.*`, which equals the freshly built
`biohub/ESMC-300M` weights in bf16), and a `vocab` of length 33 against a 64-wide model head.

Note that `train_peint_model.py` is **ESM2-only** (`--esm_model` is restricted to the
`ESM2_REGISTRY` entries). The `peint-esmc` environment is for loading, scoring, and simulating
with ESM-C checkpoints.

## Installing Flash Attention (help)

This can be fairly straightforward, but we've run into issues in the past, with different wheels
built for different PyTorch/CUDA/OS combinations.

In some cases, you may get the following error when you try to use Flash Attention, even at import
time (following installation using the default `pip install flash-attn --no-build-isolation`
command):

```bash
import flash_attn
ImportError: /envs/peint/bin/torch/lib/python3.10/site-packages/torch/lib/../../../../libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /envs/peint/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)
```

This comes from torch and flash-attn being compiled against different versions of the C++
standard library (and from the base OS). It shows up, for instance, on Ubuntu 24.04 when the
default wheels were built around Ubuntu 22.04. There are several possible fixes; the one that
works reliably is to download a prebuilt wheel from the [Flash Attention releases](https://github.com/Dao-AILab/flash-attention/releases)
page. Find the version of CUDA, Torch, and OS that matches your setup.
The final consideration is `cxx11abiTRUE` vs `cxx11abiFALSE`. This is addressed
[in this GitHub issue](https://github.com/Dao-AILab/flash-attention/issues/457#issuecomment-1681544022).

All you need to do is check `torch._C._GLIBCXX_USE_CXX11_ABI` after importing torch, and then
download the appropriate wheel.

Then it's a matter of installing it with pip, e.g.:

```bash
pip install --no-dependencies <wheel_file>
```
