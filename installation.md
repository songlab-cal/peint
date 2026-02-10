# PEINT Installation Instructions

The core set of dependencies essentially works all the way back to Pytorch 2.2, while the benchmarking dependencies require Pytorch 2.5.0 or higher.

The work as it is done here makes extensive use of a caching decorator around core functions.
This is described more in the README.

## Core Dependencies

The core dependencies of this package are:

- `numpy ==1.26.4`
- `scipy==1.13.1`
- `pandas == 2.2.0`
- `biopython`
- `git+https://github.com/songlab-cal/CherryML`
- `torch==2.5.0`
- `flash-attn==2.7.0.post2` (Requires compute capability 8.0 or higher)
- `einops==0.8.1`
- `git+https://github.com/songlab-cal/CherryML` (this will bring in a fair number of additional dependencies - e.g. ete3, which is used for tree manipulation)
- `fair-esm` (ESM2)

if you want to train your own models, you will also need:

- `lightning==2.5.0.post0`
- `wandb`

These packages are required for the core functionality of the package. That said, we also provide non-flash attention versions of the models, which can be used in Flash-Attention's absence. That said, we recommend using Flash-Attention for training and inference, as it is significantly faster.
These will allow you to run the model to either score transitions, simulate evolution, or train your own models.

### External dependencies

1 - IQTree2

IQTree2 is what we use to simulate sequence evolution with the classical models WAG and LG through [AliSim](https://academic.oup.com/bioinformatics/article/39/9/btad540/7258693). It's provided as a submodule, so you can do the following:

```bash
git submodule update --init --recursive

cd iqtree2
mkdir build
cd build
cmake ..
make -j
cd ../..  # Return to main package directory

#verify installation
git submodule status
# Should show: 977cc4324234b36fbfb80b326b8e43b73952e365 iqtree2 (v2.3.4-190-g977cc432)
```

### Installing Flash Attention (Help)

This can be fairly straightforward, but we've run into issues in the past, with different wheels built for different PyTorch/CUDA/OS combinations.

In some cases, you may get the following error when you try to use flash attention, even at import time (following installation using the default `pip install flash-attn --no-build-isolation` command):

```bash
import flash_attn
ImportError: /envs/protevo/bin/torch/lib/python3.10/site-packages/torch/lib/../../../../libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /envs/protevo/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)
```

To the best of my knowledge, this comes from different versions of torch being compiled with different versions of the C++ std library (as well as your base OS). I ran into this issue when Ubuntu was upgraded to 24.04 from 22.04. The default wheels are built around Ubuntu 22.04.
There are many ways to potentially fix this, but the one that worked for me was to download a prebuilt wheel from the [flash attention releases](https://github.com/Dao-AILab/flash-attention/releases) page.
Find the version of CUDA, Torch, and OS that matches your setup.
The final consideration is `cxx11abiFALSE` vs `cxx11abiFALSE`.
This is addressed [in this github error](https://github.com/Dao-AILab/flash-attention/issues/457#issuecomment-1681544022).

All you need to do is check `torch._C._GLIBCXX_USE_CXX11_ABI` after importing torch, and then download the appropriate wheel.

Then it's a matter of installing it with pip, e.g.:

```bash
pip install --no-dependencies <wheel_file>
```