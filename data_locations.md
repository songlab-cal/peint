# Data locations

No dataset paths are hard-coded in this repository. Every entry point takes the data it needs as
an argument, so point them at wherever you have unpacked the data.

| what | how it is supplied |
|---|---|
| VEP transitions (1,000 transitions per family) | `--data_path` on the VEP scripts under `peint/vep/` |
| Model checkpoints | `--checkpoint` / `load_peint_model(...)`; see `README.md` |
| WAG rate matrix | ships here as `data/rate_matrices/wag.txt`, and is the default |
| Evaluation and analysis outputs | `--out_path` on the script that writes them |

The published data are deposited on Zenodo with the paper, and the companion `peint-paper`
repository automates fetching and unpacking them (`scripts/fetch_local_data.py`); its
`installation.md` is the tested recipe.
