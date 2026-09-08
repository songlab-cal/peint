from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import logging
import sys
import numpy as np
import pandas as pd
import torch
from esm.data import Alphabet
from peint.models._flash_esm import ESM2Flash, ESM2Model
from peint.models._transformer import (
    PeintTransformer,
    PeintTransformerVanilla,
    PeintGenerator,
)
from peint.vep.encoders import load_esm_model

# Paths (single source of truth in _config; re-exported here for backward compat).
from peint.vep._config import (  # noqa: E402,F401
    MAIN_DIR,
    DATA_DIR,
    VEP_DATA_DIR,
    PROTEINGYM_DIR,
    CHECKPOINTS_DIR,
    TEST_TRANSITIONS_DIR,
    EMBED_DIM_TO_ESM,
)

# Module logger
logger = logging.getLogger(__name__)


# ESM2 attention-head count keyed by embedding dim. Stock ESM2 and vESM are
# architecturally identical per size, so this suffices to rebuild the encoder scaffold
# from a checkpoint (num_layers and embed_dim are read directly from the saved weights).
_ESM_ATTENTION_HEADS_BY_EMBED_DIM = {640: 20, 1280: 20, 2560: 40, 5120: 40}


def _infer_esm_arch_from_state_dict(state_dict) -> Tuple[int, int, int]:
    """Infer (num_layers, embed_dim, attention_heads) of the frozen ESM2 encoder from a
    PEINT checkpoint. The encoder weights are stored under the ``model.esm.`` prefix."""
    layer_idxs = [
        int(k.split(".")[3]) for k in state_dict if k.startswith("model.esm.layers.")
    ]
    if not layer_idxs or "model.esm.embed_tokens.weight" not in state_dict:
        raise ValueError(
            "Checkpoint has no 'model.esm.*' encoder weights, so the frozen encoder "
            "cannot be rebuilt from the checkpoint alone."
        )
    num_layers = max(layer_idxs) + 1
    embed_dim = state_dict["model.esm.embed_tokens.weight"].shape[1]
    if embed_dim not in _ESM_ATTENTION_HEADS_BY_EMBED_DIM:
        raise ValueError(f"Unrecognized ESM embed_dim in checkpoint: {embed_dim}")
    return num_layers, embed_dim, _ESM_ATTENTION_HEADS_BY_EMBED_DIM[embed_dim]


def load_model(
    model_checkpoint_path: str,
    device: torch.device,
    model_type: str = "PeintTransformer",
    use_flash: bool = True,
    **kwargs,
):
    """Load a scoring-ready PEINT model from a VEP checkpoint (CPU or GPU).

    Handles both checkpoint layouts, without instantiating a Lightning wrapper:

    * **Full** checkpoint (frozen encoder saved under ``model.esm.*``): the ESM2 architecture
      is inferred from the saved weights and encoder + head are loaded together — no
      ``which_esm``, no ESM/vESM download, fully offline. Identical for stock ESM2 and vESM.
    * **Stripped** ("PEINT-only") checkpoint: the frozen encoder is rebuilt from ``which_esm``
      (falling back to ``embed_dim`` for stock sizes) via the encoder registry, then the
      trainable head is loaded with ``strict=False``. This is what the distributed
      checkpoints are.

    ``use_flash=True`` (default) builds the FlashAttention stack (``ESM2Flash`` +
    ``PeintTransformer``), which needs a GPU. Set ``use_flash=False`` to build the standard
    PyTorch stack (``ESM2Model`` + ``PeintTransformerVanilla``) for CPU inference.
    """
    sd = torch.load(model_checkpoint_path, map_location="cpu")
    state_dict = sd["state_dict"]
    model_hparams = dict(sd["hyper_parameters"])
    model_hparams.update(kwargs)

    # Biohub ESM-C selector. Accept both the VEP name ("esmc-biohub") and the plain
    # "esmc" used by EvolutionaryScale's package: those checkpoints carry the
    # same ESMC-300M weights (verified bit-identical in bf16), so both rebuild the biohub
    # transformers backbone. Its saved encoder weights load with strict=False below; the
    # ESM2 arch-inference path does not apply.
    _esmc_backbone_names = ("esmc", "esmc-biohub")
    if (model_hparams.get("which_esm") in _esmc_backbone_names
            or model_hparams.get("encoder_backbone") in _esmc_backbone_names):
        esm_model, vocab = load_esm_model("esmc-biohub", use_flash=use_flash)
    elif any(k.startswith("model.esm.") for k in state_dict):
        # Full checkpoint: rebuild the scaffold from the checkpoint's own encoder weights.
        num_layers, embed_dim, attention_heads = _infer_esm_arch_from_state_dict(state_dict)
        esm_cls = ESM2Flash if use_flash else ESM2Model
        esm_model = esm_cls(
            num_layers=num_layers,
            embed_dim=embed_dim,
            attention_heads=attention_heads,
            alphabet="ESM-1b",
            token_dropout=True,
            dropout_p=0.0,
        )
        vocab = Alphabet.from_architecture("ESM-1b")
    else:
        # Stripped checkpoint: rebuild the frozen encoder from which_esm / embed_dim.
        which_esm = model_hparams.get("which_esm")
        if not which_esm:
            embed_dim = model_hparams["embed_dim"]
            if embed_dim not in EMBED_DIM_TO_ESM:
                raise ValueError(
                    f"Stripped checkpoint with embed_dim={embed_dim} and no 'which_esm' "
                    f"hparam — cannot infer which encoder to rebuild."
                )
            which_esm = EMBED_DIM_TO_ESM[embed_dim]
        esm_model, vocab = load_esm_model(which_esm, use_flash=use_flash)

    # PeintLightningModule stored max_seq_len + optimizer-only hparams; map/drop them for
    # the plain module constructor.
    model_kwargs = dict(model_hparams)
    if "max_seq_len" in model_kwargs:
        model_kwargs["max_len"] = model_kwargs.pop("max_seq_len")
    for k in ("lr", "num_warmup_steps", "num_training_steps", "which_esm"):
        model_kwargs.pop(k, None)

    if use_flash:
        model_cls = PeintGenerator if model_type == "Cached_Transformer" else PeintTransformer
    else:
        # Cached (generator/evaluator) variants are flash-only; CPU uses the vanilla model.
        model_cls = PeintTransformerVanilla
    model = model_cls(esm_model=esm_model, esm_vocab=vocab, **model_kwargs)

    # The checkpoint keys are prefixed with the LightningModule's "model." attribute; strip
    # it. Full checkpoints load encoder + head; for stripped checkpoints the frozen
    # encoder/embedding/lm_head keep the values PeintTransformer copied from the ESM model
    # (they are legitimately absent from the checkpoint). strict=False covers both.
    renamed = {
        k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
    }
    result = model.load_state_dict(renamed, strict=False)
    expected_absent = ("esm.", "embedding.", "lm_head.", "rot_emb", "rotary", "inv_freq")
    missing = [k for k in result.missing_keys if not any(b in k for b in expected_absent)]
    unexpected = [
        k for k in result.unexpected_keys if not any(b in k for b in expected_absent)
    ]
    if missing or unexpected:
        logger.warning(
            "load_model: missing=%s unexpected=%s", missing[:6], unexpected[:6]
        )

    return model.eval().to(device), vocab


# NOTE: _set_publication_style moved to peint.vep.plotting._style


def _ensure_stdout_logging(level: int = logging.INFO) -> None:
    """Attach a StreamHandler to stdout if no handlers exist for this logger.

    Avoids duplicate handlers across repeated calls.
    """
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


def _format_time_dir_suffix(t: float) -> str:
    """Helper to create the directory time suffix used during inference.

    For example, t=0.7 -> "t_0_7", t=0.05 -> "t_0_05".
    """
    return f"t_{str(t).replace('.', '_')}"


def _parse_time_from_dirname(dirname: str, run_prefix: str) -> Optional[float]:
    """Parse time t from a directory name of the form f"{run_prefix}-t_x_y".

    Returns float time or None if the directory does not match the pattern.
    """
    prefix = f"{run_prefix}-t_"
    if not dirname.startswith(prefix):
        return None
    tail = dirname[len(prefix) :]
    # Accept additional suffixes after time (e.g., none by default)
    # Extract up to the next '-' if present
    time_token = tail.split("-")[0]
    try:
        return float(time_token.replace("_", "."))
    except ValueError:
        return None


def _load_spearman_results_across_times(
    output_root: str | Path,
    run_prefix: str,
    times: list[float],
):
    """Load and aggregate Spearman results for a single checkpoint across times.

    Returns a DataFrame with columns ["time", "assay_type", "spearman"].
    """
    output_root = Path(output_root)
    dfs_summary = []
    for t in times:
        t_suffix = _format_time_dir_suffix(t)
        run_dir = output_root / f"{run_prefix}-{t_suffix}"
        results_path = run_dir / "spearman_results.csv"
        if not results_path.exists():
            logger.info(
                f"[load_spearman_results_across_times] Missing results for t={t} ({results_path}), skipping."
            )
            continue
        df = pd.read_csv(results_path)
        if "assay_type" not in df.columns or "spearman" not in df.columns:
            logger.info(
                f"[load_spearman_results_across_times] Unexpected format in {results_path}, skipping."
            )
            continue
        df_summary = (
            df.groupby("assay_type", as_index=False)["spearman"].mean().assign(time=t)
        )
        dfs_summary.append(df_summary)
    if not dfs_summary:
        raise FileNotFoundError(
            "No valid spearman_results.csv files found for any of the specified time points."
        )
    df_all = pd.concat(dfs_summary, ignore_index=True)
    return df_all


def _discover_time_dirs(
    output_root: Path, run_prefix: str, times: Optional[List[float]]
) -> List[Tuple[float, Path]]:
    """Discover time-specific result directories for a given run_prefix.

    If `times` is provided, only return those; otherwise, auto-discover all.
    Returns a list of (t, path) sorted by t ascending.
    """
    output_root = Path(output_root)
    time_dirs: List[Tuple[float, Path]] = []

    if times is not None:
        for t in times:
            t_suffix = _format_time_dir_suffix(t)
            d = output_root / f"{run_prefix}-{t_suffix}"
            if d.exists() and d.is_dir():
                time_dirs.append((float(t), d))
    else:
        for entry in os.listdir(output_root):
            t_val = _parse_time_from_dirname(entry, run_prefix)
            if t_val is None:
                continue
            d = output_root / entry
            if d.is_dir():
                time_dirs.append((t_val, d))

    time_dirs.sort(key=lambda x: x[0])
    return time_dirs


def _list_dataset_pred_files(time_dir: Path) -> List[str]:
    """List dataset prediction filenames (ending with _preds.txt) in a time dir."""
    return sorted([f for f in os.listdir(time_dir) if f.endswith("_preds.txt")])


def _load_preds_file(preds_path: Path) -> np.ndarray:
    """Load a preds text file as a 1D numpy array of floats."""
    with open(preds_path, "r") as fin:
        values = [float(l.rstrip("\n")) for l in fin]
    return np.asarray(values, dtype=float)


def _write_preds_file(path: Path, scores: np.ndarray) -> None:
    """Write one score per line to path, preserving simple text format."""
    with open(path, "w") as fout:
        for v in scores:
            fout.write(f"{float(v)}\n")


def _compute_mode_with_tie_break(values: List[float]) -> float:
    """Compute mode; break ties by selecting the smallest value."""
    counts: Dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values()) if counts else 0
    candidates = [t for t, c in counts.items() if c == max_count]
    return min(candidates) if candidates else float("nan")
