"""
Common utilities and classes for protein model training and fine-tuning.
"""

import os
import json
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from typing import List, Dict, Optional, Any
import lightning.pytorch as pl
from scipy.stats import spearmanr
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import Callback

from protevo.models.training import (
    ValidationLikelihoodCallback,
    GradNormCallback,
)

# Canonical encoder loader now lives in encoders.py (supports stock ESM2 + vESM).
# Re-exported here so existing `from protevo.vep._train_utils import load_esm_model`
# call sites (train_peint_vep, train_peint_vep_per_family, _vep_utils) keep working.
from protevo.vep.encoders import load_esm_model  # noqa: F401
from protevo.vep._config import EMBED_DIM
from protevo.vep._scoring import score_transition_pairs


class DMSEvaluationCallback(Callback):
    """
    A PyTorch Lightning callback that evaluates the model on DMS prediction tasks
    at specified intervals during training.
    """

    def __init__(
        self,
        dms_families: List[str],
        dms_transitions_dir: str,
        dms_labels_dir: str,
        output_dir: Optional[str] = None,
        batch_size: int = 32,
        t: float = 1.0,
        compute_every_n_epochs: int = 1,
        log_prefix: str = "dms",
        save_predictions: bool = False,
    ):
        """
        Args:
            dms_families: List of DMS families to evaluate
            dms_transitions_dir: Directory containing transitions for DMS variants
            dms_labels_dir: Directory containing DMS CSV files with labels
            output_dir: Directory to save prediction files (if save_predictions=True)
            batch_size: Batch size for inference
            t: Time parameter for the model
            compute_every_n_epochs: Compute metrics every N epochs
            log_prefix: Prefix for logging metrics
            save_predictions: Whether to save the prediction files
        """
        super().__init__()
        self.dms_families = dms_families
        self.dms_transitions_dir = dms_transitions_dir
        self.dms_labels_dir = dms_labels_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.t = t
        self.compute_every_n_epochs = compute_every_n_epochs
        self.log_prefix = log_prefix
        self.save_predictions = save_predictions

        # Create output directory if saving predictions
        if self.save_predictions and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # Cache for DMS data to avoid reloading files
        self.dms_data_cache = {}

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """
        Compute DMS metrics at the end of validation epochs (at specified intervals).
        """
        # Only compute every N epochs
        if (trainer.current_epoch + 1) % self.compute_every_n_epochs != 0:
            return

        # Get model and device
        model = pl_module
        device = next(model.parameters()).device
        vocab = model.model.vocab

        # Prepare time value
        t_value = self.t

        # Dictionary to store metrics for all families
        all_metrics = {}

        # Evaluate each family
        for family in self.dms_families:
            # Compute predictions
            scores, success = self._compute_predictions_for_family(
                trainer=trainer,
                family=family,
                model=model,
                vocab=vocab,
                device=device,
                t_value=t_value,
            )

            if not success:
                continue

            # Compute metrics
            metrics = self._compute_metrics_for_family(family, scores)

            # Add metrics to overall dict
            for metric_name, value in metrics.items():
                all_metrics[f"{self.log_prefix}/{family}/{metric_name}"] = value

        # Compute average metrics across families if we have more than 1 family
        avg_metrics = {}
        metric_types = ["spearman"]
        if len(self.dms_families) > 1:
            for metric_type in metric_types:
                metric_values = [
                    value
                    for key, value in all_metrics.items()
                    if key.endswith(f"/{metric_type}") and not np.isnan(value)
                ]

                if metric_values:
                    avg_metrics[f"{self.log_prefix}/avg/{metric_type}"] = np.mean(
                        metric_values
                    )

        # Log all metrics
        for metric_name, value in {**all_metrics, **avg_metrics}.items():
            if not np.isnan(value):
                trainer.logger.log_metrics(
                    {metric_name: value}, step=trainer.global_step
                )
                # Also expose via callback_metrics so ModelCheckpoint / EarlyStopping can
                # monitor them (logger.log_metrics alone does not populate callback_metrics).
                trainer.callback_metrics[metric_name] = torch.tensor(float(value))

        # Print average metrics
        print("\n" + "=" * 50)
        print(f"DMS Evaluation (Epoch {trainer.current_epoch}):")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("=" * 50 + "\n")

    def _compute_predictions_for_family(
        self,
        trainer,
        family: str,
        model: pl.LightningModule,
        vocab: Any,
        device: torch.device,
        t_value: float,
    ) -> tuple:
        """
        Compute model predictions (log likelihoods) for a DMS family.

        Returns:
            Tuple of (scores, success_flag)
        """
        # Check if transitions file exists
        transitions_file = os.path.join(self.dms_transitions_dir, f"{family}.txt")
        if not os.path.exists(transitions_file):
            print(f"Transitions for {family} does not exist, skipping")
            return None, False

        # Load transitions
        pairs = []
        with open(transitions_file, "r") as fin:
            for l in fin:
                pairs.append(l.strip().split())

        # Check sequence length
        if len(pairs[0][0]) > 1022:
            print(f"{family} sequences are too long (>{1022}), skipping")
            return None, False

        # Compute predictions in batches (shared scoring loop, see _scoring.py)
        scores = score_transition_pairs(
            model, vocab, device, pairs, t_value, batch_size=self.batch_size
        )

        # Save predictions if requested
        if self.save_predictions and self.output_dir:
            family_output_dir = os.path.join(
                self.output_dir, f"epoch_{trainer.current_epoch}"
            )
            os.makedirs(family_output_dir, exist_ok=True)

            output_path = os.path.join(family_output_dir, f"{family}_preds.txt")
            with open(output_path, "w") as fout:
                for score in scores:
                    fout.write(f"{score}\n")

        return scores, True

    def _compute_metrics_for_family(
        self, family: str, scores: List[float]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for a family given the model scores.

        Args:
            family: DMS family ID
            scores: List of model scores (log likelihoods)

        Returns:
            Dictionary of metrics
        """
        # Load DMS data
        dms_data = self._load_dms_data(family)
        if dms_data is None:
            return {}

        # Assign scores to DMS data
        dms_data_with_scores = dms_data.copy()
        dms_data_with_scores["model_score"] = scores

        # Get target scores (ground truth)
        target_scores = (
            dms_data_with_scores["DMS_score"].values
            if "DMS_score" in dms_data_with_scores.columns
            else dms_data_with_scores["DMS_bin_score"].values
        )

        # Get model scores
        model_scores = dms_data_with_scores["model_score"].values

        # Calculate metrics
        metrics = {}

        # Spearman correlation
        spearman_corr, p_value = spearmanr(target_scores, model_scores)
        metrics["spearman"] = spearman_corr

        return metrics

    def _load_dms_data(self, family: str) -> Optional[pd.DataFrame]:
        """
        Load DMS data for a family with caching.
        """
        # Check if already cached
        if family in self.dms_data_cache:
            return self.dms_data_cache[family]

        # Find the corresponding DMS file
        dms_files = os.listdir(self.dms_labels_dir)
        matching_files = [
            f for f in dms_files if f.startswith(family) and f.endswith(".csv")
        ]

        if not matching_files:
            print(f"No DMS label file found for {family}, skipping")
            return None

        # Load the file
        dms_file_path = os.path.join(self.dms_labels_dir, matching_files[0])
        try:
            dms_data = pd.read_csv(dms_file_path, low_memory=False)
            # Cache for future use
            self.dms_data_cache[family] = dms_data
            return dms_data
        except Exception as e:
            print(f"Error loading DMS data for {family}: {e}")
            return None


class CombinedDataset(torch.utils.data.Dataset):
    """
    Dataset that combines original and fine-tuning datasets.
    """

    def __init__(self, original_dataset, finetune_dataset):
        self.finetune_dataset = finetune_dataset
        self.original_dataset = original_dataset
        self.finetune_length = len(finetune_dataset) if finetune_dataset else 0
        self.original_length = len(original_dataset) if original_dataset else 0

    def __getitem__(self, idx):
        if idx < self.finetune_length:
            return self.finetune_dataset[idx]
        else:
            orig_idx = idx - self.finetune_length
            return self.original_dataset[orig_idx]

    def __len__(self):
        return self.finetune_length + self.original_length


class MixedDatasetSampler(torch.utils.data.Sampler):
    """
    Sampler that yields batches of indices for a mixed dataset with original and fine-tuning data.
    """

    def __init__(
        self,
        original_length,
        finetune_length,
        batch_size,
        finetune_ratio=0.5,
        shuffle=True,
    ):
        self.original_length = original_length
        self.finetune_length = finetune_length
        self.batch_size = batch_size
        self.finetune_ratio = finetune_ratio
        self.shuffle = shuffle

        # Calculate number of samples from each dataset per batch
        self.ft_samples = max(1, int(batch_size * finetune_ratio))
        self.orig_samples = batch_size - self.ft_samples

        # Calculate total batches based on finetune dataset
        self.batches_per_epoch = (
            finetune_length + self.ft_samples - 1
        ) // self.ft_samples

    def __iter__(self):
        # For finetune dataset, create indices
        ft_indices = list(range(self.finetune_length))

        # For original dataset, create indices
        orig_indices = list(range(self.original_length))

        if self.shuffle:
            # Shuffle both index lists
            ft_indices = torch.randperm(self.finetune_length).tolist()
            orig_indices = torch.randperm(self.original_length).tolist()

        # Generate indices for dataset
        current_ft_pos = 0
        current_orig_pos = 0

        ft_base_offset = 0  # Finetune data starts at index 0
        orig_base_offset = (
            self.finetune_length
        )  # Original data starts after finetune data

        # Loop through each batch
        for _ in range(self.batches_per_epoch):
            batch_indices = []

            # Get fine-tuning indices
            ft_batch_size = min(self.ft_samples, self.finetune_length - current_ft_pos)

            # Add fine-tuning indices to batch
            for i in range(ft_batch_size):
                batch_indices.append(ft_base_offset + ft_indices[current_ft_pos + i])

            current_ft_pos += ft_batch_size
            if current_ft_pos >= self.finetune_length:
                # Reset if we've gone through all fine-tuning data
                current_ft_pos = 0
                if self.shuffle:
                    ft_indices = torch.randperm(self.finetune_length).tolist()

            # Fill remaining batch with original dataset indices
            orig_needed = self.batch_size - len(batch_indices)
            for _ in range(orig_needed):
                # Wrap around if needed
                if current_orig_pos >= self.original_length:
                    current_orig_pos = 0
                    if self.shuffle:
                        orig_indices = torch.randperm(self.original_length).tolist()

                batch_indices.append(orig_base_offset + orig_indices[current_orig_pos])
                current_orig_pos += 1

            # Yield the full batch of indices
            yield batch_indices

    def __len__(self):
        return self.batches_per_epoch


class ProtevoMixedDataModule(pl.LightningDataModule):
    """Data module that efficiently mixes original and fine-tuning data."""

    def __init__(
        self,
        finetune_data_module,
        original_data_module=None,
        finetune_ratio=0.5,
        batch_size=32,
        num_workers=0,
        **kwargs,
    ):
        super().__init__()
        self.original_dm = original_data_module
        self.finetune_dm = finetune_data_module
        self.finetune_ratio = finetune_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Set up data modules
        if self.original_dm:
            self.original_dm.setup(stage)
        self.finetune_dm.setup(stage)

        # Get train datasets
        self.orig_train = getattr(self.original_dm, "train_dataset", None)
        self.ft_train = getattr(self.finetune_dm, "train_dataset", None)

        orig_size = len(self.orig_train) if self.orig_train else 0
        ft_size = len(self.ft_train) if self.ft_train else 0
        print(f"Original training dataset size: {orig_size}")
        print(f"Fine-tuning training dataset size: {ft_size}")

        # Validate datasets
        if (not self.orig_train or orig_size == 0) and (
            not self.ft_train or ft_size == 0
        ):
            raise ValueError("Both original and fine-tuning datasets are empty")

    def train_dataloader(self):
        # Get datasets
        orig_train = self.orig_train
        ft_train = self.ft_train

        # Handle single dataset cases
        if not orig_train or len(orig_train) == 0:
            print("Using only fine-tuning dataset")
            return DataLoader(
                ft_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=getattr(self.finetune_dm, "collate_fn", None),
                persistent_workers=True if self.num_workers > 0 else False,
            )
        elif not ft_train or len(ft_train) == 0:
            print("Using only original dataset")
            return DataLoader(
                orig_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=getattr(self.original_dm, "collate_fn", None),
                persistent_workers=True if self.num_workers > 0 else False,
            )

        # Create combined dataset
        combined_dataset = CombinedDataset(orig_train, ft_train)

        # Create custom sampler that yields batches of indices
        sampler = MixedDatasetSampler(
            original_length=len(orig_train),
            finetune_length=len(ft_train),
            batch_size=self.batch_size,
            finetune_ratio=self.finetune_ratio,
            shuffle=True,
        )

        # The key change: use batch_sampler instead of sampler
        return DataLoader(
            combined_dataset,
            batch_sampler=sampler,  # Use batch_sampler instead of sampler + batch_size=None
            num_workers=self.num_workers,
            collate_fn=getattr(self.finetune_dm, "collate_fn", None),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return self.finetune_dm.val_dataloader()

    def test_dataloader(self):
        return self.finetune_dm.test_dataloader()


def setup_args():
    """
    Set up and return the argument parser.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Original arguments
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument(
        "--families_file",
        type=str,
        help="Path to json file containing family information",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save model checkpoints"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--max_seq_len", type=int, default=1022, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_heads", type=int, default=20, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=5,
        help="Number of encoder transformer layers",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=5,
        help="Number of decoder transformer layers",
    )
    parser.add_argument("--embed_dim", type=int, default=640, help="Embedding size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--n_families", type=int, default=-1, help="Number of families to train on"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=2000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--devices", type=int, nargs="+", default=None, help="GPU devices to use"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Maximum number of steps to train for"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Save checkpoint every n steps (used when DMS eval is off)",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="Keep this many checkpoints: best by DMS Spearman if eval is on, else most recent",
    )
    parser.add_argument(
        "--check_val_every_n_epoch", type=int, default=1, help="Validate every n epochs"
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to model checkpoint to resume training",
    )
    parser.add_argument(
        "--dms_transitions_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to DMS transitions",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--use_attention_bias", action="store_true", help="Use attention bias"
    )
    parser.add_argument(
        "--dropout_p", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=0.1,
        help="Gradient clip value (default = 0.1)",
    )
    parser.add_argument(
        "--name_addon",
        type=str,
        nargs="?",
        default=None,
        help="additional name arguments",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.0, help="Label smoothing"
    )
    parser.add_argument(
        "--rate_matrix_path", type=str, nargs="?", help="Path to rate matrix"
    )

    # Fine-tuning specific arguments
    parser.add_argument(
        "--finetune_families_file",
        type=str,
        default=None,
        help="Path to json file containing fine-tuning family information",
    )
    parser.add_argument(
        "--finetune_data_path",
        type=str,
        default=None,
        help="Directory containing fine-tuning data",
    )
    parser.add_argument(
        "--finetune_mix_ratio",
        type=float,
        default=0.5,
        help="Ratio of fine-tuning data to mix into each batch (0-1)",
    )
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for resuming/starting fine-tuning (more specific than resume_path)",
    )

    # DMS evaluation arguments
    parser.add_argument(
        "--eval_dms", action="store_true", help="Enable DMS evaluation during training"
    )
    parser.add_argument(
        "--eval_dms_families_file",
        type=str,
        default=None,
        help="Path to json file containing the families for DMS evaluation",
    )
    parser.add_argument(
        "--eval_dms_transitions_dir",
        type=str,
        default=None,
        help="Directory containing DMS transitions",
    )
    parser.add_argument(
        "--eval_dms_labels_dir",
        type=str,
        default=None,
        help="Directory containing DMS label CSV files",
    )
    parser.add_argument(
        "--eval_dms_output_dir",
        type=str,
        default=None,
        help="Directory to save DMS prediction files",
    )
    parser.add_argument(
        "--eval_dms_batch_size",
        type=int,
        default=32,
        help="Batch size for DMS evaluation",
    )
    parser.add_argument(
        "--eval_dms_time",
        type=float,
        default=1.0,
        help="Time parameter for DMS evaluation",
    )
    parser.add_argument(
        "--eval_dms_every_n_epochs",
        type=int,
        default=1,
        help="Evaluate DMS every N epochs",
    )
    parser.add_argument(
        "--eval_dms_save_predictions",
        action="store_true",
        help="Save DMS predictions to files",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping based on Spearman correlation",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.01,
        help="Minimum change in the monitored metric to qualify as an improvement",
    )
    parser.add_argument(
        "--which_esm",
        type=str,
        default="150M",
        help="Which frozen encoder to use (stock ESM2 sizes or vESM)",
        choices=list(EMBED_DIM),
    )

    return parser


def validate_args(args):
    """
    Validate command line arguments.
    """
    if args.seed is None and (args.devices is not None and len(args.devices) > 1):
        raise ValueError("Must set seed when using multiple GPUs")

    if args.label_smoothing > 0.0 and args.rate_matrix_path is None:
        raise ValueError("Must provide rate matrix path when using label smoothing")

    # Basic validation for fine-tuning
    if args.finetune_families_file is not None and args.finetune_data_path is None:
        raise ValueError("Must provide finetune_data_path with finetune_families_file")

    # Validate DMS evaluation arguments if requested
    if args.eval_dms:
        if not args.eval_dms_families_file:
            raise ValueError(
                "Must provide eval_dms_families_file when eval_dms is enabled"
            )
        if not args.eval_dms_transitions_dir:
            raise ValueError(
                "Must provide eval_dms_transitions_dir when eval_dms is enabled"
            )
        if not args.eval_dms_labels_dir:
            raise ValueError(
                "Must provide eval_dms_labels_dir when eval_dms is enabled"
            )
        if args.eval_dms_save_predictions and not args.eval_dms_output_dir:
            raise ValueError(
                "Must provide eval_dms_output_dir when eval_dms_save_predictions is enabled"
            )

    if args.which_esm not in EMBED_DIM:
        raise ValueError(f"Invalid encoder: {args.which_esm}")
    args.embed_dim = EMBED_DIM[args.which_esm]


def setup_run_name(args):
    """
    Create a run name based on the current date/time and settings.
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    run_name = f"{date_str}_{time_str}"

    # Add fine-tuning info to run name if applicable
    if args.finetune_families_file:
        with open(args.finetune_families_file, "r") as f:
            finetune_families = json.load(f)["families"]
        run_name += f"-ft_{len(finetune_families)}fams"

    if args.finetune_mix_ratio < 1:
        run_name += f"-mix{args.finetune_mix_ratio}"

    if args.which_esm != "150M":
        run_name += f"-esm2_{args.which_esm}"

    if args.name_addon:
        run_name = run_name + "-" + args.name_addon

    return run_name


def setup_model(args, flash_esm_model, esm_vocab, model_args):
    """
    Set up the model, either creating a new one or loading from a checkpoint.
    """
    from protevo.models.training import PeintLightningModule

    if args.resume_path:
        print(f"Loading model from checkpoint: {args.resume_path}")
        model = PeintLightningModule.load_from_checkpoint(
            args.resume_path,
            esm_model=flash_esm_model,
            esm_vocab=esm_vocab,
            **model_args,
        )
    else:
        model = PeintLightningModule(
            esm_model=flash_esm_model, esm_vocab=esm_vocab, **model_args
        )

    return model


def setup_callbacks(args, output_path):
    """
    Set up the callbacks for training.

    Checkpointing keeps the best ``--save_top_k`` checkpoints by DMS Spearman when DMS
    evaluation is enabled (plus a ``last.ckpt`` safety net); otherwise it keeps the most
    recent ``--save_top_k`` step checkpoints. The frozen encoder weights are saved inside
    each checkpoint (``PeintLightningModule`` does not strip them), so ``load_model`` rebuilds
    the head with ``strict=False`` regardless.
    """
    import lightning as pl

    callbacks = []

    # DMS evaluation (added first so its avg Spearman is in callback_metrics before
    # checkpoint selection). For per-family finetuning the callback is added by the caller.
    dms_enabled = args.eval_dms and getattr(args, "finetune_mode", None) != "per_family"
    dms_families = []
    if dms_enabled:
        with open(args.eval_dms_families_file, "r") as f:
            dms_families = json.load(f)["families"]
        callbacks.append(
            DMSEvaluationCallback(
                dms_families=dms_families,
                dms_transitions_dir=args.eval_dms_transitions_dir,
                dms_labels_dir=args.eval_dms_labels_dir,
                output_dir=(
                    args.eval_dms_output_dir if args.eval_dms_save_predictions else None
                ),
                batch_size=args.eval_dms_batch_size,
                t=args.eval_dms_time,
                compute_every_n_epochs=args.eval_dms_every_n_epochs,
                save_predictions=args.eval_dms_save_predictions,
            )
        )
        print(f"Added DMS evaluation callback for {len(dms_families)} families")

    # Checkpoint: best-k by DMS Spearman when available, else most-recent-k by step.
    if dms_enabled and len(dms_families) > 1:
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_path,
            monitor="dms/avg/spearman",
            mode="max",
            save_top_k=args.save_top_k,
            save_last=True,
        )
    else:
        checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=output_path,
            save_top_k=args.save_top_k,
            every_n_train_steps=args.checkpoint_every,
            save_last=True,
        )

    callbacks += [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
        ValidationLikelihoodCallback(),
        GradNormCallback(),
    ]

    return callbacks


def setup_trainer(args, logger, callbacks):
    """
    Set up the PyTorch Lightning trainer.
    """
    import lightning as pl

    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    trainer = pl.Trainer(
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.devices,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        detect_anomaly=False,
        gradient_clip_val=args.grad_clip,
        precision="bf16",
    )

    return trainer


def prepare_model_args(args):
    """
    Prepare the model arguments dictionary.
    """
    model_args = {
        "max_seq_len": args.max_seq_len,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "lr": args.lr,
        "num_warmup_steps": args.num_warmup_steps,
        "num_training_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "use_attention_bias": args.use_attention_bias,
        "dropout_p": args.dropout_p,
        "which_esm": args.which_esm,  # saved in hparams so load_model rebuilds the right encoder
    }

    if args.label_smoothing > 0.0:
        model_args.update(
            {
                "label_smoothing": args.label_smoothing,
                "rate_matrix_path": args.rate_matrix_path,
            }
        )

    return model_args


def load_families(args):
    """
    Load the original and fine-tuning families.
    """
    # Load original families
    if args.families_file:
        with open(args.families_file, "r") as f:
            families = json.load(f)["families"]

        if args.n_families == -1:
            families = families
        else:
            assert args.n_families <= len(
                families
            ), "n_families must be less than or equal to the number of families"
            np.random.shuffle(families)
            families = families[: args.n_families]
    else:
        families = []

    # Load fine-tuning families if specified
    finetune_families = []
    if args.finetune_families_file:
        with open(args.finetune_families_file, "r") as f:
            finetune_families = json.load(f)["families"]

    return families, finetune_families
