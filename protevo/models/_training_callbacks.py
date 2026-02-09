"""Training callbacks for PEINT models.

This module contains PyTorch Lightning callbacks for monitoring
and logging during PEINT model training.

Requires: lightning, wandb
"""

from typing import Any

import numpy as np
import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb

from protevo.utils import get_quantization_points_from_geometric_grid
from protevo.models._model_utils import get_quantile_idx, gradient_norm

VALID_BINS = np.array([float(f) for f in get_quantization_points_from_geometric_grid()])


class ValidationLikelihoodCallback(Callback):
    """Aggregates validation likelihoods into time bins for visualization.

    The actual likelihoods are calculated in the model's validation_step method.
    This callback bins them by evolutionary time and logs to wandb.
    """

    def __init__(self):
        super().__init__()
        self.ppl_per_bin = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, pplbin = outputs
        for k, v in pplbin.items():
            bin_idx = get_quantile_idx(VALID_BINS, k)
            if bin_idx not in self.ppl_per_bin:
                self.ppl_per_bin[bin_idx] = []
            self.ppl_per_bin[bin_idx].extend(v)

    def on_validation_epoch_end(self, trainer, pl_module):
        arrmeans = np.zeros(len(VALID_BINS))
        for k, v in self.ppl_per_bin.items():
            arrmeans[k] = np.exp(np.mean(v))

        nonzero = np.nonzero(arrmeans)[0]
        xs = VALID_BINS[nonzero]
        ys = arrmeans[nonzero]

        data = [[x, y] for x, y in zip(xs, ys)]

        table = wandb.Table(data=data, columns=["Time", "Mean per-site Likelihood"])

        # important for multi-gpu runs, doesn't seem to affect single gpu runs
        if trainer.global_rank == 0:
            trainer.logger.experiment.log(
                {
                    "time_bin_likelihood": wandb.plot.scatter(
                        table,
                        "Time",
                        "Mean per-site Likelihood",
                        title="Loglikelihood per time bin",
                    )
                }
            )

        # clear for next epoch
        self.ppl_per_bin = {}


class GradNormCallback(Callback):
    """Logs the gradient norm during training."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        trainer.logger.experiment.log(
            {"my_model/grad_norm": gradient_norm(pl_module)}
        )
