"""PyTorch Lightning training module for PEINT models.

This module contains the Lightning wrapper for training PEINT models.

Requires: lightning
"""

import torch
import torch.nn.functional as F
import lightning as pl

from protevo.models._transformer import PeintTransformer
from protevo.models._optimization import get_polynomial_decay_schedule_with_warmup


class PeintLightningModule(pl.LightningModule):
    """PyTorch Lightning module for training PEINT models."""

    def __init__(
        self,
        esm_model,
        esm_vocab,
        max_seq_len: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embed_dim: int,
        lr: float = 1e-4,
        num_warmup_steps: int = 10000,
        num_training_steps: int = 100000,
        **kwargs,
    ):
        super().__init__()

        self.model = PeintTransformer(
            esm_model=esm_model,
            esm_vocab=esm_vocab,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_len=max_seq_len,
            **kwargs,
        )

        self.lr = lr
        self.wd = kwargs.get("weight_decay", 0.0)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_hyperparameters(ignore=["esm_model", "esm_vocab"])

    def forward(self, x, y, t, x_attn_mask, y_attn_mask):
        return self.model(x, y, t, x_attn_mask, y_attn_mask)

    def _log(self, loss_metrics, train=True):
        phase = "train" if train else "val"
        for k, v in loss_metrics.items():
            self.log(
                f"{phase}/{k}",
                v,
                on_step=train,
                on_epoch=(not train),
                sync_dist=True,
            )
            if train:
                self.log(
                    f"{phase}/{k}_epoch",
                    v,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        [x, x_targets, y, y_targets, t, x_atten_mask, y_atten_mask] = batch

        x_logits, y_logits = self(x, y, t, x_atten_mask, y_atten_mask)
        x_logits = x_logits.transpose(-1, -2)
        y_logits = y_logits.transpose(-1, -2)

        mlm_loss = self.model.x_criterion(x_logits, x_targets)
        tlm_loss = self.model.y_criterion(y_logits, y_targets)

        mlm_ppl = torch.exp(mlm_loss.detach())
        tlm_ppl = torch.exp(tlm_loss.detach())

        loss = mlm_loss + tlm_loss

        metrics = {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "tlm_loss": tlm_loss,
            "mlm_ppl": mlm_ppl,
            "tlm_ppl": tlm_ppl,
        }

        self._log(metrics, train=True)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        [x, x_targets, y, y_targets, t, x_atten_mask, y_atten_mask] = batch

        yt_mask = y_targets != self.model.vocab.padding_idx
        times = t.expand_as(y_targets)
        tbins = times[yt_mask]

        with torch.no_grad():
            x_logits, y_logits = self(x, y, t, x_atten_mask, y_atten_mask)

        y_loss = F.cross_entropy(
            y_logits.transpose(-1, -2),
            y_targets,
            ignore_index=self.model.vocab.padding_idx,
            reduction="none",
        )

        mlm_loss = F.cross_entropy(
            x_logits.transpose(-1, -2),
            x_targets,
            ignore_index=self.model.vocab.padding_idx,
            reduction="mean",
        )

        mlm_ppl = torch.exp(mlm_loss.detach())
        mask_loss = -1 * y_loss[yt_mask]
        ppl_per_bin = {b.item(): mask_loss[tbins == b].cpu().numpy() for b in t}
        acc = (
            (y_logits.argmax(-1)[yt_mask] == y_targets[yt_mask]).float().mean().detach()
        )

        metrics = {
            "loss": y_loss[yt_mask].mean(),
            "ppl": torch.exp(y_loss[yt_mask].mean()),
            "acc": acc,
            "mlm_loss": mlm_loss,
            "mlm_ppl": mlm_ppl,
        }

        self._log(metrics, train=False)
        return metrics, ppl_per_bin

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.wd},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)

        scheduler = {
            "scheduler": get_polynomial_decay_schedule_with_warmup(
                optimizer,
                self.num_warmup_steps,
                self.num_training_steps,
                power=2.0,
            ),
            "name": "inverse-sqrt-lr",
            "interval": "step",
        }
        return [optimizer], [scheduler]
