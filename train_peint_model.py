import os
import json
from argparse import ArgumentParser
import datetime

import numpy as np
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from protevo.models import ESM2_REGISTRY, get_esm_model, build_esm_backbone
from protevo.models.training import (
    PeintLightningModule,
    ValidationLikelihoodCallback,
    GradNormCallback,
)
from protevo.datasets.training import PeintDataModule


def main(args):

    if args.seed is not None:
        pl.seed_everything(args.seed)

    with open(args.families_file, "r") as f:
        families = json.load(f)['families']

    if args.n_families == -1:
        families = families
    else:
        assert args.n_families <= len(families), "n_families must be less than or equal to the number of families"
        np.random.shuffle(families)
        families = families[:args.n_families]

    _, esm_embed_dim = get_esm_model(args.esm_model)
    if args.embed_dim != esm_embed_dim:
        print(f"WARNING: --embed_dim ({args.embed_dim}) does not match "
              f"{args.esm_model} embed_dim ({esm_embed_dim}). Overriding to {esm_embed_dim}.")
        args.embed_dim = esm_embed_dim

    if args.resume_path:
        run_name = args.resume_path.split('/')[-2]
    else:
        date = str(datetime.datetime.now().date()).replace('-','')
        run_name = (f"{date}-{args.num_encoder_layers}e"
                    f"{args.num_decoder_layers}d{args.num_heads}h"
                    f"{args.embed_dim}d-{args.esm_model}-{len(families)}fams"
                    )
        if args.name_addon:
            run_name = run_name + '-' + args.name_addon

    logger = pl.pytorch.loggers.wandb.WandbLogger(name=run_name, project=args.wandb_project, entity=args.wandb_entity)

    # Single source of truth for backbone construction (shared with _loading.py).
    flash_esm_model, esm_vocab, _ = build_esm_backbone(args.esm_model, use_flash=True)
    print(f"Loaded ESM Model: {args.esm_model}")

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
        # Backbone + ablation axes (saved into hyper_parameters so the checkpoint
        # rebuilds on the right backbone and remembers which axis was ablated).
        "encoder_backbone": args.esm_model,
        "mlm_weight": args.mlm_weight,
        "use_time_conditioning": not args.no_time_conditioning,
        "esm_finetune_mode": args.esm_finetune_mode,
        "lora_rank": args.lora_rank,
        "architecture": args.architecture,
    }

    data_args = {
        'data_path': args.data_path,
        'families': families,
        'vocab': esm_vocab,
        'max_len': args.max_seq_len,
        'batch_size': args.batch_size,
        'mask_prob': args.mask_prob,
    }

    model = PeintLightningModule(
        esm_model=flash_esm_model,
        esm_vocab=esm_vocab,
        **model_args
    )

    dm = PeintDataModule(
        **data_args
    )

    output_path = os.path.join(args.output_dir, run_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=output_path,
        save_top_k=-1,
        every_n_train_steps=args.checkpoint_every,
    )

    strategy = 'ddp'

    trainer = pl.Trainer(
        strategy=strategy,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, ValidationLikelihoodCallback(), GradNormCallback()],
        accelerator=args.accelerator,
        #devices=4,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        detect_anomaly=False,
        gradient_clip_val=args.grad_clip,
        precision='bf16'
    )

    if args.resume_path is not None:
        trainer.fit(model, dm, ckpt_path=args.resume_path)
    else:
        trainer.fit(model, dm)


def build_parser():
    """Construct the training argument parser (also used by tests)."""
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='Optional YAML file of arguments (e.g. an ablation config). '
                             'Its values act as defaults; explicit CLI flags override them.')
    parser.add_argument('--data_path', type=str, default='data/processed')
    parser.add_argument('--families_file', type=str, help='Path to json file containing family information')
    parser.add_argument('--output_dir', type=str, help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_seq_len', type=int, default=1022, help='Maximum sequence length')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='Number of encoder transformer layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder transformer layers')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_families', type=int, default=300, help='Number of families to train on')
    parser.add_argument('--num_warmup_steps', type=int, default=10000, help='Number of warmup steps')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Accelerator')
    parser.add_argument('--devices', type=int, nargs='+', default=None, help='GPU devices to use')
    parser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of steps to train for')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients over')
    parser.add_argument('--checkpoint_every', type=int, default=5000, help='Save checkpoint every n steps')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Validate every n epochs')
    parser.add_argument('--resume_path', type=str, nargs='?', default=None, help='Path to model checkpoint to resume training')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--use_attention_bias', action='store_true', help='Use attention bias')
    parser.add_argument('--dropout_p', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--grad_clip', type=float, default=0.1, help="Gradient clip value (default = 0.1)")
    parser.add_argument('--name_addon', type=str, nargs="?", default=None, help="additional name arguments")
    parser.add_argument('--wandb_entity', type=str, nargs="?", default=None, help='Wandb entity name')
    parser.add_argument('--wandb_project', type=str, nargs="?", default=None, help='Wandb project name')
    parser.add_argument('--esm_model', type=str, default='ESM2-150M',
                        choices=list(ESM2_REGISTRY.keys()),
                        help='Base ESM2 backbone (determines and overrides embed_dim); '
                             'saved as encoder_backbone in the checkpoint')

    # --- Ablation axes (referee #3.3); defaults reproduce published PEINT ---
    parser.add_argument('--mlm_weight', type=float, default=1.0,
                        help='Weight on the auxiliary MLM loss (0.0 ablates it)')
    parser.add_argument('--no_time_conditioning', action='store_true',
                        help='Ablate evolutionary-time conditioning (drop the time embedding)')
    parser.add_argument('--esm_finetune_mode', type=str, default='frozen',
                        choices=['frozen', 'lora', 'full'],
                        help='How to train the backbone (default: frozen)')
    parser.add_argument('--lora_rank', type=int, default=None,
                        help='LoRA rank; required when --esm_finetune_mode lora')
    parser.add_argument('--architecture', type=str, default='encoder_decoder',
                        choices=['encoder_decoder', 'decoder_only'],
                        help='Model architecture (decoder_only is deferred)')
    parser.add_argument('--mask_prob', type=float, default=0.15,
                        help='MLM masking probability applied to the source sequence')
    return parser


def parse_args_with_config(argv=None):
    """Parse CLI args, optionally seeded by a YAML ``--config`` file.

    Precedence: hard-coded argparse defaults < YAML config < explicit CLI flags.
    Unknown keys in the YAML fail loudly (a typo in an ablation config should not
    be silently ignored). Falls back to plain CLI parsing when no config is given.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is not None:
        import yaml
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        valid_dests = {a.dest for a in parser._actions}
        unknown = set(file_cfg) - valid_dests
        if unknown:
            raise ValueError(
                f"Unknown keys in {args.config}: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_dests)}"
            )
        parser.set_defaults(**file_cfg)
        args = parser.parse_args(argv)  # re-parse so explicit CLI flags still win
    return args


if __name__ == "__main__":
    args = parse_args_with_config()
    print(args)
    if args.seed is None and (args.accelerator == 'gpu' and args.devices > 1):
        raise ValueError("Must set seed when using multiple GPUs")

    main(args)