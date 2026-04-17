import os
import json
from argparse import ArgumentParser
import datetime

import numpy as np
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
import esm

from protevo.models._flash_esm import ESM2Flash
from protevo.models import ESM2_REGISTRY, get_esm_model
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

    loader, esm_embed_dim = get_esm_model(args.esm_model)
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

    esm_model, esm_vocab = loader()
    print(f"Loaded ESM Model: {args.esm_model}")

    flash_esm_model = ESM2Flash(
        num_layers=esm_model.num_layers,
        embed_dim=esm_model.embed_dim,
        attention_heads=esm_model.attention_heads,
        alphabet="ESM-1b",
        token_dropout=True,
        dropout_p=0.0, #ESM2 does not use dropout
    )
    flash_esm_model.load_state_dict(esm_model.state_dict(), strict=False) #the rot emb is different here, so there are mismatched keys
    del(esm_model) #for some reason this is necessary for the pytorch lightning trainer??


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
    }

    data_args = {
        'data_path': args.data_path,
        'families': families,
        'vocab': esm_vocab,
        'max_len': args.max_seq_len,
        'batch_size': args.batch_size,
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


if __name__ == "__main__":

    parser = ArgumentParser()
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
    parser.add_argument('--grad_clip', type=float, default = 0.1, help="Gradient clip value (default = 0.1)")
    parser.add_argument('--name_addon', type=str, nargs="?", default=None, help="additional name arguments")
    parser.add_argument('--wandb_entity', type=str, nargs = "?", default=None, help='Wandb entity name')
    parser.add_argument('--wandb_project', type=str, nargs = "?", default=None, help='Wandb project name')
    parser.add_argument('--esm_model', type=str, default='ESM2-150M',
                        choices=list(ESM2_REGISTRY.keys()),
                        help='Base ESM2 model (determines and overrides embed_dim)')

    args = parser.parse_args()
    print(args)
    if args.seed is None and (args.accelerator =='gpu' and args.devices > 1):
        raise ValueError("Must set seed when using multiple GPUs")
    
    main(args)