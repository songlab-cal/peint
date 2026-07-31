"""
Fine-tuning script for training on all families at once.
"""

import os
import lightning as pl

from protevo.datasets._training import PeintDataModule
from protevo.vep._train_utils import (
    setup_args,
    validate_args,
    load_esm_model,
    setup_run_name,
    setup_model,
    setup_callbacks,
    setup_trainer,
    prepare_model_args,
    load_families,
    ProtevoMixedDataModule,
)


def main():
    # Parse arguments
    parser = setup_args()
    args = parser.parse_args()
    print(args)

    # Validate arguments
    validate_args(args)

    # Basic validation for per-family mode
    if not args.finetune_families_file or not args.finetune_data_path:
        raise ValueError(
            "Must provide finetune_families_file and finetune_data_path for per-family mode"
        )

    # Set random seed if provided
    if args.seed is not None:
        pl.seed_everything(args.seed)

    # Load families
    families, finetune_families = load_families(args)

    # Set up run name
    run_name = setup_run_name(args)

    # Set up logger
    logger = pl.pytorch.loggers.wandb.WandbLogger(
        name=run_name, project="protein-evolution", entity="junhaobearxiong"
    )

    # Load ESM model
    flash_esm_model, esm_vocab = load_esm_model(args.which_esm)

    # Prepare model arguments
    model_args = prepare_model_args(args)

    # Set up the model
    model = setup_model(args, flash_esm_model, esm_vocab, model_args)

    # Create fine-tuning data module
    finetune_dm = PeintDataModule(
        data_path=args.finetune_data_path,
        families=finetune_families,
        vocab=esm_vocab,
        max_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    # Create original data module if we are mixing
    if args.finetune_mix_ratio < 1:
        original_dm = PeintDataModule(
            data_path=args.data_path,
            families=families,
            vocab=esm_vocab,
            max_len=args.max_seq_len,
            batch_size=args.batch_size,
        )
    else:
        original_dm = None

    # Create mixed data module for fine-tuning
    dm = ProtevoMixedDataModule(
        finetune_data_module=finetune_dm,
        original_data_module=original_dm,
        finetune_ratio=args.finetune_mix_ratio,
        batch_size=args.batch_size,
        num_workers=0,  # Adjust as needed
    )

    # Set up output path
    output_path = os.path.join(args.output_dir, run_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=False)

    # Set up callbacks
    callbacks = setup_callbacks(args, output_path)

    # Set up trainer
    trainer = setup_trainer(args, logger, callbacks)

    # Train the model
    if args.resume_checkpoint_path:
        trainer.fit(model, dm, ckpt_path=args.resume_checkpoint_path)
    else:
        trainer.fit(model, dm)


if __name__ == "__main__":
    main()
