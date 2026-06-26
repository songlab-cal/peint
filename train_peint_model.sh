#!/bin/bash
#SBATCH --job-name=peint_train
#SBATCH --output=logs/output/%j.log
#SBATCH --error=logs/error/%j.log
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu          # set to your cluster's GPU partition
#SBATCH --gres=gpu:2

# Example training launch. Replace the placeholder paths with your own:
#   --data_path      directory of transition data
#   --families_file  json file describing the protein families
#   --output_dir     where to write checkpoints
srun python train_peint_model.py \
--data_path /path/to/transitions \
--families_file /path/to/families.json \
--output_dir checkpoints \
--batch_size 32 \
--lr 4e-4 \
--max_seq_len 1022 \
--num_heads 20 \
--num_encoder_layers 5 \
--num_decoder_layers 5 \
--embed_dim 640 \
--seed 0 \
--n_families -1 \
--accumulate_grad_batches 13 \
--checkpoint_every 3000 \
--accelerator gpu \
--devices 0 1 \
--max_steps 300000 \
--num_warmup_steps 2000 \
--dropout_p 0.0 \
--weight_decay 0.01 \
--grad_clip 1.0 \
--use_attention_bias \
--esm_model ESM2-150M \
--name_addon cts_dms
