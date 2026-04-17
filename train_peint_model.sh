#!/bin/bash
#SBATCH --job-name=1k_cts_dms
#SBATCH --output=logs/output/%j.log  
#SBATCH --error=logs/error/%j.log    
#SBATCH --time=14-00:00:00        
#SBATCH --partition=yss
#SBATCH --gres=gpu:2  

srun python train_esmtransformer.py \
--data_path /scratch/users/akoehl/protein-evolution/local_data/1k_original_gapless_512l \
--families_file /scratch/users/akoehl/protein-evolution/local_data/1k_dms_data.json \
--output_dir /scratch/users/matthew_liu/protevo_checkpoints \
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

