#!/bin/bash
#SBATCH --job-name=peint_ablation
#SBATCH --partition=yss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:00
#SBATCH --output=/scratch/users/yufan.cao/protevo_ablations/logs/%x-%j.out
#SBATCH --error=/scratch/users/yufan.cao/protevo_ablations/logs/%x-%j.err
#
# Full-scale ablation training launcher (referee #3.3).
#
# Reproduces the published PEINT setup: 4x A100, DDP, bf16, effective batch
# ~250k tokens/update (batch 32 * accumulate 6 * 4 GPUs * ~1022 tokens).
#
# Usage:
#   sbatch --job-name=peint_no_mlm train_ablation.sh configs/ablations/no_mlm.yaml
#
# The config file sets every hyper-parameter; add CLI flags after it to override
# (CLI > YAML > defaults), e.g. ... no_mlm.yaml --seed 1

set -eo pipefail

CONFIG="${1:?usage: sbatch train_ablation.sh configs/ablations/<name>.yaml [--override ...]}"
shift || true

source ~/.bashrc
conda activate prot-evo

cd /scratch/users/yufan.cao/peint-dev/rebuttal/peint

echo "Host: $(hostname)   Config: ${CONFIG}   Extra args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# One task per GPU (DDP); Lightning picks up the SLURM allocation.
srun python train_peint_model.py --config "${CONFIG}" "$@"
