#!/bin/bash
#SBATCH --job-name=peint_ablation
#SBATCH --partition=yss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A100:2
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/yufan.cao/protevo_ablations/logs/%x-%j.out
#SBATCH --error=/scratch/users/yufan.cao/protevo_ablations/logs/%x-%j.err
#
# Preemption-safe: --requeue re-runs this script on preemption; train_peint_model.py
# then auto-resumes from the latest checkpoint under --output_dir (Lightning restores
# model/optimizer/scheduler/global_step). At most `checkpoint_every` steps are re-done.
# Override partition/gres at submit for borrowed nodes, e.g.:
#   sbatch --partition=songmei --gres=gpu:H200:2 --job-name=peint_no_mlm \
#          train_ablation.sh configs/ablations/no_mlm.yaml
#
# Full-scale ablation training launcher (referee #3.3).
#
# Runs on 2x A100 (DDP, bf16) with the SAME effective batch as the published
# 4-GPU setup: the configs set accumulate_grad_batches=12, so
#   batch 32 * accumulate 12 * 2 GPUs = 768 seqs/update (~250k tokens),
# identical to the published 32 * 6 * 4. (2 GPUs is what the yss partition can
# schedule promptly; 4 contiguous A100s were days out.)
#
# Usage:
#   sbatch --job-name=peint_no_mlm train_ablation.sh configs/ablations/no_mlm.yaml
#
# The config file sets every hyper-parameter; add CLI flags after it to override
# (CLI > YAML > defaults), e.g. ... no_mlm.yaml --seed 1

set -eo pipefail

CONFIG="${1:?usage: sbatch train_ablation.sh configs/ablations/<name>.yaml [--override ...]}"
shift || true

# Initialize conda in the (non-interactive) batch shell. Sourcing ~/.bashrc is not
# reliable here (it returns early for non-interactive shells), so source conda.sh
# directly, then activate the project env.
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate prot-evo

cd /scratch/users/yufan.cao/peint-dev/rebuttal/peint

echo "Host: $(hostname)   Config: ${CONFIG}   Extra args: $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# NCCL robustness: some yss/borrowed nodes (e.g. luthien, feanor) hang on the
# GPU-to-GPU P2P transport, causing multi-GPU DDP to time out on the first
# all-reduce. Disabling P2P/IB forces a robust shared-memory/host transport. This
# is correctness-neutral (all-reduce results are identical) at a small speed cost.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Load pretrained backbones from SCRATCH, not HOME/node-local disk. fair-esm downloads
# the ESM2 weights via torch.hub (default ~/.cache/torch or node /tmp); on borrowed nodes
# that local disk can be full -> ENOSPC. A scratch cache is shared, has space, and is
# pre-populated so runs never re-download. HF_HOME does the same for ESM-C (esm3).
export TORCH_HOME=/scratch/users/yufan.cao/torch_cache
export HF_HOME=/scratch/users/yufan.cao/hf-cache

# One task per GPU (DDP); Lightning picks up the SLURM allocation.
srun python train_peint_model.py --config "${CONFIG}" "$@"
