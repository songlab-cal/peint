#!/bin/bash
# Tier-1 parity gate: the optimized tree must reproduce the baseline bit-for-bit.
#
# Run this after every Phase-1/Phase-2 commit:
#   sbatch benchmarks/inference/parity.sbatch
#   sbatch --export=ALL,TIER=2 benchmarks/inference/parity.sbatch   # deviation report
#
# Loads two full models (baseline + optimized, sequentially in subprocesses), so
# it must not run on the login node.
#SBATCH --job-name=peint_parity
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/parity_%x-%j.log

set -euo pipefail

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate prot-evo

export TORCH_HOME=/scratch/users/yufan.cao/torch_cache
export HF_HOME=/scratch/users/yufan.cao/hf-cache
# tqdm writes a progress bar per reference into the job log; useless in batch.
export TQDM_DISABLE=1

FAST_ROOT=/scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast
cd "$FAST_ROOT"

TIER=${TIER:-1}
# Space-separated, converted to commas: sbatch --export splits its own argument
# on commas, so a comma-separated WORKLOADS silently truncates to its first entry.
WORKLOADS=${WORKLOADS:-"logits likelihood generate"}
WORKLOADS=$(echo "$WORKLOADS" | tr ' ' ',')
OUT=${OUT:-results/parity_tier${TIER}.json}

echo "Host: $(hostname)  TIER: $TIER  WORKLOADS: $WORKLOADS"
python -c "import torch; assert torch.cuda.is_available(); \
print('CUDA ok:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

python benchmarks/inference/parity.py \
  --baseline-repo "${BASELINE_REPO:-../peint}" \
  --fast-repo "${FAST_REPO:-.}" \
  --workloads "$WORKLOADS" \
  --tier "$TIER" \
  --out "$OUT" \
  ${EXTRA:-}
