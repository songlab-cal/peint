#!/bin/bash
# Correctness gate. Two checks, both must pass:
#
#   1. parity      - the optimized tree reproduces the pristine tree bit-for-bit
#   2. golden      - the optimized tree is no further from the released
#                    protevo/tests/y_logits.npy than the pristine tree is
#
# The second is the only check against an *absolute* reference: the fixture
# predates this branch, whereas parity only says "same as the other tree".
#
# Run this after every change:
#   sbatch benchmarks/inference/parity.sbatch
#   sbatch --export=ALL,TIER=2 benchmarks/inference/parity.sbatch   # deviation report
#   sbatch --export=ALL,SKIP_GOLDEN=1 benchmarks/inference/parity.sbatch
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

# Auto-stages onto node-local NVMe on NFSv4.2 nodes, otherwise uses the shared
# env. Exports $PY. Replaces `conda activate`, which itself stalls on those nodes.
source /scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast/benchmarks/staging/stage_env.sh

export TORCH_HOME=/scratch/users/yufan.cao/torch_cache
export HF_HOME=/scratch/users/yufan.cao/hf-cache
# tqdm writes a progress bar per reference into the job log; useless in batch.
export TQDM_DISABLE=1
# Shared A100/H200 nodes have filled their local /tmp and killed runs on ENOSPC.
# Keep torch/wandb scratch on the parallel filesystem.
export TMPDIR=${TMPDIR:-/scratch/users/yufan.cao/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/users/yufan.cao/.cache}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

FAST_ROOT=/scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast
cd "$FAST_ROOT"

TIER=${TIER:-1}
# Space-separated, converted to commas: sbatch --export splits its own argument
# on commas, so a comma-separated WORKLOADS silently truncates to its first entry.
WORKLOADS=${WORKLOADS:-"logits likelihood generate"}
WORKLOADS=$(echo "$WORKLOADS" | tr ' ' ',')
OUT=${OUT:-results/parity_tier${TIER}.json}

echo "Host: $(hostname)  TIER: $TIER  WORKLOADS: $WORKLOADS"
"$PY" -c "import torch; assert torch.cuda.is_available(); \
print('CUDA ok:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

BASELINE_REPO=${BASELINE_REPO:-../peint}
FAST_REPO=${FAST_REPO:-.}

echo "===== 1/2 parity: optimized vs pristine ====="
"$PY" benchmarks/inference/parity.py \
  --baseline-repo "$BASELINE_REPO" \
  --fast-repo "$FAST_REPO" \
  --workloads "$WORKLOADS" \
  --tier "$TIER" \
  --out "$OUT" \
  ${EXTRA:-}

# Skipped when both sides are the same tree (the Tier-2 flag-on-vs-off mode),
# where comparing a tree to itself against the fixture says nothing.
if [ "${SKIP_GOLDEN:-0}" = "1" ]; then
  echo "===== 2/2 golden fixture: SKIPPED (SKIP_GOLDEN=1) ====="
elif [ "$(readlink -f "$BASELINE_REPO")" = "$(readlink -f "$FAST_REPO")" ]; then
  echo "===== 2/2 golden fixture: SKIPPED (both sides are the same tree) ====="
else
  echo "===== 2/2 golden fixture: both precision paths ====="
  "$PY" benchmarks/inference/golden_check.py \
    --baseline-repo "$BASELINE_REPO" --fast-repo "$FAST_REPO"
  "$PY" benchmarks/inference/golden_check.py \
    --baseline-repo "$BASELINE_REPO" --fast-repo "$FAST_REPO" --no-flash
fi

echo "===== all correctness checks passed ====="
