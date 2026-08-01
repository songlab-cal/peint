#!/bin/bash
# Benchmark one PEINT inference workload against one copy of protevo.
#
# Nothing here may run on the login node: it loads the 219 MB checkpoint and
# builds the 150M-parameter model.
#
#   # baseline, generation, on an A5000 (rainbowquartz is usually idle)
#   sbatch --job-name=bench_base_gen \
#     --export=ALL,REPO=../peint,WORKLOAD=generate,OUT=results/base_generate \
#     benchmarks/inference/bench.sbatch
#
#   # optimized, same workload, on an H200
#   sbatch --gres=gpu:H200:1 --job-name=bench_fast_gen \
#     --export=ALL,REPO=.,WORKLOAD=generate,OUT=results/fast_generate \
#     benchmarks/inference/bench.sbatch
#
# The yss partition is saturated by the ablation runs, so default to jsteinhardt.
#SBATCH --job-name=peint_bench
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/bench_%x-%j.log

set -euo pipefail

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate prot-evo

# Backbone weights come from scratch, never from a borrowed node's /tmp.
export TORCH_HOME=/scratch/users/yufan.cao/torch_cache
export HF_HOME=/scratch/users/yufan.cao/hf-cache
# tqdm writes a progress bar per reference into the job log; useless in batch.
export TQDM_DISABLE=1
# H200/A100 nodes are shared; their local /tmp has filled up and killed a run
# before. Keep torch/wandb scratch on the parallel filesystem.
export TMPDIR=${TMPDIR:-/scratch/users/yufan.cao/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/scratch/users/yufan.cao/.cache}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

FAST_ROOT=/scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast
cd "$FAST_ROOT"

REPO=${REPO:?set REPO (e.g. . or ../peint)}
WORKLOAD=${WORKLOAD:?set WORKLOAD (generate|likelihood|homology|logits)}
OUT=${OUT:?set OUT (output path prefix)}
# Space-separated, converted to the comma form run_workload.py wants. sbatch
# --export splits its own argument on commas, so a comma-separated BATCH_SIZES
# silently truncates to its first element.
BATCH_SIZES=${BATCH_SIZES:-"1 8 32 64"}
BATCH_SIZES=$(echo "$BATCH_SIZES" | tr ' ' ',')
EXTRA=${EXTRA:-}

echo "Host: $(hostname)  REPO: $REPO  WORKLOAD: $WORKLOAD  OUT: $OUT"
python -c "import torch; assert torch.cuda.is_available(); \
print('CUDA ok:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

python benchmarks/inference/run_workload.py \
  --repo "$REPO" \
  --workload "$WORKLOAD" \
  --mode "${MODE:-bench}" \
  --batch-sizes "$BATCH_SIZES" \
  --out "$OUT" \
  $EXTRA
