#!/bin/bash
# Multi-GPU sharding: correctness + scaling, on one node.
#
# Correctness: an all-vs-all hit list produced on N GPUs must be identical to the
# one produced on 1 GPU. Each reference is scored independently against the same
# query list, so identical means identical, not "close".
#
# Scaling: same workload timed at 1 GPU and at N.
#
#   sbatch benchmarks/inference/shard_check.sbatch
#   sbatch --gres=gpu:A5000:8 --export=ALL,NGPU=8 benchmarks/inference/shard_check.sbatch
#SBATCH --job-name=peint_shard
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/shard_%x-%j.log

set -uo pipefail

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate prot-evo

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

NGPU=${NGPU:-4}
# Correctness needs only enough references to cross shard boundaries; scaling
# needs enough work to amortize the one-off process spawn and per-rank model
# load, which is ~15 s of ESM2-150M construction on each GPU.
N_SEQ=${N_SEQ:-120}
N_SEQ_SCALE=${N_SEQ_SCALE:-400}
BS=${BS:-32}
OUTDIR=${OUTDIR:-results/shard}
mkdir -p "$OUTDIR"

echo "Host: $(hostname)  NGPU: $NGPU  N_SEQ: $N_SEQ"
python -c "import torch; print('visible GPUs:', torch.cuda.device_count(), torch.cuda.get_device_name(0))"

echo "=== correctness: 1 GPU vs $NGPU GPUs (identical hit lists expected) ==="
python benchmarks/inference/run_workload.py --repo . --workload homology --mode dump \
  --n-sequences "$N_SEQ" --num-gpus 1 --out "$OUTDIR/hits_1gpu"
python benchmarks/inference/run_workload.py --repo . --workload homology --mode dump \
  --n-sequences "$N_SEQ" --num-gpus "$NGPU" --out "$OUTDIR/hits_${NGPU}gpu"

python - "$OUTDIR/hits_1gpu.json" "$OUTDIR/hits_${NGPU}gpu.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["hits"]
b = json.load(open(sys.argv[2]))["hits"]
assert len(a) == len(b), f"hit count differs: {len(a)} vs {len(b)}"
bad = [(x, y) for x, y in zip(a, b) if x != y]
if bad:
    print(f"[shard] MISMATCH in {len(bad)}/{len(a)} hits; first: {bad[0]}")
    sys.exit(1)
print(f"[shard] hit lists IDENTICAL across GPU counts ({len(a)} hits)")
PY

echo "=== scaling (N=$N_SEQ_SCALE, single timed pass each) ==="
python benchmarks/inference/run_workload.py --repo . --workload homology --mode bench \
  --n-sequences "$N_SEQ_SCALE" --batch-sizes "$BS" --iters 1 --num-gpus 1 \
  --out "$OUTDIR/bench_1gpu"
python benchmarks/inference/run_workload.py --repo . --workload homology --mode bench \
  --n-sequences "$N_SEQ_SCALE" --batch-sizes "$BS" --iters 1 --num-gpus "$NGPU" \
  --out "$OUTDIR/bench_${NGPU}gpu"
