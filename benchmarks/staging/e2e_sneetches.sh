#!/bin/bash
# End-to-end validation on a far node: stage the env locally, prove the
# relocated interpreter works, then run a real benchmark through it.
#SBATCH --job-name=e2e_sneetches
#SBATCH --partition=jsteinhardt
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/e2e_%x_%j.log

set -uo pipefail
REPO=/scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast
SHARED_PY=/scratch/users/yufan.cao/conda/envs/prot-evo/bin/python
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "host=$(hostname)"

# ---------- phase 1: stage ----------
say "PHASE 1: staging env to node-local NVMe"
t0=$(date +%s)
source "$REPO/diag/stage_env.sh"
say "PHASE 1 total: $(( $(date +%s) - t0 )) s"

# ---------- phase 2: does the relocated interpreter work? ----------
say "PHASE 2: imports from the RELOCATED env"
t0=$(date +%s)
"$PY" -c "
import sys, torch, esm, flash_attn
print('  sys.prefix   :', sys.prefix)
print('  torch        :', torch.__version__)
print('  flash_attn   :', flash_attn.__version__)
print('  cuda avail   :', torch.cuda.is_available())
print('  device       :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
"
say "PHASE 2 (local import) took $(( $(date +%s) - t0 )) s  rc=$?"

# ---------- phase 3: the same import from the SHARED env, for contrast ----------
say "PHASE 3: 'import torch' from the SHARED env (capped at 900 s)"
t0=$(date +%s)
timeout 900 "$SHARED_PY" -c "import torch; print('  shared torch', torch.__version__)"
rc=$?
el=$(( $(date +%s) - t0 ))
[ $rc -eq 124 ] && say "PHASE 3 TIMED OUT after ${el}s (this is the status quo)" \
                || say "PHASE 3 took ${el}s rc=$rc"

# ---------- phase 4: a real benchmark through the staged interpreter ----------
say "PHASE 4: generation benchmark via staged interpreter"
export TORCH_HOME=/scratch/users/yufan.cao/torch_cache HF_HOME=/scratch/users/yufan.cao/hf-cache
export TQDM_DISABLE=1
cd "$REPO"
t0=$(date +%s)
"$PY" benchmarks/inference/run_workload.py --repo . --workload generate --mode bench \
      --batch-sizes 64,1024 --iters 1 --warmup 0 --out results/sneetches_staged
say "PHASE 4 took $(( $(date +%s) - t0 )) s"
say "DONE"
