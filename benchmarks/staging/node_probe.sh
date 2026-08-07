#!/bin/bash
# Filesystem probe for the far H200 nodes. Deliberately uses NO conda: activating
# the shared env is the thing under investigation, so touching it would just
# reproduce the stall we are trying to characterize.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/nodeprobe_%x_%j.log

ENV=/scratch/users/yufan.cao/conda/envs/prot-evo
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "host=$(hostname)"
say "--- candidate node-local storage ---"
for p in /tmp /var/tmp /local /localscratch /scratch_local /dev/shm "$SLURM_TMPDIR" "$TMPDIR"; do
  [ -n "$p" ] && [ -d "$p" ] && say "$(df -h "$p" 2>/dev/null | tail -1) <- $p  writable=$([ -w "$p" ] && echo yes || echo no)"
done

say "--- sequential read from shared /scratch (200 MB) ---"
BIG=$(find "$ENV/lib/python3.10/site-packages/torch/lib" -name '*.so*' -size +100M 2>/dev/null | head -1)
say "probe file: ${BIG:-none found}"
if [ -n "$BIG" ]; then
  t0=$(date +%s.%N)
  timeout 600 dd if="$BIG" of=/dev/null bs=1M count=200 2>&1 | tail -1
  t1=$(date +%s.%N)
  say "sequential 200MB took $(echo "$t1 - $t0" | bc) s"
fi

say "--- metadata / small-file read (500 files) ---"
t0=$(date +%s.%N)
timeout 600 bash -c "find '$ENV/lib/python3.10/site-packages' -type f -name '*.py' 2>/dev/null | head -500 | xargs -r cat > /dev/null"
t1=$(date +%s.%N)
say "500 small files took $(echo "$t1 - $t0" | bc) s"

say "--- write speed to node-local /tmp (1 GB) ---"
t0=$(date +%s.%N)
timeout 600 dd if=/dev/zero of=/tmp/_probe_$$ bs=1M count=1024 conv=fsync 2>&1 | tail -1
t1=$(date +%s.%N)
say "local write 1GB took $(echo "$t1 - $t0" | bc) s"
rm -f /tmp/_probe_$$

say "DONE"
