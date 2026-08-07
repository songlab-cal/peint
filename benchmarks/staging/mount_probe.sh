#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/mount_%x_%j.log
ENV=/scratch/users/yufan.cao/conda/envs/prot-evo
echo "host=$(hostname)"
echo "MOUNT: $(findmnt -no SOURCE,FSTYPE,OPTIONS --target $ENV 2>/dev/null | head -c 300)"
echo -n "60 small files: "
t0=$(date +%s.%N)
timeout 300 bash -c "find $ENV/lib/python3.10/site-packages -type f -name '*.py' 2>/dev/null | head -60 | xargs -r cat > /dev/null"
t1=$(date +%s.%N)
echo "$(echo "($t1 - $t0)/60*1000" | bc -l | cut -c1-7) ms/file"
BIG=$(find $ENV/lib/python3.10/site-packages/torch/lib -name '*.so*' -size +100M 2>/dev/null | head -1)
echo -n "sequential: "; timeout 300 dd if="$BIG" of=/dev/null bs=1M count=100 2>&1 | tail -1
