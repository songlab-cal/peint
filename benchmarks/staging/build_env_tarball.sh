#!/bin/bash
# Pack the conda env into ONE compressed archive.
#
# The far H200 nodes are slow because of per-file round-trips over a
# high-latency link, not raw bytes: the env is 7.6 GB but 74,446 files. Staging
# one archive turns ~74k metadata operations into one streaming read.
#SBATCH --job-name=build_env_tar
#SBATCH --partition=jsteinhardt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/buildenv_%j.log

set -uo pipefail
ENV=/scratch/users/yufan.cao/conda/envs/prot-evo
OUT=/scratch/users/yufan.cao/env_stage
mkdir -p "$OUT"

echo "[$(date +%T)] packing $ENV -> $OUT/prot-evo.tar.gz"
t0=$(date +%s)
# pigz across 16 threads; --numeric-owner avoids uid lookups on the far node.
tar -C "$(dirname "$ENV")" --numeric-owner -cf - "$(basename "$ENV")" \
  | pigz -p 8 -1 > "$OUT/prot-evo.tar.gz"
rc=$?
t1=$(date +%s)
echo "[$(date +%T)] tar exit=$rc in $((t1-t0)) s"
ls -lh "$OUT/prot-evo.tar.gz"
echo "compressed size: $(du -h "$OUT/prot-evo.tar.gz" | cut -f1) (from 7.6G / 74446 files)"
