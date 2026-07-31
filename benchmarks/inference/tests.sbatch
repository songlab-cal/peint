#!/bin/bash
# Run the package's own test suite inside this worktree, including the
# integration tests that need model_checkpoints/ (symlinked to ../peint).
#
#   sbatch benchmarks/inference/tests.sbatch
#   sbatch --export=ALL,PYTEST_ARGS="tests/test_model_inference.py -v" benchmarks/inference/tests.sbatch
#SBATCH --job-name=peint_fast_tests
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o /scratch/users/yufan.cao/protevo_ablations/logs/tests_%x-%j.log

set -uo pipefail

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate prot-evo

export TORCH_HOME=/scratch/users/yufan.cao/torch_cache
export HF_HOME=/scratch/users/yufan.cao/hf-cache
# tqdm writes a progress bar per reference into the job log; useless in batch.
export TQDM_DISABLE=1

cd /scratch/users/yufan.cao/peint-dev/rebuttal/peint-fast

echo "Host: $(hostname)"
python -c "import torch; print('CUDA:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

python -m pytest ${PYTEST_ARGS:-tests/ -q} 2>&1
echo "pytest exit: $?"
