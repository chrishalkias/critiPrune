#!/bin/bash
#SBATCH --job-name=critiprune_mnist
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/mnist_%j.out
#SBATCH --error=slurm_logs/mnist_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs mnist_figures

echo "Job $SLURM_JOB_ID started at $(date) | node $SLURM_NODELIST"

module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

source "$HOME/.venvs/critiPrune/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python3 pruning/mnist_scaling.py

echo "Job completed at $(date)"
