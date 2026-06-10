#!/bin/bash
#SBATCH --job-name=prune_methods
#SBATCH --partition=cpu-zen4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=slurm_logs/prune_methods_%j.out
#SBATCH --error=slurm_logs/prune_methods_%j.err

# Accuracy-recovery A(s) comparison across all 12 unstructured pruning
# methods on one representative (dataset, H, L) cell. The trained weights are
# method-agnostic, so a single checkpoint family supplies the models and every
# method is applied post-hoc -- the sweep is cheap (~seconds per repeat).
#
# Configure the cell via env vars (defaults = mnist28 H192 L5, the paper cell):
#   DATASET=mnist28  HVAL=192  LVAL=5  sbatch --account=<PROJECT> \
#       scripts/u_scripts/method_comparison.sbatch
#
# Output: unstructured_pruning/figures/method_comparison/<dataset>_H<H>_L<L>.{json,png}

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/.venvs/critiPrune/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MNIST_DATA_DIR="$HOME/.cache/critiPrune/mnist28"
export FEATURE_CACHE_DIR="$HOME/.cache/critiPrune/cifar_features"
mkdir -p "$MNIST_DATA_DIR" "$FEATURE_CACHE_DIR"

DATASET="${DATASET:-mnist28}"
HVAL="${HVAL:-192}"
LVAL="${LVAL:-5}"

echo "Job $SLURM_JOB_ID: method comparison  dataset=$DATASET H=$HVAL L=$LVAL"

python3 -m unstructured_pruning.runners.method_comparison \
    --dataset "$DATASET" --H "$HVAL" --L "$LVAL" \
    --torch-threads "${SLURM_CPUS_PER_TASK:-8}"

echo "Done at $(date)"
