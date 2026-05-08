#!/bin/bash
# SLURM script for the densification run (more_combinations.py).
# Trains only the (H, L, repeat) cells that are *missing* from the existing
# scaling_results.json — already-done cells are no-ops thanks to the resume
# logic in unstructured_pruning.core.
#
# Parameterised via env vars (set by sbatch --export from submit_more.sh):
#   DATASET   : sklearn | mnist28 | cifar_pca | cifar_resnet
#   METHOD    : random  | magnitude | wanda
#   N_REPEATS : (optional, default 1)
#
#SBATCH --job-name=u_more
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/u_more_%x_%j.out
#SBATCH --error=slurm_logs/u_more_%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

: "${DATASET:?must export DATASET}"
: "${METHOD:?must export METHOD}"
: "${N_REPEATS:=1}"
: "${EXTRA_REPEATS_ONLY:=0}"

EXTRA_FLAG=""
if [[ "$EXTRA_REPEATS_ONLY" == "1" ]]; then
    EXTRA_FLAG="--extra-repeats-only"
fi

OUTPUT_DIR="unstructured_pruning/figures/unstructured_figures_${DATASET}_${METHOD}"
mkdir -p "$OUTPUT_DIR"

echo "=================================================================="
echo "  Job       : $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "  Dataset   : $DATASET"
echo "  Method    : $METHOD"
echo "  N_REPEATS : $N_REPEATS"
echo "  EXTRA_ONLY: $EXTRA_REPEATS_ONLY"
echo "  Module    : unstructured_pruning.more_combinations"
echo "  Output    : $OUTPUT_DIR"
echo "  Node      : $SLURM_NODELIST"
echo "  Started   : $(date)"
echo "=================================================================="

module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

source "$HOME/.venvs/critiPrune/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

case "$DATASET" in
    cifar_pca|cifar_resnet)
        export FEATURE_CACHE_DIR="$HOME/.cache/cifar_features"
        mkdir -p "$FEATURE_CACHE_DIR"
        ;;
    mnist28)
        export MNIST_DATA_DIR="$HOME/.cache/mnist28"
        mkdir -p "$MNIST_DATA_DIR"
        ;;
esac

python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python3 -m unstructured_pruning.more_combinations \
    --dataset "$DATASET" \
    --method "$METHOD" \
    --output-dir "$OUTPUT_DIR" \
    --n-repeats "$N_REPEATS" \
    $EXTRA_FLAG

echo "Job completed at $(date)"
