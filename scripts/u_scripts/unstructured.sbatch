#!/bin/bash
# Generic SLURM script for unstructured-pruning scaling runs.
# Parameterised via the DATASET and METHOD env vars (exported by sbatch --export).
#
#   DATASET : sklearn | mnist28 | cifar_pca | cifar_resnet
#   METHOD  : random  | magnitude | wanda
#
# Submit through scripts/u_scripts/submit.sh, which sets time/mem overrides per dataset.
#
#SBATCH --job-name=u_pruning
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/u_%x_%j.out
#SBATCH --error=slurm_logs/u_%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

: "${DATASET:?must export DATASET}"
: "${METHOD:?must export METHOD}"
: "${N_REPEATS:=1}"

case "$DATASET" in
    sklearn)      MODULE="unstructured_pruning.runners.mnist_scaling"         ;;
    mnist28)      MODULE="unstructured_pruning.runners.mnist28_scaling"       ;;
    cifar_pca)    MODULE="unstructured_pruning.runners.cifar_scaling"         ;;
    cifar_resnet) MODULE="unstructured_pruning.runners.cifar_resnet_scaling"  ;;
    *) echo "unknown DATASET: $DATASET" >&2; exit 2 ;;
esac

OUTPUT_DIR="assets/unstructured_pruning/${DATASET}_${METHOD}"
mkdir -p "$OUTPUT_DIR"

echo "=================================================================="
echo "  Job       : $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "  Dataset   : $DATASET"
echo "  Method    : $METHOD"
echo "  N_REPEATS : $N_REPEATS"
echo "  Module    : $MODULE"
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

# Dataset-specific caches (mirroring existing scripts)
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

python3 -m "$MODULE" --method "$METHOD" --output-dir "$OUTPUT_DIR" --n-repeats "$N_REPEATS"

echo "Job completed at $(date)"
