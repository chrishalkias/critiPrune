#!/bin/bash
#SBATCH --job-name=inoise
#SBATCH --partition=cpu-zen4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/inoise_%A_%a.out
#SBATCH --error=slurm_logs/inoise_%A_%a.err

# Input-noise / pruning iso-accuracy sweep across all 12 (dataset, method)
# combinations. One array task per combination; each task walks all
# saved checkpoints in
# ``unstructured_pruning/checkpoints/unstructured_figures_<ds>_<m>/``
# and writes one JSON per cell under
# ``input_noise/results_cluster/unstructured_figures_<ds>_<m>/``.
#
# Submit (from project root on ALICE):
#   sbatch --account=<PROJECT> input_noise/submit.sbatch
#
# Hard-coded partition: cpu-zen4 (newer AMD CPUs, 7-day limit). All work
# is plain forward passes on FCNetwork; no GPU needed. Test data is
# loaded once per task and reused across all cells.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs input_noise/results_cluster

# --- Module + venv setup --------------------------------------------------
module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/.venvs/critiPrune/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# --- Constrain BLAS / torch to the allocated CPUs ------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# --- Persistent dataset caches under $HOME -------------------------------
# /tmp is per-node and non-persistent; download once into $HOME so every
# array task and every re-run picks up the same cache.
export MNIST_DATA_DIR="$HOME/.cache/critiPrune/mnist28"
export FEATURE_CACHE_DIR="$HOME/.cache/critiPrune/cifar_features"
mkdir -p "$MNIST_DATA_DIR" "$FEATURE_CACHE_DIR"

# --- Map array index -> (dataset, method) ---------------------------------
COMBOS=(
    "mnist28      random"
    "mnist28      magnitude"
    "mnist28      wanda"
    "cifar_pca    random"
    "cifar_pca    magnitude"
    "cifar_pca    wanda"
    "cifar_resnet random"
    "cifar_resnet magnitude"
    "cifar_resnet wanda"
    "sklearn      random"
    "sklearn      magnitude"
    "sklearn      wanda"
)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#COMBOS[@]}" ]; then
    echo "task index $IDX out of range (only ${#COMBOS[@]} combos)" >&2
    exit 1
fi
read -r DATASET METHOD <<< "${COMBOS[$IDX]}"

echo "Job $SLURM_JOB_ID task $IDX started at $(date)"
echo "  dataset=$DATASET  method=$METHOD"
echo "  cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}"

python3 -m input_noise.runners.cluster_sweep \
    --dataset "$DATASET" \
    --method  "$METHOD" \
    --torch-threads "${SLURM_CPUS_PER_TASK:-4}"

echo "Task $IDX completed at $(date)"
