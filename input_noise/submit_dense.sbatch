#!/bin/bash
#SBATCH --job-name=inoise_dense
#SBATCH --partition=cpu-zen4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --array=0-8
#SBATCH --output=slurm_logs/inoise_dense_%A_%a.out
#SBATCH --error=slurm_logs/inoise_dense_%A_%a.err

# High-density input-noise / pruning iso-accuracy sweep.
#
# Motivation: in the coarse run the joint grid used only ~10 retention (s)
# values, so each cell contributed ~10 iso-A=0.5 contour points. Because
# xi = (1-s)(1 + <x^2>/sigma^2(1)) is linear in s, those discrete s values
# show up as ~10 vertical stripes in the collapse plot -> a "serrated"
# triangle. Densifying the s grid to 50 values fills the triangle in.
#
# Grids (vs coarse defaults in cluster_sweep.py):
#   s     : 0.02 .. 1.00 step 0.02            (50 pts, was 10)
#   sigma : 0 .. 3.0, finer at the low end    (16 pts, was 11)
#   seeds : 3 mask x 3 noise                  (was 3 x 4)
#
# Writes to a SEPARATE results base so the existing coarse per-cell JSONs
# are not reused (the sweep is resumable by filename, and the dense run must
# not pick up coarse files). cifar_pca is omitted: it is excluded from the
# downstream analysis (EXCLUDE_DATASETS in cluster_analyze.py), so a 9-task
# array (3 datasets x 3 methods) is sufficient.
#
# Submit (from project root on ALICE):
#   sbatch --account=<PROJECT> input_noise/submit_dense.sbatch
#
# Resumable: re-submitting skips any (H, L, r) cell whose dense JSON exists.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs input_noise/results_cluster_dense

# --- Module + venv setup --------------------------------------------------
module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
source "$HOME/.venvs/critiPrune/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# --- Constrain BLAS / torch to the allocated CPUs ------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Persistent dataset caches under $HOME -------------------------------
export MNIST_DATA_DIR="$HOME/.cache/critiPrune/mnist28"
export FEATURE_CACHE_DIR="$HOME/.cache/critiPrune/cifar_features"
mkdir -p "$MNIST_DATA_DIR" "$FEATURE_CACHE_DIR"

# --- Dense sweep grids ----------------------------------------------------
S_GRID="0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 1.0"
SIGMA_GRID="0 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.65 0.8 1.0 1.25 1.5 2.0 3.0"

# --- Map array index -> (dataset, method); cifar_pca omitted --------------
COMBOS=(
    "mnist28      random"
    "mnist28      magnitude"
    "mnist28      wanda"
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
echo "  dense grid: |s|=50  |sigma|=16  (3 mask x 3 noise)"

python3 -m input_noise.runners.cluster_sweep \
    --dataset "$DATASET" \
    --method  "$METHOD" \
    --results-base input_noise/results_cluster_dense \
    --s-grid-joint     $S_GRID \
    --sigma-grid-joint $SIGMA_GRID \
    --n-mask-seeds 3 \
    --n-noise-draws 3 \
    --torch-threads "${SLURM_CPUS_PER_TASK:-8}"

echo "Task $IDX completed at $(date)"
