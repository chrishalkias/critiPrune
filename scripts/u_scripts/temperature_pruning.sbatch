#!/bin/bash
# SLURM job: full (sigma, density) sweep for temperature_pruning with
# n_trials independent (noise, mask) bundles per (cell, sigma) to produce
# error bars on the critical line.
#
# Parameterised via env vars (set by submit_temperature.sh):
#   DATASET       : sklearn | mnist28 | cifar_resnet
#   N_TRIALS      : number of independent trial bundles per (cell, sigma) (default 20)
#   N_MASK_SEEDS  : seeds per trial for random Bernoulli masks (default 2)
#   N_NOISE_SEEDS : seeds per trial for Gaussian weight noise (default 2)
#
# Output:
#   temperature_pruning/figures/<dataset>/results.json
#   temperature_pruning/figures/<dataset>/{critical_line, accuracy_curves, data_collapse}.png
#   temperature_pruning/figures/<dataset>/analysis_summary.json
#
# The sweep is *resumable*: rows already written to results.json are
# skipped, so re-submitting the job after a wallclock timeout simply
# continues where it left off.

#SBATCH --job-name=t_pruning
#SBATCH --partition=gpu-l4-24g
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/t_%x_%j.out
#SBATCH --error=slurm_logs/t_%x_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

: "${DATASET:?must export DATASET (sklearn|mnist28|cifar_resnet)}"
: "${N_TRIALS:=20}"
: "${N_MASK_SEEDS:=2}"
: "${N_NOISE_SEEDS:=2}"

OUTPUT_DIR="temperature_pruning/figures/${DATASET}"
mkdir -p "$OUTPUT_DIR"

echo "=================================================================="
echo "  Job          : $SLURM_JOB_ID  ($SLURM_JOB_NAME)"
echo "  Dataset      : $DATASET"
echo "  N_TRIALS     : $N_TRIALS"
echo "  N_MASK_SEEDS : $N_MASK_SEEDS"
echo "  N_NOISE_SEEDS: $N_NOISE_SEEDS"
echo "  Output       : $OUTPUT_DIR"
echo "  Node         : $SLURM_NODELIST"
echo "  CPUs         : $SLURM_CPUS_PER_TASK"
echo "  Mem          : $SLURM_MEM_PER_NODE MB"
echo "  Started      : $(date)"
echo "=================================================================="

# Activate the project virtualenv if it exists (cluster setup may differ).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# Cap BLAS threading to match the SLURM allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m temperature_pruning.main \
    --dataset "$DATASET" \
    --n-trials "$N_TRIALS" \
    --n-mask-seeds "$N_MASK_SEEDS" \
    --n-noise-seeds "$N_NOISE_SEEDS"

echo
echo "  Finished     : $(date)"
