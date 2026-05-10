#!/bin/bash
# Submit one temperature_pruning job per dataset to ALICE.
#
# Each job runs the full (sigma, density) sweep with N_TRIALS independent
# (noise, mask) bundles per (H, L, sigma), producing error bars on p_c.
#
# Usage:
#   bash u_scripts/submit_temperature.sh                   # 3 jobs, default account=liacs
#   bash u_scripts/submit_temperature.sh <account>
#   DATASETS="sklearn" bash u_scripts/submit_temperature.sh     # subset
#   N_TRIALS=10 bash u_scripts/submit_temperature.sh            # fewer trials
#
# Run from the project root.
set -euo pipefail

PROJECT="${1:-liacs}"
mkdir -p slurm_logs

DATASETS="${DATASETS:-sklearn mnist28 cifar_resnet}"
N_TRIALS="${N_TRIALS:-20}"
N_MASK_SEEDS="${N_MASK_SEEDS:-2}"
N_NOISE_SEEDS="${N_NOISE_SEEDS:-2}"

# Per-dataset wallclock and memory.  Rough estimates for n_trials=20,
# n_mask=2, n_noise=2 (== 80 evaluations per (cell, sigma)) on a 3x3 (H, L)
# grid with 100 sigma values.  Scale linearly with N_TRIALS.
declare -A WALLTIME=(
    [sklearn]="04:00:00"
    [mnist28]="10:00:00"
    [cifar_resnet]="10:00:00"
)
declare -A MEM=(
    [sklearn]="8G"
    [mnist28]="16G"
    [cifar_resnet]="32G"
)

# Scale walltime linearly with N_TRIALS (default scaling baseline is 20).
scale_time() {
    local base_seconds n_trials_int scaled
    base_seconds=$(awk -F: '{print $1*3600 + $2*60 + $3}' <<<"$1")
    n_trials_int="$2"
    scaled=$(( base_seconds * n_trials_int / 20 ))
    # Add a 10% safety margin.
    scaled=$(( scaled + scaled / 10 ))
    printf '%02d:%02d:%02d' $(( scaled / 3600 )) $(( (scaled % 3600) / 60 )) $(( scaled % 60 ))
}

echo "Submitting temperature_pruning jobs (account=$PROJECT)"
echo "  Datasets       : $DATASETS"
echo "  N_TRIALS       : $N_TRIALS"
echo "  N_MASK_SEEDS   : $N_MASK_SEEDS"
echo "  N_NOISE_SEEDS  : $N_NOISE_SEEDS"
echo

for DATASET in $DATASETS; do
    TIME=$(scale_time "${WALLTIME[$DATASET]}" "$N_TRIALS")
    JOB_NAME="t_${DATASET}"
    JOB_ID=$(sbatch \
        --account="$PROJECT" \
        --job-name="$JOB_NAME" \
        --time="$TIME" \
        --mem="${MEM[$DATASET]}" \
        --export=ALL,DATASET="$DATASET",N_TRIALS="$N_TRIALS",N_MASK_SEEDS="$N_MASK_SEEDS",N_NOISE_SEEDS="$N_NOISE_SEEDS" \
        --parsable \
        u_scripts/temperature_pruning.sbatch)
    printf "  %-20s  job=%s  time=%s  mem=%s\n" \
        "$JOB_NAME" "$JOB_ID" "$TIME" "${MEM[$DATASET]}"
done

echo
echo "Monitor:    squeue -u \$USER"
echo "Logs:       tail -f slurm_logs/t_*_<job_id>.out"
echo "Cancel:     scancel <job_id>"
