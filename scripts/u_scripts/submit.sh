#!/bin/bash
# Submit all 12 unstructured-pruning scaling jobs to ALICE (gpu-l4-24g).
#
# Matrix: 4 datasets × 3 methods = 12 jobs.
#
# Usage:
#   bash scripts/u_scripts/submit.sh [account]        # default account: liacs
#   DATASETS=sklearn METHODS=random bash scripts/u_scripts/submit.sh   # subset
#
# Run from the project root.
set -euo pipefail

PROJECT="${1:-liacs}"
mkdir -p slurm_logs

# Per-dataset base walltime in seconds (single-repeat).  Scaled by N_REPEATS.
declare -A TIME_SEC=(
    [sklearn]=7200       # 02:00:00
    [mnist28]=36000      # 10:00:00
    [cifar_pca]=36000    # 10:00:00
    [cifar_resnet]=43200 # 12:00:00
)
declare -A MEM=(
    [sklearn]="16G"
    [mnist28]="48G"
    [cifar_pca]="64G"
    [cifar_resnet]="64G"
)

DATASETS="${DATASETS:-sklearn mnist28 cifar_pca cifar_resnet}"
METHODS="${METHODS:-random magnitude wanda}"
N_REPEATS="${N_REPEATS:-1}"

# Format scaled seconds as HH:MM:SS
fmt_time() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

echo "Submitting jobs on partition gpu-l4-24g, account=$PROJECT"
echo "Datasets   : $DATASETS"
echo "Methods    : $METHODS"
echo "N_REPEATS  : $N_REPEATS"
echo

for DATASET in $DATASETS; do
    SCALED_SEC=$(( ${TIME_SEC[$DATASET]} * N_REPEATS ))
    SCALED_TIME=$(fmt_time "$SCALED_SEC")
    for METHOD in $METHODS; do
        JOB_NAME="u_${DATASET}_${METHOD}"
        JOB_ID=$(sbatch \
            --account="$PROJECT" \
            --job-name="$JOB_NAME" \
            --time="$SCALED_TIME" \
            --mem="${MEM[$DATASET]}" \
            --export=ALL,DATASET="$DATASET",METHOD="$METHOD",N_REPEATS="$N_REPEATS" \
            --parsable \
            scripts/u_scripts/unstructured.sbatch)
        printf "  %-25s  job=%s  time=%s  mem=%s\n" \
            "$JOB_NAME" "$JOB_ID" "$SCALED_TIME" "${MEM[$DATASET]}"
    done
done

echo
echo "Monitor:    squeue -u \$USER"
echo "Logs:       tail -f slurm_logs/u_*_<job_id>.out"
