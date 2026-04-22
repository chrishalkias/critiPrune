#!/bin/bash
# Submit all 12 unstructured-pruning scaling jobs to ALICE (gpu-l4-24g).
#
# Matrix: 4 datasets × 3 methods = 12 jobs.
#
# Usage:
#   bash u_scripts/submit.sh [account]        # default account: liacs
#   DATASETS=sklearn METHODS=random bash u_scripts/submit.sh   # subset
#
# Run from the project root.
set -euo pipefail

PROJECT="${1:-liacs}"
mkdir -p slurm_logs

# Dataset → (time, mem) overrides.  Keys match the DATASET case in the sbatch.
declare -A TIME=(
    [sklearn]="02:00:00"
    [mnist28]="10:00:00"
    [cifar_pca]="10:00:00"
    [cifar_resnet]="12:00:00"
)
declare -A MEM=(
    [sklearn]="16G"
    [mnist28]="48G"
    [cifar_pca]="64G"
    [cifar_resnet]="64G"
)

DATASETS="${DATASETS:-sklearn mnist28 cifar_pca cifar_resnet}"
METHODS="${METHODS:-random magnitude wanda}"

echo "Submitting jobs on partition gpu-l4-24g, account=$PROJECT"
echo "Datasets: $DATASETS"
echo "Methods : $METHODS"
echo

for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
        JOB_NAME="u_${DATASET}_${METHOD}"
        JOB_ID=$(sbatch \
            --account="$PROJECT" \
            --job-name="$JOB_NAME" \
            --time="${TIME[$DATASET]}" \
            --mem="${MEM[$DATASET]}" \
            --export=ALL,DATASET="$DATASET",METHOD="$METHOD" \
            --parsable \
            u_scripts/unstructured.sbatch)
        printf "  %-25s  job=%s  time=%s  mem=%s\n" \
            "$JOB_NAME" "$JOB_ID" "${TIME[$DATASET]}" "${MEM[$DATASET]}"
    done
done

echo
echo "Monitor:    squeue -u \$USER"
echo "Logs:       tail -f slurm_logs/u_*_<job_id>.out"
