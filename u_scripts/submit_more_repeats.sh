#!/bin/bash
# Submit "extra repeats" jobs: train r=1 and r=2 only on the *new*
# (densification) (H, L) cells.  Original coarse-grid cells stay at r=0.
#
# Walltimes are calibrated from the prior umore_* run (1 repeat on new cells)
# multiplied by 2 (two more repeats) and 1.5 (safety margin):
#
#   Dataset       prior elapsed   ×2 × 1.5    requested
#   sklearn       ~13 min         ~39 min     01:00:00
#   mnist28       ~1h 5min        ~3h 15min   04:00:00
#   cifar_pca     ~3h             ~9h         10:00:00
#   cifar_resnet  ~2-3h           ~9h         10:00:00
#
# Usage:
#   bash u_scripts/submit_more_repeats.sh [account]                       # default: liacs
#   DATASETS=mnist28 METHODS=random bash u_scripts/submit_more_repeats.sh # subset
#   N_REPEATS=4 bash u_scripts/submit_more_repeats.sh                     # adjust total repeats
#
# Run from the project root.
set -euo pipefail

PROJECT="${1:-liacs}"
mkdir -p slurm_logs

# Calibrated walltime per dataset for r=1 + r=2 on new cells.
declare -A TIME=(
    [sklearn]="01:00:00"
    [mnist28]="04:00:00"
    [cifar_pca]="10:00:00"
    [cifar_resnet]="10:00:00"
)
declare -A MEM=(
    [sklearn]="16G"
    [mnist28]="48G"
    [cifar_pca]="64G"
    [cifar_resnet]="64G"
)

DATASETS="${DATASETS:-sklearn mnist28 cifar_pca cifar_resnet}"
METHODS="${METHODS:-random magnitude wanda}"
N_REPEATS="${N_REPEATS:-3}"   # total repeats per new cell (r=0 already done; r=1,2 will be trained)

echo "Submitting EXTRA-REPEATS-ONLY jobs on partition gpu-l4-24g, account=$PROJECT"
echo "Datasets   : $DATASETS"
echo "Methods    : $METHODS"
echo "N_REPEATS  : $N_REPEATS  (only the new (H,L) cells run r>=1)"
echo

for DATASET in $DATASETS; do
    for METHOD in $METHODS; do
        JOB_NAME="urep_${DATASET}_${METHOD}"
        JOB_ID=$(sbatch \
            --account="$PROJECT" \
            --job-name="$JOB_NAME" \
            --time="${TIME[$DATASET]}" \
            --mem="${MEM[$DATASET]}" \
            --export=ALL,DATASET="$DATASET",METHOD="$METHOD",N_REPEATS="$N_REPEATS",EXTRA_REPEATS_ONLY=1 \
            --parsable \
            u_scripts/more_combinations.sbatch)
        printf "  %-30s  job=%s  time=%s  mem=%s\n" \
            "$JOB_NAME" "$JOB_ID" "${TIME[$DATASET]}" "${MEM[$DATASET]}"
    done
done

echo
echo "Monitor:    squeue -u \$USER"
echo "Logs:       tail -f slurm_logs/u_more_*_<job_id>.out"
