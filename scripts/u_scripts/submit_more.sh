#!/bin/bash
# Submit densification jobs (more_combinations.py) to ALICE.
#
# By default submits all 4 datasets × 3 methods = 12 jobs.  Each job only
# trains the (H, L, repeat) cells missing from the existing
# scaling_results.json — already-done cells are no-ops, so re-submitting is
# safe and cheap.
#
# Walltime per job is the base walltime scaled by the *fraction of new
# cells* being added (relative to the original coarse grid: 56 base cells
# for non-sklearn, 120 for sklearn) plus a small safety margin.  This keeps
# resource bookings honest while still leaving headroom.
#
# Usage:
#   bash scripts/u_scripts/submit_more.sh [account]                         # default: liacs
#   DATASETS=mnist28 METHODS=random bash scripts/u_scripts/submit_more.sh   # subset
#   N_REPEATS=2 bash scripts/u_scripts/submit_more.sh                       # repeats
#
# Run from the project root.
set -euo pipefail

PROJECT="${1:-liacs}"
mkdir -p slurm_logs

# Per-dataset base walltime in seconds for the *original* coarse grid.
# Mirrors scripts/u_scripts/submit.sh.
declare -A TIME_SEC=(
    [sklearn]=7200       # 02:00:00 — original coarse grid
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

# Approximate fraction of the *full densified grid* that is new, plus a
# safety margin.  Used to scale the walltime down (or up, for cifar_resnet
# where the random-method run had only 20/56 base cells done).
declare -A SCALE_NUM=(
    [sklearn]=11    # 110 new / 120 base ≈ 0.92  → 11/12
    [mnist28]=11    #  61 new /  56 base ≈ 1.10  → 11/8 ≈ 1.4
    [cifar_pca]=11
    [cifar_resnet]=15  # extra slack for the random method's missing base cells
)
declare -A SCALE_DEN=(
    [sklearn]=12
    [mnist28]=8
    [cifar_pca]=8
    [cifar_resnet]=8
)

DATASETS="${DATASETS:-sklearn mnist28 cifar_pca cifar_resnet}"
METHODS="${METHODS:-random magnitude wanda}"
N_REPEATS="${N_REPEATS:-1}"

fmt_time() { printf '%02d:%02d:%02d' $(( $1 / 3600 )) $(( ($1 % 3600) / 60 )) $(( $1 % 60 )); }

echo "Submitting densification jobs on partition gpu-l4-24g, account=$PROJECT"
echo "Datasets   : $DATASETS"
echo "Methods    : $METHODS"
echo "N_REPEATS  : $N_REPEATS"
echo

for DATASET in $DATASETS; do
    NUM=${SCALE_NUM[$DATASET]}
    DEN=${SCALE_DEN[$DATASET]}
    SCALED_SEC=$(( ${TIME_SEC[$DATASET]} * NUM * N_REPEATS / DEN ))
    # Floor at 1h, ceiling at 24h.
    if (( SCALED_SEC < 3600 )); then SCALED_SEC=3600; fi
    if (( SCALED_SEC > 86400 )); then SCALED_SEC=86400; fi
    SCALED_TIME=$(fmt_time "$SCALED_SEC")
    for METHOD in $METHODS; do
        JOB_NAME="umore_${DATASET}_${METHOD}"
        JOB_ID=$(sbatch \
            --account="$PROJECT" \
            --job-name="$JOB_NAME" \
            --time="$SCALED_TIME" \
            --mem="${MEM[$DATASET]}" \
            --export=ALL,DATASET="$DATASET",METHOD="$METHOD",N_REPEATS="$N_REPEATS" \
            --parsable \
            scripts/u_scripts/more_combinations.sbatch)
        printf "  %-30s  job=%s  time=%s  mem=%s\n" \
            "$JOB_NAME" "$JOB_ID" "$SCALED_TIME" "${MEM[$DATASET]}"
    done
done

echo
echo "Monitor:    squeue -u \$USER"
echo "Logs:       tail -f slurm_logs/u_more_*_<job_id>.out"
