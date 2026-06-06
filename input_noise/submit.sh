#!/bin/bash
# Submit the input-noise sweep array on ALICE.
#
# Usage:
#   bash input_noise/submit.sh <PROJECT_ACCOUNT>
#
# Example:
#   bash input_noise/submit.sh liacs
#
# Submits a 12-task array on cpu-zen4. Each task processes one
# (dataset, method) directory under
# unstructured_pruning/checkpoints/. Resumable: re-running the array
# will skip any (H, L, r) cells whose per-cell JSON already exists.

set -euo pipefail

PROJECT="${1:-}"
if [ -z "$PROJECT" ]; then
    echo "Usage: $0 <PROJECT_ACCOUNT>" >&2
    echo "  Find your account via: sacctmgr show associations user=\$USER format=account,partition" >&2
    exit 1
fi

mkdir -p slurm_logs input_noise/results_cluster

JOB_ID=$(sbatch --account="$PROJECT" --parsable input_noise/submit.sbatch)
echo "Submitted input-noise array job: $JOB_ID"
echo
echo "Monitor with:"
echo "  squeue -u \$USER -j $JOB_ID"
echo "  tail -f slurm_logs/inoise_${JOB_ID}_*.out"
echo
echo "After all tasks finish, aggregate to a single JSON with:"
echo "  python3 -m input_noise.runners.aggregate"
