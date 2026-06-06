#!/bin/bash
# Submit the HIGH-DENSITY input-noise sweep array on ALICE.
#
# Usage:
#   bash input_noise/submit_dense.sh <PROJECT_ACCOUNT>
#
# Example:
#   bash input_noise/submit_dense.sh liacs
#
# Submits a 9-task array on cpu-zen4 (3 datasets x 3 methods; cifar_pca is
# omitted because it is excluded from the downstream analysis). Each task
# walks one (dataset, method) checkpoint directory and runs the dense
# (s, sigma) grid. Resumable: re-running skips any (H, L, r) cell whose
# dense JSON already exists.
#
# Dense grid fills the serrated collapse triangle by sweeping 50 retention
# (s) values instead of 10. Results land in input_noise/results_cluster_dense.

set -euo pipefail

PROJECT="${1:-}"
if [ -z "$PROJECT" ]; then
    echo "Usage: $0 <PROJECT_ACCOUNT>" >&2
    echo "  Find your account via: sacctmgr show associations user=\$USER format=account,partition" >&2
    exit 1
fi

mkdir -p slurm_logs input_noise/results_cluster_dense

JOB_ID=$(sbatch --account="$PROJECT" --parsable input_noise/submit_dense.sbatch)
echo "Submitted dense input-noise array job: $JOB_ID"
echo
echo "Monitor with:"
echo "  squeue -u \$USER -j $JOB_ID"
echo "  tail -f slurm_logs/inoise_dense_${JOB_ID}_*.out"
echo
echo "After all tasks finish, aggregate and replot with:"
echo "  python3 -m input_noise.aggregate \\"
echo "      --root input_noise/results_cluster_dense \\"
echo "      --output input_noise/results_cluster_dense_all.json"
echo "  python3 -m input_noise.cluster_analyze \\"
echo "      --input input_noise/results_cluster_dense_all.json"
