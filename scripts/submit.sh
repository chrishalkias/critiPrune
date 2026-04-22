#!/bin/bash
# Submit both scaling jobs to ALICE.
# Usage:  bash scripts/submit.sh [account]
# Default account: liacs
#
# Run from the project root:
#   cd ~/critiPrune
#   bash scripts/submit.sh
set -euo pipefail

PROJECT="${1:-liacs}"
mkdir -p slurm_logs

MNIST_JOB=$(sbatch --account="$PROJECT" --parsable scripts/mnist_scaling.sbatch)
echo "MNIST  job: $MNIST_JOB  (cpu-zen4, 2h)"

CIFAR_JOB=$(sbatch --account="$PROJECT" --parsable scripts/cifar_scaling.sbatch)
echo "CIFAR  job: $CIFAR_JOB  (gpu-a100-80g, 8h)"

echo ""
echo "Monitor:    squeue -u \$USER"
echo "MNIST log:  tail -f slurm_logs/mnist_${MNIST_JOB}.out"
echo "CIFAR log:  tail -f slurm_logs/cifar_${CIFAR_JOB}.out"
