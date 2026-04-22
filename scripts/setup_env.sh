#!/bin/bash
# Set up the critiPrune virtual environment on ALICE.
# Run this once from an interactive compute session, NOT on the login node:
#
#   srun --partition=cpu-short --time=01:00:00 --cpus-per-task=4 --mem=16G \
#        --account=liacs --pty bash
#   cd ~/critiPrune
#   bash scripts/setup_env.sh
set -euo pipefail

module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

VENV_DIR="$HOME/.venvs/critiPrune"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

# PyTorch with CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Other dependencies (skip tensorflow — not used by the scaling scripts)
pip install matplotlib numpy scikit-learn scipy

echo ""
echo "Environment ready. Verify GPU access with:"
echo "  python3 -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
