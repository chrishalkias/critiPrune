#!/bin/bash
#SBATCH --job-name=critiprune_test
#SBATCH --partition=gpu-short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/test_%j.out
#SBATCH --error=slurm_logs/test_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs

echo "=== START $(date) | job $SLURM_JOB_ID | node $SLURM_NODELIST ==="

module purge
module load ALICE/default
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.4.0

source "$HOME/.venvs/critiPrune/bin/activate"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# --- 1. GPU check ---
python3 -c "
import torch
avail = torch.cuda.is_available()
name  = torch.cuda.get_device_name(0) if avail else 'N/A'
print(f'CUDA available: {avail} | device: {name}')
assert avail, 'GPU not detected'
"

# --- 2. Import check ---
python3 -c "
from pruning.pruning import FCNetwork, precompute_pruning_scores, evaluate_path_accuracy
import torch
print(f'Imports OK | torch {torch.__version__}')
"

# --- 3. Minimal end-to-end: H=64 L=2 5 epochs ---
python3 - <<'PYEOF'
import torch, numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pruning.pruning import FCNetwork, precompute_pruning_scores, evaluate_path_accuracy

device = torch.device("cuda")
print(f"Running on {torch.cuda.get_device_name(0)}")

digits = load_digits()
X, y = digits.data.astype(np.float64), digits.target
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr); X_val = sc.transform(X_val); X_te = sc.transform(X_te)

model = FCNetwork(input_size=64, hidden_size=64, num_hidden_layers=2, num_classes=10, seed=42)
model = model.to(device)
val_acc = model.train_model(X_tr, y_tr, X_val, y_val, epochs=5, verbose=True)
print(f"val_acc={100*val_acc:.1f}%")

scores = precompute_pruning_scores(model, X_tr, y_tr, methods=["wanda"])
k_values = list(range(1, 65))
accs, normal_acc = evaluate_path_accuracy(model, X_te, y_te, k_values, scores["wanda"], "wanda")
print(f"normal_acc={100*normal_acc:.1f}%  K=32 acc={100*accs[32]:.1f}%")
print("=== SMOKE TEST PASSED ===")
PYEOF

echo "=== DONE $(date) ==="
