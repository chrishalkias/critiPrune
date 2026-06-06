#!/usr/bin/env python3
"""C5: seed replicates for the pilot cells.

For each of the five pilot cells (digits H in {64, 128}, L in {2, 4};
mnist28 H in {128, 256}, L = 2), train three fresh FCNetworks with
deterministic seeds {42, 142, 242}, run the joint (s, sigma) grid,
fit the rational curve at A = 0.5, and report per-cell R^2 mean +/- std.

Crucial verdict: is the L = 4 R^2 = 0.62 outlier statistically
significant relative to seed-to-seed scatter?

Run::

    .venv/bin/python -m input_noise.extensions.seed_replicates.run
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pruning.pruning import FCNetwork                                # noqa: E402
from pruning.mnist_scaling import load_data as load_digits           # noqa: E402
from pruning.mnist28_scaling import load_mnist28                     # noqa: E402

from input_noise.core import evaluate_joint                          # noqa: E402
from input_noise.extensions._analysis import analyze_cell            # noqa: E402

# Match input_noise/run_experiment.py exactly for grid + epochs.
S_GRID_JOINT     = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00]
SIGMA_GRID_JOINT = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.30, 1.75,
                    2.25, 3.00]

CELLS = [
    ('digits',  64,  2),
    ('digits', 128,  2),
    ('digits', 128,  4),
    ('mnist28', 128, 2),
    ('mnist28', 256, 2),
]
SEEDS = [42, 142, 242]

EPOCHS_DIGITS  = 300
EPOCHS_MNIST28 = 100
BS_DIGITS      = 64
BS_MNIST28     = 256
LR             = 1e-3

OUTPUT_JSON = 'assets/input_noise/extensions/seed_replicates/results.json'


_DATA_CACHE = {}


def load(key):
    if key not in _DATA_CACHE:
        if key == 'digits':
            _DATA_CACHE[key] = load_digits()
        elif key == 'mnist28':
            _DATA_CACHE[key] = load_mnist28()
        else:
            raise ValueError(key)
    X_tr, X_val, X_te, y_tr, y_val, y_te = _DATA_CACHE[key]
    D = X_tr.shape[1]
    x2 = float((np.asarray(X_te, dtype=float) ** 2).mean())
    return X_tr, X_val, X_te, y_tr, y_val, y_te, D, x2


def train_and_sweep(dataset_key, H, L, D, X_tr, X_val, X_te,
                    y_tr, y_val, y_te, seed):
    """Train one cell with the given seed, then run the joint grid."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if dataset_key == 'digits':
        epochs, bs = EPOCHS_DIGITS, BS_DIGITS
    else:
        epochs, bs = EPOCHS_MNIST28, BS_MNIST28

    model = FCNetwork(input_size=D, hidden_size=H, num_hidden_layers=L,
                      num_classes=10, seed=int(seed))
    val_acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                epochs=epochs, bs=bs, lr=LR, verbose=False)
    model.eval()

    joint = evaluate_joint(
        model, X_te, y_te, S_GRID_JOINT, SIGMA_GRID_JOINT,
        n_mask_seeds=3, n_noise_draws=4,
        base_seed=int(seed) + 13)
    # Convert flat dict back to the {s_grid, sigma_grid, mean, std} shape.
    joint_dict = {
        's_grid':     [float(s)  for s in S_GRID_JOINT],
        'sigma_grid': [float(sg) for sg in SIGMA_GRID_JOINT],
        'mean':       [[joint[(float(s), float(sg))][0]
                        for sg in SIGMA_GRID_JOINT]
                       for s in S_GRID_JOINT],
        'std':        [[joint[(float(s), float(sg))][1]
                        for sg in SIGMA_GRID_JOINT]
                       for s in S_GRID_JOINT],
    }
    return joint_dict, float(val_acc)


def main():
    per_cell = []
    t0 = time.time()
    for (ds, H, L) in CELLS:
        print(f'\n  === {ds} H={H} L={L} ===')
        X_tr, X_val, X_te, y_tr, y_val, y_te, D, x2 = load(ds)

        per_seed = []
        for seed in SEEDS:
            t_seed = time.time()
            joint, val_acc = train_and_sweep(
                ds, H, L, D, X_tr, X_val, X_te, y_tr, y_val, y_te, seed)
            res = analyze_cell(joint, x2, [0.5])[0.5]
            per_seed.append({
                'seed':     int(seed),
                'val_acc':  val_acc,
                'sigma2_1': res['sigma2_1'],
                'R2':       res['R2'],
                'n':        res['n'],
            })
            print(f'    seed={seed}: val_acc={val_acc:.3f}  '
                  f'sigma2_1={res["sigma2_1"]:.2f}  R2={res["R2"]:.3f}  '
                  f'n={res["n"]}  ({time.time() - t_seed:.1f}s)')

        R2s = [s['R2'] for s in per_seed
               if np.isfinite(s['R2']) and s['n'] >= 2]
        s2s = [s['sigma2_1'] for s in per_seed
               if np.isfinite(s['sigma2_1']) and s['n'] >= 2]
        rec = {
            'dataset': ds, 'H': int(H), 'L': int(L), 'x2_mean': x2,
            'per_seed': per_seed,
            'R2_mean':       float(np.mean(R2s)) if R2s else float('nan'),
            'R2_std':        float(np.std(R2s, ddof=1))
                             if len(R2s) >= 2 else float('nan'),
            'sigma2_1_mean': float(np.mean(s2s)) if s2s else float('nan'),
            'sigma2_1_std':  float(np.std(s2s, ddof=1))
                             if len(s2s) >= 2 else float('nan'),
        }
        per_cell.append(rec)
        print(f'    -> R2 = {rec["R2_mean"]:.3f} +/- {rec["R2_std"]:.3f}  '
              f'(n_seeds={len(R2s)})')

    # Compare L=4 R2 to seed scatter of the L=2 cells.
    l4 = next((r for r in per_cell if r['L'] == 4), None)
    l2 = [r for r in per_cell if r['L'] == 2]
    if l4 is not None and l4['R2_mean'] is not None:
        # Pooled std of L=2 R2s as a reference scatter.
        l2_R2s = [s['R2'] for r in l2 for s in r['per_seed']
                  if np.isfinite(s['R2']) and s['n'] >= 2]
        l2_pooled_std = float(np.std(l2_R2s, ddof=1)) if len(l2_R2s) >= 2 else float('nan')
        l2_pooled_mean = float(np.mean(l2_R2s)) if l2_R2s else float('nan')
        z = ((l4['R2_mean'] - l2_pooled_mean) / l2_pooled_std
             if np.isfinite(l2_pooled_std) and l2_pooled_std > 0
             else float('nan'))
        print(f'\n  L=4 cell:  R^2 = {l4["R2_mean"]:.3f} +/- {l4["R2_std"]:.3f}')
        print(f'  L=2 cells: R^2 pooled = {l2_pooled_mean:.3f} +/- {l2_pooled_std:.3f}'
              f'  (n={len(l2_R2s)})')
        print(f'  z-score of L=4 from L=2 pooled distribution: {z:.2f}')

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({'seeds':    SEEDS,
                   'per_cell': per_cell}, f, indent=2)
    print(f'\n  Total wall: {time.time() - t0:.1f}s -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
