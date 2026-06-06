#!/usr/bin/env python3
"""Pilot experiment: input noise vs. pruning iso-accuracy on FC ReLU nets.

For each (dataset, H, L) cell we

  1. train a fresh FCNetwork with the same hyperparameters as the
     scaling-law scripts in ``unstructured_pruning/``;
  2. sweep ``A(s)`` under random Bernoulli pruning (s-axis sweep);
  3. sweep ``A(sigma)`` with N(0, sigma^2) added to the inputs of the
     unpruned model;
  4. sweep the joint (s, sigma) grid;
  5. fit sigmoids to the two 1-D sweeps and dump everything to
     ``input_noise/results.json``.

The downstream plots and writeup consume only that JSON; this driver
is the only thing that touches torch.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pruning.mnist_scaling import load_data as load_digits        # noqa: E402
from pruning.mnist28_scaling import load_mnist28                  # noqa: E402
from pruning.pruning import FCNetwork                             # noqa: E402

from input_noise.core import (                                    # noqa: E402
    evaluate_joint,
    evaluate_noisy_accuracy,
    evaluate_pruned_accuracy,
    fit_sigmoid_1d,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
S_GRID_FINE = sorted({0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                      0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                      0.80, 0.85, 0.90, 0.95, 1.00})  # 21 points

# Input-noise grid: covers from clean to past-saturation. Inputs are
# standardised (std ~ 1) so sigma = 1 already injects unit-variance noise.
SIGMA_GRID_FINE = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25,
                   0.30, 0.40, 0.50, 0.60, 0.75, 1.00, 1.25, 1.50,
                   2.00, 3.00]  # 18 points

S_GRID_JOINT     = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00]
SIGMA_GRID_JOINT = [0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.30, 1.75,
                    2.25, 3.00]

CELLS = [
    # (dataset_key, H, L)
    ('digits',  64,  2),
    ('digits', 128,  2),
    ('digits', 128,  4),
    ('mnist28', 128, 2),
    ('mnist28', 256, 2),
]

EPOCHS_DIGITS  = 300
EPOCHS_MNIST28 = 100  # MNIST trains faster per epoch
BS_DIGITS      = 64
BS_MNIST28     = 256
LR             = 1e-3
SEED           = 42

OUTPUT_JSON = 'input_noise/results.json'
CKPT_DIR    = 'input_noise/checkpoints'


# ---------------------------------------------------------------------------
# Per-dataset wrappers
# ---------------------------------------------------------------------------
def load_dataset(key):
    """Return (X_tr, X_val, X_te, y_tr, y_val, y_te, input_size, x2_mean)."""
    if key == 'digits':
        X_tr, X_val, X_te, y_tr, y_val, y_te = load_digits()
        D = 64
    elif key == 'mnist28':
        X_tr, X_val, X_te, y_tr, y_val, y_te = load_mnist28()
        D = 784
    else:
        raise ValueError(f'unknown dataset {key}')
    x2_mean = float((np.asarray(X_te, dtype=float) ** 2).mean())
    return X_tr, X_val, X_te, y_tr, y_val, y_te, D, x2_mean


def train_or_load(dataset_key, H, L, D, X_tr, X_val, y_tr, y_val):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f'{dataset_key}_H{H}_L{L}.pt')
    if dataset_key == 'digits':
        epochs, bs = EPOCHS_DIGITS, BS_DIGITS
    else:
        epochs, bs = EPOCHS_MNIST28, BS_MNIST28

    torch.manual_seed(SEED)
    model = FCNetwork(input_size=D, hidden_size=H, num_hidden_layers=L,
                      num_classes=10, seed=SEED)

    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        val_acc = float(ckpt.get('val_acc', float('nan')))
        print(f'    loaded {path}  (val_acc cached = {val_acc:.4f})')
        return model, val_acc

    t0 = time.time()
    val_acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                epochs=epochs, bs=bs, lr=LR, verbose=False)
    print(f'    trained in {time.time() - t0:.1f}s  val_acc={val_acc:.4f}')
    torch.save({'state_dict': model.state_dict(), 'val_acc': float(val_acc)},
               path)
    return model, val_acc


# ---------------------------------------------------------------------------
# Per-cell pipeline
# ---------------------------------------------------------------------------
def run_cell(dataset_key, H, L):
    print(f'\n  === {dataset_key}  H={H}  L={L} ===')
    data = load_dataset(dataset_key)
    X_tr, X_val, X_te, y_tr, y_val, y_te, D, x2_mean = data
    print(f'    D={D}  N_te={X_te.shape[0]}  <x^2>={x2_mean:.4f}')

    model, val_acc = train_or_load(
        dataset_key, H, L, D, X_tr, X_val, y_tr, y_val)
    model.eval()

    # ------------- Sweep 1: pruning -----------------------------------
    print('    pruning sweep ...')
    s_accs, normal_acc = evaluate_pruned_accuracy(
        model, X_te, y_te, S_GRID_FINE,
        n_mask_seeds=4, base_seed=SEED)
    s_sorted = sorted(s_accs)
    s_means  = [s_accs[s][0] for s in s_sorted]
    s_stds   = [s_accs[s][1] for s in s_sorted]
    fit_s = fit_sigmoid_1d(s_sorted, s_means, normal_acc)
    print(f'      A(s=1)={normal_acc:.4f}  s_0={fit_s[2]:.3f}  '
          f'beta={fit_s[3]:.2f}  R2={fit_s[4]:.3f}')

    # ------------- Sweep 2: input noise -------------------------------
    print('    input-noise sweep ...')
    n_accs = evaluate_noisy_accuracy(
        model, X_te, y_te, SIGMA_GRID_FINE,
        n_draws=10, base_seed=SEED + 7)
    sig_sorted = sorted(n_accs)
    sig_means  = [n_accs[s][0] for s in sig_sorted]
    sig_stds   = [n_accs[s][1] for s in sig_sorted]
    # Fit in sigma (descending sigmoid: flip sign or accept that beta < 0).
    fit_sigma  = fit_sigmoid_1d(sig_sorted, sig_means, normal_acc)
    fit_sigma2 = fit_sigmoid_1d([x * x for x in sig_sorted], sig_means,
                                normal_acc)
    print(f'      sigma_0={fit_sigma[2]:.3f}  R2(sigma)={fit_sigma[4]:.3f}  '
          f'R2(sigma^2)={fit_sigma2[4]:.3f}')

    # ------------- Sweep 3: joint (s, sigma) --------------------------
    print('    joint (s, sigma) grid ...')
    joint = evaluate_joint(
        model, X_te, y_te, S_GRID_JOINT, SIGMA_GRID_JOINT,
        n_mask_seeds=3, n_noise_draws=4, base_seed=SEED + 13)

    return {
        'dataset':    dataset_key,
        'H':          int(H),
        'L':          int(L),
        'D':          int(D),
        'x2_mean':    float(x2_mean),
        'val_acc':    float(val_acc),
        'normal_acc': float(normal_acc),
        'pruning': {
            's':          [float(s) for s in s_sorted],
            'accs_mean':  s_means,
            'accs_std':   s_stds,
            'fit':        {
                'A_inf':  fit_s[0], 'A_0': fit_s[1],
                's_0':    fit_s[2], 'beta': fit_s[3], 'R2': fit_s[4],
            },
        },
        'noise': {
            'sigma':      [float(s) for s in sig_sorted],
            'accs_mean':  sig_means,
            'accs_std':   sig_stds,
            'fit_in_sigma': {
                'A_inf':  fit_sigma[0], 'A_0': fit_sigma[1],
                'sigma_0': fit_sigma[2], 'beta': fit_sigma[3],
                'R2': fit_sigma[4],
            },
            'fit_in_sigma2': {
                'A_inf':  fit_sigma2[0], 'A_0': fit_sigma2[1],
                'sigma2_0': fit_sigma2[2], 'beta': fit_sigma2[3],
                'R2': fit_sigma2[4],
            },
        },
        'joint': {
            's_grid':     [float(s) for s in S_GRID_JOINT],
            'sigma_grid': [float(s) for s in SIGMA_GRID_JOINT],
            'mean':       [[joint[(float(s), float(sg))][0]
                            for sg in SIGMA_GRID_JOINT]
                           for s in S_GRID_JOINT],
            'std':        [[joint[(float(s), float(sg))][1]
                            for sg in SIGMA_GRID_JOINT]
                           for s in S_GRID_JOINT],
        },
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    results = []
    t0 = time.time()
    for key, H, L in CELLS:
        results.append(run_cell(key, H, L))
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'\n  Total wall: {time.time() - t0:.1f}s  '
          f'-> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
