#!/usr/bin/env python3
"""MA-1 seed sweep: bound Jacobian under-reporting on Table I error bars.

Cell: (MNIST-28, magnitude, H=256, L=2). Five seeds trained from scratch.
Per seed: magnitude pruning sweep on DEFAULT_DENSITIES, fit 4-param logistic
(pruning.fit_sigmoid) -> (s_0, beta) + Jacobian sigma; then weight-noise sweep
on the same model (Gaussian N(0, sigma_W^2) added to hidden weights, RMS-scaled)
at a coarse sigma_W grid; refit sigmoid at each sigma_W to extract s_0(sigma_W);
fit s_0(sigma_W) = lambda + mu*sigma_W + nu*sigma_W^2 (Eq. 4) -> nu + Jacobian sigma.

Aggregate: seed-std vs mean-Jacobian-sigma for s_0, beta, nu.
Under-reporting ratio = seed_std / mean(Jacobian sigma).

CPU only. Deterministic seeding everywhere.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import curve_fit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pruning.pruning import FCNetwork, fit_sigmoid
from pruning.mnist28_scaling import load_mnist28
from unstructured_pruning.core import DEFAULT_DENSITIES, evaluate_masked_accuracy
from unstructured_pruning.methods import magnitude_masks
from temperature_pruning.noise import add_weight_noise


# --- Configuration -----------------------------------------------------------

H, L = 256, 2
INPUT_SIZE, N_CLASSES = 784, 10
EPOCHS, BS, LR = 300, 256, 1e-3
SEEDS = [0, 1, 2, 3, 4]

# Coarse sigma_W grid for the weight-noise quadratic shift (Eq. 4).
SIGMA_W_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]

OUT_DIR = _HERE
RESULTS_PATH = os.path.join(OUT_DIR, 'results.json')
DEVICE = torch.device('cpu')


# --- Per-seed pipeline -------------------------------------------------------

def fit_s0_at(model, X_te, y_te, densities, seed):
    """Magnitude sweep + sigmoid fit on `model`. Returns (popt, perr, r2, ...)."""
    mask_sets = magnitude_masks(model, densities, n_seeds=1, base_seed=seed)
    accs_stats, normal_acc = evaluate_masked_accuracy(model, X_te, y_te, mask_sets)
    s_values = sorted(accs_stats.keys())
    accs_mean = {s: accs_stats[s][0] for s in s_values}
    popt, perr, r2 = fit_sigmoid(s_values, accs_mean, normal_acc)
    return popt, perr, r2, normal_acc, s_values, accs_mean


def quadratic(sigma, lam, mu, nu):
    return lam + mu * sigma + nu * sigma ** 2


def fit_quadratic_s0(sigma_arr, s0_arr):
    """s_0(sigma_W) = lambda + mu*sigma_W + nu*sigma_W^2; perr = sqrt(diag(pcov))."""
    sigma_arr = np.asarray(sigma_arr, dtype=float)
    s0_arr = np.asarray(s0_arr, dtype=float)
    try:
        popt, pcov = curve_fit(quadratic, sigma_arr, s0_arr,
                               p0=[float(s0_arr[0]), 0.0, 1.0], maxfev=10_000)
        perr = np.sqrt(np.diag(pcov))
        resid = s0_arr - quadratic(sigma_arr, *popt)
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((s0_arr - s0_arr.mean()) ** 2).sum())
        n, p = len(s0_arr), len(popt)
        r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
              if (ss_tot > 0 and n > p) else float('nan'))
        return popt, perr, r2
    except Exception as exc:
        print(f"    quadratic fit failed: {exc}")
        return None, None, None


def run_one_seed(seed, data):
    """Train + baseline sigmoid fit + weight-noise quadratic fit for one seed."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = data

    print(f"\n--- seed={seed} ---", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    model = FCNetwork(input_size=INPUT_SIZE, hidden_size=H,
                      num_hidden_layers=L, num_classes=N_CLASSES, seed=seed)
    model = model.to(DEVICE)
    val_acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                epochs=EPOCHS, bs=BS, lr=LR, verbose=False)
    print(f"  trained: val={100*val_acc:.2f}%  [{time.time()-t0:.0f}s]", flush=True)

    # Baseline sigmoid fit (sigma_W = 0).
    popt, perr, r2, normal_acc, s_values, accs_mean = fit_s0_at(
        model, X_te, y_te, DEFAULT_DENSITIES, seed)
    if popt is None:
        raise RuntimeError(f"seed {seed}: baseline sigmoid fit failed")
    A_inf, A_0, s_0, beta = popt
    s_0_err, beta_err = float(perr[2]), float(perr[3])
    print(f"  baseline: s_0={s_0:.4f} +/- {s_0_err:.4f}  "
          f"beta={beta:.2f} +/- {beta_err:.2f}  R2={r2:.3f}", flush=True)

    # Weight-noise sweep on the SAME trained model.
    sigma_rows = []
    rng = np.random.default_rng(seed + 7919)
    for sigma_W in SIGMA_W_GRID:
        noisy = model if sigma_W == 0.0 else add_weight_noise(
            model, sigma=sigma_W, rng=rng, scale='rms')
        popt_n, perr_n, r2_n, _, _, _ = fit_s0_at(
            noisy, X_te, y_te, DEFAULT_DENSITIES, seed)
        if popt_n is None:
            print(f"    sigma_W={sigma_W:.3f}: sigmoid fit FAILED — skip")
            continue
        sigma_rows.append({
            'sigma_W': float(sigma_W),
            's_0': float(popt_n[2]), 's_0_err_jac': float(perr_n[2]),
            'beta': float(popt_n[3]), 'beta_err_jac': float(perr_n[3]),
            'sigmoid_R2': float(r2_n),
        })
        print(f"    sigma_W={sigma_W:.3f}: s_0={popt_n[2]:.4f} "
              f"beta={popt_n[3]:.2f} R2={r2_n:.3f}", flush=True)

    # Quadratic fit for nu.
    sig_arr = [r['sigma_W'] for r in sigma_rows]
    s0_arr = [r['s_0'] for r in sigma_rows]
    qpopt, qperr, qr2 = fit_quadratic_s0(sig_arr, s0_arr)
    if qpopt is None:
        lam = mu = nu = nu_err = float('nan')
    else:
        lam, mu, nu = [float(x) for x in qpopt]
        nu_err = float(qperr[2])
    qr2_str = 'n/a' if qr2 is None else f'{qr2:.3f}'
    print(f"  quadratic: lambda={lam:.4f} mu={mu:.4f} "
          f"nu={nu:.4f} +/- {nu_err:.4f}  R2={qr2_str}", flush=True)

    return {
        'seed': int(seed),
        'val_acc': float(val_acc),
        'normal_acc': float(normal_acc),
        'sigmoid_baseline': {
            'A_inf': float(A_inf), 'A_0': float(A_0),
            's_0': float(s_0), 'beta': float(beta),
            's_0_err_jac': s_0_err, 'beta_err_jac': beta_err,
            'A_inf_err_jac': float(perr[0]), 'A_0_err_jac': float(perr[1]),
            'sigmoid_R2': float(r2),
            'densities': [float(s) for s in s_values],
            'accs_mean': [float(accs_mean[s]) for s in s_values],
        },
        'sigma_W_sweep': sigma_rows,
        'quadratic': {
            'lambda': lam, 'mu': mu, 'nu': nu, 'nu_err_jac': nu_err,
            'R2': (float(qr2) if qr2 is not None else None),
            'sigma_W_grid': [float(s) for s in sig_arr],
            's_0_at_sigma': [float(s) for s in s0_arr],
        },
    }


# --- Aggregation -------------------------------------------------------------

def under_reporting(vals, jacs):
    """seed-std (ddof=1) / mean(Jacobian sigma)."""
    vals = np.asarray(vals, dtype=float)
    jacs = np.asarray(jacs, dtype=float)
    seed_std = float(vals.std(ddof=1))
    mean_jac = float(jacs.mean())
    ratio = (seed_std / mean_jac) if mean_jac > 0 else float('nan')
    return {'seed_mean': float(vals.mean()), 'seed_std': seed_std,
            'mean_jac_sigma': mean_jac, 'under_reporting_ratio': ratio}


def aggregate(records):
    s0 = under_reporting(
        [r['sigmoid_baseline']['s_0'] for r in records],
        [r['sigmoid_baseline']['s_0_err_jac'] for r in records])
    beta = under_reporting(
        [r['sigmoid_baseline']['beta'] for r in records],
        [r['sigmoid_baseline']['beta_err_jac'] for r in records])
    nu = under_reporting(
        [r['quadratic']['nu'] for r in records],
        [r['quadratic']['nu_err_jac'] for r in records])
    return {'s_0': s0, 'beta': beta, 'nu': nu}


def _persist(records):
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'meta': {
                'H': H, 'L': L, 'dataset': 'MNIST-28', 'method': 'magnitude',
                'epochs': EPOCHS, 'bs': BS, 'lr': LR,
                'densities': [float(s) for s in DEFAULT_DENSITIES],
                'sigma_W_grid': [float(s) for s in SIGMA_W_GRID],
                'seeds': list(SEEDS),
            },
            'per_seed': records,
            'aggregate': aggregate(records) if records else None,
        }, f, indent=2)


def main():
    print(f"MA-1 seed sweep on (MNIST-28, magnitude, H={H}, L={L})")
    print(f"  densities: {DEFAULT_DENSITIES} ({len(DEFAULT_DENSITIES)} pts)")
    print(f"  seeds: {SEEDS}")
    print(f"  sigma_W grid: {SIGMA_W_GRID}")

    t_total = time.time()
    data = load_mnist28()

    records = []
    for seed in SEEDS:
        records.append(run_one_seed(seed, data))
        _persist(records)  # save after each seed

    agg = aggregate(records)
    print("\n=== AGGREGATE ===")
    for k in ('s_0', 'beta', 'nu'):
        a = agg[k]
        print(f"  {k}: mean={a['seed_mean']:.4f}  "
              f"seed_std={a['seed_std']:.4f}  "
              f"mean_jac_sigma={a['mean_jac_sigma']:.4f}  "
              f"ratio={a['under_reporting_ratio']:.2f}")
    print(f"\n  Total runtime: {time.time() - t_total:.0f}s")
    print(f"  Saved: {RESULTS_PATH}")


if __name__ == '__main__':
    main()
