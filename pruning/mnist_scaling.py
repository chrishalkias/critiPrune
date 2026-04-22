#!/usr/bin/env python3
"""
Scaling Laws for Neural Network Effective Coupling Constants
=============================================================
Train networks with varying width H and depth L, extract sigmoid
parameters (K_0, beta, g_eff), and search for scaling laws relating
these to architecture.

Produces three figures:
  scaling_curves.png       - accuracy curves and parameter vs width
  scaling_laws.png         - K_0/H ratio, coupling vs depth, compressibility
  parameter_heatmaps.png   - K_0, beta, g_eff over the (H, L) grid
"""

import os
import time
import json
import warnings

import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'mnist_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Import from refactored pruning module ------------------------------------

from pruning import (
    FCNetwork, accuracy, sigmoid_fn, fit_sigmoid,
)


# --- Data ---------------------------------------------------------------------

def load_data():
    """Load sklearn digits, split into train/val/test, standardise."""
    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te = sc.transform(X_te)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# --- Path pruning (signal-based, vectorised) ----------------------------------

def _sparsify_batch(current, k, H):
    """Keep top-K entries per row by absolute value. current: (N, D, H)"""
    if k >= H:
        return current
    kth = H - k
    top_idx = np.argpartition(np.abs(current), kth, axis=2)[:, :, kth:]
    result = np.zeros_like(current)
    n_idx = np.arange(current.shape[0])[:, np.newaxis, np.newaxis]
    d_idx = np.arange(current.shape[1])[np.newaxis, :, np.newaxis]
    result[n_idx, d_idx, top_idx] = current[n_idx, d_idx, top_idx]
    return result


def evaluate_pruned_accuracy(model, X_test, y_test, k_values):
    """Evaluate accuracy for each K using signal-based (magnitude) pruning.

    Parameters
    ----------
    model   : FCNetwork
    X_test  : ndarray [N, D]
    y_test  : ndarray [N]
    k_values: list of int

    Returns
    -------
    accs       : dict {K: float}
    normal_acc : float
    """
    W = model.W
    H = model.H
    actual_logits, relu_masks_list = model.numpy_forward_with_masks(X_test)
    normal_acc = accuracy(actual_logits, y_test)

    # current: (N, D, H) — path matrix batched over all samples
    # X_test[:, :, None] is (N, D, 1), W[0].T[None] is (1, D, H)
    accs = {}
    for K in k_values:
        current = X_test[:, :, np.newaxis] * W[0].T[np.newaxis]
        current *= relu_masks_list[0][:, np.newaxis, :]
        current = _sparsify_batch(current, K, H)

        for l in range(1, model.L):
            current = current @ W[l].T  # (N, D, H) @ (H, H) → (N, D, H)
            current *= relu_masks_list[l][:, np.newaxis, :]
            current = _sparsify_batch(current, K, H)

        pl = (current @ W[-1].T).sum(axis=1)  # (N, D, C) → (N, C)
        accs[K] = accuracy(pl, y_test)

    return accs, normal_acc


# --- Sigmoid and exponential fits ---------------------------------------------

def exp_fn(K, A_inf, A_0, tau):
    """Exponential recovery: A_inf - (A_inf - A_0) * exp(-K / tau)."""
    K = np.asarray(K, dtype=float)
    return A_inf - (A_inf - A_0) * np.exp(-K / tau)


def fit_exponential(k_values, accuracies, normal_acc):
    """Fit an exponential recovery curve. Returns (popt, r2) or (None, None)."""
    k_arr = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values])
    try:
        p0 = [normal_acc, acc_arr[0], np.median(k_arr)]
        bounds = ([0, 0, 0.1], [1.0, 1.0, float(max(k_arr)) * 5])
        popt, pcov = curve_fit(exp_fn, k_arr, acc_arr, p0=p0,
                               bounds=bounds, maxfev=30000)
        resid = acc_arr - exp_fn(k_arr, *popt)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((acc_arr - acc_arr.mean()) ** 2)
        n, p = len(k_arr), len(popt)
        r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
               if (ss_tot > 0 and n > p) else float('nan'))
        return popt, r2
    except Exception:
        return None, None


# --- Scaling scan -------------------------------------------------------------

def run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Train networks over an (H, L) grid and extract sigmoid parameters.

    Returns a list of result dicts, one per successfully trained configuration.
    """
    # Small H so K_0 lands visibly in the middle of [1, H].
    # Large H on easy 64-dim digits means even K=1 achieves high accuracy,
    # hiding the sigmoid transition.
    H_values = [8, 16, 24, 32, 48, 56, 64, 96]
    L_values = [1, 2, 3, 4, 5, 7, 8, 10]

    results = []
    checkpoints = {}
    total = len(H_values) * len(L_values)
    count = 0

    for H in H_values:
        for L in L_values:
            count += 1
            t0 = time.time()
            print(f"\n  [{count}/{total}] H={H}, L={L}", end="", flush=True)

            model = FCNetwork(input_size=64, hidden_size=H,
                              num_hidden_layers=L, num_classes=10, seed=42)
            epochs = 300 if H >= 32 else 500
            val_acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                        epochs=epochs, verbose=False)
            print(f"  val={100 * val_acc:.1f}%", end="", flush=True)

            if val_acc < 0.20:
                print("  SKIP (val too low)")
                continue

            k_values = list(range(1, H + 1))
            accs, normal_acc = evaluate_pruned_accuracy(model, X_te, y_te, k_values)
            popt_sig, perr_sig, r2_sig = fit_sigmoid(k_values, accs, normal_acc)
            popt_exp, r2_exp = fit_exponential(k_values, accs, normal_acc)

            checkpoints[(H, L)] = {
                'state_dict': model.state_dict(),
                'arch': {
                    'input_size': 64,
                    'hidden_size': H,
                    'num_hidden_layers': L,
                    'num_classes': 10,
                },
                'val_acc': float(val_acc),
                'normal_acc': float(normal_acc),
            }

            n_params = sum(p.numel() for p in model.parameters())
            res = {
                'H': H, 'L': L,
                'val_acc': float(val_acc),
                'normal_acc': float(normal_acc),
                'n_params': n_params,
                'accs': {int(k): float(v) for k, v in accs.items()},
            }

            if popt_sig is not None:
                A_inf, A_0, K_0, beta = popt_sig
                g_eff = np.exp(-beta)
                res.update({
                    'sigmoid_A_inf': float(A_inf),
                    'sigmoid_A_0': float(A_0),
                    'sigmoid_K_0': float(K_0),
                    'sigmoid_beta': float(beta),
                    'sigmoid_g_eff': float(g_eff),
                    'sigmoid_R2': float(r2_sig),
                })
                print(f"  K0={K_0:.1f} beta={beta:.3f}"
                      f" g={g_eff:.3f} R2={r2_sig:.3f}", end="")
            else:
                res['sigmoid_R2'] = None
                print("  sigmoid fit FAILED", end="")

            if popt_exp is not None:
                res.update({
                    'exp_A_inf': float(popt_exp[0]),
                    'exp_A_0': float(popt_exp[1]),
                    'exp_tau': float(popt_exp[2]),
                    'exp_R2': float(r2_exp),
                })

            dt = time.time() - t0
            print(f"  [{dt:.0f}s]")
            results.append(res)

    return results, checkpoints


# --- Scaling law fitting ------------------------------------------------------

def _power_law(x, a, b):
    return a * np.power(x, b)

def _power_law_2d(HL, a, alpha, gamma):
    """K0 = a * H^alpha * L^gamma."""
    H, L = HL
    return a * np.power(H, alpha) * np.power(L, gamma)


def fit_scaling_laws(results):
    """Fit K_0 and beta as power-law functions of H and L.

    Returns a dict of fitted coefficients and R-squared values.
    """
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 5:
        print("  Not enough good fits for scaling law analysis")
        return None

    H_arr = np.array([r['H'] for r in good], dtype=float)
    L_arr = np.array([r['L'] for r in good], dtype=float)
    K0_arr = np.array([r['sigmoid_K_0'] for r in good])
    beta_arr = np.array([r['sigmoid_beta'] for r in good])
    g_arr = np.array([r['sigmoid_g_eff'] for r in good])

    scaling_results = {}

    print(f"\n{'=' * 70}")
    print(f"  SCALING LAW ANALYSIS ({len(good)} good fits, R2 > 0.80)")
    print(f"{'=' * 70}")

    # K0 = a * H^alpha * L^gamma
    try:
        popt, pcov = curve_fit(_power_law_2d, (H_arr, L_arr), K0_arr,
                               p0=[1.0, 0.5, 0.5], maxfev=10000,
                               bounds=([0, -3, -3], [1000, 3, 3]))
        K0_pred = _power_law_2d((H_arr, L_arr), *popt)
        ss_res = np.sum((K0_arr - K0_pred) ** 2)
        ss_tot = np.sum((K0_arr - K0_arr.mean()) ** 2)
        n, p = len(K0_arr), len(popt)
        r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
               if (ss_tot > 0 and n > p) else 0)
        perr = np.sqrt(np.diag(pcov))
        a, alpha, gamma = popt
        scaling_results['K0'] = {
            'a': float(a), 'alpha': float(alpha), 'gamma': float(gamma),
            'R2': float(r2),
            'formula': f'K0 = {a:.3f} * H^{alpha:.3f} * L^{gamma:.3f}',
        }
        print(f"\n  K_0 = {a:.3f} * H^{alpha:.3f} * L^{gamma:.3f}")
        print(f"       +/- ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R2 = {r2:.4f}")
    except Exception as e:
        print(f"  K_0 power-law fit failed: {e}")

    # beta = a * H^alpha * L^gamma
    try:
        popt, pcov = curve_fit(_power_law_2d, (H_arr, L_arr), beta_arr,
                               p0=[1.0, -0.5, -0.5], maxfev=10000,
                               bounds=([0, -3, -3], [100, 3, 3]))
        beta_pred = _power_law_2d((H_arr, L_arr), *popt)
        ss_res = np.sum((beta_arr - beta_pred) ** 2)
        ss_tot = np.sum((beta_arr - beta_arr.mean()) ** 2)
        n, p = len(beta_arr), len(popt)
        r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
               if (ss_tot > 0 and n > p) else 0)
        perr = np.sqrt(np.diag(pcov))
        a, alpha, gamma = popt
        scaling_results['beta'] = {
            'a': float(a), 'alpha': float(alpha), 'gamma': float(gamma),
            'R2': float(r2),
            'formula': f'beta = {a:.3f} * H^{alpha:.3f} * L^{gamma:.3f}',
        }
        print(f"\n  beta = {a:.3f} * H^{alpha:.3f} * L^{gamma:.3f}")
        print(f"       +/- ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R2 = {r2:.4f}")
    except Exception as e:
        print(f"  beta power-law fit failed: {e}")

    # g_eff = a * H^alpha * L^gamma
    try:
        popt, pcov = curve_fit(_power_law_2d, (H_arr, L_arr), g_arr,
                               p0=[0.5, 0.1, 0.1], maxfev=10000,
                               bounds=([0, -3, -3], [2, 3, 3]))
        g_pred = _power_law_2d((H_arr, L_arr), *popt)
        ss_res = np.sum((g_arr - g_pred) ** 2)
        ss_tot = np.sum((g_arr - g_arr.mean()) ** 2)
        n, p = len(g_arr), len(popt)
        r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
               if (ss_tot > 0 and n > p) else 0)
        perr = np.sqrt(np.diag(pcov))
        a, alpha, gamma = popt
        scaling_results['g_eff'] = {
            'a': float(a), 'alpha': float(alpha), 'gamma': float(gamma),
            'R2': float(r2),
            'formula': f'g = {a:.3f} * H^{alpha:.3f} * L^{gamma:.3f}',
        }
        print(f"\n  g_eff = {a:.3f} * H^{alpha:.3f} * L^{gamma:.3f}")
        print(f"       +/- ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R2 = {r2:.4f}")
    except Exception as e:
        print(f"  g_eff power-law fit failed: {e}")

    # K0/H ratio analysis
    K0_over_H = K0_arr / H_arr
    print(f"\n  K_0/H ratio statistics:")
    print(f"    mean = {K0_over_H.mean():.3f} +/- {K0_over_H.std():.3f}")
    print(f"    range = [{K0_over_H.min():.3f}, {K0_over_H.max():.3f}]")
    scaling_results['K0_over_H'] = {
        'mean': float(K0_over_H.mean()), 'std': float(K0_over_H.std()),
    }

    # Fixed-L slices: K0 vs H for each L
    print("\n  Fixed-L slices (K_0 vs H):")
    unique_L = sorted(set(r['L'] for r in good))
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if len(subset) >= 3:
            Hs = np.array([r['H'] for r in subset], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in subset])
            try:
                po, _ = curve_fit(_power_law, Hs, K0s, p0=[1, 0.5], maxfev=5000)
                pred = _power_law(Hs, *po)
                ss_r = np.sum((K0s - pred) ** 2)
                ss_t = np.sum((K0s - K0s.mean()) ** 2)
                n, p = len(K0s), len(po)
                r2v = (1 - (ss_r / (n - p)) / (ss_t / (n - 1))
                       if (ss_t > 0 and n > p) else 0)
                print(f"    L={L_val}: K_0 = {po[0]:.3f} * H^{po[1]:.3f}"
                      f"  R2={r2v:.3f}")
            except Exception:
                print(f"    L={L_val}: fit failed")

    # Fixed-H slices: K0 vs L
    print("\n  Fixed-H slices (K_0 vs L):")
    unique_H = sorted(set(r['H'] for r in good))
    for H_val in unique_H:
        subset = [r for r in good if r['H'] == H_val]
        if len(subset) >= 3:
            Ls = np.array([r['L'] for r in subset], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in subset])
            try:
                po, _ = curve_fit(_power_law, Ls, K0s, p0=[1, 0.5], maxfev=5000)
                pred = _power_law(Ls, *po)
                ss_r = np.sum((K0s - pred) ** 2)
                ss_t = np.sum((K0s - K0s.mean()) ** 2)
                n, p = len(K0s), len(po)
                r2v = (1 - (ss_r / (n - p)) / (ss_t / (n - 1))
                       if (ss_t > 0 and n > p) else 0)
                print(f"    H={H_val}: K_0 = {po[0]:.3f} * L^{po[1]:.3f}"
                      f"  R2={r2v:.3f}")
            except Exception:
                print(f"    H={H_val}: fit failed")

    print(f"{'=' * 70}")
    return scaling_results


# --- Visualisation ------------------------------------------------------------

def make_scaling_plots(results, scaling_results, output_dir=None):
    """Two figures: accuracy curves + K_0 scaling (dot+line | heatmap)."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 3:
        print("  Not enough data for plots")
        return []

    H_arr  = np.array([r['H'] for r in good], dtype=float)
    L_arr  = np.array([r['L'] for r in good], dtype=float)
    K0_arr = np.array([r['sigmoid_K_0'] for r in good])

    unique_L = sorted(set(r['L'] for r in good))
    unique_H = sorted(set(r['H'] for r in good))
    L_color = dict(zip(unique_L, plt.cm.viridis(np.linspace(0.15, 0.85, max(len(unique_L), 1)))))
    H_color = dict(zip(unique_H, plt.cm.plasma( np.linspace(0.15, 0.85, max(len(unique_H), 1)))))

    # Figure 1: Accuracy curves (top) + K_0 vs H with lines (bottom)
    n_L_panels = min(len(unique_L), 3)
    fig = plt.figure(figsize=(6 * n_L_panels, 11))
    top_axes = [fig.add_subplot(2, n_L_panels, i + 1) for i in range(n_L_panels)]
    ax_k0 = fig.add_subplot(2, 1, 2)
    fig.suptitle('Path-Pruning Accuracy Curves & $K_0$ Scaling\n'
                 '$A(K) = A_0 + (A_\\infty - A_0)/(1 + e^{-\\beta(K-K_0)})$',
                 fontsize=13, y=1.02)

    for idx, L_val in enumerate(unique_L[:n_L_panels]):
        ax = top_axes[idx]
        for r in sorted([r for r in good if r['L'] == L_val], key=lambda x: x['H']):
            k_vals = sorted(r['accs'].keys())
            acc_vals = [r['accs'][k] * 100 for k in k_vals]
            k_fine = np.linspace(1, max(k_vals), 300)
            ax.scatter(k_vals, acc_vals, s=12, color=H_color[r['H']], alpha=0.7)
            if r.get('sigmoid_R2') and r['sigmoid_R2'] > 0.80:
                fit_line = sigmoid_fn(k_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                      r['sigmoid_K_0'], r['sigmoid_beta']) * 100
                ax.plot(k_fine, fit_line, color=H_color[r['H']], lw=1.5,
                        label=f'H={r["H"]}  K₀={r["sigmoid_K_0"]:.1f}')
            ax.axhline(r['normal_acc'] * 100, color=H_color[r['H']],
                       ls=':', lw=0.5, alpha=0.4)
        ax.set_title(f'L = {L_val} layers', fontsize=11)
        ax.set_xlabel('K (paths per pixel)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    for L_val in unique_L:
        sub = sorted([r for r in good if r['L'] == L_val], key=lambda x: x['H'])
        if sub:
            ax_k0.plot([r['H'] for r in sub], [r['sigmoid_K_0'] for r in sub],
                       'o-', color=L_color[L_val], lw=1.8, ms=8,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label=f'L={L_val}', zorder=5)
    if scaling_results and 'K0' in scaling_results:
        sr = scaling_results['K0']
        H_fine = np.linspace(min(unique_H), max(unique_H), 200)
        for L_val in unique_L:
            ax_k0.plot(H_fine, sr['a'] * H_fine ** sr['alpha'] * L_val ** sr['gamma'],
                       '--', color=L_color[L_val], alpha=0.45, lw=1.2)
        formula = (f"$K_0 = {sr['a']:.3f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$  $R^2={sr['R2']:.3f}$")
        ax_k0.text(0.05, 0.95, formula, transform=ax_k0.transAxes, fontsize=9,
                   va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax_k0.set_xlabel('H (hidden size)', fontsize=11)
    ax_k0.set_ylabel('$K_0$ (inflection point)', fontsize=11)
    ax_k0.set_title('$K_0$ vs Width H', fontsize=12)
    ax_k0.legend(fontsize=8)
    ax_k0.grid(alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(output_dir, 'scaling_curves.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Figure 2: K_0 dot+line plot | K_0 heatmap
    fig, (ax_dot, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('sklearn digits — $K_0$ Scaling Law', fontsize=13, y=1.02)

    for L_val in unique_L:
        sub = sorted([r for r in good if r['L'] == L_val], key=lambda x: x['H'])
        if sub:
            Hs  = np.array([r['H'] for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            ax_dot.plot(Hs, K0s, 'o-', color=L_color[L_val], lw=2, ms=9,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=f'L={L_val}', zorder=5)
    if scaling_results and 'K0' in scaling_results:
        sr = scaling_results['K0']
        H_fine = np.linspace(min(unique_H), max(unique_H), 200)
        for L_val in unique_L:
            ax_dot.plot(H_fine, sr['a'] * H_fine ** sr['alpha'] * L_val ** sr['gamma'],
                        '--', color=L_color[L_val], alpha=0.45, lw=1.2)
        formula = (f"$K_0 = {sr['a']:.3f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$  $R^2={sr['R2']:.3f}$")
        ax_dot.text(0.05, 0.95, formula, transform=ax_dot.transAxes, fontsize=9,
                    va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax_dot.set_xlabel('H (hidden size)', fontsize=11)
    ax_dot.set_ylabel('$K_0$', fontsize=11)
    ax_dot.set_title('$K_0$ vs Width  (lines per L)', fontsize=11)
    ax_dot.legend(fontsize=8)
    ax_dot.grid(alpha=0.3)

    H_grid = sorted(set(r['H'] for r in good))
    L_grid = sorted(set(r['L'] for r in good))
    data = np.full((len(L_grid), len(H_grid)), np.nan)
    for r in good:
        data[L_grid.index(r['L']), H_grid.index(r['H'])] = r['sigmoid_K_0']
    im = ax_heat.imshow(data, aspect='auto', cmap='YlOrRd', origin='lower')
    ax_heat.set_xticks(range(len(H_grid))); ax_heat.set_xticklabels(H_grid)
    ax_heat.set_yticks(range(len(L_grid))); ax_heat.set_yticklabels(L_grid)
    ax_heat.set_xlabel('H (hidden size)', fontsize=11)
    ax_heat.set_ylabel('L (layers)', fontsize=11)
    ax_heat.set_title('$K_0$ Heatmap over $(H, L)$ Grid', fontsize=11)
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label='$K_0$')
    for i in range(len(L_grid)):
        for j in range(len(H_grid)):
            if not np.isnan(data[i, j]):
                val = data[i, j]
                c = 'white' if val > np.nanmean(data) * 1.3 else 'black'
                ax_heat.text(j, i, f'{val:.1f}', ha='center', va='center',
                             fontsize=9, fontweight='bold', color=c)

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'k0_scaling.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    return [path1, path2]


# --- Main ---------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    t_total = time.time()

    print("=" * 70)
    print("  SCALING LAW ANALYSIS FOR NEURAL NETWORK EFFECTIVE COUPLING")
    print("=" * 70)

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    print(f"  Data: Train={X_tr.shape[0]}, Val={X_val.shape[0]},"
          f" Test={X_te.shape[0]}")

    print("\n  Starting architecture scan...")
    results, checkpoints = run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te)

    with open(os.path.join(OUTPUT_DIR, 'scaling_results.json'), 'w') as f:
        clean_results = []
        for r in results:
            cr = {}
            for k, v in r.items():
                if k == 'accs':
                    cr[k] = {str(kk): vv for kk, vv in v.items()}
                elif isinstance(v, (np.floating, np.integer)):
                    cr[k] = float(v)
                else:
                    cr[k] = v
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2)

    scaling_results = fit_scaling_laws(results)

    print(f"\n{'=' * 100}")
    print(f"  {'H':>4}  {'L':>3}  {'Params':>8}  {'ValAcc':>7}  "
          f"{'A_inf':>6}  {'A_0':>6}  {'K_0':>6}  {'beta':>7}  "
          f"{'g_eff':>7}  {'R2':>6}  {'K0/H':>5}")
    print(f"{'-' * 100}")
    for r in sorted(results, key=lambda x: (x['L'], x['H'])):
        if r.get('sigmoid_R2') is not None:
            K0_H = r['sigmoid_K_0'] / r['H']
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  "
                  f"{100 * r['val_acc']:>6.1f}%  "
                  f"{100 * r['sigmoid_A_inf']:>5.1f}%  "
                  f"{100 * r['sigmoid_A_0']:>5.1f}%  "
                  f"{r['sigmoid_K_0']:>6.1f}  {r['sigmoid_beta']:>7.4f}  "
                  f"{r['sigmoid_g_eff']:>7.4f}  {r['sigmoid_R2']:>6.3f}  "
                  f"{K0_H:>5.2f}")
        else:
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  "
                  f"{100 * r['val_acc']:>6.1f}%  "
                  f"{'--':>6}  {'--':>6}  {'--':>6}  {'--':>7}  "
                  f"{'--':>7}  {'--':>6}  {'--':>5}")
    print(f"{'=' * 100}")

    print("\n  Generating visualizations...")
    plot_paths = make_scaling_plots(results, scaling_results)

    if scaling_results:
        with open(os.path.join(OUTPUT_DIR, 'scaling_laws.json'), 'w') as f:
            json.dump(scaling_results, f, indent=2)

    ckpt_path = os.path.join(OUTPUT_DIR, 'checkpoints.pt')
    torch.save(checkpoints, ckpt_path)
    print(f"\n  Saved {len(checkpoints)} model checkpoints → {ckpt_path}")
    print("  To load: ckpts = torch.load('checkpoints.pt'); arch = ckpts[(H, L)]['arch']")

    dt = time.time() - t_total
    print(f"\n  Total runtime: {dt:.0f}s")
    print("  Done!")