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

def _sparsify(current, k, H):
    """Keep top-K entries per row by absolute value."""
    if k >= H:
        return current
    s = np.abs(current)
    kth = H - k
    top_idx = np.argpartition(s, kth, axis=1)[:, kth:]
    result = np.zeros_like(current)
    rows = np.arange(current.shape[0])[:, np.newaxis]
    result[rows, top_idx] = current[rows, top_idx]
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
    N = X_test.shape[0]
    H = model.H
    actual_logits, relu_masks_list = model.numpy_forward_with_masks(X_test)
    normal_acc = accuracy(actual_logits, y_test)

    accs = {}
    bias_offset = None
    for K in k_values:
        pl = np.zeros((N, model.C))
        for n in range(N):
            x_n = X_test[n]
            current = W[0].T * x_n[:, np.newaxis]
            current *= relu_masks_list[0][n][np.newaxis, :]
            current = _sparsify(current, K, H)

            for l in range(1, model.L):
                current = current @ W[l].T
                current *= relu_masks_list[l][n][np.newaxis, :]
                current = _sparsify(current, K, H)

            pl[n] = (current @ W[-1].T).sum(axis=0)

        if K == H:
            bias_offset = actual_logits - pl
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
    H_values = [8, 16, 24, 32, 48, 56, 64, 96]
    L_values = [1, 2, 3, 4, 5, 7, 8, 10]

    results = []
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

    return results


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

def make_scaling_plots(results, scaling_results):
    """Create comprehensive scaling law visualisations (three figures)."""
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 3:
        print("  Not enough data for plots")
        return []

    H_arr = np.array([r['H'] for r in good], dtype=float)
    L_arr = np.array([r['L'] for r in good], dtype=float)
    K0_arr = np.array([r['sigmoid_K_0'] for r in good])
    beta_arr = np.array([r['sigmoid_beta'] for r in good])
    g_arr = np.array([r['sigmoid_g_eff'] for r in good])
    normal_arr = np.array([r['normal_acc'] for r in good])

    unique_L = sorted(set(r['L'] for r in good))
    unique_H = sorted(set(r['H'] for r in good))
    colors_L = plt.cm.viridis(np.linspace(0.15, 0.85, len(unique_L)))
    colors_H = plt.cm.plasma(np.linspace(0.15, 0.85, len(unique_H)))
    L_color = dict(zip(unique_L, colors_L))
    H_color = dict(zip(unique_H, colors_H))

    # Figure 1: Accuracy curves by architecture
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Path-Pruning Accuracy Curves by Architecture\n'
                 'Sigmoid Fit: $A(K) = A_0 + (A_\\infty - A_0)'
                 '/(1 + e^{-\\beta(K-K_0)})$',
                 fontsize=13, y=1.02)

    for idx, L_val in enumerate(unique_L[:3]):
        ax = axes[0, idx]
        subset = [r for r in good if r['L'] == L_val]
        for r in sorted(subset, key=lambda x: x['H']):
            k_vals = sorted(r['accs'].keys())
            acc_vals = [r['accs'][k] * 100 for k in k_vals]
            k_fine = np.linspace(1, max(k_vals), 300)
            ax.scatter(k_vals, acc_vals, s=12, color=H_color[r['H']], alpha=0.7)
            if r.get('sigmoid_R2') and r['sigmoid_R2'] > 0.80:
                fit_line = sigmoid_fn(k_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                      r['sigmoid_K_0'], r['sigmoid_beta']) * 100
                ax.plot(k_fine, fit_line, color=H_color[r['H']], lw=1.5,
                        label=f'H={r["H"]} (g={r["sigmoid_g_eff"]:.2f})')
            ax.axhline(r['normal_acc'] * 100, color=H_color[r['H']],
                        ls=':', lw=0.5, alpha=0.4)
        ax.set_title(f'L = {L_val} layers', fontsize=11)
        ax.set_xlabel('K (paths per pixel)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    # Bottom-left: K0 vs H
    ax = axes[1, 0]
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if subset:
            Hs = [r['H'] for r in subset]
            K0s = [r['sigmoid_K_0'] for r in subset]
            ax.scatter(Hs, K0s, s=80, color=L_color[L_val], edgecolors='black',
                       lw=0.5, label=f'L={L_val}', zorder=5)
    if scaling_results and 'K0' in scaling_results:
        sr = scaling_results['K0']
        for L_val in unique_L:
            H_fine = np.linspace(min(H_arr), max(H_arr), 100)
            K0_pred = sr['a'] * H_fine ** sr['alpha'] * L_val ** sr['gamma']
            ax.plot(H_fine, K0_pred, '--', color=L_color[L_val], alpha=0.5, lw=1)
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$K_0$ (inflection point)')
    ax.set_title('$K_0$ vs Width H')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-center: beta vs H
    ax = axes[1, 1]
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if subset:
            Hs = [r['H'] for r in subset]
            betas = [r['sigmoid_beta'] for r in subset]
            ax.scatter(Hs, betas, s=80, color=L_color[L_val], edgecolors='black',
                       lw=0.5, label=f'L={L_val}', zorder=5)
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$\\beta$ (growth rate)')
    ax.set_title('$\\beta$ vs Width H')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: g_eff vs H
    ax = axes[1, 2]
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if subset:
            Hs = [r['H'] for r in subset]
            gs = [r['sigmoid_g_eff'] for r in subset]
            ax.scatter(Hs, gs, s=80, color=L_color[L_val], edgecolors='black',
                       lw=0.5, label=f'L={L_val}', zorder=5)
    ax.axhline(1.0, color='red', ls=':', lw=1.5, alpha=0.5,
               label='$g=1$ (strongly coupled)')
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$g_{eff} = e^{-\\beta}$')
    ax.set_title('Effective Coupling vs Width')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'scaling_curves.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # Figure 2: Scaling law summary
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('Scaling Laws for Effective Coupling Parameters',
                 fontsize=13, y=1.03)

    ax = axes[0]
    K0_over_H = K0_arr / H_arr
    for L_val in unique_L:
        mask = L_arr == L_val
        ax.scatter(H_arr[mask], K0_over_H[mask], s=80, color=L_color[L_val],
                   edgecolors='black', lw=0.5, label=f'L={L_val}', zorder=5)
    ax.axhline(K0_over_H.mean(), color='gray', ls='--', lw=1.5, alpha=0.6,
               label=f'mean={K0_over_H.mean():.2f}')
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$K_0 / H$')
    ax.set_title('Critical Path Fraction $K_0/H$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for H_val in unique_H:
        subset = [r for r in good if r['H'] == H_val]
        if len(subset) >= 2:
            Ls = sorted([r['L'] for r in subset])
            gs = [next(r['sigmoid_g_eff'] for r in subset if r['L'] == L)
                  for L in Ls]
            ax.plot(Ls, gs, 'o-', color=H_color[H_val], lw=1.5, ms=7,
                    label=f'H={H_val}')
    ax.axhline(1.0, color='red', ls=':', lw=1.5, alpha=0.5)
    ax.set_xlabel('L (number of hidden layers)')
    ax.set_ylabel('$g_{eff}$')
    ax.set_title('Coupling Strength vs Depth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for r in good:
        K_half = max(1, int(r['sigmoid_K_0'] / 2))
        K_half = min(K_half, max(r['accs'].keys()))
        acc_half = r['accs'].get(K_half, list(r['accs'].values())[0])
        ax.scatter(r['sigmoid_g_eff'], acc_half * 100, s=80,
                   color=L_color[r['L']], edgecolors='black', lw=0.5, zorder=5)
    ax.set_xlabel('$g_{eff} = e^{-\\beta}$')
    ax.set_ylabel('Accuracy at $K = K_0/2$ (%)')
    ax.set_title('Compressibility vs Coupling')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'scaling_laws.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # Figure 3: Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Parameter Heatmaps Across Architecture Grid',
                 fontsize=13, y=1.03)

    for idx, (param, label, cmap) in enumerate([
        ('sigmoid_K_0', '$K_0$', 'YlOrRd'),
        ('sigmoid_beta', '$\\beta$', 'YlGnBu'),
        ('sigmoid_g_eff', '$g_{eff}$', 'RdYlGn_r'),
    ]):
        ax = axes[idx]
        H_grid = sorted(set(r['H'] for r in good))
        L_grid = sorted(set(r['L'] for r in good))
        data = np.full((len(L_grid), len(H_grid)), np.nan)
        for r in good:
            i = L_grid.index(r['L'])
            j = H_grid.index(r['H'])
            data[i, j] = r[param]

        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks(range(len(H_grid)))
        ax.set_xticklabels(H_grid)
        ax.set_yticks(range(len(L_grid)))
        ax.set_yticklabels(L_grid)
        ax.set_xlabel('H (hidden size)')
        ax.set_ylabel('L (layers)')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(L_grid)):
            for j in range(len(H_grid)):
                if not np.isnan(data[i, j]):
                    c = ('white' if data[i, j] > np.nanmean(data) * 1.3
                         else 'black')
                    ax.text(j, i, f'{data[i, j]:.2f}', ha='center',
                            va='center', fontsize=8, fontweight='bold', color=c)

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'parameter_heatmaps.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    return [path1, path2, path3]


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
    results = run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te)

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

    dt = time.time() - t_total
    print(f"\n  Total runtime: {dt:.0f}s")
    print("  Done!")