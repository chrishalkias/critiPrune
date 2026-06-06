#!/usr/bin/env python3
"""
Scaling Laws for Neural Network Effective Coupling — MNIST 28×28
=================================================================
Same experiment as mnist_scaling.py but on the full MNIST dataset
(28×28 = 784 input dimensions) using WANDA pruning.

Motivation
----------
mnist_scaling.py uses the sklearn 8×8 digits dataset (64-dim, ~1800 train
samples) as a fast proxy.  This script uses the real MNIST (784-dim, 60k
train samples) to test whether the sigmoid phase-transition and power-law
scaling laws hold at richer input resolution, enabling direct comparison
of exponents:

  sklearn digits:  K_0 = 0.112 * H^0.588 * L^0.915   (R²=0.95)
  CIFAR-10:        K_0 = 2.747 * H^0.782 * L^0.089   (R²=0.95)
  MNIST 28×28:     (this script)

Key differences from mnist_scaling.py
--------------------------------------
* Input dimension: 784 instead of 64 — uses batched path-tracing engine
  (evaluate_path_accuracy from pruning.py) for memory efficiency.
* Pruning method: WANDA only (Sun et al., 2023) — same as cifar_scaling.py,
  enabling direct exponent comparison across datasets.
* Architecture grid: H ∈ {64, 128, 256, 512} × L ∈ {2, 3, 5, 7, 10}
  (20 configurations).

Outputs → mnist28_figures/
  mnist28_scaling_curves.png
  mnist28_scaling_laws.png
  mnist28_parameter_heatmaps.png
  mnist28_scaling_results.json
  mnist28_scaling_laws.json
"""

import os
import time
import json
import warnings

import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from pruning import (
    FCNetwork, accuracy,
    precompute_pruning_scores, evaluate_path_accuracy,
    sigmoid_fn, fit_sigmoid,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED       = 42
OUTPUT_DIR = 'mnist28_figures'
# Dataset cache — overridable via env var so cluster jobs can use persistent $HOME storage
# instead of per-node /tmp, which is cleared between jobs.
MNIST_DATA_DIR = os.environ.get('MNIST_DATA_DIR', '/tmp/mnist28')

H_VALUES   = [64, 128, 256, 512]
L_VALUES   = [2, 3, 5, 7, 10]

EPOCHS     = 300          # MNIST converges fast
LR         = 1e-3
BS         = 256          # larger batch for 60k training set
TEST_SIZE  = 3000         # subsample test set for evaluation speed
MAX_MEM_MB = 512          # path-tracing tensor budget [B, 784, H]
N_K_POINTS = 60           # max K values sampled per architecture (all K if H<=this)

# Device selection: training runs on GPU when available.
# FCNetwork uses float32 so the L4's full ~120 TFLOPS FP32 throughput is used.
# Evaluation (path-tracing) always runs on CPU via model.W
# (.detach().cpu().numpy()), so is unaffected by device choice.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_torchvision():
    """Load MNIST via torchvision (preferred)."""
    import torchvision
    os.makedirs(MNIST_DATA_DIR, exist_ok=True)
    tr = torchvision.datasets.MNIST(root=MNIST_DATA_DIR, train=True,  download=True)
    te = torchvision.datasets.MNIST(root=MNIST_DATA_DIR, train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float64)
    y_tr = tr.targets.numpy()
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float64)
    y_te = te.targets.numpy()
    return X_tr, y_tr, X_te, y_te


def _load_sklearn():
    """Fallback: load MNIST via sklearn fetch_openml.

    ``parser='liac-arff'`` is used explicitly because ``parser='auto'``
    requires ``pandas``, which is not part of the project's hard
    dependency set. liac-arff is bundled with scikit-learn and needs
    no extra install.
    """
    from sklearn.datasets import fetch_openml
    print("  Downloading MNIST via fetch_openml (may take a while)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False,
                         parser='liac-arff')
    X = mnist.data.astype(np.float64)
    y = mnist.target.astype(int)
    X_tr, X_te = X[:60000], X[60000:]
    y_tr, y_te = y[:60000], y[60000:]
    return X_tr, y_tr, X_te, y_te


def load_mnist28():
    """Load MNIST 28×28, split into train/val/test, standardise.

    Returns
    -------
    X_tr, X_val, X_te : ndarray [N, 784]
    y_tr, y_val, y_te : ndarray [N]
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for loader, name in [(_load_torchvision, 'torchvision'),
                         (_load_sklearn,     'sklearn fetch_openml')]:
        try:
            X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = loader()
            print(f"  MNIST loaded via {name}")
            break
        except Exception as e:
            print(f"  {name} failed: {e}")
            continue
    else:
        raise RuntimeError("Cannot load MNIST — install torchvision or scikit-learn>=1.0")

    # Validation split from training set
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_raw, y_tr_raw, test_size=0.10, random_state=SEED, stratify=y_tr_raw)

    # Subsample test set for evaluation speed
    rng = np.random.default_rng(SEED)
    te_idx = rng.choice(len(y_te_raw), size=min(TEST_SIZE, len(y_te_raw)), replace=False)
    X_te = X_te_raw[te_idx]
    y_te = y_te_raw[te_idx]

    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)

    print(f"  Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Test: {X_te.shape[0]}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def make_k_values(H):
    """Return evenly-spaced K sample points in [1, H], capped at N_K_POINTS."""
    if H <= N_K_POINTS:
        return list(range(1, H + 1))
    pts = np.unique(np.round(np.linspace(1, H, N_K_POINTS)).astype(int))
    return sorted(set([1] + pts.tolist() + [H]))


# ---------------------------------------------------------------------------
# Architecture grid scan
# ---------------------------------------------------------------------------

def run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Train FC networks over (H, L) grid, prune with WANDA, fit sigmoids.

    Returns a list of result dicts.
    """
    results = []
    total = len(H_VALUES) * len(L_VALUES)
    count = 0

    for H in H_VALUES:
        for L in L_VALUES:
            count += 1
            t0 = time.time()
            print(f"\n  [{count}/{total}] H={H}, L={L}", end="", flush=True)

            model = FCNetwork(input_size=784, hidden_size=H,
                              num_hidden_layers=L, num_classes=10, seed=SEED)
            model = model.to(DEVICE)
            val_acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                        epochs=EPOCHS, bs=BS, lr=LR,
                                        verbose=False)
            print(f"  val={100 * val_acc:.1f}%", end="", flush=True)

            if val_acc < 0.20:
                print("  SKIP (below chance)")
                continue

            # WANDA importance scores
            scores = precompute_pruning_scores(
                model, X_tr, y_tr, methods=['wanda'], seed=SEED)

            k_values = make_k_values(H)
            accs, normal_acc = evaluate_path_accuracy(
                model, X_te, y_te, k_values,
                scores['wanda'], method_name='wanda',
                max_mem_mb=MAX_MEM_MB)

            popt, perr, r2 = fit_sigmoid(k_values, accs, normal_acc)

            n_params = sum(p.numel() for p in model.parameters())
            res = {
                'H': H, 'L': L,
                'val_acc': float(val_acc),
                'normal_acc': float(normal_acc),
                'n_params': n_params,
                'accs': {int(k): float(v) for k, v in accs.items()},
            }

            if popt is not None:
                A_inf, A_0, K_0, beta = popt
                g_eff = np.exp(-beta)
                res.update({
                    'sigmoid_A_inf':  float(A_inf),
                    'sigmoid_A_0':    float(A_0),
                    'sigmoid_K_0':    float(K_0),
                    'sigmoid_beta':   float(beta),
                    'sigmoid_g_eff':  float(g_eff),
                    'sigmoid_R2':     float(r2),
                    'sigmoid_perr':   [float(e) for e in perr],
                })
                print(f"  K0={K_0:.1f} beta={beta:.3f} g={g_eff:.3f}"
                      f" R2={r2:.3f}", end="")
            else:
                res['sigmoid_R2'] = None
                print("  sigmoid FAILED", end="")

            dt = time.time() - t0
            print(f"  [{dt:.0f}s]")
            results.append(res)

    return results


# ---------------------------------------------------------------------------
# Scaling law fits
# ---------------------------------------------------------------------------

def _power_law_2d(HL, a, alpha, gamma):
    H, L = HL
    return a * np.power(H, alpha) * np.power(L, gamma)


def _power_law_1d(x, a, b):
    return a * np.power(x, b)


def fit_scaling_laws(results, r2_threshold=0.80):
    """Fit K_0, beta, g_eff as power laws in H and L.

    Returns a dict with fitted coefficients, uncertainties, and R².
    """
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > r2_threshold]
    if len(good) < 5:
        print(f"  Only {len(good)} good fits (need >= 5) — skipping scaling law analysis")
        return None

    H   = np.array([r['H'] for r in good], dtype=float)
    L   = np.array([r['L'] for r in good], dtype=float)
    K0  = np.array([r['sigmoid_K_0']  for r in good])
    beta = np.array([r['sigmoid_beta'] for r in good])
    g    = np.array([r['sigmoid_g_eff'] for r in good])

    scaling = {}

    print(f"\n{'=' * 70}")
    print(f"  SCALING LAW ANALYSIS  ({len(good)} configs, R2 > {r2_threshold})")
    print(f"{'=' * 70}")

    for name, arr, p0, bnd in [
        ('K0',    K0,   [1.0,  0.5,  0.5], ([0, -3, -3], [1e4, 3, 3])),
        ('beta',  beta, [10.0, -0.5, -0.5], ([0, -3, -3], [1e4, 3, 3])),
        ('g_eff', g,    [0.5,  0.1,  0.1], ([0, -3, -3], [2,   3, 3])),
    ]:
        try:
            popt, pcov = curve_fit(_power_law_2d, (H, L), arr,
                                   p0=p0, bounds=bnd, maxfev=10000)
            pred  = _power_law_2d((H, L), *popt)
            ss_res = np.sum((arr - pred) ** 2)
            ss_tot = np.sum((arr - arr.mean()) ** 2)
            n, p   = len(arr), len(popt)
            r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
                   if (ss_tot > 0 and n > p) else 0)
            perr = np.sqrt(np.diag(pcov))
            a_fit, al, ga = popt
            scaling[name] = dict(
                a=float(a_fit), alpha=float(al), gamma=float(ga),
                R2=float(r2),
                a_err=float(perr[0]), alpha_err=float(perr[1]), gamma_err=float(perr[2]),
                formula=f'{name} = {a_fit:.4f} * H^{al:.3f} * L^{ga:.3f}')
            sym = {'K0': 'K_0', 'beta': 'beta', 'g_eff': 'g_eff'}[name]
            print(f"\n  {sym} = {a_fit:.4f} * H^{al:.3f} * L^{ga:.3f}")
            print(f"       ± ({perr[0]:.4f}, {perr[1]:.3f}, {perr[2]:.3f})")
            print(f"       R² = {r2:.4f}")
        except Exception as e:
            print(f"  {name} fit failed: {e}")

    ratio = K0 / H
    scaling['K0_over_H'] = dict(mean=float(ratio.mean()), std=float(ratio.std()))
    print(f"\n  K_0/H: mean = {ratio.mean():.3f} ± {ratio.std():.3f}")

    print("\n  Fixed-L slices (K_0 vs H):")
    for Lv in sorted(set(r['L'] for r in good)):
        sub = [r for r in good if r['L'] == Lv]
        if len(sub) >= 3:
            Hs  = np.array([r['H'] for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            try:
                po, _ = curve_fit(_power_law_1d, Hs, K0s, p0=[1, 0.5], maxfev=5000)
                pred  = _power_law_1d(Hs, *po)
                ss_r  = np.sum((K0s - pred) ** 2)
                ss_t  = np.sum((K0s - K0s.mean()) ** 2)
                n, p  = len(K0s), len(po)
                r2v   = (1 - (ss_r / (n - p)) / (ss_t / (n - 1))
                          if (ss_t > 0 and n > p) else 0)
                print(f"    L={Lv}: K_0 = {po[0]:.3f} * H^{po[1]:.3f}  R²={r2v:.3f}")
            except Exception:
                print(f"    L={Lv}: fit failed")

    print("\n  Fixed-H slices (K_0 vs L):")
    for Hv in sorted(set(r['H'] for r in good)):
        sub = [r for r in good if r['H'] == Hv]
        if len(sub) >= 3:
            Ls  = np.array([r['L'] for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            try:
                po, _ = curve_fit(_power_law_1d, Ls, K0s, p0=[1, 0.5], maxfev=5000)
                pred  = _power_law_1d(Ls, *po)
                ss_r  = np.sum((K0s - pred) ** 2)
                ss_t  = np.sum((K0s - K0s.mean()) ** 2)
                n, p  = len(K0s), len(po)
                r2v   = (1 - (ss_r / (n - p)) / (ss_t / (n - 1))
                          if (ss_t > 0 and n > p) else 0)
                print(f"    H={Hv}: K_0 = {po[0]:.3f} * L^{po[1]:.3f}  R²={r2v:.3f}")
            except Exception:
                print(f"    H={Hv}: fit failed")

    print(f"{'=' * 70}")
    return scaling


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_plots(results, scaling, output_dir):
    """Two figures: accuracy curves + K_0 scaling (dot+line | heatmap)."""
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 3:
        print("  Not enough data for plots")
        return []

    unique_L = sorted(set(int(r['L']) for r in good))
    unique_H = sorted(set(int(r['H']) for r in good))
    L_col = dict(zip(unique_L, plt.cm.viridis(np.linspace(0.15, 0.85, max(len(unique_L), 1)))))
    H_col = dict(zip(unique_H, plt.cm.plasma( np.linspace(0.15, 0.85, max(len(unique_H), 1)))))

    paths = []
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Figure 1: Accuracy curves (top) + K_0 vs H with lines (bottom)
    # ----------------------------------------------------------------
    n_panels = min(len(unique_L), 5)
    fig = plt.figure(figsize=(6 * n_panels, 11))
    top_axes = [fig.add_subplot(2, n_panels, i + 1) for i in range(n_panels)]
    ax_k0 = fig.add_subplot(2, 1, 2)
    fig.suptitle(
        'MNIST 28×28 / WANDA Pruning — Accuracy Curves & $K_0$ Scaling\n'
        '$A(K) = A_0 + (A_\\infty - A_0)/(1 + e^{-\\beta(K-K_0)})$',
        fontsize=13, y=1.02)

    for idx, Lv in enumerate(unique_L[:n_panels]):
        ax = top_axes[idx]
        for r in sorted([r for r in good if r['L'] == Lv], key=lambda r: r['H']):
            ks = sorted(r['accs'].keys())
            accs_v = [r['accs'][k] * 100 for k in ks]
            k_fine = np.geomspace(max(1, min(ks)), max(ks), 300)
            ax.scatter(ks, accs_v, s=12, color=H_col[r['H']], alpha=0.7)
            if r.get('sigmoid_R2') and r['sigmoid_R2'] > 0.80:
                if r.get('sigmoid_perr'):
                    ax.axvspan(r['sigmoid_K_0'] - r['sigmoid_perr'][2],
                               r['sigmoid_K_0'] + r['sigmoid_perr'][2],
                               alpha=0.08, color=H_col[r['H']])
                fit = sigmoid_fn(k_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                 r['sigmoid_K_0'], r['sigmoid_beta']) * 100
                ax.plot(k_fine, fit, color=H_col[r['H']], lw=1.5,
                        label=f'H={r["H"]}  K₀={r["sigmoid_K_0"]:.1f}')
            ax.axhline(r['normal_acc'] * 100, color=H_col[r['H']],
                       ls=':', lw=0.5, alpha=0.4)
        ax.set_title(f'L = {Lv} layers', fontsize=11)
        ax.set_xlabel('K (neurons kept)')
        ax.set_xscale('log')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3, which='both')

    for Lv in unique_L:
        sub = sorted([r for r in good if r['L'] == Lv], key=lambda r: r['H'])
        if sub:
            ax_k0.plot([r['H'] for r in sub], [r['sigmoid_K_0'] for r in sub],
                       'o-', color=L_col[Lv], lw=1.8, ms=8,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label=f'L={Lv}', zorder=5)
    if scaling and 'K0' in scaling:
        sr = scaling['K0']
        H_fine = np.geomspace(min(unique_H), max(unique_H), 200)
        for Lv in unique_L:
            ax_k0.plot(H_fine, sr['a'] * H_fine ** sr['alpha'] * Lv ** sr['gamma'],
                       '--', color=L_col[Lv], alpha=0.45, lw=1.2)
        formula = (f"$K_0 = {sr['a']:.3f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$  $R^2={sr['R2']:.3f}$")
        ax_k0.text(0.05, 0.95, formula, transform=ax_k0.transAxes, fontsize=9,
                   va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax_k0.set_xlabel('H (hidden size)', fontsize=11)
    ax_k0.set_xscale('log')
    ax_k0.set_ylabel('$K_0$ (inflection point)', fontsize=11)
    ax_k0.set_title('$K_0$ vs Width H', fontsize=12)
    ax_k0.legend(fontsize=8)
    ax_k0.grid(alpha=0.3, which='both')

    plt.tight_layout()
    p = os.path.join(output_dir, 'mnist28_scaling_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p)
    print(f"  Saved: {p}")

    # ----------------------------------------------------------------
    # Figure 2: K_0 dot+line plot | K_0 heatmap
    # ----------------------------------------------------------------
    fig, (ax_dot, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('MNIST 28×28 / WANDA — $K_0$ Scaling Law', fontsize=13, y=1.02)

    for Lv in unique_L:
        sub = sorted([r for r in good if r['L'] == Lv], key=lambda r: r['H'])
        if sub:
            Hs  = np.array([r['H'] for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            ax_dot.plot(Hs, K0s, 'o-', color=L_col[Lv], lw=2, ms=9,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=f'L={Lv}', zorder=5)
    if scaling and 'K0' in scaling:
        sr = scaling['K0']
        H_fine = np.geomspace(min(unique_H), max(unique_H), 200)
        for Lv in unique_L:
            ax_dot.plot(H_fine, sr['a'] * H_fine ** sr['alpha'] * Lv ** sr['gamma'],
                        '--', color=L_col[Lv], alpha=0.45, lw=1.2)
        formula = (f"$K_0 = {sr['a']:.3f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$  $R^2={sr['R2']:.3f}$")
        ax_dot.text(0.05, 0.95, formula, transform=ax_dot.transAxes, fontsize=9,
                    va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax_dot.set_xlabel('H (hidden size)', fontsize=11)
    ax_dot.set_xscale('log')
    ax_dot.set_ylabel('$K_0$', fontsize=11)
    ax_dot.set_title('$K_0$ vs Width  (lines per L)', fontsize=11)
    ax_dot.legend(fontsize=8)
    ax_dot.grid(alpha=0.3, which='both')

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
    p = os.path.join(output_dir, 'mnist28_k0_scaling.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p)
    print(f"  Saved: {p}")

    return paths


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(results):
    print(f"\n{'=' * 100}")
    print("  MNIST 28×28 / WANDA  --  Sigmoid fit results")
    print(f"{'=' * 100}")
    print(f"  {'H':>4}  {'L':>3}  {'Params':>8}  {'ValAcc':>7}  "
          f"{'A_inf':>6}  {'A_0':>6}  {'K_0':>6}  {'±':>5}  "
          f"{'beta':>7}  {'g_eff':>7}  {'R2':>6}  {'K0/H':>5}")
    print(f"{'-' * 100}")
    for r in sorted(results, key=lambda x: (x['L'], x['H'])):
        if r.get('sigmoid_R2') is not None:
            K0H  = r['sigmoid_K_0'] / r['H']
            perr = r.get('sigmoid_perr', [0]*4)
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  "
                  f"{100 * r['val_acc']:>6.1f}%  "
                  f"{100 * r['sigmoid_A_inf']:>5.1f}%  "
                  f"{100 * r['sigmoid_A_0']:>5.1f}%  "
                  f"{r['sigmoid_K_0']:>6.1f}  {perr[2]:>5.2f}  "
                  f"{r['sigmoid_beta']:>7.4f}  {r['sigmoid_g_eff']:>7.4f}  "
                  f"{r['sigmoid_R2']:>6.3f}  {K0H:>5.2f}")
        else:
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  "
                  f"{100 * r['val_acc']:>6.1f}%  {'--':>6}  {'--':>6}  "
                  f"{'--':>6}  {'--':>5}  {'--':>7}  {'--':>7}  {'--':>6}  {'--':>5}")
    print(f"{'=' * 100}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    t_total = time.time()

    print("=" * 70)
    print("  MNIST 28×28 SCALING LAW ANALYSIS  (WANDA pruning)")
    print("=" * 70)
    print(f"  Device: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else ''))
    print("  Training on GPU; path-tracing evaluation on CPU (numpy).")

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_mnist28()

    print(f"\n  Grid: H in {H_VALUES}  x  L in {L_VALUES}"
          f"  = {len(H_VALUES) * len(L_VALUES)} configs")
    results = run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te)

    # Save raw results
    with open(os.path.join(OUTPUT_DIR, 'mnist28_scaling_results.json'), 'w') as f:
        clean = []
        for r in results:
            cr = {}
            for k, v in r.items():
                if k == 'accs':
                    cr[k] = {str(kk): vv for kk, vv in v.items()}
                elif isinstance(v, (np.floating, np.integer)):
                    cr[k] = float(v)
                else:
                    cr[k] = v
            clean.append(cr)
        json.dump(clean, f, indent=2)

    print_results_table(results)

    scaling = fit_scaling_laws(results)
    if scaling:
        with open(os.path.join(OUTPUT_DIR, 'mnist28_scaling_laws.json'), 'w') as f:
            json.dump(scaling, f, indent=2)

    print("\n  Generating visualisations ...")
    make_plots(results, scaling, OUTPUT_DIR)

    dt = time.time() - t_total
    print(f"\n  Total runtime: {dt:.0f}s")
    print("  Done!")
