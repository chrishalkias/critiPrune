#!/usr/bin/env python3
"""
Scaling Laws for Neural Network Effective Coupling — CIFAR-10
==============================================================
Tests whether the scaling laws discovered on MNIST also hold on CIFAR-10.

Uses WANDA pruning only (Sun et al., 2023), the recommended method for
LLM-scale comparisons.

Requires
--------
  pruning.py (refactored PyTorch version, same directory or on PYTHONPATH)
  torchvision (for CIFAR-10 download)

Outputs
-------
  {OUTPUT_DIR}/
    cifar_scaling_curves.png     – sigmoid fits per architecture
    cifar_scaling_laws.png       – K_0/H, g vs L, compressibility
    cifar_parameter_heatmaps.png – K_0, beta, g_eff heatmaps over (H, L) grid
    cifar_scaling_results.json   – raw numerical results
    cifar_scaling_laws.json      – fitted scaling exponents
"""

import os
import sys
import time
import json
import warnings

import numpy as np
import torch
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from pruning import (
    FCNetwork, accuracy,
    precompute_pruning_scores, evaluate_path_accuracy,
    sigmoid_fn, fit_sigmoid,
)


# --- Configuration ------------------------------------------------------------

SEED = 42
OUTPUT_DIR = 'cifar_figures'
PCA_DIM = 200       # reduce CIFAR from 3072 -> 200 for tractable FC
EPOCHS = 500
LR = 1e-3
BS = 128

H_VALUES = [32, 64, 128, 256]
L_VALUES = [2, 3, 5, 7, 10]

MAX_MEM_MB = 256


# --- CIFAR-10 loading ---------------------------------------------------------

def _load_cifar10_torchvision():
    """Load CIFAR-10 via torchvision."""
    import torchvision
    ds_tr = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True,
                                          download=True)
    ds_te = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False,
                                          download=True)
    X_tr = np.array(ds_tr.data)
    y_tr = np.array(ds_tr.targets)
    X_te = np.array(ds_te.data)
    y_te = np.array(ds_te.targets)
    return X_tr, y_tr, X_te, y_te


def _load_cifar10_raw():
    """Download from the official URL and unpickle (fallback)."""
    import pickle
    import tarfile
    import tempfile
    import urllib.request

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    tmp = os.path.join(tempfile.gettempdir(), 'cifar-10-python.tar.gz')
    data_dir = os.path.join(tempfile.gettempdir(), 'cifar-10-batches-py')

    if not os.path.isdir(data_dir):
        print(f"    Downloading CIFAR-10 from {url} ...")
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp, 'r:gz') as tar:
            tar.extractall(tempfile.gettempdir())

    def _unpickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    X_tr, y_tr = [], []
    for i in range(1, 6):
        d = _unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        X_tr.append(d[b'data'])
        y_tr.append(np.array(d[b'labels']))
    X_tr = np.concatenate(X_tr).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_tr = np.concatenate(y_tr)

    d = _unpickle(os.path.join(data_dir, 'test_batch'))
    X_te = d[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_te = np.array(d[b'labels'])
    return X_tr, y_tr, X_te, y_te


def load_cifar10():
    """Load CIFAR-10, flatten, PCA-reduce, and standardise.

    Returns
    -------
    X_tr, X_val, X_te : ndarray [N, PCA_DIM]
    y_tr, y_val, y_te : ndarray [N]
    """
    X_tr_raw = None
    for loader, name in [(_load_cifar10_torchvision, 'torchvision'),
                         (_load_cifar10_raw, 'raw download')]:
        try:
            X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = loader()
            print(f"  CIFAR-10 loaded via {name}")
            break
        except Exception:
            continue

    if X_tr_raw is None:
        raise RuntimeError(
            "Cannot load CIFAR-10.  Install torchvision or ensure"
            " internet access for raw download.")

    X_tr_flat = X_tr_raw.reshape(X_tr_raw.shape[0], -1).astype(np.float64)
    X_te_flat = X_te_raw.reshape(X_te_raw.shape[0], -1).astype(np.float64)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_flat, y_tr_raw, test_size=0.10,
        random_state=SEED, stratify=y_tr_raw)
    X_te = X_te_flat
    y_te = y_te_raw

    # Subsample test set for speed
    rng = np.random.default_rng(SEED)
    te_idx = rng.choice(len(y_te), size=min(1500, len(y_te)), replace=False)
    X_te, y_te = X_te[te_idx], y_te[te_idx]

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te = sc.transform(X_te)

    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr = pca.fit_transform(X_tr)
    X_val = pca.transform(X_val)
    X_te = pca.transform(X_te)

    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: 3072 -> {PCA_DIM}  ({100 * var_explained:.1f}% variance retained)")
    print(f"  Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Test: {X_te.shape[0]}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# --- Scaling grid scan (WANDA only) -------------------------------------------

def run_cifar_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Train FC networks over the (H, L) grid, prune with WANDA,
    fit sigmoids, return list of result dicts."""
    results = []
    total = len(H_VALUES) * len(L_VALUES)
    count = 0

    for H in H_VALUES:
        for L in L_VALUES:
            count += 1
            t0 = time.time()
            print(f"\n  [{count}/{total}] H={H}, L={L}", end="", flush=True)

            model = FCNetwork(input_size=PCA_DIM, hidden_size=H,
                              num_hidden_layers=L, num_classes=10, seed=SEED)
            val_acc = model.train_model(X_tr, y_tr, X_val, y_val,
                                        epochs=EPOCHS, bs=BS, lr=LR,
                                        verbose=False)
            print(f"  val={100 * val_acc:.1f}%", end="", flush=True)

            if val_acc < 0.15:
                print("  SKIP (below chance)")
                continue

            scores = precompute_pruning_scores(
                model, X_tr, y_tr, methods=['wanda'], seed=SEED)

            k_values = list(range(1, H + 1))
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
                    'sigmoid_A_inf': float(A_inf),
                    'sigmoid_A_0': float(A_0),
                    'sigmoid_K_0': float(K_0),
                    'sigmoid_beta': float(beta),
                    'sigmoid_g_eff': float(g_eff),
                    'sigmoid_R2': float(r2),
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


# --- Scaling law fits ---------------------------------------------------------

def _power_law_2d(HL, a, alpha, gamma):
    H, L = HL
    return a * np.power(H, alpha) * np.power(L, gamma)


def _power_law_1d(x, a, b):
    return a * np.power(x, b)


def fit_scaling_laws(results, r2_threshold=0.80):
    """Fit K_0, beta, g_eff as power-law functions of H and L.

    Returns a dict of fitted coefficients and R-squared values.
    """
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > r2_threshold]
    if len(good) < 5:
        print(f"  Only {len(good)} good fits (need >= 5)"
              " -- skipping scaling law analysis")
        return None

    H = np.array([r['H'] for r in good], dtype=float)
    L = np.array([r['L'] for r in good], dtype=float)
    K0 = np.array([r['sigmoid_K_0'] for r in good])
    beta = np.array([r['sigmoid_beta'] for r in good])
    g = np.array([r['sigmoid_g_eff'] for r in good])

    scaling = {}

    print(f"\n{'=' * 70}")
    print(f"  SCALING LAW ANALYSIS  ({len(good)} configs with R2 > {r2_threshold})")
    print(f"{'=' * 70}")

    for name, arr in [('K0', K0), ('beta', beta), ('g_eff', g)]:
        try:
            p0 = [1.0, 0.5, 0.5] if name != 'beta' else [10.0, -0.5, -0.5]
            bnd = ([0, -3, -3], [1e4, 3, 3])
            popt, pcov = curve_fit(_power_law_2d, (H, L), arr,
                                   p0=p0, bounds=bnd, maxfev=10000)
            pred = _power_law_2d((H, L), *popt)
            ss_res = np.sum((arr - pred) ** 2)
            ss_tot = np.sum((arr - arr.mean()) ** 2)
            n, p = len(arr), len(popt)
            r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
                   if (ss_tot > 0 and n > p) else 0)
            perr = np.sqrt(np.diag(pcov))
            a, al, ga = popt
            scaling[name] = dict(
                a=float(a), alpha=float(al), gamma=float(ga),
                R2=float(r2),
                formula=f'{name} = {a:.3f} * H^{al:.3f} * L^{ga:.3f}')
            sym = {'K0': 'K_0', 'beta': 'beta', 'g_eff': 'g_eff'}[name]
            print(f"\n  {sym} = {a:.4f} * H^{al:.3f} * L^{ga:.3f}")
            print(f"       +/- ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
            print(f"       R2 = {r2:.4f}")
        except Exception as e:
            print(f"  {name} power-law fit failed: {e}")

    ratio = K0 / H
    scaling['K0_over_H'] = dict(mean=float(ratio.mean()), std=float(ratio.std()))
    print(f"\n  K_0/H: mean = {ratio.mean():.3f} +/- {ratio.std():.3f}")

    print("\n  Fixed-L slices (K_0 vs H):")
    for Lv in sorted(set(r['L'] for r in good)):
        sub = [r for r in good if r['L'] == Lv]
        if len(sub) >= 3:
            Hs = np.array([r['H'] for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            try:
                po, _ = curve_fit(_power_law_1d, Hs, K0s, p0=[1, 0.5],
                                  maxfev=5000)
                pred = _power_law_1d(Hs, *po)
                ss_r = np.sum((K0s - pred) ** 2)
                ss_t = np.sum((K0s - K0s.mean()) ** 2)
                r2v = 1 - ss_r / ss_t if ss_t > 0 else 0
                print(f"    L={Lv}: K_0 = {po[0]:.3f} * H^{po[1]:.3f}"
                      f"  R2={r2v:.3f}")
            except Exception:
                print(f"    L={Lv}: fit failed")

    print("\n  Fixed-H slices (K_0 vs L):")
    for Hv in sorted(set(r['H'] for r in good)):
        sub = [r for r in good if r['H'] == Hv]
        if len(sub) >= 3:
            Ls = np.array([r['L'] for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            try:
                po, _ = curve_fit(_power_law_1d, Ls, K0s, p0=[1, 0.5],
                                  maxfev=5000)
                pred = _power_law_1d(Ls, *po)
                ss_r = np.sum((K0s - pred) ** 2)
                ss_t = np.sum((K0s - K0s.mean()) ** 2)
                r2v = 1 - ss_r / ss_t if ss_t > 0 else 0
                print(f"    H={Hv}: K_0 = {po[0]:.3f} * L^{po[1]:.3f}"
                      f"  R2={r2v:.3f}")
            except Exception:
                print(f"    H={Hv}: fit failed")

    print(f"{'=' * 70}")
    return scaling


# --- Visualisation ------------------------------------------------------------

def make_plots(results, scaling, output_dir):
    """Produce three figures mirroring the MNIST output for CIFAR-10 / WANDA."""
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

    unique_L = sorted(set(int(r['L']) for r in good))
    unique_H = sorted(set(int(r['H']) for r in good))
    cmap_L = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(unique_L), 1)))
    cmap_H = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(unique_H), 1)))
    L_col = dict(zip(unique_L, cmap_L))
    H_col = dict(zip(unique_H, cmap_H))

    paths = []

    # Figure 1: Accuracy curves by architecture
    n_L_panels = min(len(unique_L), 5)
    fig, axes = plt.subplots(2, max(n_L_panels, 3),
                             figsize=(6 * max(n_L_panels, 3), 11))
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('CIFAR-10 / WANDA Pruning — Accuracy Curves by Architecture\n'
                 '$A(K) = A_0 + (A_\\infty - A_0)/(1 + e^{-\\beta(K-K_0)})$',
                 fontsize=13, y=1.02)

    for idx, Lv in enumerate(unique_L[:n_L_panels]):
        ax = axes[0, idx]
        subset = sorted([r for r in good if r['L'] == Lv],
                        key=lambda r: r['H'])
        for r in subset:
            ks = sorted(r['accs'].keys())
            accs = [r['accs'][k] * 100 for k in ks]
            k_fine = np.linspace(1, max(ks), 300)
            ax.scatter(ks, accs, s=12, color=H_col[r['H']], alpha=0.7)
            if r.get('sigmoid_R2') and r['sigmoid_R2'] > 0.80:
                fit = sigmoid_fn(k_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                 r['sigmoid_K_0'], r['sigmoid_beta']) * 100
                ax.plot(k_fine, fit, color=H_col[r['H']], lw=1.5,
                        label=f'H={r["H"]} (g={r["sigmoid_g_eff"]:.2f})')
        ax.set_title(f'L = {Lv} layers', fontsize=11)
        ax.set_xlabel('K (paths per feature)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)
    for idx in range(n_L_panels, axes.shape[1]):
        axes[0, idx].axis('off')

    for pidx, (arr, ylabel, title) in enumerate([
        (K0_arr, '$K_0$', '$K_0$ vs Width H'),
        (beta_arr, '$\\beta$', '$\\beta$ vs Width H'),
        (g_arr, '$g_{eff} = e^{-\\beta}$', 'Effective Coupling vs Width'),
    ]):
        ax = axes[1, pidx]
        for Lv in unique_L:
            mask = L_arr == Lv
            ax.scatter(H_arr[mask], arr[mask], s=80, color=L_col[Lv],
                       edgecolors='black', lw=0.5, label=f'L={Lv}', zorder=5)
        if pidx == 2:
            ax.axhline(1.0, color='red', ls=':', lw=1.5, alpha=0.5,
                       label='$g=1$')
        ax.set_xlabel('H (hidden size)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for idx in range(3, axes.shape[1]):
        axes[1, idx].axis('off')

    plt.tight_layout()
    p = os.path.join(output_dir, 'cifar_scaling_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p)
    print(f"  Saved: {p}")

    # Figure 2: Scaling law summary
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('CIFAR-10 — Scaling Laws for Effective Coupling (WANDA)',
                 fontsize=13, y=1.03)

    ax = axes[0]
    ratio = K0_arr / H_arr
    for Lv in unique_L:
        mask = L_arr == Lv
        ax.scatter(H_arr[mask], ratio[mask], s=80, color=L_col[Lv],
                   edgecolors='black', lw=0.5, label=f'L={Lv}', zorder=5)
    ax.axhline(ratio.mean(), color='gray', ls='--', lw=1.5, alpha=0.6,
               label=f'mean={ratio.mean():.2f}')
    ax.set_xlabel('H')
    ax.set_ylabel('$K_0 / H$')
    ax.set_title('Critical Path Fraction $K_0/H$')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for Hv in unique_H:
        sub = sorted([r for r in good if r['H'] == Hv], key=lambda r: r['L'])
        if len(sub) >= 2:
            Ls = [r['L'] for r in sub]
            gs = [r['sigmoid_g_eff'] for r in sub]
            ax.plot(Ls, gs, 'o-', color=H_col[Hv], lw=1.5, ms=7,
                    label=f'H={Hv}')
    ax.axhline(1.0, color='red', ls=':', lw=1.5, alpha=0.5)
    ax.set_xlabel('L (hidden layers)')
    ax.set_ylabel('$g_{eff}$')
    ax.set_title('Coupling Strength vs Depth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for r in good:
        Kh = max(1, int(r['sigmoid_K_0'] / 2))
        Kh = min(Kh, max(r['accs'].keys()))
        ah = r['accs'].get(Kh, list(r['accs'].values())[0])
        ax.scatter(r['sigmoid_g_eff'], ah * 100, s=80,
                   color=L_col[r['L']], edgecolors='black', lw=0.5, zorder=5)
    ax.set_xlabel('$g_{eff}$')
    ax.set_ylabel('Accuracy at $K = K_0/2$ (%)')
    ax.set_title('Compressibility vs Coupling')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, 'cifar_scaling_laws.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p)
    print(f"  Saved: {p}")

    # Figure 3: Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('CIFAR-10 / WANDA — Parameter Heatmaps Across Architecture Grid',
                 fontsize=13, y=1.03)

    for pidx, (param, label, cmap) in enumerate([
        ('sigmoid_K_0', '$K_0$', 'YlOrRd'),
        ('sigmoid_beta', '$\\beta$', 'YlGnBu'),
        ('sigmoid_g_eff', '$g_{eff}$', 'RdYlGn_r'),
    ]):
        ax = axes[pidx]
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
                    val = data[i, j]
                    c = 'white' if val > np.nanmean(data) * 1.3 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold', color=c)

    plt.tight_layout()
    p = os.path.join(output_dir, 'cifar_parameter_heatmaps.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p)
    print(f"  Saved: {p}")

    return paths


# --- Results table ------------------------------------------------------------

def print_results_table(results):
    """Print a formatted table of all sigmoid fit results."""
    print(f"\n{'=' * 100}")
    print("  CIFAR-10 / WANDA  --  Sigmoid fit results")
    print(f"{'=' * 100}")
    print(f"  {'H':>4}  {'L':>3}  {'Params':>8}  {'ValAcc':>7}  "
          f"{'A_inf':>6}  {'A_0':>6}  {'K_0':>6}  {'beta':>7}  "
          f"{'g_eff':>7}  {'R2':>6}  {'K0/H':>5}")
    print(f"{'-' * 100}")
    for r in sorted(results, key=lambda x: (x['L'], x['H'])):
        if r.get('sigmoid_R2') is not None:
            K0H = r['sigmoid_K_0'] / r['H']
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  "
                  f"{100 * r['val_acc']:>6.1f}%  "
                  f"{100 * r['sigmoid_A_inf']:>5.1f}%  "
                  f"{100 * r['sigmoid_A_0']:>5.1f}%  "
                  f"{r['sigmoid_K_0']:>6.1f}  {r['sigmoid_beta']:>7.4f}  "
                  f"{r['sigmoid_g_eff']:>7.4f}  {r['sigmoid_R2']:>6.3f}  "
                  f"{K0H:>5.2f}")
        else:
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  "
                  f"{100 * r['val_acc']:>6.1f}%  {'--':>6}  {'--':>6}  "
                  f"{'--':>6}  {'--':>7}  {'--':>7}  {'--':>6}  {'--':>5}")
    print(f"{'=' * 100}")


# --- Main ---------------------------------------------------------------------

if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  CIFAR-10 SCALING LAW ANALYSIS  (WANDA pruning)")
    print("=" * 70)

    X_tr, X_val, X_te, y_tr, y_val, y_te = load_cifar10()

    print(f"\n  Grid: H in {H_VALUES}  x  L in {L_VALUES}  "
          f"= {len(H_VALUES) * len(L_VALUES)} configs")
    results = run_cifar_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te)

    with open(os.path.join(OUTPUT_DIR, 'cifar_scaling_results.json'), 'w') as f:
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
        with open(os.path.join(OUTPUT_DIR, 'cifar_scaling_laws.json'), 'w') as f:
            json.dump(scaling, f, indent=2)

    print("\n  Generating visualisations ...")
    make_plots(results, scaling, OUTPUT_DIR)

    dt = time.time() - t_total
    print(f"\n  Total runtime: {dt:.0f}s")
    print("  Done!")