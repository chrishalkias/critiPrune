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
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from unstructured_pruning.base import (
    FCNetwork, accuracy,
    precompute_pruning_scores, evaluate_path_accuracy,
    sigmoid_fn, fit_sigmoid,
)


# --- Configuration ------------------------------------------------------------

SEED = 42
OUTPUT_DIR = 'cifar_figures'
BACKBONE          = 'resnet18'
FEATURE_DIM       = 512   # ResNet18 avgpool output dim
FEATURE_CACHE_DIR = os.environ.get('FEATURE_CACHE_DIR', '/tmp/cifar_features')
EPOCHS     = 300          # ResNet features converge faster than PCA
LR         = 1e-3
BS         = 128
N_K_POINTS = 60           # max K values sampled per architecture

H_VALUES = [64, 128, 256, 512]
L_VALUES = [2, 3, 5, 7, 10]

MAX_MEM_MB = 512


# --- ResNet18 feature extractor -----------------------------------------------

def extract_cnn_features(X_hwc, split_name):
    """Extract frozen pretrained ResNet18 features from raw HWC uint8 images.

    Caches to FEATURE_CACHE_DIR so re-runs skip extraction entirely.

    Parameters
    ----------
    X_hwc      : ndarray [N, H, W, C] uint8
    split_name : str  e.g. 'train', 'val', 'test'

    Returns
    -------
    feats : ndarray [N, 512] float32
    """
    import torchvision.models as models

    cache_path = os.path.join(FEATURE_CACHE_DIR, f'{BACKBONE}_{split_name}.npy')
    if os.path.exists(cache_path):
        feats = np.load(cache_path)
        print(f"  Loaded cached {split_name} features: {feats.shape}")
        return feats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.resnet18(weights='IMAGENET1K_V1')
    net.fc = nn.Identity()
    net = net.to(device).eval()

    transform = T.Compose([
        T.Resize(224, antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # HWC uint8 → CHW float [0, 1]
    X_tensor = torch.from_numpy(X_hwc.transpose(0, 3, 1, 2)).float() / 255.0

    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 256):
            batch = transform(X_tensor[i:i + 256].to(device))
            chunks.append(net(batch).cpu().numpy())

    feats = np.concatenate(chunks, axis=0)
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    np.save(cache_path, feats)
    print(f"  Extracted & cached {split_name} features: {feats.shape}")
    return feats


def make_k_values(H):
    """Evenly-spaced K sample points in [1, H], capped at N_K_POINTS."""
    if H <= N_K_POINTS:
        return list(range(1, H + 1))
    pts = np.unique(np.round(np.linspace(1, H, N_K_POINTS)).astype(int))
    return sorted(set([1] + pts.tolist() + [H]))


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
    """Load CIFAR-10, extract ResNet18 features, standardise.

    Returns
    -------
    X_tr, X_val, X_te : ndarray [N, FEATURE_DIM]  float64
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

    # Train/val split on raw HWC data
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_tr_raw, y_tr_raw, test_size=0.10,
        random_state=SEED, stratify=y_tr_raw)

    # Subsample test set for evaluation speed
    rng = np.random.default_rng(SEED)
    te_idx = rng.choice(len(y_te_raw), size=min(1500, len(y_te_raw)), replace=False)
    X_te_raw, y_te = X_te_raw[te_idx], y_te_raw[te_idx]

    # Extract frozen ResNet18 features (cached to disk)
    X_tr  = extract_cnn_features(X_tr_raw,  'train').astype(np.float64)
    X_val = extract_cnn_features(X_val_raw, 'val').astype(np.float64)
    X_te  = extract_cnn_features(X_te_raw,  'test').astype(np.float64)

    # Standardise the 512-dim embedding space
    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)

    print(f"  {BACKBONE} → {FEATURE_DIM}-dim features | "
          f"Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Test: {X_te.shape[0]}")
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

            model = FCNetwork(input_size=FEATURE_DIM, hidden_size=H,
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
    """Produce two figures for CIFAR-10 / WANDA.

    Figure 1 (cifar_scaling_curves.png):
        Top row  — accuracy curves per L slice with sigmoid fits.
        Bottom   — K_0 vs H scatter with connecting lines per L.

    Figure 2 (cifar_k0_scaling.png):
        Left  — K_0 vs H with connecting lines + power-law fit overlay.
        Right — K_0 heatmap over the (H, L) grid.
    """
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 3:
        print("  Not enough data for plots")
        return []

    H_arr  = np.array([r['H'] for r in good], dtype=float)
    L_arr  = np.array([r['L'] for r in good], dtype=float)
    K0_arr = np.array([r['sigmoid_K_0'] for r in good])

    unique_L = sorted(set(int(r['L']) for r in good))
    unique_H = sorted(set(int(r['H']) for r in good))
    cmap_L = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(unique_L), 1)))
    cmap_H = plt.cm.plasma( np.linspace(0.15, 0.85, max(len(unique_H), 1)))
    L_col  = dict(zip(unique_L, cmap_L))
    H_col  = dict(zip(unique_H, cmap_H))

    paths = []

    # ----------------------------------------------------------------
    # Figure 1: Accuracy curves (top) + K_0 vs H with lines (bottom)
    # ----------------------------------------------------------------
    n_L_panels = min(len(unique_L), 5)
    n_cols = max(n_L_panels, 1)
    fig = plt.figure(figsize=(6 * n_cols, 11))
    # Top row: one axis per L slice
    top_axes = [fig.add_subplot(2, n_cols, i + 1) for i in range(n_L_panels)]
    # Bottom: single wide axis spanning all columns
    ax_k0 = fig.add_subplot(2, 1, 2)

    fig.suptitle('CIFAR-10 / WANDA Pruning — Accuracy Curves & $K_0$ Scaling\n'
                 '$A(K) = A_0 + (A_\\infty - A_0)/(1 + e^{-\\beta(K-K_0)})$',
                 fontsize=13, y=1.02)

    for idx, Lv in enumerate(unique_L[:n_L_panels]):
        ax = top_axes[idx]
        subset = sorted([r for r in good if r['L'] == Lv], key=lambda r: r['H'])
        for r in subset:
            ks     = sorted(r['accs'].keys())
            accs_v = [r['accs'][k] * 100 for k in ks]
            k_fine = np.geomspace(max(1, min(ks)), max(ks), 300)
            ax.scatter(ks, accs_v, s=12, color=H_col[r['H']], alpha=0.7)
            if r.get('sigmoid_R2') and r['sigmoid_R2'] > 0.80:
                fit = sigmoid_fn(k_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                 r['sigmoid_K_0'], r['sigmoid_beta']) * 100
                ax.plot(k_fine, fit, color=H_col[r['H']], lw=1.5,
                        label=f'H={r["H"]}  K₀={r["sigmoid_K_0"]:.1f}')
        ax.set_title(f'L = {Lv} layers', fontsize=11)
        ax.set_xlabel('K (paths per feature)')
        ax.set_xscale('log')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3, which='both')

    # K_0 vs H with dots connected per L
    for Lv in unique_L:
        sub = sorted([r for r in good if r['L'] == Lv], key=lambda r: r['H'])
        if sub:
            Hs  = [r['H']           for r in sub]
            K0s = [r['sigmoid_K_0'] for r in sub]
            ax_k0.plot(Hs, K0s, 'o-', color=L_col[Lv], lw=1.8, ms=8,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label=f'L={Lv}', zorder=5)
    ax_k0.set_xlabel('H (hidden size)', fontsize=11)
    ax_k0.set_xscale('log')
    ax_k0.set_ylabel('$K_0$ (inflection point)', fontsize=11)
    ax_k0.set_title('$K_0$ vs Width H', fontsize=12)
    ax_k0.legend(fontsize=8)
    ax_k0.grid(alpha=0.3, which='both')

    plt.tight_layout()
    p = os.path.join(output_dir, 'cifar_scaling_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(p)
    print(f"  Saved: {p}")

    # ----------------------------------------------------------------
    # Figure 2: K_0 dot plot (with lines + power-law fit) | K_0 heatmap
    # ----------------------------------------------------------------
    fig, (ax_dot, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('CIFAR-10 / WANDA — $K_0$ Scaling Law', fontsize=13, y=1.02)

    # Left: K_0 vs H with connecting lines per L + power-law fit overlay
    for Lv in unique_L:
        sub = sorted([r for r in good if r['L'] == Lv], key=lambda r: r['H'])
        if sub:
            Hs  = np.array([r['H']           for r in sub], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in sub])
            ax_dot.plot(Hs, K0s, 'o-', color=L_col[Lv], lw=2, ms=9,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=f'L={Lv}', zorder=5)

    if scaling and 'K0' in scaling:
        sr = scaling['K0']
        H_fine = np.geomspace(min(unique_H), max(unique_H), 200)
        for Lv in unique_L:
            K0_fit = sr['a'] * H_fine ** sr['alpha'] * Lv ** sr['gamma']
            ax_dot.plot(H_fine, K0_fit, '--', color=L_col[Lv], alpha=0.45, lw=1.2)
        formula = (f"$K_0 = {sr['a']:.3f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$  $R^2={sr['R2']:.3f}$")
        ax_dot.text(0.05, 0.95, formula, transform=ax_dot.transAxes,
                    fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax_dot.set_xlabel('H (hidden size)', fontsize=11)
    ax_dot.set_xscale('log')
    ax_dot.set_ylabel('$K_0$', fontsize=11)
    ax_dot.set_title('$K_0$ vs Width  (lines per L)', fontsize=11)
    ax_dot.legend(fontsize=8)
    ax_dot.grid(alpha=0.3, which='both')

    # Right: K_0 heatmap
    H_grid = sorted(set(r['H'] for r in good))
    L_grid = sorted(set(r['L'] for r in good))
    data = np.full((len(L_grid), len(H_grid)), np.nan)
    for r in good:
        data[L_grid.index(r['L']), H_grid.index(r['H'])] = r['sigmoid_K_0']

    im = ax_heat.imshow(data, aspect='auto', cmap='YlOrRd', origin='lower')
    ax_heat.set_xticks(range(len(H_grid)))
    ax_heat.set_xticklabels(H_grid)
    ax_heat.set_yticks(range(len(L_grid)))
    ax_heat.set_yticklabels(L_grid)
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
    p = os.path.join(output_dir, 'cifar_k0_scaling.png')
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