#!/usr/bin/env python3
r"""Validation-loss vs. critical-density scaling.

For every $(H, L)$ configuration of a chosen dataset/method this script

    1. loads each saved checkpoint $(H, L, r)$ from
       ``unstructured_pruning/checkpoints/unstructured_figures_<dataset>_<method>/``,
    2. builds pruning masks at every density $s \in$ ``DEFAULT_DENSITIES``
       using the same routine as the scaling experiment,
    3. measures the validation cross-entropy
       $\mathcal{L}_{\mathrm{val}}(s)$ for each masked model,
    4. pools all per-density / per-mask-seed / per-repeat losses into a
       single ``(loss_mean, loss_std)`` summary for that $(H, L)$ cell,
    5. reads the fitted $s_0$ from ``scaling_results.json`` and produces a
       publication-quality scatter of $\langle \mathcal{L}_{\mathrm{val}}
       \rangle$ versus $s_0$ across all $\sim 10^2$ cells.

The script is dataset-agnostic; the default invocation runs only the
MNIST-28 dataset (3 methods $\rightarrow$ 3 plots) so the user can
inspect the result quickly.  Per-cell losses are cached as JSON so
re-renders skip the expensive forward passes.

Outputs land in
``assets/unstructured_pruning/loss_scaling/``.

Usage
-----
    python -m unstructured_pruning.loss_scaling
    python -m unstructured_pruning.loss_scaling --datasets mnist28 cifar_pca
    python -m unstructured_pruning.loss_scaling --force      # ignore cache
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unstructured_pruning.core import (  # noqa: E402
    apply_mask, DEFAULT_DENSITIES, CHECKPOINT_BASE, load_fc_checkpoint,
)
from unstructured_pruning.methods import (  # noqa: E402
    UNSTRUCTURED_METHODS, random_masks, magnitude_masks, wanda_masks,
)


# ---------------------------------------------------------------------------
# Dataset registry (loader + metadata).  Mirrors more_combinations.py.
# ---------------------------------------------------------------------------
DATASETS = {
    'sklearn': {
        'data_module':   'pruning.mnist_scaling',
        'data_loader':   'load_data',
        'dataset_label': 'sklearn digits',
    },
    'mnist28': {
        'data_module':   'pruning.mnist28_scaling',
        'data_loader':   'load_mnist28',
        'dataset_label': r'MNIST-28',
    },
    'cifar_pca': {
        'data_module':   'unstructured_pruning.cifar_scaling',
        'data_loader':   'load_cifar_pca',
        'dataset_label': r'CIFAR-10 + PCA(200)',
    },
    'cifar_resnet': {
        'data_module':   'pruning.cifar_scaling',
        'data_loader':   'load_cifar10',
        'dataset_label': r'CIFAR-10 + ResNet18',
    },
}


# ---------------------------------------------------------------------------
# Style: real LaTeX where available.
# ---------------------------------------------------------------------------
def _configure_style():
    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex':            have_latex,
        'font.family':            'serif',
        'font.serif':             ['Computer Modern Roman', 'CMU Serif',
                                   'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':       'cm',
        'mathtext.rm':            'serif',
        'axes.labelsize':         12,
        'axes.titlesize':         13,
        'axes.linewidth':         0.7,
        'xtick.labelsize':        10,
        'ytick.labelsize':        10,
        'legend.fontsize':        9,
        'legend.frameon':         True,
        'legend.framealpha':      0.92,
        'legend.edgecolor':       '0.6',
        'figure.titlesize':       14,
        'figure.dpi':             120,
        'savefig.dpi':            300,
        'savefig.bbox':           'tight',
        'savefig.pad_inches':     0.08,
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}'
        )
    return have_latex


# ---------------------------------------------------------------------------
# Data + mask helpers
# ---------------------------------------------------------------------------
def _load_data(cfg):
    mod = importlib.import_module(cfg['data_module'])
    return getattr(mod, cfg['data_loader'])()


def _build_mask_sets(method, model, densities, base_seed, X_calib,
                     n_seeds=3):
    if method == 'random':
        return random_masks(model, densities, n_seeds=n_seeds,
                            base_seed=base_seed)
    if method == 'magnitude':
        return magnitude_masks(model, densities, n_seeds=1,
                               base_seed=base_seed)
    if method == 'wanda':
        return wanda_masks(model, densities, X_calib,
                           n_seeds=1, base_seed=base_seed)
    raise ValueError(f"unknown method: {method}")


def _val_loss_per_density(model, X_val, y_val, mask_sets, device):
    """Return ``{s: list_of_loss_values}`` over all mask seeds for one model."""
    X_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(np.asarray(y_val), dtype=torch.long, device=device)
    out = {}
    for s, seed_masks in mask_sets.items():
        per_seed = []
        for masks in seed_masks:
            pruned = apply_mask(model, masks).to(device)
            pruned.eval()
            with torch.no_grad():
                logits = pruned(X_t)
                per_seed.append(float(F.cross_entropy(logits, y_t).item()))
        out[float(s)] = per_seed
    return out


# ---------------------------------------------------------------------------
# Per-config loss computation (cached)
# ---------------------------------------------------------------------------
def compute_per_config(dataset, method, *, force=False, cache_root):
    """Compute or read cached ``(H, L) -> loss/s_0`` summary."""
    cfg = DATASETS[dataset]
    cache_dir = os.path.join(cache_root, f'{dataset}_{method}')
    os.makedirs(cache_dir, exist_ok=True)
    cache_p = os.path.join(cache_dir, 'loss_per_config.json')

    if not force and os.path.isfile(cache_p):
        with open(cache_p) as f:
            cached = json.load(f)
        print(f'  [cache] {cache_p}  ({len(cached)} cells)')
        return list(cached.values())

    # -- need to actually compute --
    figures_dir = os.path.join(
        'unstructured_pruning', 'figures',
        f'unstructured_figures_{dataset}_{method}')
    sr_p = os.path.join(figures_dir, 'scaling_results.json')
    if not os.path.isfile(sr_p):
        print(f'  no scaling_results.json at {sr_p}')
        return []

    rows = json.load(open(sr_p))
    rows = [r for r in rows if r.get('sigmoid_s_0') is not None]
    if not rows:
        return []

    print(f'  Loading data for {dataset} ...')
    data = _load_data(cfg)
    X_tr, X_val, _X_te, _y_tr, y_val, _y_te = data

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')
    print(f'  device: {device}   |  cells: {len(rows)}')

    ckpt_dir = os.path.join(
        CHECKPOINT_BASE, f'unstructured_figures_{dataset}_{method}')

    per_repeat = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        H, L = int(row['H']), int(row['L'])
        r = int(row.get('repeat', 0))
        s0 = float(row['sigmoid_s_0'])
        ckpt_p = os.path.join(ckpt_dir, f'H{H}_L{L}_r{r}.pt')
        if not os.path.isfile(ckpt_p):
            print(f'    [{i:3d}/{len(rows)}] H={H:4d} L={L:2d} r={r}  '
                  f'no checkpoint - skip')
            continue

        model, ckpt = load_fc_checkpoint(ckpt_p)
        seed_r = int(ckpt.get('train_seed', 42 + 1000 * r))
        mask_sets = _build_mask_sets(method, model, DEFAULT_DENSITIES,
                                     base_seed=seed_r, X_calib=X_tr,
                                     n_seeds=3)
        model = model.to(device)
        loss_curve = _val_loss_per_density(
            model, X_val, y_val, mask_sets, device)

        # pool every (density, mask-seed) loss into one flat list
        flat = [v for vs in loss_curve.values() for v in vs]
        per_repeat.append({
            'H': H, 'L': L, 'repeat': r, 'sigmoid_s_0': s0,
            'losses': flat,
        })
        if i % 10 == 0 or i == len(rows):
            dt = time.time() - t0
            print(f'    [{i:3d}/{len(rows)}] H={H:4d} L={L:2d} r={r}  '
                  f'loss_mu={np.mean(flat):.3f}  s0={s0:.3f}  '
                  f'(elapsed {dt:.0f}s)')

    # aggregate per (H, L) across repeats: pool every loss, every s_0
    bins = defaultdict(lambda: {'losses': [], 's0s': []})
    for p in per_repeat:
        bins[(p['H'], p['L'])]['losses'].extend(p['losses'])
        bins[(p['H'], p['L'])]['s0s'].append(p['sigmoid_s_0'])

    per_config = {}
    for (H, L), d in bins.items():
        L_arr = np.asarray(d['losses'])
        S_arr = np.asarray(d['s0s'])
        # Loss distribution is heavy-tailed (bounded below by 0, unbounded
        # above), so symmetric mean +/- std bars dip negative when std > mean.
        # Use percentile-based asymmetric bars instead — Gaussian-equivalent
        # 16th/84th plus 5th/95th for inner/outer bands.
        p16, p50, p84 = (np.percentile(L_arr, [16, 50, 84])
                         if len(L_arr) > 1
                         else (float(L_arr[0]),) * 3)
        per_config[f'{H},{L}'] = {
            'H': int(H), 'L': int(L),
            'n_repeats':  int(len(S_arr)),
            'n_loss_obs': int(len(L_arr)),
            'loss_mean':   float(L_arr.mean()),
            'loss_median': float(p50),
            'loss_p16':    float(p16),
            'loss_p84':    float(p84),
            'loss_std':    float(L_arr.std()) if len(L_arr) > 1 else 0.0,
            's0_mean':     float(S_arr.mean()),
            's0_std':      float(S_arr.std()) if len(S_arr) > 1 else 0.0,
        }

    with open(cache_p, 'w') as f:
        json.dump(per_config, f, indent=2)
    print(f'  Saved cache: {cache_p}')
    return list(per_config.values())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss_vs_s0(rows, output_path, *, dataset_label, method):
    if not rows:
        return None
    # Re-assert style: importing pruning.pruning earlier clobbers
    # text.usetex back to False at module load time.
    _configure_style()
    H_arr  = np.array([r['H']  for r in rows], dtype=float)
    L_arr  = np.array([r['L']  for r in rows], dtype=int)
    s0     = np.array([r['s0_mean']    for r in rows])
    s0_e   = np.array([r['s0_std']     for r in rows])
    # Use median + 16th/84th-percentile asymmetric bars (Gaussian-equivalent
    # on skewed distributions).  Falls back to mean for legacy caches.
    loss      = np.array([r.get('loss_median', r['loss_mean']) for r in rows])
    loss_lo   = np.array([r.get('loss_p16', max(r['loss_mean']
                                                - r['loss_std'], 0.0))
                          for r in rows])
    loss_hi   = np.array([r.get('loss_p84', r['loss_mean']
                                + r['loss_std'])
                          for r in rows])
    yerr_low  = np.maximum(loss - loss_lo, 0.0)
    yerr_high = np.maximum(loss_hi - loss,  0.0)

    L_unique = sorted(set(int(l) for l in L_arr))
    cmap = plt.cm.viridis
    norm = Normalize(vmin=min(L_unique), vmax=max(L_unique))

    fig, ax = plt.subplots(figsize=(8.6, 6.4), facecolor='white')

    # error-bar layer (drawn first, no markers, so the ones above sit on top)
    for L_val in L_unique:
        m = L_arr == L_val
        if not m.any():
            continue
        c = cmap(norm(L_val))
        yerr = np.vstack([yerr_low[m], yerr_high[m]])
        ax.errorbar(s0[m], loss[m],
                    xerr=s0_e[m] if (s0_e[m] > 0).any() else None,
                    yerr=yerr if yerr.any() else None,
                    fmt='o', ms=6, mec='black', mew=0.45,
                    mfc=c, ecolor=c, elinewidth=0.7, capsize=2.4,
                    alpha=0.92, zorder=5,
                    label=rf'$L = {L_val}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Critical density $s_{0}$')
    ax.set_ylabel(
        r'$\mathrm{median}\,\mathcal{L}_{\mathrm{val}}$ '
        r'(cross-entropy; $16$--$84$th\,\%\,bars)')
    method_pretty = {
        'random':    r'random',
        'magnitude': r'magnitude',
        'wanda':     r'\textsc{wanda}',
    }.get(method, method)
    ax.set_title(
        rf'{dataset_label}\,---\,{method_pretty}: '
        r'validation loss vs.\ critical density'
        rf'\quad ($N={len(rows)}$ cells)',
        pad=8)
    ax.grid(which='both', alpha=0.32, linewidth=0.5)
    ax.tick_params(which='both', direction='out')

    # legend split into two columns when many depths
    ncol = 2 if len(L_unique) > 5 else 1
    ax.legend(loc='best', ncol=ncol, title=r'\textbf{Depth}',
              title_fontsize=9, columnspacing=1.0, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(datasets, methods, force=False):
    have_latex = _configure_style()
    print(f'  Rendering with text.usetex = {have_latex}')

    base_out = 'assets/unstructured_pruning/loss_scaling'
    os.makedirs(base_out, exist_ok=True)

    saved = []
    for ds in datasets:
        if ds not in DATASETS:
            print(f'  unknown dataset {ds!r} - skip'); continue
        for m in methods:
            if m not in UNSTRUCTURED_METHODS:
                print(f'  unknown method {m!r} - skip'); continue
            print(f'\n=== {ds} / {m} ===')
            rows = compute_per_config(ds, m, force=force, cache_root=base_out)
            if not rows:
                print('  no rows; skipping plot'); continue
            out = os.path.join(base_out, f'loss_vs_s0_{ds}_{m}.png')
            p = plot_loss_vs_s0(
                rows, out,
                dataset_label=DATASETS[ds]['dataset_label'],
                method=m)
            if p:
                print(f'  Saved: {p}')
                saved.append(p)

    print(f'\nDone. {len(saved)} plot(s) written to {base_out}/.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['mnist28'],
                    choices=list(DATASETS),
                    help='datasets to render (default: mnist28 only)')
    ap.add_argument('--methods', nargs='+',
                    default=['random', 'magnitude', 'wanda'],
                    choices=list(UNSTRUCTURED_METHODS))
    ap.add_argument('--force', action='store_true',
                    help='recompute losses even if a JSON cache exists')
    args = ap.parse_args()
    main(args.datasets, args.methods, force=args.force)
