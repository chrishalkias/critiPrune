#!/usr/bin/env python3
"""Fit s_0 ~ a * P^phi where P is total parameter count.

Tests whether the critical weight density scales as a single power law in the
total number of parameters rather than separately in H and L.

Produces a 4x3 log-log figure (dataset x method) saved to
assets/unstructured_pruning/param_scaling.png.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = 'assets/unstructured_pruning'
OUTPUT_PATH = os.path.join(FIGURES_DIR, 'param_scaling.png')
MIN_R2 = 0.80

DATASETS = ['sklearn', 'mnist28', 'cifar_pca', 'cifar_resnet']
METHODS  = ['random', 'magnitude', 'wanda']

DATASET_LABELS = {
    'sklearn':      'sklearn digits',
    'mnist28':      'MNIST 28×28',
    'cifar_pca':    'CIFAR-10 + PCA(200)',
    'cifar_resnet': 'CIFAR-10 + ResNet18',
}
METHOD_LABELS = {
    'random':    'Random',
    'magnitude': 'Magnitude',
    'wanda':     'WANDA',
}


# --- Fitting -----------------------------------------------------------------

def _adj_r2(y, y_pred, n_fit_params):
    n = len(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0 or n <= n_fit_params:
        return float('nan')
    return 1.0 - (ss_res / (n - n_fit_params)) / (ss_tot / (n - 1))


def fit_param_scaling(P_arr, s0_arr):
    """Fit s_0 = a * P^phi in log-log space.

    Filters out non-positive ``s_0`` or ``P`` entries before taking logs,
    since a sigmoid fit can return ``s_0 = 0`` at the bound and ``log(0)``
    would silently propagate NaN through the whole fit.

    Returns (phi, a, r2_adj) where phi and a are the power-law exponent and
    prefactor, and r2_adj is the adjusted R² computed in log space.
    """
    P_arr = np.asarray(P_arr, dtype=float)
    s0_arr = np.asarray(s0_arr, dtype=float)
    valid = (P_arr > 0) & (s0_arr > 0) & np.isfinite(P_arr) & np.isfinite(s0_arr)
    if valid.sum() < 2:
        return float('nan'), float('nan'), float('nan')
    log_P  = np.log(P_arr[valid])
    log_s0 = np.log(s0_arr[valid])
    phi, log_a = np.polyfit(log_P, log_s0, 1)
    a = np.exp(log_a)
    log_s0_pred = phi * log_P + log_a
    r2 = _adj_r2(log_s0, log_s0_pred, n_fit_params=2)
    return float(phi), float(a), float(r2)


# --- Data loading ------------------------------------------------------------

def load_good_rows(dataset, method):
    path = os.path.join(
        FIGURES_DIR,
        f'unstructured_figures_{dataset}_{method}',
        'scaling_results.json',
    )
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = json.load(f)
    return [
        r for r in rows
        if r.get('sigmoid_R2') is not None
        and r['sigmoid_R2'] > MIN_R2
        and r.get('sigmoid_s_0') is not None
        and r.get('n_params') is not None
    ]


# --- Plot --------------------------------------------------------------------

fig, axes = plt.subplots(
    len(DATASETS), len(METHODS),
    figsize=(5 * len(METHODS), 4 * len(DATASETS)),
)
fig.suptitle(
    r'Unstructured pruning: $s_0 \sim a\,P^{\phi}$  —  log-log, coloured by depth $L$',
    fontsize=13, y=1.01,
)

print(f"{'dataset':14s}  {'method':9s}  {'phi':>7s}  {'a':>7s}  {'R2_adj':>7s}  {'n':>5s}")
print('-' * 55)

for row_idx, dataset in enumerate(DATASETS):
    for col_idx, method in enumerate(METHODS):
        ax = axes[row_idx][col_idx]
        rows = load_good_rows(dataset, method)

        ax.set_title(
            f'{DATASET_LABELS[dataset]} / {METHOD_LABELS[method]}',
            fontsize=8.5,
        )
        ax.set_xlabel('Parameters $P$', fontsize=8)
        ax.set_ylabel('$s_0$', fontsize=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(labelsize=7)

        if not rows:
            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=9)
            continue

        P_arr  = np.array([r['n_params']    for r in rows], dtype=float)
        s0_arr = np.array([r['sigmoid_s_0'] for r in rows])

        # colour by L so depth structure is visible
        unique_L = sorted(set(r['L'] for r in rows))
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(unique_L), 1)))
        L_col = dict(zip(unique_L, cmap))

        for r in rows:
            ax.scatter(r['n_params'], r['sigmoid_s_0'],
                       s=14, color=L_col[r['L']], alpha=0.6, zorder=5,
                       linewidths=0)

        # power-law fit and overlay
        phi, a, r2 = fit_param_scaling(P_arr, s0_arr)
        P_fine = np.geomspace(P_arr.min() * 0.9, P_arr.max() * 1.1, 300)
        ax.plot(P_fine, a * P_fine ** phi, color='crimson',
                lw=1.8, ls='--', zorder=6)

        label = (
            f'$s_0 = {a:.3f}\\,P^{{{phi:+.3f}}}$\n'
            f'$R^2_{{\\rm adj}}={r2:.3f}$   $n={len(rows)}$'
        )
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=7,
                va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

        print(f'{dataset:14s}  {method:9s}  {phi:+7.3f}  {a:7.4f}  {r2:7.3f}  {len(rows):5d}')

# shared depth-L colourbar legend (approximate: uses the last subplot's L range)
sm = plt.cm.ScalarMappable(
    cmap='viridis',
    norm=plt.Normalize(vmin=min(unique_L), vmax=max(unique_L)),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, aspect=40)
cbar.set_label('Depth $L$', fontsize=9)

plt.tight_layout()
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {OUTPUT_PATH}')
