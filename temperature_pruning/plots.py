"""Plots for the temperature/pruning experiment.

Three figures:
    accuracy_curves.png  -- A(s) for every sigma, one panel per (H, L) cell.
    critical_line.png    -- p_c(sigma) scatter + linear fit per cell. Headline.
    data_collapse.png    -- A vs s/sigma for sigma > 0, one panel per cell.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pruning.pruning import sigmoid_fn

from .analysis import collapse_score, fit_critical_line, group_pc_by_cell


def _cell_grid(cells, n_cols=3):
    n = len(cells)
    n_cols = max(1, min(n_cols, n))
    n_rows = int(np.ceil(n / n_cols))
    return n_rows, n_cols


def _sigma_norm(sigmas, cmap_name='viridis'):
    """Continuous (Normalize, Colormap) mapping for a sigma grid.

    Returned objects are usable both for individual ``cmap(norm(sigma))``
    color lookups and as inputs to a ``ScalarMappable`` for the colorbar.
    Densely sampled sigmas (~100 values) make a per-curve legend illegible,
    so all sigma plots share one colorbar instead.
    """
    smin = float(min(sigmas))
    smax = float(max(sigmas))
    if smax <= smin:
        smax = smin + 1e-9
    norm = Normalize(vmin=smin, vmax=smax)
    cmap = plt.colormaps.get_cmap(cmap_name)
    return norm, cmap


def _add_sigma_colorbar(fig, axes, norm, cmap, label=r'Temperature $\sigma$'):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    flat_axes = np.atleast_1d(axes).ravel().tolist()
    cbar = fig.colorbar(sm, ax=flat_axes, shrink=0.85, pad=0.02,
                        aspect=30, label=label)
    return cbar


def plot_accuracy_curves(results, output_path, min_r2=0.0):
    """One panel per (H, L); colour by sigma; sigmoid fit overlaid."""
    by_cell = defaultdict(list)
    for r in results:
        by_cell[(int(r['H']), int(r['L']))].append(r)
    cells = sorted(by_cell)
    if not cells:
        print("  [accuracy_curves] no rows -- skip")
        return

    sigmas_all = sorted({float(r['sigma']) for r in results})
    norm, cmap = _sigma_norm(sigmas_all)

    n_rows, n_cols = _cell_grid(cells)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False, layout='constrained')

    for idx, (H, L) in enumerate(cells):
        ax = axes[idx // n_cols][idx % n_cols]
        rows = sorted(by_cell[(H, L)], key=lambda r: float(r['sigma']))
        for r in rows:
            sigma = float(r['sigma'])
            color = cmap(norm(sigma))
            s_vals = np.array(r['densities'])
            a_mean = np.array(r['accs_mean']) * 100
            a_std = np.array(r['accs_std']) * 100
            ax.errorbar(s_vals, a_mean, yerr=a_std, fmt='o', ms=2.5,
                        color=color, alpha=0.7, lw=0.5)
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] >= min_r2:
                s_fine = np.linspace(s_vals.min(), s_vals.max(), 300)
                fit = sigmoid_fn(s_fine,
                                 r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                 r['sigmoid_s_0'], r['sigmoid_beta']) * 100
                ax.plot(s_fine, fit, color=color, lw=0.9, alpha=0.8)
        ax.set_title(f'H={H}, L={L}')
        ax.set_xlabel('Density $s$')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(alpha=0.3)

    for j in range(len(cells), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(r'Recovery curves $A(s)$ across temperature $\sigma$',
                 fontsize=12)
    _add_sigma_colorbar(fig, axes, norm, cmap)
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_critical_line(results, output_path, min_r2=0.80):
    """Per-cell linear fit of p_c(sigma); slope = 1/J_0_eff."""
    pc = group_pc_by_cell(results, min_r2=min_r2)
    fits = fit_critical_line(pc)
    if not fits:
        print("  [critical_line] no fits -- skip")
        return

    cells = sorted(fits)
    n_rows, n_cols = _cell_grid(cells)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for idx, cell in enumerate(cells):
        ax = axes[idx // n_cols][idx % n_cols]
        H, L = cell
        f = fits[cell]
        sigmas = np.array(f['sigmas'])
        p_cs = np.array(f['p_cs'])
        ax.scatter(sigmas, p_cs, s=30, color='C0', zorder=5,
                   edgecolor='black', linewidth=0.5)
        # Polynomial fit curve, evaluated on a fine grid.
        x_line = np.linspace(0, sigmas.max() * 1.05, 200)
        # f['coeffs'] is ascending [c0, c1, c2, ...]; np.polyval expects
        # highest-degree-first, so reverse.
        y_line = np.polyval(list(reversed(f['coeffs'])), x_line)
        deg = f['degree']
        if deg == 2:
            label = (f"$p_c = {f['a']:.3f} + {f['b']:.3f}\\,\\sigma + "
                     f"{f['c']:.3f}\\,\\sigma^2$\n"
                     f"$R^2={f['R2']:.3f}$")
        elif deg == 1:
            label = (f"$p_c = {f['a']:.3f} + {f['b']:.3f}\\,\\sigma$\n"
                     f"$R^2={f['R2']:.3f}$  $1/J_0={f['b']:.2f}$")
        else:
            terms = [f"{c:+.3f}\\sigma^{{{i}}}" if i > 1 else
                     (f"{c:+.3f}\\sigma" if i == 1 else f"{c:.3f}")
                     for i, c in enumerate(f['coeffs'])]
            label = f"$p_c = {' '.join(terms)}$\n$R^2={f['R2']:.3f}$"
        ax.plot(x_line, y_line, 'r-', lw=1.5, label=label)
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(f'H={H}, L={L}')
        ax.set_xlabel('Temperature $\\sigma$')
        ax.set_ylabel('$p_c$ (sigmoid inflection)')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)

    for j in range(len(cells), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    deg = next(iter(fits.values()))['degree']
    if deg == 2:
        title = (r"Critical line $p_c(\sigma) = a + b\,\sigma + c\,\sigma^2$"
                 "\n"
                 r"Sherrington-Kirkpatrick bond-disorder prediction: "
                 r"$p_c = T_0/J_0 + \sigma^2/(2 J_0^2)$  "
                 r"(i.e. $b=0$, $c=1/(2J_0^2)$)")
    elif deg == 1:
        title = (r"Critical line $p_c(\sigma) = a + b\,\sigma$"
                 "\n"
                 r"diluted Curie-Weiss prediction: linear with $b=1/J_0$")
    else:
        title = r"$p_c(\sigma)$ polynomial fit"
    fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return fits


def plot_data_collapse(results, output_path, min_r2=0.80):
    """A vs s/sigma overlay for sigma > 0 curves; one panel per cell."""
    scores = collapse_score(results, min_r2=min_r2)
    if not scores:
        print("  [data_collapse] not enough data -- skip")
        return

    by_cell = defaultdict(list)
    for r in results:
        if float(r['sigma']) <= 0:
            continue
        if r.get('sigmoid_R2') is None or r['sigmoid_R2'] < min_r2:
            continue
        by_cell[(int(r['H']), int(r['L']))].append(r)
    cells = sorted(by_cell)
    n_rows, n_cols = _cell_grid(cells)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False, layout='constrained')

    sigmas_all = sorted({float(r['sigma']) for r in results
                         if float(r['sigma']) > 0})
    norm, cmap = _sigma_norm(sigmas_all)

    for idx, cell in enumerate(cells):
        ax = axes[idx // n_cols][idx % n_cols]
        H, L = cell
        rows = sorted(by_cell[cell], key=lambda r: float(r['sigma']))
        for r in rows:
            sigma = float(r['sigma'])
            xs = np.array(r['densities']) / sigma
            ys = np.array(r['accs_mean']) * 100
            ax.plot(xs, ys, 'o-', ms=2.5, color=cmap(norm(sigma)),
                    lw=0.6, alpha=0.75)
        sc = scores.get(cell)
        if sc is not None:
            ax.text(0.05, 0.95, f"collapse score = {sc['score']:.2f}",
                    transform=ax.transAxes, fontsize=8,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',
                              fc='white', alpha=0.85))
        ax.set_xscale('log')
        ax.set_xlabel(r'$s / \sigma$  (rescaled density)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'H={H}, L={L}')
        ax.grid(alpha=0.3, which='both')

    for j in range(len(cells), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle(r'Data collapse: $A(s, \sigma)$ vs $s / \sigma$  '
                 r'(score $\to 1$ means perfect collapse)',
                 fontsize=12)
    _add_sigma_colorbar(fig, axes, norm, cmap)
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return scores


def make_all_plots(results, output_dir, min_r2=0.80):
    os.makedirs(output_dir, exist_ok=True)
    plot_accuracy_curves(results,
                         os.path.join(output_dir, 'accuracy_curves.png'),
                         min_r2=min_r2)
    fits = plot_critical_line(results,
                              os.path.join(output_dir, 'critical_line.png'),
                              min_r2=min_r2)
    scores = plot_data_collapse(results,
                                os.path.join(output_dir, 'data_collapse.png'),
                                min_r2=min_r2)
    return fits, scores
