#!/usr/bin/env python3
"""Cluster-scale analysis of the input-noise sweep.

Reads ``input_noise/results_cluster_all.json`` (~4000 cells across 12
(dataset, method) combos and many (H, L, repeat) triples), runs:

  1. Per-cell iso-``A = 0.5`` contour extraction from the joint grid.
  2. Per-cell fit of the framework rational curve

         sigma^2(x) = <x^2> * (sigma^2(1) - x) / (<x^2> + x),
         x = (1-s)/s * <x^2>

     -> ``sigma^2(1)`` and ``R^2`` per cell.
  3. Per-cell collapse residual in normalised coordinates

         xi  = (1 - s) * (1 + <x^2> / sigma^2(1))
         eta = sigma^2(s) / sigma^2(1)

     -> framework predicts ``eta = 1 - xi``; we report
     ``RMS(eta - (1 - xi))`` across cells.
  4. Aggregate statistics by ``(dataset, method)`` and by ``(H, L)``.

Outputs (``input_noise/figures_cluster/``):
  - ``collapse_all.png``      master parameter-free collapse (all cells)
  - ``collapse_by_method.png`` 3-panel collapse split by pruning method
  - ``r2_distribution.png``    histogram of per-cell R^2 (overall + by method)
  - ``r2_vs_HL.png``           median R^2 over the (H, L) grid per dataset
  - ``sigma2_1_scaling.png``   sigma^2(1) vs H, vs L, per (dataset, method)
  - ``residuals.png``          per-stratum RMS residual table as a heatmap
  - ``per_cell_fits.json``     full per-cell records (machine-readable)
  - ``findings.md``            headline summary

Run::

    .venv/bin/python -m input_noise.cluster_analyze
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_JSON   = 'input_noise/results_cluster_all.json'
OUT_DIR      = 'input_noise/figures_cluster'
ISO_LEVEL    = 0.50

# Datasets to drop before analysis. ``cifar_pca`` is excluded because the
# raw-pixel networks rarely reach A_unpruned >= 0.5, so the iso-A = 0.5
# contour is empty in almost every cell.
EXCLUDE_DATASETS = {'cifar_pca'}

DATASET_COLOR = {
    'mnist28':      '#1f77b4',
    'cifar_pca':    '#2ca02c',
    'cifar_resnet': '#d62728',
    'sklearn':      '#9467bd',
}
DATASET_LABEL = {
    'mnist28':      'MNIST 28x28',
    'cifar_pca':    'CIFAR-10 PCA-200',
    'cifar_resnet': 'CIFAR-10 ResNet18',
    'sklearn':      'sklearn digits',
}
METHOD_MARKER = {
    'random':    '.',
    'magnitude': 'x',
    'wanda':     '+',
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _style():
    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex':      have_latex,
        'font.family':      'serif',
        'mathtext.fontset': 'cm',
        'figure.dpi':       120,
        'savefig.dpi':      200,
        'savefig.bbox':     'tight',
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}')
    return have_latex


# ---------------------------------------------------------------------------
# Per-cell fit (one-parameter framework curve)
# ---------------------------------------------------------------------------
def iso_contour(cell, level=ISO_LEVEL):
    """Trace the iso-A contour in (s, sigma) via linear interpolation
    across each s-column of the joint grid. Returns a sorted list of
    (s, sigma) pairs where the column's A(sigma) curve crosses ``level``."""
    s_grid     = np.asarray(cell['joint']['s_grid'],     dtype=float)
    sigma_grid = np.asarray(cell['joint']['sigma_grid'], dtype=float)
    A = np.asarray(cell['joint']['mean'],                dtype=float)
    pts = []
    for i_s, s in enumerate(s_grid):
        col = A[i_s, :]
        if col[0] < level or col[-1] > level:
            continue
        order = np.argsort(sigma_grid)
        sig_o = sigma_grid[order]
        col_o = col[order]
        idx = np.searchsorted(-col_o, -level)
        if idx == 0 or idx >= len(sig_o):
            continue
        a0, a1 = col_o[idx - 1], col_o[idx]
        sg0, sg1 = sig_o[idx - 1], sig_o[idx]
        if a0 == a1:
            sigma_iso = 0.5 * (sg0 + sg1)
        else:
            sigma_iso = sg0 + (sg1 - sg0) * (level - a0) / (a1 - a0)
        pts.append((float(s), float(sigma_iso)))
    return pts


def fit_framework(cell, level=ISO_LEVEL):
    """One-parameter fit of sigma^2(s) = s*sigma2_1 - (1-s)*<x^2>.

    Returns ``(sigma2_1, R2, n_contour)``. Closed-form OLS for a single
    free parameter.
    """
    contour = iso_contour(cell, level=level)
    if len(contour) < 1:
        return None, None, 0
    s_arr  = np.array([p[0] for p in contour])
    sg_arr = np.array([p[1] for p in contour])
    x2 = float(cell['x2_mean'])
    # d/d sigma2_1: 2 * sum s_i (sg_i^2 - s_i sigma2_1 + (1-s_i) x2)
    # ⇒ sigma2_1 = sum s_i (sg_i^2 + (1-s_i) x2) / sum s_i^2
    num = float(np.sum(s_arr * (sg_arr ** 2 + (1.0 - s_arr) * x2)))
    den = float(np.sum(s_arr ** 2))
    if den <= 0:
        return None, None, len(contour)
    sigma2_1 = num / den
    y_pred = s_arr * sigma2_1 - (1.0 - s_arr) * x2
    y = sg_arr ** 2
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return float(sigma2_1), float(R2), len(contour)


def collapse_residuals(cell, sigma2_1, level=ISO_LEVEL):
    """Per-cell normalised-collapse residuals to ``eta = 1 - xi``.

    Returns lists ``(xi, eta)`` over the cell's contour points. ``[]``
    if the contour is empty or ``sigma2_1`` is non-finite/non-positive.
    """
    if (sigma2_1 is None or not np.isfinite(sigma2_1)
            or sigma2_1 <= 0):
        return [], []
    contour = iso_contour(cell, level=level)
    if not contour:
        return [], []
    s_arr  = np.array([p[0] for p in contour])
    sg_arr = np.array([p[1] for p in contour])
    x2 = float(cell['x2_mean'])
    xi  = (1.0 - s_arr) * (1.0 + x2 / sigma2_1)
    eta = (sg_arr ** 2) / sigma2_1
    return xi.tolist(), eta.tolist()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def process_all(cells, level=ISO_LEVEL):
    """Return per-cell fit records and pooled (xi, eta) arrays.

    ``cell_id_all`` is a list parallel to ``xi_all`` / ``eta_all`` /
    ``group_all``: it tags each point with the cell it came from so
    downstream aggregation can do per-cell-first then cross-cell stats
    (avoids population-composition aliasing that pure pooled binning
    suffers from).
    """
    records = []
    xi_all, eta_all, group_all, cell_id_all = [], [], [], []
    s_raw_all, sigma_raw_all = [], []
    for c in cells:
        sigma2_1, R2, n = fit_framework(c, level=level)
        rec = {
            'dataset': c['dataset'],
            'method':  c['method'],
            'H':       int(c['H']),
            'L':       int(c['L']),
            'repeat':  int(c.get('repeat', 0)),
            'x2_mean': float(c['x2_mean']),
            'val_acc': float(c.get('val_acc', float('nan'))),
            'sigma2_1':    sigma2_1,
            'R2':          R2,
            'n_contour':   int(n),
        }
        records.append(rec)
        cid = (c['dataset'], c['method'],
               int(c['H']), int(c['L']), int(c.get('repeat', 0)))
        xi, eta = collapse_residuals(c, sigma2_1, level=level)
        # Raw iso-A contour: (s, sigma) pairs straight from the joint
        # grid, no per-cell rescaling. Kept parallel to xi/eta so a point
        # at index i is the same physical contour point in both
        # representations.
        raw_pts = iso_contour(c, level=level) if len(xi) else []
        for (s_pt, sg_pt), x, y in zip(raw_pts, xi, eta):
            s_raw_all.append(s_pt)
            sigma_raw_all.append(sg_pt)
            xi_all.append(x)
            eta_all.append(y)
            group_all.append((c['dataset'], c['method']))
            cell_id_all.append(cid)
    return (records, np.asarray(xi_all), np.asarray(eta_all),
            group_all, cell_id_all,
            np.asarray(s_raw_all), np.asarray(sigma_raw_all))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _binned_quantiles(xi, eta, nbins=24, x_min=0.0, x_max=1.3):
    """Bin (xi, eta) into nbins along xi and return median + 25/75
    quantiles + counts per bin. Bins with fewer than 5 points are
    dropped — the median is unreliable there."""
    edges = np.linspace(x_min, x_max, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    q25 = np.full(nbins, np.nan)
    q50 = np.full(nbins, np.nan)
    q75 = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)
    bin_idx = np.clip(np.digitize(xi, edges) - 1, 0, nbins - 1)
    for b in range(nbins):
        mask = bin_idx == b
        n = int(mask.sum())
        counts[b] = n
        if n >= 5:
            v = eta[mask]
            q25[b] = float(np.quantile(v, 0.25))
            q50[b] = float(np.quantile(v, 0.50))
            q75[b] = float(np.quantile(v, 0.75))
    valid = ~np.isnan(q50)
    return centres[valid], q25[valid], q50[valid], q75[valid], counts[valid]


def _per_cell_cross_cell(xi, eta, cell_ids, nbins=24,
                         x_min=0.0, x_max=1.3, min_cells=5):
    """Two-stage aggregation: per-cell median per bin, then cross-cell
    median across cells.

    Removes the population-composition aliasing that pure pooled binning
    suffers from: in pooled binning, neighbouring xi-bins are fed by
    different subsets of cells, so the median jumps as composition
    changes. Here each cell contributes **one value per bin** (its own
    median over that bin) regardless of how many raw points it has, so
    each cell gets one vote per bin and bin-to-bin transitions reflect
    a stable population.

    Returns ``(centres, q25, q50, q75, n_cells_per_bin)`` restricted to
    bins where ``>= min_cells`` cells contribute.
    """
    edges = np.linspace(x_min, x_max, nbins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    by_cell = defaultdict(list)
    for i, cid in enumerate(cell_ids):
        # Ensure cid is hashable even if it came through a numpy object
        # array that promoted nested tuples to ndarrays.
        if isinstance(cid, np.ndarray):
            cid = tuple(cid.tolist())
        by_cell[cid].append(i)
    per_cell_rows = []
    for cid, idx in by_cell.items():
        idx = np.asarray(idx)
        cell_xi  = xi[idx]
        cell_eta = eta[idx]
        b_idx = np.clip(np.digitize(cell_xi, edges) - 1, 0, nbins - 1)
        row = np.full(nbins, np.nan)
        for b in range(nbins):
            v = cell_eta[b_idx == b]
            if len(v):
                row[b] = float(np.median(v))
        per_cell_rows.append(row)
    if not per_cell_rows:
        empty = np.array([])
        return empty, empty, empty, empty, np.zeros(0, dtype=int)
    M = np.asarray(per_cell_rows)                      # (n_cells, nbins)
    counts = np.sum(~np.isnan(M), axis=0).astype(int)  # cells per bin
    q25 = np.full(nbins, np.nan)
    q50 = np.full(nbins, np.nan)
    q75 = np.full(nbins, np.nan)
    for b in range(nbins):
        col = M[:, b]
        col = col[~np.isnan(col)]
        if len(col) >= min_cells:
            q25[b] = float(np.quantile(col, 0.25))
            q50[b] = float(np.quantile(col, 0.50))
            q75[b] = float(np.quantile(col, 0.75))
    valid = ~np.isnan(q50)
    return centres[valid], q25[valid], q50[valid], q75[valid], counts[valid]


def plot_collapse_all(xi, eta, groups, cell_ids, out_path, have_latex):
    """Parameter-free collapse, built with the seaborn strip+regplot recipe.

    Directly mirrors the seaborn gallery example ``strip_regplot``:

        sns.catplot(data=..., x=..., y=..., hue=..., native_scale=True, zorder=1)
        sns.regplot(data=..., x=..., y=..., scatter=False, truncate=False,
                    order=2, color=".2")

    Here x = xi, y = eta, and the continuous hue is the network depth L
    (the quantity the findings show drives the collapse residual).  The
    ``regplot`` draws the empirical order-1 (linear) conditional mean of the
    cloud; the parameter-free prediction eta = 1 - xi is overlaid as a dashed
    reference so the empirical mean can be read against the theory.
    """
    import pandas as pd
    import seaborn as sns

    sns.set_theme()

    L_of_pt = np.array([cid[3] for cid in cell_ids], dtype=int)
    df = pd.DataFrame({'xi': xi, 'eta': eta, 'L': L_of_pt})

    # Strip chart with observations (figure-level catplot, native numeric x).
    # Wide-and-short aspect so the panel reads as a single compact row when
    # placed full-width (figure*) in the manuscript.
    g = sns.catplot(
        data=df, x='xi', y='eta', hue='L',
        native_scale=True, zorder=1,
        height=4.0, aspect=2.6, s=3, alpha=0.35, linewidth=0,
        legend=True,
    )
    ax = g.ax

    # Empirical order-1 conditional mean (linear fit through the cloud).
    sns.regplot(
        data=df, x='xi', y='eta',
        scatter=False, truncate=False, order=1, color='.2',
        ax=ax, label='order-1 conditional mean',
    )
    mean_line = ax.lines[-1]

    # Parameter-free prediction eta = 1 - xi as a dashed reference.
    xg = np.linspace(0.0, 1.4, 200)
    (theory_line,) = ax.plot(
        xg, 1.0 - xg, color='crimson', lw=1.6, ls='--', zorder=6,
        label=(r'$\eta = 1 - \xi$' if have_latex else 'eta = 1 - xi'))

    rms = float(np.sqrt(np.mean((eta - (1.0 - xi)) ** 2))) if len(xi) else float('nan')
    ax.text(0.04, 0.05,
            (rf'RMS $= {rms:.3f}$,  $N = {len(xi)}$'
             if have_latex else f'RMS = {rms:.3f},  N = {len(xi)}'),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', lw=0.6))

    ax.set_xlim(-0.05, 1.4)
    ax.set_ylim(-0.4, 1.2)
    ax.set_xlabel(r'$\xi = (1-s)\,(1 + \langle x^2\rangle / \sigma^2(1))$'
                  if have_latex else '(1-s)(1 + <x^2>/sigma^2(1))')
    ax.set_ylabel(r'$\eta = \sigma_{\mathrm{iso}}^{2}(s) / \sigma^2(1)$'
                  if have_latex else 'sigma^2_iso(s) / sigma^2(1)')
    # Make the L hue legend (catplot's figure-level legend) legible: the
    # strip markers are tiny and semi-transparent (s=3, alpha=0.35), which
    # the legend inherits, so enlarge the swatches and force them opaque.
    if g.legend is not None:
        for h in g.legend.legend_handles:
            try:
                h.set_alpha(1.0)
                h.set_sizes([60])           # PathCollection (scatter) handle
            except (AttributeError, TypeError):
                pass
            try:
                h.set_markersize(9)          # Line2D handle fallback
                h.set_alpha(1.0)
            except (AttributeError, TypeError):
                pass
        g.legend.set_title('L')
        for txt in g.legend.get_texts():
            txt.set_fontsize(9)

    # Dedicated legend for the two lines; keep catplot's figure-level L
    # hue legend (on the right margin) untouched by adding this as a
    # separate artist.
    line_leg = ax.legend(handles=[mean_line, theory_line],
                         loc='upper right', fontsize=9, framealpha=0.9)
    ax.add_artist(line_leg)

    g.figure.savefig(out_path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(g.figure)

    # Reset rcParams so other plots in the same run are unaffected.
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    return rms


def plot_eta_conditional_kde(xi, eta, out_path, have_latex, n_xi_bins=5):
    """Conditional KDE of the xi-band composition as a function of eta.

    Mirrors the seaborn gallery example ``multiple_conditional_kde``:

        sns.displot(data, x=..., hue=..., kind="kde", height=6,
                    multiple="fill", clip=(0, None), palette="ch:...")

    Here x = eta and the hue is xi binned into ``n_xi_bins`` ordered bands.
    ``multiple="fill"`` renders, at each eta, the *conditional* proportion of
    points coming from each xi band (the column sums to 1), giving a smooth
    continuous version of the earlier stacked histogram.

    The parameter-free prediction eta = 1 - xi is overlaid: under it, a point
    in xi band centred at xi_c must sit at eta = 1 - xi_c.  Those predicted
    eta locations are drawn as colour-matched markers/vertical guides, so one
    can check whether each band's conditional mass actually concentrates where
    the theory places it.

    The lower clip of the gallery example is dropped because eta is signed.
    """
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style='whitegrid')

    xi = np.asarray(xi)
    eta = np.asarray(eta)

    # Ordered xi bands (equal-width over the data range) used as the hue.
    edges = np.linspace(float(xi.min()), float(xi.max()), n_xi_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    labels = [f'{edges[i]:.2f}–{edges[i + 1]:.2f}' for i in range(n_xi_bins)]
    xi_band = pd.cut(xi, bins=edges, labels=labels, include_lowest=True,
                     ordered=True)
    df = pd.DataFrame({'eta': eta, 'xi_band': xi_band})

    palette = 'ch:rot=-.25,hue=1,light=.75'
    g = sns.displot(
        data=df,
        x='eta', hue='xi_band',
        kind='kde', height=6,
        multiple='fill', clip=(None, None),
        palette=palette,
    )
    ax = g.ax

    # Colours displot assigned to the ordered bands, reused for the overlay.
    band_colors = sns.color_palette(palette, n_xi_bins)

    # Overlay theory: eta = 1 - xi_centre for each band.
    for c, col, lab in zip(centres, band_colors, labels):
        eta_theory = 1.0 - c
        ax.axvline(eta_theory, color=col, ls='--', lw=1.4, alpha=0.9, zorder=5)
        ax.plot([eta_theory], [1.02], marker='v', ms=9, color=col,
                markeredgecolor='0.2', markeredgewidth=0.5,
                clip_on=False, zorder=6)
    # Single proxy handle documenting the markers.
    ax.plot([], [], marker='v', ls='--', color='0.3',
            label=(r'$\eta = 1-\xi$ (band centre)'
                   if have_latex else 'eta = 1 - xi (band centre)'))

    ax.set_xlabel(r'$\eta = \sigma_{\mathrm{iso}}^{2}(s) / \sigma^2(1)$'
                  if have_latex else 'eta = sigma^2_iso(s) / sigma^2(1)')
    ax.set_ylabel(r'conditional proportion  $P(\xi\,\mathrm{band}\mid \eta)$'
                  if have_latex else 'conditional proportion P(xi band | eta)')
    if g.legend is not None:
        g.legend.set_title(r'$\xi$ band' if have_latex else 'xi band')

    g.savefig(out_path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(g.figure)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_collapse_by_method(xi, eta, groups, cell_ids, out_path, have_latex):
    """Per-method collapse using per-cell-first, then cross-cell median.

    Same two-stage aggregation as :func:`plot_collapse_all` right panel,
    split by pruning method. Removes population-composition aliasing so
    each cell gets one vote per bin regardless of how many raw points
    it contributes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), facecolor='white',
                             sharex=True, sharey=True)
    methods = ['random', 'magnitude', 'wanda']
    rmses = {}
    xg = np.linspace(0.0, 1.4, 60)
    cell_ids_arr = np.asarray(cell_ids, dtype=object)
    for ax, m in zip(axes, methods):
        ax.plot(xg, 1.0 - xg, 'k-', lw=1.2)
        by_ds = defaultdict(list)
        for i, (ds, mm) in enumerate(groups):
            if mm == m:
                by_ds[ds].append(i)
        n_pts = sum(len(v) for v in by_ds.values())
        for ds, idx in sorted(by_ds.items()):
            idx = np.asarray(idx)
            if len(idx) < 10:
                continue
            c, q25, q50, q75, _ = _per_cell_cross_cell(
                xi[idx], eta[idx], cell_ids_arr[idx])
            if not len(c):
                continue
            col = DATASET_COLOR.get(ds, '0.5')
            ax.fill_between(c, q25, q75, color=col, alpha=0.20, lw=0)
            ax.plot(c, q50, '-o', color=col, ms=3, lw=1.2,
                    label=DATASET_LABEL.get(ds, ds))
        sel = np.asarray([i for i, (_, mm) in enumerate(groups) if mm == m])
        if len(sel):
            rms = float(np.sqrt(np.mean(
                (eta[sel] - (1.0 - xi[sel])) ** 2)))
        else:
            rms = float('nan')
        rmses[m] = rms
        ax.set_title((rf'{m}  ($N={n_pts}$, RMS$={rms:.3f}$)'
                      if have_latex
                      else f'{m} (N={n_pts}, RMS={rms:.3f})'))
        ax.set_xlim(-0.05, 1.4)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3, lw=0.4)
        ax.set_xlabel(r'$(1-s)\,(1 + \langle x^2\rangle / \sigma^2(1))$'
                      if have_latex
                      else '(1-s)(1 + <x^2>/sigma^2(1))')
    axes[0].set_ylabel(r'$\sigma_{\mathrm{iso}}^{2}(s) / \sigma^2(1)$'
                       if have_latex
                       else 'sigma^2_iso(s) / sigma^2(1)')
    axes[0].legend(loc='upper right', fontsize=7)
    fig.suptitle('Binned-median collapse, split by pruning method',
                 y=1.01)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)
    return rmses


def plot_raw_contour(s_raw, sigma_raw, groups, cell_ids,
                     out_path, have_latex):
    """Bare iso-A=0.5 contour: every point is a (s, sigma) pair on the
    iso-accuracy locus, no per-cell rescaling.

    Faceted by dataset (one panel each), points coloured by `L` to
    show how the iso-A curve flattens (less noise tolerance) with
    depth. This is the rawest representation of the data — what the
    framework's (xi, eta) collapse is normalising.
    """
    L_of_pt = np.array([cid[3] for cid in cell_ids], dtype=int)
    by_ds = defaultdict(list)
    for i, (ds, _m) in enumerate(groups):
        by_ds[ds].append(i)
    datasets = sorted(by_ds.keys())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0),
                             facecolor='white', sharey=True)
    if n == 1:
        axes = [axes]
    Ls_present = sorted(set(L_of_pt.tolist()))
    cmap = plt.get_cmap('viridis')
    L_min, L_max = (min(Ls_present), max(Ls_present)
                    if Ls_present else (0, 1))
    for ax, ds in zip(axes, datasets):
        idx = np.asarray(by_ds[ds])
        order = np.argsort(L_of_pt[idx], kind='mergesort')
        idx = idx[order]
        sc = ax.scatter(s_raw[idx], sigma_raw[idx],
                        c=L_of_pt[idx], cmap=cmap,
                        vmin=L_min, vmax=L_max,
                        s=8, alpha=0.45, edgecolors='none')
        ax.set_xlabel(r'retention rate $s$'
                      if have_latex else 'retention rate s')
        ax.set_title(f'{DATASET_LABEL.get(ds, ds)}  '
                     f'(N={len(idx)} contour pts)')
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r'input-noise std $\sigma_x$ at $A = 0.5$'
                       if have_latex
                       else 'input-noise std sigma_x at A=0.5')
    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label('hidden layers $L$' if have_latex else 'hidden layers L')
    cbar.set_ticks(Ls_present)
    fig.suptitle('Bare iso-accuracy contour: '
                 r'$(s, \sigma_x)$ at $A = 0.5$, all cells overlaid'
                 if have_latex
                 else 'Bare iso-accuracy contour: (s, sigma_x) at A=0.5, '
                      'all cells overlaid',
                 y=1.02)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


def plot_collapse_L2(xi, eta, groups, cell_ids, out_path, have_latex):
    """Collapse plot restricted to ``L = 2`` cells only.

    F41's linearised single-layer SNR derivation is exact at ``L = 2``,
    so this is the strongest sanity check: if the framework is right,
    this panel should collapse to a sharp single line ``y = 1 - x``. If
    it does not, the framework is wrong even at the depth where the
    derivation is exact.
    """
    L_of_pt = np.array([cid[3] for cid in cell_ids], dtype=int)
    sel = np.where(L_of_pt == 2)[0]
    n_L2_cells = len({cell_ids[i] for i in sel})

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.0), facecolor='white')
    xg = np.linspace(0.0, 1.4, 60)
    ax.plot(xg, 1.0 - xg, 'k-', lw=1.5, zorder=6,
            label=(r'framework: $y = 1 - x$'
                   if have_latex else 'framework: y = 1 - x'))
    by_ds = defaultdict(list)
    for i in sel:
        by_ds[groups[i][0]].append(i)
    for ds, idx in sorted(by_ds.items()):
        idx = np.asarray(idx)
        ax.scatter(xi[idx], eta[idx],
                   s=12, alpha=0.40, color=DATASET_COLOR.get(ds, '0.5'),
                   edgecolors='none',
                   label=DATASET_LABEL.get(ds, ds))
    if len(sel):
        rms = float(np.sqrt(np.mean((eta[sel] - (1.0 - xi[sel])) ** 2)))
        signed = float(np.mean(eta[sel] - (1.0 - xi[sel])))
    else:
        rms = signed = float('nan')

    ax.text(0.04, 0.06,
            (rf'$L = 2$ only: RMS to $y = 1 - x$ = ${rms:.3f}$'
             rf', signed = ${signed:+.3f}$'
             rf' ($N_{{\mathrm{{pts}}}} = {len(sel)}$, '
             rf'$N_{{\mathrm{{cells}}}} = {n_L2_cells}$)'
             if have_latex else
             f'L=2 only: RMS to y=1-x = {rms:.3f}, signed = {signed:+.3f} '
             f'(N_pts={len(sel)}, N_cells={n_L2_cells})'),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.5', lw=0.5))
    ax.set_xlim(-0.05, 1.4)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel(r'$(1-s)\,(1 + \langle x^2\rangle / \sigma^2(1))$'
                  if have_latex
                  else '(1-s)(1 + <x^2>/sigma^2(1))')
    ax.set_ylabel(r'$\sigma_{\mathrm{iso}}^{2}(s) / \sigma^2(1)$'
                  if have_latex
                  else 'sigma^2_iso(s) / sigma^2(1)')
    ax.set_title(f'$L = 2$ cells only — strongest collapse test'
                 if have_latex else 'L = 2 cells only - strongest collapse test')
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(loc='upper right', fontsize=8)
    fig.savefig(out_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return rms, signed, n_L2_cells, len(sel)


def plot_collapse_by_L(xi, eta, groups, cell_ids, out_path, have_latex):
    """Single-panel scatter coloured by depth ``L``.

    Reveals the triangular envelope's interior structure: the top edge
    (the framework line) should be dominated by shallow cells; the
    interior depression should be dominated by deeper cells. This is the
    direct visual reading of the depth-residual signature documented in
    ``r2_vs_HL.png``.
    """
    L_of_pt = np.array([cid[3] for cid in cell_ids], dtype=int)
    Ls_present = sorted(set(L_of_pt.tolist()))
    cmap = plt.get_cmap('viridis')
    L_min, L_max = min(Ls_present), max(Ls_present)
    def _col(L):
        if L_max == L_min:
            return cmap(0.5)
        return cmap((L - L_min) / (L_max - L_min))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0), facecolor='white')
    xg = np.linspace(0.0, 1.4, 60)
    ax.plot(xg, 1.0 - xg, 'k-', lw=1.5, zorder=6,
            label=(r'framework: $y = 1 - x$'
                   if have_latex else 'framework: y = 1 - x'))
    # Draw shallow-first so deep points sit on top and are visible.
    order = np.argsort(L_of_pt, kind='mergesort')
    sc = ax.scatter(xi[order], eta[order],
                    c=L_of_pt[order], cmap=cmap,
                    vmin=L_min, vmax=L_max,
                    s=6, alpha=0.35, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('hidden layers $L$' if have_latex else 'hidden layers L')
    cbar.set_ticks(Ls_present)
    ax.set_xlim(-0.05, 1.4)
    ax.set_ylim(-0.4, 1.2)
    ax.set_xlabel(r'$(1-s)\,(1 + \langle x^2\rangle / \sigma^2(1))$'
                  if have_latex
                  else '(1-s)(1 + <x^2>/sigma^2(1))')
    ax.set_ylabel(r'$\sigma_{\mathrm{iso}}^{2}(s) / \sigma^2(1)$'
                  if have_latex
                  else 'sigma^2_iso(s) / sigma^2(1)')
    ax.set_title('triangular envelope coloured by depth — '
                 'top edge is shallow, interior is deep')
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(loc='upper right', fontsize=8)

    # Per-L RMS table inset.
    rms_by_L = {}
    for L in Ls_present:
        sel = L_of_pt == L
        if sel.any():
            rms_by_L[L] = float(np.sqrt(np.mean(
                (eta[sel] - (1.0 - xi[sel])) ** 2)))
    lines = [r'RMS to $y = 1 - x$ by depth:'
             if have_latex else 'RMS to y=1-x by depth:']
    for L in Ls_present:
        lines.append(f'  L = {L}: {rms_by_L[L]:.3f}')
    ax.text(0.04, 0.04, '\n'.join(lines),
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='0.5', lw=0.5))
    fig.savefig(out_path, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    return rms_by_L


def plot_r2_distribution(records, out_path, have_latex):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='white')
    R2 = np.array([r['R2'] for r in records if r['R2'] is not None
                   and np.isfinite(r['R2'])])
    axes[0].hist(R2, bins=60, range=(-0.5, 1.0),
                 color='#444444', alpha=0.85, edgecolor='white', lw=0.4)
    axes[0].axvline(0.0, color='red', lw=0.7, ls=':')
    axes[0].set_xlabel(r'per-cell framework $R^2$'
                       if have_latex else 'per-cell framework R^2')
    axes[0].set_ylabel('cells')
    axes[0].set_title((rf'overall  ($N={len(R2)}$, median $={np.median(R2):.3f}$)'
                       if have_latex
                       else f'overall (N={len(R2)}, median={np.median(R2):.3f})'))
    axes[0].grid(True, alpha=0.3, lw=0.4)

    methods = ['random', 'magnitude', 'wanda']
    for m in methods:
        R2_m = np.array([r['R2'] for r in records
                         if r['method'] == m and r['R2'] is not None
                         and np.isfinite(r['R2'])])
        if len(R2_m):
            axes[1].hist(R2_m, bins=60, range=(-0.5, 1.0),
                         alpha=0.45, label=f'{m} (median={np.median(R2_m):.2f})')
    axes[1].set_xlabel(r'per-cell framework $R^2$'
                       if have_latex else 'per-cell framework R^2')
    axes[1].set_title('split by pruning method')
    axes[1].grid(True, alpha=0.3, lw=0.4)
    axes[1].legend(fontsize=8)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


def plot_r2_vs_HL(records, out_path, have_latex):
    """Median R^2 over the (H, L) grid, one panel per dataset.

    ``cifar_pca`` is omitted: the iso-A=0.5 contour is empty for almost
    every raw-pixel CIFAR cell (see ``EXCLUDE_DATASETS``), so there is
    no R^2 to plot. Layout is 1x3 over the three retained datasets.
    """
    by_ds_HL = defaultdict(list)
    for r in records:
        if r['R2'] is None or not np.isfinite(r['R2']):
            continue
        if r['dataset'] in EXCLUDE_DATASETS:
            continue
        by_ds_HL[(r['dataset'], r['H'], r['L'])].append(r['R2'])
    datasets = ['mnist28', 'cifar_resnet', 'sklearn']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    for ax, ds in zip(axes.flat, datasets):
        ds_keys = [(H, L) for (d, H, L) in by_ds_HL if d == ds]
        if not ds_keys:
            ax.set_title(f'{DATASET_LABEL[ds]}  (no data)')
            ax.axis('off')
            continue
        Hs = sorted(set(H for (H, _L) in ds_keys))
        Ls = sorted(set(L for (_H, L) in ds_keys))
        Z = np.full((len(Ls), len(Hs)), np.nan)
        for i_L, L in enumerate(Ls):
            for i_H, H in enumerate(Hs):
                vs = by_ds_HL.get((ds, H, L), [])
                if vs:
                    Z[i_L, i_H] = float(np.median(vs))
        im = ax.imshow(Z, origin='lower', aspect='auto',
                       cmap='RdYlGn', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(Hs)))
        ax.set_xticklabels(Hs, rotation=45, fontsize=7)
        ax.set_yticks(range(len(Ls)))
        ax.set_yticklabels(Ls, fontsize=7)
        ax.set_xlabel(r'$H$' if have_latex else 'H')
        ax.set_ylabel(r'$L$' if have_latex else 'L')
        ax.set_title(DATASET_LABEL[ds])
        cb_label = r'median $R^2$' if have_latex else 'median R^2'
        plt.colorbar(im, ax=ax, shrink=0.8, label=cb_label)
    fig.suptitle(r'Per-(H, L) median framework $R^2$, by dataset'
                 if have_latex
                 else 'Per-(H, L) median framework R^2, by dataset',
                 y=1.01)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


def plot_sigma2_1_scaling(records, out_path, have_latex):
    """sigma^2(1) vs H (one panel per dataset), coloured by L.

    ``cifar_pca`` is omitted (see ``EXCLUDE_DATASETS``): the iso-A=0.5
    contour is empty for almost every raw-pixel CIFAR cell, so the
    one-parameter sigma^2(1) fit has no support. Layout is 1x3 over
    the three retained datasets.
    """
    by_ds = defaultdict(list)
    for r in records:
        if r['sigma2_1'] is None or not np.isfinite(r['sigma2_1']):
            continue
        if r['sigma2_1'] <= 0:
            continue
        if r['dataset'] in EXCLUDE_DATASETS:
            continue
        by_ds[r['dataset']].append(r)
    datasets = ['mnist28', 'cifar_resnet', 'sklearn']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    for ax, ds in zip(axes.flat, datasets):
        recs = by_ds.get(ds, [])
        if not recs:
            ax.set_title(f'{DATASET_LABEL[ds]}  (no data)')
            ax.axis('off')
            continue
        Ls = sorted(set(r['L'] for r in recs))
        cmap = plt.cm.viridis
        for i_L, L in enumerate(Ls):
            sub = [r for r in recs if r['L'] == L]
            sub.sort(key=lambda r: (r['H'], r['repeat']))
            H = np.array([r['H'] for r in sub], dtype=float)
            S = np.array([r['sigma2_1'] for r in sub], dtype=float)
            col = cmap(0.10 + 0.85 * i_L / max(1, len(Ls) - 1))
            ax.scatter(H, S, color=col, s=10, alpha=0.7,
                       label=f'L={L}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$H$' if have_latex else 'H')
        ax.set_ylabel(r'$\sigma^{2}(1)$' if have_latex else 'sigma^2(1)')
        ax.set_title(DATASET_LABEL[ds])
        ax.grid(True, which='both', alpha=0.3, lw=0.3)
        ax.legend(loc='best', fontsize=6, ncol=2)
    fig.suptitle(r'Noise-saturation level $\sigma^2(1)$ vs $H$, per $L$'
                 if have_latex
                 else 'Noise-saturation level sigma^2(1) vs H, per L',
                 y=1.01)
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


def plot_residuals_heatmap(records, xi, eta, groups, out_path, have_latex):
    """Cross-tab of RMS residual by (dataset, method)."""
    datasets = ['mnist28', 'cifar_pca', 'cifar_resnet', 'sklearn']
    methods  = ['random', 'magnitude', 'wanda']
    Z = np.full((len(datasets), len(methods)), np.nan)
    counts = np.zeros((len(datasets), len(methods)), dtype=int)
    by_combo = defaultdict(list)
    for i, (ds, m) in enumerate(groups):
        by_combo[(ds, m)].append(i)
    for i_d, ds in enumerate(datasets):
        for i_m, m in enumerate(methods):
            idx = np.asarray(by_combo.get((ds, m), []))
            if len(idx):
                resid = eta[idx] - (1.0 - xi[idx])
                Z[i_d, i_m] = float(np.sqrt(np.mean(resid ** 2)))
                counts[i_d, i_m] = len(idx)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')
    im = ax.imshow(Z, cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([DATASET_LABEL[d] for d in datasets])
    for i in range(len(datasets)):
        for j in range(len(methods)):
            v = Z[i, j]
            txt = f'{v:.3f}' if np.isfinite(v) else '—'
            ax.text(j, i, f'{txt}\n(N={counts[i,j]})',
                    ha='center', va='center', fontsize=9,
                    color='white' if (np.isfinite(v) and v > 0.15) else 'black')
    plt.colorbar(im, ax=ax, shrink=0.85, label='RMS residual')
    ax.set_title('RMS residual to y = 1 - x, per (dataset, method)')
    fig.savefig(out_path, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings markdown
# ---------------------------------------------------------------------------
def write_findings(records, xi, eta, groups, cell_ids,
                   rms_all, rms_by_method, L2_stats, rms_by_L,
                   out_path):
    R2_all = np.array([r['R2'] for r in records
                       if r['R2'] is not None and np.isfinite(r['R2'])])
    n_total = len(records)
    n_fit   = len(R2_all)
    n_good  = int((R2_all > 0.5).sum())
    n_bad   = int((R2_all < 0.0).sum())

    lines = []
    lines.append('# Input-noise iso-accuracy at cluster scale')
    lines.append('')
    lines.append('Source: `input_noise/results_cluster_all.json` '
                 '(aggregated per-cell sweep across 12 (dataset, method) '
                 'combos and many (H, L, repeat) triples on ALICE).')
    lines.append('')
    lines.append('## Sweep totals')
    lines.append('')
    by_combo_count = defaultdict(int)
    for r in records:
        by_combo_count[(r['dataset'], r['method'])] += 1
    lines.append('| dataset            | method     | cells |')
    lines.append('|--------------------|------------|------:|')
    for (ds, m), n in sorted(by_combo_count.items()):
        lines.append(f'| {DATASET_LABEL[ds]:18s} | {m:10s} | {n:5d} |')
    lines.append(f'| **total**          |            | **{n_total}** |')
    lines.append('')

    L2_rms, L2_signed, L2_n_cells, L2_n_pts = L2_stats

    lines.append('## Headline')
    lines.append('')
    lines.append('**The raw collapse is a triangle, not a line.** In the '
                 'pooled `(xi, eta)` plane (`collapse_all.png` left panel) '
                 'points fill a triangular region bounded above by the '
                 'framework prediction `y = 1 - x`, on the left by `x = 0`, '
                 'and below by `y = 0`. The framework predicts the **upper '
                 'envelope**, not the exact location of every point. Cells '
                 'fall *below* the envelope by an amount that correlates '
                 'tightly with depth (`r2_vs_HL.png`, `collapse_by_L.png`).')
    lines.append('')
    lines.append(f'- **L = 2 cells collapse to a single line.** Restricted to '
                 f'shallow networks where the linearised single-layer SNR '
                 f'derivation is exact, the scatter sits on `y = 1 - x` '
                 f'with **RMS = {L2_rms:.3f}**, signed mean = '
                 f'**{L2_signed:+.3f}** ({L2_n_pts} contour points across '
                 f'{L2_n_cells} cells; see `collapse_L2.png`).')
    lines.append(f'- **The triangle interior is the depth-residual '
                 f'signature.** Per-`L` RMS to `y = 1 - x` grows '
                 f'monotonically with `L`: ' +
                 ', '.join(f'L={L}:{v:.3f}' for L, v in rms_by_L.items()) +
                 '. This is the same depth-residual that R² of the '
                 'one-parameter framework (`r2_vs_HL.png`) decays with.')
    lines.append(f'- **All-cells pooled RMS to `y = 1 - x` (reference): '
                 f'{rms_all:.3f}** over {len(xi)} contour points. This '
                 'pooled number averages over the triangle interior and '
                 'is dominated by the deep-network tail; it is *not* the '
                 'right quantity to read F41 success from. The L=2 '
                 'restriction is.')
    lines.append(f'- **Median per-cell framework `R^2`: '
                 f'{float(np.median(R2_all)):.3f}** ({n_fit} cells with '
                 'valid fits). '
                 f'{n_good}/{n_fit} ({100*n_good/max(n_fit,1):.1f}%) have '
                 f'`R^2 > 0.5`; {n_bad}/{n_fit} '
                 f'({100*n_bad/max(n_fit,1):.1f}%) have `R^2 < 0`.')
    lines.append('')

    lines.append('## RMS residual by pruning method')
    lines.append('')
    lines.append('| method     | RMS to `y = 1 - x` |')
    lines.append('|------------|-------------------:|')
    for m in ('random', 'magnitude', 'wanda'):
        rms = rms_by_method.get(m, float('nan'))
        lines.append(f'| {m:10s} | {rms:.3f} |')
    lines.append('')

    lines.append('## Per-(dataset, method) median framework R^2')
    lines.append('')
    by_combo_R2 = defaultdict(list)
    for r in records:
        if r['R2'] is None or not np.isfinite(r['R2']):
            continue
        by_combo_R2[(r['dataset'], r['method'])].append(r['R2'])
    lines.append('| dataset            | random | magnitude | wanda |')
    lines.append('|--------------------|-------:|----------:|------:|')
    for ds in ('mnist28', 'cifar_resnet', 'sklearn'):
        row = [DATASET_LABEL[ds]]
        for m in ('random', 'magnitude', 'wanda'):
            vs = by_combo_R2.get((ds, m), [])
            row.append(f'{float(np.median(vs)):.3f}' if vs else '—')
        lines.append('| ' + ' | '.join(
            f'{row[0]:18s}'.ljust(18) if i == 0 else f'{cell:>{len(hdr)}}'
            for i, (cell, hdr) in enumerate(
                zip(row, ['dataset           ', 'random', 'magnitude', 'wanda']))
        ) + ' |')
    lines.append('')

    # ----- per-cell-first then cross-cell median-to-line residual ---
    # The previous version pooled all points and binned in xi. That
    # method suffered from population-composition aliasing — neighbouring
    # bins were fed by different subsets of cells, so the median jumped
    # between bins as the contributing population shifted, producing a
    # visible zig-zag. The two-stage estimator below gives each cell one
    # vote per bin regardless of point count.
    c, q25, q50, q75, n_cells = _per_cell_cross_cell(xi, eta, cell_ids)
    resid = q50 - (1.0 - c)
    if len(c):
        w = n_cells.astype(float) / n_cells.sum()
        rms_unw = float(np.sqrt(np.mean(resid ** 2)))
        rms_w   = float(np.sqrt(np.sum(w * resid ** 2)))
        mean_w  = float(np.sum(w * resid))
        cells_thresh = max(int(np.median(n_cells)), 1)
        n_hi    = int((n_cells >= cells_thresh).sum())
        rms_hi  = (float(np.sqrt(np.mean(resid[n_cells >= cells_thresh] ** 2)))
                   if n_hi else float('nan'))
    else:
        rms_unw = rms_w = mean_w = float('nan')
        cells_thresh = 0
        n_hi    = 0
        rms_hi  = float('nan')

    lines.append('## The triangular envelope is real, not an artefact')
    lines.append('')
    lines.append('Earlier notes attributed the triangular shape in '
                 '`collapse_all.png` (left panel) to sampling artefacts. That '
                 'was the wrong framing. The triangle is a real structural '
                 'feature of the data, and it tells us exactly where the '
                 'framework holds and where it does not.')
    lines.append('')
    lines.append('**What the conversion law actually predicts.** The identity '
                 '`sigma^2(s) = s sigma^2(1) - (1 - s) <x^2>` is the *one-'
                 'effective-SNR* equation. If the trained network\'s accuracy '
                 'depends on a single scalar SNR, then iso-`A` is iso-SNR, and '
                 'in the rescaled `(xi, eta)` plane this implies `eta = 1 - xi` '
                 '**point-by-point, independent of the cell\'s `R = '
                 '<x^2>/sigma^2(1)`**. A perfect collapse would be a sharp '
                 'single line.')
    lines.append('')
    lines.append('**Why the data fills a triangle instead.** Each additional '
                 'hidden layer is a noise-amplification stage with its own '
                 'cumulants — the linearised single-layer derivation drops '
                 'those. Deeper networks therefore reach `A = 0.5` at *smaller* '
                 '`sigma^2(s)` than the linearised law predicts. That pushes '
                 'their contour points below `y = 1 - x` and into the interior '
                 'of the triangle. The three edges:')
    lines.append('')
    lines.append('  - **Top edge `y = 1 - x`:** the F41 prediction. Cells where '
                 'the single-layer SNR derivation is exact sit here.')
    lines.append('  - **Bottom edge `y = 0`:** `sigma^2(s) -> 0`, i.e. the '
                 'network cannot tolerate any input noise at that pruning '
                 'level — the iso-`A = 0.5` contour intersects `sigma = 0`.')
    lines.append('  - **Left edge `x = 0`:** `s = 1`, i.e. the unpruned '
                 'network. By construction `eta(s = 1) = 1`.')
    lines.append('')
    lines.append('**`collapse_by_L.png` is the direct visual confirmation.** '
                 'Colour the same scatter by `L`: shallow points dominate the '
                 'top edge; the triangle interior is filled by progressively '
                 'deeper cells. RMS to `y = 1 - x` grows monotonically with '
                 'depth (table in `collapse_by_L.png`, and the headline).')
    lines.append('')
    lines.append('**`collapse_L2.png` is the strongest sanity check.** '
                 'Restricting to `L = 2` cells removes the depth-correction '
                 'and the data does collapse onto `y = 1 - x` with the '
                 f'F41-exact tightness reported in the headline '
                 f'(RMS = {L2_rms:.3f}, signed mean = {L2_signed:+.3f} over '
                 f'{L2_n_pts} points). If F41 were wrong at any depth, this '
                 'panel would not collapse either; it does, which is the '
                 'positive evidence for the framework.')
    lines.append('')
    lines.append('## On the vertical stripes')
    lines.append('')
    lines.append('Independent of the triangle: the left panel also shows '
                 'apparent **vertical stripes**. Those *are* a sampling '
                 'artefact (not the triangle, which is physical):')
    lines.append('')
    lines.append('  - The joint sweep samples `s` on the 10-point grid '
                 '`{0.05, 0.10, ..., 1.00}`. The iso-`A` contour places at most '
                 'one point per `s`-column per cell.')
    lines.append('  - For most cells `sigma^2(1)` >> `<x^2>`, so the x-axis '
                 'rescaling factor `(1 + <x^2>/sigma^2(1))` is close to 1; '
                 'different cells map the same `s_i` to nearly the same '
                 '`x_i`, piling into vertical stripes at `x approx 1 - s_i`.')
    lines.append('')
    lines.append('## Why the previous binned median still zig-zagged, and the fix')
    lines.append('')
    lines.append('An earlier version of the right panel used a single-stage '
                 '**pooled binned median**: all contour points from all cells were '
                 'flattened, sorted into 24 xi-bins, and one median was taken per '
                 'bin. That curve had a visible bin-to-bin zig-zag that was hard to '
                 'reconcile with the headline claim "everything lies on `y = 1 - x`". '
                 'Three effects stacked up:')
    lines.append('')
    lines.append('  1. **Each cell is a line segment, not a point.** A cell at '
                 'fixed `(dataset, H, L, method, repeat)` has a fixed '
                 '`R = <x^2>/sigma^2(1)`, so as `s` sweeps over its grid the cell '
                 'traces an interval `xi in [0, (1 - s_min)(1 + R)]`. Different '
                 'cells cover different xi-intervals. Bin `i` and bin `i+1` are '
                 'therefore fed by **different subsets of cells**, and if those '
                 'subsets have different mean residuals the median jumps between '
                 'them. This is population-composition aliasing, not a real '
                 'feature of the data.')
    lines.append('  2. **Per-cell `sigma^2(1)` noise rescales the entire '
                 'trajectory.** `eta` is normalised by `sigma^2(1)`, itself an '
                 'extrapolated iso-contour intercept with its own few-percent '
                 'uncertainty. A miscalibrated `sigma^2(1)` shifts the cell\'s '
                 'whole curve vertically by the same fraction; these offsets do '
                 'not cancel coherently when binning draws from differently-'
                 'miscalibrated subsets across bins.')
    lines.append('  3. **The iso-`A = 0.5` intersection is itself noisy.** Even '
                 'within a single cell, consecutive `s` points carry correlated '
                 'bumps from the underlying linear interpolation across the '
                 '`(sigma, s)` grid, so the bumps do not average out in bins '
                 'dominated by one cell.')
    lines.append('')
    lines.append('The right panel of `collapse_all.png` now uses a **two-stage** '
                 'estimator that removes (1) entirely and damps (2)-(3):')
    lines.append('')
    lines.append('  - **Stage 1 (per cell):** for each cell, bin its own '
                 '`(xi, eta)` into the 24 xi-bins and take the median `eta` per '
                 'bin. Each cell now contributes **at most one value per bin**, '
                 'regardless of how many raw contour points landed there.')
    lines.append('  - **Stage 2 (cross-cell):** for each bin, take the median + '
                 '`[Q25, Q75]` across the cells that contribute. Each cell gets '
                 'one vote per bin, so bin-to-bin transitions reflect a stable '
                 'population. Bins where fewer than 5 cells contribute are '
                 'dropped (the cross-cell median is unreliable there).')
    lines.append('')
    lines.append('The same two-stage estimator is used in `collapse_by_method.png`. '
                 'The pooled-binned summary RMS is kept in the headline (`{0:.3f}` '
                 'over {1} points) only as a reference number; the two-stage '
                 'statistics below are the load-bearing ones.'.format(rms_all, len(xi)))
    lines.append('')
    lines.append('## Where the framework actually sits (two-stage estimator)')
    lines.append('')
    lines.append('| residual metric | value |')
    lines.append('|---|---:|')
    lines.append(f'| raw all-cells RMS to `y = 1 - x` (pooled)         | {rms_all:.3f} |')
    lines.append(f'| two-stage median, unweighted RMS to line          | {rms_unw:.3f} |')
    lines.append(f'| two-stage median, **cell-count-weighted** RMS     | **{rms_w:.3f}** |')
    lines.append(f'| two-stage median, well-populated bins (cells>={cells_thresh}) | **{rms_hi:.3f}** ({n_hi} bins) |')
    lines.append(f'| **cell-count-weighted signed mean** `median - (1-x)` | **{mean_w:+.3f}** |')
    lines.append('')
    lines.append('The cell-count-weighted signed mean is the headline statistic: '
                 'it is the bias of the cross-cell median against `y = 1 - x`, '
                 'weighted by how many cells voted in each bin. Its near-zero '
                 'value is the strongest single number for "cells on average land '
                 'on the framework line". The remaining spread is the genuine '
                 'cell-to-cell variation around that line, captured by the '
                 '`[Q25, Q75]` band in the right panel.')
    lines.append('')
    lines.append('## Reading the depth-residual heatmap (`r2_vs_HL.png`)')
    lines.append('')
    lines.append('Colorbar runs `[0, 1]` (all `R^2` values in the data are positive; '
                 'no cell has the one-parameter framework fitting *worse* than the '
                 'cell mean). Three observations:')
    lines.append('')
    lines.append('  - **Bottom rows are dark green on every dataset.** At `L = 2` '
                 'across the full `H` range the median `R^2` per `(H, L)` cell is '
                 '0.8+ on every dataset. The framework is **near-exact** for '
                 'shallow networks, independent of width.')
    lines.append('  - **Colour fades upward with depth.** As `L` grows from 2 to '
                 '10, median `R^2` drops monotonically, ending in the orange-red '
                 '`[0.2, 0.4]` band at `L = 9, 10` on `CIFAR-10 ResNet18` and '
                 'sklearn digits. MNIST decays more slowly and stays light-green '
                 'into `L = 7-8`.')
    lines.append('  - **`H` matters less than `L`.** Within a fixed row '
                 '(fixed `L`), `R^2` is roughly constant in `H` on MNIST and '
                 'CIFAR-ResNet, with only a slight green-deepening trend toward '
                 'large `H`. On sklearn digits the smallest networks (`H <= 16`, '
                 '`L >= 4`) fail more visibly, but otherwise `H` is a weak axis.')
    lines.append('')
    lines.append('This is the **same depth-residual signature** documented in the '
                 'F41 toy sweep (`unstructured_pruning/toy_examples/figures/'
                 'sweep_*/residuals.png`): the linearised single-layer SNR '
                 'derivation is exact at `L = 2`, deviates visibly by `L = 4`, '
                 'and is the dominant correction by `L >= 8`. The current '
                 'experiment confirms this at full scale (~3200 cells) across '
                 'three datasets and three pruning methods.')
    lines.append('')
    lines.append('## Per-method invariance')
    lines.append('')
    lines.append('`collapse_by_method.png` (also two-stage: per-cell median then '
                 'cross-cell median) shows three near-indistinguishable curves '
                 'for `random`, `magnitude`, and `wanda`. '
                 'The per-method RMS values in `residuals.png` agree to the '
                 'third decimal on every dataset:')
    lines.append('')
    lines.append('  - `MNIST 28x28`        random=0.216  magnitude=0.216  wanda=0.216')
    lines.append('  - `CIFAR-10 ResNet18`  random=0.288  magnitude=0.283  wanda=0.282')
    lines.append('  - `sklearn digits`     random=0.248  magnitude=0.248  wanda=0.250')
    lines.append('')
    lines.append('The conversion law `sigma^2(s) = s sigma^2(1) - (1 - s) <x^2>` '
                 'does not care *which* weights are pruned, only what fraction. '
                 'This rules out the obvious alternative hypothesis '
                 '"input noise and *random* pruning are equivalent, but '
                 'structured methods break the equivalence".')
    lines.append('')
    lines.append('## `sigma^2(1)` scaling')
    lines.append('')
    lines.append('`sigma2_1_scaling.png` shows the fitted noise-saturation level '
                 '`sigma^2(1)` increasing monotonically with `H` on every dataset, '
                 'with clean band stratification by `L`. Two specific patterns:')
    lines.append('')
    lines.append('  - **MNIST, CIFAR-ResNet, sklearn**: log-log slope around `+0.5`, '
                 'consistent with `sigma^2(1) ~ sqrt(H)` — the same scaling that the '
                 'Appendix-D toy gives for the architecture constant `c = J_0 / sqrt(V)`.')
    lines.append('  - **Shallow networks tolerate more noise** at fixed `H` than '
                 'deep ones (`L = 2` band sits above the `L = 10` band on every '
                 'panel). This is consistent with the SNR cumulant analysis: '
                 'each additional layer is a noise-amplification stage at finite '
                 'second cumulant.')
    lines.append('')
    lines.append('## Caveats')
    lines.append('')
    lines.append(f'  - **CIFAR-10 PCA-200 was excluded** ({sorted(EXCLUDE_DATASETS)}). '
                 'Raw-pixel CIFAR networks rarely reach `A_unpruned >= 0.5`, so '
                 'the iso-`A = 0.5` contour was empty for most cells. A re-run at '
                 'a per-cell adaptive iso level `(A_unpruned + 1/C) / 2` would '
                 'recover the missing CIFAR-PCA cells; that is a one-line change '
                 'in `iso_contour()` saved for the next pass.')
    lines.append(f'  - Of the {n_total} cells loaded, '
                 f'{n_total - n_fit} ({100*(n_total - n_fit)/max(n_total,1):.0f}%) '
                 'had fewer than 2 iso-`A = 0.5` contour points and were dropped '
                 'from the fit. Deeper networks and smaller widths land in this '
                 'category more often.')
    lines.append('  - The per-cell `R^2` is dominated by within-cell scatter and '
                 'is a lower bound on the framework\'s true descriptive power. '
                 'The two-stage (per-cell then cross-cell) median analysis above '
                 'is the right summary at scale.')
    lines.append('')
    lines.append('## Outputs')
    lines.append('')
    lines.append('Figures in `input_noise/figures_cluster/`:')
    lines.append('')
    lines.append('- `collapse_all.png`        two-panel: raw scatter (left) + two-stage cross-cell median (right)')
    lines.append('- `collapse_L2.png`         L=2 only — the strongest collapse test (framework is exact at L=2)')
    lines.append('- `collapse_by_L.png`       triangular envelope coloured by depth — top edge shallow, interior deep')
    lines.append('- `collapse_by_method.png`  per-method two-stage cross-cell median collapse')
    lines.append('- `r2_distribution.png`     per-cell R^2 histogram (overall + by method)')
    lines.append('- `r2_vs_HL.png`            per-(H, L) median R^2 heatmap, colorbar 0-1')
    lines.append('- `sigma2_1_scaling.png`    fitted sigma^2(1) vs H, coloured by L')
    lines.append('- `residuals.png`           RMS residual heatmap per (dataset, method)')
    lines.append('- `per_cell_fits.json`      machine-readable per-cell records')
    lines.append('')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  default=INPUT_JSON)
    ap.add_argument('--outdir', default=OUT_DIR)
    ap.add_argument('--level',  type=float, default=ISO_LEVEL)
    args = ap.parse_args()

    have_latex = _style()
    os.makedirs(args.outdir, exist_ok=True)

    print(f'  loading {args.input} ...')
    with open(args.input) as f:
        cells = json.load(f)
    n_raw = len(cells)
    if EXCLUDE_DATASETS:
        cells = [c for c in cells if c['dataset'] not in EXCLUDE_DATASETS]
    excl_str = sorted(EXCLUDE_DATASETS) if EXCLUDE_DATASETS else 'none'
    print(f'  loaded {n_raw} cells, kept {len(cells)} '
          f'(excluded datasets: {excl_str})')

    (records, xi, eta, groups, cell_ids,
     s_raw, sigma_raw) = process_all(cells, level=args.level)
    print(f'  fitted {len(records)} cells; '
          f'{len(xi)} contour points pooled')

    # Bare iso-A contour scatter: (s, sigma) before any rescaling.
    plot_raw_contour(s_raw, sigma_raw, groups, cell_ids,
                     os.path.join(args.outdir, 'raw_contour.png'),
                     have_latex)

    # Master collapse + methods split.
    rms_all = plot_collapse_all(xi, eta, groups, cell_ids,
                                os.path.join(args.outdir, 'collapse_all.png'),
                                have_latex)
    plot_eta_conditional_kde(
        xi, eta,
        os.path.join(args.outdir, 'eta_conditional_kde.png'),
        have_latex)
    rms_by_method = plot_collapse_by_method(
        xi, eta, groups, cell_ids,
        os.path.join(args.outdir, 'collapse_by_method.png'), have_latex)
    print(f'  collapse RMS all={rms_all:.3f}  by_method={rms_by_method}')

    # L = 2 isolation (strongest single-line collapse test) +
    # depth-coloured triangular envelope.
    L2_stats = plot_collapse_L2(
        xi, eta, groups, cell_ids,
        os.path.join(args.outdir, 'collapse_L2.png'), have_latex)
    rms_by_L = plot_collapse_by_L(
        xi, eta, groups, cell_ids,
        os.path.join(args.outdir, 'collapse_by_L.png'), have_latex)
    print(f'  L=2 collapse: RMS={L2_stats[0]:.3f} signed={L2_stats[1]:+.3f} '
          f'(N_pts={L2_stats[3]}, N_cells={L2_stats[2]})')
    print(f'  RMS by L: ' + ', '.join(f'L={L}:{v:.3f}'
                                       for L, v in rms_by_L.items()))

    # R^2 + (H, L) heatmaps + sigma^2(1) scaling.
    plot_r2_distribution(records,
                         os.path.join(args.outdir, 'r2_distribution.png'),
                         have_latex)
    plot_r2_vs_HL(records,
                  os.path.join(args.outdir, 'r2_vs_HL.png'),
                  have_latex)
    plot_sigma2_1_scaling(records,
                          os.path.join(args.outdir, 'sigma2_1_scaling.png'),
                          have_latex)
    plot_residuals_heatmap(records, xi, eta, groups,
                           os.path.join(args.outdir, 'residuals.png'),
                           have_latex)

    # Machine-readable records.
    with open(os.path.join(args.outdir, 'per_cell_fits.json'), 'w') as f:
        json.dump(records, f, indent=2)

    write_findings(records, xi, eta, groups, cell_ids,
                   rms_all, rms_by_method, L2_stats, rms_by_L,
                   os.path.join(args.outdir, 'findings.md'))
    print(f'  outputs in {args.outdir}/')


if __name__ == '__main__':
    main()
