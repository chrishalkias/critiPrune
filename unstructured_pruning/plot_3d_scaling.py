#!/usr/bin/env python3
r"""Publication-quality 3D comparison plots: empirical vs. theoretical
$s_0(H, L)$ manifolds.

For each ``assets/unstructured_pruning/unstructured_figures_<dataset>_<method>``
directory containing both ``scaling_results.json`` and ``scaling_laws.json``,
this script writes ``s0_3d.png`` next to the existing 2D plots.

Layout: two side-by-side 3D panels sharing axes/limits/view.
    Left  -- empirical manifold: a triangulated surface stitched directly
             from the measured $(H, L, s_0)$ points; data points and
             their $z$-error bars are overlaid so the relationship between
             samples and the empirical surface is visible.
    Right -- theoretical manifold: the smooth power-law fit
             $s_0 = a\,H^{\alpha}\,L^{\gamma}$ over the same $(H, L)$
             ranges.

All text is rendered through LaTeX (``text.usetex=True``) with a
Computer-Modern serif body font for publication-grade typography.

Usage
-----
    python -m unstructured_pruning.plot_3d_scaling
    python -m unstructured_pruning.plot_3d_scaling --base FIGS_DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ----------------------------------------------------------------------------
# Publication style: real LaTeX rendering when available, mathtext fallback.
# ----------------------------------------------------------------------------
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
        'axes.labelsize':         11,
        'axes.titlesize':         12,
        'axes.linewidth':         0.7,
        'xtick.labelsize':        9,
        'ytick.labelsize':        9,
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
_log10 = lambda x: np.log10(np.maximum(np.asarray(x, dtype=float), 1e-12))


def _aggregate(rows, min_r2=0.80):
    """Group good fits by (H, L) and return mean +/- std arrays."""
    bins = defaultdict(list)
    for r in rows:
        if r.get('sigmoid_R2') is None or r['sigmoid_R2'] < min_r2:
            continue
        bins[(int(r['H']), int(r['L']))].append(float(r['sigmoid_s_0']))
    if not bins:
        return None
    H, L, mu, sd, n = [], [], [], [], []
    for (h, l), vals in sorted(bins.items()):
        arr = np.asarray(vals)
        H.append(h); L.append(l)
        mu.append(arr.mean()); sd.append(arr.std()); n.append(len(arr))
    return (np.array(H,  dtype=float),
            np.array(L,  dtype=float),
            np.array(mu, dtype=float),
            np.array(sd, dtype=float),
            np.array(n,  dtype=int))


def _style_3d_axes(ax):
    """Clean, paper-quality 3D pane styling."""
    pane_face = (1.0, 1.0, 1.0, 1.0)
    pane_edge = (0.55, 0.55, 0.55, 0.6)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane_face)
        axis.pane.set_edgecolor(pane_edge)
        axis.pane.set_linewidth(0.6)
        axis._axinfo['grid'].update({
            'color': (0.75, 0.75, 0.75, 0.45),
            'linewidth': 0.45, 'linestyle': '-',
        })
    ax.tick_params(axis='both', which='major', labelsize=9, pad=1)


def _log_ticks(lo, hi):
    return np.arange(int(np.floor(np.log10(lo))),
                     int(np.ceil(np.log10(hi))) + 1)


def _set_common_axes(ax, H_lo, H_hi, L_lo, L_hi, z_lo, z_hi):
    h_ticks = _log_ticks(H_lo, H_hi)
    z_ticks = _log_ticks(z_lo, z_hi)
    ax.set_xticks(h_ticks)
    ax.set_xticklabels([rf'$10^{{{int(t)}}}$' for t in h_ticks])
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([rf'$10^{{{int(t)}}}$' for t in z_ticks])
    ax.set_xlim(_log10(H_lo) - 0.05, _log10(H_hi) + 0.05)
    ax.set_ylim(L_lo - 0.3, L_hi + 0.3)
    ax.set_zlim(_log10(z_lo), _log10(z_hi))
    ax.set_xlabel(r'Width $H$', labelpad=10)
    ax.set_ylabel(r'Depth $L$', labelpad=10)
    ax.set_zlabel(r'Critical density $s_{0}$', labelpad=10)
    ax.view_init(elev=22, azim=-58)


def _safe_label(name: str) -> str:
    """Convert a directory token into a LaTeX-safe display token."""
    pretty = {
        'sklearn':      r'\textsc{sklearn} digits',
        'mnist28':      r'\textsc{mnist}-28',
        'cifar':        r'\textsc{cifar}-10',
        'pca':          r'\textsc{pca}',
        'resnet':       r'\textsc{ResNet}',
        'random':       r'random',
        'magnitude':    r'magnitude',
        'wanda':        r'\textsc{wanda}',
    }
    parts = name.split('_')
    return ' '.join(pretty.get(p, p.replace('_', r'\_')) for p in parts)


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def render_3d(results, scaling, output_path, dataset_method=''):
    agg = _aggregate(results)
    if agg is None or scaling is None or 's0' not in scaling:
        return None
    H, L, mu, sd, _ = agg
    if len(H) < 4:
        return None
    sr = scaling['s0']
    has_err = (sd > 0).any()

    H_lo, H_hi = float(H.min()), float(H.max())
    L_lo, L_hi = float(L.min()), float(L.max())
    if H_hi == H_lo: H_hi *= 1.05
    if L_hi == L_lo: L_hi += 1
    z_lo = max(float((mu - np.where(sd > 0, sd, 0)).min()) * 0.7, 1e-5)
    z_hi = float((mu + sd).max()) * 1.4

    Hl = _log10(H)
    sl = _log10(mu)

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(15.5, 7.5), facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0],
                          left=0.04, right=0.96, top=0.90, bottom=0.06,
                          wspace=0.08)
    ax_data = fig.add_subplot(gs[0, 0], projection='3d')
    ax_pred = fig.add_subplot(gs[0, 1], projection='3d')

    for ax in (ax_data, ax_pred):
        ax.set_box_aspect((1.0, 1.0, 0.8))
        _style_3d_axes(ax)
        _set_common_axes(ax, H_lo, H_hi, L_lo, L_hi, z_lo, z_hi)

    cmap = plt.cm.viridis
    # Shared colour scale: mapped to s_0 (the z-axis) in log10 space so the
    # two manifolds are directly comparable cell-for-cell.  s_0 lives in
    # (0, 1]; we centre the norm on the observed range across BOTH the
    # empirical points and the predicted surface so neither panel saturates.
    SM_pred_preview = (sr['a']
                       * np.geomspace(H_lo, H_hi, 50)[None, :] ** sr['alpha']
                       * np.linspace(L_lo, L_hi, 50)[:, None] ** sr['gamma'])
    z_norm_lo = max(min(float(mu.min()), float(SM_pred_preview.min())),
                    1e-5)
    z_norm_hi = min(max(float(mu.max()), float(SM_pred_preview.max())),
                    1.0)
    cnorm = Normalize(vmin=_log10(z_norm_lo), vmax=_log10(z_norm_hi))

    # ---- LEFT: empirical manifold from the data --------------------------
    # Render order matters in 3D: plot_trisurf occludes anything drawn
    # before it.  We split each error bar into a lower half (drawn BEFORE
    # the surface) and an upper half (drawn AFTER) so the manifold sits
    # between the two halves and the upper bars stay visible above it.

    # ---- LEFT: empirical manifold from the data --------------------------
    tri = mtri.Triangulation(Hl, L)
    ax_data.plot_trisurf(tri, sl, cmap=cmap, norm=cnorm,
                         edgecolor='black', linewidth=0.25,
                         alpha=0.82, antialiased=True, shade=True)

    for h, l, m, e, hl_, sl_ in zip(H, L, mu, sd, Hl, sl):
        c = cmap(cnorm(sl_))
        if has_err and e > 0:
            zlo = _log10(max(m - e, z_lo * 0.5))
            zhi = _log10(m + e)
            ax_data.plot([hl_, hl_], [l, l], [zlo, zhi],
                         color='black', lw=0.7, alpha=0.7, zorder=4)
            for zz in (zlo, zhi):
                ax_data.plot([hl_], [l], [zz], marker='_', mew=0.7,
                             ms=4, color='black', alpha=0.7, zorder=4)
        ax_data.scatter(hl_, l, sl_, color=c,
                        edgecolor='black', linewidth=0.5,
                        s=42, depthshade=True, zorder=5)

    ax_data.set_title(r'\textbf{Empirical manifold}\quad'
                      r'$\hat{s}_{0}(H, L)$', pad=4)

    # ---- RIGHT: theoretical scaling-law surface --------------------------
    Hg = np.geomspace(H_lo, H_hi, 80)
    Lg = np.linspace(L_lo, L_hi, 80)
    HM, LM = np.meshgrid(Hg, Lg)
    SM = sr['a'] * HM ** sr['alpha'] * LM ** sr['gamma']

    ax_pred.plot_surface(_log10(HM), LM, _log10(SM),
                         cmap=cmap, norm=cnorm,
                         edgecolor='black', linewidth=0.18,
                         alpha=0.88, antialiased=True, shade=True,
                         rstride=4, cstride=4)
    ax_pred.set_title(r'\textbf{Power-law manifold}\quad'
                      r'$s_{0} = a\,H^{\alpha}\,L^{\gamma}$', pad=4)

    # ---- super-title + formula inset ------------------------------------
    if dataset_method:
        fig.suptitle(
            rf'{_safe_label(dataset_method)}: '
            r'empirical vs.\ predicted $s_{0}(H, L)$',
            y=0.985)

    formula = (
        r'\begin{aligned}'
        rf's_{{0}} &= {sr["a"]:.4f}\,H^{{{sr["alpha"]:.3f}}}'
        rf'\,L^{{{sr["gamma"]:.3f}}}\\[2pt]'
        rf'R^{{2}}_{{\mathrm{{adj}}}} &= {sr["R2"]:.3f}'
        rf'\quad (N = {len(H)}\ \mathrm{{cells}})'
        r'\end{aligned}'
    )
    ax_pred.text2D(0.02, 0.98, rf'${formula}$',
                   transform=ax_pred.transAxes,
                   fontsize=10, va='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.4', fc='white',
                             ec='0.55', lw=0.6, alpha=0.94))

    fig.savefig(output_path, facecolor='white')
    plt.close(fig)
    return output_path


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main(base='assets/unstructured_pruning'):
    have_latex = _configure_style()
    print(f'  Rendering with text.usetex = {have_latex}')

    saved, skipped = [], []
    for d in sorted(glob.glob(os.path.join(base, 'unstructured_figures_*'))):
        results_p = os.path.join(d, 'scaling_results.json')
        scaling_p = os.path.join(d, 'scaling_laws.json')
        if not os.path.isfile(results_p):
            skipped.append((d, 'no scaling_results.json')); continue
        results = json.load(open(results_p))
        scaling = (json.load(open(scaling_p))
                   if os.path.isfile(scaling_p) else None)
        token = (os.path.basename(d)
                 .replace('unstructured_figures_', ''))
        out = os.path.join(d, 's0_3d.png')
        p = render_3d(results, scaling, out, dataset_method=token)
        if p:
            print(f'  Saved: {p}')
            saved.append(p)
        else:
            skipped.append((d, 'insufficient data or no s_0 fit'))

    print(f'\nDone. {len(saved)} plots saved, {len(skipped)} skipped.')
    for d, why in skipped:
        print(f'  skipped: {os.path.basename(d)}  ({why})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='assets/unstructured_pruning',
                    help='directory containing unstructured_figures_* subdirs')
    args = ap.parse_args()
    main(base=args.base)
