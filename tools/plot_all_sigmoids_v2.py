#!/usr/bin/env python3
r"""v2 sigmoid overlay: uses three-parameter fit with A_0 = 1/C fixed.

Sibling of ``tools/plot_all_sigmoids.py``. Reads ``sigmoid_*_v2`` fields
written by ``tools/refit_sigmoids_v2.py`` and plots

    A(s) = A_floor + (A_inf - A_floor) / (1 + exp(-beta * (s - s_0)))

with ``A_floor = 1/C = 0.1``. Overwrites the same output paths as v1.

Usage:
    .venv/bin/python tools/plot_all_sigmoids_v2.py --mode fit
    .venv/bin/python tools/plot_all_sigmoids_v2.py --mode data
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


A_FLOOR = 0.1   # 10-class random-guess floor

DATASET_CMAPS = {
    'mnist28':      'Blues',
    'cifar_pca':    'Greys',
    'cifar_resnet': 'Reds',
    'sklearn':      'Greens',
}
DATASET_PRETTY = {
    'mnist28':      'MNIST 28x28',
    'cifar_pca':    'CIFAR-10 PCA-200',
    'cifar_resnet': 'CIFAR-10 ResNet18',
    'sklearn':      'sklearn digits',
}
METHOD_LS = {'magnitude': '-', 'random': '--', 'wanda': ':'}
DIR_RE = re.compile(r'unstructured_figures_(.+)_(magnitude|random|wanda)$')


def _parse_dir(path):
    m = DIR_RE.match(os.path.basename(os.path.dirname(path)))
    return (m.group(1), m.group(2)) if m else (None, None)


def _sigmoid_v2(s, A_inf, s_0, beta):
    x = -beta * (np.asarray(s, dtype=float) - s_0)
    return A_FLOOR + (A_inf - A_FLOOR) / (1.0 + np.exp(np.clip(x, -500, 500)))


def _load_cells(root):
    pattern = os.path.join(root, 'unstructured_pruning', 'figures',
                           'unstructured_figures_*', 'scaling_results.json')
    paths = sorted(glob.glob(pattern))
    out = []
    for p in paths:
        dataset, method = _parse_dir(p)
        if dataset is None:
            continue
        with open(p) as f:
            cells = json.load(f)
        by_HL = defaultdict(list)
        for c in cells:
            r2 = c.get('sigmoid_R2_v2')
            if r2 is None or not np.isfinite(r2):
                continue
            by_HL[(c['H'], c['L'])].append(c)
        for (H, L), grp in by_HL.items():
            densities_all = [g.get('densities') for g in grp]
            accs_all = [g.get('accs_mean') for g in grp]
            densities = (np.asarray(densities_all[0], dtype=float)
                         if densities_all and densities_all[0] is not None
                         else None)
            if (densities is not None and accs_all and all(
                    a is not None and len(a) == len(densities)
                    for a in accs_all)):
                accs_mean = np.mean(np.asarray(accs_all, dtype=float), axis=0)
            else:
                accs_mean = None
            out.append({
                'dataset': dataset, 'method': method,
                'H': int(H), 'L': int(L),
                'n_params': int(np.mean([g['n_params'] for g in grp])),
                'A_inf': float(np.mean([g['sigmoid_A_inf_v2'] for g in grp])),
                's_0':   float(np.mean([g['sigmoid_s_0_v2']   for g in grp])),
                'beta':  float(np.mean([g['sigmoid_beta_v2']  for g in grp])),
                'R2':    float(np.mean([g['sigmoid_R2_v2']    for g in grp])),
                'densities': densities,
                'accs_mean': accs_mean,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--mode', choices=('fit', 'data'), default='fit')
    ap.add_argument('--output', default=None)
    ap.add_argument('--x-min', type=float, default=-0.5)
    ap.add_argument('--x-max', type=float, default=+0.5)
    ap.add_argument('--n-x', type=int, default=400)
    ap.add_argument('--r2-min', type=float, default=0.85)
    ap.add_argument('--alpha', type=float, default=0.28)
    ap.add_argument('--lw', type=float, default=0.55)
    ap.add_argument('--exclude', nargs='*', default=['cifar_pca'])
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(
            'unstructured_pruning', 'figures',
            f'sigmoid_overlay_{args.mode}.png')

    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex': have_latex, 'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    cells = _load_cells(args.root)
    n_all = len(cells)
    excluded = set(args.exclude or [])
    cells = [c for c in cells
             if c['R2'] >= args.r2_min and c['dataset'] not in excluded]
    print(f'  {len(cells)} of {n_all} cells pass R^2_v2 >= {args.r2_min}')

    arch_by_ds = defaultdict(set)
    for c in cells:
        arch_by_ds[c['dataset']].add(c['n_params'])
    arch_rank = {ds: {n: i for i, n in enumerate(sorted(arch_by_ds[ds]))}
                 for ds in arch_by_ds}

    fig, ax = plt.subplots(figsize=(9.5, 6.0), facecolor='white')
    s_grid = np.linspace(args.x_min, args.x_max, args.n_x)

    counts = defaultdict(int)
    dropped = 0
    for c in cells:
        cmap = plt.colormaps[DATASET_CMAPS.get(c['dataset'], 'Greys')]
        n_arch = max(1, len(arch_by_ds[c['dataset']]) - 1)
        rk = arch_rank[c['dataset']][c['n_params']]
        intensity = 0.25 + 0.70 * (rk / n_arch)
        color = cmap(intensity)
        ls = METHOD_LS.get(c['method'], '-')
        if args.mode == 'fit':
            A = _sigmoid_v2(c['s_0'] + s_grid, c['A_inf'], c['s_0'], c['beta'])
            ax.plot(s_grid, A, color=color, ls=ls, lw=args.lw, alpha=args.alpha)
        else:
            if c['densities'] is None or c['accs_mean'] is None:
                dropped += 1; continue
            x = np.asarray(c['densities']) - c['s_0']
            order = np.argsort(x)
            x = x[order]
            y = np.asarray(c['accs_mean'])[order]
            keep = (x >= args.x_min) & (x <= args.x_max)
            if keep.sum() < 2:
                dropped += 1; continue
            ax.plot(x[keep], y[keep], color=color, ls=ls,
                    lw=args.lw, alpha=args.alpha, marker='.',
                    ms=1.8, markeredgewidth=0)
        counts[(c['dataset'], c['method'])] += 1
    if dropped:
        print(f'  (dropped {dropped} cells in data mode)')

    ax.axvline(0.0, color='0.5', lw=0.5, linestyle=':')
    ax.axhline((A_FLOOR + 1.0) / 2, color='0.5', lw=0.5, linestyle=':')
    ax.axhline(A_FLOOR, color='0.7', lw=0.4, linestyle='-')
    ax.set_xlabel(r'$s - s_0$' if have_latex else 's - s_0')
    ax.set_ylabel(r'$A(s)$' if have_latex else 'A(s)')
    title_word = 'fits' if args.mode == 'fit' else 'curves'
    ax.set_title(f'All sigmoid {title_word} (v2, A_0 = 1/C fixed) '
                 f'centred at s_0 (N = {len(cells)})')
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, lw=0.4)

    # ---- Inset zoom on the s_0 crossing region (top-left corner) -----
    zoom_x_half = 0.06         # x in [-0.06, +0.06]
    zoom_y_half = 0.18         # y in [mid-0.18, mid+0.18]
    y_mid = (A_FLOOR + 1.0) / 2
    ax_inset = inset_axes(ax, width="38%", height="40%",
                          loc='upper left', borderpad=1.2)
    s_grid_zoom = np.linspace(-zoom_x_half, zoom_x_half, 200)
    for c in cells:
        cmap = plt.colormaps[DATASET_CMAPS.get(c['dataset'], 'Greys')]
        n_arch = max(1, len(arch_by_ds[c['dataset']]) - 1)
        rk = arch_rank[c['dataset']][c['n_params']]
        intensity = 0.25 + 0.70 * (rk / n_arch)
        color = cmap(intensity)
        ls = METHOD_LS.get(c['method'], '-')
        if args.mode == 'fit':
            A = _sigmoid_v2(c['s_0'] + s_grid_zoom,
                            c['A_inf'], c['s_0'], c['beta'])
            ax_inset.plot(s_grid_zoom, A, color=color, ls=ls,
                          lw=args.lw + 0.15, alpha=min(args.alpha + 0.1, 1.0))
        else:
            if c['densities'] is None or c['accs_mean'] is None:
                continue
            x = np.asarray(c['densities']) - c['s_0']
            order = np.argsort(x)
            x = x[order]
            y = np.asarray(c['accs_mean'])[order]
            keep = (x >= -zoom_x_half) & (x <= zoom_x_half)
            if keep.sum() < 1:
                continue
            ax_inset.plot(x[keep], y[keep], color=color, ls=ls,
                          lw=args.lw + 0.15,
                          alpha=min(args.alpha + 0.1, 1.0),
                          marker='.', ms=2.4, markeredgewidth=0)
    ax_inset.axvline(0.0, color='0.4', lw=0.6, linestyle=':')
    ax_inset.axhline(y_mid, color='0.4', lw=0.6, linestyle=':')
    ax_inset.set_xlim(-zoom_x_half, zoom_x_half)
    ax_inset.set_ylim(y_mid - zoom_y_half, y_mid + zoom_y_half)
    ax_inset.tick_params(labelsize=7, length=2, pad=1)
    ax_inset.set_xticks([-zoom_x_half, 0, zoom_x_half])
    ax_inset.set_yticks([y_mid - zoom_y_half, y_mid, y_mid + zoom_y_half])
    ax_inset.grid(True, alpha=0.25, lw=0.3)
    inset_title = (r'zoom: $s \to s_0$' if have_latex
                   else 'zoom: s -> s_0')
    ax_inset.set_title(inset_title, fontsize=8, pad=2)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(0.7)
    try:
        mark_inset(ax, ax_inset, loc1=2, loc2=4,
                   fc='none', ec='0.45', lw=0.5, ls='--')
    except Exception:
        pass

    # ---- Legends (both bottom-right) ---------------------------------
    ds_handles = []
    for ds in sorted(DATASET_PRETTY):
        if ds not in arch_by_ds:
            continue
        cmap = plt.colormaps[DATASET_CMAPS.get(ds, 'Greys')]
        ds_handles.append(Line2D([0], [0], color=cmap(0.78), lw=2.2,
                                 label=DATASET_PRETTY[ds]))
    method_handles = [Line2D([0], [0], color='0.25', lw=1.4, ls=ls, label=m)
                      for m, ls in METHOD_LS.items()]
    leg_methods = ax.legend(handles=method_handles,
                            title='Pruning (linestyle)',
                            loc='lower right',
                            bbox_to_anchor=(0.998, 0.012),
                            fontsize=8, title_fontsize=8,
                            framealpha=0.92)
    ax.add_artist(leg_methods)
    ax.legend(handles=ds_handles, title='Dataset (hue)',
              loc='lower right',
              bbox_to_anchor=(0.998, 0.215),
              fontsize=8, title_fontsize=8,
              framealpha=0.92)

    for (ds, m) in sorted(counts):
        print(f'    {ds:14s} {m:9s}  N={counts[(ds, m)]:4d}')
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, facecolor='white')
    plt.close(fig)
    print(f'Saved figure: {args.output}')


if __name__ == '__main__':
    main()
