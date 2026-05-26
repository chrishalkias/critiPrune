#!/usr/bin/env python3
r"""Single-plot overlay of all sigmoid fits across (dataset, method, arch).

For every cell in the main-text unstructured_pruning scaling JSONs
(``unstructured_pruning/figures/unstructured_figures_<dataset>_<method>/
scaling_results.json``), average the cached sigmoid parameters across
repeats and plot

    A(s)  =  A_0 + (A_inf - A_0) / (1 + exp(-beta * (s - s_0)))

centred at its own ``s_0`` so the x-axis is ``s - s_0``. All curves
share the same horizontal centre, exposing the spread in slope
(``beta``) and asymptotes (``A_inf``, ``A_0``).

Color encoding
--------------
- **Hue**       : dataset (one colormap per dataset)
- **Intensity** : architecture rank within the dataset's colormap
  (sorted by ``n_params``; small networks pale, large ones saturated)
- **Linestyle** : pruning method (``-`` magnitude, ``--`` random, ``:`` wanda)

Run::

    .venv/bin/python -m tools.plot_all_sigmoids
    # custom threshold + zoom
    .venv/bin/python -m tools.plot_all_sigmoids --r2-min 0.9 \
        --x-min -0.4 --x-max 0.4
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
METHOD_LS = {
    'magnitude': '-',
    'random':    '--',
    'wanda':     ':',
}
DIR_RE = re.compile(
    r'unstructured_figures_(.+)_(magnitude|random|wanda)$')


def _parse_dir(path: str):
    """Return ``(dataset, method)`` from the parent directory name."""
    m = DIR_RE.match(os.path.basename(os.path.dirname(path)))
    return (m.group(1), m.group(2)) if m else (None, None)


def _sigmoid(s, A_inf, A_0, s_0, beta):
    """Same as ``pruning.pruning.sigmoid_fn``."""
    x = -beta * (np.asarray(s, dtype=float) - s_0)
    return A_0 + (A_inf - A_0) / (1.0 + np.exp(np.clip(x, -500, 500)))


def _load_cells(root: str):
    """Walk the 12 main-text scaling JSONs and return per-cell records,
    averaging sigmoid parameters across repeats per ``(H, L)``."""
    pattern = os.path.join(
        root, 'unstructured_pruning', 'figures',
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
            r2 = c.get('sigmoid_R2')
            if r2 is None or not np.isfinite(r2):
                continue
            by_HL[(c['H'], c['L'])].append(c)
        for (H, L), grp in by_HL.items():
            # Repeats share densities; just average accs_mean point-wise.
            densities_all = [g.get('densities') for g in grp]
            accs_all      = [g.get('accs_mean') for g in grp]
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
                'dataset':   dataset,
                'method':    method,
                'H':         int(H),
                'L':         int(L),
                'n_params':  int(np.mean([g['n_params']        for g in grp])),
                'A_inf':     float(np.mean([g['sigmoid_A_inf'] for g in grp])),
                'A_0':       float(np.mean([g['sigmoid_A_0']   for g in grp])),
                's_0':       float(np.mean([g['sigmoid_s_0']   for g in grp])),
                'beta':      float(np.mean([g['sigmoid_beta']  for g in grp])),
                'R2':        float(np.mean([g['sigmoid_R2']    for g in grp])),
                'densities': densities,
                'accs_mean': accs_mean,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--mode', choices=('fit', 'data'), default='fit',
                    help='"fit" plots the cached sigmoid curves; "data" '
                         'plots the cached (density, accs_mean) points '
                         'centred at the fit s_0. Same color encoding.')
    ap.add_argument('--output', default=None,
                    help='override output path; default depends on --mode')
    ap.add_argument('--x-min',   type=float, default=-0.5)
    ap.add_argument('--x-max',   type=float, default=+0.5)
    ap.add_argument('--n-x',     type=int,   default=400)
    ap.add_argument('--r2-min',  type=float, default=0.85,
                    help='discard cells whose sigmoid R^2 is below this')
    ap.add_argument('--alpha',   type=float, default=0.28)
    ap.add_argument('--lw',      type=float, default=0.55)
    ap.add_argument('--exclude', nargs='*', default=['cifar_pca'],
                    help='dataset keys to skip; default: cifar_pca')
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(
            'unstructured_pruning', 'figures',
            f'sigmoid_overlay_{args.mode}.png')

    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex':      have_latex,
        'font.family':      'serif',
        'mathtext.fontset': 'cm',
        'figure.dpi':       120,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}')

    cells = _load_cells(args.root)
    n_all = len(cells)
    excluded = set(args.exclude or [])
    cells = [c for c in cells
             if c['R2'] >= args.r2_min and c['dataset'] not in excluded]
    print(f'  {len(cells)} of {n_all} cells pass R^2 >= {args.r2_min}'
          + (f' and dataset not in {sorted(excluded)}' if excluded else ''))

    # Architecture rank by parameter count, per dataset (sets intensity).
    arch_by_ds = defaultdict(set)
    for c in cells:
        arch_by_ds[c['dataset']].add(c['n_params'])
    arch_rank = {
        ds: {n: i for i, n in enumerate(sorted(arch_by_ds[ds]))}
        for ds in arch_by_ds
    }

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
            A = _sigmoid(c['s_0'] + s_grid, c['A_inf'], c['A_0'],
                         c['s_0'], c['beta'])
            ax.plot(s_grid, A, color=color, ls=ls,
                    lw=args.lw, alpha=args.alpha)
        else:  # 'data'
            if c['densities'] is None or c['accs_mean'] is None:
                dropped += 1
                continue
            x = np.asarray(c['densities']) - c['s_0']
            order = np.argsort(x)
            x = x[order]
            y = np.asarray(c['accs_mean'])[order]
            keep = (x >= args.x_min) & (x <= args.x_max)
            if keep.sum() < 2:
                dropped += 1
                continue
            ax.plot(x[keep], y[keep], color=color, ls=ls,
                    lw=args.lw, alpha=args.alpha, marker='.',
                    ms=1.8, markeredgewidth=0)
        counts[(c['dataset'], c['method'])] += 1
    if dropped:
        print(f'  (dropped {dropped} cells in data mode: missing samples '
              'or empty window)')

    # --- Legends -----------------------------------------------------
    ds_handles = []
    for ds in sorted(DATASET_PRETTY):
        if ds not in arch_by_ds:
            continue
        cmap = plt.colormaps[DATASET_CMAPS.get(ds, 'Greys')]
        ds_handles.append(Line2D([0], [0], color=cmap(0.78), lw=2.2,
                                 label=DATASET_PRETTY[ds]))
    leg_ds = ax.legend(handles=ds_handles, title='Dataset (hue)',
                       loc='upper left', fontsize=8, title_fontsize=8)
    ax.add_artist(leg_ds)
    method_handles = [Line2D([0], [0], color='0.25', lw=1.4, ls=ls,
                             label=m) for m, ls in METHOD_LS.items()]
    ax.legend(handles=method_handles, title='Pruning (linestyle)',
              loc='lower right', fontsize=8, title_fontsize=8)

    ax.axvline(0.0, color='0.5', lw=0.5, linestyle=':')
    ax.axhline(0.5, color='0.5', lw=0.5, linestyle=':')
    ax.set_xlabel(r'$s - s_0$' if have_latex else 's - s_0')
    ax.set_ylabel(r'$A(s)$' if have_latex else 'A(s)')
    if args.mode == 'fit':
        title_tex = (rf'All sigmoid fits centred at $s_0$ '
                     rf'(N = {len(cells)})')
        title_plain = f'All sigmoid fits centred at s_0 (N = {len(cells)})'
    else:
        title_tex = (rf'All empirical $A(s)$ curves centred at $s_0$ '
                     rf'(N = {len(cells)})')
        title_plain = (f'All empirical A(s) curves centred at s_0 '
                       f'(N = {len(cells)})')
    ax.set_title(title_tex if have_latex else title_plain)
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, lw=0.4)

    # Per-cell breakdown printout (also dumped to title space below).
    print('  cells per (dataset, method):')
    for (ds, m) in sorted(counts):
        print(f'    {ds:14s} {m:9s}  N={counts[(ds, m)]:4d}')

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, facecolor='white')
    plt.close(fig)
    print(f'Saved figure: {args.output}')


if __name__ == '__main__':
    main()
