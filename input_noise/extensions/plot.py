#!/usr/bin/env python3
"""Plots for the §5.2 extensions (C1, C3, C5).

Reads the three results.json under input_noise/extensions/{iso_levels,
depth_cells, seed_replicates}/ and emits

  - extensions/collapse_multi_iso.png  parameter-free (xi, eta) collapse
                                       across 4 iso levels (C1)
  - extensions/depth_r2.png            per-cell R^2 vs L scatter (C3)
  - extensions/seed_error_bars.png     per-cell R^2 (mean +/- std)
                                       across 3 seeds (C5)

Run::

    .venv/bin/python -m input_noise.extensions.plot
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

EXT = 'input_noise/extensions'

ISO_COLORS = {0.30: '#1f77b4', 0.50: '#2ca02c',
              0.70: '#d62728', 0.90: '#9467bd'}


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
    return have_latex


# ---------------------------------------------------------------------------
# C1: multi-iso collapse
# ---------------------------------------------------------------------------
def plot_multi_iso(path_in, path_out, have_latex):
    with open(path_in) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(7.0, 5.5), facecolor='white')
    xg = np.linspace(0.0, 1.3, 50)
    ax.plot(xg, 1.0 - xg, 'k-', lw=1.0, alpha=0.6,
            label=(r'framework: $\eta = 1 - \xi$' if have_latex
                   else 'framework: eta = 1 - xi'))

    for lvl_str in sorted(data['pooled']):
        lvl = float(lvl_str)
        col = ISO_COLORS.get(lvl, 'gray')
        xis, etas = [], []
        for rec in data['per_cell']:
            r = rec['per_iso'][lvl_str]
            if r['n'] < 2 or not np.isfinite(r['sigma2_1']) or r['sigma2_1'] <= 0:
                continue
            s_arr  = np.array([p[0] for p in r['contour']])
            sg_arr = np.array([p[1] for p in r['contour']])
            x2 = rec['x2_mean']
            xi  = (1.0 - s_arr) * (1.0 + x2 / r['sigma2_1'])
            eta = (sg_arr ** 2) / r['sigma2_1']
            xis.extend(xi.tolist()); etas.extend(eta.tolist())
        rms = data['pooled'][lvl_str]['rms_to_line']
        ax.scatter(xis, etas, color=col, s=30, alpha=0.75,
                   label=(rf'$A_\star={lvl:.1f}$:  '
                          rf'RMS$={rms:.3f}$  (n={len(xis)})'
                          if have_latex
                          else f'A*={lvl:.1f}: RMS={rms:.3f} (n={len(xis)})'))

    ax.set_xlabel(r'$\xi = (1-s)\,(1 + \langle x^2\rangle / \sigma^2(1;A_\star))$'
                  if have_latex
                  else 'xi = (1-s)(1 + <x^2>/sigma^2(1;A*))')
    ax.set_ylabel(r'$\eta = \sigma_{\mathrm{iso}}^2(s;A_\star)/\sigma^2(1;A_\star)$'
                  if have_latex
                  else 'eta = sigma_iso^2(s;A*)/sigma^2(1;A*)')
    ax.set_title('Multi-iso parameter-free collapse (5 pilot cells)')
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(fontsize=9, loc='upper right')
    fig.savefig(path_out, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# C3: depth R^2 scatter
# ---------------------------------------------------------------------------
def plot_depth_r2(path_in, path_out, have_latex):
    with open(path_in) as f:
        data = json.load(f)
    cells = data['per_cell']

    # Also overlay the pilot L=2 R^2 values for reference, if iso_levels
    # results exist (read from C1 output).
    pilot_path = os.path.join(EXT, 'iso_levels/results.json')
    pilot_R2_at_half = []  # one per pilot cell
    if os.path.exists(pilot_path):
        with open(pilot_path) as f:
            pilot = json.load(f)
        for rec in pilot['per_cell']:
            r = rec['per_iso']['0.50']
            if r['n'] >= 2 and np.isfinite(r['R2']):
                pilot_R2_at_half.append((rec['L'], r['R2'],
                                         rec['dataset'], rec['H']))

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor='white')
    ds_color = {'sklearn': '#9467bd', 'mnist28': '#1f77b4'}
    jitter_rng = np.random.default_rng(0)
    for rec in cells:
        r = rec['per_iso']['0.50']
        if r['n'] < 2 or not np.isfinite(r['R2']):
            continue
        L = rec['L']; ds = rec['dataset']
        Lj = L + jitter_rng.uniform(-0.12, 0.12)
        ax.scatter(Lj, r['R2'], color=ds_color.get(ds, 'gray'),
                   marker='o', s=55, alpha=0.85, edgecolor='k', lw=0.4)
        ax.annotate(f'H={rec["H"]}', (Lj, r['R2']),
                    fontsize=6, alpha=0.6,
                    xytext=(3, 2), textcoords='offset points')

    for (L, R2, ds, H) in pilot_R2_at_half:
        Lj = L + jitter_rng.uniform(-0.12, 0.12)
        ax.scatter(Lj, R2, color=ds_color.get(ds, 'gray'),
                   marker='s', s=60, alpha=0.95, edgecolor='k', lw=0.7)

    # Median per L over depth cells.
    by_L = {}
    for rec in cells:
        r = rec['per_iso']['0.50']
        if r['n'] >= 2 and np.isfinite(r['R2']):
            by_L.setdefault(rec['L'], []).append(r['R2'])
    Ls = sorted(by_L)
    meds = [float(np.median(by_L[L])) for L in Ls]
    ax.plot(Ls, meds, 'k--', lw=1.2, marker='D', ms=6, alpha=0.7,
            label=('median per $L$' if have_latex else 'median per L'))

    # Legend handles for dataset colours + pilot markers.
    h_sklearn = plt.scatter([], [], color=ds_color['sklearn'], marker='o',
                            s=55, label='sklearn digits (cluster)',
                            edgecolor='k', lw=0.4)
    h_mnist = plt.scatter([], [], color=ds_color['mnist28'], marker='o',
                          s=55, label='MNIST 28x28 (cluster)',
                          edgecolor='k', lw=0.4)
    h_pilot = plt.scatter([], [], color='gray', marker='s', s=60,
                          label='pilot cells (square)',
                          edgecolor='k', lw=0.7)
    h_med = plt.Line2D([], [], color='k', ls='--', marker='D',
                       label='median per $L$' if have_latex else 'median per L')
    ax.legend(handles=[h_sklearn, h_mnist, h_pilot, h_med],
              fontsize=9, loc='lower left')

    ax.set_xticks([2, 3, 4, 5])
    ax.set_xlabel(r'depth $L$' if have_latex else 'depth L')
    ax.set_ylabel(r'$R^2$ of rational fit at $A=0.5$'
                  if have_latex else 'R^2 of rational fit at A=0.5')
    ax.set_title(r'Depth breakdown: framework $R^2$ falls with $L$'
                 if have_latex
                 else 'Depth breakdown: framework R^2 falls with L')
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_ylim(0.0, 1.05)
    fig.savefig(path_out, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# C5: seed replicates
# ---------------------------------------------------------------------------
def plot_seed_bars(path_in, path_out, have_latex):
    with open(path_in) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), facecolor='white')
    cells = data['per_cell']
    labels = [f'{c["dataset"]}\nH={c["H"]} L={c["L"]}' for c in cells]
    xs = np.arange(len(cells))
    means = [c['R2_mean'] for c in cells]
    stds  = [c['R2_std']  for c in cells]
    # Highlight the L=4 cell
    colors = ['#d62728' if c['L'] == 4 else '#1f77b4' for c in cells]

    ax.bar(xs, means, yerr=stds, color=colors, alpha=0.7,
           ecolor='k', capsize=6, edgecolor='k', lw=0.5)
    for i, c in enumerate(cells):
        for s in c['per_seed']:
            if np.isfinite(s['R2']) and s['n'] >= 2:
                ax.scatter(i, s['R2'], color='k', s=15, alpha=0.7,
                           zorder=3)

    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r'$R^2$ of rational fit at $A=0.5$'
                  if have_latex else 'R^2 (rational fit, A=0.5)')
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Seed replicates (n=3): mean +/- std per pilot cell')
    ax.grid(True, axis='y', alpha=0.3, lw=0.4)
    fig.savefig(path_out, facecolor='white')
    plt.close(fig)


def main():
    have_latex = _style()

    print('  plotting collapse_multi_iso.png (C1) ...')
    plot_multi_iso(os.path.join(EXT, 'iso_levels/results.json'),
                   os.path.join(EXT, 'collapse_multi_iso.png'),
                   have_latex)

    print('  plotting depth_r2.png (C3) ...')
    plot_depth_r2(os.path.join(EXT, 'depth_cells/results.json'),
                  os.path.join(EXT, 'depth_r2.png'),
                  have_latex)

    print('  plotting seed_error_bars.png (C5) ...')
    plot_seed_bars(os.path.join(EXT, 'seed_replicates/results.json'),
                   os.path.join(EXT, 'seed_error_bars.png'),
                   have_latex)
    print('  done.')


if __name__ == '__main__':
    main()
