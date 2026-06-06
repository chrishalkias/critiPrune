#!/usr/bin/env python3
"""Plots and conversion-law fit for the input-noise iso-accuracy experiment.

Reads ``input_noise/results.json`` produced by ``run_experiment.py`` and
emits

  - per-cell A(s) and A(sigma) side-by-side panels
    (``figures/<dataset>/cell_H<H>_L<L>_curves.png``);
  - per-cell iso-accuracy contour map in (s, sigma)
    (``figures/<dataset>/cell_H<H>_L<L>_contours.png``);
  - one combined SNR-collapse plot across all cells
    (``figures/snr_collapse.png``);
  - the cross-cell conversion fit  ``sigma^2_iso(s) = A - B (1-s)/s``
    (``figures/conversion_fit.png``).
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from input_noise.core import iso_accuracy_contour


RESULTS_PATH = 'input_noise/results.json'
FIG_ROOT     = 'input_noise/figures'

ISO_LEVELS = [0.30, 0.50, 0.70, 0.90]


def _style():
    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex': have_latex,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'figure.dpi': 120,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
    })
    return have_latex


def _sigmoid(x, A_inf, A_0, x_0, beta):
    z = -beta * (np.asarray(x, dtype=float) - x_0)
    return A_0 + (A_inf - A_0) / (1.0 + np.exp(np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# Plot 1: A(s) and A(sigma) side by side
# ---------------------------------------------------------------------------
def plot_curves(cell, output_path, have_latex):
    H, L = cell['H'], cell['L']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4), facecolor='white')

    s = np.array(cell['pruning']['s'])
    a = np.array(cell['pruning']['accs_mean'])
    sd = np.array(cell['pruning']['accs_std'])
    axL.errorbar(s, a, yerr=sd, fmt='o', ms=4, color='C0', capsize=0)
    f = cell['pruning']['fit']
    if np.isfinite(f['s_0']):
        sg = np.linspace(0.0, 1.0, 200)
        axL.plot(sg, _sigmoid(sg, f['A_inf'], f['A_0'], f['s_0'], f['beta']),
                 '-', color='C0', lw=1.2,
                 label=(rf'$s_0={f["s_0"]:.3f}$, $\beta={f["beta"]:.1f}$'
                        if have_latex
                        else f's_0={f["s_0"]:.3f}, beta={f["beta"]:.1f}'))
        axL.axvline(f['s_0'], color='C0', lw=0.5, ls=':')
    axL.set_xlabel(r'$s$ (Bernoulli density)' if have_latex else 's (density)')
    axL.set_ylabel(r'$A(s)$' if have_latex else 'A')
    axL.set_ylim(-0.02, 1.02)
    axL.set_title('Pruning sweep')
    axL.grid(True, alpha=0.3, lw=0.4)
    axL.legend(fontsize=8, loc='lower right')

    sg = np.array(cell['noise']['sigma'])
    a  = np.array(cell['noise']['accs_mean'])
    sd = np.array(cell['noise']['accs_std'])
    axR.errorbar(sg, a, yerr=sd, fmt='o', ms=4, color='C3', capsize=0)
    f2 = cell['noise']['fit_in_sigma']
    if np.isfinite(f2['sigma_0']):
        sgg = np.linspace(0.0, sg.max(), 300)
        axR.plot(sgg, _sigmoid(sgg, f2['A_inf'], f2['A_0'],
                               f2['sigma_0'], f2['beta']),
                 '-', color='C3', lw=1.2,
                 label=(rf'$\sigma_0={f2["sigma_0"]:.3f}$, '
                        rf'$R^2={f2["R2"]:.3f}$'
                        if have_latex
                        else f'sigma_0={f2["sigma_0"]:.3f}, '
                             f'R^2={f2["R2"]:.3f}'))
        axR.axvline(f2['sigma_0'], color='C3', lw=0.5, ls=':')
    axR.set_xlabel(r'$\sigma$ (input noise std)' if have_latex
                   else 'sigma (input noise std)')
    axR.set_ylabel(r'$A(\sigma)$' if have_latex else 'A')
    axR.set_ylim(-0.02, 1.02)
    axR.set_title('Input-noise sweep')
    axR.grid(True, alpha=0.3, lw=0.4)
    axR.legend(fontsize=8, loc='lower left')

    fig.suptitle(f'{cell["dataset"]}  H={H}  L={L}  '
                 f'A_full={cell["normal_acc"]:.3f}', y=1.02)
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: iso-accuracy contour map per cell
# ---------------------------------------------------------------------------
def plot_contours(cell, output_path, have_latex):
    H, L = cell['H'], cell['L']
    s_grid = np.array(cell['joint']['s_grid'])
    sg_grid = np.array(cell['joint']['sigma_grid'])
    A = np.array(cell['joint']['mean'])  # shape (n_s, n_sigma)

    # Reshape to 2D mesh: rows = sigma, cols = s, so contour sees (X=s, Y=sigma)
    S, SG = np.meshgrid(s_grid, sg_grid, indexing='xy')
    A_T = A.T  # rows = sigma, cols = s

    fig, ax = plt.subplots(figsize=(6.5, 5.0), facecolor='white')
    im = ax.pcolormesh(S, SG, A_T, cmap='viridis', shading='auto',
                       vmin=0.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'$A(s, \sigma)$' if have_latex else 'A')

    cs = ax.contour(S, SG, A_T, levels=ISO_LEVELS,
                    colors='white', linewidths=1.0)
    ax.clabel(cs, fmt='%.2f', fontsize=7, inline=True)
    ax.set_xlabel(r'$s$' if have_latex else 's')
    ax.set_ylabel(r'$\sigma$ (input noise)' if have_latex
                  else 'sigma (input noise)')
    ax.set_title(f'{cell["dataset"]}  H={H}  L={L}  iso-A contours')

    fig.savefig(output_path, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: SNR-collapse
# ---------------------------------------------------------------------------
def plot_snr_collapse(cells, output_path, have_latex):
    """For both pruning and noise sweeps, plot A vs the framework-predicted
    SNR-like quantity.  Pruning prediction: ``chi_prune = sqrt(s/(1-s))``.
    Input-noise prediction (single hidden layer, normalised inputs):
    ``chi_noise = 1 / sqrt(sigma^2 / <x^2> + (1-s_0)/s_0)``, with s_0 fixed
    per cell from the pruning fit; for s=1 this reduces to
    ``1/sqrt(sigma^2/<x^2>)``.

    A linear-axis SNR plot with Phi overlay tests whether both sweeps lie on
    the same A = Phi(c * chi) curve up to a single per-cell scale ``c``.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor='white')
    chi_grid = np.linspace(0.0, 5.0, 400)
    ax.plot(chi_grid, norm.cdf(chi_grid), 'k--', lw=0.8,
            label=(r'$\Phi(\chi)$' if have_latex else 'Phi(chi)'))

    cmap = plt.cm.viridis
    n_cells = len(cells)
    for i, cell in enumerate(cells):
        col = cmap(0.15 + 0.7 * i / max(1, n_cells - 1))
        x2 = cell['x2_mean']
        # Pruning side: chi = sqrt(s/(1-s)) * c, where c is fit so the
        # curve lands on Phi.  We extract c from the half-transition:
        # at s_0, A = (A_inf + A_0)/2 = Phi(0) = 0.5, so c * chi(s_0) is
        # the offset.  We just plot chi_prune (no c) and let the family
        # collapse by colour.
        s = np.array(cell['pruning']['s'])
        a = np.array(cell['pruning']['accs_mean'])
        chi_p = np.sqrt(s / np.clip(1 - s, 1e-6, None))
        ax.scatter(chi_p, a, color=col, marker='o', s=20, alpha=0.7,
                   label=(f'{cell["dataset"]} H={cell["H"]} L={cell["L"]}'
                          if i < 6 else None))

        # Input-noise side, treated as adding sigma^2 to <x^2>:
        # the framework says the effective SNR-like variable is
        # chi_noise = 1 / sqrt(1 + sigma^2/<x^2>),
        # which goes from 1 at sigma=0 to 0 as sigma -> infty.
        sigma = np.array(cell['noise']['sigma'])
        an    = np.array(cell['noise']['accs_mean'])
        chi_n = 1.0 / np.sqrt(1.0 + sigma * sigma / x2)
        # Rescale so chi_n at sigma=0 lands at chi_prune(s=1) = inf;
        # use chi_n's range mapped to the maximum chi_prune on the cell.
        scale = max(chi_p) * 1.05
        ax.scatter(chi_n * scale, an, color=col, marker='x', s=22,
                   alpha=0.6)

    ax.set_xlabel(r'$\chi$ (pruning: $\sqrt{s/(1-s)}$, '
                  r'noise: $1/\sqrt{1+\sigma^2/\langle x^2\rangle}$ '
                  r'rescaled)'
                  if have_latex
                  else 'chi (pruning vs. noise)')
    ax.set_ylabel(r'$A$' if have_latex else 'A')
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(
        r'SNR collapse: $A$ vs framework-predicted $\chi$ '
        r'(\textbullet\ pruning, $\times$ input noise)'
        if have_latex
        else 'SNR collapse: A vs framework-predicted chi '
             '(o pruning, x input noise)')
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Conversion fit
# ---------------------------------------------------------------------------
def fit_conversion(cells, level=0.5):
    """Per-cell extraction of the iso-A contour ``sigma^2_iso(s)`` plus the
    framework-predicted curve

        sigma^2(s)  =  s * sigma^2(1)  -  (1 - s) * <x^2>

    (linear in ``s``, with a single free parameter ``sigma^2(1)`` per
    cell). Equivalent in the conventional ``x = (1-s)/s * <x^2>`` axis to
    the Möbius / rational curve

        sigma^2(x)  =  <x^2> * (sigma^2(1) - x) / (<x^2> + x).

    Per-cell ``sigma^2(1)`` is fit by least-squares against the linear-in-s
    form (one parameter, n_contour points). R^2 is reported in the
    linear-in-s coordinates.
    """
    per_cell = []
    for c in cells:
        x2 = c['x2_mean']
        contour = iso_accuracy_contour(
            {(s, sg): (c['joint']['mean'][i_s][i_sg],
                       c['joint']['std'][i_s][i_sg])
             for i_s, s in enumerate(c['joint']['s_grid'])
             for i_sg, sg in enumerate(c['joint']['sigma_grid'])},
            c['joint']['s_grid'], c['joint']['sigma_grid'], level)
        if len(contour) < 1:
            per_cell.append({'cell': (c['dataset'], c['H'], c['L']),
                             'contour': contour, 'fit': None})
            continue
        s_arr  = np.array([p[0] for p in contour])
        sg_arr = np.array([p[1] for p in contour])
        # One-parameter fit: minimise ||sg^2 - (s * sigma2_1 - (1-s) * x2)||^2
        #   d/d(sigma2_1) = 2 * sum_i s_i * (sg_i^2 - s_i * sigma2_1 + (1-s_i)*x2)
        # ⇒  sigma2_1 = (sum s_i (sg_i^2 + (1-s_i) x2)) / sum s_i^2
        num = float(np.sum(s_arr * (sg_arr ** 2 + (1.0 - s_arr) * x2)))
        den = float(np.sum(s_arr ** 2))
        sigma2_1 = num / den if den > 0 else float('nan')
        y_pred = s_arr * sigma2_1 - (1.0 - s_arr) * x2
        y = sg_arr ** 2
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        per_cell.append({
            'cell':    (c['dataset'], c['H'], c['L']),
            'contour': contour,
            'fit':     {'sigma2_1': float(sigma2_1),
                        'R2':       float(R2),
                        'x2_mean':  float(x2),
                        'n':        int(len(contour))},
        })
    return per_cell


def plot_collapse(per_cell, output_path, have_latex):
    """Framework prediction: under additive-variance SNR, iso-accuracy gives

        sigma^2(s) = s * sigma^2(1) - (1 - s) * <x^2>

    so rearranging,

        sigma^2(s) / sigma^2(1)  =  1  -  (1 - s) * (1 + <x^2> / sigma^2(1)).

    Plotting LHS vs (1-s)*(1+<x^2>/sigma^2(1)) should collapse all cells
    onto the unit-slope line ``y = 1 - x`` — *parameter-free* across (H, L,
    dataset). Deviations from that line are the framework's residual.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.0), facecolor='white')
    xg = np.linspace(0.0, 1.3, 50)
    ax.plot(xg, 1.0 - xg, 'k-', lw=1.2,
            label=(r'framework: $y = 1 - x$' if have_latex
                   else 'framework: y = 1 - x'))

    cmap = plt.cm.viridis
    valid = [r for r in per_cell if r['fit'] is not None]
    n = max(1, len(valid))
    all_x, all_y = [], []
    for i, rec in enumerate(valid):
        ds, H, L = rec['cell']
        s_arr  = np.array([p[0] for p in rec['contour']])
        sg_arr = np.array([p[1] for p in rec['contour']])
        x2 = rec['fit']['x2_mean']
        # Use the per-cell framework-fit sigma^2(1) directly.
        sigma2_at_1 = float(rec['fit']['sigma2_1'])
        if not np.isfinite(sigma2_at_1) or sigma2_at_1 <= 0:
            continue
        x_norm = (1.0 - s_arr) * (1.0 + x2 / sigma2_at_1)
        y_norm = (sg_arr ** 2) / sigma2_at_1
        col = cmap(0.12 + 0.76 * i / n)
        ax.scatter(x_norm, y_norm, color=col, s=26, alpha=0.85,
                   label=f'{ds} H={H} L={L}')
        all_x.extend(x_norm.tolist())
        all_y.extend(y_norm.tolist())

    if len(all_x) >= 2:
        ax_x = np.asarray(all_x); ax_y = np.asarray(all_y)
        # Residual versus the parameter-free framework line.
        resid = ax_y - (1.0 - ax_x)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        ax.text(0.04, 0.06, (rf'RMS residual to $y=1-x$: {rms:.3f}'
                             if have_latex
                             else f'RMS residual to y=1-x: {rms:.3f}'),
                transform=ax.transAxes, ha='left', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='0.5', lw=0.5))
    else:
        rms = float('nan')

    ax.set_xlabel(r'$(1-s)\,(1 + \langle x^2\rangle / \sigma^2(1))$'
                  if have_latex
                  else '(1-s) * (1 + <x^2>/sigma^2(1))')
    ax.set_ylabel(r'$\sigma_{\mathrm{iso}}^{2}(s) / \sigma^2(1)$'
                  if have_latex
                  else 'sigma^2_iso(s) / sigma^2(1)')
    ax.set_title('Parameter-free framework collapse (iso-A=0.5)')
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(fontsize=7, loc='upper right')
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)
    return {'rms_residual': rms, 'n_points': len(all_x)}


def plot_conversion(per_cell, output_path, have_latex):
    """Overlay the framework-predicted *rational* curve

        sigma^2(s)  =  <x^2> * (sigma^2(1) - x) / (<x^2> + x),    x = (1-s)/s * <x^2>

    on the iso-A data points, one curve per cell. This is the correct
    functional form derived from iso-SNR (linear in ``s``, rational in the
    plotted axis ``x``). Earlier linear-regression overlays were the wrong
    functional form and have been removed.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor='white')
    cmap = plt.cm.viridis
    valid = [r for r in per_cell if r['fit'] is not None]
    n = max(1, len(valid))

    R2_list = []
    for i, rec in enumerate(valid):
        ds, H, L = rec['cell']
        s_arr  = np.array([p[0] for p in rec['contour']])
        sg_arr = np.array([p[1] for p in rec['contour']])
        x2     = rec['fit']['x2_mean']
        sigma2_1 = rec['fit']['sigma2_1']
        R2_list.append(rec['fit']['R2'])

        x = (1.0 - s_arr) / np.clip(s_arr, 1e-6, None) * x2
        y = sg_arr ** 2
        col = cmap(0.15 + 0.7 * i / n)
        ax.scatter(x, y, color=col, s=26, alpha=0.85,
                   label=(rf'{ds} $H={H}$ $L={L}$: '
                          rf'$\sigma^2(1)={sigma2_1:.2f}$, '
                          rf'$R^2={rec["fit"]["R2"]:.3f}$'
                          if have_latex
                          else f'{ds} H={H} L={L}: '
                               f'sigma^2(1)={sigma2_1:.2f}, '
                               f'R²={rec["fit"]["R2"]:.3f}'))
        # Framework rational curve from x = 0 (s=1) to x where sigma^2 -> 0.
        x_max = max(x.max(), sigma2_1) * 1.1
        xg = np.linspace(0.0, x_max, 200)
        yg = x2 * (sigma2_1 - xg) / (x2 + xg)
        ax.plot(xg, yg, color=col, lw=1.0, alpha=0.85)

    R2_mean = float(np.mean(R2_list)) if R2_list else float('nan')
    ax.plot([], [], color='0.3', lw=1.0,
            label=(rf'framework rational curve '
                   rf'$\sigma^2(x)=\langle x^2\rangle\,'
                   rf'(\sigma^2(1)-x)/(\langle x^2\rangle+x)$, '
                   rf'$\langle R^2\rangle={R2_mean:.3f}$'
                   if have_latex
                   else f'framework: sigma^2(x) = <x^2>(sigma^2(1)-x)/'
                        f'(<x^2>+x), <R²>={R2_mean:.3f}'))

    ax.axhline(0.0, color='0.7', lw=0.5, ls=':')
    ax.set_xlabel(r'$(1-s)/s \cdot \langle x^2 \rangle$' if have_latex
                  else '(1-s)/s * <x^2>')
    ax.set_ylabel(r'$\sigma_{\mathrm{iso}}^{2}(s)$' if have_latex
                  else 'sigma_iso^2(s)')
    ax.set_title(
        r'Conversion: $\sigma^2_{\mathrm{iso}}(s)$ vs $(1-s)/s\,\langle x^2\rangle$ '
        r'with framework rational curve'
        if have_latex
        else 'Conversion: sigma^2_iso(s) vs (1-s)/s <x^2>')
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.legend(fontsize=7, loc='best')
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)
    return {'R2_mean': R2_mean,
            'R2_per_cell': [float(r) for r in R2_list],
            'n_cells': len(valid)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    have_latex = _style()
    with open(RESULTS_PATH) as f:
        cells = json.load(f)
    print(f'  loaded {len(cells)} cells from {RESULTS_PATH}')

    for c in cells:
        ds_dir = os.path.join(FIG_ROOT, c['dataset'])
        os.makedirs(ds_dir, exist_ok=True)
        stem = f'cell_H{c["H"]}_L{c["L"]}'
        plot_curves(c, os.path.join(ds_dir, f'{stem}_curves.png'),
                    have_latex)
        plot_contours(c, os.path.join(ds_dir, f'{stem}_contours.png'),
                      have_latex)
        print(f'    {c["dataset"]} H={c["H"]} L={c["L"]}: figures saved')

    plot_snr_collapse(cells, os.path.join(FIG_ROOT, 'snr_collapse.png'),
                      have_latex)
    print(f'    snr_collapse.png saved')

    per_cell = fit_conversion(cells, level=0.5)
    global_fit = plot_conversion(
        per_cell, os.path.join(FIG_ROOT, 'conversion_fit.png'), have_latex)
    print(f'    conversion_fit.png saved  global={global_fit}')

    collapse = plot_collapse(
        per_cell, os.path.join(FIG_ROOT, 'collapse.png'), have_latex)
    print(f'    collapse.png saved  {collapse}')

    out_json = os.path.join(FIG_ROOT, 'conversion_fit.json')
    with open(out_json, 'w') as f:
        json.dump({'per_cell': [{'cell': list(r['cell']),
                                 'fit': r['fit'],
                                 'n_contour': len(r['contour'])}
                                for r in per_cell],
                   'global':   global_fit,
                   'collapse': collapse}, f, indent=2)
    print(f'    {out_json} saved')


if __name__ == '__main__':
    main()
