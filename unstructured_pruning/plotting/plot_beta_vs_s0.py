#!/usr/bin/env python3
r"""Plot sigmoid steepness :math:`\beta` as a function of the critical
density :math:`s_0` for every (dataset, pruning-method) combination, and
compare against the theoretical prediction

.. math::
    \beta(s_0; c) \;=\; \frac{2c}{\sqrt{2\pi}}\;
        \frac{e^{-z_{1/2}^{2}/2}}{s_0^{1/2}\,(1 - s_0)^{3/2}},
    \qquad z_{1/2}\approx 0.6745,

derived in ``docs/pruning_sigmoid_derivation.md`` by matching the slope of
the logistic ansatz with that of the exact Gaussian-cumulant result at
:math:`s = s_0`. The constant :math:`c = \mathcal{J}_0/\sqrt{\mathcal{V}}`
is the effective signal-to-noise of the architecture; the shape
:math:`1/[s_0^{1/2}(1-s_0)^{3/2}]` is parameter-free. We fit a single
:math:`c` per (dataset, method) panel by least-squares on
:math:`\log\beta`, restricted to resolved (non-saturated) cells, and
overlay the resulting curve.

Each ``assets/unstructured_pruning/<dataset>_<method>``
directory holds one ``scaling_results.json`` with one row per
``(H, L, repeat)`` cell, including the per-cell sigmoid fit parameters
``sigmoid_s_0`` and ``sigmoid_beta``. This script scatters
:math:`\beta` against :math:`s_0` for every cell, coloured by depth
:math:`L` and marker-sized by width :math:`H`, so the dependence (if any)
across the architecture grid is visible.

The script writes two flavours of figure:

1. ``beta_vs_s0.png`` inside every per-(dataset, method) directory --
   one cell, full detail.
2. ``beta_vs_s0_grid.png`` at the top level -- a 4 x 3 grid covering
   the whole dataset x method matrix at a glance.

Beta is fitted with a curve-fit upper bound of 20, so points with
:math:`\beta \gtrsim 19.5` are flagged as saturated (the fit hit the
bound, the true steepness is unresolved) and drawn as open markers.

Usage
-----
    python -m unstructured_pruning.plotting.plot_beta_vs_s0
    python -m unstructured_pruning.plotting.plot_beta_vs_s0 --base FIGS_DIR
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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D


# Beta values fitted at the upper bound of the optimiser. Anything at or above
# this cutoff is treated as a saturated / unresolved fit.
BETA_CAP = 200.0
BETA_SAT_THRESHOLD = BETA_CAP - 0.5

# Half-quantile of the standard normal: z = sqrt(2) * erf^{-1}(1/2) ~ 0.6745.
# This is the value f(s_0) takes at the midpoint of the excess accuracy g(s),
# i.e. the constant appearing in the derivation in docs/pruning_sigmoid_derivation.md.
Z_HALF = 0.6744897501960817

# Prefactor that survives when (2/sqrt(2 pi)) e^{-z^2/2} is absorbed into the
# formula. With c = J_0 / sqrt(V) a free architecture-dependent constant,
#   beta_theory(s_0; c) = c * K0 / [ s_0^{1/2} (1 - s_0)^{3/2} ]
K0 = (2.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * Z_HALF ** 2)


def beta_theory(s0, c):
    """Theoretical sigmoid steepness as a function of the critical density.

    Derived in ``docs/pruning_sigmoid_derivation.md`` by matching the
    derivative of the logistic ansatz with that of the exact Gaussian-cumulant
    result at :math:`s = s_0`:

    .. math::
        \\beta = \\frac{2c}{\\sqrt{2\\pi}}\\;
                 \\frac{e^{-z_{1/2}^{2}/2}}
                       {s_{0}^{1/2}\\,(1 - s_{0})^{3/2}},
        \\qquad z_{1/2}\\approx 0.6745,

    where :math:`c = \\mathcal{J}_0/\\sqrt{\\mathcal{V}}` is the effective
    signal-to-noise ratio of the architecture / dataset.
    """
    s = np.asarray(s0, dtype=float)
    s = np.clip(s, 1e-9, 1.0 - 1e-9)
    return c * K0 / (np.sqrt(s) * (1.0 - s) ** 1.5)


# Parameter-free prediction: in the bare theory ``c`` is not free but is
# determined by s_0 itself through eq. (F33) / (D19) of the paper:
#     c * sqrt(s_0 / (1 - s_0)) = z_{1/2}    ==>    c = z * sqrt((1 - s_0)/s_0)
# Substituting back into beta_theory collapses the s_0^{1/2}, (1-s_0)^{3/2}
# factors to a clean 1 / [s_0 (1 - s_0)] shape with NO free parameters:
#     beta_pf(s_0) = K_PF / [ s_0 (1 - s_0) ],   K_PF = 2 z e^{-z^2/2} / sqrt(2 pi).
K_PF = 2.0 * Z_HALF * np.exp(-0.5 * Z_HALF ** 2) / np.sqrt(2.0 * np.pi)


def beta_theory_parameter_free(s0):
    """Bare-theory prediction with c eliminated via :math:`c = z\\sqrt{(1-s_0)/s_0}`.

    No free parameters. Returns :math:`\\beta = K_{\\mathrm{PF}}/[s_0(1-s_0)]`
    with :math:`K_{\\mathrm{PF}} = 2 z_{1/2} e^{-z_{1/2}^{2}/2} / \\sqrt{2\\pi}
    \\approx 0.4286`.
    """
    s = np.asarray(s0, dtype=float)
    s = np.clip(s, 1e-9, 1.0 - 1e-9)
    return K_PF / (s * (1.0 - s))


def parameter_free_loglog_r2(s0, beta):
    """R^2 in log-beta of the parameter-free prediction against the data.

    No fit -- this is a direct comparison with a fixed curve. Restricted to
    the same resolved cells the c-fit uses.
    """
    s = np.asarray(s0, dtype=float)
    b = np.asarray(beta, dtype=float)
    mask = (s > 0.0) & (s < 1.0) & (b > 0.0)
    if mask.sum() < 3:
        return None, int(mask.sum())
    log_b = np.log(b[mask])
    log_pf = np.log(beta_theory_parameter_free(s[mask]))
    ss_res = float(np.sum((log_b - log_pf) ** 2))
    ss_tot = float(np.sum((log_b - log_b.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return r2, int(mask.sum())


def fit_c_loglog(s0, beta):
    """Single-parameter LS fit of ``c`` in log-beta space.

    Returns ``(c_hat, R2_loglog, N)`` where R^2 is computed on the natural
    logarithm of beta -- the right scale because the model is multiplicative
    in c and we plot on log-log axes. The cells passed in should already
    exclude saturated fits.
    """
    s = np.asarray(s0, dtype=float)
    b = np.asarray(beta, dtype=float)
    mask = (s > 0.0) & (s < 1.0) & (b > 0.0)
    if mask.sum() < 3:
        return None, None, int(mask.sum())
    shape = K0 / (np.sqrt(s[mask]) * (1.0 - s[mask]) ** 1.5)
    log_b = np.log(b[mask])
    log_shape = np.log(shape)
    # log b = log c + log shape  =>  log c = mean(log b - log shape)
    log_c = float(np.mean(log_b - log_shape))
    c_hat = float(np.exp(log_c))
    resid = log_b - (log_c + log_shape)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((log_b - log_b.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return c_hat, r2, int(mask.sum())

DATASETS_ORDER = ['sklearn', 'mnist28', 'cifar_pca', 'cifar_resnet']
METHODS_ORDER = ['random', 'magnitude', 'wanda']

DATASET_LABEL = {
    'sklearn':      r'\textsc{sklearn} digits',
    'mnist28':      r'\textsc{mnist}-28',
    'cifar_pca':    r'\textsc{cifar}-10 + PCA',
    'cifar_resnet': r'\textsc{cifar}-10 + ResNet',
}
METHOD_LABEL = {
    'random':    r'random',
    'magnitude': r'magnitude',
    'wanda':     r'\textsc{wanda}',
}

# Fallback labels when LaTeX is unavailable.
DATASET_LABEL_PLAIN = {
    'sklearn':      'sklearn digits',
    'mnist28':      'MNIST-28',
    'cifar_pca':    'CIFAR-10 + PCA',
    'cifar_resnet': 'CIFAR-10 + ResNet',
}
METHOD_LABEL_PLAIN = {
    'random':    'random',
    'magnitude': 'magnitude',
    'wanda':     'WANDA',
}


def _configure_style():
    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex':       have_latex,
        'font.family':       'serif',
        'font.serif':        ['Computer Modern Roman', 'CMU Serif',
                              'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':  'cm',
        'axes.labelsize':    11,
        'axes.titlesize':    12,
        'axes.linewidth':    0.7,
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   8,
        'legend.frameon':    True,
        'legend.framealpha': 0.92,
        'legend.edgecolor':  '0.6',
        'figure.titlesize':  14,
        'figure.dpi':        120,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.08,
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}'
        )
    return have_latex


def _parse_token(token):
    """Split a ``<dataset>_<method>`` directory token, accounting for
    multi-word dataset names like ``cifar_pca`` and ``cifar_resnet``."""
    for ds in DATASETS_ORDER:
        for m in METHODS_ORDER:
            if token == f'{ds}_{m}':
                return ds, m
    # Fallback: last underscore separates dataset and method.
    if '_' in token:
        ds, m = token.rsplit('_', 1)
        return ds, m
    return token, ''


def _extract(rows, min_r2=0.80):
    """Pull (s_0, beta, H, L, R^2, saturated) per usable row."""
    out = []
    for r in rows:
        r2 = r.get('sigmoid_R2')
        s0 = r.get('sigmoid_s_0')
        beta = r.get('sigmoid_beta')
        if r2 is None or s0 is None or beta is None:
            continue
        if r2 < min_r2:
            continue
        if not (0.0 < float(s0) <= 1.0):
            continue
        if float(beta) <= 0.0:
            continue
        out.append({
            'H':         int(r['H']),
            'L':         int(r['L']),
            's0':        float(s0),
            'beta':      float(beta),
            'R2':        float(r2),
            'saturated': float(beta) >= BETA_SAT_THRESHOLD,
        })
    return out


def _size_for_H(H, H_lo, H_hi, s_lo=22, s_hi=110):
    """Linear marker-size interpolation in log H so wide and narrow
    networks are visually distinguishable."""
    if H_hi <= H_lo:
        return np.full_like(H, (s_lo + s_hi) / 2, dtype=float)
    t = (np.log(H) - np.log(H_lo)) / (np.log(H_hi) - np.log(H_lo))
    return s_lo + (s_hi - s_lo) * t


def _theory_label(c_hat, r2, r2_pf, n_fit, have_latex):
    if c_hat is None:
        return None
    pf_str = (f'{r2_pf:+.2f}' if r2_pf is not None else 'n/a')
    if have_latex:
        return (rf'$\beta_{{\mathrm{{fit}}}} = \dfrac{{2c\,e^{{-z_{{1/2}}^{{2}}/2}}}}'
                rf'{{\sqrt{{2\pi}}\;s_{{0}}^{{1/2}}(1-s_{{0}})^{{3/2}}}}$'
                '\n'
                rf'$\beta_{{\mathrm{{PF}}}} = \dfrac{{K_{{\mathrm{{PF}}}}}}{{s_{{0}}(1-s_{{0}})}},\;'
                rf'K_{{\mathrm{{PF}}}}\!\approx\!{K_PF:.3f}$'
                '\n'
                rf'$c_{{\mathrm{{fit}}}} = {c_hat:.2f}$'
                '\n'
                rf'$R^{{2}}_{{\log\beta}}:\;\mathrm{{fit}}={r2:+.2f},\;\mathrm{{PF}}={pf_str}$'
                '\n'
                rf'$N_{{\mathrm{{fit}}}} = {n_fit}$')
    return (f'beta_fit = 2c e^(-z^2/2) / [sqrt(2 pi) s0^0.5 (1-s0)^1.5]\n'
            f'beta_PF  = {K_PF:.3f} / [s0 (1-s0)]   (no free params)\n'
            f'c_fit = {c_hat:.2f}\n'
            f'R^2 (log beta): fit={r2:+.2f}, PF={pf_str}\n'
            f'N_fit = {n_fit}')


def _plot_one(ax, cells, *, have_latex, l_norm, cmap, H_range,
              show_xlabel=True, show_ylabel=True, title=None):
    """Render one (dataset, method) scatter on ``ax``.

    Returns the scatter handle (or None) for colourbar reuse.
    """
    if not cells:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='0.4')
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, pad=4)
        return None

    s0 = np.array([c['s0']        for c in cells])
    beta = np.array([c['beta']    for c in cells])
    L = np.array([c['L']          for c in cells])
    H = np.array([c['H']          for c in cells])
    sat = np.array([c['saturated'] for c in cells])

    sizes = _size_for_H(H, *H_range)

    sc = None
    # Resolved fits: filled markers.
    if (~sat).any():
        sc = ax.scatter(s0[~sat], beta[~sat],
                        c=L[~sat], cmap=cmap, norm=l_norm,
                        s=sizes[~sat],
                        edgecolor='black', linewidth=0.4,
                        alpha=0.85, zorder=3)
    # Saturated fits: hollow markers in matching colour so the value bound
    # is visible without claiming a resolved beta.
    if sat.any():
        face = cmap(l_norm(L[sat]))
        face[:, -1] = 0.0  # fully transparent fill
        ax.scatter(s0[sat], beta[sat],
                   facecolors=face,
                   edgecolors=cmap(l_norm(L[sat])),
                   linewidth=0.9,
                   s=sizes[sat],
                   marker='o',
                   alpha=0.9, zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linewidth=0.4, alpha=0.4)

    # Saturation guide line.
    ax.axhline(BETA_CAP, color='0.35', linestyle=':', linewidth=0.8,
               alpha=0.85, zorder=1)

    # Theoretical beta(s_0): fit one c on resolved cells, and also overlay
    # the parameter-free prediction obtained by eliminating c via the
    # critical-point relation c = z sqrt((1 - s_0)/s_0).
    c_hat, r2, n_fit = fit_c_loglog(s0[~sat], beta[~sat])
    r2_pf, _ = parameter_free_loglog_r2(s0[~sat], beta[~sat])
    s_lo = max(float(s0.min()) * 0.7, 1e-3)
    s_hi = min(float(s0.max()) * 1.4, 0.999)
    s_grid = np.geomspace(s_lo, s_hi, 256)
    if c_hat is not None:
        ax.plot(s_grid, beta_theory(s_grid, c_hat),
                color='crimson', linestyle='--', linewidth=1.4,
                alpha=0.95, zorder=4,
                label=(rf'fit: $c={c_hat:.2f}$' if have_latex
                       else f'fit: c={c_hat:.2f}'))
    ax.plot(s_grid, beta_theory_parameter_free(s_grid),
            color='navy', linestyle='-', linewidth=1.4,
            alpha=0.85, zorder=4,
            label=(r'parameter-free $\beta = K_{\mathrm{PF}}/[s_{0}(1-s_{0})]$'
                   if have_latex
                   else 'parameter-free (no fit)'))

    if show_xlabel:
        ax.set_xlabel(r'Critical density $s_{0}$' if have_latex
                      else r'Critical density s_0')
    if show_ylabel:
        ax.set_ylabel(r'Sigmoid steepness $\beta$' if have_latex
                      else r'Sigmoid steepness beta')
    if title:
        ax.set_title(title, pad=4)

    txt = _theory_label(c_hat, r2, r2_pf, n_fit, have_latex)
    if txt is not None:
        ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                ha='left', va='top', fontsize=8.0,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='0.55', lw=0.5, alpha=0.92))

    return sc


def _l_norm(all_L):
    L_lo = max(1, int(min(all_L)))
    L_hi = max(L_lo + 1, int(max(all_L)))
    return Normalize(vmin=L_lo, vmax=L_hi), L_lo, L_hi


def _H_range(all_H):
    return float(min(all_H)), float(max(all_H))


def _size_legend(H_lo, H_hi, have_latex):
    """Build a 3-point legend showing the H -> marker-size mapping."""
    if H_hi <= H_lo:
        Hs = [H_lo]
    else:
        Hs = sorted({
            int(round(H_lo)),
            int(round(np.exp(0.5 * (np.log(H_lo) + np.log(H_hi))))),
            int(round(H_hi)),
        })
    handles = []
    for h in Hs:
        s = float(_size_for_H(np.array([h]), H_lo, H_hi)[0])
        label = (rf'$H = {h}$' if have_latex else f'H = {h}')
        handles.append(Line2D([0], [0], marker='o', linestyle='',
                              markerfacecolor='0.85',
                              markeredgecolor='black',
                              markersize=np.sqrt(s),
                              label=label))
    handles.append(Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor='none',
                          markeredgecolor='0.3',
                          markeredgewidth=0.9,
                          markersize=8,
                          label=(r'$\beta$ at fit bound (saturated)'
                                 if have_latex
                                 else 'beta at fit bound (saturated)')))
    return handles


def render_per_cell(cells, output_path, *, dataset, method, have_latex):
    """One-panel figure per (dataset, method) directory."""
    if not cells:
        return None
    all_L = [c['L'] for c in cells]
    all_H = [c['H'] for c in cells]
    norm, _, _ = _l_norm(all_L)
    H_lo, H_hi = _H_range(all_H)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor='white')
    fig.subplots_adjust(left=0.11, right=0.82, top=0.91, bottom=0.11)

    ds_label = (DATASET_LABEL if have_latex else DATASET_LABEL_PLAIN).get(
        dataset, dataset)
    m_label = (METHOD_LABEL if have_latex else METHOD_LABEL_PLAIN).get(
        method, method)
    title = (rf'{ds_label} -- {m_label}\quad $\beta(s_{{0}})$'
             if have_latex else f'{ds_label} -- {m_label}: beta(s_0)')

    sc = _plot_one(ax, cells, have_latex=have_latex,
                   l_norm=norm, cmap=cmap, H_range=(H_lo, H_hi),
                   title=title)

    cax = fig.add_axes([0.84, 0.11, 0.025, 0.80])
    cb = fig.colorbar(sc, cax=cax) if sc is not None else None
    if cb is not None:
        cb.set_label(r'Depth $L$' if have_latex else 'Depth L')

    handles = _size_legend(H_lo, H_hi, have_latex)
    ax.legend(handles=handles, loc='lower left',
              bbox_to_anchor=(0.0, 0.0), borderaxespad=0.3,
              handletextpad=0.6)

    fig.savefig(output_path, facecolor='white')
    plt.close(fig)
    return output_path


def render_grid(per_cell_data, output_path, *, have_latex):
    """A single 4 x 3 (dataset x method) overview figure."""
    # Shared colour and size scales so panels are comparable.
    all_L = [c['L'] for cells in per_cell_data.values() for c in cells]
    all_H = [c['H'] for cells in per_cell_data.values() for c in cells]
    if not all_L:
        return None
    norm, _, _ = _l_norm(all_L)
    H_lo, H_hi = _H_range(all_H)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(15.5, 16.0), facecolor='white')
    gs = fig.add_gridspec(len(DATASETS_ORDER), len(METHODS_ORDER),
                          left=0.06, right=0.88, top=0.94, bottom=0.07,
                          wspace=0.22, hspace=0.32)
    sc_for_cb = None

    for i, ds in enumerate(DATASETS_ORDER):
        for j, m in enumerate(METHODS_ORDER):
            ax = fig.add_subplot(gs[i, j])
            cells = per_cell_data.get((ds, m), [])
            ds_label = (DATASET_LABEL if have_latex
                        else DATASET_LABEL_PLAIN).get(ds, ds)
            m_label = (METHOD_LABEL if have_latex
                       else METHOD_LABEL_PLAIN).get(m, m)
            title = (rf'{ds_label} -- {m_label}'
                     if have_latex else f'{ds_label} -- {m_label}')
            sc = _plot_one(ax, cells, have_latex=have_latex,
                           l_norm=norm, cmap=cmap,
                           H_range=(H_lo, H_hi),
                           show_xlabel=(i == len(DATASETS_ORDER) - 1),
                           show_ylabel=(j == 0),
                           title=title)
            if sc is not None and sc_for_cb is None:
                sc_for_cb = sc

    fig.suptitle((r'Sigmoid steepness $\beta$ vs.\ critical density '
                  r'$s_{0}$ across the $(H, L)$ architecture grid'
                  if have_latex
                  else 'Sigmoid steepness beta vs. critical density s_0 '
                       'across the (H, L) architecture grid'),
                 y=0.985)

    if sc_for_cb is not None:
        cax = fig.add_axes([0.90, 0.20, 0.018, 0.60])
        cb = fig.colorbar(sc_for_cb, cax=cax)
        cb.set_label(r'Depth $L$' if have_latex else 'Depth L')

    handles = _size_legend(H_lo, H_hi, have_latex)
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.005),
               ncol=len(handles), frameon=True, edgecolor='0.6')

    fig.savefig(output_path, facecolor='white')
    plt.close(fig)
    return output_path


def main(base='assets/unstructured_pruning', min_r2=0.80):
    have_latex = _configure_style()
    print(f'  Rendering with text.usetex = {have_latex}')

    per_cell_data = {}
    saved, skipped = [], []
    for d in sorted(p for m in ('magnitude', 'random', 'wanda')
                    for p in glob.glob(os.path.join(base, f'*_{m}'))):
        results_p = os.path.join(d, 'scaling_results.json')
        if not os.path.isfile(results_p):
            skipped.append((d, 'no scaling_results.json'))
            continue
        token = os.path.basename(d)
        dataset, method = _parse_token(token)
        with open(results_p) as f:
            rows = json.load(f)
        cells = _extract(rows, min_r2=min_r2)
        if not cells:
            skipped.append((d, 'no rows pass min_r2 filter'))
            continue
        per_cell_data[(dataset, method)] = cells

        out = os.path.join(d, 'beta_vs_s0.png')
        p = render_per_cell(cells, out,
                            dataset=dataset, method=method,
                            have_latex=have_latex)
        if p:
            saved.append(p)
            print(f'  Saved: {p}')

    if per_cell_data:
        grid_out = os.path.join(base, 'beta_vs_s0_grid.png')
        p = render_grid(per_cell_data, grid_out, have_latex=have_latex)
        if p:
            saved.append(p)
            print(f'  Saved: {p}')

    print(f'\nDone. {len(saved)} plots saved, {len(skipped)} skipped.')
    for d, why in skipped:
        print(f'  skipped: {os.path.basename(d)}  ({why})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='assets/unstructured_pruning',
                    help='directory containing <dataset>_<method> subdirs')
    ap.add_argument('--min-r2', type=float, default=0.80,
                    help='minimum sigmoid R^2 to keep a cell')
    args = ap.parse_args()
    main(base=args.base, min_r2=args.min_r2)
