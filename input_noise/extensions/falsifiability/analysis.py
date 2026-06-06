#!/usr/bin/env python3
"""Falsifiability re-analysis for the eta = 1 - xi upper-envelope claim.

Implements A1 (prior prediction), A2 (signed mean residual at L = 2),
A3 (baseline-monotone null control). See ``__init__.py`` docstring for
the full specification.

Run::

    .venv/bin/python -m input_noise.extensions.falsifiability.analysis
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from input_noise.cluster_analyze import iso_contour, fit_framework  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_JSON      = 'input_noise/results_cluster_all.json'
OUT_DIR         = 'input_noise/extensions/falsifiability'
ISO_LEVEL       = 0.50
EXCLUDE         = {'cifar_pca'}
NULL_EXPONENTS  = (0.5, 1.0, 2.0)
MID_RANGE       = (0.3, 0.7)
DATASET_LABEL = {
    'mnist28':      'MNIST 28x28',
    'cifar_resnet': 'CIFAR-10 ResNet18',
    'sklearn':      'sklearn digits',
}
DATASET_COLOR = {
    'mnist28':      '#1f77b4',
    'cifar_resnet': '#d62728',
    'sklearn':      '#9467bd',
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


# ---------------------------------------------------------------------------
# A1. Prior fit of sigma^2(1) from the s = 1 column alone
# ---------------------------------------------------------------------------
def fit_sigma2_1_prior(cell, level=ISO_LEVEL):
    """Fit a logistic in sigma^2 to the s=1 column of the joint grid, then
    return sigma2_1_prior = the sigma^2 at which A = ``level``.

    The s=1 column is the *unpruned* network's accuracy as a function of
    input-noise sigma. This estimate of sigma^2(1) uses only that column —
    it never touches the iso-A contour, so the rescaling that follows is
    no longer self-referential.

    Returns ``(sigma2_1_prior, R2, n_used)`` or ``(nan, nan, 0)`` if the
    fit is undefined.
    """
    s_grid     = np.asarray(cell['joint']['s_grid'],     dtype=float)
    sigma_grid = np.asarray(cell['joint']['sigma_grid'], dtype=float)
    A = np.asarray(cell['joint']['mean'], dtype=float)
    # Locate the s = 1 row.
    idx = np.where(np.isclose(s_grid, 1.0))[0]
    if len(idx) == 0:
        return float('nan'), float('nan'), 0
    col = A[idx[0], :]
    # Monotone-decreasing in sigma; transform to x = sigma^2 and fit a
    # 2-parameter logistic anchored at A_unpruned and 1/C (we don't know
    # C exactly here, so let the lower asymptote float).
    x = sigma_grid ** 2
    y = col
    valid = np.isfinite(y) & np.isfinite(x)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return float('nan'), float('nan'), int(len(x))
    A_hi = float(y.max())
    A_lo = float(y.min())
    if A_hi <= level or A_lo >= level:
        # The unpruned curve does not cross ``level``; we cannot extract
        # sigma2_1_prior unambiguously.
        return float('nan'), float('nan'), int(len(x))

    from scipy.optimize import curve_fit

    def f(x, A_inf, A_0, x0, beta):
        z = -beta * (x - x0)
        return A_0 + (A_inf - A_0) / (1.0 + np.exp(np.clip(z, -500, 500)))

    # Initial guess: high asymptote ~ A_hi, low ~ A_lo, mid at the empirical
    # 50% crossing (linear interp), beta from a typical slope.
    order = np.argsort(x)
    xs_, ys_ = x[order], y[order]
    # Bracket the level crossing on the descending curve.
    # ys_ is decreasing in xs_ overall; find first idx with ys_ < level.
    cross_idx = np.searchsorted(-ys_, -level)
    if 0 < cross_idx < len(xs_):
        x0_0 = float(0.5 * (xs_[cross_idx - 1] + xs_[cross_idx]))
    else:
        x0_0 = float(np.median(xs_))
    beta_0 = 1.0 / max(x0_0, 1e-3)  # positive; sigmoid descending in x
    p0 = [A_hi, A_lo, x0_0, beta_0]
    bounds = (
        [0.0, 0.0, 0.0,            1e-4],
        [1.0, 1.0, 10.0 * xs_[-1], 1e3],
    )
    try:
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=30_000)
    except Exception:
        return float('nan'), float('nan'), int(len(x))
    A_inf, A_0, x0, beta = popt
    y_pred = f(x, *popt)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    # Invert: solve f(x*) = level for x*.
    # level = A_0 + (A_inf - A_0) / (1 + exp(-beta (x - x0)))
    # => exp(-beta (x - x0)) = (A_inf - A_0) / (level - A_0) - 1
    rhs = (A_inf - A_0) / (level - A_0) - 1.0
    if rhs <= 0 or not np.isfinite(rhs):
        return float('nan'), float(R2), int(len(x))
    sigma2_1_prior = float(x0 - np.log(rhs) / beta)
    if not np.isfinite(sigma2_1_prior) or sigma2_1_prior <= 0:
        return float('nan'), float(R2), int(len(x))
    return sigma2_1_prior, float(R2), int(len(x))


# ---------------------------------------------------------------------------
# Per-cell records
# ---------------------------------------------------------------------------
def process_cells(cells, level=ISO_LEVEL):
    """Build the per-cell records for A1 + A2 + A3.

    Returns a list of dicts; each dict carries the prior sigma^2(1)_prior,
    the measured sigma^2(1) (re-fit from the contour for reference),
    the contour residuals (predicted vs measured), and the (xi, eta)
    points used by A2.
    """
    out = []
    for c in cells:
        if c['dataset'] in EXCLUDE:
            continue
        contour = iso_contour(c, level=level)  # list of (s, sigma_iso)
        sigma2_1_meas, R2_meas, _ = fit_framework(c, level=level)
        sigma2_1_prior, R2_prior, n_prior = fit_sigma2_1_prior(c, level=level)
        x2 = float(c['x2_mean'])

        rec = {
            'dataset':        c['dataset'],
            'method':         c['method'],
            'H':              int(c['H']),
            'L':              int(c['L']),
            'repeat':         int(c.get('repeat', 0)),
            'x2_mean':        x2,
            'n_contour':      int(len(contour)),
            'sigma2_1_meas':  sigma2_1_meas,
            'R2_meas':        R2_meas,
            'sigma2_1_prior': sigma2_1_prior,
            'R2_prior':       R2_prior,
            'n_prior_pts':    n_prior,
        }

        # A1 prediction errors in sigma^2 space, using the PRIOR estimate.
        if (np.isfinite(sigma2_1_prior) and sigma2_1_prior > 0
                and len(contour) >= 1):
            s_arr  = np.array([p[0] for p in contour])
            sg_arr = np.array([p[1] for p in contour])
            sigma2_meas = sg_arr ** 2
            sigma2_pred = s_arr * sigma2_1_prior - (1.0 - s_arr) * x2
            resid = sigma2_meas - sigma2_pred
            rec['a1_rms_sigma2']  = float(np.sqrt(np.mean(resid ** 2)))
            rec['a1_mean_sigma2'] = float(np.mean(resid))
            # Also the (xi, eta) residuals using the PRIOR rescaling, so
            # that the parameter-free collapse claim is honest.
            xi_prior  = (1.0 - s_arr) * (1.0 + x2 / sigma2_1_prior)
            eta_prior = sigma2_meas / sigma2_1_prior
            line_resid = eta_prior - (1.0 - xi_prior)
            rec['a1_rms_eta_prior']  = float(np.sqrt(np.mean(line_resid ** 2)))
            rec['a1_mean_eta_prior'] = float(np.mean(line_resid))
            rec['xi_prior']  = xi_prior.tolist()
            rec['eta_prior'] = eta_prior.tolist()
        else:
            rec['a1_rms_sigma2']     = float('nan')
            rec['a1_mean_sigma2']    = float('nan')
            rec['a1_rms_eta_prior']  = float('nan')
            rec['a1_mean_eta_prior'] = float('nan')
            rec['xi_prior']  = []
            rec['eta_prior'] = []

        # A2 residuals at L = 2 in measured-rescaling space (the original
        # upper-envelope coordinates).
        if (np.isfinite(sigma2_1_meas) and sigma2_1_meas > 0
                and len(contour) >= 1):
            s_arr  = np.array([p[0] for p in contour])
            sg_arr = np.array([p[1] for p in contour])
            xi  = (1.0 - s_arr) * (1.0 + x2 / sigma2_1_meas)
            eta = (sg_arr ** 2) / sigma2_1_meas
            rec['xi_meas']  = xi.tolist()
            rec['eta_meas'] = eta.tolist()
            rec['mean_signed_resid']     = float(np.mean(eta - (1.0 - xi)))
            mid_mask = (xi >= MID_RANGE[0]) & (xi <= MID_RANGE[1])
            if mid_mask.any():
                rec['mean_signed_resid_mid'] = float(
                    np.mean((eta - (1.0 - xi))[mid_mask]))
                rec['n_mid'] = int(mid_mask.sum())
            else:
                rec['mean_signed_resid_mid'] = float('nan')
                rec['n_mid'] = 0
        else:
            rec['xi_meas']  = []
            rec['eta_meas'] = []
            rec['mean_signed_resid']     = float('nan')
            rec['mean_signed_resid_mid'] = float('nan')
            rec['n_mid'] = 0

        # A3 null-control residuals: sigma^2_null(s) = sigma2_1_meas * s^k
        # rescaled with the same (sigma2_1_meas, x2).
        rec['a3_null'] = {}
        if (np.isfinite(sigma2_1_meas) and sigma2_1_meas > 0
                and len(contour) >= 1):
            s_arr = np.array([p[0] for p in contour])
            for k in NULL_EXPONENTS:
                sigma2_null = sigma2_1_meas * (s_arr ** k)
                xi_null  = (1.0 - s_arr) * (1.0 + x2 / sigma2_1_meas)
                eta_null = sigma2_null / sigma2_1_meas
                resid    = eta_null - (1.0 - xi_null)
                rec['a3_null'][f'k={k}'] = {
                    'rms_to_line':  float(np.sqrt(np.mean(resid ** 2))),
                    'mean_to_line': float(np.mean(resid)),
                    'xi':           xi_null.tolist(),
                    'eta':          eta_null.tolist(),
                }

        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------
def weighted_mean(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return float('nan'), 0
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask])), int(mask.sum())


def boot_ci(values, weights, n_boot=2000, seed=0):
    """Cell-count-weighted bootstrap 95% CI on the weighted mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[mask], w[mask]
    if len(v) < 3:
        return float('nan'), float('nan')
    means = np.empty(n_boot)
    n = len(v)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.sum(v[idx] * w[idx]) / np.sum(w[idx])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_prior_prediction(records, out_path):
    """A1: per-dataset predicted vs measured contour in sigma^2 space."""
    _style()
    datasets = sorted({r['dataset'] for r in records
                       if np.isfinite(r.get('a1_rms_sigma2', float('nan')))})
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        subset = [r for r in records
                  if r['dataset'] == ds
                  and np.isfinite(r.get('sigma2_1_prior', float('nan')))
                  and r['n_contour'] >= 1]
        xs, ys = [], []
        for r in subset:
            for x, y in zip(r['xi_prior'], r['eta_prior']):
                xs.append(x); ys.append(y)
        if xs:
            ax.scatter(xs, ys, s=4, alpha=0.25,
                       color=DATASET_COLOR.get(ds, '#444'),
                       label=f'{len(subset)} cells')
        xx = np.linspace(0.0, 1.3, 200)
        ax.plot(xx, 1.0 - xx, 'k-', lw=1.2, label=r'$\eta = 1 - \xi$')
        ax.axhline(0, color='k', lw=0.4)
        ax.axvline(0, color='k', lw=0.4)
        ax.set_xlim(-0.05, 1.3)
        ax.set_ylim(-0.05, 1.2)
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$ (prior rescaling)')
        ax.set_title(DATASET_LABEL.get(ds, ds))
        ax.legend(loc='upper right', fontsize=8)
    fig.suptitle(r'A1: prior $\sigma^2(1)$ from s=1 column, no contour leakage',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_signed_residual_L2(records, weighted, ci, out_path):
    """A2: per-cell signed mean residual at L=2 + middle-range overlay."""
    _style()
    l2 = [r for r in records if r['L'] == 2
          and np.isfinite(r.get('mean_signed_resid', float('nan')))]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, key, title in [
        (axes[0], 'mean_signed_resid',     r'all $\xi$'),
        (axes[1], 'mean_signed_resid_mid', r'$\xi \in [0.3, 0.7]$'),
    ]:
        vals_by_ds = defaultdict(list)
        for r in l2:
            v = r.get(key, float('nan'))
            if np.isfinite(v):
                vals_by_ds[r['dataset']].append(v)
        # One column per dataset; jitter for visibility.
        pos = 0
        ticks, labels = [], []
        rng = np.random.default_rng(0)
        for ds, vals in vals_by_ds.items():
            xs = pos + 0.12 * rng.standard_normal(len(vals))
            ax.scatter(xs, vals, s=8, alpha=0.4,
                       color=DATASET_COLOR.get(ds, '#444'))
            ax.scatter([pos], [np.mean(vals)], marker='_', s=200,
                       color='k', zorder=5)
            ticks.append(pos)
            labels.append(f'{DATASET_LABEL.get(ds, ds)}\n(n={len(vals)})')
            pos += 1
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(r'$\langle \eta - (1 - \xi) \rangle$ per cell')
        ax.set_title(title)
        ax.set_ylim(-0.4, 0.4)
        if key == 'mean_signed_resid_mid':
            wm, _ = weighted
            lo, hi = ci
            ax.axhline(wm, color='red', lw=1.2,
                       label=f'weighted mean = {wm:+.3f}')
            ax.axhspan(lo, hi, color='red', alpha=0.12,
                       label=f'95% CI = [{lo:+.3f}, {hi:+.3f}]')
            ax.legend(loc='lower right', fontsize=8)
    fig.suptitle('A2: signed mean residual at $L = 2$',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_null_control(records, out_path):
    """A3: framework line + three null monotone surrogates at L = 2."""
    _style()
    l2 = [r for r in records if r['L'] == 2 and r['a3_null']]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, k in zip(axes, NULL_EXPONENTS):
        xs, ys = [], []
        for r in l2:
            key = f'k={k}'
            if key in r['a3_null']:
                xs.extend(r['a3_null'][key]['xi'])
                ys.extend(r['a3_null'][key]['eta'])
        if xs:
            ax.scatter(xs, ys, s=4, alpha=0.25, color='#888',
                       label=fr'null $\sigma^2(s)\propto s^{{{k}}}$')
        # Also overlay measured points for direct comparison.
        xm, ym = [], []
        for r in l2:
            xm.extend(r['xi_meas']); ym.extend(r['eta_meas'])
        ax.scatter(xm, ym, s=2, alpha=0.18, color='#1f77b4',
                   label='measured (L=2)')
        xx = np.linspace(0.0, 1.3, 200)
        ax.plot(xx, 1.0 - xx, 'k-', lw=1.2, label=r'$\eta = 1 - \xi$')
        ax.set_xlim(-0.05, 1.3)
        ax.set_ylim(-0.05, 1.2)
        ax.set_xlabel(r'$\xi$')
        ax.set_title(fr'$k = {k}$')
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(0, color='k', lw=0.4)
        ax.axvline(0, color='k', lw=0.4)
    axes[0].set_ylabel(r'$\eta$')
    fig.suptitle('A3: null monotone surrogates vs framework prediction (L=2)',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def verdict_a1(records):
    """A1: per-cell median RMS between predicted and measured contour, in
    sigma^2 units AND in eta units (parameter-free). Verdict is PARTIAL/
    YES/NO based on the eta-units RMS vs the L=2 cluster-scale RMS of
    0.108 (which is the upper bound for "as good as the self-referential
    fit")."""
    rms_eta = np.array([r['a1_rms_eta_prior'] for r in records
                        if np.isfinite(r.get('a1_rms_eta_prior',
                                             float('nan')))])
    rms_eta_l2 = np.array([r['a1_rms_eta_prior'] for r in records
                            if r['L'] == 2
                            and np.isfinite(r.get('a1_rms_eta_prior',
                                                  float('nan')))])
    rms_sigma2 = np.array([r['a1_rms_sigma2'] for r in records
                            if np.isfinite(r.get('a1_rms_sigma2',
                                                 float('nan')))])
    median_eta = float(np.median(rms_eta)) if len(rms_eta) else float('nan')
    median_eta_l2 = (float(np.median(rms_eta_l2))
                     if len(rms_eta_l2) else float('nan'))
    # Threshold: 1.5x the cluster-scale L=2 self-referential RMS (0.108).
    threshold = 1.5 * 0.108
    if median_eta_l2 <= 0.108 * 1.05:
        verdict = 'YES'
    elif median_eta_l2 <= threshold:
        verdict = 'PARTIAL'
    else:
        verdict = 'NO'
    return {
        'verdict':        verdict,
        'median_rms_eta_all_cells':   median_eta,
        'median_rms_eta_L2':           median_eta_l2,
        'median_rms_sigma2_all_cells': (float(np.median(rms_sigma2))
                                         if len(rms_sigma2)
                                         else float('nan')),
        'n_cells_with_prior_fit':      int(len(rms_eta)),
        'n_cells_L2_with_prior_fit':   int(len(rms_eta_l2)),
        'threshold_used':              threshold,
        'comment': ('verdict: YES if median L=2 RMS(eta) <= 1.05*0.108 '
                    '(self-referential L=2 baseline); '
                    'PARTIAL if <= 1.5*0.108; NO otherwise'),
    }


def verdict_a2(records):
    """A2: cell-count-weighted signed mean residual at L = 2 in middle
    range. Verdict is YES (distinguishable from zero) if the bootstrap
    95% CI excludes zero AND |weighted mean| > 0.02; otherwise NO."""
    l2 = [r for r in records if r['L'] == 2]
    vals = [r['mean_signed_resid_mid'] for r in l2]
    weights = [r['n_mid'] for r in l2]
    wm_mid, n_mid = weighted_mean(vals, weights)
    lo_mid, hi_mid = boot_ci(vals, weights)
    vals_all = [r['mean_signed_resid'] for r in l2]
    weights_all = [r['n_contour'] for r in l2]
    wm_all, n_all = weighted_mean(vals_all, weights_all)
    lo_all, hi_all = boot_ci(vals_all, weights_all)
    ci_excludes_zero = np.isfinite(lo_mid) and (lo_mid > 0 or hi_mid < 0)
    verdict = ('YES' if (ci_excludes_zero and abs(wm_mid) > 0.02)
               else 'NO')
    return {
        'verdict':                       verdict,
        'weighted_mean_signed_mid':      wm_mid,
        'ci_95_mid':                     [lo_mid, hi_mid],
        'n_cells_mid':                   n_mid,
        'weighted_mean_signed_all_xi':   wm_all,
        'ci_95_all_xi':                  [lo_all, hi_all],
        'n_cells_all_xi':                n_all,
        'comment': ('YES means the framework over- or under-predicts '
                    'eta systematically (signed bias) in the geometrically '
                    'unconstrained middle range; NO means the framework '
                    'line passes through the cell-count-weighted middle.'),
    }


def verdict_a3(records, a2_summary):
    """A3: rule out generic-monotone artefact. Verdict is YES if every
    null-k median RMS to the framework line is substantially larger than
    the measured-L=2 median RMS (factor >= 1.5)."""
    l2 = [r for r in records if r['L'] == 2 and r['a3_null']]
    measured_rms = np.array([
        float(np.sqrt(np.mean(
            (np.array(r['eta_meas']) - (1.0 - np.array(r['xi_meas']))) ** 2)))
        for r in l2 if r['xi_meas'] and r['eta_meas']])
    median_measured = (float(np.median(measured_rms))
                       if len(measured_rms) else float('nan'))
    per_k = {}
    pass_count = 0
    for k in NULL_EXPONENTS:
        key = f'k={k}'
        null_rms = np.array([r['a3_null'][key]['rms_to_line']
                              for r in l2 if key in r['a3_null']])
        null_mean = np.array([r['a3_null'][key]['mean_to_line']
                               for r in l2 if key in r['a3_null']])
        med_null = (float(np.median(null_rms))
                    if len(null_rms) else float('nan'))
        ratio = med_null / median_measured if median_measured > 0 else float('nan')
        per_k[key] = {
            'median_rms':  med_null,
            'median_mean': (float(np.median(null_mean))
                             if len(null_mean) else float('nan')),
            'ratio_vs_measured': ratio,
        }
        if np.isfinite(ratio) and ratio >= 1.5:
            pass_count += 1
    # For k = 1.0 (which is *also* monotone but linear-in-s with zero
    # intercept), expect a large residual because the framework's
    # intercept is -(1-s)<x^2>, not zero.
    verdict = 'YES' if pass_count == len(NULL_EXPONENTS) else (
        'PARTIAL' if pass_count >= 1 else 'NO')
    return {
        'verdict':                  verdict,
        'measured_L2_median_rms':   median_measured,
        'per_k':                    per_k,
        'n_cells_L2':               int(len(l2)),
        'comment': ('YES iff every k in {0.5, 1, 2} gives a null '
                    'median RMS to the framework line that is at least '
                    '1.5x the measured L=2 median RMS.'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'  loading {INPUT_JSON}')
    with open(INPUT_JSON) as f:
        cells = json.load(f)
    print(f'  loaded {len(cells)} cells; processing '
          f'(excluding {EXCLUDE})')

    records = process_cells(cells, level=ISO_LEVEL)
    print(f'  built records for {len(records)} cells')
    n_with_prior = sum(1 for r in records
                       if np.isfinite(r['sigma2_1_prior']))
    n_L2 = sum(1 for r in records if r['L'] == 2)
    print(f'    {n_with_prior} cells have a valid prior sigma2_1 fit')
    print(f'    {n_L2} cells are at L = 2')

    summary_a1 = verdict_a1(records)
    summary_a2 = verdict_a2(records)
    summary_a3 = verdict_a3(records, summary_a2)

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_prior_prediction(records, os.path.join(OUT_DIR, 'prior_prediction.png'))
    plot_signed_residual_L2(records,
                             (summary_a2['weighted_mean_signed_mid'],
                              summary_a2['n_cells_mid']),
                             summary_a2['ci_95_mid'],
                             os.path.join(OUT_DIR, 'signed_residual_L2.png'))
    plot_null_control(records, os.path.join(OUT_DIR, 'null_control.png'))

    # Strip the per-cell xi/eta arrays from the JSON to keep it small;
    # keep summary scalars per cell.
    def _strip(r):
        return {k: v for k, v in r.items()
                if k not in ('xi_meas', 'eta_meas', 'xi_prior', 'eta_prior')
                and k != 'a3_null'}

    results = {
        'iso_level':    ISO_LEVEL,
        'exclude':      sorted(EXCLUDE),
        'null_exponents': list(NULL_EXPONENTS),
        'mid_range':    list(MID_RANGE),
        'per_cell':     [_strip(r) for r in records],
        'a1':           summary_a1,
        'a2':           summary_a2,
        'a3':           summary_a3,
    }
    out_json = os.path.join(OUT_DIR, 'results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'  -> {out_json}')
    print()
    print(f'  A1 verdict: {summary_a1["verdict"]}')
    print(f'  A2 verdict: {summary_a2["verdict"]}')
    print(f'  A3 verdict: {summary_a3["verdict"]}')

    # REPORT.md
    report_path = os.path.join(OUT_DIR, 'REPORT.md')
    with open(report_path, 'w') as f:
        f.write(_format_report(records, summary_a1, summary_a2, summary_a3))
    print(f'  -> {report_path}')


def _format_report(records, a1, a2, a3):
    n_total = len(records)
    n_L2 = sum(1 for r in records if r['L'] == 2)

    # Per-dataset breakdown for A1 (median RMS in eta units).
    by_ds_a1 = defaultdict(list)
    for r in records:
        if np.isfinite(r.get('a1_rms_eta_prior', float('nan'))):
            by_ds_a1[r['dataset']].append(r['a1_rms_eta_prior'])

    # Per-dataset A2 weighted mean in middle range.
    by_ds_a2 = defaultdict(lambda: ([], []))
    for r in records:
        if r['L'] == 2 and np.isfinite(r.get('mean_signed_resid_mid',
                                              float('nan'))):
            v, w = by_ds_a2[r['dataset']]
            v.append(r['mean_signed_resid_mid'])
            w.append(r['n_mid'])

    lines = []
    lines.append('# Falsifiability re-analysis of the eta = 1 - xi claim')
    lines.append('')
    lines.append('Source: `input_noise/results_cluster_all.json` '
                 f'({n_total} cells after excluding {sorted(EXCLUDE)}); '
                 f'{n_L2} of those cells are at L = 2.')
    lines.append('')
    lines.append('## Verdicts (one-liners)')
    lines.append('')
    lines.append(f'- **A1 (prior prediction)**: **{a1["verdict"]}** — '
                 f'median per-cell RMS(eta) between predicted and measured '
                 f'contour, using sigma^2(1) extracted *only* from the s=1 '
                 f'column: {a1["median_rms_eta_all_cells"]:.3f} all cells, '
                 f'{a1["median_rms_eta_L2"]:.3f} at L = 2 '
                 f'(self-referential L=2 baseline: 0.108).')
    lines.append(f'- **A2 (signed mean residual at L = 2, middle range)**: '
                 f'**{a2["verdict"]}** — cell-count-weighted '
                 f'<eta - (1 - xi)> in xi in [0.3, 0.7] is '
                 f'{a2["weighted_mean_signed_mid"]:+.3f} '
                 f'(95% CI [{a2["ci_95_mid"][0]:+.3f}, '
                 f'{a2["ci_95_mid"][1]:+.3f}]; '
                 f'{a2["n_cells_mid"]} L=2 cells voting).')
    measured_rms = a3["measured_L2_median_rms"]
    null_rms = a3["per_k"]["k=1.0"]["median_rms"]
    lines.append(f'- **A3 (null-control)**: **{a3["verdict"]}** — '
                 f'measured L=2 median RMS to framework line = '
                 f'{measured_rms:.3f}; null surrogate sigma^2~s gives '
                 f'{null_rms:.3f} '
                 f'(ratio {a3["per_k"]["k=1.0"]["ratio_vs_measured"]:.2f}x).')
    lines.append('')

    lines.append('## A1. Prior-prediction reframe')
    lines.append('')
    lines.append('Per-cell procedure (no contour leakage):')
    lines.append('')
    lines.append('1. Fit a logistic A(sigma^2) to the s = 1 column of the '
                 'joint grid alone.')
    lines.append('2. Invert at A = 0.5 to get sigma^2(1)_prior.')
    lines.append('3. Predict the full contour via sigma^2(s) = '
                 's * sigma^2(1)_prior - (1-s) * <x^2>.')
    lines.append('4. Compare predicted vs measured iso-A contour points.')
    lines.append('')
    lines.append('| dataset | n_cells with prior fit | median RMS(eta) | '
                 'median RMS(sigma^2) |')
    lines.append('|---|---:|---:|---:|')
    by_ds_sigma = defaultdict(list)
    for r in records:
        if np.isfinite(r.get('a1_rms_sigma2', float('nan'))):
            by_ds_sigma[r['dataset']].append(r['a1_rms_sigma2'])
    for ds in sorted(by_ds_a1):
        lines.append(f'| {DATASET_LABEL.get(ds, ds)} | '
                     f'{len(by_ds_a1[ds])} | '
                     f'{np.median(by_ds_a1[ds]):.3f} | '
                     f'{np.median(by_ds_sigma[ds]):.3f} |')
    lines.append('')
    lines.append(f'**Verdict A1**: {a1["verdict"]}. {a1["comment"]}.')
    lines.append('')

    lines.append('## A2. Signed mean residual at L = 2')
    lines.append('')
    lines.append('The framework predicts eta = 1 - xi point-by-point at '
                 'L = 2 (where the linearised SNR derivation is exact). '
                 'A non-zero signed mean residual in the middle range '
                 'xi in [0.3, 0.7] (away from the geometrically-forced '
                 'endpoints) is the cleanest falsifiable shape prediction.')
    lines.append('')
    lines.append('| dataset | L=2 cells | weighted <signed resid> (mid xi) |')
    lines.append('|---|---:|---:|')
    for ds in sorted(by_ds_a2):
        v, w = by_ds_a2[ds]
        wm, _ = weighted_mean(v, w)
        lines.append(f'| {DATASET_LABEL.get(ds, ds)} | '
                     f'{len(v)} | {wm:+.3f} |')
    lines.append('')
    lines.append(f'**Pooled cell-count-weighted signed mean (mid xi)**: '
                 f'{a2["weighted_mean_signed_mid"]:+.3f} '
                 f'(95% CI [{a2["ci_95_mid"][0]:+.3f}, '
                 f'{a2["ci_95_mid"][1]:+.3f}], n_cells = '
                 f'{a2["n_cells_mid"]}).')
    lines.append('')
    lines.append(f'**All-xi (reference)**: '
                 f'{a2["weighted_mean_signed_all_xi"]:+.3f} '
                 f'(95% CI [{a2["ci_95_all_xi"][0]:+.3f}, '
                 f'{a2["ci_95_all_xi"][1]:+.3f}], n_cells = '
                 f'{a2["n_cells_all_xi"]}).')
    lines.append('')
    lines.append(f'**Verdict A2**: {a2["verdict"]}. {a2["comment"]}')
    lines.append('')

    lines.append('## A3. Baseline-monotone null control')
    lines.append('')
    lines.append('Per L=2 cell, replace the measured iso-A contour with the '
                 'monotone surrogate sigma^2_null(s) = sigma^2(1) * s^k for '
                 'k in {0.5, 1, 2}. Each surrogate satisfies the boundary '
                 'conditions sigma^2_null(0) = 0 and sigma^2_null(1) = '
                 'sigma^2(1) but is *not* the framework prediction. Apply '
                 'the same (xi, eta) rescaling and compare RMS to '
                 'eta = 1 - xi.')
    lines.append('')
    lines.append('| k | median RMS to line | median signed mean | '
                 'ratio vs measured |')
    lines.append('|---:|---:|---:|---:|')
    for k in NULL_EXPONENTS:
        key = f'k={k}'
        s = a3['per_k'][key]
        lines.append(f'| {k} | {s["median_rms"]:.3f} | '
                     f'{s["median_mean"]:+.3f} | '
                     f'{s["ratio_vs_measured"]:.2f}x |')
    lines.append('')
    lines.append(f'**Measured L=2 median RMS (reference)**: '
                 f'{a3["measured_L2_median_rms"]:.3f} '
                 f'(over {a3["n_cells_L2"]} L=2 cells).')
    lines.append('')
    lines.append(f'**Verdict A3**: {a3["verdict"]}. {a3["comment"]}')
    lines.append('')

    lines.append('## Manuscript hook')
    lines.append('')
    lines.append('All three verdicts feed §V.B of the Phase 4 revision: '
                 'A1 turns the upper-envelope claim into a *prior* '
                 'shape prediction; A2 quantifies whether the prediction '
                 'is point-by-point or only upper-envelope; A3 rules out '
                 '(or fails to rule out) the generic-monotone artefact '
                 'that R1.C1 and R4.MAJOR-1 flagged.')
    lines.append('')
    lines.append('Figures: `prior_prediction.png` (A1), '
                 '`signed_residual_L2.png` (A2), `null_control.png` (A3). '
                 'Machine-readable per-cell numbers in `results.json`.')
    lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
