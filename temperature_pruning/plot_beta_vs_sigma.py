"""Plot sigmoid steepness beta vs Gaussian-noise amplitude sigma.

Hypothesis: more weight noise -> a less-sharp pruning transition,
so the sigmoid slope parameter beta from
    A(s) = A_0 + (A_inf - A_0) / (1 + exp(-beta (s - s_0)))
should decrease with sigma.

Reads ``assets/temperature_pruning/<dataset>/results.json`` and writes
``beta_vs_sigma.png`` next to it.

Usage:
    python -m temperature_pruning.plot_beta_vs_sigma                # mnist28
    python -m temperature_pruning.plot_beta_vs_sigma --dataset sklearn
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .analysis import fit_critical_line, group_pc_by_cell


def _gaussian_plus_const(sigma, a, Sigma, b):
    """beta(sigma) = a * exp(-sigma^2 / Sigma^2) + b."""
    return a * np.exp(-sigma ** 2 / (Sigma ** 2)) + b


# scipy.optimize.curve_fit bound used in the sweep -- beta values at this
# value mean the fit saturated and the true steepness is unresolved.
BETA_CAP = 200.0


def _enable_latex():
    try:
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })
        return True
    except Exception:
        return False


def _group_by_cell(rows, min_r2=0.80):
    """Return {(H, L): [(sigma, beta, R2, saturated_bool), ...]}.

    Beta values pinned at the curve_fit upper bound (>= BETA_CAP - 0.5)
    are flagged as saturated. Saturated rows are kept for plotting so the
    low-sigma region is visible, but downstream callers can choose to
    exclude them from fits.
    """
    out = defaultdict(list)
    for r in rows:
        if r.get('sigmoid_R2') is None:
            continue
        if r['sigmoid_R2'] < min_r2:
            continue
        beta = r.get('sigmoid_beta')
        if beta is None:
            continue
        saturated = beta >= BETA_CAP - 0.5
        out[(int(r['H']), int(r['L']))].append((
            float(r['sigma']), float(beta),
            float(r['sigmoid_R2']), bool(saturated)))
    for k in out:
        out[k].sort()
    return dict(out)


def plot_beta_vs_sigma(results_path, output_path,
                       min_r2=0.80, drop_saturated=True):
    with open(results_path) as f:
        rows = json.load(f)

    by_cell = _group_by_cell(rows, min_r2=min_r2,
                             drop_saturated=drop_saturated)
    cells = sorted(by_cell)
    if not cells:
        print(f"  no usable rows in {results_path}")
        return

    # F-regime cutoffs per cell, from the same residual-based detector used
    # for the critical_line plots.  Cells with no truncation get cutoff=
    # max(sigma) in the data.
    pc_by_cell = group_pc_by_cell(rows, min_r2=min_r2)
    fits = fit_critical_line(pc_by_cell)
    cutoffs = {cell: float(f.get('sigma_cutoff', 1.0))
               for cell, f in fits.items()}

    n = len(cells)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             squeeze=False, layout='constrained')

    # Common colour scale by depth L so the panels share a reading.
    Ls = sorted({L for (_H, L) in cells})
    L_colors = {L: plt.cm.plasma(0.15 + 0.7 * i / max(len(Ls) - 1, 1))
                for i, L in enumerate(Ls)}

    for idx, (H, L) in enumerate(cells):
        ax = axes[idx // n_cols][idx % n_cols]
        entries = by_cell[(H, L)]
        sigmas = np.array([e[0] for e in entries])
        betas = np.array([e[1] for e in entries])
        cutoff = cutoffs.get((H, L), float(sigmas.max()))
        sigma_max = float(sigmas.max())
        in_F = sigmas <= cutoff + 1e-9

        # Shade the SG / thermalisation region so it sits below the data.
        if cutoff < sigma_max:
            ax.axvspan(cutoff, sigma_max * 1.05,
                       facecolor='lightgray', alpha=0.35, zorder=0)
            ax.axvline(cutoff, color='dimgray', ls='--', lw=1.0,
                       alpha=0.85, zorder=1)

        # F-regime points (solid)
        ax.scatter(sigmas[in_F], betas[in_F], s=18, color=L_colors[L],
                   edgecolor='black', linewidth=0.4, zorder=4,
                   label=f'$L={L}$ (F regime)')
        # SG / thermalised points (faint, same color but lighter)
        if (~in_F).any():
            ax.scatter(sigmas[~in_F], betas[~in_F], s=14,
                       color=L_colors[L], alpha=0.35, marker='x',
                       zorder=3, label=r'$\sigma > J_0^{\rm eff}$')

        # Gaussian+offset fit beta = a * exp(-sigma^2 / Sigma^2) + b,
        # restricted to the F regime.
        if in_F.sum() >= 4 and (betas[in_F] > 0).all():
            try:
                s_F = sigmas[in_F]
                b_F = betas[in_F]
                p0 = (max(b_F.max() - b_F.min(), 0.1),  # a
                      max(s_F.max() / 2.0, 0.1),         # Sigma
                      float(b_F.min()))                  # b
                bounds = ([0.0, 1e-3, 0.0],
                          [b_F.max() * 5, s_F.max() * 5,
                           b_F.max()])
                popt, _pcov = curve_fit(
                    _gaussian_plus_const, s_F, b_F,
                    p0=p0, bounds=bounds, maxfev=10_000)
                a_fit, Sigma_fit, b_fit = popt
                x_fit = np.linspace(0.0, cutoff, 200)
                ax.plot(x_fit, _gaussian_plus_const(x_fit, *popt),
                        'r--', lw=1.1, alpha=0.9,
                        label=(fr'$\beta = {a_fit:.1f}\,'
                               fr'e^{{-\sigma^2/\Sigma^2}} + {b_fit:.1f}$'
                               '\n'
                               fr'$\Sigma = {Sigma_fit:.2f}$ (F regime)'))
            except Exception:
                pass

        ax.set_title(f'$H={H}$, $L={L}$')
        ax.set_xlabel(r'Temperature $\sigma$')
        ax.set_ylabel(r'Sigmoid steepness $\beta$')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    for j in range(len(cells), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    if drop_saturated:
        cap_note = (rf'  (dropped $\beta \ge {BETA_CAP:.0f}$ saturated fits;'
                    rf' $R^2 \ge {min_r2:.2f}$;'
                    r' Gaussian+offset fit on F regime only)')
    else:
        cap_note = (rf'  ($R^2 \ge {min_r2:.2f}$;'
                    r' Gaussian+offset fit on F regime only)')
    fig.suptitle(r'Sigmoid steepness $\beta(\sigma)$ vs Gaussian-noise '
                 r'amplitude' + '\n'
                 r'(transition becomes rounder as $\sigma$ grows '
                 r'$\Longleftrightarrow$ $\beta$ decreases)' + cap_note,
                 fontsize=11)
    fig.savefig(output_path, dpi=240, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='mnist28',
                   choices=['sklearn_digits', 'mnist28', 'cifar_resnet'])
    p.add_argument('--results-path', default=None,
                   help='Override results.json path.')
    p.add_argument('--output-path', default=None,
                   help='Override output PNG path.')
    p.add_argument('--min-r2', type=float, default=0.80)
    p.add_argument('--keep-saturated', action='store_true',
                   help='Include rows where the sigmoid beta saturated '
                        'against its 20 upper bound.')
    args = p.parse_args()

    base = f'assets/temperature_pruning/{args.dataset}'
    results_path = args.results_path or os.path.join(base, 'results.json')
    output_path = args.output_path or os.path.join(base, 'beta_vs_sigma.png')

    _enable_latex()
    plot_beta_vs_sigma(results_path, output_path,
                       min_r2=args.min_r2,
                       drop_saturated=not args.keep_saturated)


if __name__ == '__main__':
    main()
