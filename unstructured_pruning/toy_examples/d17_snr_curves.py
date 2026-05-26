#!/usr/bin/env python3
r"""Plot the Appendix D closed form

    A(s) = Phi( SNR * sqrt(s / (1 - s)) ),    SNR = J_0 / sqrt(V)

as a family of curves, one per SNR value, with hue carrying SNR.

The same shape appears in every dataset of this folder; here we just
isolate it from the trained-weight machinery and look at how the
sigmoid steepens and shifts as the single parameter SNR varies.

Run::

    .venv/bin/python -m unstructured_pruning.toy_examples.d17_snr_curves
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snr-min',   type=float, default=0.5)
    ap.add_argument('--snr-max',   type=float, default=60.0)
    ap.add_argument('--n-snr',     type=int,   default=30)
    ap.add_argument('--n-s',       type=int,   default=400)
    ap.add_argument('--s-min',     type=float, default=1e-4)
    ap.add_argument('--s-max',     type=float, default=0.999)
    ap.add_argument('--output',    default=os.path.join(
        'unstructured_pruning', 'toy_examples', 'figures', 'd17',
        'd17_snr_curves.png'))
    args = ap.parse_args()

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

    s = np.geomspace(args.s_min, args.s_max, args.n_s)
    u = np.sqrt(s / (1.0 - s))
    snrs = np.geomspace(args.snr_min, args.snr_max, args.n_snr)
    norm_obj = matplotlib.colors.LogNorm(vmin=snrs.min(), vmax=snrs.max())
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor='white')
    for snr in snrs:
        A = norm.cdf(snr * u)
        ax.plot(s, A, color=cmap(norm_obj(snr)), lw=1.6)
    ax.set_xscale('log')
    ax.set_xlim(args.s_min, 1.0)
    ax.set_ylim(0.48, 1.02)
    ax.set_xlabel(r'$s$' if have_latex else 's')
    ax.set_ylabel(r'$A(s)$' if have_latex else 'A(s)')
    ax.set_title(r'Accuracy curve $A(s)$ for single hidden layer, binary classification')
        # ((r'$A(s)=\Phi\!\left(\mathrm{SNR}\,\sqrt{s/(1-s)}\right)$, '
        #  r'$\mathrm{SNR}=\mathcal{J}_0/\sqrt{\mathcal{V}}$'
        #  if have_latex
        #  else 'A(s) = Phi(SNR * sqrt(s/(1-s)))'))
    ax.axhline(0.5, color='0.5', lw=0.5, linestyle=':')
    ax.grid(True, which='both', alpha=0.3, lw=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\mathrm{SNR}=\mathcal{J}_0/\sqrt{\mathcal{V}}$'
                   if have_latex else 'SNR = J_0 / sqrt(V)')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, facecolor='white')
    plt.close(fig)
    print(f'Saved figure: {args.output}')


if __name__ == '__main__':
    main()
