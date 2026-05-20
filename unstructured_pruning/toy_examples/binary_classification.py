#!/usr/bin/env python3
r"""Appendix D toy: scalar binary classification under random pruning.

Setup (exactly the network of paper Appendix D, eqs D1--D6)
-----------------------------------------------------------
* scalar input :math:`x \sim \mathcal{N}(0, 1)`
* label :math:`y = \mathrm{sign}(x)`
* one hidden layer, **linear** (no activation):

  .. math:: \hat y \;=\; \sum_{a=1}^{H} W^{(2)}_a\,W^{(1)}_a\,x
                   \;\equiv\; x\,\mathcal{J}(\mathbf{m}),
           \quad \mathcal{J}(\mathbf{m}) = \sum_a m_a w_a,
           \quad w_a = W^{(2)}_a W^{(1)}_a.

* random Bernoulli mask :math:`m_a \sim \text{Bern}(s)` is applied to the
  hidden layer only.

Theoretical prediction (eq. D17)
--------------------------------
.. math::
    A(s) \;=\; \Phi\!\left(\frac{\mathcal{J}_0}{\sqrt{\mathcal{V}}}\,
                 \sqrt{\frac{s}{1-s}}\right),
    \qquad \mathcal{J}_0 = \sum_a w_a,\;\;
            \mathcal{V}   = \sum_a w_a^{2},
    \qquad c \equiv \mathcal{J}_0 / \sqrt{\mathcal{V}}.

Two diagnostics
---------------
1. ``A(s)`` panel: empirical mean +/- std over mask seeds, overlaid with
   the parameter-free :math:`\Phi(c\sqrt{s/(1-s)})` (c computed directly
   from the trained weights, not fitted).
2. **Probit linearization**: plot :math:`\Phi^{-1}(A_{\mathrm{emp}}(s))`
   against :math:`\sqrt{s/(1-s)}`. Under D17 this is a straight line
   through the origin with slope c. Any departure from a straight line is
   direct visual evidence that D17 fails.

A grid is run across several hidden widths H so we can also see whether
the predicted scaling :math:`c \propto \sqrt{H}` (uniform-weight limit) is
borne out.

Usage
-----
    .venv/bin/python -m unstructured_pruning.toy_examples.binary_classification
    .venv/bin/python -m unstructured_pruning.toy_examples.binary_classification \
        --hs 4 16 64 256 --n-seeds 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_HS = (4, 16, 64, 256, 512, 4096, 8192, 16384)


# ---------------------------------------------------------------------------
# Model & training
# ---------------------------------------------------------------------------
class LinearHiddenNet(nn.Module):
    """:math:`\\hat y = \\sum_a W^{(2)}_a W^{(1)}_a x` -- exactly eq. (D2)."""

    def __init__(self, H: int):
        super().__init__()
        self.H = int(H)
        # Initialise with non-zero mean so the trained solution lands at
        # J_0 > 0 quickly. Variance shrinks with H to keep |y| ~ O(1).
        scale = 1.0 / np.sqrt(self.H)
        self.W1 = nn.Parameter(torch.randn(self.H) * scale)
        self.W2 = nn.Parameter(torch.randn(self.H) * scale)

    def forward(self, x, mask=None):
        # x: (B,) -- broadcast against (H,) hidden weights.
        if mask is None:
            return (self.W1 * self.W2).sum() * x
        # Pre-mask path weights so the per-neuron multiplication is explicit.
        w = self.W1 * self.W2 * mask
        return w.sum() * x

    @torch.no_grad()
    def path_weights(self) -> np.ndarray:
        return (self.W1 * self.W2).detach().cpu().numpy()


def train(model: LinearHiddenNet, *, n_train: int, n_epochs: int,
          lr: float, batch_size: int, seed: int) -> None:
    """Plain SGD on MSE. The optimum for x~N(0,1), y=sign(x) is the Wiener
    filter J = E[xy]/E[x^2] = sqrt(2/pi); any J > 0 already classifies
    perfectly, so training is just to give us a definite weight pattern.
    """
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal(n_train).astype(np.float32))
    y = torch.sign(x)
    y[y == 0] = 1.0
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        idx = torch.randperm(n_train)[:batch_size]
        pred = model(x[idx])
        loss = ((pred - y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# Pruning sweep
# ---------------------------------------------------------------------------
def evaluate_mask(model: LinearHiddenNet,
                  x: torch.Tensor, y: torch.Tensor,
                  mask: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(x, mask)
        # sign of zero -> +1 (consistent with how labels were sanitised)
        psign = torch.sign(pred)
        psign[psign == 0] = 1.0
        return float((psign == y).float().mean())


def sweep(model: LinearHiddenNet, *, densities: np.ndarray,
          n_seeds: int, n_test: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 1)
    x = torch.from_numpy(rng.standard_normal(n_test).astype(np.float32))
    y = torch.sign(x)
    y[y == 0] = 1.0

    H = model.H
    A_mean = np.zeros_like(densities)
    A_std = np.zeros_like(densities)
    for i, s in enumerate(densities):
        accs = np.empty(n_seeds, dtype=float)
        for k in range(n_seeds):
            m = torch.from_numpy(
                (rng.random(H) < s).astype(np.float32))
            accs[k] = evaluate_mask(model, x, y, m)
        A_mean[i] = accs.mean()
        A_std[i] = accs.std()
    return {'densities': densities.tolist(),
            'A_mean':    A_mean.tolist(),
            'A_std':     A_std.tolist()}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _configure_style():
    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex':        have_latex,
        'font.family':        'serif',
        'font.serif':         ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset':   'cm',
        'axes.labelsize':     11,
        'axes.titlesize':     11,
        'legend.fontsize':    8,
        'figure.dpi':         120,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.08,
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    return have_latex


def theory_A(s, c):
    s = np.clip(np.asarray(s, dtype=float), 1e-12, 1.0 - 1e-12)
    return norm.cdf(c * np.sqrt(s / (1.0 - s)))


def render_weights(per_H: dict, output_dir: str, have_latex: bool) -> str:
    """Histogram of trained path-weights w_a per H, with the cumulants that
    drive the Edgeworth correction to D17 annotated."""
    os.makedirs(output_dir, exist_ok=True)
    Hs = sorted(per_H)
    cols = min(4, len(Hs))
    n_bands = int(np.ceil(len(Hs) / cols))
    panel_w = 4.0
    panel_h = 3.4
    fig_w = max(15.5, panel_w * cols + 1.0)
    fig_h = max(4.0, panel_h * n_bands + 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = fig.add_gridspec(
        n_bands, cols,
        left=0.05, right=0.99,
        top=1.0 - 1.0 / fig_h,
        bottom=0.6 / fig_h,
        wspace=0.30, hspace=0.45,
    )
    cmap = plt.cm.viridis

    for idx, H in enumerate(Hs):
        band = idx // cols
        j = idx % cols
        ax = fig.add_subplot(gs[band, j])
        d = per_H[H]
        w = np.asarray(d['w'], dtype=float)
        J0 = float(np.sum(w))
        V = float(np.sum(w * w))
        c = J0 / np.sqrt(V) if V > 0 else 0.0
        # Standardised third cumulant of the path-weight distribution:
        #   skew(w) = E[(w - mean)^3] / std(w)^3
        wbar = float(np.mean(w))
        wstd = float(np.std(w))
        skew_w = (float(np.mean((w - wbar) ** 3)) / wstd ** 3) if wstd > 0 else 0.0
        # Standardised third cumulant of the SUM J(m) at very small s,
        #   sum w^3 / (sum w^2)^(3/2),
        # which is the quantity that multiplies 1/sqrt(s) in the Edgeworth
        # skewness correction of D17.
        sum_w3 = float(np.sum(w ** 3))
        edgeworth = sum_w3 / (V ** 1.5) if V > 0 else 0.0

        # Symmetric x-range so the offset at zero is visible.
        lim = float(np.max(np.abs(w))) * 1.02
        ax.hist(w, bins=60, range=(-lim, lim),
                color=cmap(0.4), edgecolor='black', linewidth=0.3, alpha=0.85)
        ax.axvline(0.0, color='0.4', lw=0.6, linestyle=':')
        ax.axvline(wbar, color='crimson', lw=1.2, linestyle='--',
                   label=(rf'$\bar w = {wbar:+.2e}$'
                          if have_latex else f'mean = {wbar:+.2e}'))
        ax.set_xlabel(r'$w_a = W^{(1)}_a\,W^{(2)}_a$'
                      if have_latex else 'w_a = W1_a * W2_a')
        if j == 0:
            ax.set_ylabel('count')
        ax.set_title(rf'$H = {H}$' if have_latex else f'H = {H}')
        ax.grid(True, alpha=0.3, lw=0.4)
        # Stats annotation.
        if have_latex:
            stats = ('\n'.join([
                rf'$\mathcal{{J}}_{{0}}=\sum w_a={J0:+.3f}$',
                rf'$\mathcal{{V}}=\sum w_a^{{2}}={V:.2e}$',
                rf'$c=\mathcal{{J}}_0/\sqrt{{\mathcal{{V}}}}={c:.2f}$',
                rf'$\mathrm{{skew}}(w)={skew_w:+.2f}$',
                rf'$\sum w^{{3}}/\mathcal{{V}}^{{3/2}}={edgeworth:+.3f}$',
            ]))
        else:
            stats = ('\n'.join([
                f'J_0 = {J0:+.3f}',
                f'V   = {V:.2e}',
                f'c   = {c:.2f}',
                f'skew(w) = {skew_w:+.2f}',
                f'sum w^3 / V^1.5 = {edgeworth:+.3f}',
            ]))
        ax.text(0.02, 0.98, stats, transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec='0.55', lw=0.5, alpha=0.92))
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

    fig.suptitle((r'Trained path-weight distributions $w_a = W^{(1)}_a W^{(2)}_a$ '
                  r'and the cumulants driving the D17 Edgeworth correction'
                  if have_latex
                  else 'Trained path-weight distributions and Edgeworth-correction '
                       'cumulants'),
                 y=1.0 - 0.25 / fig_h)
    out = os.path.join(output_dir, 'weight_distributions.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


def render(per_H: dict, output_dir: str, have_latex: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    Hs = sorted(per_H)

    # Lay panels out so each band has at most 4 H columns. For each band we
    # use two sub-rows: A(s) on top, probit linearisation on bottom.
    cols = min(4, len(Hs))
    n_bands = int(np.ceil(len(Hs) / cols))
    panel_w = 4.0
    panel_h = 3.6
    fig_w = max(15.5, panel_w * cols + 1.0)
    fig_h = max(8.0, panel_h * 2 * n_bands + 1.5)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = fig.add_gridspec(
        2 * n_bands, cols,
        left=0.05, right=0.99,
        top=1.0 - 1.0 / fig_h,
        bottom=0.5 / fig_h,
        wspace=0.30, hspace=0.42,
    )

    cmap = plt.cm.viridis

    for idx, H in enumerate(Hs):
        band = idx // cols
        j = idx % cols
        d = per_H[H]
        s = np.array(d['densities'])
        A = np.array(d['A_mean'])
        Asd = np.array(d['A_std'])
        c = float(d['c'])
        J0 = float(d['J0'])
        V = float(d['V'])

        # ------- TOP: A(s) on log-s, with parameter-free theory overlay
        ax = fig.add_subplot(gs[2 * band, j])
        # Cover the full empirical s-range -- for large H the transition is
        # below s = 1e-3 so we must not clip there.
        s_fine = np.geomspace(max(float(s.min()), 1e-9),
                              min(float(s.max()), 1.0 - 1e-9), 400)
        ax.errorbar(s, A, yerr=Asd, fmt='o', ms=4,
                    color=cmap(0.25), alpha=0.85,
                    ecolor=cmap(0.25), capsize=2,
                    label=('empirical' if have_latex else 'empirical'))
        ax.plot(s_fine, theory_A(s_fine, c),
                color='crimson', lw=1.6,
                label=(rf'$\Phi(c\sqrt{{s/(1-s)}}),\;c={c:.2f}$'
                       if have_latex
                       else f'theory: Phi(c·sqrt(s/(1-s))), c={c:.2f}'))
        ax.axhline(0.5, color='0.5', lw=0.5, linestyle=':')
        ax.set_xscale('log')
        ax.set_xlim(min(s), 1.0)
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel(r'Density $s$' if have_latex else 'Density s')
        if j == 0:
            ax.set_ylabel(r'Accuracy $A(s)$' if have_latex else 'Accuracy A(s)')
        ax.set_title(rf'$H = {H}$' if have_latex else f'H = {H}')
        ax.grid(True, which='both', alpha=0.3, lw=0.4)
        ax.legend(loc='lower right', framealpha=0.9)

        # ------- BOTTOM: probit linearization, Phi^{-1}(A) vs sqrt(s/(1-s))
        # Probit inversion has a finite dynamic range (Phi^{-1}(0.999) ~ 3.1
        # while the c*u line is unbounded). We therefore restrict the panel
        # to the informative window 0.55 < A_emp < 0.99; both axes are then
        # comparable and any departure from linearity is visible.
        ax2 = fig.add_subplot(gs[2 * band + 1, j])
        u = np.sqrt(s / (1.0 - s))
        sel = (A > 0.55) & (A < 0.99)
        u_sel = u[sel]
        A_sel = A[sel]
        if u_sel.size >= 2:
            z_emp = norm.ppf(A_sel)
            u_hi = float(max(u_sel.max() * 1.05, 1.0))
            u_fine = np.linspace(0, u_hi, 200)
            ax2.plot(u_fine, c * u_fine,
                     color='crimson', lw=1.6,
                     label=(rf'$y = c\,u,\;c={c:.2f}$'
                            if have_latex else f'y = c u, c={c:.2f}'))
            ax2.plot(u_sel, z_emp, 'o', ms=5, color=cmap(0.25),
                     label=(r'$\Phi^{-1}(A_{\mathrm{emp}})$'
                            if have_latex else 'Phi^-1(A_emp)'))
            # Least-squares slope through the origin gives the data's own
            # estimate of c.
            c_emp = float(np.sum(u_sel * z_emp) / np.sum(u_sel ** 2))
            ax2.plot(u_fine, c_emp * u_fine,
                     color='navy', lw=1.0, linestyle='--',
                     label=(rf'best-slope fit, $c_{{\mathrm{{emp}}}}={c_emp:.2f}$'
                            if have_latex
                            else f'best-slope fit, c_emp={c_emp:.2f}'))
            resid = z_emp - c * u_sel
            mae = float(np.mean(np.abs(resid)))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((z_emp - z_emp.mean()) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
            txt = ((rf'$c_{{\mathrm{{emp}}}}/c = {c_emp / c:.3f}$'
                    '\n'
                    rf'$\langle|\Phi^{{-1}}(A) - c\,u|\rangle = {mae:.3f}$'
                    '\n'
                    rf'$R^{{2}} = {r2:+.3f}\quad (N={int(sel.sum())})$')
                   if have_latex
                   else f'c_emp/c = {c_emp / c:.3f}\n'
                        f'<|Phi^-1(A) - c u|> = {mae:.3f}\n'
                        f'R^2 = {r2:+.3f}  (N={int(sel.sum())})')
            ax2.text(0.97, 0.03, txt, transform=ax2.transAxes,
                     ha='right', va='bottom', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='0.55', lw=0.5, alpha=0.92))
        else:
            ax2.text(0.5, 0.5, 'transition outside sampled range',
                     transform=ax2.transAxes, ha='center', va='center',
                     fontsize=10, color='0.4')
        ax2.axhline(0, color='0.5', lw=0.5, linestyle=':')
        ax2.axvline(0, color='0.5', lw=0.5, linestyle=':')
        ax2.set_xlabel(r'$\sqrt{s/(1-s)}$' if have_latex else 'sqrt(s/(1-s))')
        if j == 0:
            ax2.set_ylabel(r'$\Phi^{-1}(A_{\mathrm{emp}}(s))$'
                           if have_latex else 'Phi^-1(A_emp(s))')
        ax2.set_title(rf'Probit linearization, $H={H}$'
                      if have_latex else f'Probit linearization, H={H}')
        ax2.grid(True, alpha=0.3, lw=0.4)
        ax2.legend(loc='upper left', framealpha=0.9)

    fig.suptitle((r'Appendix D toy: $A(s) \overset{?}{=} '
                  r'\Phi\!\left(c\,\sqrt{s/(1-s)}\right)$'
                  r' for a trained linear $1$-hidden-layer network'
                  if have_latex
                  else 'Appendix D toy: A(s) ?= Phi(c·sqrt(s/(1-s))) '
                       'for a trained linear 1-hidden-layer net'),
                 y=1.0 - 0.25 / fig_h)

    out = os.path.join(output_dir, 'binary_classification.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hs', type=int, nargs='+', default=list(DEFAULT_HS),
                    help='hidden widths to test')
    ap.add_argument('--n-seeds', type=int, default=400,
                    help='mask seeds per density')
    ap.add_argument('--n-test', type=int, default=20000,
                    help='test-set size for the accuracy estimate')
    ap.add_argument('--n-train', type=int, default=20000,
                    help='training-set size')
    ap.add_argument('--n-epochs', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-densities', type=int, default=150,
                    help='densities sweep size; log-spaced in '
                         'u = sqrt(s/(1-s)) so points spread evenly across '
                         'the transition for every H')
    ap.add_argument('--u-min', type=float, default=0.001,
                    help='lower bound on u = sqrt(s/(1-s)); s_min ~ u_min^2. '
                         'Needs to be small enough that for the largest H '
                         'the transition midpoint u_mid ~ 0.674/c is sampled')
    ap.add_argument('--u-max', type=float, default=30.0,
                    help='upper bound on u; s_max = u^2 / (1 + u^2)')
    ap.add_argument('--output-dir',
                    default='unstructured_pruning/toy_examples/figures/binary')
    args = ap.parse_args()

    have_latex = _configure_style()
    torch.manual_seed(args.seed)

    # Density grid log-spaced in u = sqrt(s/(1-s)). This is the natural
    # transition variable: A(s) = Phi(c*u) under D17, so log-uniform u puts
    # points evenly across the sigmoid for every H. Converting back:
    # s = u^2 / (1 + u^2).
    u_grid = np.geomspace(args.u_min, args.u_max, args.n_densities)
    densities = (u_grid ** 2) / (1.0 + u_grid ** 2)

    per_H = {}
    for H in args.hs:
        print(f'  H = {H}: training...')
        model = LinearHiddenNet(H=H)
        train(model,
              n_train=args.n_train, n_epochs=args.n_epochs,
              lr=args.lr, batch_size=args.batch_size, seed=args.seed)
        w = model.path_weights()
        J0 = float(np.sum(w))
        V = float(np.sum(w ** 2))
        c = J0 / np.sqrt(V) if V > 0 else 0.0
        # Unpruned baseline classification rate (should be ~1).
        baseline = norm.cdf(c * 1e6) if J0 > 0 else 0.5

        print(f'    J_0 = {J0:+.4f}   V = {V:.4f}   '
              f'c = J_0/sqrt(V) = {c:+.4f}   '
              f'sqrt(H) = {np.sqrt(H):.4f}   '
              f'A_theory(s=1)~{baseline:.4f}')
        print(f'    sweeping densities (n={len(densities)}, '
              f'mask seeds={args.n_seeds})...')
        out = sweep(model, densities=densities,
                    n_seeds=args.n_seeds, n_test=args.n_test,
                    seed=args.seed)
        out.update({'H': int(H), 'J0': J0, 'V': V, 'c': c,
                    'w': [float(x) for x in w]})
        per_H[int(H)] = out

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({str(H): v for H, v in per_H.items()}, f, indent=2)
    png = render(per_H, args.output_dir, have_latex)
    print(f'\nSaved figure: {png}')
    png_w = render_weights(per_H, args.output_dir, have_latex)
    print(f'Saved figure: {png_w}')
    print(f'Saved data:   {os.path.join(args.output_dir, "results.json")}')


if __name__ == '__main__':
    main()
