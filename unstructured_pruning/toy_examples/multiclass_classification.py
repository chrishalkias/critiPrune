#!/usr/bin/env python3
r"""Appendix F toy: C-class linear classification under random pruning.

Setup (paper Appendix F, with :math:`L = 1` for clean closed form)
------------------------------------------------------------------
* input :math:`\mathbf{x}\in\mathbb{R}^{D}` from a :math:`C`-component
  Gaussian mixture with unit-norm centroids;
* one linear hidden layer of width :math:`H`:
  :math:`\hat z_{k} = \sum_{a} W^{(2)}_{ka}\,h_{a},\;
                   h_{a} = \sum_{b} m_{ab}\,W^{(1)}_{ab}\,x_{b}`;
* :math:`C`-class argmax readout;
* :math:`m_{ab}\sim\text{Bern}(s)` iid on the hidden weights only.

Theoretical predictions
-----------------------
**(F41) independent-competitor (exact in the Gaussian / no-correlation limit).**
For a test example :math:`(\mathbf{x}, y)`, define the per-competitor SNR

.. math::
    r_{k}(\mathbf{x}, y) \;=\;
    \frac{\bar z_{y} - \bar z_{k}}{\sqrt{\sum_{a}(\Delta W_{ka})^{2}\,v_{a}(\mathbf{x})}},
    \quad
    \Delta W_{ka} = W^{(2)}_{ya} - W^{(2)}_{ka},
    \quad
    v_{a}(\mathbf{x}) = \sum_{b} (W^{(1)}_{ab}\,x_{b})^{2}.

Then equation (F41) gives

.. math::
    A(s) \;\approx\; \mathbb{E}_{(\mathbf{x},y)}
    \prod_{k\neq y}\Phi\!\left(r_{k}(\mathbf{x}, y)\,\sqrt{\frac{s}{1-s}}\right).

**(F42) symmetric-competitor (clean closed form).** If all
:math:`r_{k}(\mathbf{x}, y)` are roughly equal to some effective :math:`c`,
the product becomes :math:`\Phi(c\,u)^{C-1}` with :math:`u = \sqrt{s/(1-s)}`.

**Probit linearisation (generalised).** Under (F42),
:math:`\Phi^{-1}\!\big(A(s)^{1/(C-1)}\big) = c\,u` -- a straight line
through the origin with slope :math:`c`. The C=2 case reduces to the
binary (D17) test of ``binary_classification.py``.

Usage
-----
    .venv/bin/python -m unstructured_pruning.toy_examples.multiclass_classification
    .venv/bin/python -m unstructured_pruning.toy_examples.multiclass_classification \
        --cs 2 3 5 10 --H 64 --n-seeds 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_CS = (2, 3, 5, 10)


# ---------------------------------------------------------------------------
# Model & data
# ---------------------------------------------------------------------------
class LinearMultiClassNet(nn.Module):
    """One hidden layer, linear: :math:`z = W^{(2)}\\,(m\\odot W^{(1)})\\,x`."""

    def __init__(self, D: int, H: int, C: int):
        super().__init__()
        self.D, self.H, self.C = int(D), int(H), int(C)
        self.W1 = nn.Parameter(torch.randn(self.H, self.D) / np.sqrt(self.D))
        self.W2 = nn.Parameter(torch.randn(self.C, self.H) / np.sqrt(self.H))

    def forward(self, x, mask=None):
        # x: (B, D), mask: (H, D) or None
        W = self.W1 if mask is None else (self.W1 * mask)
        h = x @ W.t()                  # (B, H)
        return h @ self.W2.t()         # (B, C)


def make_centroids(C: int, D: int, seed: int) -> np.ndarray:
    """Random unit-norm class centroids in :math:`\\mathbb{R}^{D}`."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((C, D))
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def gen_data(centroids: np.ndarray, N: int, sigma: float, seed: int):
    """C-class isotropic-noise Gaussian mixture using pre-built centroids."""
    rng = np.random.default_rng(seed)
    C, D = centroids.shape
    Y = rng.integers(0, C, size=N)
    X = centroids[Y] + sigma * rng.standard_normal((N, D))
    return (torch.from_numpy(X.astype(np.float32)),
            torch.from_numpy(Y.astype(np.int64)))


def train(model, X, Y, *, n_epochs, batch_size, lr, momentum=0.9):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    N = X.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Theoretical predictions: (F41) and (F42)
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_example_snr(model: LinearMultiClassNet,
                    X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    r"""Compute :math:`r_{k}(\mathbf{x}, y)` for each test example and class
    :math:`k \neq y`. Returns an ``(N, C)`` array; entries with :math:`k = y`
    are set to ``+inf`` so that ``Phi(inf) = 1`` is harmless in the product.
    """
    h_bar = X @ model.W1.t()                           # (N, H)
    v = (X ** 2) @ (model.W1 ** 2).t()                 # (N, H) -- v_a(x)
    z_bar = h_bar @ model.W2.t()                       # (N, C)
    z_bar_y = z_bar.gather(1, Y.unsqueeze(1))          # (N, 1)
    delta_z = z_bar_y - z_bar                          # (N, C)
    W2_y = model.W2[Y]                                 # (N, H)
    delta_W = W2_y.unsqueeze(1) - model.W2.unsqueeze(0)  # (N, C, H)
    Sigma_unit = (delta_W ** 2 * v.unsqueeze(1)).sum(dim=-1)  # (N, C)
    denom = torch.sqrt(Sigma_unit.clamp(min=1e-30))
    r = delta_z / denom                                # (N, C)
    mask_y = (torch.arange(model.C).unsqueeze(0) == Y.unsqueeze(1))
    r = torch.where(mask_y, torch.full_like(r, float('inf')), r)
    return r.cpu().numpy()


def theory_F41(r_table: np.ndarray, s: float) -> float:
    """Independent-competitor prediction (F41)."""
    if not (0.0 < s < 1.0):
        # Defer to argmax tie-breaking at s=0; saturate to 1 at s=1.
        return 1.0 if s >= 1.0 else 1.0 / r_table.shape[1]
    u = float(np.sqrt(s / (1.0 - s)))
    args = r_table * u                       # (N, C); k=y entries remain inf
    per_ex = norm.cdf(args).prod(axis=1)     # (N,)
    return float(per_ex.mean())


def theory_F42(c_eff: float, s: float, C: int) -> float:
    """Symmetric-competitor prediction (F42)."""
    if not (0.0 < s < 1.0):
        return 1.0 if s >= 1.0 else 1.0 / C
    u = float(np.sqrt(s / (1.0 - s)))
    return float(norm.cdf(c_eff * u) ** (C - 1))


# ---------------------------------------------------------------------------
# Empirical sweep
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mask(model, X, Y, mask) -> float:
    logits = model(X, mask)
    pred = logits.argmax(dim=1)
    return float((pred == Y).float().mean())


def sweep(model, X, Y, *, densities, n_seeds, seed):
    rng = np.random.default_rng(seed + 7)
    H, D = model.H, model.D
    A_mean = np.zeros_like(densities)
    A_std = np.zeros_like(densities)
    for i, s in enumerate(densities):
        accs = np.empty(n_seeds, dtype=float)
        for k in range(n_seeds):
            m = torch.from_numpy(
                (rng.random((H, D)) < s).astype(np.float32))
            accs[k] = evaluate_mask(model, X, Y, m)
        A_mean[i] = accs.mean()
        A_std[i] = accs.std()
    return A_mean, A_std


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


def render(per_C: dict, output_dir: str, have_latex: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    Cs = sorted(per_C)
    fig = plt.figure(figsize=(15.5, 8.0), facecolor='white')
    gs = fig.add_gridspec(2, len(Cs), left=0.06, right=0.985,
                          top=0.92, bottom=0.09, wspace=0.28, hspace=0.34)
    cmap = plt.cm.viridis

    for j, C in enumerate(Cs):
        d = per_C[C]
        s = np.array(d['densities'])
        A_emp = np.array(d['A_emp_mean'])
        A_std = np.array(d['A_emp_std'])
        A_F41 = np.array(d['A_F41'])
        A_F42 = np.array(d['A_F42'])
        c_eff = float(d['c_eff'])
        A_full = float(d['A_unpruned'])

        # ------- TOP: A(s) on log-s with both predictions overlaid
        ax = fig.add_subplot(gs[0, j])
        ax.errorbar(s, A_emp, yerr=A_std, fmt='o', ms=4,
                    color=cmap(0.25), alpha=0.85, capsize=2,
                    label='empirical')
        ax.plot(s, A_F41, color='crimson', lw=1.6,
                label=(r'(F41) independent-competitor'
                       if have_latex else 'F41: indep. competitor'))
        ax.plot(s, A_F42, color='navy', lw=1.2, linestyle='--',
                label=(rf'(F42) $\Phi(c\,u)^{{C-1}},\;c={c_eff:.2f}$'
                       if have_latex
                       else f'F42 symmetric, c={c_eff:.2f}'))
        ax.axhline(1.0 / C, color='0.5', lw=0.5, linestyle=':',
                   label=(rf'chance $1/C$' if have_latex else '1/C'))
        ax.axhline(A_full, color='0.7', lw=0.5, linestyle='-.',
                   label=(rf'$A_{{\mathrm{{unpruned}}}}={A_full:.3f}$'
                          if have_latex else f'A_unpruned={A_full:.3f}'))
        ax.set_xscale('log')
        ax.set_xlim(min(s), 1.0)
        ax.set_ylim(max(0.0, 1.0 / C - 0.05), 1.02)
        ax.set_xlabel(r'Density $s$' if have_latex else 'Density s')
        if j == 0:
            ax.set_ylabel(r'Accuracy $A(s)$'
                          if have_latex else 'Accuracy A(s)')
        ax.set_title(rf'$C = {C}$' if have_latex else f'C = {C}')
        ax.grid(True, which='both', alpha=0.3, lw=0.4)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=7)

        # ------- BOTTOM: generalised probit linearisation
        #   under (F42):  Phi^{-1}( A^{1/(C-1)} )  =  c * sqrt(s/(1-s))
        ax2 = fig.add_subplot(gs[1, j])
        u = np.sqrt(s / (1.0 - s))
        # Restrict to A in an informative window away from chance and ceiling.
        # The ceiling cutoff scales with C: under (F42), A=0.99 corresponds to
        # different Phi^{-1} values for different C, but in all cases anything
        # very close to 1 collapses the (C-1)-th-root transform onto a flat
        # plateau, killing the slope test.
        A_lo = max((1.0 / C) + 0.02, 0.55)
        A_hi = 1.0 - max(0.005, 0.05 / max(C - 1, 1))
        sel = (A_emp > A_lo) & (A_emp < A_hi)
        if sel.sum() >= 3:
            A_sel = A_emp[sel]
            u_sel = u[sel]
            z_emp = norm.ppf(np.power(A_sel, 1.0 / (C - 1)))
            # Display window: focus on the transition window only.
            u_lo = 0.0
            u_hi = float(u_sel.max() * 1.1)
            u_fine = np.linspace(u_lo, u_hi, 200)
            ax2.plot(u_fine, c_eff * u_fine,
                     color='crimson', lw=1.6,
                     label=(rf'$y = c_{{\mathrm{{eff}}}}\,u,'
                            rf'\;c_{{\mathrm{{eff}}}}={c_eff:.2f}$'
                            if have_latex
                            else f'y = c_eff u, c_eff={c_eff:.2f}'))
            ax2.plot(u_sel, z_emp, 'o', ms=5, color=cmap(0.25),
                     label=(r'$\Phi^{-1}(A_{\mathrm{emp}}^{1/(C-1)})$'
                            if have_latex
                            else 'Phi^-1(A_emp^{1/(C-1)})'))
            c_emp = float(np.sum(u_sel * z_emp) / np.sum(u_sel ** 2))
            ax2.plot(u_fine, c_emp * u_fine,
                     color='navy', linestyle='--', lw=1.0,
                     label=(rf'best-slope, $c_{{\mathrm{{emp}}}}={c_emp:.2f}$'
                            if have_latex
                            else f'best-slope, c_emp={c_emp:.2f}'))
            ax2.set_xlim(u_lo, u_hi)
            ax2.set_ylim(0.0,
                         float(max(z_emp.max(), c_eff * u_hi) * 1.05))
            resid = z_emp - c_eff * u_sel
            mae = float(np.mean(np.abs(resid)))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((z_emp - z_emp.mean()) ** 2))
            r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
            txt = ((rf'$c_{{\mathrm{{emp}}}}/c_{{\mathrm{{eff}}}} = '
                    rf'{c_emp / c_eff:.3f}$' '\n'
                    rf'$\langle|\Delta|\rangle = {mae:.3f}$' '\n'
                    rf'$R^{{2}} = {r2:+.3f}\quad(N={int(sel.sum())})$')
                   if have_latex
                   else (f'c_emp/c_eff = {c_emp / c_eff:.3f}\n'
                         f'<|delta|> = {mae:.3f}\n'
                         f'R^2 = {r2:+.3f}  (N={int(sel.sum())})'))
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
        ax2.set_xlabel(r'$\sqrt{s/(1-s)}$' if have_latex
                       else 'sqrt(s/(1-s))')
        if j == 0:
            ax2.set_ylabel(
                (r'$\Phi^{-1}\!\left(A_{\mathrm{emp}}^{\,1/(C-1)}\right)$'
                 if have_latex
                 else 'Phi^-1( A_emp^{1/(C-1)} )'))
        ax2.set_title(rf'Probit linearisation, $C={C}$'
                      if have_latex else f'Probit linearisation, C={C}')
        ax2.grid(True, alpha=0.3, lw=0.4)
        ax2.legend(loc='upper left', framealpha=0.9, fontsize=7)

    fig.suptitle(
        (r'Appendix F multi-class toy: $A(s) \overset{?}{=} '
         r'\Phi(c\,\sqrt{s/(1-s)})^{\,C-1}$ '
         r'on a trained linear $1$-hidden-layer net'
         if have_latex
         else 'Appendix F multi-class toy: A(s) ?= Phi(c sqrt(s/(1-s)))^(C-1)'),
        y=0.985)
    out = os.path.join(output_dir, 'multiclass_classification.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cs', type=int, nargs='+', default=list(DEFAULT_CS),
                    help='number-of-classes values to sweep')
    ap.add_argument('--H', type=int, default=64, help='hidden width')
    ap.add_argument('--D-mult', type=int, default=2,
                    help='input dim D = D_mult * C')
    ap.add_argument('--sigma', type=float, default=0.15,
                    help='Gaussian noise on the mixture centroids')
    ap.add_argument('--n-train', type=int, default=8000)
    ap.add_argument('--n-test', type=int, default=4000)
    ap.add_argument('--n-epochs', type=int, default=80,
                    help='full passes over the dataset')
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-seeds', type=int, default=300,
                    help='mask seeds per density')
    ap.add_argument('--n-densities', type=int, default=150,
                    help='densities sweep size; log-spaced in '
                         'u = sqrt(s/(1-s)) so points spread evenly across '
                         'the transition for every C')
    ap.add_argument('--u-min', type=float, default=0.01,
                    help='lower bound on u = sqrt(s/(1-s))')
    ap.add_argument('--u-max', type=float, default=30.0,
                    help='upper bound on u; s_max = u^2 / (1 + u^2)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output-dir',
                    default='assets/unstructured_pruning/toy_examples/multiclass')
    args = ap.parse_args()

    have_latex = _configure_style()
    torch.manual_seed(args.seed)

    # Density grid log-spaced in u = sqrt(s/(1-s)) -- the natural transition
    # variable for (F41)/(F42), so points populate the transition evenly
    # across every C. Inversion: s = u^2 / (1 + u^2).
    u_grid = np.geomspace(args.u_min, args.u_max, args.n_densities)
    densities = (u_grid ** 2) / (1.0 + u_grid ** 2)

    per_C = {}
    for C in args.cs:
        D = args.D_mult * C
        print(f'  C = {C} (D = {D}, H = {args.H}): training...')
        model = LinearMultiClassNet(D=D, H=args.H, C=C)
        centroids = make_centroids(C, D, seed=args.seed)
        X_tr, Y_tr = gen_data(centroids, args.n_train,
                              args.sigma, seed=args.seed + 1)
        X_te, Y_te = gen_data(centroids, args.n_test,
                              args.sigma, seed=args.seed + 100)
        train(model, X_tr, Y_tr,
              n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
        with torch.no_grad():
            A_full = float((model(X_te).argmax(1) == Y_te).float().mean())

        r_table = per_example_snr(model, X_te, Y_te)
        finite_r = r_table[np.isfinite(r_table)]
        c_eff = float(np.median(finite_r))
        c_mean = float(np.mean(finite_r))
        print(f'    A_unpruned = {A_full:.4f}, '
              f'c_eff (median r_k) = {c_eff:.3f}, mean = {c_mean:.3f}')

        print(f'    sweeping densities (n={len(densities)}, '
              f'mask seeds={args.n_seeds})...')
        A_mean, A_std = sweep(model, X_te, Y_te,
                              densities=densities,
                              n_seeds=args.n_seeds, seed=args.seed)
        A_F41 = np.array([theory_F41(r_table, float(s)) for s in densities])
        A_F42 = np.array([theory_F42(c_eff, float(s), C) for s in densities])

        per_C[int(C)] = {
            'C': int(C), 'D': int(D), 'H': int(args.H),
            'A_unpruned': A_full,
            'c_eff': c_eff, 'c_mean': c_mean,
            'densities': densities.tolist(),
            'A_emp_mean': A_mean.tolist(),
            'A_emp_std':  A_std.tolist(),
            'A_F41':      A_F41.tolist(),
            'A_F42':      A_F42.tolist(),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({str(C): v for C, v in per_C.items()}, f, indent=2)
    png = render(per_C, args.output_dir, have_latex)
    print(f'\nSaved figure: {png}')
    print(f'Saved data:   {os.path.join(args.output_dir, "results.json")}')


if __name__ == '__main__':
    main()
