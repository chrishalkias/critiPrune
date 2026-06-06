#!/usr/bin/env python3
r"""Extend the Appendix F toy to MNIST with L hidden layers and ReLU.

Setup
-----
* MNIST 28x28 normalised (mean 0.1307, std 0.3081), all 60k train and
  10k test samples.
* Fully-connected ReLU classifier:
    784 -> H -> H -> ... -> H (L hidden layers) -> 10
  with biases on every layer.
* Random Bernoulli(s) mask applied independently to every hidden weight
  matrix; the read-out matrix is left intact (matches paper Appendix F.5).
* Standard SGD + cross-entropy training.

Theoretical prediction tested
-----------------------------
For each test example :math:`(\mathbf{x}, y)`, propagate the per-neuron
post-ReLU moments :math:`(\mu, q, v) = (E[h], E[h^2], \mathrm{Var}[h])`
through the L hidden layers under random masks at density :math:`s` via
eqs. (F22)-(F28):

  preact mean       :math:`\mu^{\mathrm{pre}} = s\,W\,\mu^{(l-1)} + b`
  preact variance   :math:`\sigma^{2\,\mathrm{pre}} = (W^{2})\,[s\,q^{(l-1)} - s^{2}\,(\mu^{(l-1)})^{2}]`
  ReLU moments via the exact :math:`\Phi`/:math:`\varphi` formulas (F8)-(F11).

The (un-masked) read-out then gives

.. math::
    M_{k}(s, \mathbf{x}) &= \sum_{a}(W^{(L+1)}_{ya} - W^{(L+1)}_{ka})\,\mu^{(L)}_{a}
                          + (b_{y} - b_{k}) \\
    \Sigma_{kk}(s, \mathbf{x}) &= \sum_{a}(W^{(L+1)}_{ya} - W^{(L+1)}_{ka})^{2}\,v^{(L)}_{a},

and the F41 prediction averages over the test set:

.. math::
    A_{\mathrm{F41}}(s) = \mathbb{E}_{(\mathbf{x},y)}
        \prod_{k\neq y}\Phi\!\left(\frac{M_{k}(s, \mathbf{x})}
                                        {\sqrt{\Sigma_{kk}(s, \mathbf{x})}}\right).

We sweep ``s`` log-uniform in :math:`u = \sqrt{s/(1-s)}` and check the
prediction directly against empirical accuracy on the masked network.

Usage
-----
    .venv/bin/python -m unstructured_pruning.toy_examples.mnist_relu_multilayer
    .venv/bin/python -m unstructured_pruning.toy_examples.mnist_relu_multilayer \
        --Ls 1 2 3 --H 128 --n-seeds 30 --n-densities 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_LS = (1, 2, 3)
DEFAULT_MNIST_DIR = os.environ.get('MNIST_DATA_DIR', '/tmp/mnist28')


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FCReLU(nn.Module):
    """L-hidden-layer ReLU network. Hidden weights are maskable; the
    read-out matrix and all biases are kept intact."""

    def __init__(self, D: int, H: int, L: int, C: int):
        super().__init__()
        self.D, self.H, self.L, self.C = int(D), int(H), int(L), int(C)
        dims = [self.D] + [self.H] * self.L
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(dims[i + 1], dims[i]).normal_(
                0.0, 1.0 / float(np.sqrt(dims[i]))))
            for i in range(self.L)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(dims[i + 1])) for i in range(self.L)
        ])
        self.W_out = nn.Parameter(torch.empty(self.C, self.H).normal_(
            0.0, 1.0 / float(np.sqrt(self.H))))
        self.b_out = nn.Parameter(torch.zeros(self.C))

    def forward(self, x, masks=None):
        h = x
        for l in range(self.L):
            W = self.W[l]
            if masks is not None and masks[l] is not None:
                W = W * masks[l]
            h = F.relu(h @ W.t() + self.b[l])
        return h @ self.W_out.t() + self.b_out


# ---------------------------------------------------------------------------
# F41 moment-propagation recursion (paper eqs F22-F28 + F8-F11)
# ---------------------------------------------------------------------------
@torch.no_grad()
def f41_recursion(model: FCReLU,
                  X: torch.Tensor,
                  Y: torch.Tensor,
                  s: float) -> tuple[float, np.ndarray]:
    """Return ``(A_F41(s), ratios)`` where ratios is the (N, C) array of
    :math:`r_{k}(\\mathbf{x}, y) = M_{k}/\\sqrt{\\Sigma_{kk}}` with k=y
    entries set to ``+inf`` so they contribute :math:`\\Phi(\\infty)=1`."""
    mu = X
    q = X ** 2
    v = torch.zeros_like(X)

    for l in range(model.L):
        W = model.W[l]
        Wsq = W ** 2
        pm = s * (mu @ W.t()) + model.b[l]
        # diagonal (no cross-neuron covariance) approximation; same as the
        # paper's (F25) law-of-total-variance step
        pv = (s * q - s * s * mu * mu) @ Wsq.t()
        pv = pv.clamp(min=1e-12)
        ps = torch.sqrt(pv)
        rho = pm / ps

        rho_np = rho.cpu().numpy()
        phi = torch.from_numpy(norm.pdf(rho_np).astype(np.float32))
        Phi = torch.from_numpy(norm.cdf(rho_np).astype(np.float32))

        # ReLU moments (F8) and (F10)
        #   E[h]   = sigma * phi(rho) + mu * Phi(rho)
        #   E[h^2] = (mu^2 + sigma^2) * Phi(rho) + mu * sigma * phi(rho)
        mu_new = ps * phi + pm * Phi
        q_new = (pm * pm + pv) * Phi + pm * ps * phi
        v_new = (q_new - mu_new * mu_new).clamp(min=0.0)

        mu, q, v = mu_new, q_new, v_new

    # Read-out (unmasked): z_k = sum_a W_out[k, a] * h^(L)_a + b_out[k]
    z_bar = mu @ model.W_out.t() + model.b_out                     # (N, C)
    W_out = model.W_out
    z_y = z_bar.gather(1, Y.unsqueeze(1))                          # (N, 1)
    M = z_y - z_bar                                                # (N, C)
    W_y = W_out[Y]                                                 # (N, H)
    dW = W_y.unsqueeze(1) - W_out.unsqueeze(0)                     # (N, C, H)
    Sigma_kk = (dW * dW * v.unsqueeze(1)).sum(dim=-1).clamp(min=1e-30)
    r = M / torch.sqrt(Sigma_kk)
    mask_y = (torch.arange(model.C).unsqueeze(0) == Y.unsqueeze(1))
    r = torch.where(mask_y, torch.full_like(r, float('inf')), r)
    r_np = r.cpu().numpy()
    A = norm.cdf(r_np).prod(axis=1)
    return float(A.mean()), r_np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_mnist(data_dir: str):
    """Load MNIST 28x28 to flat float32 tensors, normalised."""
    from torchvision import datasets, transforms  # local import
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    os.makedirs(data_dir, exist_ok=True)
    train = datasets.MNIST(data_dir, train=True, download=True,
                           transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True,
                          transform=transform)
    X_tr = train.data.float().div(255.0).sub(0.1307).div(0.3081)
    X_te = test.data.float().div(255.0).sub(0.1307).div(0.3081)
    X_tr = X_tr.view(-1, 28 * 28)
    X_te = X_te.view(-1, 28 * 28)
    Y_tr = train.targets.long()
    Y_te = test.targets.long()
    return X_tr, Y_tr, X_te, Y_te


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, X, Y, *, n_epochs, batch_size, lr, momentum=0.9,
          weight_decay=0.0, verbose=False):
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    N = X.shape[0]
    for ep in range(n_epochs):
        perm = torch.randperm(N)
        running = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss)
            n_batches += 1
        if verbose:
            print(f'      epoch {ep + 1:3d}: avg loss = '
                  f'{running / max(n_batches, 1):.4f}')


# ---------------------------------------------------------------------------
# Empirical sweep
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_with_masks(model, X, Y, masks) -> float:
    pred = model(X, masks).argmax(dim=1)
    return float((pred == Y).float().mean())


def sweep(model, X, Y, *, densities, n_seeds, seed):
    rng = np.random.default_rng(seed + 11)
    A_mean = np.zeros_like(densities)
    A_std = np.zeros_like(densities)
    for i, s in enumerate(densities):
        accs = np.empty(n_seeds, dtype=float)
        for k in range(n_seeds):
            masks = []
            for l in range(model.L):
                shp = tuple(model.W[l].shape)
                masks.append(torch.from_numpy(
                    (rng.random(shp) < s).astype(np.float32)))
            accs[k] = evaluate_with_masks(model, X, Y, masks)
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
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}')
    return have_latex


def render(per_L: dict, output_dir: str, have_latex: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    Ls = sorted(per_L)
    cols = len(Ls)
    fig_w = max(15.5, 4.5 * cols + 1.0)
    fig = plt.figure(figsize=(fig_w, 8.5), facecolor='white')
    gs = fig.add_gridspec(2, cols, left=0.06, right=0.985,
                          top=0.92, bottom=0.08,
                          wspace=0.28, hspace=0.34)
    cmap = plt.cm.viridis
    for j, L in enumerate(Ls):
        d = per_L[L]
        s = np.array(d['densities'])
        A_emp = np.array(d['A_emp_mean'])
        A_std = np.array(d['A_emp_std'])
        A_F41 = np.array(d['A_F41'])
        A_full = float(d['A_unpruned'])
        C = int(d['C'])

        # ------- TOP: A(s) on log-s
        ax = fig.add_subplot(gs[0, j])
        ax.errorbar(s, A_emp, yerr=A_std, fmt='o', ms=4,
                    color=cmap(0.25), alpha=0.85, capsize=2,
                    label=('empirical' if have_latex else 'empirical'))
        ax.plot(s, A_F41, color='crimson', lw=1.6,
                label=(r'(F41) parameter-free' if have_latex
                       else 'F41 (parameter-free)'))
        ax.axhline(1.0 / C, color='0.5', lw=0.5, linestyle=':',
                   label=(r'chance $1/C$' if have_latex else '1/C'))
        ax.axhline(A_full, color='0.6', lw=0.5, linestyle='-.',
                   label=(rf'$A_{{\mathrm{{unpruned}}}}={A_full:.3f}$'
                          if have_latex else f'A_unpruned={A_full:.3f}'))
        ax.set_xscale('log')
        ax.set_xlim(min(s), 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(r'Density $s$' if have_latex else 'Density s')
        if j == 0:
            ax.set_ylabel(r'Accuracy $A(s)$' if have_latex
                          else 'Accuracy A(s)')
        ax.set_title(rf'$L = {L}$ hidden ReLU layer{"s" if L > 1 else ""}'
                     if have_latex else f'L = {L}')
        ax.grid(True, which='both', alpha=0.3, lw=0.4)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=7)

        # ------- BOTTOM: residual A_emp - A_F41
        ax2 = fig.add_subplot(gs[1, j])
        resid = A_emp - A_F41
        ax2.errorbar(s, resid, yerr=A_std, fmt='o', ms=4,
                     color=cmap(0.45), alpha=0.85, capsize=2)
        ax2.axhline(0, color='crimson', lw=1.2)
        # Mean absolute residual over the transition window
        sel = (A_emp > 1.0 / C + 0.05) & (A_emp < 0.99)
        if sel.any():
            mae = float(np.mean(np.abs(resid[sel])))
            mbe = float(np.mean(resid[sel]))
            txt = ((rf'$\langle|\Delta|\rangle = {mae:.3f}$' '\n'
                    rf'$\langle\Delta\rangle = {mbe:+.3f}$' '\n'
                    rf'$N_{{\mathrm{{window}}}} = {int(sel.sum())}$')
                   if have_latex
                   else (f'<|delta|> = {mae:.3f}\n'
                         f'<delta>  = {mbe:+.3f}\n'
                         f'N_window = {int(sel.sum())}'))
            ax2.text(0.03, 0.97, txt, transform=ax2.transAxes,
                     ha='left', va='top', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white',
                               ec='0.55', lw=0.5, alpha=0.92))
        ax2.set_xscale('log')
        ax2.set_xlim(min(s), 1.0)
        ax2.set_xlabel(r'Density $s$' if have_latex else 'Density s')
        if j == 0:
            ax2.set_ylabel(
                r'$A_{\mathrm{emp}} - A_{\mathrm{F41}}$' if have_latex
                else 'A_emp - A_F41')
        ax2.set_title(rf'Residual, $L = {L}$' if have_latex
                      else f'Residual, L = {L}')
        ax2.grid(True, which='both', alpha=0.3, lw=0.4)

    fig.suptitle(
        (r'Appendix F toy on real MNIST: '
         r'$A(s)\overset{?}{=}\mathbb{E}_{\mathbf{x}}\prod_{k\neq y}'
         r'\Phi(M_k/\sqrt{\Sigma_{kk}})$, '
         r'$L$ hidden ReLU layers, parameter-free'
         if have_latex
         else 'MNIST + L hidden ReLU layers: '
              'A(s) ?= E_x Pi Phi(M_k / sqrt(Sigma_kk))'),
        y=0.985)
    out = os.path.join(output_dir, 'mnist_relu_multilayer.png')
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Ls', type=int, nargs='+', default=list(DEFAULT_LS),
                    help='number-of-hidden-layers values to sweep')
    ap.add_argument('--H', type=int, default=128, help='hidden width')
    ap.add_argument('--n-train', type=int, default=60000,
                    help='cap on training-set size')
    ap.add_argument('--n-test', type=int, default=4000,
                    help='cap on test-set size used for the sweep')
    ap.add_argument('--n-epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-seeds', type=int, default=30,
                    help='mask seeds per density')
    ap.add_argument('--n-densities', type=int, default=40,
                    help='log-uniform in u = sqrt(s/(1-s))')
    ap.add_argument('--u-min', type=float, default=0.005)
    ap.add_argument('--u-max', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default=DEFAULT_MNIST_DIR)
    ap.add_argument('--output-dir',
                    default='assets/unstructured_pruning/toy_examples/mnist_relu')
    args = ap.parse_args()

    have_latex = _configure_style()
    torch.manual_seed(args.seed)

    print(f'  Loading MNIST from {args.data_dir} ...')
    X_tr, Y_tr, X_te, Y_te = load_mnist(args.data_dir)
    if args.n_train and X_tr.shape[0] > args.n_train:
        X_tr, Y_tr = X_tr[:args.n_train], Y_tr[:args.n_train]
    if args.n_test and X_te.shape[0] > args.n_test:
        X_te, Y_te = X_te[:args.n_test], Y_te[:args.n_test]
    D = X_tr.shape[1]
    C = int(Y_tr.max().item()) + 1
    print(f'  X_tr {tuple(X_tr.shape)}, X_te {tuple(X_te.shape)}, '
          f'D={D}, C={C}')

    u_grid = np.geomspace(args.u_min, args.u_max, args.n_densities)
    densities = (u_grid ** 2) / (1.0 + u_grid ** 2)

    per_L = {}
    for L in args.Ls:
        print(f'  L = {L} (H = {args.H}): training...')
        model = FCReLU(D=D, H=args.H, L=L, C=C)
        train(model, X_tr, Y_tr,
              n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
        with torch.no_grad():
            A_full = float((model(X_te).argmax(1) == Y_te).float().mean())
        print(f'    A_unpruned = {A_full:.4f}')

        print(f'    F41 recursion across {len(densities)} densities...')
        A_F41 = np.zeros_like(densities)
        for i, s in enumerate(densities):
            A_F41[i], _ = f41_recursion(model, X_te, Y_te, float(s))

        print(f'    empirical sweep, n_seeds={args.n_seeds}...')
        A_mean, A_std = sweep(model, X_te, Y_te,
                              densities=densities,
                              n_seeds=args.n_seeds, seed=args.seed)
        per_L[int(L)] = {
            'L': int(L), 'H': args.H, 'D': D, 'C': C,
            'A_unpruned': A_full,
            'densities': densities.tolist(),
            'A_emp_mean': A_mean.tolist(),
            'A_emp_std':  A_std.tolist(),
            'A_F41':      A_F41.tolist(),
        }
        # Quick numerical summary
        sel = (A_mean > 1.0 / C + 0.05) & (A_mean < 0.99)
        if sel.any():
            mbe = float(np.mean(A_mean[sel] - A_F41[sel]))
            mae = float(np.mean(np.abs(A_mean[sel] - A_F41[sel])))
            print(f'    transition window: mean(A_emp - A_F41) = {mbe:+.4f}, '
                  f'mean|.| = {mae:.4f}, N = {int(sel.sum())}')

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({str(L): v for L, v in per_L.items()}, f, indent=2)
    png = render(per_L, args.output_dir, have_latex)
    print(f'\nSaved figure: {png}')
    print(f'Saved data:   {os.path.join(args.output_dir, "results.json")}')


if __name__ == '__main__':
    main()
