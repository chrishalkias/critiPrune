#!/usr/bin/env python3
r"""F41 prediction vs empirical on a (H, L) grid.

Trains a fully-connected ReLU network for every (H, L) combination on
the chosen dataset, applies random Bernoulli(s) masks on each hidden
weight matrix at log-uniform-in-:math:`u = \sqrt{s/(1-s)}` densities,
and compares the empirical accuracy :math:`A(s)` against the
parameter-free F41 prediction computed via the (F22)-(F28) recursion in
:mod:`mnist_relu_multilayer`.

This is the generalised counterpart of ``mnist_relu_multilayer.py``: the
same per-test-example moment propagation, but sweeping over both width
:math:`H` and depth :math:`L` simultaneously so we can map where the
prediction holds, where it breaks, and how.

Datasets
--------
Selected via ``--dataset``. See :data:`DATASETS` below; adding a new
dataset is a one-line registry entry that points at a loader returning
``(X_tr, Y_tr, X_te, Y_te)`` as flat torch tensors.

Pruning method
--------------
``--method`` is currently restricted to ``random``: F41 is derived for
random Bernoulli masks. ``magnitude`` and ``wanda`` would each require a
different theoretical surrogate; the CLI is left in place so future
extensions slot in cleanly.

Usage
-----
    .venv/bin/python -m unstructured_pruning.toy_examples.f41_sweep
    .venv/bin/python -m unstructured_pruning.toy_examples.f41_sweep \
        --dataset mnist --method random \
        --Hs 64 128 256 --Ls 1 2 3 \
        --n-seeds 40 --n-densities 60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import torch
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .mnist_relu_multilayer import (
    DEFAULT_MNIST_DIR,
    FCReLU,
    f41_recursion,
    load_mnist,
    sweep,
    train,
)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
def _mnist_loader(data_dir):
    return load_mnist(data_dir)


def _cifar_pca_loader(data_dir):
    """Wrap ``unstructured_pruning.cifar_scaling.load_cifar_pca`` to the
    four-tuple torch contract used by this sweep. ``data_dir`` is accepted
    for signature uniformity and currently ignored (the underlying loader
    caches under ``/tmp/cifar10``)."""
    del data_dir  # not configurable through the underlying loader
    from unstructured_pruning.cifar_scaling import load_cifar_pca
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_cifar_pca()
    # Fold val back into train: the sweep has no validation concept.
    X_tr = np.concatenate([X_tr, X_val], axis=0)
    y_tr = np.concatenate([y_tr, y_val], axis=0)
    X_tr = torch.from_numpy(X_tr).float()
    X_te = torch.from_numpy(X_te).float()
    Y_tr = torch.from_numpy(np.asarray(y_tr)).long()
    Y_te = torch.from_numpy(np.asarray(y_te)).long()
    return X_tr, Y_tr, X_te, Y_te


DATASETS = {
    'mnist': {
        'loader':   _mnist_loader,
        'data_dir': DEFAULT_MNIST_DIR,
        'name':     'MNIST 28x28',
    },
    'cifar_pca': {
        'loader':   _cifar_pca_loader,
        'data_dir': '/tmp/cifar10',
        'name':     'CIFAR-10 PCA-200',
    },
}


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
def _configure_style():
    have_latex = (shutil.which('latex') is not None
                  and shutil.which('dvipng') is not None)
    plt.rcParams.update({
        'text.usetex':        have_latex,
        'font.family':        'serif',
        'font.serif':         ['Computer Modern Roman', 'DejaVu Serif'],
        'mathtext.fontset':   'cm',
        'axes.labelsize':     10,
        'axes.titlesize':     10,
        'legend.fontsize':    7,
        'figure.dpi':         120,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
        'savefig.pad_inches': 0.08,
    })
    if have_latex:
        plt.rcParams['text.latex.preamble'] = (
            r'\usepackage{amsmath}\usepackage{amssymb}')
    return have_latex


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def render_overlay_grid(cells, output_path, *, have_latex,
                        dataset_name: str, method: str) -> None:
    """One small `A_emp` vs `A_F41` panel per (H, L) cell."""
    Hs = sorted({c['H'] for c in cells})
    Ls = sorted({c['L'] for c in cells})
    n_rows = len(Ls)
    n_cols = len(Hs)
    panel_w = 3.0
    panel_h = 2.3
    fig_w = panel_w * n_cols + 1.4
    fig_h = panel_h * n_rows + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    gs = fig.add_gridspec(n_rows, n_cols,
                          left=0.07, right=0.99,
                          top=1.0 - 0.60 / fig_h,
                          bottom=0.55 / fig_h,
                          wspace=0.22, hspace=0.34)
    cell_map = {(c['H'], c['L']): c for c in cells}
    cmap = plt.cm.viridis
    for i, L in enumerate(Ls):
        for j, H in enumerate(Hs):
            ax = fig.add_subplot(gs[i, j])
            cd = cell_map.get((H, L))
            if cd is None:
                ax.set_xticks([]); ax.set_yticks([])
                continue
            s = np.array(cd['densities'])
            A_emp = np.array(cd['A_emp_mean'])
            A_std = np.array(cd['A_emp_std'])
            A_F41 = np.array(cd['A_F41'])
            C = int(cd['C'])
            ax.errorbar(s, A_emp, yerr=A_std, fmt='o', ms=2.2,
                        color=cmap(0.25), alpha=0.75, capsize=0.0, lw=0.6)
            ax.plot(s, A_F41, color='crimson', lw=1.2)
            ax.axhline(1.0 / C, color='0.5', lw=0.4, linestyle=':')
            ax.axhline(cd['A_unpruned'], color='0.6', lw=0.4, linestyle='-.')
            ax.set_xscale('log')
            ax.set_xlim(min(s), 1.0)
            ax.set_ylim(0.0, 1.02)
            # Tick / label policy: outer panels carry labels, interior ones
            # share axes for compactness.
            if i == n_rows - 1:
                ax.set_xlabel(r'$s$' if have_latex else 's')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(rf'$L = {L}$' if have_latex
                              else f'L = {L}')
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(rf'$H = {H}$' if have_latex
                             else f'H = {H}', fontsize=10)
            ax.grid(True, which='both', alpha=0.25, lw=0.3)
            # Compact text: A_unpruned and mean residual on transition window
            A_arr = A_emp
            sel = (A_arr > 1.0 / C + 0.05) & (A_arr < 0.99)
            mbe = (float(np.mean(A_arr[sel] - A_F41[sel])) if sel.any()
                   else float('nan'))
            txt = ((rf'$A_{{\mathrm{{full}}}}={cd["A_unpruned"]:.3f}$'
                    '\n'
                    rf'$\langle\Delta\rangle={mbe:+.3f}$')
                   if have_latex
                   else (f'A_full={cd["A_unpruned"]:.3f}\n'
                         f'<Δ>={mbe:+.3f}'))
            ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                    ha='left', va='top', fontsize=6.8,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec='0.55', lw=0.4, alpha=0.92))

    fig.suptitle(
        (rf'$A(s)$ empirical (dots) vs parameter-free F41 (line) -- '
         rf'{dataset_name}, {method} pruning'
         if have_latex
         else f'A(s): empirical vs F41 — {dataset_name}, {method} pruning'),
        y=1.0 - 0.14 / fig_h)
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)


def render_heatmap(cells, output_path, *, have_latex,
                   dataset_name: str, method: str) -> None:
    """Two heatmaps over the (H, L) grid: mean signed residual and
    mean absolute residual, both averaged over the A in (1/C + 0.05, 0.99)
    transition window."""
    Hs = sorted({c['H'] for c in cells})
    Ls = sorted({c['L'] for c in cells})
    Hi = {H: i for i, H in enumerate(Hs)}
    Li = {L: i for i, L in enumerate(Ls)}
    mbe = np.full((len(Ls), len(Hs)), np.nan)
    mae = np.full((len(Ls), len(Hs)), np.nan)
    for c in cells:
        s = np.array(c['densities'])
        A_emp = np.array(c['A_emp_mean'])
        A_F41 = np.array(c['A_F41'])
        C = int(c['C'])
        sel = (A_emp > 1.0 / C + 0.05) & (A_emp < 0.99)
        if not sel.any():
            continue
        mbe[Li[c['L']], Hi[c['H']]] = float(np.mean(A_emp[sel] - A_F41[sel]))
        mae[Li[c['L']], Hi[c['H']]] = float(
            np.mean(np.abs(A_emp[sel] - A_F41[sel])))

    fig_w = 3.0 + 1.5 * len(Hs)
    fig_h = max(3.5, 0.9 * len(Ls) + 2.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                   facecolor='white')

    vmax = float(np.nanmax(np.abs(mbe))) if np.isfinite(mbe).any() else 0.05
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 0.05
    im1 = ax1.imshow(mbe, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                     aspect='auto', origin='lower')
    ax1.set_xticks(range(len(Hs))); ax1.set_xticklabels(Hs)
    ax1.set_yticks(range(len(Ls))); ax1.set_yticklabels(Ls)
    ax1.set_xlabel(r'$H$' if have_latex else 'H')
    ax1.set_ylabel(r'$L$' if have_latex else 'L')
    ax1.set_title(r'mean$(A_{\mathrm{emp}} - A_{\mathrm{F41}})$'
                  if have_latex else 'mean(A_emp - A_F41)')
    plt.colorbar(im1, ax=ax1, shrink=0.85)
    for i in range(len(Ls)):
        for j in range(len(Hs)):
            v = mbe[i, j]
            if np.isfinite(v):
                ax1.text(j, i, f'{v:+.3f}', ha='center', va='center',
                         fontsize=8.5, color='black')

    vmax2 = float(np.nanmax(mae)) if np.isfinite(mae).any() else 0.05
    if not np.isfinite(vmax2) or vmax2 == 0.0:
        vmax2 = 0.05
    im2 = ax2.imshow(mae, cmap='Reds', aspect='auto', origin='lower',
                     vmin=0.0, vmax=vmax2)
    ax2.set_xticks(range(len(Hs))); ax2.set_xticklabels(Hs)
    ax2.set_yticks(range(len(Ls))); ax2.set_yticklabels(Ls)
    ax2.set_xlabel(r'$H$' if have_latex else 'H')
    ax2.set_ylabel(r'$L$' if have_latex else 'L')
    ax2.set_title(r'mean$|A_{\mathrm{emp}} - A_{\mathrm{F41}}|$'
                  if have_latex else 'mean|A_emp - A_F41|')
    plt.colorbar(im2, ax=ax2, shrink=0.85)
    for i in range(len(Ls)):
        for j in range(len(Hs)):
            v = mae[i, j]
            if np.isfinite(v):
                ax2.text(j, i, f'{v:.3f}', ha='center', va='center',
                         fontsize=8.5, color='black')

    fig.suptitle(
        (rf'F41 vs empirical residuals -- {dataset_name}, '
         rf'{method} pruning, transition window only'
         if have_latex
         else f'F41 vs empirical residuals — {dataset_name}, '
              f'{method} pruning, transition window only'),
        y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, facecolor='white')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='mnist', choices=list(DATASETS))
    ap.add_argument('--method', default='random', choices=['random'])
    ap.add_argument('--Hs', type=int, nargs='+',
                    default=[64, 128, 256, 512])
    ap.add_argument('--Ls', type=int, nargs='+',
                    default=[1, 2, 3, 4])
    ap.add_argument('--n-train', type=int, default=60000)
    ap.add_argument('--n-test', type=int, default=2500)
    ap.add_argument('--n-epochs', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-seeds', type=int, default=40,
                    help='mask seeds per density (controls error bars)')
    ap.add_argument('--n-densities', type=int, default=70,
                    help='densities sweep size; log-uniform in '
                         'u = sqrt(s/(1-s))')
    ap.add_argument('--u-min', type=float, default=0.005)
    ap.add_argument('--u-max', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()

    have_latex = _configure_style()
    torch.manual_seed(args.seed)

    ds = DATASETS[args.dataset]
    data_dir = args.data_dir or ds['data_dir']
    output_dir = args.output_dir or (
        f'unstructured_pruning/toy_examples/figures/'
        f'sweep_{args.dataset}_{args.method}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'  Dataset: {ds["name"]}  method: {args.method}')
    print(f'  Loading from {data_dir} ...')
    X_tr, Y_tr, X_te, Y_te = ds['loader'](data_dir)
    if args.n_train < X_tr.shape[0]:
        X_tr, Y_tr = X_tr[:args.n_train], Y_tr[:args.n_train]
    if args.n_test < X_te.shape[0]:
        X_te, Y_te = X_te[:args.n_test], Y_te[:args.n_test]
    D = int(X_tr.shape[1])
    C = int(Y_tr.max().item()) + 1
    print(f'  X_tr {tuple(X_tr.shape)}, X_te {tuple(X_te.shape)}, '
          f'D={D}, C={C}')

    u_grid = np.geomspace(args.u_min, args.u_max, args.n_densities)
    densities = (u_grid ** 2) / (1.0 + u_grid ** 2)
    print(f'  Sweep: {len(args.Hs)} widths x {len(args.Ls)} depths = '
          f'{len(args.Hs) * len(args.Ls)} cells, '
          f'{args.n_densities} densities, {args.n_seeds} mask seeds')

    cells = []
    for L in args.Ls:
        for H in args.Hs:
            print(f'  (H={H}, L={L}): training...')
            model = FCReLU(D=D, H=H, L=L, C=C)
            train(model, X_tr, Y_tr,
                  n_epochs=args.n_epochs,
                  batch_size=args.batch_size, lr=args.lr)
            with torch.no_grad():
                A_full = float(
                    (model(X_te).argmax(1) == Y_te).float().mean())
            print(f'    A_unpruned = {A_full:.4f}')

            print(f'    F41 recursion ({len(densities)} densities) ...')
            A_F41 = np.zeros_like(densities)
            for i, s in enumerate(densities):
                A_F41[i], _ = f41_recursion(model, X_te, Y_te, float(s))

            print(f'    empirical sweep, n_seeds={args.n_seeds} ...')
            A_mean, A_std = sweep(model, X_te, Y_te,
                                  densities=densities,
                                  n_seeds=args.n_seeds, seed=args.seed)

            sel = (A_mean > 1.0 / C + 0.05) & (A_mean < 0.99)
            mbe = (float(np.mean(A_mean[sel] - A_F41[sel]))
                   if sel.any() else float('nan'))
            mae = (float(np.mean(np.abs(A_mean[sel] - A_F41[sel])))
                   if sel.any() else float('nan'))
            print(f'    (H={H}, L={L}) -> mean(A_emp-A_F41)={mbe:+.4f}, '
                  f'mean|.|={mae:.4f}, N={int(sel.sum())}')

            cells.append({
                'H': int(H), 'L': int(L), 'D': D, 'C': C,
                'A_unpruned': A_full,
                'densities':  densities.tolist(),
                'A_emp_mean': A_mean.tolist(),
                'A_emp_std':  A_std.tolist(),
                'A_F41':      A_F41.tolist(),
            })

    out_json = os.path.join(output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({'dataset': args.dataset, 'method': args.method,
                   'cells': cells}, f, indent=2)
    print(f'\nSaved data:   {out_json}')

    overlay_path = os.path.join(output_dir, 'overlay.png')
    render_overlay_grid(cells, overlay_path, have_latex=have_latex,
                        dataset_name=ds['name'], method=args.method)
    print(f'Saved figure: {overlay_path}')

    heat_path = os.path.join(output_dir, 'heatmap.png')
    render_heatmap(cells, heat_path, have_latex=have_latex,
                   dataset_name=ds['name'], method=args.method)
    print(f'Saved figure: {heat_path}')


if __name__ == '__main__':
    main()
