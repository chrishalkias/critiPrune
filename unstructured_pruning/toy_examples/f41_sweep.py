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


def _six_tuple_to_four(loader_fn):
    """Adapter from (X_tr, X_val, X_te, y_tr, y_val, y_te) numpy six-tuples
    to the (X_tr, Y_tr, X_te, Y_te) torch four-tuple this sweep expects.
    Folds val back into train (the sweep has no validation concept)."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = loader_fn()
    X_tr = np.concatenate([X_tr, X_val], axis=0)
    y_tr = np.concatenate([y_tr, y_val], axis=0)
    X_tr = torch.from_numpy(np.asarray(X_tr)).float()
    X_te = torch.from_numpy(np.asarray(X_te)).float()
    Y_tr = torch.from_numpy(np.asarray(y_tr)).long()
    Y_te = torch.from_numpy(np.asarray(y_te)).long()
    return X_tr, Y_tr, X_te, Y_te


def _cifar_pca_loader(data_dir):
    """Wrap ``unstructured_pruning.cifar_scaling.load_cifar_pca`` to the
    four-tuple torch contract. ``data_dir`` is accepted for signature
    uniformity and currently ignored (the underlying loader caches under
    ``/tmp/cifar10``)."""
    del data_dir
    from unstructured_pruning.cifar_scaling import load_cifar_pca
    return _six_tuple_to_four(load_cifar_pca)


def _cifar_resnet_loader(data_dir):
    """Wrap ``pruning.cifar_scaling.load_cifar10`` (ResNet18 features at
    FEATURE_DIM = 512). ``data_dir`` is accepted for signature uniformity
    and currently ignored (the underlying loader caches features under
    ``$FEATURE_CACHE_DIR``)."""
    del data_dir
    from pruning.cifar_scaling import load_cifar10
    return _six_tuple_to_four(load_cifar10)


def _digits_loader(data_dir):
    """Wrap ``pruning.mnist_scaling.load_data`` (sklearn digits, 8x8 = 64
    dims). ``data_dir`` is accepted for signature uniformity and ignored
    (sklearn ships the data with the wheel)."""
    del data_dir
    from pruning.mnist_scaling import load_data
    return _six_tuple_to_four(load_data)


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
    'cifar_resnet': {
        'loader':   _cifar_resnet_loader,
        'data_dir': '/tmp/cifar10',
        'name':     'CIFAR-10 ResNet18 features',
    },
    'digits': {
        'loader':   _digits_loader,
        'data_dir': '',
        'name':     'sklearn digits 8x8',
    },
}


# ---------------------------------------------------------------------------
# Critical-point extraction
# ---------------------------------------------------------------------------
def _extract_s_0(densities, A_values, target):
    """Half-transition critical density: the smallest ``s`` for which
    ``A(s) >= target``, computed by linear interpolation on the cached
    ``(densities, A_values)`` curve.

    ``A_values`` is assumed monotone non-decreasing in ``densities`` (true
    for ``A_F41`` by construction and empirically for the random-mask
    ``A_emp`` in this folder). Returns ``nan`` if ``target`` lies outside
    the data's reachable range."""
    densities = np.asarray(densities, dtype=float)
    A_values  = np.asarray(A_values,  dtype=float)
    order = np.argsort(densities)
    s_sorted = densities[order]
    A_sorted = A_values[order]
    if target < A_sorted[0] or target > A_sorted[-1]:
        return float('nan')
    return float(np.interp(target, A_sorted, s_sorted))


def _ensure_s_0_fields(cell):
    """Populate cell['s0_F41'], cell['s0_emp'], cell['target'] if absent.
    Mutates and returns whether anything changed."""
    if 's0_F41' in cell and 's0_emp' in cell and 'target' in cell:
        return False
    target = 0.5 * (float(cell['A_unpruned']) + 1.0 / float(cell['C']))
    cell['target'] = target
    cell['s0_F41'] = _extract_s_0(cell['densities'], cell['A_F41'], target)
    cell['s0_emp'] = _extract_s_0(cell['densities'], cell['A_emp_mean'],
                                  target)
    return True


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


def render_residual_grid(cells, output_path, *, have_latex,
                         dataset_name: str, method: str) -> None:
    """One per-(H, L) panel showing the residual curve
    ``A_emp(s) - A_F41(s)`` as a function of density ``s``, with seed
    error bars on the empirical accuracy. Same layout as the overlay
    grid so panels line up visually with the corresponding A(s) plots.

    A shared y-range across all panels makes magnitudes comparable across
    the grid; a horizontal y = 0 reference line marks where empirical
    matches prediction; the transition window over which mean residuals
    are computed (1/C + 0.05 < A_emp < 0.99) is shaded."""
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

    # Shared y-limit across all panels; floor so tiny residuals stay
    # readable, ceiling so large outliers don't squash small ones.
    max_abs = 0.0
    for c in cells:
        A_emp = np.array(c['A_emp_mean'])
        A_F41 = np.array(c['A_F41'])
        diff = A_emp - A_F41
        if np.isfinite(diff).any():
            max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
    y_lim = max(0.02, 1.1 * max_abs)

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
            diff = A_emp - A_F41
            C = int(cd['C'])
            sel = (A_emp > 1.0 / C + 0.05) & (A_emp < 0.99)
            mbe = (float(np.mean(diff[sel])) if sel.any() else float('nan'))
            mae = (float(np.mean(np.abs(diff[sel]))) if sel.any()
                   else float('nan'))

            # Shade the transition window so the eye knows which points
            # contribute to the window-mean annotations.
            if sel.any():
                s_in = s[sel]
                ax.axvspan(float(s_in.min()), float(s_in.max()),
                           color='0.92', zorder=0)
            ax.axhline(0.0, color='0.4', lw=0.6, linestyle='--', zorder=1)
            ax.errorbar(s, diff, yerr=A_std, fmt='o', ms=2.4,
                        color=cmap(0.25), alpha=0.85, capsize=0.0, lw=0.6,
                        zorder=2)
            # Connect samples to make the trend with s readable.
            ax.plot(s, diff, color=cmap(0.25), lw=0.5, alpha=0.55, zorder=2)
            ax.set_xscale('log')
            ax.set_xlim(min(s), 1.0)
            ax.set_ylim(-y_lim, y_lim)

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
            txt = ((rf'$\langle\Delta\rangle={mbe:+.3f}$'
                    '\n'
                    rf'$\langle|\Delta|\rangle={mae:.3f}$')
                   if have_latex
                   else (f'<Δ>={mbe:+.3f}\n<|Δ|>={mae:.3f}'))
            ax.text(0.03, 0.97, txt, transform=ax.transAxes,
                    ha='left', va='top', fontsize=6.8,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white',
                              ec='0.55', lw=0.4, alpha=0.92))

    fig.suptitle(
        (rf'Residual $A_{{\mathrm{{emp}}}}(s) - A_{{\mathrm{{F41}}}}(s)$ '
         rf'-- {dataset_name}, {method} pruning'
         if have_latex
         else f'Residual A_emp(s) − A_F41(s) — {dataset_name}, '
              f'{method} pruning'),
        y=1.0 - 0.14 / fig_h)
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
    ap.add_argument('--n-densities', type=int, default=200,
                    help='densities sweep size; log-uniform in '
                         'u = sqrt(s/(1-s))')
    # Default range covers s in [0.01, 0.999]; below s=0.01 every cell is
    # already at chance, so cropping there isolates the transition window
    # and lets the same n_densities resolve it ~10x finer.
    ap.add_argument('--u-min', type=float, default=0.1005,
                    help='lower u bound; default 0.1005 = sqrt(0.01/0.99) '
                         'so s_min = 0.01')
    ap.add_argument('--u-max', type=float, default=31.61,
                    help='upper u bound; default 31.61 -> s_max ~ 0.999')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--from-json', default=None,
                    help='skip training; re-render plots from this '
                         'results.json file (uses cached cells only)')
    args = ap.parse_args()

    have_latex = _configure_style()
    torch.manual_seed(args.seed)

    ds = DATASETS[args.dataset]
    data_dir = args.data_dir or ds['data_dir']
    output_dir = args.output_dir or (
        f'assets/unstructured_pruning/toy_examples/'
        f'sweep_{args.dataset}_{args.method}')
    os.makedirs(output_dir, exist_ok=True)

    # ----- re-render from cached JSON ---------------------------------
    if args.from_json is not None:
        with open(args.from_json) as f:
            blob = json.load(f)
        cells = blob['cells']
        mutated = False
        for c in cells:
            mutated |= _ensure_s_0_fields(c)
        if mutated:
            with open(args.from_json, 'w') as f:
                json.dump(blob, f, indent=2)
            print(f'  Back-filled s_0 fields; wrote {args.from_json}')
        print(f'  Re-rendering {len(cells)} cells from {args.from_json}')
        overlay_path = os.path.join(output_dir, 'overlay.png')
        render_overlay_grid(cells, overlay_path, have_latex=have_latex,
                            dataset_name=ds['name'], method=args.method)
        print(f'Saved figure: {overlay_path}')
        residual_path = os.path.join(output_dir, 'residuals.png')
        render_residual_grid(cells, residual_path, have_latex=have_latex,
                             dataset_name=ds['name'], method=args.method)
        print(f'Saved figure: {residual_path}')
        return

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

            target = 0.5 * (A_full + 1.0 / C)
            s0_F41 = _extract_s_0(densities, A_F41, target)
            s0_emp = _extract_s_0(densities, A_mean, target)
            print(f'    s_0(F41)={s0_F41:.4f}, s_0(emp)={s0_emp:.4f}, '
                  f'target={target:.4f}')

            cells.append({
                'H': int(H), 'L': int(L), 'D': D, 'C': C,
                'A_unpruned': A_full,
                'densities':  densities.tolist(),
                'A_emp_mean': A_mean.tolist(),
                'A_emp_std':  A_std.tolist(),
                'A_F41':      A_F41.tolist(),
                'target':     target,
                's0_F41':     s0_F41,
                's0_emp':     s0_emp,
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

    residual_path = os.path.join(output_dir, 'residuals.png')
    render_residual_grid(cells, residual_path, have_latex=have_latex,
                         dataset_name=ds['name'], method=args.method)
    print(f'Saved figure: {residual_path}')


if __name__ == '__main__':
    main()
