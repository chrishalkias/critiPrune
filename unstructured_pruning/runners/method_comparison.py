#!/usr/bin/env python3
"""Compare unstructured pruning methods on one representative architecture.

For a single (dataset, H, L) cell this sweeps the accuracy recovery curve
``A(s)`` (test accuracy vs retention/density ``s``) for every method in
``methods.UNSTRUCTURED_METHODS``, averaging over the available checkpoint
repeats and over mask seeds (for the stochastic methods). Results are written
to a JSON consumed by :func:`plot_method_comparison`, which renders the
seaborn ``timeseries_facets`` style figure (one panel per method, each
highlighting its own curve against the faint ensemble of all methods).

The trained weights are method-agnostic, so a single checkpoint family (the
``_random`` directory) supplies the models; the 12 pruning methods are all
applied post-hoc to those same nets.

Run (local, fast for sklearn; mnist28 is the paper target)::

    .venv/bin/python -m unstructured_pruning.runners.method_comparison \\
        --dataset mnist28 --H 192 --L 5

then plot::

    .venv/bin/python -m unstructured_pruning.runners.method_comparison \\
        --dataset mnist28 --H 192 --L 5 --plot-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pruning.pruning import FCNetwork                          # noqa: E402
from unstructured_pruning.core import evaluate_masked_accuracy  # noqa: E402
from unstructured_pruning.methods import (                      # noqa: E402
    UNSTRUCTURED_METHODS, build_masks,
)

CHECKPOINT_BASE = 'unstructured_pruning/checkpoints'
OUTPUT_BASE     = 'assets/unstructured_pruning/method_comparison'

# Dense density grid -> smooth A(s) curves.
DEFAULT_DENSITIES = [round(v, 3) for v in np.linspace(0.02, 1.0, 25)]


def _load_dataset(dataset):
    if dataset == 'mnist28':
        from pruning.mnist28_scaling import load_mnist28
        return load_mnist28()
    if dataset == 'sklearn':
        from pruning.mnist_scaling import load_data
        return load_data()
    if dataset == 'cifar_resnet':
        from pruning.cifar_scaling import load_cifar10
        return load_cifar10()
    if dataset == 'cifar_pca':
        from unstructured_pruning.runners.cifar_scaling import load_cifar_pca
        return load_cifar_pca()
    raise ValueError(f'unknown dataset: {dataset}')


def _checkpoint_paths(dataset, H, L):
    """All repeats of the (H, L) cell from the method-agnostic _random family."""
    d = os.path.join(CHECKPOINT_BASE, f'unstructured_figures_{dataset}_random')
    paths = sorted(glob.glob(os.path.join(d, f'H{H}_L{L}_r*.pt')))
    if not paths:
        raise SystemExit(f'no checkpoints for H{H}_L{L} under {d}')
    return paths


def run_sweep(dataset, H, L, densities, n_seeds, calib_n, seed, out_path):
    """Sweep A(s) for every method; average over repeats and mask seeds."""
    X_tr, X_val, X_te, y_tr, y_val, y_te = _load_dataset(dataset)
    X_tr = np.asarray(X_tr); y_tr = np.asarray(y_tr)
    X_te = np.asarray(X_te); y_te = np.asarray(y_te)
    calib_X = X_tr[:calib_n]
    calib_y = y_tr[:calib_n]

    ckpts = _checkpoint_paths(dataset, H, L)
    print(f'  {dataset} H{H} L{L}: {len(ckpts)} repeat(s), '
          f'{len(densities)} densities, {len(UNSTRUCTURED_METHODS)} methods')

    # method -> density -> list of per-repeat mean accuracies
    acc = {m: {float(s): [] for s in densities} for m in UNSTRUCTURED_METHODS}
    normal_accs = []
    t0 = time.time()

    for ckpt_path in ckpts:
        c = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model = FCNetwork(**c['arch'])
        model.load_state_dict(c['state_dict'])
        model.eval()

        for method in UNSTRUCTURED_METHODS:
            mask_sets = build_masks(method, model, densities,
                                    X_calib=calib_X, y_calib=calib_y,
                                    n_seeds=n_seeds, base_seed=seed)
            accs, normal = evaluate_masked_accuracy(model, X_te, y_te, mask_sets)
            for s in densities:
                acc[method][float(s)].append(accs[float(s)][0])
        normal_accs.append(normal)

    methods_out = {}
    for method in UNSTRUCTURED_METHODS:
        means = [float(np.mean(acc[method][float(s)])) for s in densities]
        stds  = [float(np.std(acc[method][float(s)])) for s in densities]
        methods_out[method] = {
            'label': UNSTRUCTURED_METHODS[method],
            'mean':  means,
            'std':   stds,
        }

    rec = {
        'dataset': dataset, 'H': int(H), 'L': int(L),
        'densities': [float(s) for s in densities],
        'normal_acc': float(np.mean(normal_accs)),
        'n_repeats': len(ckpts),
        'n_seeds': int(n_seeds),
        'methods': methods_out,
        'wall_time_s': time.time() - t0,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(rec, f, indent=2)
    print(f'  wrote {out_path}  ({rec["wall_time_s"]:.1f}s)')
    return rec


def plot_method_comparison(rec, out_path):
    """seaborn ``timeseries_facets`` style: one panel per method.

    Each panel draws every method's A(s) faintly in grey, then highlights the
    panel's own method in colour, so the reader can place each method against
    the whole field at a glance.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style='dark')

    densities = np.array(rec['densities'])
    methods = list(rec['methods'])
    # Long-form frame for seaborn.
    rows = []
    for m in methods:
        for s, a in zip(densities, rec['methods'][m]['mean']):
            rows.append({'s': float(s), 'acc': a * 100.0,
                         'method': rec['methods'][m]['label'], 'key': m})
    df = pd.DataFrame(rows)
    labels = [rec['methods'][m]['label'] for m in methods]

    pal = sns.color_palette('husl', len(methods))
    g = sns.relplot(
        data=df, x='s', y='acc',
        col='method', hue='method',
        kind='line', palette=pal,
        col_wrap=4, height=2.4, aspect=1.25,
        linewidth=2.2, zorder=5,
        col_order=labels, hue_order=labels,
        legend=False,
    )

    # Faint grey ensemble of all curves behind each panel's highlighted line.
    for method_label, ax in g.axes_dict.items():
        for other in labels:
            sub = df[df['method'] == other]
            ax.plot(sub['s'], sub['acc'], color='.7', lw=0.8,
                    alpha=0.5, zorder=1)
        ax.text(.04, .12, method_label, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='center')

    # Reference: unpruned accuracy.
    normal = rec['normal_acc'] * 100.0
    for ax in g.axes.flat:
        ax.axhline(normal, color='k', ls=':', lw=0.8, alpha=0.6, zorder=2)

    g.set_titles('')
    g.set_axis_labels('density $s$', 'test accuracy (%)')
    g.set(xlim=(0, 1))
    g.figure.suptitle(
        f'Pruning-method accuracy recovery $A(s)$ — '
        f'{rec["dataset"]} (H={rec["H"]}, L={rec["L"]}), '
        f'mean of {rec["n_repeats"]} repeat(s); dotted = unpruned',
        y=1.02, fontsize=12)
    g.figure.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(g.figure)
    print(f'  saved {out_path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='mnist28',
                    choices=['mnist28', 'sklearn', 'cifar_resnet', 'cifar_pca'])
    ap.add_argument('--H', type=int, default=192)
    ap.add_argument('--L', type=int, default=5)
    ap.add_argument('--densities', type=float, nargs='+', default=DEFAULT_DENSITIES)
    ap.add_argument('--n-seeds', type=int, default=3,
                    help='mask realisations for stochastic methods')
    ap.add_argument('--calib-n', type=int, default=512,
                    help='calibration batch size for saliency/activation methods')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', default=OUTPUT_BASE)
    ap.add_argument('--torch-threads', type=int, default=None)
    ap.add_argument('--plot-only', action='store_true',
                    help='skip the sweep; replot from the existing JSON')
    args = ap.parse_args()

    if args.torch_threads:
        torch.set_num_threads(int(args.torch_threads))

    tag = f'{args.dataset}_H{args.H}_L{args.L}'
    json_path = os.path.join(args.output_dir, f'{tag}.json')
    fig_path  = os.path.join(args.output_dir, f'{tag}.png')

    if args.plot_only:
        with open(json_path) as f:
            rec = json.load(f)
    else:
        rec = run_sweep(args.dataset, args.H, args.L, args.densities,
                        args.n_seeds, args.calib_n, args.seed, json_path)

    plot_method_comparison(rec, fig_path)


if __name__ == '__main__':
    main()
