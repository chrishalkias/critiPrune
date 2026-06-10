#!/usr/bin/env python3
"""Densify (H, L) scaling grids by training only the *new* intermediate cells.

Delegates to ``unstructured_pruning.core.run_scaling_experiment``, whose
resume logic skips every (H, L, repeat) triple already present in
``<output_dir>/scaling_results.json``. This script defines, per dataset,
additional intermediate H and L values that fill gaps in the existing
coarse grid; passing the merged-and-sorted lists to the runner means
already-trained cells are cheap no-ops and only new (H, L) pairs incur
GPU time.

Usage
-----
    python -m unstructured_pruning.runners.more_combinations \\
        --dataset mnist28 --method random [--n-repeats 1]

Outputs land in ``assets/unstructured_pruning/<dataset>_<method>/``
(same directory as the original scan), so plots and scaling-law fits get
regenerated with the denser grid automatically.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unstructured_pruning.core import run_scaling_experiment  # noqa: E402
from unstructured_pruning.methods import UNSTRUCTURED_METHODS  # noqa: E402


SEED = 42

# Per-dataset config:
#   H_base, L_base : the original coarse grid (matches the existing scaling
#                    scripts; documented here only for clarity / reporting).
#   H_extra, L_extra : intermediate values that densify the grid.  Any (H, L)
#                    triple already on disk is skipped by the runner.
DATASETS = {
    'sklearn': {
        'data_module':  'pruning.mnist_scaling',
        'data_loader':  'load_data',
        'input_size':   64,
        'H_base':   [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96],
        'L_base':   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'H_extra':  [10, 14, 18, 22, 28, 36, 44, 52, 60, 72, 88],
        'L_extra':  [],
        'epochs_fn':     lambda H, L: 300 if H >= 32 else 500,
        'bs': 64,  'lr': 1e-3, 'val_acc_floor': 0.20,
        'dataset_label': 'sklearn digits',
    },
    'mnist28': {
        'data_module':  'pruning.mnist28_scaling',
        'data_loader':  'load_mnist28',
        'input_size':   784,
        'H_base':   [64, 96, 128, 192, 256, 384, 512],
        'L_base':   [2, 3, 4, 5, 6, 7, 8, 10],
        'H_extra':  [80, 112, 160, 224, 320, 448],
        'L_extra':  [9],
        'epochs_fn':     lambda H, L: 300,
        'bs': 256, 'lr': 1e-3, 'val_acc_floor': 0.20,
        'dataset_label': 'MNIST 28x28',
    },
    'cifar_pca': {
        'data_module':  'unstructured_pruning.runners.cifar_scaling',
        'data_loader':  'load_cifar_pca',
        'input_size':   200,  # PCA_DIM in cifar_scaling.py
        'H_base':   [64, 96, 128, 192, 256, 384, 512],
        'L_base':   [2, 3, 4, 5, 6, 7, 8, 10],
        'H_extra':  [80, 112, 160, 224, 320, 448],
        'L_extra':  [9],
        'epochs_fn':     lambda H, L: 500,
        'bs': 128, 'lr': 1e-3, 'val_acc_floor': 0.15,
        'dataset_label': 'CIFAR-10 + PCA(200)',
    },
    'cifar_resnet': {
        'data_module':  'pruning.cifar_scaling',
        'data_loader':  'load_cifar10',
        'input_size':   None,  # filled at runtime from FEATURE_DIM
        'H_base':   [64, 96, 128, 192, 256, 384, 512],
        'L_base':   [2, 3, 4, 5, 6, 7, 8, 10],
        'H_extra':  [80, 112, 160, 224, 320, 448],
        'L_extra':  [9],
        'epochs_fn':     lambda H, L: 300,
        'bs': 128, 'lr': 1e-3, 'val_acc_floor': 0.15,
        'dataset_label': 'CIFAR-10 + ResNet18',
    },
}


def _resolve_input_size(dataset, cfg):
    if cfg['input_size'] is not None:
        return cfg['input_size']
    if dataset == 'cifar_resnet':
        from unstructured_pruning.base.cifar_scaling import FEATURE_DIM
        return FEATURE_DIM
    raise RuntimeError(f"input_size unresolved for dataset {dataset}")


def _load_data(cfg):
    mod = importlib.import_module(cfg['data_module'])
    return getattr(mod, cfg['data_loader'])()


def _existing_pairs(output_dir):
    """Return the set of (H, L) pairs already on disk for this output dir."""
    p = os.path.join(output_dir, 'scaling_results.json')
    if not os.path.isfile(p):
        return set()
    with open(p) as f:
        rows = json.load(f)
    return {(int(r['H']), int(r['L'])) for r in rows}


def main(dataset, method, output_dir=None, n_repeats=1,
         extra_repeats_only=False):
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset '{dataset}'. "
                         f"Choose from: {list(DATASETS)}")
    if method not in UNSTRUCTURED_METHODS:
        raise SystemExit(f"unknown method '{method}'. "
                         f"Choose from: {list(UNSTRUCTURED_METHODS)}")

    cfg = DATASETS[dataset]
    if output_dir is None:
        output_dir = ('assets/unstructured_pruning/'
                      f'{dataset}_{method}')

    H_values = sorted(set(cfg['H_base']) | set(cfg['H_extra']))
    L_values = sorted(set(cfg['L_base']) | set(cfg['L_extra']))

    done = _existing_pairs(output_dir)
    full = {(H, L) for H in H_values for L in L_values}
    base_pairs = {(H, L) for H in cfg['H_base'] for L in cfg['L_base']}
    new_pairs = full - base_pairs
    todo = sorted(full - done)
    new_H = sorted(set(cfg['H_extra']) - set(cfg['H_base']))
    new_L = sorted(set(cfg['L_extra']) - set(cfg['L_base']))

    extra_repeat_cells = new_pairs if extra_repeats_only else None

    np.random.seed(SEED); torch.manual_seed(SEED)
    print('=' * 70)
    print(f'  DENSIFY — {dataset} — {UNSTRUCTURED_METHODS[method]}')
    print('=' * 70)
    print(f'  Output dir       : {output_dir}')
    print(f'  Merged H ({len(H_values):2d}): {H_values}')
    print(f'  Merged L ({len(L_values):2d}): {L_values}')
    print(f'  New H values     : {new_H}')
    print(f'  New L values     : {new_L}')
    print(f'  Cells already done: {len(done)} / {len(full)}')
    print(f'  Cells to train   : {len(todo)}')
    print(f'  n_repeats        : {n_repeats}')
    if extra_repeats_only:
        print(f'  Extra-repeats-only: True  '
              f'({len(new_pairs)} new pairs eligible for r>=1; '
              f'{len(base_pairs)} base pairs locked at r=0)')
    print('=' * 70)

    if not todo:
        print('  Nothing to do — grid is already complete. '
              'Re-running plots only.')

    data = _load_data(cfg)
    input_size = _resolve_input_size(dataset, cfg)

    run_scaling_experiment(
        data,
        input_size=input_size,
        h_values=H_values,
        l_values=L_values,
        method=method,
        output_dir=output_dir,
        dataset_label=cfg['dataset_label'],
        epochs_fn=cfg['epochs_fn'],
        bs=cfg['bs'], lr=cfg['lr'],
        n_seeds=3, n_repeats=n_repeats,
        extra_repeat_cells=extra_repeat_cells,
        seed=SEED, val_acc_floor=cfg['val_acc_floor'],
    )
    print('  Done!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--method',  required=True, choices=list(UNSTRUCTURED_METHODS))
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--n-repeats', type=int, default=1)
    ap.add_argument('--extra-repeats-only', action='store_true',
                    help='Only the *new* (extra) (H, L) cells get r>=1; '
                         'original coarse-grid cells stay at r=0.')
    args = ap.parse_args()
    main(dataset=args.dataset, method=args.method,
         output_dir=args.output_dir, n_repeats=args.n_repeats,
         extra_repeats_only=args.extra_repeats_only)
