#!/usr/bin/env python3
"""Unstructured pruning scaling scan on sklearn digits (8x8, 64-dim).

Thin wrapper: delegates to ``core.run_scaling_experiment``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pruning.mnist_scaling import load_data  # noqa: E402
from unstructured_pruning.core import run_scaling_experiment, DEFAULT_DENSITIES  # noqa: E402
from unstructured_pruning.methods import UNSTRUCTURED_METHODS  # noqa: E402


H_VALUES = [8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96]
L_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEED = 42


def main(method='random', output_dir=None, n_repeats=1):
    if method not in UNSTRUCTURED_METHODS:
        raise SystemExit(f"unknown method '{method}'. "
                         f"Choose from: {list(UNSTRUCTURED_METHODS)}")
    if output_dir is None:
        output_dir = f'assets/unstructured_pruning/unstructured_figures_sklearn_{method}'

    np.random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 70)
    print(f"  UNSTRUCTURED PRUNING — sklearn digits — "
          f"{UNSTRUCTURED_METHODS[method]}")
    print("=" * 70)

    data = load_data()
    print(f"  Train={data[0].shape[0]}  Val={data[1].shape[0]}  Test={data[2].shape[0]}")
    print(f"  Densities ({len(DEFAULT_DENSITIES)}): {DEFAULT_DENSITIES}")

    run_scaling_experiment(
        data,
        input_size=64,
        h_values=H_VALUES,
        l_values=L_VALUES,
        method=method,
        output_dir=output_dir,
        dataset_label='sklearn digits',
        epochs_fn=lambda H, L: 300 if H >= 32 else 500,
        bs=64, lr=1e-3, n_seeds=3, n_repeats=n_repeats,
        seed=SEED, val_acc_floor=0.20,
    )
    print("  Done!")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='random',
                    choices=list(UNSTRUCTURED_METHODS))
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--n-repeats', type=int, default=1,
                    help='independent (train, mask, fit) trials per (H, L)')
    args = ap.parse_args()
    main(method=args.method, output_dir=args.output_dir,
         n_repeats=args.n_repeats)
