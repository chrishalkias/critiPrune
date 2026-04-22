#!/usr/bin/env python3
"""Unstructured pruning scaling scan on CIFAR-10 with frozen ResNet18 features (512-d).

Reuses ``pruning.cifar_scaling.load_cifar10`` which extracts & caches ResNet18
features from raw CIFAR-10.
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

from pruning.cifar_scaling import load_cifar10, FEATURE_DIM  # noqa: E402
from unstructured_pruning.core import run_scaling_experiment, DEFAULT_DENSITIES  # noqa: E402
from unstructured_pruning.methods import UNSTRUCTURED_METHODS  # noqa: E402


H_VALUES = [64, 128, 256, 512]
L_VALUES = [2, 3, 5, 7, 10]
SEED = 42


def main(method='random', output_dir=None):
    if method not in UNSTRUCTURED_METHODS:
        raise SystemExit(f"unknown method '{method}'")
    if output_dir is None:
        output_dir = f'unstructured_figures_cifar_resnet_{method}'

    np.random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 70)
    print(f"  UNSTRUCTURED PRUNING — CIFAR-10 + ResNet18 — "
          f"{UNSTRUCTURED_METHODS[method]}")
    print("=" * 70)

    data = load_cifar10()

    run_scaling_experiment(
        data,
        input_size=FEATURE_DIM,
        h_values=H_VALUES,
        l_values=L_VALUES,
        method=method,
        output_dir=output_dir,
        dataset_label='CIFAR-10 + ResNet18',
        epochs_fn=lambda H, L: 300,
        bs=128, lr=1e-3, n_seeds=3, seed=SEED, val_acc_floor=0.15,
    )
    print("  Done!")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='random',
                    choices=list(UNSTRUCTURED_METHODS))
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()
    main(method=args.method, output_dir=args.output_dir)
