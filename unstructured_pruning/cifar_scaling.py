#!/usr/bin/env python3
"""Unstructured pruning scaling scan on CIFAR-10 with PCA-reduced raw pixels.

Loads CIFAR-10 via the same fallback chain as ``pruning.cifar_scaling`` (torchvision
or raw download), flattens to 3072-dim pixel vectors, then reduces to ``PCA_DIM``
components fitted on the training set.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse the raw-loader fallbacks from the existing CIFAR script.
from pruning.cifar_scaling import (  # noqa: E402
    _load_cifar10_torchvision, _load_cifar10_raw,
)
from unstructured_pruning.core import run_scaling_experiment, DEFAULT_DENSITIES  # noqa: E402
from unstructured_pruning.methods import UNSTRUCTURED_METHODS  # noqa: E402


PCA_DIM = 200
TEST_SIZE = 1500
SEED = 42

H_VALUES = [64, 96, 128, 192, 256, 384, 512]
L_VALUES = [2, 3, 4, 5, 6, 7, 8, 10]


def load_cifar_pca():
    """Load CIFAR-10, flatten to pixels, reduce to ``PCA_DIM`` dims."""
    X_tr_raw = None
    for loader, name in [(_load_cifar10_torchvision, 'torchvision'),
                         (_load_cifar10_raw, 'raw download')]:
        try:
            X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = loader()
            print(f"  CIFAR-10 loaded via {name}")
            break
        except Exception:
            continue
    if X_tr_raw is None:
        raise RuntimeError("Cannot load CIFAR-10")

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_tr_raw, y_tr_raw, test_size=0.10,
        random_state=SEED, stratify=y_tr_raw)

    rng = np.random.default_rng(SEED)
    te_idx = rng.choice(len(y_te_raw),
                        size=min(TEST_SIZE, len(y_te_raw)), replace=False)
    X_te_raw, y_te = X_te_raw[te_idx], y_te_raw[te_idx]

    # Flatten HWC → 3072-d float64, standardise in pixel space
    def flat(X): return X.reshape(X.shape[0], -1).astype(np.float64) / 255.0
    X_tr_flat  = flat(X_tr_raw)
    X_val_flat = flat(X_val_raw)
    X_te_flat  = flat(X_te_raw)

    sc = StandardScaler(with_mean=True, with_std=True)
    X_tr_flat  = sc.fit_transform(X_tr_flat)
    X_val_flat = sc.transform(X_val_flat)
    X_te_flat  = sc.transform(X_te_flat)

    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_tr  = pca.fit_transform(X_tr_flat)
    X_val = pca.transform(X_val_flat)
    X_te  = pca.transform(X_te_flat)
    print(f"  PCA({PCA_DIM}) fitted  "
          f"explained_var={pca.explained_variance_ratio_.sum():.3f}")
    print(f"  Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  "
          f"Test: {X_te.shape[0]}")

    return X_tr, X_val, X_te, y_tr, y_val, y_te


def main(method='random', output_dir=None, n_repeats=1):
    if method not in UNSTRUCTURED_METHODS:
        raise SystemExit(f"unknown method '{method}'")
    if output_dir is None:
        output_dir = f'assets/unstructured_pruning/unstructured_figures_cifar_pca_{method}'

    np.random.seed(SEED); torch.manual_seed(SEED)
    print("=" * 70)
    print(f"  UNSTRUCTURED PRUNING — CIFAR-10 + PCA({PCA_DIM}) — "
          f"{UNSTRUCTURED_METHODS[method]}")
    print("=" * 70)

    data = load_cifar_pca()

    run_scaling_experiment(
        data,
        input_size=PCA_DIM,
        h_values=H_VALUES,
        l_values=L_VALUES,
        method=method,
        output_dir=output_dir,
        dataset_label=f'CIFAR-10 + PCA({PCA_DIM})',
        epochs_fn=lambda H, L: 500,
        bs=128, lr=1e-3, n_seeds=3, n_repeats=n_repeats,
        seed=SEED, val_acc_floor=0.15,
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
