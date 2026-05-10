"""CLI entry: temperature/pruning sweep on a chosen dataset.

Sweeps the (sigma, density) grid with random Bernoulli pruning on the
``unstructured_figures_{dataset}_random`` checkpoint family and writes
results.json + the three diagnostic figures.

Usage:

    python -m temperature_pruning.main --dataset sklearn
    python -m temperature_pruning.main --dataset mnist28
    python -m temperature_pruning.main --dataset cifar_resnet

Smoke test (single cell, three sigmas):

    python -m temperature_pruning.main --dataset sklearn \\
        --h-values 32 --l-values 3 --sigmas 0.0 0.05 0.1 \\
        --n-mask-seeds 2 --n-noise-seeds 2 \\
        --output-dir temperature_pruning/figures/smoke
"""

from __future__ import annotations

import argparse
import json
import os

from unstructured_pruning.core import DEFAULT_DENSITIES

from .core import DEFAULT_SIGMAS, run_temperature_pruning_experiment
from .plots import make_all_plots


_CKPT_BASE = 'unstructured_pruning/checkpoints'


def _load_sklearn():
    from pruning.pruning import load_digits_data
    return load_digits_data()


def _load_mnist28():
    from pruning.mnist28_scaling import load_mnist28
    return load_mnist28()


def _load_cifar_resnet():
    """Test-only ResNet18 features for CIFAR-10.

    The temperature/pruning runner only needs the test split (no retraining
    happens here). Extracting features for the full 54k-image train set on
    CPU takes ~45 min because ``extract_cnn_features`` upsamples 32->224 and
    runs ResNet18 forward in batches of 256. Instead, we extract features for
    only the test split plus a 3000-image train subsample used to fit the
    StandardScaler -- enough samples for stable per-feature mean/std on the
    512-dim embedding (within ~1% of the full-train scaler).
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from pruning.cifar_scaling import (
        _load_cifar10_torchvision, _load_cifar10_raw, extract_cnn_features,
    )

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
        raise RuntimeError("Cannot load CIFAR-10 (need torchvision or internet).")

    rng = np.random.default_rng(42)
    tr_idx = rng.choice(len(y_tr_raw), size=min(3000, len(y_tr_raw)), replace=False)
    te_idx = rng.choice(len(y_te_raw), size=min(1500, len(y_te_raw)), replace=False)

    X_tr_sub, y_tr = X_tr_raw[tr_idx], y_tr_raw[tr_idx]
    X_te_sub, y_te = X_te_raw[te_idx], y_te_raw[te_idx]

    X_tr = extract_cnn_features(X_tr_sub, 'train_sub3k').astype(np.float64)
    X_te = extract_cnn_features(X_te_sub, 'test_sub1500').astype(np.float64)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    # Val placeholder = test (the runner only consumes the test split).
    print(f"  ResNet18 -> 512-dim | Train(scaler-fit): {X_tr.shape[0]}  "
          f"Test: {X_te.shape[0]}")
    return X_tr, X_te, X_te, y_tr, y_te, y_te


# Per-dataset registry: loader, checkpoint directory, default (H, L) grid,
# default checkpoint repeat to use. Grids chosen so every (H, L, repeat) cell
# actually has a checkpoint on disk (verified manually).
DATASETS = {
    'sklearn': {
        'loader': _load_sklearn,
        'ckpt_dir': f'{_CKPT_BASE}/unstructured_figures_sklearn_random',
        'default_h': [16, 32, 64],
        'default_l': [2, 3, 5],
        'default_repeat': 2,
        'output_subdir': 'sklearn_digits',
    },
    'mnist28': {
        'loader': _load_mnist28,
        'ckpt_dir': f'{_CKPT_BASE}/unstructured_figures_mnist28_random',
        'default_h': [128, 192, 256],
        'default_l': [3, 5, 7],
        'default_repeat': 2,
        'output_subdir': 'mnist28',
    },
    'cifar_resnet': {
        'loader': _load_cifar_resnet,
        'ckpt_dir': f'{_CKPT_BASE}/unstructured_figures_cifar_resnet_random',
        'default_h': [160, 192, 224],
        'default_l': [3, 5, 7],
        'default_repeat': 0,
        'output_subdir': 'cifar_resnet',
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=sorted(DATASETS), default='sklearn',
                   help='Which dataset/checkpoint family to evaluate.')
    p.add_argument('--ckpt-dir', default=None,
                   help='Override checkpoint directory (default: per-dataset).')
    p.add_argument('--output-dir', default=None,
                   help='Override output directory (default: per-dataset).')
    p.add_argument('--h-values', type=int, nargs='+', default=None,
                   help='Hidden-size grid (default: per-dataset).')
    p.add_argument('--l-values', type=int, nargs='+', default=None,
                   help='Depth grid (default: per-dataset).')
    p.add_argument('--sigmas', type=float, nargs='+',
                   default=DEFAULT_SIGMAS,
                   help='Temperature grid (Gaussian noise std, RMS-scaled).')
    p.add_argument('--densities', type=float, nargs='+',
                   default=DEFAULT_DENSITIES,
                   help='Pruning density grid.')
    p.add_argument('--repeat-ids', type=int, nargs='+', default=None,
                   help='Which checkpoint repeats (default: per-dataset).')
    p.add_argument('--n-mask-seeds', type=int, default=3)
    p.add_argument('--n-noise-seeds', type=int, default=3)
    p.add_argument('--n-trials', type=int, default=1,
                   help='Independent (noise, mask) bundles per (cell, sigma). '
                        'Each trial produces one sigmoid fit; the std across '
                        'trial s_0 values becomes the error bar on p_c(sigma).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise-scale', choices=['rms', 'absolute'], default='rms',
                   help='Per-layer RMS-relative or absolute noise std.')
    p.add_argument('--min-r2', type=float, default=0.80,
                   help='Sigmoid R^2 threshold for downstream analysis.')
    p.add_argument('--analysis-only', action='store_true',
                   help='Skip the sweep; re-fit and re-plot from results.json.')
    args = p.parse_args()

    cfg = DATASETS[args.dataset]
    if args.ckpt_dir is None:
        args.ckpt_dir = cfg['ckpt_dir']
    if args.output_dir is None:
        args.output_dir = f"temperature_pruning/figures/{cfg['output_subdir']}"
    if args.h_values is None:
        args.h_values = list(cfg['default_h'])
    if args.l_values is None:
        args.l_values = list(cfg['default_l'])
    if args.repeat_ids is None:
        args.repeat_ids = [cfg['default_repeat']]
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.analysis_only:
        results_path = os.path.join(args.output_dir, 'results.json')
        if not os.path.exists(results_path):
            raise FileNotFoundError(
                f"--analysis-only requested but {results_path} does not exist")
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} rows from {results_path}")
    else:
        cfg = DATASETS[args.dataset]
        print(f"  Loading dataset: {args.dataset}...")
        data = cfg['loader']()
        print(f"  Running temperature/pruning sweep")
        print(f"    H={args.h_values}  L={args.l_values}")
        print(f"    sigmas={args.sigmas}")
        print(f"    densities ({len(args.densities)})={args.densities[:3]}..."
              f"{args.densities[-3:]}")
        results = run_temperature_pruning_experiment(
            data,
            h_values=args.h_values,
            l_values=args.l_values,
            sigmas=args.sigmas,
            densities=args.densities,
            ckpt_dir=args.ckpt_dir,
            output_dir=args.output_dir,
            repeat_ids=tuple(args.repeat_ids),
            n_mask_seeds=args.n_mask_seeds,
            n_noise_seeds=args.n_noise_seeds,
            n_trials=args.n_trials,
            seed=args.seed,
            noise_scale=args.noise_scale,
        )

    print("\n  Generating figures...")
    fits, scores = make_all_plots(results, args.output_dir,
                                  min_r2=args.min_r2)

    summary = {
        'critical_line_fits': {
            f'H{H}_L{L}': {k: v for k, v in f.items() if k != 'sigmas' and k != 'p_cs'}
            for (H, L), f in (fits or {}).items()
        },
        'collapse_scores': {
            f'H{H}_L{L}': float(s['score']) for (H, L), s in (scores or {}).items()
        },
    }
    summary_path = os.path.join(args.output_dir, 'analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    if fits:
        any_fit = next(iter(fits.values()))
        deg = any_fit.get('degree', 1)
        any_restricted = any(f.get('restricted', False) for f in fits.values())
        if deg == 2:
            header = "p_c = a + b*sigma + c*sigma^2"
        elif deg == 1:
            header = "p_c = a + b*sigma"
        else:
            header = f"p_c = polynomial degree {deg}"
        scope = ("F-regime restricted (per cell)" if any_restricted
                 else "no truncation: F regime extends past data")
        print(f"\n  Critical-line fits  {header}  [{scope}]:")
        for (H, L), f in sorted(fits.items()):
            parts = [f"H={H} L={L}:"]
            for i, c in enumerate(f['coeffs']):
                name = ['a', 'b', 'c', 'd'][i] if i < 4 else f'c{i}'
                se = f['coeffs_se'][i]
                parts.append(f"{name}={c:+.4f} +/- {se:.4f}")
            parts.append(f"R2={f['R2']:.3f}  n={f['n']}")
            if 'J0_eff_iter' in f and f['J0_eff_iter'] is not None:
                parts.append(f"J0={f['J0_eff_iter']:.2f}")
            if f.get('restricted'):
                fmax = f.get('sigma_fit_max', f.get('sigma_cutoff'))
                cutoff = f.get('sigma_cutoff')
                parts.append(f"fit_window<= {fmax:.2f}  SG_line={cutoff:.2f}")
            print("    " + "  ".join(parts))


if __name__ == '__main__':
    main()
