#!/usr/bin/env python3
"""Cluster-scale input-noise iso-accuracy sweep.

Walks every saved FCNetwork checkpoint under
``checkpoints/<dataset>_<method>/``,
loads the trained weights, and runs:

  - a 1-D ``A(sigma; s=1)`` input-noise sweep,
  - a joint ``(s, sigma)`` grid sweep (random Bernoulli mask + Gaussian
    input noise),

then saves one JSON per cell. Resumable: if the output JSON already
exists, the cell is skipped.

The pruning-only ``A(s; sigma=0)`` curve is *not* recomputed --
the existing scaling JSONs already have it. Each cell's output JSON
records ``H, L, repeat, dataset, method`` plus the sweep arrays.

Run::

    .venv/bin/python -m input_noise.runners.cluster_sweep --dataset mnist28 --method random

CLI flags allow tuning the grid; the defaults match
``input_noise/findings.md``'s pilot for direct comparability.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path / dataset registry
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unstructured_pruning.base.pruning import FCNetwork                          # noqa: E402

from input_noise.core import (                                 # noqa: E402
    evaluate_joint,
    evaluate_noisy_accuracy,
)

CHECKPOINT_BASE = 'checkpoints'
RESULTS_BASE    = 'input_noise/results_cluster'

# (X_tr, X_val, X_te, y_tr, y_val, y_te) per dataset key.
def _load_mnist28():
    from unstructured_pruning.base.mnist28_scaling import load_mnist28
    return load_mnist28()

def _load_cifar_pca():
    from unstructured_pruning.runners.cifar_scaling import load_cifar_pca
    return load_cifar_pca()

def _load_cifar_resnet():
    from unstructured_pruning.base.cifar_scaling import load_cifar10
    return load_cifar10()

def _load_sklearn():
    from unstructured_pruning.base.mnist_scaling import load_data
    return load_data()

DATASET_LOADERS = {
    'mnist28':      _load_mnist28,
    'cifar_pca':    _load_cifar_pca,
    'cifar_resnet': _load_cifar_resnet,
    'sklearn':      _load_sklearn,
}

VALID_METHODS = ('random', 'magnitude', 'wanda')

CKPT_NAME_RE = re.compile(r'^H(\d+)_L(\d+)_r(\d+)\.pt$')


# ---------------------------------------------------------------------------
# Grid configuration (defaults; CLI-overridable)
# ---------------------------------------------------------------------------
SIGMA_GRID_1D = [
    0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30,
    0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00,
]
S_GRID_JOINT = [
    0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.00,
]
SIGMA_GRID_JOINT = [
    0.0, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.30, 1.75, 2.25, 3.00,
]


# ---------------------------------------------------------------------------
# Per-cell processing
# ---------------------------------------------------------------------------
def process_cell(ckpt_path: str, X_te, y_te, args,
                 dataset: str, method: str, H: int, L: int, r: int,
                 output_dir: str):
    """Run input-noise sweeps on one checkpoint and dump a per-cell JSON.

    Idempotent: returns ``None`` if the output file already exists.
    """
    out_path = os.path.join(output_dir, f'H{H}_L{L}_r{r}.json')
    if os.path.exists(out_path):
        return None

    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = FCNetwork(**ckpt['arch'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    x2_mean = float((np.asarray(X_te, dtype=float) ** 2).mean())

    # ---- 1D input-noise sweep (s = 1) ---------------------------------
    n_accs = evaluate_noisy_accuracy(
        model, X_te, y_te, args.sigma_grid_1d,
        n_draws=args.n_draws_1d,
        base_seed=args.seed + 7 * (H * 1000 + L))
    sig_sorted = sorted(n_accs)
    sig_means  = [n_accs[s][0] for s in sig_sorted]
    sig_stds   = [n_accs[s][1] for s in sig_sorted]

    # ---- joint (s, sigma) grid ---------------------------------------
    joint = evaluate_joint(
        model, X_te, y_te, args.s_grid_joint, args.sigma_grid_joint,
        n_mask_seeds=args.n_mask_seeds, n_noise_draws=args.n_noise_draws,
        base_seed=args.seed + 13 * (H * 1000 + L))

    rec = {
        'dataset':    dataset,
        'method':     method,
        'H':          int(H),
        'L':          int(L),
        'repeat':     int(r),
        'val_acc':    float(ckpt.get('val_acc', float('nan'))),
        'train_seed': int(ckpt.get('train_seed', -1)),
        'x2_mean':    x2_mean,
        'D':          int(ckpt['arch']['input_size']),
        'C':          int(ckpt['arch']['num_classes']),
        'noise': {
            'sigma':     [float(s) for s in sig_sorted],
            'accs_mean': sig_means,
            'accs_std':  sig_stds,
            'n_draws':   int(args.n_draws_1d),
        },
        'joint': {
            's_grid':     [float(s)  for s in args.s_grid_joint],
            'sigma_grid': [float(sg) for sg in args.sigma_grid_joint],
            'mean':       [[joint[(float(s), float(sg))][0]
                            for sg in args.sigma_grid_joint]
                           for s in args.s_grid_joint],
            'std':        [[joint[(float(s), float(sg))][1]
                            for sg in args.sigma_grid_joint]
                           for s in args.s_grid_joint],
            'n_mask_seeds':  int(args.n_mask_seeds),
            'n_noise_draws': int(args.n_noise_draws),
        },
        'wall_time_s': time.time() - t0,
    }
    with open(out_path, 'w') as f:
        json.dump(rec, f, indent=2)
    return rec


def list_cells(checkpoint_dir: str):
    """Yield (H, L, r, full_path) for every checkpoint in directory."""
    for p in sorted(glob.glob(os.path.join(checkpoint_dir, '*.pt'))):
        m = CKPT_NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        yield int(m.group(1)), int(m.group(2)), int(m.group(3)), p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_LOADERS))
    ap.add_argument('--method',  required=True, choices=VALID_METHODS)
    ap.add_argument('--checkpoint-base', default=CHECKPOINT_BASE)
    ap.add_argument('--results-base',    default=RESULTS_BASE)
    # Subsetting (mostly for testing).
    ap.add_argument('--max-cells', type=int, default=None)
    ap.add_argument('--cell-stride', type=int, default=1,
                    help='process every N-th cell only (testing)')
    # Sweep knobs.
    ap.add_argument('--sigma-grid-1d',     type=float, nargs='+',
                    default=SIGMA_GRID_1D)
    ap.add_argument('--s-grid-joint',      type=float, nargs='+',
                    default=S_GRID_JOINT)
    ap.add_argument('--sigma-grid-joint',  type=float, nargs='+',
                    default=SIGMA_GRID_JOINT)
    ap.add_argument('--n-draws-1d',     type=int, default=8)
    ap.add_argument('--n-mask-seeds',   type=int, default=3)
    ap.add_argument('--n-noise-draws',  type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--torch-threads', type=int, default=None,
                    help='set torch.set_num_threads(); default = inherited')
    args = ap.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(int(args.torch_threads))

    dirname = f'{args.dataset}_{args.method}'
    ckpt_dir   = os.path.join(args.checkpoint_base, dirname)
    output_dir = os.path.join(args.results_base,    dirname)
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f'checkpoint dir not found: {ckpt_dir}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'  ckpt dir:   {ckpt_dir}')
    print(f'  output dir: {output_dir}')
    print(f'  loading test data for dataset={args.dataset} ...')
    X_tr, X_val, X_te, y_tr, y_val, y_te = DATASET_LOADERS[args.dataset]()
    del X_tr, X_val, y_tr, y_val  # only need test set
    X_te = np.asarray(X_te)
    y_te = np.asarray(y_te)
    print(f'    X_te {X_te.shape}, y_te {y_te.shape}, '
          f'<x^2>={float((X_te.astype(float) ** 2).mean()):.4f}')

    cells = list(list_cells(ckpt_dir))
    cells = cells[::args.cell_stride]
    if args.max_cells is not None:
        cells = cells[:args.max_cells]
    print(f'  found {len(cells)} cells '
          f'(stride={args.cell_stride}, '
          f'max={args.max_cells})')

    t_total = time.time()
    done, skipped = 0, 0
    for i, (H, L, r, ckpt_path) in enumerate(cells):
        out_path = os.path.join(output_dir, f'H{H}_L{L}_r{r}.json')
        if os.path.exists(out_path):
            skipped += 1
            continue
        rec = process_cell(ckpt_path, X_te, y_te, args,
                           args.dataset, args.method, H, L, r,
                           output_dir)
        if rec is None:
            skipped += 1
            continue
        done += 1
        if done % 5 == 0 or i + 1 == len(cells):
            elapsed = time.time() - t_total
            rate = done / max(elapsed, 1e-6)
            print(f'    [{i+1}/{len(cells)}] H={H} L={L} r={r}  '
                  f'took {rec["wall_time_s"]:.1f}s  '
                  f'(done={done}, skipped={skipped}, '
                  f'rate={rate * 60:.1f}/min)')

    elapsed = time.time() - t_total
    print(f'\n  Total: done={done}  skipped={skipped}  '
          f'wall={elapsed:.0f}s  ({elapsed / 60:.1f} min)')


if __name__ == '__main__':
    main()
