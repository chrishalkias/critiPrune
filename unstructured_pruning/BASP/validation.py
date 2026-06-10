#!/usr/bin/env python3
"""Validate BASP against the baseline criteria on trained sklearn-digits cells.

For a handful of trained checkpoints (varying depth L), build masks with each
method across the standard density grid, evaluate frozen test accuracy, fit the
sigmoid, and report the critical density ``s_0`` plus the accuracy retained at
very low density. Lower ``s_0`` (and higher low-density accuracy) is better.

Run from the project root:
    .venv/bin/python -m unstructured_pruning.BASP.validation
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unstructured_pruning.base.mnist_scaling import load_data                          # noqa: E402
from unstructured_pruning.base.pruning import fit_sigmoid                              # noqa: E402
from unstructured_pruning.core import (                              # noqa: E402
    DEFAULT_DENSITIES, evaluate_masked_accuracy, load_fc_checkpoint)
from unstructured_pruning import methods as M                        # noqa: E402

CKPT_DIR = 'checkpoints/sklearn_magnitude'
CELLS = ['H72_L4_r0', 'H72_L6_r0', 'H72_L8_r0', 'H44_L6_r0']
METHODS = ['random', 'magnitude', 'wanda', 'snip', 'lamp', 'basp']
LOW_S = [0.02, 0.05, 0.10]   # report accuracy at these high-sparsity points


def _build(method, model, densities, X_calib, y_calib):
    if method == 'snip':
        return M.snip_masks(model, densities, X_calib, y_calib, n_seeds=1)
    return M.build_masks(method, model, densities,
                         X_calib=X_calib, y_calib=y_calib, n_seeds=3)


def main():
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    X_calib, y_calib = X_tr[:512], y_tr[:512]
    densities = sorted(set(DEFAULT_DENSITIES) | set(LOW_S))

    for cell in CELLS:
        path = os.path.join(CKPT_DIR, f'{cell}.pt')
        if not os.path.exists(path):
            print(f"  (skip {cell}: no checkpoint)")
            continue
        model, ck = load_fc_checkpoint(path)
        arch = ck['arch']
        print('\n' + '=' * 78)
        print(f"  {cell}   H={arch['hidden_size']} L={arch['num_hidden_layers']}"
              f"   val_acc={ck['val_acc']:.3f}")
        print('=' * 78)
        header = (f"  {'method':<12}  {'s_0':>7}  {'beta':>7}  {'R2':>6}  "
                  + '  '.join(f"A(s={s})" for s in LOW_S))
        print(header)
        print('  ' + '-' * (len(header) - 2))

        rows = []
        for method in METHODS:
            masks = _build(method, model, densities, X_calib, y_calib)
            accs, normal_acc = evaluate_masked_accuracy(model, X_te, y_te, masks)
            s_vals = sorted(accs.keys())
            mean = {s: accs[s][0] for s in s_vals}
            popt, perr, r2 = fit_sigmoid(s_vals, mean, normal_acc)
            s0 = popt[2] if popt is not None else float('nan')
            beta = popt[3] if popt is not None else float('nan')
            low = [mean[s] for s in LOW_S]
            rows.append((method, s0, low))
            print(f"  {method:<12}  {s0:>7.3f}  {beta:>7.2f}  {r2:>6.3f}  "
                  + '  '.join(f"{a*100:5.1f}%" for a in low)
                  + ('   <-- unpruned %.1f%%' % (normal_acc * 100)
                     if method == METHODS[0] else ''))

        best = min(rows, key=lambda r: r[1])
        print(f"\n  lowest s_0: {best[0]} (s_0={best[1]:.3f})")


if __name__ == '__main__':
    main()
