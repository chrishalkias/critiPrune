#!/usr/bin/env python3
r"""Regenerate ``scaling_curves.png`` and ``s0_scaling.png`` from the cached
per-cell JSON, without retraining.

Each ``assets/unstructured_pruning/unstructured_figures_<dataset>_<method>/``
directory holds ``scaling_results.json`` (per-cell sigmoid fits + raw
recovery curves) and ``scaling_laws.json`` (bivariate power-law fits).
Both PNGs are produced by ``unstructured_pruning.core.make_plots``,
which reads only those two JSONs -- so after a refit
(``tools/refit_sigmoids.py``) we can refresh the figures without touching
``torch`` or the cached checkpoints.

Usage
-----
    .venv/bin/python -m unstructured_pruning.replot_from_json
    .venv/bin/python -m unstructured_pruning.replot_from_json --base FIGS_DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os

# matplotlib must use the non-interactive backend before make_plots imports it.
import matplotlib
matplotlib.use('Agg')

from .core import make_plots


def _safe_label(token: str) -> str:
    """Render a ``<dataset>_<method>`` token in a human-friendly form."""
    pretty = {
        'sklearn':   'sklearn digits',
        'mnist28':   'MNIST-28',
        'cifar':     'CIFAR-10',
        'pca':       'PCA',
        'resnet':    'ResNet',
        'random':    'random',
        'magnitude': 'magnitude',
        'wanda':     'WANDA',
    }
    return ' '.join(pretty.get(p, p) for p in token.split('_'))


def main(base='assets/unstructured_pruning'):
    dirs = sorted(glob.glob(os.path.join(base, 'unstructured_figures_*')))
    if not dirs:
        print(f'No unstructured_figures_* directories under {base}')
        return

    written, skipped = [], []
    for d in dirs:
        results_p = os.path.join(d, 'scaling_results.json')
        scaling_p = os.path.join(d, 'scaling_laws.json')
        if not os.path.isfile(results_p):
            skipped.append((d, 'no scaling_results.json'))
            continue
        with open(results_p) as f:
            results = json.load(f)
        scaling = (json.load(open(scaling_p))
                   if os.path.isfile(scaling_p) else None)
        token = os.path.basename(d).replace('unstructured_figures_', '')
        paths = make_plots(results, scaling, d, title_prefix=_safe_label(token))
        if paths:
            written.extend(paths)
        else:
            skipped.append((d, 'make_plots produced no figures'))

    print(f'\nDone. {len(written)} figures written, {len(skipped)} skipped.')
    for p in written:
        print(f'  wrote: {p}')
    for d, why in skipped:
        print(f'  skipped: {os.path.basename(d)}  ({why})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='assets/unstructured_pruning',
                    help='directory containing unstructured_figures_* subdirs')
    args = ap.parse_args()
    main(base=args.base)
