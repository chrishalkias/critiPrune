#!/usr/bin/env python3
"""Pool per-cell JSONs from ``input_noise/results_cluster/`` into a single
aggregated JSON for analysis. Light-weight: just concatenates records.

Usage::

    .venv/bin/python -m input_noise.aggregate \\
        --root input_noise/results_cluster \\
        --output input_noise/results_cluster_all.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',   default='input_noise/results_cluster')
    ap.add_argument('--output', default='input_noise/results_cluster_all.json')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, '*', 'H*_L*_r*.json')))
    print(f'  pooling {len(paths)} cell JSONs from {args.root}')
    cells = []
    for p in paths:
        with open(p) as f:
            cells.append(json.load(f))

    by_combo = {}
    for c in cells:
        key = f'{c["dataset"]}_{c["method"]}'
        by_combo[key] = by_combo.get(key, 0) + 1
    print('  per (dataset, method):')
    for k, n in sorted(by_combo.items()):
        print(f'    {k:40s} {n:4d}')

    with open(args.output, 'w') as f:
        json.dump(cells, f, indent=2)
    print(f'\n  Saved {len(cells)} records -> {args.output}')


if __name__ == '__main__':
    main()
