#!/usr/bin/env python3
"""C3: depth extension at L in {3, 4, 5}.

The ALICE cluster sweep already covered L in {3, 4, 5} on sklearn
digits and MNIST-28 with random pruning (see
``assets/input_noise/cluster/findings.md``). We re-use those joint
grids directly — no retraining needed.

For each (dataset, H, L) selection we extract iso-A contours at
A in {0.3, 0.5, 0.7, 0.9} via the same helper used in C1 and report:

  - per-cell rational-curve R^2 at each iso level,
  - whether the depth-breakdown story (deep nets harder than the
    single-layer derivation predicts) is confirmed across the new
    cells.

Writes assets/input_noise/extensions/depth_cells/results.json.

Run::

    .venv/bin/python -m input_noise.extensions.depth_cells.run
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from input_noise.extensions._analysis import analyze_cell    # noqa: E402

CLUSTER_JSON = 'input_noise/results_cluster_all.json'
OUTPUT_JSON  = 'assets/input_noise/extensions/depth_cells/results.json'
ISO_LEVELS   = [0.30, 0.50, 0.70, 0.90]

# 2-3 cells per L per dataset. Sklearn is cheaper and has L=2 baselines
# in the pilot; mnist28 provides scale for the deeper-net comparison.
# Each (dataset, H, L) -> one representative cell from cluster repeats.
SELECTIONS = [
    # sklearn digits: H in {32, 64, 96}, L in {3, 4, 5}
    ('sklearn', 'random', 32, 3),
    ('sklearn', 'random', 64, 3),
    ('sklearn', 'random', 96, 3),
    ('sklearn', 'random', 32, 4),
    ('sklearn', 'random', 64, 4),
    ('sklearn', 'random', 96, 4),
    ('sklearn', 'random', 32, 5),
    ('sklearn', 'random', 64, 5),
    ('sklearn', 'random', 96, 5),
    # MNIST-28: H in {128, 256}, L in {3, 4, 5}
    ('mnist28', 'random', 128, 3),
    ('mnist28', 'random', 256, 3),
    ('mnist28', 'random', 128, 4),
    ('mnist28', 'random', 256, 4),
    ('mnist28', 'random', 128, 5),
    ('mnist28', 'random', 256, 5),
]


def pick_cell(cells, dataset, method, H, L):
    """Return the lowest-repeat cell matching (dataset, method, H, L), or
    None if no match."""
    matches = [c for c in cells if (c['dataset'], c['method'], c['H'], c['L'])
               == (dataset, method, H, L)]
    if not matches:
        return None
    matches.sort(key=lambda c: c['repeat'])
    return matches[0]


def main():
    print(f'  loading {CLUSTER_JSON} ...')
    with open(CLUSTER_JSON) as f:
        all_cells = json.load(f)
    print(f'  {len(all_cells)} cells loaded')

    per_cell = []
    for (ds, method, H, L) in SELECTIONS:
        cell = pick_cell(all_cells, ds, method, H, L)
        if cell is None:
            print(f'  WARN: no cluster cell for {ds} {method} H={H} L={L}')
            continue
        analyzed = analyze_cell(cell['joint'], cell['x2_mean'], ISO_LEVELS)
        per_cell.append({
            'dataset': ds,
            'H':       int(H),
            'L':       int(L),
            'repeat':  int(cell['repeat']),
            'x2_mean': float(cell['x2_mean']),
            'val_acc': float(cell['val_acc']),
            'per_iso': {f'{lvl:.2f}': analyzed[lvl] for lvl in ISO_LEVELS},
        })
        line = f'  {ds:8s} H={H:3d} L={L}: '
        for lvl in ISO_LEVELS:
            r = analyzed[lvl]
            if np.isfinite(r['R2']) and r['n'] >= 2:
                line += f' A={lvl:.1f}:R2={r["R2"]:+.2f}(n={r["n"]})'
            else:
                line += f' A={lvl:.1f}:--'
        print(line)

    # Verdict prep: R^2 summary at A = 0.5 across new cells.
    r2_at_half = [rec['per_iso']['0.50']['R2'] for rec in per_cell
                  if rec['per_iso']['0.50']['n'] >= 2
                  and np.isfinite(rec['per_iso']['0.50']['R2'])]
    if r2_at_half:
        print(f'\n  A=0.5 R^2 across {len(r2_at_half)} depth cells: '
              f'min={min(r2_at_half):.3f}, median={float(np.median(r2_at_half)):.3f}, '
              f'max={max(r2_at_half):.3f}, mean={float(np.mean(r2_at_half)):.3f}')

    # Also: per-L median R^2 to show the depth trend explicitly.
    by_L = {}
    for rec in per_cell:
        L = rec['L']
        r = rec['per_iso']['0.50']
        if r['n'] >= 2 and np.isfinite(r['R2']):
            by_L.setdefault(L, []).append(r['R2'])
    print('\n  median R^2 at A=0.5 by depth L:')
    for L in sorted(by_L):
        rs = by_L[L]
        print(f'    L={L}: n={len(rs)}  median R^2 = {float(np.median(rs)):.3f}  '
              f'[min={min(rs):.3f}, max={max(rs):.3f}]')

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({'iso_levels':     ISO_LEVELS,
                   'per_cell':       per_cell,
                   'r2_at_half':     {
                       'min':    float(min(r2_at_half)) if r2_at_half else None,
                       'median': float(np.median(r2_at_half)) if r2_at_half else None,
                       'max':    float(max(r2_at_half)) if r2_at_half else None,
                       'mean':   float(np.mean(r2_at_half)) if r2_at_half else None,
                       'n':      int(len(r2_at_half)),
                   },
                   'by_L_median_R2': {
                       int(L): float(np.median(rs)) for L, rs in by_L.items()
                   }}, f, indent=2)
    print(f'\n  -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
