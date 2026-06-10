#!/usr/bin/env python3
"""C1: multi-iso-level collapse on the pilot cells.

Re-uses the existing joint (s, sigma) grid in assets/input_noise/pilot/results.json
(no retraining). For each pilot cell, extracts iso-A contours at
A in {0.3, 0.5, 0.7, 0.9}, fits the per-cell rational curve
(eq. 12 of .docs/input_noise.md), and computes the parameter-free
collapse residual RMS(eta - (1 - xi)) per iso level (eq. 14).

Writes assets/input_noise/extensions/iso_levels/results.json.

Run::

    .venv/bin/python -m input_noise.extensions.iso_levels.run
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

PILOT_RESULTS = 'assets/input_noise/pilot/results.json'
OUTPUT_JSON   = 'assets/input_noise/extensions/iso_levels/results.json'
ISO_LEVELS    = [0.30, 0.50, 0.70, 0.90]


def main():
    with open(PILOT_RESULTS) as f:
        cells = json.load(f)
    print(f'  loaded {len(cells)} pilot cells from {PILOT_RESULTS}')

    per_cell = []
    for c in cells:
        analyzed = analyze_cell(c['joint'], c['x2_mean'], ISO_LEVELS)
        per_cell.append({
            'dataset': c['dataset'],
            'H':       int(c['H']),
            'L':       int(c['L']),
            'x2_mean': float(c['x2_mean']),
            'per_iso': {f'{lvl:.2f}': analyzed[lvl] for lvl in ISO_LEVELS},
        })
        line = f'  {c["dataset"]:8s} H={c["H"]:3d} L={c["L"]}: '
        for lvl in ISO_LEVELS:
            r = analyzed[lvl]
            if np.isfinite(r['R2']):
                line += f' A={lvl:.1f}:R2={r["R2"]:.2f}(n={r["n"]})'
            else:
                line += f' A={lvl:.1f}:--'
        print(line)

    # Per-iso-level pooled RMS to y = 1 - x (eq. 14 verdict).
    per_iso_summary = {}
    for lvl in ISO_LEVELS:
        xis, etas = [], []
        for rec in per_cell:
            r = rec['per_iso'][f'{lvl:.2f}']
            if not np.isfinite(r['sigma2_1']) or r['sigma2_1'] <= 0 or r['n'] < 2:
                continue
            s_arr  = np.array([p[0] for p in r['contour']])
            sg_arr = np.array([p[1] for p in r['contour']])
            x2 = rec['x2_mean']
            xis.extend(((1.0 - s_arr) * (1.0 + x2 / r['sigma2_1'])).tolist())
            etas.extend(((sg_arr ** 2) / r['sigma2_1']).tolist())
        if len(xis) == 0:
            per_iso_summary[f'{lvl:.2f}'] = {'n_points': 0,
                                              'rms_to_line': float('nan')}
            continue
        xis = np.asarray(xis); etas = np.asarray(etas)
        rms = float(np.sqrt(np.mean((etas - (1.0 - xis)) ** 2)))
        per_iso_summary[f'{lvl:.2f}'] = {
            'n_points':    int(len(xis)),
            'n_cells':     int(sum(1 for rec in per_cell
                                  if rec['per_iso'][f'{lvl:.2f}']['n'] >= 2)),
            'rms_to_line': rms,
        }
        print(f'  A={lvl}: pooled n={len(xis)} pts, RMS to y=1-x: {rms:.3f}')

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump({'iso_levels': ISO_LEVELS,
                   'per_cell':   per_cell,
                   'pooled':     per_iso_summary}, f, indent=2)
    print(f'\n  -> {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
