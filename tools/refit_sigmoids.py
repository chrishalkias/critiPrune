#!/usr/bin/env python3
r"""Re-fit cached sigmoid parameters using the current ``fit_sigmoid`` cap.

Every per-cell results JSON in this repo stores the raw curve data
(``densities`` and ``accs_mean`` for the s-space subpackages,
``accs`` keyed by K for the K-space subpackages). When we change the
:math:`\beta` upper bound or the initial-guess heuristic in
``pruning.pruning.fit_sigmoid``, we can refresh every fit without
retraining a single network.

This script:

  1. Discovers every results JSON under the repo (or a chosen --root).
  2. For each row, re-calls ``fit_sigmoid`` with the cached curve.
  3. Overwrites only the ``sigmoid_*`` fields in place.
  4. Reports how many cells un-saturate as a result.

Usage
-----
    .venv/bin/python tools/refit_sigmoids.py [--dry-run] [--no-backup] [--root .]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import sys
from typing import List, Tuple

import numpy as np

# Make ``import pruning.pruning`` work no matter where the script is invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pruning.pruning import fit_sigmoid  # noqa: E402


# Match the post-hoc saturation flag used by the plot scripts (BETA_CAP - 0.5).
NEW_BETA_CAP = 200.0
SAT_BEFORE = 19.5    # rows with beta >= this were saturated under the old cap
SAT_AFTER = NEW_BETA_CAP - 0.5


def _discover(root: str) -> List[Tuple[str, str]]:
    """Return ``(json_path, kind)`` for every results file we know about.

    ``kind`` is ``"s"`` (s-space; ``densities`` + ``accs_mean`` arrays) or
    ``"K"`` (K-space; ``accs`` dict keyed by K).
    """
    found = []
    patterns_s = [
        os.path.join(root, 'unstructured_pruning', 'figures',
                     'unstructured_figures_*', 'scaling_results.json'),
        os.path.join(root, 'temperature_pruning', 'figures', '*',
                     'results.json'),
    ]
    patterns_k = [
        os.path.join(root, 'mnist_figures', 'scaling_results.json'),
        os.path.join(root, 'cifar_figures', 'scaling_results.json'),
    ]
    for p in patterns_s:
        for m in sorted(glob.glob(p)):
            found.append((m, 's'))
    for p in patterns_k:
        for m in sorted(glob.glob(p)):
            found.append((m, 'K'))
    return found


def _refit_row(row: dict, kind: str) -> Tuple[dict | None, bool]:
    """Return ``(updated_row, ok)``. ``updated_row`` is None on fit failure."""
    if kind == 's':
        xs = list(row.get('densities') or [])
        ys = list(row.get('accs_mean') or [])
        if not xs or len(xs) != len(ys):
            return None, False
        K0_key = 'sigmoid_s_0'
    else:
        accs = row.get('accs') or {}
        if not accs:
            return None, False
        # JSON keys are strings; sort numerically.
        items = sorted(((float(k), float(v)) for k, v in accs.items()),
                       key=lambda kv: kv[0])
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        K0_key = 'sigmoid_K_0' if 'sigmoid_K_0' in row else 'sigmoid_s_0'

    normal_acc = float(row.get('normal_acc') or row.get('val_acc') or max(ys))

    # fit_sigmoid expects accuracies as a mapping keyed by each k value.
    acc_map = dict(zip(xs, ys))
    popt, _perr, r2 = fit_sigmoid(xs, acc_map, normal_acc)
    if popt is None:
        return None, False

    A_inf, A_0, K_0, beta = (float(x) for x in popt)
    row = dict(row)  # shallow copy so we never mutate the caller's mapping
    row['sigmoid_A_inf'] = A_inf
    row['sigmoid_A_0']   = A_0
    row[K0_key]          = K_0
    row['sigmoid_beta']  = beta
    row['sigmoid_R2']    = float(r2) if r2 is not None else None
    if 'sigmoid_g_eff' in row:
        row['sigmoid_g_eff'] = float(np.exp(-beta))
    return row, True


def _process(path: str, kind: str, *, dry_run: bool, backup: bool):
    with open(path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        print(f'  skip {path}: top-level is not a list')
        return

    before_sat = sum(1 for r in rows
                     if (r.get('sigmoid_beta') or 0) >= SAT_BEFORE)

    new_rows = []
    failed = 0
    for r in rows:
        upd, ok = _refit_row(r, kind)
        if not ok:
            failed += 1
            new_rows.append(r)
        else:
            new_rows.append(upd)

    after_sat = sum(1 for r in new_rows
                    if (r.get('sigmoid_beta') or 0) >= SAT_AFTER)
    after_resolved_above_old = sum(
        1 for r in new_rows
        if SAT_BEFORE <= (r.get('sigmoid_beta') or 0) < SAT_AFTER)

    rel = os.path.relpath(path, _REPO)
    print(f'  {rel}')
    print(f'    rows={len(rows):4d}  fit_failed={failed:3d}  '
          f'before_sat(>={SAT_BEFORE})={before_sat:4d}  '
          f'now_resolved_above_old={after_resolved_above_old:4d}  '
          f'now_sat(>={SAT_AFTER})={after_sat:4d}')

    if dry_run:
        return

    if backup:
        ts = _dt.datetime.now().strftime('%Y%m%dT%H%M%S')
        shutil.copy2(path, f'{path}.bak.{ts}')

    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(new_rows, f, indent=2)
        f.write('\n')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=_REPO,
                    help='repo root to walk (default: repo root)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report changes without writing')
    ap.add_argument('--no-backup', action='store_true',
                    help='skip writing <path>.bak.<ts> before overwriting')
    args = ap.parse_args()

    paths = _discover(args.root)
    if not paths:
        print('No results JSON found under', args.root)
        sys.exit(1)

    mode = 'DRY RUN' if args.dry_run else 'WRITING'
    print(f'Refit sigmoids ({mode}, new beta cap = {NEW_BETA_CAP})')
    print(f'Discovered {len(paths)} JSON files:')
    for path, kind in paths:
        _process(path, kind,
                 dry_run=args.dry_run,
                 backup=not args.no_backup)
    print('Done.')


if __name__ == '__main__':
    main()
