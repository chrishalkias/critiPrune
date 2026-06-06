#!/usr/bin/env python3
r"""v2 s_0 manifold plots: uses sigmoid_s_0_v2 (A_0=1/C fixed) and re-fits
the power law inline on v2 cells.

Sibling of ``plot_3d_scaling.py``. Reuses the ``render_3d`` rendering
function from that module after swapping the aggregation field names
and re-fitting the s_0 = a H^alpha L^gamma power law on v2 data.

Writes outputs to ``unstructured_pruning/figures/unstructured_figures_*/
s0_3d_v2.png`` (sibling of the v1 ``s0_3d.png``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import unstructured_pruning.plot_3d_scaling as base


def _aggregate_v2(rows, min_r2=0.80):
    bins = defaultdict(list)
    for r in rows:
        r2 = r.get('sigmoid_R2_v2')
        s0 = r.get('sigmoid_s_0_v2')
        if r2 is None or s0 is None or r2 < min_r2:
            continue
        bins[(int(r['H']), int(r['L']))].append(float(s0))
    if not bins:
        return None
    H, L, mu, sd, n = [], [], [], [], []
    for (h, l), vals in sorted(bins.items()):
        arr = np.asarray(vals)
        H.append(h); L.append(l)
        mu.append(arr.mean()); sd.append(arr.std()); n.append(len(arr))
    return (np.array(H, dtype=float), np.array(L, dtype=float),
            np.array(mu, dtype=float), np.array(sd, dtype=float),
            np.array(n, dtype=int))


def _powerlaw_v2(rows, min_r2=0.80):
    """Refit s_0 = a H^alpha L^gamma on v2 cells. Returns dict like
    scaling_laws.json['s0']."""
    agg = _aggregate_v2(rows, min_r2)
    if agg is None:
        return None
    H, L, mu, _, _ = agg
    if len(H) < 6:
        return None

    def fn(HL, a, alpha, gamma):
        h, l = HL
        return a * np.power(h, alpha) * np.power(l, gamma)

    try:
        popt, pcov = curve_fit(fn, (H, L), mu,
                               p0=[1.0, -0.3, 0.8], maxfev=30_000)
        resid = mu - fn((H, L), *popt)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((mu - mu.mean()) ** 2))
        n, p = len(mu), len(popt)
        r2_adj = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
                  if ss_tot > 0 and n > p else float('nan'))
        return {
            'a': float(popt[0]),
            'alpha': float(popt[1]),
            'gamma': float(popt[2]),
            'R2': float(r2_adj),
            'formula': (f"s_0 = {popt[0]:.4f} * H^{popt[1]:+.3f} "
                        f"* L^{popt[2]:+.3f}"),
        }
    except Exception:
        return None


def main(base_dir='unstructured_pruning/figures'):
    have_latex = base._configure_style()
    print(f'  Rendering with text.usetex = {have_latex}')

    # Monkey-patch the aggregator so render_3d uses v2 fields.
    base._aggregate = _aggregate_v2

    saved, skipped = [], []
    for d in sorted(glob.glob(os.path.join(base_dir, 'unstructured_figures_*'))):
        results_p = os.path.join(d, 'scaling_results.json')
        if not os.path.isfile(results_p):
            skipped.append((d, 'no scaling_results.json'))
            continue
        results = json.load(open(results_p))
        s0_pl = _powerlaw_v2(results)
        if s0_pl is None:
            skipped.append((d, 'power-law fit failed or too few v2 cells'))
            continue
        token = (os.path.basename(d).replace('unstructured_figures_', ''))
        out = os.path.join(d, 's0_3d_v2.png')
        scaling = {'s0': s0_pl}
        p = base.render_3d(results, scaling, out, dataset_method=token)
        if p:
            print(f'  Saved: {p}   {s0_pl["formula"]}  R2_adj={s0_pl["R2"]:.3f}')
            saved.append(p)
        else:
            skipped.append((d, 'render_3d returned None'))

    print(f'\nDone. {len(saved)} plots saved, {len(skipped)} skipped.')
    for d, why in skipped:
        print(f'  skipped: {os.path.basename(d)}  ({why})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='unstructured_pruning/figures')
    args = ap.parse_args()
    main(base_dir=args.base)
