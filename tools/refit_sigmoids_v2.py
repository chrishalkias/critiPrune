#!/usr/bin/env python3
r"""Three-parameter sigmoid refit with A_0 fixed at 1/C.

Phase 5 follow-up. The v1 fit (``pruning.pruning.fit_sigmoid``) treats
``A_0`` as the s -> -inf asymptote with bounds [-0.05, 1.0]; on most
cells the optimiser pins A_0 to the lower bound, which is unphysical
(random guess for a 10-class problem is 1/C = 0.1). This v2 routine
constrains A_0 := 1/C and fits only (A_inf, s_0, beta), writing the new
parameters as ``sigmoid_*_v2`` fields alongside the existing v1 fields.

Then aggregates per (dataset, method): bins by (H, L) averaging over
``repeat``, fits the power law ``s_0 = c H^alpha L^gamma`` on the
v2 cells with R^2_adj >= 0.80, and writes
``tools/refit_summary.md`` with a v1-vs-v2 comparison table.

Usage:
    .venv/bin/python tools/refit_sigmoids_v2.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

C_CLASSES = 10            # all our datasets are 10-class
A_FLOOR = 1.0 / C_CLASSES # 0.1


def sigmoid_v2(s, A_inf, s_0, beta):
    s = np.asarray(s, dtype=float)
    return A_FLOOR + (A_inf - A_FLOOR) / (
        1.0 + np.exp(np.clip(-beta * (s - s_0), -500, 500))
    )


def fit_v2(densities, accs):
    densities = np.asarray(densities, dtype=float)
    accs = np.asarray(accs, dtype=float)
    if len(densities) < 4:
        return None
    A_hi = max(float(accs.max()), A_FLOOR + 1e-3)
    span = max(A_hi - A_FLOOR, 1e-3)
    dk = np.diff(densities)
    valid = dk > 0
    if valid.any():
        slopes = np.abs(np.diff(accs)[valid] / dk[valid])
        s_max = float(slopes.max())
    else:
        s_max = 0.2
    beta0 = float(np.clip(4.0 * s_max / span, 0.2, 100.0))
    p0 = [A_hi, float(np.median(densities)), beta0]
    bounds = (
        [A_FLOOR, 0.0, 1e-4],
        [1.0, 2.0 * float(densities.max()), 200.0],
    )
    try:
        popt, pcov = curve_fit(
            sigmoid_v2, densities, accs, p0=p0,
            bounds=bounds, maxfev=30_000,
        )
        perr = np.sqrt(np.diag(pcov))
        resid = accs - sigmoid_v2(densities, *popt)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((accs - accs.mean()) ** 2))
        n, p = len(densities), len(popt)
        r2 = (
            1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
            if ss_tot > 0 and n > p else float('nan')
        )
        return {
            'sigmoid_A_inf_v2': float(popt[0]),
            'sigmoid_s_0_v2': float(popt[1]),
            'sigmoid_beta_v2': float(popt[2]),
            'sigmoid_A_inf_err_v2': float(perr[0]),
            'sigmoid_s_0_err_v2': float(perr[1]),
            'sigmoid_beta_err_v2': float(perr[2]),
            'sigmoid_R2_v2': float(r2),
        }
    except Exception:
        return {
            'sigmoid_A_inf_v2': None,
            'sigmoid_s_0_v2': None,
            'sigmoid_beta_v2': None,
            'sigmoid_R2_v2': None,
        }


def discover_strata():
    pattern = os.path.join(
        _REPO, 'unstructured_pruning', 'figures',
        'unstructured_figures_*', 'scaling_results.json'
    )
    paths = sorted(glob.glob(pattern))
    out = []
    for p in paths:
        stem = os.path.basename(os.path.dirname(p))
        tag = stem.replace('unstructured_figures_', '')
        # split last "_method" off; methods are random, wanda, magnitude
        for m in ('random', 'wanda', 'magnitude'):
            if tag.endswith('_' + m):
                dataset = tag[: -(len(m) + 1)]
                method = m
                break
        else:
            dataset, method = tag, 'unknown'
        out.append((dataset, method, p))
    return out


def refit_stratum(json_path):
    with open(json_path) as f:
        cells = json.load(f)
    refit_count = 0
    for cell in cells:
        d = cell.get('densities')
        a = cell.get('accs_mean')
        if d is None or a is None:
            continue
        new = fit_v2(d, a)
        if new is None:
            continue
        cell.update(new)
        refit_count += 1
    with open(json_path, 'w') as f:
        json.dump(cells, f, indent=2)
    return cells, refit_count


def powerlaw(HL, c, alpha, gamma):
    H, L = HL
    return c * np.power(H, alpha) * np.power(L, gamma)


def fit_powerlaw(cells, key='sigmoid_s_0_v2', r2_key='sigmoid_R2_v2',
                 r2_min=0.80):
    # bin by (H, L), average across repeat
    by_hl = defaultdict(list)
    for c in cells:
        v = c.get(key)
        r2 = c.get(r2_key)
        if v is None or r2 is None or r2 < r2_min:
            continue
        by_hl[(int(c['H']), int(c['L']))].append(float(v))
    if not by_hl:
        return None
    Hs, Ls, ys = [], [], []
    for (h, l), vals in by_hl.items():
        Hs.append(h); Ls.append(l); ys.append(np.mean(vals))
    Hs = np.array(Hs); Ls = np.array(Ls); ys = np.array(ys)
    if len(ys) < 6:
        return {'n': len(ys), 'fail': 'too_few_cells'}
    try:
        popt, pcov = curve_fit(powerlaw, (Hs, Ls), ys,
                               p0=[1.0, -0.3, 0.8], maxfev=30_000)
        perr = np.sqrt(np.diag(pcov))
        resid = ys - powerlaw((Hs, Ls), *popt)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        n, p = len(ys), len(popt)
        r2_adj = (
            1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
            if ss_tot > 0 and n > p else float('nan')
        )
        return {
            'n': int(n),
            'c': float(popt[0]),
            'alpha': float(popt[1]),
            'alpha_err': float(perr[1]),
            'gamma': float(popt[2]),
            'gamma_err': float(perr[2]),
            'R2_adj': float(r2_adj),
        }
    except Exception as e:
        return {'n': len(ys), 'fail': str(e)}


def main():
    strata = discover_strata()
    print(f"Found {len(strata)} strata to refit.")
    summary = []
    for dataset, method, path in strata:
        print(f"  refitting {dataset:20s} | {method:10s} ... ", end='', flush=True)
        cells, n = refit_stratum(path)
        v1 = fit_powerlaw(cells, key='sigmoid_s_0', r2_key='sigmoid_R2')
        v2 = fit_powerlaw(cells, key='sigmoid_s_0_v2', r2_key='sigmoid_R2_v2')
        summary.append({
            'dataset': dataset, 'method': method,
            'n_cells_refit': n, 'v1': v1, 'v2': v2,
        })
        print(f"n={n}; v1 alpha={v1['alpha']:+.3f}+/-{v1['alpha_err']:.3f} gamma={v1['gamma']:+.3f}+/-{v1['gamma_err']:.3f}; v2 alpha={v2['alpha']:+.3f}+/-{v2['alpha_err']:.3f} gamma={v2['gamma']:+.3f}+/-{v2['gamma_err']:.3f}")

    # Write summary
    out_md = os.path.join(_HERE, 'refit_summary.md')
    with open(out_md, 'w') as f:
        f.write("# Sigmoid refit v2: A_0 := 1/C fixed at random-guess floor\n\n")
        f.write(
            "Three-parameter fit A(s) = 0.1 + (A_inf - 0.1)/(1+exp(-beta(s - s_0))) "
            "with A_0 fixed at 1/C = 0.1 (10-class random guess). v1 (4-parameter) "
            "leaves A_0 unconstrained in [-0.05, 1.0] and on most cells pins it to "
            "the lower bound, which is unphysical.\n\n"
        )
        f.write("## Per-stratum comparison\n\n")
        f.write("| dataset | method | n | v1 alpha | v2 alpha | v1 gamma | v2 gamma | v1 R2adj | v2 R2adj |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            v1 = row['v1']; v2 = row['v2']
            if 'fail' in v1: v1_alpha = v1_gamma = v1_r2 = 'n/a'
            else:
                v1_alpha = f"{v1['alpha']:+.3f}+/-{v1['alpha_err']:.3f}"
                v1_gamma = f"{v1['gamma']:+.3f}+/-{v1['gamma_err']:.3f}"
                v1_r2 = f"{v1['R2_adj']:.3f}"
            if 'fail' in v2: v2_alpha = v2_gamma = v2_r2 = 'n/a'
            else:
                v2_alpha = f"{v2['alpha']:+.3f}+/-{v2['alpha_err']:.3f}"
                v2_gamma = f"{v2['gamma']:+.3f}+/-{v2['gamma_err']:.3f}"
                v2_r2 = f"{v2['R2_adj']:.3f}"
            f.write(f"| {row['dataset']} | {row['method']} | {row['n_cells_refit']} | "
                    f"{v1_alpha} | {v2_alpha} | {v1_gamma} | {v2_gamma} | "
                    f"{v1_r2} | {v2_r2} |\n")
        # Sigma agreement verdict
        f.write("\n## v1 vs v2 1-sigma agreement\n\n")
        agree_alpha = agree_gamma = 0
        total = 0
        for row in summary:
            v1 = row['v1']; v2 = row['v2']
            if 'fail' in v1 or 'fail' in v2:
                continue
            total += 1
            da = abs(v1['alpha'] - v2['alpha'])
            sa = max(v1['alpha_err'], v2['alpha_err'])
            dg = abs(v1['gamma'] - v2['gamma'])
            sg = max(v1['gamma_err'], v2['gamma_err'])
            agree_alpha += int(da <= sa)
            agree_gamma += int(dg <= sg)
        f.write(f"- Strata where |alpha_v1 - alpha_v2| <= max(SE): {agree_alpha} / {total}\n")
        f.write(f"- Strata where |gamma_v1 - gamma_v2| <= max(SE): {agree_gamma} / {total}\n")
    print(f"\nWrote {out_md}")


if __name__ == '__main__':
    main()
