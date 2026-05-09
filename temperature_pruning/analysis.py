"""Statistical tests of the diluted Curie-Weiss prediction.

Two diagnostics on the (H, L, sigma) -> p_c table produced by ``core.py``:

1. **Linear critical line** (headline test)
   For each (H, L) cell fit p_c(sigma) = a + b*sigma. The model predicts
   ``a == 0`` and ``b > 0``; ``b = 1/J_0_eff`` is the empirical estimate
   of the toy-model coupling constant.

2. **Data collapse** (universality stress-test)
   Plot all A(s, sigma) curves on the rescaled axis x = s / sigma. The
   T/p equivalence implies they all collapse onto a single master curve.
   The collapse score is the ratio of inter-curve scatter to intra-curve
   scatter at matched x; near 1 means good collapse.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def group_pc_by_cell(results, min_r2=0.80):
    """Build the per-cell p_c(sigma) table.

    Parameters
    ----------
    results : list of dict
        Output of ``core.run_temperature_pruning_experiment``.
    min_r2 : float
        Sigmoid R^2 threshold; rows below this are skipped.

    Returns
    -------
    dict {(H, L): list of (sigma, s_0, beta, R2, repeat)}
    """
    out = defaultdict(list)
    for r in results:
        if r.get('sigmoid_R2') is None:
            continue
        if r['sigmoid_R2'] < min_r2:
            continue
        out[(int(r['H']), int(r['L']))].append((
            float(r['sigma']),
            float(r['sigmoid_s_0']),
            float(r['sigmoid_beta']),
            float(r['sigmoid_R2']),
            int(r['repeat']),
        ))
    return dict(out)


def _poly_fit(x, y, degree=2):
    """Least-squares fit ``y = c0 + c1 x + c2 x^2 + ...`` with R^2 + std errors.

    Returns
    -------
    dict with
        ``coeffs``      list ascending order [c0, c1, ..., c_degree]
        ``coeffs_se``   per-coefficient standard error
        ``R2``          coefficient of determination
        ``n``, ``degree``
    Backwards-compatible aliases for degree==2:
        ``a``, ``b``, ``c``  -> c0, c1, c2 (constant, linear, quadratic)
        ``a_se``, ``b_se``, ``c_se``
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= degree:
        return None
    # numpy returns highest-degree-first; we flip to ascending for clarity.
    coeffs_hi, cov = np.polyfit(x, y, deg=degree, cov=True)
    coeffs = list(coeffs_hi[::-1])
    coeffs_se = list(np.sqrt(np.diag(cov))[::-1])
    y_hat = np.polyval(coeffs_hi, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    out = {
        'coeffs': [float(c) for c in coeffs],
        'coeffs_se': [float(s) for s in coeffs_se],
        'R2': float(r2), 'n': int(n), 'degree': int(degree),
    }
    # convenience aliases
    names = ['a', 'b', 'c', 'd', 'e', 'f']
    for i, c in enumerate(coeffs):
        out[names[i]] = float(c)
        out[f'{names[i]}_se'] = float(coeffs_se[i])
    return out


def fit_critical_line(pc_by_cell, degree=2):
    """Per (H, L) polynomial fit p_c(sigma) = a + b sigma + c sigma^2 + ...

    The diluted Curie-Weiss model predicts a strict linear ``p_c = T/J_0``;
    empirical curves often look parabolic at finite size, so a degree-2 fit
    captures both the linear slope and the leading curvature.

    Returns
    -------
    dict {(H, L): {coeffs, coeffs_se, R2, n, degree, a, b, c, ..., sigmas, p_cs}}
    """
    out = {}
    for (H, L), entries in pc_by_cell.items():
        by_sigma = defaultdict(list)
        for (sigma, s0, _beta, _r2, _rep) in entries:
            by_sigma[float(sigma)].append(s0)
        sigmas = sorted(by_sigma)
        p_cs = [float(np.mean(by_sigma[s])) for s in sigmas]
        if len(sigmas) <= degree:
            continue
        fit = _poly_fit(sigmas, p_cs, degree=degree)
        if fit is None:
            continue
        fit['sigmas'] = list(sigmas)
        fit['p_cs'] = p_cs
        out[(H, L)] = fit
    return out


def collapse_score(results, sigma_min=1e-6, n_grid=40, min_r2=0.80):
    """Quantify the s/sigma data collapse for sigma > 0 curves.

    For each (H, L), interpolate every A(s, sigma) curve onto a common
    log-spaced grid in x = s / sigma. The score is

        score = mean over x of (inter-curve std at x) / (intra-curve std at x).

    A perfect collapse gives ``score -> 1``: the variation across sigmas at
    fixed x equals the noise within a single curve.

    Returns
    -------
    dict {(H, L): {score, x_grid, mean_curve, per_sigma_mean, n_sigmas}}
    """
    out = {}
    by_cell = defaultdict(list)
    for r in results:
        if r.get('sigmoid_R2') is None or r['sigmoid_R2'] < min_r2:
            continue
        if float(r['sigma']) <= sigma_min:
            continue
        by_cell[(int(r['H']), int(r['L']))].append(r)

    for cell, rows in by_cell.items():
        if len(rows) < 2:
            continue
        # Per-curve x = density / sigma.
        per_sigma = {}
        for r in rows:
            sigma = float(r['sigma'])
            xs = np.array(r['densities']) / sigma
            ys = np.array(r['accs_mean'])
            stds = np.array(r['accs_std'])
            per_sigma.setdefault(sigma, []).append((xs, ys, stds))

        # Common log-grid in x covering the overlap of all curves.
        x_lo = max(min(xs.min() for xs, _, _ in lst) for lst in per_sigma.values())
        x_hi = min(max(xs.max() for xs, _, _ in lst) for lst in per_sigma.values())
        if not (x_hi > x_lo > 0):
            continue
        x_grid = np.geomspace(x_lo, x_hi, n_grid)

        # Per-sigma mean curve (averaging repeats at same sigma).
        per_sigma_mean = {}
        per_sigma_intra_var = {}
        for sigma, lst in per_sigma.items():
            ys_interp = []
            stds_interp = []
            for (xs, ys, stds) in lst:
                order = np.argsort(xs)
                ys_interp.append(np.interp(x_grid, xs[order], ys[order]))
                stds_interp.append(np.interp(x_grid, xs[order], stds[order]))
            per_sigma_mean[sigma] = np.mean(ys_interp, axis=0)
            per_sigma_intra_var[sigma] = np.mean(np.array(stds_interp) ** 2, axis=0)

        Y = np.stack([per_sigma_mean[s] for s in sorted(per_sigma_mean)], axis=0)
        intra = np.stack([per_sigma_intra_var[s] for s in sorted(per_sigma_intra_var)],
                         axis=0)
        inter_std = Y.std(axis=0)
        intra_std = np.sqrt(intra.mean(axis=0))
        # Score: ratio averaged over the grid; clip intra to avoid div0.
        ratio = inter_std / np.maximum(intra_std, 1e-6)
        score = float(np.mean(ratio))
        out[cell] = {
            'score': score,
            'x_grid': x_grid.tolist(),
            'mean_curve': Y.mean(axis=0).tolist(),
            'per_sigma_mean': {float(s): per_sigma_mean[s].tolist()
                               for s in per_sigma_mean},
            'n_sigmas': int(Y.shape[0]),
        }
    return out
