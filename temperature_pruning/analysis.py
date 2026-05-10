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


def _J0_from_curvature(coeffs):
    """Extract J_0_eff = 1/sqrt(2c) from a degree->=2 polynomial fit.

    Returns ``None`` when the curvature ``c`` is non-positive (degenerate
    fit, no F-regime interpretation).
    """
    if len(coeffs) < 3:
        return None
    c = coeffs[2]
    if c <= 0:
        return None
    return 1.0 / np.sqrt(2.0 * c)


def _residual_based_F_regime_fit(sigmas, p_cs, degree=2,
                                 initial_sigma_max=0.3,
                                 R2_drop_tol=0.01,
                                 min_points=4):
    """Data-driven F -> SG / thermalisation cutoff via running R^2.

    We walk outward in sigma starting from the bootstrap window
    sigma <= ``initial_sigma_max``. After each candidate extension we
    refit the polynomial and check the resulting R^2:

        - Accept the new point as long as R^2 has not dropped more than
          ``R2_drop_tol`` below the *bootstrap* R^2 (the in-window fit
          quality at sigma <= ``initial_sigma_max``).
        - Otherwise, the previous sigma is declared the F -> SG /
          thermalisation boundary and we stop.

    Anchoring against the bootstrap R^2 (rather than an absolute floor or
    the running maximum) is forgiving for cells that are intrinsically
    noisy in the F regime and decisive for cells where the curve cleanly
    breaks down at large sigma.

    Returned fields
    ---------------
    ``sigma_cutoff``   -- empirical F regime boundary (used for plotting).
    ``sigma_fit_max``  -- alias for ``sigma_cutoff`` (fit window == cutoff
                          here by construction).
    ``J0_eff_iter``    -- J_0 = 1 / sqrt(2 c) from the truncated fit.
    ``restricted``     -- True if at least one point was excluded.
    ``n_iters``        -- number of points beyond the bootstrap that were
                          successfully appended.
    """
    sigmas = np.asarray(sigmas, dtype=float)
    p_cs = np.asarray(p_cs, dtype=float)
    sigma_max_data = float(sigmas.max())

    order = np.argsort(sigmas)
    sigmas_s = sigmas[order]
    p_cs_s = p_cs[order]
    n_total = len(sigmas_s)

    full_fit = _poly_fit(sigmas, p_cs, degree=degree)
    if full_fit is None:
        return None
    full_fit['sigma_fit_max'] = sigma_max_data
    full_fit['sigma_cutoff'] = sigma_max_data
    full_fit['J0_eff_iter'] = _J0_from_curvature(full_fit['coeffs'])
    full_fit['restricted'] = False
    full_fit['n_iters'] = 0

    init_count = int((sigmas_s <= initial_sigma_max + 1e-9).sum())
    init_count = max(init_count, min_points)
    if init_count >= n_total:
        return full_fit

    cur_fit = _poly_fit(sigmas_s[:init_count], p_cs_s[:init_count],
                        degree=degree)
    if cur_fit is None:
        return full_fit

    R2_boot = float(cur_fit['R2']) if cur_fit['R2'] == cur_fit['R2'] else 1.0
    R2_threshold = R2_boot - R2_drop_tol
    n_used = init_count
    n_accepted = 0

    for i in range(init_count, n_total):
        candidate = _poly_fit(sigmas_s[:i + 1], p_cs_s[:i + 1],
                              degree=degree)
        if candidate is None:
            break
        r2 = float(candidate['R2']) if candidate['R2'] == candidate['R2'] else 0.0
        if r2 < R2_threshold:
            break
        cur_fit = candidate
        n_used = i + 1
        n_accepted += 1

    if n_used == n_total:
        full_fit['J0_eff_iter'] = _J0_from_curvature(cur_fit['coeffs'])
        return full_fit

    cutoff = float(sigmas_s[n_used - 1])
    cur_fit['sigma_fit_max'] = cutoff
    cur_fit['sigma_cutoff'] = cutoff
    cur_fit['J0_eff_iter'] = _J0_from_curvature(cur_fit['coeffs'])
    cur_fit['restricted'] = True
    cur_fit['n_iters'] = int(n_accepted)
    return cur_fit


def fit_critical_line(pc_by_cell, degree=2,
                      restrict_to_F_regime=True,
                      initial_sigma_max=0.3):
    """Per (H, L) polynomial fit p_c(sigma) = a + b sigma + c sigma^2 + ...

    By default the fit is restricted to the SK ferromagnetic regime
    sigma <= J_0_eff, with J_0_eff iteratively self-consistent with the
    fitted curvature. Set ``restrict_to_F_regime=False`` to fit the full
    sigma range.

    Returns
    -------
    dict {(H, L): {coeffs, coeffs_se, R2, n, degree, a, b, c, ...,
                   sigmas, p_cs, sigma_cutoff, J0_eff_iter, restricted,
                   n_iters}}
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
        if restrict_to_F_regime:
            fit = _residual_based_F_regime_fit(
                sigmas, p_cs, degree=degree,
                initial_sigma_max=initial_sigma_max,
            )
        else:
            fit = _poly_fit(sigmas, p_cs, degree=degree)
            if fit is not None:
                fit['sigma_cutoff'] = float(sigmas[-1])
                fit['J0_eff_iter'] = _J0_from_curvature(fit['coeffs'])
                fit['restricted'] = False
                fit['n_iters'] = 0
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
