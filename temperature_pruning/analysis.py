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

3. **Model comparison** (reviewer diagnostic)
   Per cell, compare
       Model A: p_c = a + b*sigma + c*sigma^2  (k=3)
       Model B: p_c = a + c*sigma^2            (k=2, no linear term)
   using AIC/BIC and a two-sided t-test for H0: b=0.  The SK bond-disorder
   prediction is b=0 (pure quadratic); ΔAIC and the p-value quantify how
   much evidence the data actually hold against that null.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def group_pc_by_cell(results, min_r2=0.80):
    """Build the per-cell p_c(sigma) table.

    Returns
    -------
    dict {(H, L): list of (sigma, s_0, s_0_std, beta, R2, repeat)}
        ``s_0_std`` is the across-trials std when ``n_trials > 1`` was
        used in the sweep; 0.0 otherwise.
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
            float(r.get('sigmoid_s_0_std', 0.0) or 0.0),
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
        by_sigma_vals = defaultdict(list)
        by_sigma_stds = defaultdict(list)
        for (sigma, s0, s0_std, _beta, _r2, _rep) in entries:
            by_sigma_vals[float(sigma)].append(s0)
            by_sigma_stds[float(sigma)].append(s0_std)
        sigmas = sorted(by_sigma_vals)
        p_cs = [float(np.mean(by_sigma_vals[s])) for s in sigmas]
        # If trials produced an std, prefer that (averaged over repeats);
        # otherwise fall back to the across-repeats std at this sigma.
        p_cs_std = []
        for s in sigmas:
            trial_stds = [v for v in by_sigma_stds[s] if v > 0]
            if trial_stds:
                p_cs_std.append(float(np.mean(trial_stds)))
            else:
                vals = by_sigma_vals[s]
                p_cs_std.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
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
        fit['p_cs_std'] = p_cs_std
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


# ---------------------------------------------------------------------------
# Model comparison: full quadratic vs. pure quadratic (no linear term)
# ---------------------------------------------------------------------------

def _aic(rss, n, k):
    """Akaike Information Criterion for an OLS model."""
    if rss <= 0 or n <= k:
        return float('nan')
    return n * float(np.log(rss / n)) + 2 * k


def _bic(rss, n, k):
    """Bayesian Information Criterion for an OLS model."""
    if rss <= 0 or n <= k:
        return float('nan')
    return n * float(np.log(rss / n)) + k * float(np.log(n))


def _fit_no_linear(x, y):
    """Fit p_c = a + c*x^2 (no linear term) via ordinary least-squares.

    Uses the design matrix [1, x^2] so the linear coefficient is structurally
    absent — this is the SK bond-disorder null hypothesis.

    Returns
    -------
    dict with keys: a, c, a_se, c_se, R2, rss, n, k, AIC, BIC
        or None when the system is underdetermined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return None
    X = np.stack([np.ones(n), x ** 2], axis=1)
    coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    rss = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float('nan')
    k = 2
    # Covariance via (X^T X)^{-1} * s^2, s^2 = RSS/(n-k)
    s2 = rss / max(n - k, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    return {
        'a': float(coeffs[0]),
        'c': float(coeffs[1]),
        'a_se': float(se[0]),
        'c_se': float(se[1]),
        'R2': float(r2),
        'rss': rss,
        'n': n,
        'k': k,
        'AIC': _aic(rss, n, k),
        'BIC': _bic(rss, n, k),
    }


def fit_model_comparison(pc_by_cell, restrict_to_F_regime=True,
                         initial_sigma_max=0.3):
    """Per-cell AIC/BIC model comparison and t-test for the linear coefficient.

    Compares two models fitted on the same F-regime sigma window:

        Model A  p_c = a + b*sigma + c*sigma^2   (k=3 parameters)
        Model B  p_c = a + c*sigma^2             (k=2, b forced to zero)

    The SK bond-disorder prediction corresponds to Model B (b=0).  A small
    ΔAIC = AIC_B - AIC_A (< 2) and a non-significant p-value for b support
    Model B as the parsimonious description.

    The F-regime window is determined identically to ``fit_critical_line``
    (running-R² cutoff from ``_residual_based_F_regime_fit``), so both models
    are compared on exactly the same data points — no circularity between the
    window choice and the coefficient estimates, because the window uses the
    full-quadratic R² and is finalized *before* computing AIC/t-test.

    Parameters
    ----------
    pc_by_cell : dict from ``group_pc_by_cell``
    restrict_to_F_regime : bool
        If True (default), truncate to the iterative F-regime window.
    initial_sigma_max : float
        Bootstrap window for the running-R² cutoff.

    Returns
    -------
    dict {(H, L): {
        'sigma_cutoff': float,
        'sigmas': list[float],
        'p_cs': list[float],
        'model_A': dict,        # full quadratic fit + t-test fields
        'model_B': dict,        # no-linear fit
        'delta_AIC': float,     # AIC_B - AIC_A  (>0 => A preferred)
        'delta_BIC': float,
    }}
    """
    from scipy import stats as _st

    out = {}
    for (H, L), entries in pc_by_cell.items():
        by_sigma_vals = defaultdict(list)
        for (sigma, s0, _s0_std, _beta, _r2, _rep) in entries:
            by_sigma_vals[float(sigma)].append(s0)
        sigmas_all = sorted(by_sigma_vals)
        p_cs_all = [float(np.mean(by_sigma_vals[s])) for s in sigmas_all]

        if len(sigmas_all) < 4:
            continue

        if restrict_to_F_regime:
            window_fit = _residual_based_F_regime_fit(
                sigmas_all, p_cs_all, degree=2,
                initial_sigma_max=initial_sigma_max,
            )
            if window_fit is None:
                continue
            cutoff = float(window_fit['sigma_cutoff'])
        else:
            cutoff = float(sigmas_all[-1])

        # Select the F-regime points (same for both models).
        mask = np.array(sigmas_all) <= cutoff + 1e-9
        x = np.array(sigmas_all)[mask]
        y = np.array(p_cs_all)[mask]
        if len(x) < 3:
            continue

        # --- Model A: full quadratic ---
        fa = _poly_fit(x, y, degree=2)
        if fa is None:
            continue
        y_hat_a = np.polyval(list(reversed(fa['coeffs'])), x)
        rss_a = float(np.sum((y - y_hat_a) ** 2))
        n, k_a = fa['n'], 3
        fa['rss'] = rss_a
        fa['k'] = k_a
        fa['AIC'] = _aic(rss_a, n, k_a)
        fa['BIC'] = _bic(rss_a, n, k_a)
        # Two-sided t-test for H0: b = 0
        b, b_se = fa['b'], fa['b_se']
        t_b = b / b_se if b_se > 0 else float('nan')
        df = n - k_a
        p_b = float(2 * _st.t.sf(abs(t_b), df=df)) if np.isfinite(t_b) else float('nan')
        t_crit = float(_st.t.ppf(0.975, df=df)) if df > 0 else float('nan')
        fa['t_b'] = float(t_b)
        fa['p_b'] = float(p_b)
        fa['ci95_b_lo'] = float(b - t_crit * b_se) if np.isfinite(t_crit) else float('nan')
        fa['ci95_b_hi'] = float(b + t_crit * b_se) if np.isfinite(t_crit) else float('nan')

        # --- Model B: no linear term ---
        fb = _fit_no_linear(x, y)
        if fb is None:
            continue

        out[(H, L)] = {
            'sigma_cutoff': cutoff,
            'sigmas': x.tolist(),
            'p_cs': y.tolist(),
            'model_A': fa,
            'model_B': fb,
            'delta_AIC': fb['AIC'] - fa['AIC'],
            'delta_BIC': fb['BIC'] - fa['BIC'],
        }
    return out
