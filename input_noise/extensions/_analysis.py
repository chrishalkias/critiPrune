"""Shared analysis helpers for §5.2 extensions.

All three drivers (iso_levels, depth_cells, seed_replicates) use the same
two operations:

  - extract iso-A contour from a joint (s, sigma) accuracy grid
  - fit the rational curve (eq. 12 of .docs/input_noise.md) per cell
    and the parameter-free collapse residual (eq. 14)

These wrap input_noise.core.iso_accuracy_contour and use the same fit
formula as input_noise.plotting.plots.fit_conversion, so numbers reproduce.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from input_noise.core import iso_accuracy_contour


def joint_dict(joint: dict) -> Dict[Tuple[float, float], Tuple[float, float]]:
    """Convert the {s_grid, sigma_grid, mean[i_s][i_sg], std[i_s][i_sg]}
    storage format into the {(s, sigma): (mean, std)} dict shape that
    ``iso_accuracy_contour`` expects.
    """
    return {(float(s), float(sg)):
            (joint['mean'][i_s][i_sg], joint['std'][i_s][i_sg])
            for i_s, s in enumerate(joint['s_grid'])
            for i_sg, sg in enumerate(joint['sigma_grid'])}


def fit_rational(contour: List[Tuple[float, float]],
                 x2: float) -> Dict[str, float]:
    """Fit the single-parameter rational curve

        sigma^2(s)  =  s * sigma2_1  -  (1 - s) * <x^2>

    by least squares in ``sigma^2`` space (closed form).

    Returns ``{'sigma2_1', 'R2', 'n'}`` or ``{'sigma2_1': nan, ...}`` if
    the fit is undefined.
    """
    n = len(contour)
    if n < 2:
        return {'sigma2_1': float('nan'), 'R2': float('nan'), 'n': n}
    s_arr  = np.array([p[0] for p in contour], dtype=float)
    sg_arr = np.array([p[1] for p in contour], dtype=float)
    num = float(np.sum(s_arr * (sg_arr ** 2 + (1.0 - s_arr) * x2)))
    den = float(np.sum(s_arr ** 2))
    if den <= 0:
        return {'sigma2_1': float('nan'), 'R2': float('nan'), 'n': n}
    sigma2_1 = num / den
    y_pred = s_arr * sigma2_1 - (1.0 - s_arr) * x2
    y      = sg_arr ** 2
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'sigma2_1': float(sigma2_1), 'R2': float(R2), 'n': n}


def collapse_coords(contour, sigma2_1, x2):
    """Return (xi, eta) per eq. (13) of .docs/input_noise.md.

    xi  = (1 - s) * (1 + <x^2> / sigma^2(1))
    eta = sigma^2_iso(s) / sigma^2(1)
    """
    s_arr  = np.array([p[0] for p in contour], dtype=float)
    sg_arr = np.array([p[1] for p in contour], dtype=float)
    xi  = (1.0 - s_arr) * (1.0 + x2 / sigma2_1)
    eta = (sg_arr ** 2) / sigma2_1
    return xi, eta


def rms_to_line(xi: np.ndarray, eta: np.ndarray) -> float:
    """RMS residual to the framework prediction ``eta = 1 - xi`` (eq. 14)."""
    if len(xi) == 0:
        return float('nan')
    return float(np.sqrt(np.mean((eta - (1.0 - xi)) ** 2)))


def analyze_cell(joint: dict, x2: float, iso_levels) -> Dict[float, dict]:
    """Per-cell, per-iso-level: extract contour, fit rational, compute
    collapse residual.

    Returns ``{iso_level: {'contour', 'sigma2_1', 'R2', 'n', 'rms_to_line'}}``.
    """
    jd = joint_dict(joint)
    s_grid     = joint['s_grid']
    sigma_grid = joint['sigma_grid']
    out = {}
    for level in iso_levels:
        contour = iso_accuracy_contour(jd, s_grid, sigma_grid, level)
        fit = fit_rational(contour, x2)
        rms = float('nan')
        if np.isfinite(fit['sigma2_1']) and fit['sigma2_1'] > 0 and fit['n'] >= 2:
            xi, eta = collapse_coords(contour, fit['sigma2_1'], x2)
            rms = rms_to_line(xi, eta)
        out[float(level)] = {
            'contour':     [(float(s), float(sg)) for s, sg in contour],
            'sigma2_1':    fit['sigma2_1'],
            'R2':          fit['R2'],
            'n':           fit['n'],
            'rms_to_line': rms,
        }
    return out
