"""Unit tests for the raw-contour upper-envelope helpers in cluster_analyze.

Pure-numeric only -- importing cluster_analyze must not require seaborn/pandas
(those are lazily imported inside the collapse plotters), so these run fast.
"""
import math

import numpy as np
import pytest

from input_noise.analysis.cluster_analyze import (
    _iso_conversion_sigma,
    _envelope_params,
    _upper_envelope,
)


# ---------------------------------------------------------------------------
# _iso_conversion_sigma:  sigma_x(s) = sqrt(s*sigma2_1 - (1-s)*x2)
# ---------------------------------------------------------------------------
def test_iso_conversion_at_s1_is_sqrt_sigma2_1():
    # at s=1 the (1-s) term vanishes: sigma_x(1) = sqrt(sigma2_1)
    out = _iso_conversion_sigma(np.array([1.0]), sigma2_1=4.0, x2=1.0)
    assert out[0] == pytest.approx(2.0)


def test_iso_conversion_zero_at_s0():
    # s0 = x2 / (sigma2_1 + x2); the curve hits sigma_x = 0 there
    sigma2_1, x2 = 4.0, 1.0
    s0 = x2 / (sigma2_1 + x2)            # = 0.2
    out = _iso_conversion_sigma(np.array([s0]), sigma2_1, x2)
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_iso_conversion_nan_below_s0():
    # below s0 the radicand is negative -> NaN (so the line starts at s0)
    sigma2_1, x2 = 4.0, 1.0
    out = _iso_conversion_sigma(np.array([0.1]), sigma2_1, x2)   # 0.4-0.9<0
    assert math.isnan(out[0])


def test_iso_conversion_monotone_increasing():
    s = np.linspace(0.2, 1.0, 50)
    out = _iso_conversion_sigma(s, sigma2_1=4.0, x2=1.0)
    assert np.all(np.diff(out) > 0)


# ---------------------------------------------------------------------------
# _envelope_params
# ---------------------------------------------------------------------------
def _recs(dataset, rows):
    """rows: list of (L, sigma2_1). x2_mean fixed at 1.0."""
    return [{'dataset': dataset, 'L': L, 'sigma2_1': s, 'x2_mean': 1.0}
            for (L, s) in rows]


def test_envelope_params_basic():
    recs = _recs('d', [(2, 4.0), (2, 6.0), (3, 3.0), (4, 2.0)])
    p = _envelope_params(recs, 'd', anchor_L=2, env_q=0.99)
    assert p['x2'] == pytest.approx(1.0)
    assert p['anchor_L_used'] == 2
    assert p['sigma2_1_fw'] == pytest.approx(5.0)        # mean of L=2 -> (4+6)/2
    assert p['n_fw'] == 2
    assert p['n_all'] == 4
    # p99 of all sigma2_1 -> near the max (6.0)
    assert p['sigma2_1_env'] == pytest.approx(np.quantile([4, 6, 3, 2], 0.99))
    # s0 = x2 / (sigma2_1 + x2)
    assert p['s0_fw'] == pytest.approx(1.0 / (5.0 + 1.0))


def test_envelope_params_falls_back_to_min_L():
    # no L=2 present -> fall back to the smallest available L (=3)
    recs = _recs('d', [(3, 3.0), (3, 5.0), (4, 2.0)])
    p = _envelope_params(recs, 'd', anchor_L=2)
    assert p['anchor_L_used'] == 3
    assert p['sigma2_1_fw'] == pytest.approx(4.0)        # mean of L=3 -> (3+5)/2


def test_envelope_params_ignores_nonpositive_and_nonfinite():
    recs = _recs('d', [(2, 4.0), (2, -1.0), (2, float('nan')), (2, 6.0)])
    p = _envelope_params(recs, 'd', anchor_L=2)
    assert p['n_fw'] == 2                                # only 4.0 and 6.0 kept
    assert p['sigma2_1_fw'] == pytest.approx(5.0)


def test_envelope_params_filters_by_dataset():
    recs = _recs('d', [(2, 4.0)]) + _recs('other', [(2, 99.0)])
    p = _envelope_params(recs, 'd', anchor_L=2)
    assert p['n_all'] == 1
    assert p['sigma2_1_fw'] == pytest.approx(4.0)


def test_envelope_params_returns_none_when_empty():
    p = _envelope_params(_recs('d', []), 'd', anchor_L=2)
    assert p is None


# ---------------------------------------------------------------------------
# _upper_envelope: nonparametric per-s-bin max -> true top boundary
# ---------------------------------------------------------------------------
def test_upper_envelope_no_points_above():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.05, 1.0, 5000)
    sigma = rng.uniform(0.0, 2.0, 5000)
    centres, tops = _upper_envelope(s, sigma, nbins=20, min_count=3)
    # every point must lie at or below the envelope value of its own bin
    edges = np.linspace(0.0, 1.0, 21)
    for si, gi in zip(s, sigma):
        b = min(int(np.digitize(si, edges) - 1), 19)
        bc = 0.5 * (edges[b] + edges[b + 1])
        j = np.argmin(np.abs(centres - bc))
        assert gi <= tops[j] + 1e-12


def test_upper_envelope_picks_bin_max():
    # one bin around s=0.5 with a known max
    s = np.array([0.50, 0.51, 0.52, 0.49])
    sigma = np.array([1.0, 2.5, 0.3, 1.7])
    centres, tops = _upper_envelope(s, sigma, nbins=10, min_count=3)
    # the bin covering [0.5,0.6) holds 0.50/0.51/0.52 -> max 2.5
    j = np.argmin(np.abs(centres - 0.55))
    assert tops[j] == pytest.approx(2.5)


def test_upper_envelope_drops_sparse_bins():
    s = np.array([0.11, 0.12, 0.13, 0.90])     # one lonely point at 0.90
    sigma = np.array([0.5, 0.6, 0.7, 9.9])
    centres, tops = _upper_envelope(s, sigma, nbins=10, min_count=3)
    # the sparse high-s bin (1 point) is dropped; the 9.9 outlier never appears
    assert tops.max() == pytest.approx(0.7)
    assert np.all(np.diff(centres) > 0)        # returned sorted by s
