"""Input-noise / pruning iso-accuracy experiment — shared helpers.

Uses :mod:`unstructured_pruning.methods` and
:mod:`unstructured_pruning.core` for the Bernoulli-mask side; the
input-noise side is a plain forward pass on perturbed inputs.
"""

from __future__ import annotations

import numpy as np
import torch

from unstructured_pruning.core import apply_mask
from unstructured_pruning.methods import random_masks


def _seed_for(label, *idxs, base=0):
    """Deterministic non-negative seed for a (sigma_idx, draw_idx, ...) cell."""
    s = base + abs(hash((label, *idxs))) % (2 ** 31)
    return int(s)


def add_gaussian_noise(X, sigma, seed):
    """Return ``X + N(0, sigma^2 I)`` with a fresh per-call RNG."""
    if sigma <= 0:
        return X
    rng = np.random.default_rng(int(seed))
    eps = rng.standard_normal(X.shape).astype(X.dtype)
    return X + sigma * eps


@torch.no_grad()
def _forward_acc(model, X_np, y_np):
    p = next(model.parameters())
    X_t = torch.as_tensor(X_np, dtype=p.dtype, device=p.device)
    pred = model(X_t).argmax(1).cpu().numpy()
    return float((pred == np.asarray(y_np)).mean())


def evaluate_noisy_accuracy(model, X_test, y_test, sigma_grid,
                            n_draws=10, base_seed=0):
    """Per-sigma accuracy of the unpruned model on noisy inputs.

    Returns
    -------
    dict ``{sigma: (mean_acc, std_acc)}``
    """
    model.eval()
    out = {}
    for i, sigma in enumerate(sigma_grid):
        if sigma <= 0:
            a = _forward_acc(model, X_test, y_test)
            out[float(sigma)] = (a, 0.0)
            continue
        accs = []
        for d in range(n_draws):
            X_noisy = add_gaussian_noise(
                X_test, sigma, _seed_for('noise', i, d, base=base_seed))
            accs.append(_forward_acc(model, X_noisy, y_test))
        out[float(sigma)] = (float(np.mean(accs)), float(np.std(accs)))
    return out


def evaluate_pruned_accuracy(model, X_test, y_test, s_grid,
                             n_mask_seeds=3, base_seed=42):
    """Per-density accuracy under random Bernoulli pruning, no input noise.

    Returns
    -------
    dict ``{s: (mean_acc, std_acc)}`` plus ``normal_acc``.
    """
    mask_sets = random_masks(model, list(s_grid),
                             n_seeds=n_mask_seeds, base_seed=base_seed)
    accs = {}
    for s, seed_masks in mask_sets.items():
        per_seed = []
        for masks in seed_masks:
            pruned = apply_mask(model, masks)
            pruned.eval()
            per_seed.append(_forward_acc(pruned, X_test, y_test))
        accs[float(s)] = (float(np.mean(per_seed)), float(np.std(per_seed)))
    return accs, _forward_acc(model, X_test, y_test)


def evaluate_joint(model, X_test, y_test, s_grid, sigma_grid,
                   n_mask_seeds=3, n_noise_draws=5, base_seed=42):
    """Joint (s, sigma) grid: random mask + Gaussian input noise.

    Mask realisations and noise realisations are independent; the cell-mean
    accuracy is the mean over ``n_mask_seeds * n_noise_draws`` combined
    realisations.

    Returns
    -------
    dict ``{(s, sigma): (mean_acc, std_acc)}``
    """
    model.eval()
    mask_sets = random_masks(model, list(s_grid),
                             n_seeds=n_mask_seeds, base_seed=base_seed)
    pruned_models = {
        s: [apply_mask(model, masks).eval() for masks in seed_masks]
        for s, seed_masks in mask_sets.items()
    }
    out = {}
    for i_sigma, sigma in enumerate(sigma_grid):
        for s, models in pruned_models.items():
            accs = []
            if sigma <= 0:
                for m in models:
                    accs.append(_forward_acc(m, X_test, y_test))
            else:
                for d in range(n_noise_draws):
                    X_noisy = add_gaussian_noise(
                        X_test, sigma,
                        _seed_for('joint', i_sigma, d, base=base_seed))
                    for m in models:
                        accs.append(_forward_acc(m, X_noisy, y_test))
            out[(float(s), float(sigma))] = (
                float(np.mean(accs)), float(np.std(accs)))
    return out


# ---------------------------------------------------------------------------
# Curve fits
# ---------------------------------------------------------------------------
def fit_sigmoid_1d(xs, accs_mean, normal_acc):
    """Fit ``A(x) = A_0 + (A_inf - A_0) / (1 + exp(-beta * (x - x_0)))`` to
    ``(xs, accs_mean)``. Handles both ascending (``beta > 0``) and descending
    (``beta < 0``) sigmoids, so it works for the pruning sweep (A increases
    with ``s``) and the input-noise sweep (A decreases with ``sigma``)
    without sign games on the x-axis.

    Returns ``(A_inf, A_0, x_0, beta, R2)`` or all-NaN if the fit fails.
    """
    from scipy.optimize import curve_fit

    x = np.asarray(xs, dtype=float)
    y = np.asarray(accs_mean, dtype=float)
    if len(x) < 5 or not np.all(np.isfinite(y)):
        return (float('nan'),) * 5

    def f(x, A_inf, A_0, x_0, beta):
        z = -beta * (x - x_0)
        return A_0 + (A_inf - A_0) / (1.0 + np.exp(np.clip(z, -500, 500)))

    A_hi = float(max(y.max(), normal_acc))
    A_lo = float(max(y.min(), 0.0))
    span = max(A_hi - A_lo, 1e-3)
    # Data-driven beta_0 from the largest |interior secant| slope.
    order = np.argsort(x)
    xs_, ys_ = x[order], y[order]
    dx = np.diff(xs_)
    valid = dx > 0
    s_max = (float(np.max(np.abs(np.diff(ys_)[valid] / dx[valid])))
             if valid.any() else 0.2)
    beta_sign = +1.0 if (ys_[-1] >= ys_[0]) else -1.0
    beta_0 = beta_sign * float(np.clip(4.0 * s_max / span, 0.2, 100.0))
    p0 = [A_hi, A_lo, float(np.median(x)), beta_0]
    x_range = float(xs_[-1] - xs_[0])
    bounds = (
        [0.0, -0.05, float(xs_[0]) - x_range, -200.0],
        [1.0,  1.0,  float(xs_[-1]) + x_range, 200.0],
    )
    try:
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=30_000)
    except Exception:
        return (float('nan'),) * 5
    y_pred = f(x, *popt)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return (float(popt[0]), float(popt[1]), float(popt[2]),
            float(popt[3]), float(r2))


def iso_accuracy_contour(joint, s_grid, sigma_grid, level):
    """Trace the iso-accuracy contour ``A(s, sigma) = level`` in the (s, sigma)
    plane.

    For each ``s`` column of the joint grid, finds the sigma at which the
    column's accuracy curve crosses ``level`` (linear interpolation).
    Returns the list of ``(s, sigma)`` pairs where the crossing exists.
    """
    sigmas = np.asarray(sorted(set(sigma_grid)), dtype=float)
    pts = []
    for s in sorted(set(s_grid)):
        col = np.array([joint[(float(s), float(sg))][0] for sg in sigmas])
        # Accuracy is non-increasing in sigma; flip to ascending for interp.
        if col[0] < level or col[-1] > level:
            continue  # contour doesn't cross this column
        # Locate the bracket then linear-interp on (sigma -> acc).
        idx = np.searchsorted(-col, -level)
        if idx == 0 or idx >= len(sigmas):
            continue
        a0, a1 = col[idx - 1], col[idx]
        sg0, sg1 = sigmas[idx - 1], sigmas[idx]
        if a0 == a1:
            sigma_iso = 0.5 * (sg0 + sg1)
        else:
            sigma_iso = sg0 + (sg1 - sg0) * (level - a0) / (a1 - a0)
        pts.append((float(s), float(sigma_iso)))
    return pts
