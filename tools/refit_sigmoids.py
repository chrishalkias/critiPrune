#!/usr/bin/env python3
r"""Refit cached pruning curves with ``A_0 = 1/C`` fixed.

All datasets used by the unstructured-pruning overlay are 10-class problems,
so the low-density accuracy floor is fixed at ``A_0 = 0.1``. The refit stores
the three free parameters and their uncertainties in the existing
``sigmoid_*_v2`` fields without retraining any model.

Usage:
    .venv/bin/python tools/refit_sigmoids.py
    .venv/bin/python tools/refit_sigmoids.py --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.optimize import curve_fit


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

C_CLASSES = 10
A_FLOOR = 1.0 / C_CLASSES


def sigmoid(s, A_inf, s_0, beta):
    s = np.asarray(s, dtype=float)
    exponent = -beta * (s - s_0)
    return A_FLOOR + (A_inf - A_FLOOR) / (
        1.0 + np.exp(np.clip(exponent, -500, 500))
    )


def fit_sigmoid(densities, accuracies):
    """Fit ``(A_inf, s_0, beta)`` on physical density samples."""
    densities = np.asarray(densities, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    if (
        densities.ndim != 1
        or accuracies.ndim != 1
        or len(densities) != len(accuracies)
        or len(densities) < 4
        or not np.all(np.isfinite(densities))
        or not np.all(np.isfinite(accuracies))
        or np.any(densities < 0.0)
        or np.any(densities > 1.0)
    ):
        return None

    high_accuracy = max(float(accuracies.max()), A_FLOOR + 1e-3)
    accuracy_span = max(high_accuracy - A_FLOOR, 1e-3)
    density_steps = np.diff(densities)
    valid_steps = density_steps > 0
    if valid_steps.any():
        slopes = np.abs(np.diff(accuracies)[valid_steps] / density_steps[valid_steps])
        maximum_slope = float(slopes.max())
    else:
        maximum_slope = 0.2

    initial = [
        high_accuracy,
        float(np.median(densities)),
        float(np.clip(4.0 * maximum_slope / accuracy_span, 0.2, 100.0)),
    ]
    bounds = (
        [A_FLOOR, 0.0, 1e-4],
        [1.0, 1.0, 200.0],
    )
    try:
        parameters, covariance = curve_fit(
            sigmoid,
            densities,
            accuracies,
            p0=initial,
            bounds=bounds,
            maxfev=30_000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    errors = np.sqrt(np.diag(covariance))
    residuals = accuracies - sigmoid(densities, *parameters)
    residual_sum = float(np.sum(residuals**2))
    total_sum = float(np.sum((accuracies - accuracies.mean()) ** 2))
    n_observations = len(densities)
    n_parameters = len(parameters)
    adjusted_r2 = (
        1.0
        - (residual_sum / (n_observations - n_parameters))
        / (total_sum / (n_observations - 1))
        if total_sum > 0 and n_observations > n_parameters
        else float("nan")
    )
    return {
        "sigmoid_A_inf_v2": float(parameters[0]),
        "sigmoid_s_0_v2": float(parameters[1]),
        "sigmoid_beta_v2": float(parameters[2]),
        "sigmoid_A_inf_err_v2": float(errors[0]),
        "sigmoid_s_0_err_v2": float(errors[1]),
        "sigmoid_beta_err_v2": float(errors[2]),
        "sigmoid_R2_v2": float(adjusted_r2),
    }


def discover_results(root):
    pattern = os.path.join(
        os.fspath(root),
        "assets",
        "unstructured_pruning",
        "*",
        "scaling_results.json",
    )
    return sorted(glob.glob(pattern))


def refit_file(path, *, dry_run=False):
    with open(path) as handle:
        rows = json.load(handle)

    fitted = failed = skipped = 0
    updated_rows = []
    for row in rows:
        densities = row.get("densities")
        accuracies = row.get("accs_mean")
        if densities is None or accuracies is None:
            skipped += 1
            updated_rows.append(row)
            continue
        fit = fit_sigmoid(densities, accuracies)
        if fit is None:
            failed += 1
            updated_rows.append(row)
            continue
        updated = dict(row)
        updated.update(fit)
        updated_rows.append(updated)
        fitted += 1

    if not dry_run:
        temporary_path = path + ".tmp"
        with open(temporary_path, "w") as handle:
            json.dump(updated_rows, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    return fitted, failed, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=_REPO, help="repository root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = discover_results(args.root)
    if not paths:
        raise SystemExit("No unstructured-pruning scaling results were found.")

    mode = "dry run" if args.dry_run else "writing"
    print(f"Refitting {len(paths)} strata with A_0 = {A_FLOOR:.1f} ({mode})")
    total_fitted = total_failed = total_skipped = 0
    for path in paths:
        fitted, failed, skipped = refit_file(path, dry_run=args.dry_run)
        total_fitted += fitted
        total_failed += failed
        total_skipped += skipped
        relative_path = os.path.relpath(path, args.root)
        print(
            f"  {relative_path}: fitted={fitted}, failed={failed}, skipped={skipped}"
        )
    print(
        f"Total: fitted={total_fitted}, failed={total_failed}, "
        f"skipped={total_skipped}"
    )


if __name__ == "__main__":
    main()
