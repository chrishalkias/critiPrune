#!/usr/bin/env python3
r"""Overlay raw pruning curves with their physical-domain sigmoid fits.

The main axes show the cached experimental accuracy curves, centred at each
cell's fitted critical density ``s_0``. The inset is a separate plot of the
three-parameter fits with ``A_0 = 1/C`` fixed at the random-guess floor.

The centred coordinate ``s - s_0`` may be negative, but the underlying density
is always physical. Fitted curves are evaluated only for ``0 <= s <= 1`` and
are never extrapolated toward ``s -> -inf``.

Usage:
    .venv/bin/python tools/plot_all_sigmoids.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

A_FLOOR = 0.1
DATASET_CMAPS = {
    "mnist28": "Blues",
    "cifar_pca": "Greys",
    "cifar_resnet": "Reds",
    "sklearn": "Greens",
}
DATASET_PRETTY = {
    "mnist28": "MNIST 28x28",
    "cifar_pca": "CIFAR-10 PCA-200",
    "cifar_resnet": "CIFAR-10 ResNet18",
    "sklearn": "sklearn digits",
}
METHOD_LS = {"magnitude": "-", "random": "--", "wanda": ":"}
DIR_RE = re.compile(r"(.+)_(magnitude|random|wanda)$")


def _parse_dir(path):
    match = DIR_RE.match(os.path.basename(os.path.dirname(path)))
    return (match.group(1), match.group(2)) if match else (None, None)


def _sigmoid(s, A_inf, s_0, beta):
    s = np.asarray(s, dtype=float)
    exponent = -beta * (s - s_0)
    return A_FLOOR + (A_inf - A_FLOOR) / (
        1.0 + np.exp(np.clip(exponent, -500, 500))
    )


def _load_cells(root):
    """Load and average repeats from the current ``assets/`` layout."""
    pattern = os.path.join(
        os.fspath(root),
        "assets",
        "unstructured_pruning",
        "*",
        "scaling_results.json",
    )
    cells_out = []
    for path in sorted(glob.glob(pattern)):
        dataset, method = _parse_dir(path)
        if dataset is None:
            continue
        with open(path) as handle:
            rows = json.load(handle)

        by_architecture = defaultdict(list)
        for row in rows:
            r2 = row.get("sigmoid_R2_v2")
            if r2 is None or not np.isfinite(r2):
                continue
            by_architecture[(row["H"], row["L"])].append(row)

        for (width, depth), repeats in by_architecture.items():
            densities = np.asarray(repeats[0].get("densities"), dtype=float)
            accuracy_rows = [repeat.get("accs_mean") for repeat in repeats]
            valid_raw = (
                densities.ndim == 1
                and densities.size > 0
                and all(
                    accuracy is not None and len(accuracy) == len(densities)
                    for accuracy in accuracy_rows
                )
            )
            accuracy_mean = (
                np.mean(np.asarray(accuracy_rows, dtype=float), axis=0)
                if valid_raw
                else None
            )
            cells_out.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "H": int(width),
                    "L": int(depth),
                    "n_params": int(np.mean([row["n_params"] for row in repeats])),
                    "A_inf": float(
                        np.mean([row["sigmoid_A_inf_v2"] for row in repeats])
                    ),
                    "s_0": float(
                        np.mean([row["sigmoid_s_0_v2"] for row in repeats])
                    ),
                    "beta": float(
                        np.mean([row["sigmoid_beta_v2"] for row in repeats])
                    ),
                    "R2": float(
                        np.mean([row["sigmoid_R2_v2"] for row in repeats])
                    ),
                    "densities": densities if valid_raw else None,
                    "accs_mean": accuracy_mean,
                }
            )
    return cells_out


def _physical_fit_curve(cell, *, x_min, x_max, n_x):
    """Sample a centred fit without leaving the physical density interval."""
    lower = max(float(x_min), -float(cell["s_0"]))
    upper = min(float(x_max), 1.0 - float(cell["s_0"]))
    if upper < lower:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = np.linspace(lower, upper, int(n_x))
    density = cell["s_0"] + x
    return x, _sigmoid(density, cell["A_inf"], cell["s_0"], cell["beta"])


def _style_for(cell, architecture_ranks):
    dataset = cell["dataset"]
    ranks = architecture_ranks[dataset]
    denominator = max(1, len(ranks) - 1)
    intensity = 0.25 + 0.70 * (ranks[cell["n_params"]] / denominator)
    color = plt.colormaps[DATASET_CMAPS.get(dataset, "Greys")](intensity)
    return color, METHOD_LS.get(cell["method"], "-")


def build_combined_figure(
    cells,
    *,
    x_min=-0.5,
    x_max=0.5,
    n_x=400,
    raw_alpha=0.28,
    fit_alpha=0.32,
    line_width=0.55,
):
    """Return a figure with raw curves on the main axes and fits in the inset."""
    architectures = defaultdict(set)
    for cell in cells:
        architectures[cell["dataset"]].add(cell["n_params"])
    architecture_ranks = {
        dataset: {value: rank for rank, value in enumerate(sorted(values))}
        for dataset, values in architectures.items()
    }

    fig, main_ax = plt.subplots(figsize=(9.5, 6.0), facecolor="white")
    inset_ax = inset_axes(
        main_ax,
        width="39%",
        height="40%",
        loc="upper left",
        borderpad=1.2,
    )

    for cell in cells:
        color, line_style = _style_for(cell, architecture_ranks)

        if cell["densities"] is not None and cell["accs_mean"] is not None:
            centred_density = np.asarray(cell["densities"]) - cell["s_0"]
            order = np.argsort(centred_density)
            centred_density = centred_density[order]
            accuracy = np.asarray(cell["accs_mean"])[order]
            keep = (centred_density >= x_min) & (centred_density <= x_max)
            if np.count_nonzero(keep) >= 2:
                line = main_ax.plot(
                    centred_density[keep],
                    accuracy[keep],
                    color=color,
                    linestyle=line_style,
                    linewidth=line_width,
                    alpha=raw_alpha,
                    marker=".",
                    markersize=1.8,
                    markeredgewidth=0,
                )[0]
                line.set_gid("raw-data")

        fit_x, fit_accuracy = _physical_fit_curve(
            cell, x_min=x_min, x_max=x_max, n_x=n_x
        )
        if fit_x.size:
            line = inset_ax.plot(
                fit_x,
                fit_accuracy,
                color=color,
                linestyle=line_style,
                linewidth=line_width + 0.15,
                alpha=fit_alpha,
            )[0]
            line.set_gid("sigmoid-fit")

    main_ax.axvline(0.0, color="0.5", linewidth=0.5, linestyle=":")
    main_ax.axhline(A_FLOOR, color="0.7", linewidth=0.4)
    main_ax.set_xlabel(r"$s - s_0$")
    main_ax.set_ylabel(r"$A(s)$")
    main_ax.set_title(
        "Experimental pruning curves centred at $s_0$ "
        f"($A_0 = 1/C$ fixed, N = {len(cells)})"
    )
    main_ax.set_xlim(x_min, x_max)
    main_ax.set_ylim(-0.02, 1.02)
    main_ax.grid(True, alpha=0.3, linewidth=0.4)

    inset_ax.axvline(0.0, color="0.4", linewidth=0.6, linestyle=":")
    inset_ax.axhline((A_FLOOR + 1.0) / 2.0, color="0.4", linewidth=0.6, linestyle=":")
    inset_ax.set_xlim(x_min, x_max)
    inset_ax.set_ylim(-0.02, 1.02)
    inset_ax.tick_params(labelsize=7, length=2, pad=1)
    inset_ax.grid(True, alpha=0.25, linewidth=0.3)
    inset_ax.set_title("Fitted accuracy curves", fontsize=8, pad=2)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.7)

    dataset_handles = []
    for dataset in sorted(DATASET_PRETTY):
        if dataset not in architectures:
            continue
        cmap = plt.colormaps[DATASET_CMAPS.get(dataset, "Greys")]
        dataset_handles.append(
            Line2D(
                [0],
                [0],
                color=cmap(0.78),
                linewidth=2.2,
                label=DATASET_PRETTY[dataset],
            )
        )
    method_handles = [
        Line2D([0], [0], color="0.25", linewidth=1.4, linestyle=style, label=method)
        for method, style in METHOD_LS.items()
    ]
    method_legend = main_ax.legend(
        handles=method_handles,
        title="Pruning (linestyle)",
        loc="lower right",
        bbox_to_anchor=(0.998, 0.012),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.92,
    )
    main_ax.add_artist(method_legend)
    main_ax.legend(
        handles=dataset_handles,
        title="Dataset (hue)",
        loc="lower right",
        bbox_to_anchor=(0.998, 0.215),
        fontsize=8,
        title_fontsize=8,
        framealpha=0.92,
    )
    return fig, main_ax, inset_ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=_REPO, help="repository root")
    parser.add_argument(
        "--output",
        default=os.path.join(
            _REPO, "assets", "unstructured_pruning", "sigmoid_overlay.png"
        ),
    )
    parser.add_argument("--x-min", type=float, default=-0.5)
    parser.add_argument("--x-max", type=float, default=0.5)
    parser.add_argument("--n-x", type=int, default=400)
    parser.add_argument("--r2-min", type=float, default=0.85)
    parser.add_argument("--raw-alpha", type=float, default=0.28)
    parser.add_argument("--fit-alpha", type=float, default=0.32)
    parser.add_argument("--line-width", type=float, default=0.55)
    parser.add_argument("--exclude", nargs="*", default=["cifar_pca"])
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    all_cells = _load_cells(args.root)
    excluded = set(args.exclude)
    cells = [
        cell
        for cell in all_cells
        if cell["R2"] >= args.r2_min and cell["dataset"] not in excluded
    ]
    if not cells:
        raise SystemExit("No cells passed the requested dataset and R^2 filters.")

    print(
        f"{len(cells)} of {len(all_cells)} cells pass "
        f"R^2_v2 >= {args.r2_min} after exclusions"
    )
    counts = defaultdict(int)
    for cell in cells:
        counts[(cell["dataset"], cell["method"])] += 1
    for (dataset, method), count in sorted(counts.items()):
        print(f"  {dataset:14s} {method:9s} N={count:4d}")

    figure, _, _ = build_combined_figure(
        cells,
        x_min=args.x_min,
        x_max=args.x_max,
        n_x=args.n_x,
        raw_alpha=args.raw_alpha,
        fit_alpha=args.fit_alpha,
        line_width=args.line_width,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, facecolor="white")
    plt.close(figure)
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
