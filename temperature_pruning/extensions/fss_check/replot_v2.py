"""v2 FSS check: uses sigmoid_beta_v2 (3-param fit, A_0=1/C fixed) and
plots beta vs H with panels grouped by method (one row per method),
datasets across columns.

Overwrites the same output PNG name (beta_vs_H_FSS.png) used in the
manuscript by the parent agent's symlink.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/chrischalkias/Projects/critiPrune")
OUT = ROOT / "assets" / "temperature_pruning" / "fss_check"
OUT.mkdir(parents=True, exist_ok=True)

R2_MIN, BETA_LO, BETA_HI = 0.80, 0.1, 1e4

_DENS_DS = [("sklearn", "sklearn_digits"), ("mnist28", "mnist28"),
            ("cifar_pca", "cifar_pca"), ("cifar_resnet", "cifar_resnet")]
SOURCES = [
    (f"assets/unstructured_pruning/unstructured_figures_{slug}_{m}/scaling_results.json",
     label, m)
    for slug, label in _DENS_DS for m in ("random", "magnitude", "wanda")
]

DATASET_PRETTY = {
    "sklearn_digits":  "sklearn digits",
    "mnist28":         "MNIST 28x28",
    "cifar_pca":       "CIFAR-10 PCA-200",
    "cifar_resnet":    "CIFAR-10 ResNet18",
}
METHOD_ORDER = ["random", "magnitude", "wanda"]
DATASET_ORDER = ["sklearn_digits", "mnist28", "cifar_pca", "cifar_resnet"]


def load_long():
    rows = []
    for rel, dataset, method in SOURCES:
        p = ROOT / rel
        if not p.exists():
            continue
        for d in json.load(open(p)):
            beta, r2 = d.get("sigmoid_beta_v2"), d.get("sigmoid_R2_v2")
            if beta is None or r2 is None:
                continue
            if r2 < R2_MIN or not (BETA_LO < beta < BETA_HI):
                continue
            rows.append({"dataset": dataset, "method": method,
                         "H": int(d["H"]), "L": int(d["L"]),
                         "repeat": int(d.get("repeat", 0)),
                         "beta": float(beta), "r2": float(r2)})
    return rows


def fit_power(x, y):
    m = (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 3 or np.unique(x).size < 3:
        return None
    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    var_x = np.sum((lx - lx.mean()) ** 2)
    if var_x <= 0:
        return None
    se = float(np.sqrt(np.sum(resid ** 2) / (x.size - 2) / var_x))
    return float(slope), se, float(intercept)


def aggregate_by_L(rows, dataset, method):
    """Per L: (H array, mean beta over repeats and H bins, slope/SE)."""
    sub = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
    out = []
    for L in sorted({r["L"] for r in sub}):
        cells = [r for r in sub if r["L"] == L]
        Hs = sorted({r["H"] for r in cells})
        ys = [np.mean([r["beta"] for r in cells if r["H"] == h]) for h in Hs]
        out.append({"L": L, "Hs": np.array(Hs, dtype=float),
                    "betas": np.array(ys, dtype=float)})
    return out


def plot_grid_by_method(rows, fname="beta_vs_H_FSS.png"):
    n_rows = len(METHOD_ORDER)
    n_cols = len(DATASET_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.8 * n_rows),
                             squeeze=False, sharex=False, sharey=False)
    for i, method in enumerate(METHOD_ORDER):
        for j, dataset in enumerate(DATASET_ORDER):
            ax = axes[i, j]
            series = aggregate_by_L(rows, dataset, method)
            if not series:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="0.5")
                ax.axis("off")
                continue
            cmap = plt.cm.viridis(np.linspace(0, 1, max(len(series), 2)))
            for c, s in zip(cmap, series):
                ax.plot(s["Hs"], s["betas"], "o-", color=c, ms=4, lw=1.0,
                        label=f"L={s['L']}")
                fit = fit_power(s["Hs"], s["betas"])
                if fit is not None and s["Hs"].size >= 3:
                    p, _, intercept = fit
                    xg = np.linspace(s["Hs"].min(), s["Hs"].max(), 50)
                    ax.plot(xg, np.exp(intercept) * xg ** p, "--",
                            color=c, lw=0.7, alpha=0.7)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(DATASET_PRETTY.get(dataset, dataset), fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("H", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{method}\n" + r"$\beta$", fontsize=9)
            ax.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(r"$\beta(H)$ vs hidden width $H$ at fixed depth $L$ "
                 "(v2 fits, A_0=1/C fixed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def fss_summary(rows):
    """Per-method median of p (beta ~ H^p) for the new fits."""
    p_by_method = {m: [] for m in METHOD_ORDER}
    p_by_dataset = {d: [] for d in DATASET_ORDER}
    triples = 0
    for method in METHOD_ORDER:
        for dataset in DATASET_ORDER:
            for s in aggregate_by_L(rows, dataset, method):
                if s["Hs"].size < 3:
                    continue
                fit = fit_power(s["Hs"], s["betas"])
                if fit is None:
                    continue
                p_by_method[method].append(fit[0])
                p_by_dataset[dataset].append(fit[0])
                triples += 1
    all_p = sum(p_by_method.values(), [])
    return {
        "n_triples": triples,
        "all_p_median": float(np.median(all_p)) if all_p else float("nan"),
        "all_p_iqr": (
            float(np.percentile(all_p, 25)),
            float(np.percentile(all_p, 75))
        ) if all_p else (float("nan"), float("nan")),
        "by_method": {m: float(np.median(v)) if v else float("nan")
                      for m, v in p_by_method.items()},
        "by_dataset": {d: float(np.median(v)) if v else float("nan")
                       for d, v in p_by_dataset.items()},
    }


def main():
    rows = load_long()
    print(f"loaded {len(rows)} rows after v2 R^2 >= {R2_MIN} filter")
    plot_grid_by_method(rows, "beta_vs_H_FSS.png")
    print(f"wrote {OUT / 'beta_vs_H_FSS.png'}")
    s = fss_summary(rows)
    print(f"\nv2 FSS summary across {s['n_triples']} (method, dataset, L) triples:")
    print(f"  all-triples median p = {s['all_p_median']:+.3f}, IQR ["
          f"{s['all_p_iqr'][0]:+.3f}, {s['all_p_iqr'][1]:+.3f}]")
    print("  per-method medians:")
    for m, p in s["by_method"].items():
        print(f"    {m:10s} {p:+.3f}")
    print("  per-dataset medians:")
    for d, p in s["by_dataset"].items():
        print(f"    {d:18s} {p:+.3f}")


if __name__ == "__main__":
    main()
