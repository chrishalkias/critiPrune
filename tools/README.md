# `tools/` — cross-cutting post-processing utilities

Generic helpers that operate on **any** scaling-results JSON with the
`sigmoid_*` schema (not tied to one dataset or experiment). They read the
per-`(dataset, method)` results under `assets/unstructured_pruning/` and refresh
fits or overlay plots.

## Files

| File | Role |
|---|---|
| `refit_sigmoids.py` | Re-fit every sigmoid in all results JSONs with the current `fit_sigmoid` cap; rewrites the JSON fields in place and reports how many cells un-saturate. |
| `refit_sigmoids_v2.py` | Three-parameter refit variant with `A₀ := 1/C` fixed; writes `sigmoid_*_v2` fields alongside v1. Compared in `refit_summary.md`. |
| `plot_all_sigmoids.py` | Overlay of all sigmoid fits across `(dataset, method, H, L)`, each centred at its own `s₀`, to expose the spread in slope `β` and asymptotes. |
| `plot_all_sigmoids_v2.py` | Sibling using the `sigmoid_*_v2` fields (`--mode fit|data`). |
| `refit_summary.md` | Notes on the v1-vs-v2 refit comparison. |

> The `_v2` scripts are kept separate from their v1 counterparts because they
> use a different fit family (`A₀=1/C` fixed) and some of their outputs feed
> manuscript figures; they are documented here rather than merged.

## Outputs

Figures/JSON write to `assets/unstructured_pruning/...`. These tools never
train or sweep — they only post-process existing results.

## Run

```bash
python tools/refit_sigmoids.py
python tools/plot_all_sigmoids.py
```
