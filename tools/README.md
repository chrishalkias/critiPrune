# `tools/` — cross-cutting post-processing utilities

Generic helpers that operate on **any** scaling-results JSON with the
`sigmoid_*` schema (not tied to one dataset or experiment). They read the
per-`(dataset, method)` results under `assets/unstructured_pruning/` and refresh
fits or overlay plots.

## Files

| File | Role |
|---|---|
| `refit_sigmoids.py` | Three-parameter refit with `A₀ := 1/C` fixed. Refreshes the `sigmoid_*_v2` fields in the cached scaling JSONs. |
| `plot_all_sigmoids.py` | Builds one combined figure: raw centred accuracy curves on the main axes and physical-domain v2 fits in the inset. |
| `test_sigmoid_tools.py` | Regression coverage for current asset discovery, fixed-floor fitting, physical-density clamping, and main/inset composition. |

The old four-parameter fit path was removed because its unconstrained
`s -> -inf` asymptote is outside the physical pruning domain. The centred
coordinate `s - s₀` may be negative, but all fitted samples satisfy
`0 <= s <= 1`.

## Outputs

Figures/JSON write to `assets/unstructured_pruning/...`. These tools never
train or sweep — they only post-process existing results.

## Run

```bash
python3 tools/refit_sigmoids.py
python3 tools/plot_all_sigmoids.py
```
