# `temperature_pruning/` — weight-noise vs. pruning experiment

> **Naming note:** the package is called `temperature_pruning` for historical
> reasons, but the mechanism is **Gaussian weight noise**, not Boltzmann
> temperature. The noise amplitude `σ` maps to the Sherrington–Kirkpatrick
> bond-disorder amplitude `J₁` in the paper's framework.

Sweeps weight-noise amplitude `σ` jointly with pruning density `s` on trained
checkpoints, fits a sigmoid per `(H, L, repeat, σ)`, extracts the critical
density `p_c(σ)`, and tests the SK prediction that the critical line is
**purely quadratic** (no linear term). Depends on `unstructured_pruning`
(`load_fc_checkpoint`, `evaluate_masked_accuracy`, `random_masks`) and
`pruning` (`fit_sigmoid`).

## Layout

```
temperature_pruning/
├── core.py  noise.py  analysis.py  main.py   (kept at root; main is the CLI entry)
├── plotting/    plots, plot_beta_vs_sigma
└── extensions/  fss_check, seed_sweep_ma1
```

## Files

| File | Role |
|---|---|
| `core.py` | `(σ, density)` sweep runner on checkpoints; per-cell sigmoid fits; extracts `p_c(σ)`. |
| `noise.py` | The weight-noise knob: `add_weight_noise` adds `N(0, σ²·rms(Wₗ)²)` to hidden-layer weights (per-layer RMS-scaled, so `σ` is a fractional perturbation). |
| `main.py` | CLI driver with a per-dataset registry (`sklearn`, `mnist28`, `cifar_resnet`); `--analysis-only` re-renders from the cached JSON. |
| `analysis.py` | Per-cell quadratic fit `p_c(σ)=a+bσ+cσ²`, β-collapse F-regime cutoff, AIC/BIC + t-test model comparison vs. the `b=0` SK null, data-collapse diagnostic. |
| `plotting/plots.py` | `accuracy_curves`, `critical_line`, `data_collapse`, `model_comparison` figures. |
| `plotting/plot_beta_vs_sigma.py` | Sigmoid steepness `β` vs. noise amplitude `σ`. |
| `extensions/fss_check/` | Finite-size-scaling check of `β(H)`; `replot.py` builds the manuscript `beta_vs_H.png` (v1). `replot_v2.py` is the `A₀=1/C`-fixed refit variant. |
| `extensions/seed_sweep_ma1/` | Seed-replicate robustness probe. |

## Outputs vs. data

- **Figures + accompanying JSON → `assets/temperature_pruning/<dataset>/` and `assets/temperature_pruning/fss_check/`.**
- Checkpoints are reused from `checkpoints/` (not duplicated here).

## Run

```bash
python -m temperature_pruning.main --dataset sklearn
python -m temperature_pruning.main --dataset mnist28 --analysis-only
python temperature_pruning/extensions/fss_check/replot.py
```
