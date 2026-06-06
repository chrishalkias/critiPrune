# `unstructured_pruning/` — Unstructured (weight-level) pruning and density scaling laws

This subpackage masks **individual weights** of fully-connected ReLU networks at
prescribed densities `s ∈ (0, 1]` and measures the resulting accuracy as a
function of `s`. The recovery curve is again sigmoidal in shape, with a
critical density `s_0` whose dependence on width `H` and depth `L` (and total
parameter count `P`) constitutes the central object of study.

It is the **unstructured / weight-level** counterpart to [`pruning/`](../pruning/),
which prunes whole-neuron paths via top-`K` selection. The two subpackages
share the `FCNetwork` model class and the sigmoid-fitting utilities.

## Layout

```
unstructured_pruning/
├── core.py  methods.py        shared engine + pruning methods (kept at root)
├── runners/    per-dataset scaling drivers + method_comparison + more_combinations (CLI: python -m unstructured_pruning.runners.<name>)
├── analysis/   param_scaling, loss_scaling, heldout_s0_prediction(+multi)
├── plotting/   plot_3d_scaling(+_v2), plot_beta_vs_s0, replot_from_json
├── toy_examples/   analytically tractable minimal experiments
├── extensions/     single_axis_stratified probe
└── checkpoints/    trained weights (data, stays)
```

All generated figures/JSON now write under `assets/unstructured_pruning/`.

---

## Scientific motivation

For each (`H`, `L`) cell, we sweep through a prescribed list of densities
and apply a method-specific mask. Test accuracy follows a sigmoid in `s`:

```
A(s) = A_0 + (A_∞ - A_0) / (1 + exp(-β·(s - s_0)))
```

with `s_0` interpreted as the **critical density** at which the network
transitions from random-guessing to its unpruned regime. Bivariate fits
across the architecture grid yield power laws

```
s_0  = a · H^α · L^γ
β    = a · H^α · L^γ
g_eff = exp(-β) = a · H^α · L^γ
```

and an additional cross-dataset/method comparison checks whether
`s_0 ~ a · P^φ` (pure parameter-count scaling) holds. The empirical answer
is mostly **no**: in FC networks `α < 0` (wider helps pruning tolerance) and
`γ > 0` (deeper hurts), so the H- and L-effects partly cancel when projected
onto total `P`. See `param_scaling.py` for the details and the in-repo
analysis comparing this with Rosenfeld et al. (arXiv:2006.10621).

Representative observed exponents (R² on the bivariate `H, L` fit):

| Dataset       | Method    | `s_0 = a · H^α · L^γ`                   | R²    |
|---------------|-----------|-----------------------------------------|-------|
| sklearn digits | wanda    | `0.252 · H^{-0.348} · L^{0.755}`        | 0.894 |
| MNIST 28×28    | wanda    | `0.322 · H^{-0.439} · L^{0.642}`        | 0.910 |
| CIFAR + PCA    | wanda    | `0.359 · H^{-0.316} · L^{0.602}`        | 0.830 |
| CIFAR + PCA    | magnitude | `0.540 · H^{-0.145} · L^{0.373}`        | 0.809 |

`g_eff` typically fits poorly (very small absolute scale, near noise).

---

## Module layout

| File | Purpose |
|------|---------|
| `core.py`                  | The shared engine: `apply_mask`, `evaluate_masked_accuracy`, `run_scaling_experiment`, sigmoid + power-law fitting, plotting (`scaling_curves.png`, `s0_scaling.png`). Implements **resume / checkpointing** so re-running a job continues from saved `(H, L, repeat)` triples. |
| `methods.py`               | The three pruning strategies: `random_masks`, `magnitude_masks`, `wanda_masks`. Each returns `{s: [mask_set_seed_0, …]}`. Only hidden layers are masked; the classifier head stays intact. |
| `mnist_scaling.py`         | Per-dataset driver: `(H, L)` scan on **sklearn digits** (8×8, 64-dim). |
| `mnist28_scaling.py`       | Same on **MNIST 28×28** (784-dim). |
| `cifar_scaling.py`         | Same on **CIFAR-10 + PCA(200)** features. |
| `cifar_resnet_scaling.py`  | Same on **CIFAR-10 + frozen ResNet18** (512-dim) features. |
| `param_scaling.py`         | Cross-dataset/method analysis: fits `s_0 ~ a · P^φ` across the 4 datasets × 3 methods grid; produces `assets/unstructured_pruning/param_scaling.png`. |
| `__init__.py`              | Public API re-exports. |

---

## Sigmoid fit interpretation

| symbol | meaning |
|--------|---------|
| `A_∞`  | high-density plateau ≈ unpruned val accuracy |
| `A_0`  | low-density plateau ≈ random-chance level |
| `s_0`  | inflection density — half-recovered point |
| `β`    | steepness of the transition (inverse-temperature) |
| `g_eff = exp(-β)` | "effective coupling constant" of the phase transition |

`s_0` is the central quantity; smaller is better (the network can be pruned
to a smaller fraction of weights before collapsing).

---

## Pruning strategies (`methods.py`)

`UNSTRUCTURED_METHODS` registers three method keys:

```python
UNSTRUCTURED_METHODS = {
    'random':    'Random (unstructured)',
    'magnitude': 'Weight magnitude (unstructured)',
    'wanda':     'WANDA (unstructured)',
}
```

| key | algorithm | data needed | n_seeds |
|-----|-----------|-------------|---------|
| `random`    | Per-element Bernoulli with keep-prob `s`; one mask set per seed. | none | typically 3 |
| `magnitude` | Global-per-layer top-`s·N` weights by `\|W\|`. Deterministic. | none | 1 (replicated) |
| `wanda`     | Row-wise top-`s·fan_in` by score `\|W_ij\| · \|\|x_j\|\|_2` with `\|\|x_j\|\|` from a calibration set (Sun et al., 2023, adapted to FC layers). | calibration `X` | 1 (replicated) |

Each method returns `{s: [mask_set_seed_0, mask_set_seed_1, …]}` where a
*mask set* is a list of one float32 `{0,1}` tensor per hidden layer with the
shape of `layer.weight`.

A `taylor_masks` stub exists but raises `NotImplementedError`.

---

## Core engine: `core.py`

### `DEFAULT_DENSITIES`

```python
[0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
```

A 15-point logarithmic-ish sweep over `(0, 1]` chosen to resolve the sigmoid
transition without wasting compute on the saturated tails.

### Key functions

- `apply_mask(model, masks)` — returns a deep-copied `FCNetwork` with hidden
  layer weights multiplied element-wise by the masks. Original is untouched.
- `evaluate_masked_accuracy(model, X_test, y_test, mask_sets)` — for each
  density, evaluates accuracy over all seed realisations and returns
  `{s: (mean, std)}` plus the unpruned reference.
- `run_scaling_experiment(data, *, input_size, h_values, l_values, method,
  output_dir, dataset_label, ...)` — orchestrates the full `(H, L, repeat)`
  scan, calls `_build_masks`, fits sigmoids per cell, fits 2D power laws over
  the grid, writes JSON + PNG outputs.
- `fit_scaling_laws(results, min_r2=0.80)` — bivariate `a · H^α · L^γ` fit
  for `s_0`, `β`, `g_eff`. Filters cells with `sigmoid_R² > min_r2`.

### Repeats and resumability

`run_scaling_experiment(..., n_repeats=N)` runs each `(H, L)` cell `N` times
with independent training seeds (`seed + 1000·r`) so the bivariate scaling-law
fit sees `N×` more datapoints and variance is quantified via mean ± std bars
on `s0_scaling.png`.

After every `(H, L)` bundle the engine appends to `output_dir/scaling_results.json`.
On a re-run, the file is loaded back and the set of finished
`(H, L, repeat)` triples is skipped — so an interrupted job (or one that
needs an extended grid) resumes cleanly. Trained weights for each cell are
cached under `unstructured_pruning/checkpoints/<output_dir_name>/H{H}_L{L}_r{r}.pt`.

> The `checkpoints/` directory can grow to several GB and is excluded from
> the repository via `.gitignore`. Delete it freely if disk pressure is an issue;
> the JSON results stay intact and the next run will simply retrain those cells.

### Plots

Two PNGs per `(dataset, method)` cell:

| filename            | content |
|---------------------|---------|
| `scaling_curves.png` | Top row: per-`L` panels of `A(s)` with errorbars (across mask seeds), sigmoid fits, and a dotted line at the unpruned reference. Bottom: `s_0` vs. `H` lines per `L` (mean ± std across `n_repeats`) with overlaid bivariate fit. |
| `s0_scaling.png`     | Left: same `s_0` vs. `H` view; right: `s_0` heatmap over the `(H, L)` grid with mean ± std cell labels. |

---

## Per-dataset drivers

All four drivers are thin wrappers around `run_scaling_experiment`. They
share `--method {random,magnitude,wanda}`, `--output-dir`, `--n-repeats` CLI
flags and write to `assets/unstructured_pruning/unstructured_figures_<dataset>_<method>/`.

| Driver                    | Dataset                          | Input dim | Default `H × L`                                         | Train epochs | Calib `X` for WANDA |
|---------------------------|----------------------------------|-----------|---------------------------------------------------------|--------------|---------------------|
| `mnist_scaling.py`        | sklearn digits (8×8)             | 64        | `[8,12,16,20,24,32,40,48,56,64,80,96] × [1..10]`        | 300 / 500    | training set        |
| `mnist28_scaling.py`      | MNIST 28×28                      | 784       | `[64,96,128,192,256,384,512] × [2..8,10]`               | 300          | training set        |
| `cifar_scaling.py`        | CIFAR-10 + PCA(200)              | 200       | (same large grid as cifar_resnet)                       | 300          | training set        |
| `cifar_resnet_scaling.py` | CIFAR-10 + ResNet18 features     | 512       | `[64,96,128,192,256,384,512] × [2..8,10]`               | 300          | training set        |

CIFAR drivers cache features under `$FEATURE_CACHE_DIR` (default
`/tmp/cifar_features`); MNIST 28×28 caches under `$MNIST_DATA_DIR` (default
`/tmp/mnist28`).

`val_acc_floor` (default `0.15` or `0.20`) skips cells that fail to train above
chance; this happens occasionally for very small `H` on harder datasets.

---

## Cross-dataset analysis: `param_scaling.py`

Reads every `assets/unstructured_pruning/unstructured_figures_<dataset>_<method>/scaling_results.json`,
keeps rows with `sigmoid_R² > 0.80`, and fits

```
log s_0 = log a + φ · log P
```

per `(dataset, method)` cell, where `P` is the network's total parameter
count. The result is a 4 × 3 panel (`assets/unstructured_pruning/param_scaling.png`) — one
subplot per `(dataset, method)` — with:

- scatter of `(P, s_0)` coloured by depth `L`,
- the OLS power-law fit overlaid in dashed crimson,
- the formula `s_0 = a · P^φ` and adjusted-R² annotated per panel,
- a shared `L`-colourbar.

Across all 12 cells the observed `φ` is much weaker (≈ −0.03 to −0.16) than
the clean `s_0 ~ P^δ` reported for ResNets in Rosenfeld et al. (arXiv:2006.10621).
This is consistent with FC architectures where width and depth contribute
in opposite directions to pruning tolerance.

---

## How to run

From the repository root:

```bash
# Single (dataset, method) cell — resumes if scaling_results.json exists
python -m unstructured_pruning.runners.mnist_scaling        --method wanda --n-repeats 3
python -m unstructured_pruning.runners.mnist28_scaling      --method magnitude
python -m unstructured_pruning.runners.cifar_scaling        --method wanda
python -m unstructured_pruning.runners.cifar_resnet_scaling --method random

# After all 12 (dataset, method) cells have run, regenerate the cross-dataset figure
python -m unstructured_pruning.analysis.param_scaling
```

CLI flags (common to all four dataset drivers):

| flag            | default     | meaning                                              |
|-----------------|-------------|------------------------------------------------------|
| `--method`      | `random`    | one of `{random, magnitude, wanda}`                  |
| `--output-dir`  | autogen     | where JSON / PNG land                                |
| `--n-repeats`   | `1`         | independent train-mask-fit trials per `(H, L)` cell  |

### Cluster usage

`u_scripts/submit.sh` submits all `dataset × method` jobs to SLURM with
per-dataset walltime presets. `N_REPEATS` is propagated as both a Python
flag and a walltime multiplier:

```bash
N_REPEATS=3 DATASETS="cifar_pca cifar_resnet" METHODS="wanda magnitude" \
    bash u_scripts/submit.sh
```

### Dependencies

`numpy`, `torch`, `scipy`, `matplotlib`, `scikit-learn`, `torchvision` (for
MNIST / CIFAR raw data and the frozen ResNet18). The package shares
`FCNetwork`, `sigmoid_fn`, and `fit_sigmoid` with `pruning/` — that import
is lazy to avoid circularity.

---

## Output directory layout

```
assets/unstructured_pruning/
├── param_scaling.png                                    ← cross-dataset φ figure
├── unstructured_figures_<dataset>_<method>/
│   ├── scaling_results.json     # one row per (H, L, repeat); sigmoid params + R²
│   ├── scaling_laws.json        # fitted bivariate exponents for s_0, β, g_eff
│   ├── scaling_curves.png       # recovery curves + s_0 vs. H
│   └── s0_scaling.png           # s_0 vs. H + heatmap
└── ...
```

`<dataset>` ∈ `{sklearn, mnist28, cifar_pca, cifar_resnet}`,
`<method>` ∈ `{random, magnitude, wanda}` ⇒ 12 cells total.

---

## Relationship to the rest of the repo

- Reuses `pruning.pruning.FCNetwork`, `sigmoid_fn`, `fit_sigmoid` from the
  sister [`pruning/`](../pruning/) subpackage.
- The `data` tuple expected by `run_scaling_experiment` is the same
  `(X_tr, X_val, X_te, y_tr, y_val, y_te)` produced by the loaders in
  `pruning/{mnist,mnist28,cifar}_scaling.py`.
- Aggregated assets used in the top-level README live under
  `assets/legacy/<dataset>/`.
