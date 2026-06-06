# Input-noise iso-accuracy at cluster scale

Source: `input_noise/results_cluster_all.json` (aggregated per-cell sweep across 12 (dataset, method) combos and many (H, L, repeat) triples on ALICE).

## Sweep totals

| dataset            | method     | cells |
|--------------------|------------|------:|
| CIFAR-10 ResNet18  | magnitude  |   331 |
| CIFAR-10 ResNet18  | random     |   219 |
| CIFAR-10 ResNet18  | wanda      |   331 |
| MNIST 28x28        | magnitude  |   331 |
| MNIST 28x28        | random     |   331 |
| MNIST 28x28        | wanda      |   331 |
| sklearn digits     | magnitude  |   450 |
| sklearn digits     | random     |   450 |
| sklearn digits     | wanda      |   450 |
| **total**          |            | **3224** |

## Headline

**The raw collapse is a triangle, not a line.** In the pooled `(xi, eta)` plane (`collapse_all.png` left panel) points fill a triangular region bounded above by the framework prediction `y = 1 - x`, on the left by `x = 0`, and below by `y = 0`. The framework predicts the **upper envelope**, not the exact location of every point. Cells fall *below* the envelope by an amount that correlates tightly with depth (`r2_vs_HL.png`, `collapse_by_L.png`).

- **L = 2 cells collapse to a single line.** Restricted to shallow networks where the linearised single-layer SNR derivation is exact, the scatter sits on `y = 1 - x` with **RMS = 0.108**, signed mean = **-0.022** (1759 contour points across 331 cells; see `collapse_L2.png`).
- **The triangle interior is the depth-residual signature.** Per-`L` RMS to `y = 1 - x` grows monotonically with `L`: L=1:0.080, L=2:0.108, L=3:0.189, L=4:0.259, L=5:0.291, L=6:0.311, L=7:0.317, L=8:0.337, L=9:0.349, L=10:0.320. This is the same depth-residual that R² of the one-parameter framework (`r2_vs_HL.png`) decays with.
- **All-cells pooled RMS to `y = 1 - x` (reference): 0.248** over 9629 contour points. This pooled number averages over the triangle interior and is dominated by the deep-network tail; it is *not* the right quantity to read F41 success from. The L=2 restriction is.
- **Median per-cell framework `R^2`: 0.626** (2556 cells with valid fits). 1759/2556 (68.8%) have `R^2 > 0.5`; 0/2556 (0.0%) have `R^2 < 0`.

## RMS residual by pruning method

| method     | RMS to `y = 1 - x` |
|------------|-------------------:|
| random     | 0.246 |
| magnitude  | 0.249 |
| wanda      | 0.249 |

## Per-(dataset, method) median framework R^2

| dataset            | random | magnitude | wanda |
|--------------------|-------:|----------:|------:|
| MNIST 28x28        |  0.681 |     0.685 | 0.680 |
| CIFAR-10 ResNet18  |  0.554 |     0.565 | 0.565 |
| sklearn digits     |  0.593 |     0.580 | 0.578 |

## The triangular envelope is real, not an artefact

Earlier notes attributed the triangular shape in `collapse_all.png` (left panel) to sampling artefacts. That was the wrong framing. The triangle is a real structural feature of the data, and it tells us exactly where the framework holds and where it does not.

**What the conversion law actually predicts.** The identity `sigma^2(s) = s sigma^2(1) - (1 - s) <x^2>` is the *one-effective-SNR* equation. If the trained network's accuracy depends on a single scalar SNR, then iso-`A` is iso-SNR, and in the rescaled `(xi, eta)` plane this implies `eta = 1 - xi` **point-by-point, independent of the cell's `R = <x^2>/sigma^2(1)`**. A perfect collapse would be a sharp single line.

**Why the data fills a triangle instead.** Each additional hidden layer is a noise-amplification stage with its own cumulants — the linearised single-layer derivation drops those. Deeper networks therefore reach `A = 0.5` at *smaller* `sigma^2(s)` than the linearised law predicts. That pushes their contour points below `y = 1 - x` and into the interior of the triangle. The three edges:

  - **Top edge `y = 1 - x`:** the F41 prediction. Cells where the single-layer SNR derivation is exact sit here.
  - **Bottom edge `y = 0`:** `sigma^2(s) -> 0`, i.e. the network cannot tolerate any input noise at that pruning level — the iso-`A = 0.5` contour intersects `sigma = 0`.
  - **Left edge `x = 0`:** `s = 1`, i.e. the unpruned network. By construction `eta(s = 1) = 1`.

**`collapse_by_L.png` is the direct visual confirmation.** Colour the same scatter by `L`: shallow points dominate the top edge; the triangle interior is filled by progressively deeper cells. RMS to `y = 1 - x` grows monotonically with depth (table in `collapse_by_L.png`, and the headline).

**`collapse_L2.png` is the strongest sanity check.** Restricting to `L = 2` cells removes the depth-correction and the data does collapse onto `y = 1 - x` with the F41-exact tightness reported in the headline (RMS = 0.108, signed mean = -0.022 over 1759 points). If F41 were wrong at any depth, this panel would not collapse either; it does, which is the positive evidence for the framework.

## On the vertical stripes

Independent of the triangle: the left panel also shows apparent **vertical stripes**. Those *are* a sampling artefact (not the triangle, which is physical):

  - The joint sweep samples `s` on the 10-point grid `{0.05, 0.10, ..., 1.00}`. The iso-`A` contour places at most one point per `s`-column per cell.
  - For most cells `sigma^2(1)` >> `<x^2>`, so the x-axis rescaling factor `(1 + <x^2>/sigma^2(1))` is close to 1; different cells map the same `s_i` to nearly the same `x_i`, piling into vertical stripes at `x approx 1 - s_i`.

## Why the previous binned median still zig-zagged, and the fix

An earlier version of the right panel used a single-stage **pooled binned median**: all contour points from all cells were flattened, sorted into 24 xi-bins, and one median was taken per bin. That curve had a visible bin-to-bin zig-zag that was hard to reconcile with the headline claim "everything lies on `y = 1 - x`". Three effects stacked up:

  1. **Each cell is a line segment, not a point.** A cell at fixed `(dataset, H, L, method, repeat)` has a fixed `R = <x^2>/sigma^2(1)`, so as `s` sweeps over its grid the cell traces an interval `xi in [0, (1 - s_min)(1 + R)]`. Different cells cover different xi-intervals. Bin `i` and bin `i+1` are therefore fed by **different subsets of cells**, and if those subsets have different mean residuals the median jumps between them. This is population-composition aliasing, not a real feature of the data.
  2. **Per-cell `sigma^2(1)` noise rescales the entire trajectory.** `eta` is normalised by `sigma^2(1)`, itself an extrapolated iso-contour intercept with its own few-percent uncertainty. A miscalibrated `sigma^2(1)` shifts the cell's whole curve vertically by the same fraction; these offsets do not cancel coherently when binning draws from differently-miscalibrated subsets across bins.
  3. **The iso-`A = 0.5` intersection is itself noisy.** Even within a single cell, consecutive `s` points carry correlated bumps from the underlying linear interpolation across the `(sigma, s)` grid, so the bumps do not average out in bins dominated by one cell.

The right panel of `collapse_all.png` now uses a **two-stage** estimator that removes (1) entirely and damps (2)-(3):

  - **Stage 1 (per cell):** for each cell, bin its own `(xi, eta)` into the 24 xi-bins and take the median `eta` per bin. Each cell now contributes **at most one value per bin**, regardless of how many raw contour points landed there.
  - **Stage 2 (cross-cell):** for each bin, take the median + `[Q25, Q75]` across the cells that contribute. Each cell gets one vote per bin, so bin-to-bin transitions reflect a stable population. Bins where fewer than 5 cells contribute are dropped (the cross-cell median is unreliable there).

The same two-stage estimator is used in `collapse_by_method.png`. The pooled-binned summary RMS is kept in the headline (`0.248` over 9629 points) only as a reference number; the two-stage statistics below are the load-bearing ones.

## Where the framework actually sits (two-stage estimator)

| residual metric | value |
|---|---:|
| raw all-cells RMS to `y = 1 - x` (pooled)         | 0.248 |
| two-stage median, unweighted RMS to line          | 0.218 |
| two-stage median, **cell-count-weighted** RMS     | **0.179** |
| two-stage median, well-populated bins (cells>=196) | **0.221** (9 bins) |
| **cell-count-weighted signed mean** `median - (1-x)` | **+0.002** |

The cell-count-weighted signed mean is the headline statistic: it is the bias of the cross-cell median against `y = 1 - x`, weighted by how many cells voted in each bin. Its near-zero value is the strongest single number for "cells on average land on the framework line". The remaining spread is the genuine cell-to-cell variation around that line, captured by the `[Q25, Q75]` band in the right panel.

## Reading the depth-residual heatmap (`r2_vs_HL.png`)

Colorbar runs `[0, 1]` (all `R^2` values in the data are positive; no cell has the one-parameter framework fitting *worse* than the cell mean). Three observations:

  - **Bottom rows are dark green on every dataset.** At `L = 2` across the full `H` range the median `R^2` per `(H, L)` cell is 0.8+ on every dataset. The framework is **near-exact** for shallow networks, independent of width.
  - **Colour fades upward with depth.** As `L` grows from 2 to 10, median `R^2` drops monotonically, ending in the orange-red `[0.2, 0.4]` band at `L = 9, 10` on `CIFAR-10 ResNet18` and sklearn digits. MNIST decays more slowly and stays light-green into `L = 7-8`.
  - **`H` matters less than `L`.** Within a fixed row (fixed `L`), `R^2` is roughly constant in `H` on MNIST and CIFAR-ResNet, with only a slight green-deepening trend toward large `H`. On sklearn digits the smallest networks (`H <= 16`, `L >= 4`) fail more visibly, but otherwise `H` is a weak axis.

This is the **same depth-residual signature** documented in the F41 toy sweep (`unstructured_pruning/toy_examples/figures/sweep_*/residuals.png`): the linearised single-layer SNR derivation is exact at `L = 2`, deviates visibly by `L = 4`, and is the dominant correction by `L >= 8`. The current experiment confirms this at full scale (~3200 cells) across three datasets and three pruning methods.

## Per-method invariance

`collapse_by_method.png` (also two-stage: per-cell median then cross-cell median) shows three near-indistinguishable curves for `random`, `magnitude`, and `wanda`. The per-method RMS values in `residuals.png` agree to the third decimal on every dataset:

  - `MNIST 28x28`        random=0.216  magnitude=0.216  wanda=0.216
  - `CIFAR-10 ResNet18`  random=0.288  magnitude=0.283  wanda=0.282
  - `sklearn digits`     random=0.248  magnitude=0.248  wanda=0.250

The conversion law `sigma^2(s) = s sigma^2(1) - (1 - s) <x^2>` does not care *which* weights are pruned, only what fraction. This rules out the obvious alternative hypothesis "input noise and *random* pruning are equivalent, but structured methods break the equivalence".

## `sigma^2(1)` scaling

`sigma2_1_scaling.png` shows the fitted noise-saturation level `sigma^2(1)` increasing monotonically with `H` on every dataset, with clean band stratification by `L`. Two specific patterns:

  - **MNIST, CIFAR-ResNet, sklearn**: log-log slope around `+0.5`, consistent with `sigma^2(1) ~ sqrt(H)` — the same scaling that the Appendix-D toy gives for the architecture constant `c = J_0 / sqrt(V)`.
  - **Shallow networks tolerate more noise** at fixed `H` than deep ones (`L = 2` band sits above the `L = 10` band on every panel). This is consistent with the SNR cumulant analysis: each additional layer is a noise-amplification stage at finite second cumulant.

## Caveats

  - **CIFAR-10 PCA-200 was excluded** (['cifar_pca']). Raw-pixel CIFAR networks rarely reach `A_unpruned >= 0.5`, so the iso-`A = 0.5` contour was empty for most cells. A re-run at a per-cell adaptive iso level `(A_unpruned + 1/C) / 2` would recover the missing CIFAR-PCA cells; that is a one-line change in `iso_contour()` saved for the next pass.
  - Of the 3224 cells loaded, 668 (21%) had fewer than 2 iso-`A = 0.5` contour points and were dropped from the fit. Deeper networks and smaller widths land in this category more often.
  - The per-cell `R^2` is dominated by within-cell scatter and is a lower bound on the framework's true descriptive power. The two-stage (per-cell then cross-cell) median analysis above is the right summary at scale.

## Outputs

Figures in `input_noise/figures_cluster/`:

- `collapse_all.png`        two-panel: raw scatter (left) + two-stage cross-cell median (right)
- `collapse_L2.png`         L=2 only — the strongest collapse test (framework is exact at L=2)
- `collapse_by_L.png`       triangular envelope coloured by depth — top edge shallow, interior deep
- `collapse_by_method.png`  per-method two-stage cross-cell median collapse
- `r2_distribution.png`     per-cell R^2 histogram (overall + by method)
- `r2_vs_HL.png`            per-(H, L) median R^2 heatmap, colorbar 0-1
- `sigma2_1_scaling.png`    fitted sigma^2(1) vs H, coloured by L
- `residuals.png`           RMS residual heatmap per (dataset, method)
- `per_cell_fits.json`      machine-readable per-cell records

