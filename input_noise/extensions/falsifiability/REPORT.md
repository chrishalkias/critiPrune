# Falsifiability re-analysis of the eta = 1 - xi claim

Source: `input_noise/results_cluster_all.json` (3224 cells after excluding ['cifar_pca']); 331 of those cells are at L = 2.

## Verdicts (one-liners)

- **A1 (prior prediction)**: **PARTIAL** — median per-cell RMS(eta) between predicted and measured contour, using sigma^2(1) extracted *only* from the s=1 column: 0.248 all cells, 0.156 at L = 2 (self-referential L=2 baseline: 0.108).
- **A2 (signed mean residual at L = 2, middle range)**: **YES** — cell-count-weighted <eta - (1 - xi)> in xi in [0.3, 0.7] is -0.043 (95% CI [-0.052, -0.035]; 328 L=2 cells voting).
- **A3 (null-control)**: **PARTIAL** — measured L=2 median RMS to framework line = 0.085; null surrogate sigma^2~s gives 0.084 (ratio 0.99x).

## A1. Prior-prediction reframe

Per-cell procedure (no contour leakage):

1. Fit a logistic A(sigma^2) to the s = 1 column of the joint grid alone.
2. Invert at A = 0.5 to get sigma^2(1)_prior.
3. Predict the full contour via sigma^2(s) = s * sigma^2(1)_prior - (1-s) * <x^2>.
4. Compare predicted vs measured iso-A contour points.

| dataset | n_cells with prior fit | median RMS(eta) | median RMS(sigma^2) |
|---|---:|---:|---:|
| CIFAR-10 ResNet18 | 881 | 0.236 | 1.122 |
| MNIST 28x28 | 438 | 0.255 | 1.895 |
| sklearn digits | 385 | 0.261 | 0.735 |

**Verdict A1**: PARTIAL. verdict: YES if median L=2 RMS(eta) <= 1.05*0.108 (self-referential L=2 baseline); PARTIAL if <= 1.5*0.108; NO otherwise.

## A2. Signed mean residual at L = 2

The framework predicts eta = 1 - xi point-by-point at L = 2 (where the linearised SNR derivation is exact). A non-zero signed mean residual in the middle range xi in [0.3, 0.7] (away from the geometrically-forced endpoints) is the cleanest falsifiable shape prediction.

| dataset | L=2 cells | weighted <signed resid> (mid xi) |
|---|---:|---:|
| CIFAR-10 ResNet18 | 91 | -0.021 |
| MNIST 28x28 | 105 | -0.007 |
| sklearn digits | 132 | -0.099 |

**Pooled cell-count-weighted signed mean (mid xi)**: -0.043 (95% CI [-0.052, -0.035], n_cells = 328).

**All-xi (reference)**: -0.022 (95% CI [-0.024, -0.020], n_cells = 331).

**Verdict A2**: YES. YES means the framework over- or under-predicts eta systematically (signed bias) in the geometrically unconstrained middle range; NO means the framework line passes through the cell-count-weighted middle.

## A3. Baseline-monotone null control

Per L=2 cell, replace the measured iso-A contour with the monotone surrogate sigma^2_null(s) = sigma^2(1) * s^k for k in {0.5, 1, 2}. Each surrogate satisfies the boundary conditions sigma^2_null(0) = 0 and sigma^2_null(1) = sigma^2(1) but is *not* the framework prediction. Apply the same (xi, eta) rescaling and compare RMS to eta = 1 - xi.

| k | median RMS to line | median signed mean | ratio vs measured |
|---:|---:|---:|---:|
| 0.5 | 0.248 | +0.208 | 2.93x |
| 1.0 | 0.084 | +0.070 | 0.99x |
| 2.0 | 0.108 | -0.095 | 1.28x |

**Measured L=2 median RMS (reference)**: 0.085 (over 331 L=2 cells).

**Verdict A3**: PARTIAL. YES iff every k in {0.5, 1, 2} gives a null median RMS to the framework line that is at least 1.5x the measured L=2 median RMS.

## Manuscript hook

All three verdicts feed §V.B of the Phase 4 revision: A1 turns the upper-envelope claim into a *prior* shape prediction; A2 quantifies whether the prediction is point-by-point or only upper-envelope; A3 rules out (or fails to rule out) the generic-monotone artefact that R1.C1 and R4.MAJOR-1 flagged.

Figures: `prior_prediction.png` (A1), `signed_residual_L2.png` (A2), `null_control.png` (A3). Machine-readable per-cell numbers in `results.json`.
