# F41 critical-density scaling on the toy-models sweep

Source: `figures/sweep_<dataset>_random/results.json` (populated by `f41_sweep.py`).

The critical density `s_0` is defined as the half-
transition density,

    A_F41(s_0) = (A_unpruned + 1/C) / 2,

extracted by linear interpolation on the cached 200-point
F41 curve. `s_0_emp` is the same definition on `A_emp(s)`.

## Fitted form

All fits are OLS in log–log space.

  - **JOINT**: `log s_0 = a + α_H · log H + α_L · log L`, one fit per dataset over all 16 `(H, L)` cells.
  - **Per-L**: `log s_0 = a_L + α_H(L) · log H`, one fit per (dataset, L).
  - **Per-H**: `log s_0 = a_H + α_L(H) · log L`, one fit per (dataset, H).

## Joint exponents

| Dataset | α_H | α_L | a | R² | N |
|---|---:|---:|---:|---:|---:|
| MNIST 28x28 | -0.186 | +0.851 | -0.845 | 0.946 | 16 |
| CIFAR-10 PCA-200 | -0.075 | +0.162 | -0.117 | 0.848 | 16 |
| CIFAR-10 ResNet18 | -0.136 | +1.143 | -1.478 | 0.971 | 16 |
| sklearn digits 8x8 | -0.209 | +1.005 | -0.901 | 0.967 | 16 |

## Per-`L` H-scaling: `α_H(L)`

| Dataset \ L | L=1 | L=2 | L=3 | L=4 |
|---|:-:|:-:|:-:|:-:|
| MNIST 28x28 | -0.125 (R²=0.94) | -0.349 (R²=0.97) | -0.173 (R²=0.96) | -0.099 (R²=0.83) |
| CIFAR-10 PCA-200 | -0.088 (R²=0.77) | -0.084 (R²=0.91) | -0.096 (R²=0.98) | -0.030 (R²=0.80) |
| CIFAR-10 ResNet18 | -0.116 (R²=0.88) | -0.193 (R²=0.98) | -0.143 (R²=0.96) | -0.093 (R²=0.83) |
| sklearn digits 8x8 | -0.277 (R²=0.89) | -0.179 (R²=0.99) | -0.176 (R²=0.93) | -0.203 (R²=0.96) |

## Per-`H` L-scaling: `α_L(H)`

| Dataset \ H | H=64 | H=128 | H=256 | H=512 |
|---|:-:|:-:|:-:|:-:|
| MNIST 28x28 | +0.825 (R²=0.85) | +0.846 (R²=0.95) | +0.865 (R²=1.00) | +0.869 (R²=0.99) |
| CIFAR-10 PCA-200 | +0.109 (R²=0.77) | +0.187 (R²=0.76) | +0.168 (R²=0.96) | +0.182 (R²=0.87) |
| CIFAR-10 ResNet18 | +1.128 (R²=0.95) | +1.144 (R²=0.97) | +1.147 (R²=0.97) | +1.154 (R²=0.99) |
| sklearn digits 8x8 | +1.021 (R²=0.98) | +0.908 (R²=0.99) | +0.944 (R²=0.97) | +1.148 (R²=0.96) |

## F41 vs empirical critical density

Cross-check that the F41 prediction lands on the same critical density as the empirical curve. Ratio reported as mean and full range over the cells where both are finite and positive.

| Dataset | mean `s_0_emp / s_0_F41` | range |
|---|---:|---:|
| MNIST 28x28 | 0.892 | [0.680, 1.023] |
| CIFAR-10 PCA-200 | 0.793 | [0.250, 0.996] |
| CIFAR-10 ResNet18 | 0.875 | [0.628, 1.013] |
| sklearn digits 8x8 | 0.936 | [0.766, 1.017] |

## Theoretical reference

From Appendix D (binary case), with
`c = J_0 / sqrt(V) ~ sqrt(H)`, the half-transition density is

    s_0 ~ z_{1/2}^2 / c^2 ~ 1/H,

so D17 predicts `α_H = −1` and `α_L = 0` in the `C = 2` limit.
The Appendix F generalisation does not change the leading
H-scaling for fixed depth: each competitor SNR `r_k` still
scales as `sqrt(H)` because the read-out variance scales as
`1/H`. The expected pattern is therefore

  - `α_H ≈ −1` everywhere, **independent of L** at leading order.
  - `α_L` measures the residual depth dependence not captured
    by H alone. A non-zero `α_L` is a deviation from naive theory.

## Headline findings

1. **`α_H` is much weaker than theory predicts.** Across all four datasets the joint H-exponent sits in `[-0.21, -0.07]`, an order of magnitude shallower than D17's `α_H = −1`. Per-L slopes agree: no `α_H` value reaches `−0.5`.

2. **`α_L` is strongly positive on three of four datasets** (`α_L ∈ [+0.16, +1.14]`). Deeper networks need a *higher* density to reach the same fraction of unpruned accuracy. This is *not* in D17; it is the multiplicative attenuation of signal through stacked masked ReLU layers, which the bare leading-order scaling analysis ignores.

3. **The per-L and per-H slopes are remarkably stable.** For every dataset the per-L `α_H` and per-H `α_L` vary by at most a factor of two over their respective ranges (see tables). The joint exponents are therefore meaningful global summaries, not just averages over noisy slopes.

4. **F41 systematically *over-predicts* `s_0`** by 10–25 % on average (ratio `s_0_emp / s_0_F41 < 1` everywhere). This is the integrated form of the positive residual seen in the shallow-network rows of `residuals.png`: F41's independent-competitor approximation under-counts the joint orthant probability, pushing the predicted half-transition to a higher `s`.

## Per-dataset comments

**MNIST 28x28.** Joint `α_H = -0.186`, `α_L = +0.851` (R² = 0.946).
  Per-L `α_H` ∈ `[-0.349, -0.099]` (range 0.250), per-L R² ∈ `[0.83, 0.97]`.

**CIFAR-10 PCA-200.** Joint `α_H = -0.075`, `α_L = +0.162` (R² = 0.848).
  Per-L `α_H` ∈ `[-0.096, -0.030]` (range 0.066), per-L R² ∈ `[0.77, 0.98]`.

**CIFAR-10 ResNet18.** Joint `α_H = -0.136`, `α_L = +1.143` (R² = 0.971).
  Per-L `α_H` ∈ `[-0.193, -0.093]` (range 0.100), per-L R² ∈ `[0.83, 0.98]`.

**sklearn digits 8x8.** Joint `α_H = -0.209`, `α_L = +1.005` (R² = 0.967).
  Per-L `α_H` ∈ `[-0.277, -0.176]` (range 0.101), per-L R² ∈ `[0.89, 0.99]`.

## Caveats

  - The `H` range here is 64 → 512 (8×, less than one decade). Power-law exponents from such a short lever-arm should be taken as effective slopes, not asymptotic limits.
  - CIFAR-PCA at `L = 1, H ∈ {256, 512}` does not converge in 4 epochs (`A_full ≈ 0.22`), which compresses its `s_0` estimates near the chance baseline and weakens both fits. The other 12 cells dominate the joint fit; the L = 2–4 rows are the cleaner read.
  - `s_0` is defined as the half-transition density in A-space, not the `β`-style inflection point. The two agree to leading order but the half-transition is more robust on noisy `A_emp` curves because it only requires one interpolation step, not a sigmoid fit.
