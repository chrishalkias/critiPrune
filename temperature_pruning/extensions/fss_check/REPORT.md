# FSS check: does the sigmoid slope beta grow with system size H?

## Data aggregated
- Long-format rows after filter (R2 >= 0.8, beta in (0.1, 10000.0)): **5180** total, **5117** in density-space.
- Per dataset: {'sklearn_digits_K': 63, 'sklearn_digits': 2070, 'mnist28': 1053, 'cifar_pca': 1053, 'cifar_resnet': 941}
- Per method: {'signal_K': 63, 'random': 1631, 'magnitude': 1743, 'wanda': 1743}  (3 mask seeds per (H, L) cell in unstructured sweeps)
- Triples (dataset, method, fixed-L) with >=3 H values: **111** density-space, 119 total.
- Triples (dataset, method, fixed-H) with >=3 L values: **186** density-space, 194 total.

## Missing inputs (reported, not regenerated)
- `mnist_figures/scaling_results.json` (not on disk)
- `cifar_figures/scaling_results.json` (not on disk)

## Power-law fit results (density-space, headline)
- beta(H) ~ H^p at fixed L: median p = **+0.164**, IQR [-0.022, +0.357]
  - 46% of triples have p >= +0.20 (sharpening with size)
  - 47% have |p| < 0.20 (no scaling)
  - 7% have p <= -0.20 (softening)
- beta(L) ~ L^q at fixed H: median q = **+0.231**, IQR [-0.236, +0.505]

### p by method (median over (dataset, L))
- magnitude   +0.101
- random      -0.012
- wanda       +0.339

### p by dataset (median over (method, L))
- cifar_pca           +0.023
- cifar_resnet        +0.097
- mnist28             +0.326
- sklearn_digits      +0.235

### Caveat: K-space (path-tracing) sweep on sklearn digits
The legacy K-space fits in `assets/mnist_figures/scaling_results.json` show p = -0.69, which
looks contrary but is a unit artifact: beta_K = beta_density / H, so p_K = p_density - 1. In
density units this becomes p_density ~ +0.31, consistent with the density-space methods above.
The K-space series is excluded from the headline statistics for this reason and shown only as
a labeled outlier in `p_exponent_summary.png` if reinstated.

## Verdict
**FSS-neutral**  (rule: |median p| < 0.20 -> neutral; >= 0.20 -> supportive; <= -0.20 -> contrary)

## Recommendation
Beta neither grows nor declines consistently with H (median p=+0.16, within the neutral band). However the picture is method-dependent: WANDA shows clear sharpening (+0.34), magnitude weak sharpening (+0.10), random none (-0.01). 46% of triples are FSS-supportive vs only 7% contrary, so the data lean weakly toward sharpening but do not justify unqualified 'phase transition' claims. The user's default of operational softening is the right call: keep 'critical density' for K_0 / s_0, explicitly disclaim singular-limit behavior, and cite the measured per-method p distribution as the basis. Optionally mention WANDA's positive p as a tantalizing data-aware-pruning effect worth following up.