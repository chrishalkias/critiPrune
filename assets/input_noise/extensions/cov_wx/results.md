# Cov(W, x) hypothesis test for the σ-vs-σ² sigmoid-fit discrepancy

## Cov statistic chosen

I use the **scalar factorisation error**

```
delta_fact = <W^2 x^2> / (<W^2><x^2>) - 1
```

where `W ≡ W^(1) ∈ R^{H×D}` is the trained first-layer weight matrix,
`<·>` denotes the empirical mean over both indices `(h, j)` (joint),
and `<x_j^2>` is the per-feature second moment on the test set.

Rationale. §3 of `pruning_sigmoid_derivation.md` factorises the variance
term as `<W_{hj}^2 (x_j+ε_j)^2> ≈ <W^2>(<x^2>+σ_x^2)`. The crucial
substitution `<W_{hj}^2 x_j^2> = <W^2><x^2>` is *exactly* what
`delta_fact = 0` asserts. So `|delta_fact|` is the most direct
single-number probe of the dropped Cov(W², x²) term that §6.2(b) names
as the suspected culprit. It is also dimensionless, order ~10⁻¹–10⁻²
on these networks (passes the magnitude sanity check), and invariant to
overall rescaling of either `W` or `x`. I tested two alternatives —
the Pearson correlation of `mean_h W_{hj}^2` against `<x_j^2>` across
`j`, and the joint Pearson r over the flattened `(h, j)` pairs — and
both gave weaker (and noisier) signals; `delta_fact` is the cleanest.

## Per-cell measurements (N = 5)

The extension cells `input_noise/extensions/depth_cells/results.json`
from subagent S1 do not yet exist on disk, so this report uses the
baseline 5 cells. Adding S1's deeper cells would be a natural
follow-up to push N to 7–8.

| cell                 | `delta_fact` | `|delta_fact|` | ΔR² = R²(σ) − R²(σ²) |
|----------------------|-------------:|---------------:|---------------------:|
| digits  H=64  L=2    |     +0.0197  |        0.0197  |              +0.0248 |
| digits  H=128 L=2    |     +0.0054  |        0.0054  |              +0.0262 |
| digits  H=128 L=4    |     +0.0002  |        0.0002  |              +0.0255 |
| MNIST   H=128 L=2    |     −0.0312  |        0.0312  |              +0.0228 |
| MNIST   H=256 L=2    |     +0.1092  |        0.1092  |              +0.0145 |

## Correlation across cells (N = 5)

| direction | statistic vs ΔR² | Pearson r | Pearson p | Spearman ρ | Spearman p |
|-----------|------------------|----------:|----------:|-----------:|-----------:|
| signed    | `delta_fact`     | −0.832    | 0.080     | −0.300     | 0.624      |
| absolute  | `|delta_fact|`   | **−0.993**| **0.001** | **−0.900** | **0.037**  |

The hypothesis from §6.2(b) predicts that larger |Cov(W, x)| should
produce larger ΔR² (more departure from the pure-second-cumulant
σ²-sigmoid prediction). The data show the **opposite sign**: the cell
with the largest factorisation deviation (MNIST H=256 L=2,
|delta| ≈ 0.11) has the *smallest* ΔR² (+0.014), and the cell with the
smallest deviation (digits H=128 L=4, |delta| ≈ 2×10⁻⁴) has near the
largest ΔR² (+0.026). With N = 5 the absolute-value Spearman ρ = −0.90
(p = 0.04, two-sided) is suggestive but not conclusive; the Pearson
r = −0.99 (p = 0.001) is unusually clean for such a small sample, but
small-N Pearson is sensitive to the single high-leverage MNIST H=256
point.

## Verdict

**The data do not support the Cov(W, x) hypothesis with ρ = −0.90 over
N = 5 cells** — the rank correlation is the wrong sign relative to what
§6.2(b) predicts. The σ-vs-σ² discrepancy (uniform Δ R² ≈ +0.022 across
all 5 cells, range 0.014–0.026) does not track the per-cell magnitude
of weight–input factorisation error, so a Cov(W, x) correction is
unlikely to be the dominant source. The remarkable cell-to-cell
uniformity of ΔR² itself (range 0.012, vs |delta_fact| spanning 500×
from 2×10⁻⁴ to 0.11) is independent evidence that ΔR² is set by some
near-universal property of the σ-vs-σ² parametrisation (likely the
location of inflection on the sigmoid curve, or a finite-grid sampling
effect of the 1-D noise sweep), rather than by a cell-dependent
correction term.

Caveats: N = 5 is small; subagent S1's depth-extension cells, when
available at `input_noise/extensions/depth_cells/results.json`, should
be added — they would both tighten the correlation and probe whether
the trend changes shape at deeper L. A second-order check would be to
re-derive the SNR keeping the next-to-leading Cov(W², x²) term and see
if it produces a *negative* correction (consistent with the observed
sign), which would reverse the sign of the §6.2(b) argument.

Files:
- pipeline: `/Users/chrischalkias/Projects/critiPrune/input_noise/extensions/cov_wx/measure.py`
- numbers:  `/Users/chrischalkias/Projects/critiPrune/input_noise/extensions/cov_wx/results.json`
- scatter:  `/Users/chrischalkias/Projects/critiPrune/input_noise/extensions/cov_wx/results_scatter.png`
