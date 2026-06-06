# `pruning/` — Structured (path/neuron) pruning and scaling laws

This subpackage implements the **Feynman path-integral analogy** for fully-connected
ReLU networks: each forward pass is decomposed into the sum of contributions from
discrete computation paths through the network, and pruning corresponds to keeping
only the strongest `K` paths per token at each hidden layer. By sweeping `K` from
`1` to `H` (the hidden width), the test accuracy traces a sigmoidal "phase
transition" with a well-defined inflection point `K_0` whose dependence on the
architecture's width `H` and depth `L` reveals power-law scaling.

It is the **structured / neuron-level** counterpart to [`unstructured_pruning/`](../unstructured_pruning/),
which prunes individual weights instead of whole neuron activations.

---

## Scientific motivation

For an FC ReLU network with hidden width `H` and depth `L`, a single forward pass
on input `x ∈ R^I` can be written as a sum over paths:

```
logit_c(x) = Σ_{p}  W_out[c, h_L(p)] · ∏_{l=0}^{L-1} W[h_{l+1}(p), h_l(p)] · x[h_0(p)] · M(p)
```

where `M(p)` is the product of the ReLU masks along path `p`. We approximate
this exact sum by retaining only the top-`K` paths by absolute contribution at
each layer and ask: **how few paths suffice to recover the unpruned accuracy?**

The recovery curve `A(K)` is consistently sigmoidal:

```
A(K) = A_0 + (A_∞ - A_0) / (1 + exp(-β·(K - K_0)))
```

This package fits `(A_0, A_∞, K_0, β)` per architecture and discovers that
`K_0` follows a clean two-parameter power law:

```
K_0 = a · H^α · L^γ
```

with `R² > 0.9` typically. Examples observed:

| Dataset           | Method | Fit                                  |
|-------------------|--------|--------------------------------------|
| sklearn digits    | signal | `K_0 = 0.112 · H^0.588 · L^0.915`  (R²=0.95) |
| CIFAR-10 (ResNet18 features) | WANDA | `K_0 = 2.747 · H^0.782 · L^0.089` (R²=0.95) |

The companion quantity `g_eff = exp(-β)` plays the role of an *effective
coupling constant* and `β` the inverse-temperature of the phase transition.

---

## Module layout

| File | Purpose |
|------|---------|
| `pruning.py`         | Core engine: `FCNetwork`, the 5 pruning methods, batched path-tracing, sigmoid fitting, comparison plotting. Importable; also runs as a script on sklearn digits. |
| `mnist_scaling.py`   | `(H, L)` grid scan on **sklearn digits** (8×8, 64-dim) using the **signal** method. Fits `K_0`, `β`, `g_eff` as bivariate power laws. |
| `mnist28_scaling.py` | Same scan on **MNIST 28×28** (784-dim) with **WANDA** scoring. |
| `cifar_scaling.py`   | Same scan on **CIFAR-10** with frozen-ResNet18 features (512-dim) and **WANDA**. |
| `pythia_scaling.py`  | Plotting utilities for **Pythia-family LLMs** (160M → 12B). Sigmoid recovery curves of perplexity-recovery vs. MLP-neuron sparsity, fitted across model scale. |
| `test.py`            | `pytest` suite covering `FCNetwork` shapes, forward consistency (torch ↔ numpy), training, ReLU masks, all 5 score functions, the sparsify utilities, sigmoid/exponential fitting, scaling-law fitting, and plot generation. |
| `__init__.py`        | Public API re-exports. |

---

## Core engine: `pruning.py`

### `FCNetwork`

A PyTorch `nn.Module` of shape `[I] → [H]·L → [C]` with He init and Adam optimiser.

```python
FCNetwork(input_size=64, hidden_size=64, num_hidden_layers=3,
          num_classes=10, seed=42)
```

Provides:
- `forward(x)` — standard.
- `forward_with_masks(x)` — also returns the per-layer ReLU masks.
- `forward_cache(x)` — also returns pre-/post-activations for manual backprop.
- `train_model(X_tr, y_tr, X_val, y_val, epochs, bs, lr, verbose)` — Adam + cross-entropy.
- `W`, `b` properties — weights as numpy arrays for the path-tracing engine.
- `numpy_forward(X)` / `numpy_forward_with_masks(X)` — float64 numpy parity.

### Pruning methods (`PRUNING_METHODS`)

| key       | description                                              |
|-----------|----------------------------------------------------------|
| `signal`  | Dynamic per-sample, per-pixel `\|W·x\|` magnitude.       |
| `weight`  | Static row-norm `\|\|W_i\|\|_2` (data-free).             |
| `wanda`   | `\|W_ij\| · \|\|x_j\|\|_2` (Sun et al., 2023).           |
| `taylor`  | First-order sensitivity `\|∂L/∂W ⊙ W\|`.                 |
| `random`  | Uniform baseline.                                         |

Static methods produce per-layer column masks once via
`_precompute_column_masks`; `signal` is recomputed per sample/per layer
inside `_sparsify_dynamic`.

### Path-tracing engine

`evaluate_path_accuracy(model, X_test, y_test, k_values, layer_scores, method_name)`
returns `(accs, normal_acc)` where `accs[K]` is the test accuracy when only
the top-`K` paths per pixel are kept at every hidden layer.

Key design:
- Tensor shape `[B, I, H]` traces the contribution of every input dimension
  along every neuron, batched over `B` samples chosen by
  `_estimate_batch_size(I, H, max_mem_mb)` to fit a memory budget.
- ReLU masks are computed once on the full forward pass and reused across
  all `K`.
- A K=H sanity check captures the bias offset so that K<H accuracies
  remain comparable to the unpruned reference within `1e-6`.

### Sigmoid fit

```python
sigmoid_fn(K, A_inf, A_0, K_0, beta)        # the curve
fit_sigmoid(k_values, accuracies, normal_acc)  # → (popt, perr, r2)
```

Bounds: `A_inf ∈ [0,1]`, `A_0 ∈ [-0.05,1]`, `K_0 ∈ [0, 2·max_K]`, `β ∈ [1e-4, 20]`.

### Comparison figure

`make_comparison_plot(...)` renders all 5 methods on one grid and a bar
chart of fit parameters per method. Standalone script writes
`assets/legacy/mnist/pruning_comparison.png`.

---

## Scaling experiments

All three scaling scripts share the same workflow:

```
for H in H_VALUES:
  for L in L_VALUES:
    train FC[I→H·L→10] → val_acc
    compute pruning scores                       (signal | wanda)
    sweep K, evaluate accuracy via path-tracing
    fit sigmoid → (A_∞, A_0, K_0, β, R²)
fit  K_0 = a · H^α · L^γ   (and same for β, g_eff)
emit JSON results, JSON laws, two PNG figures
```

| Script | Dataset | Input dim | Method | Grid `H × L` |
|--------|---------|-----------|--------|--------------|
| `mnist_scaling.py`   | sklearn digits | 64  | signal | `[8,16,24,32,48,56,64,96] × [1..10]` |
| `mnist28_scaling.py` | MNIST 28×28    | 784 | wanda  | `[64,128,256,512] × [2,3,5,7,10]`     |
| `cifar_scaling.py`   | CIFAR-10 + ResNet18 | 512 | wanda | `[64,128,256,512] × [2,3,5,7,10]` |

CIFAR features are extracted once via a frozen ImageNet-pretrained ResNet18
and cached to `$FEATURE_CACHE_DIR` (default `/tmp/cifar_features`).

### Outputs (written to `assets/legacy/<dataset>/` or `assets/legacy/<dataset>/`)

| File | Content |
|------|---------|
| `*_scaling_results.json` | One row per `(H, L)`: `val_acc`, `n_params`, `accs[K]`, sigmoid params + R², per-param errors. |
| `*_scaling_laws.json`    | Fitted exponents `a, α, γ, R²` for `K_0`, `β`, `g_eff`. |
| `scaling_curves.png`     | Top: per-`L` panels of `A(K)` curves with sigmoid fits; bottom: `K_0` vs. `H` lines per `L` with overlaid power-law fit. |
| `k0_scaling.png`         | Left: same `K_0` vs. `H`; right: `K_0` heatmap over the `(H, L)` grid. |
| `checkpoints.pt`         | (sklearn) trained `state_dict` per cell, for replay. |

---

## Pythia (LLM) extension

`pythia_scaling.py` exposes two plotting helpers used by external evaluation
scripts (which run perplexity recovery on Pythia-160M through Pythia-12B with
WANDA pruning of MLP intermediate neurons):

- `make_sigmoid_curves_plot(results, output_dir)` — recovery curve overlay
  across the 7-model family with sigmoid fits.
- `make_k0_scaling_plot(results, scaling, output_dir)` — `K_0_abs` vs. `d_ff`
  with power-law fit `K_0 = a · d_ff^α · L^γ`, plus a `K_0(%)` heatmap over
  `(d_ff, L)`. The y-axis here is *MLP neurons kept* and the recovery is
  `log PPL_base / log PPL_sparse` rather than classification accuracy.

The driver notebook is `Pythia_test.ipynb`; results land in `assets/legacy/pythia/`.

---

## How to run

From the repository root:

```bash
# Quick smoke test (sklearn digits, all 5 methods, ~1 minute)
python -m pruning.pruning

# Full scaling scans
python -m pruning.mnist_scaling      # sklearn 8x8, signal,  ~10 min CPU
python -m pruning.mnist28_scaling    # MNIST 28x28, WANDA, GPU recommended
python -m pruning.cifar_scaling      # CIFAR-10 + ResNet18, GPU recommended

# Tests
pytest pruning/test.py -v
```

Cluster jobs are submitted via `u_scripts/` (see `submit.sh` and the
`*.sbatch` files at the repo root).

### Dependencies

`numpy`, `torch`, `scipy`, `matplotlib`, `scikit-learn`, `torchvision`
(for MNIST / CIFAR raw data). LaTeX rendering is disabled by default for
portability.

---

## Reading the figures

In a recovery curve:
- **`A_∞`** ≈ unpruned validation accuracy — the high plateau.
- **`A_0`** ≈ random-guess accuracy — the low plateau.
- **`K_0`** marks the crossover point (half-recovered).
- **`β`** is the steepness of the transition; `g_eff = exp(-β)` ∈ (0,1] is
  reported as the "effective coupling".

In `k0_scaling.png`:
- Solid lines connect measured `K_0` per `L` slice.
- Dashed lines are the fitted power law `a · H^α · L^γ`.
- The right-hand heatmap reports `K_0` directly over the architecture grid;
  white text marks values above the grid mean × 1.3.

---

## Relationship to the rest of the repo

- The `pruning.py` engine and `FCNetwork` class are also re-used by the
  unstructured pipeline ([`unstructured_pruning/`](../unstructured_pruning/));
  only the masking step differs.
- Plot regeneration (without re-running the experiments) lives in
  `replot.py` at the repo root.
- Aggregated assets used in the top-level README are mirrored under
  `assets/legacy/{mnist,mnist28,cifar,pythia}/`.
