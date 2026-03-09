# Neural Network Pruning as a Phase Transition

> *Discovering that structured pruning of neural networks, from tiny FC networks to multi-million-parameter LLMs, obeys a universal sigmoid law with a sharp critical threshold, analogous to a second-order phase transition in statistical mechanics.*

---

## Overview

This repository investigates the hypothesis that **neural network pruning is a phase transition**, drawing on quantum field theory and statistical physics.

When you progressively restore active neurons (paths) to a pruned network, performance does not recover gradually, it undergoes a sharp, sigmoidal transition at a critical threshold K₀. This behaviour mirrors the order-parameter jump at a second-order phase transition, and the parameters of the sigmoid curve (`K₀`, `β`, `g_eff`) obey precise **scaling laws** across network architecture.

The framework is validated on:
- Fully-connected ReLU networks trained on **MNIST / sklearn Digits** and **CIFAR-10**
- The **Pythia** transformer family (14M → 6.9B parameters, EleutherAI)
- Mixed open-source LLMs: **TinyLlama**, **Qwen**, **SmolLM**

---

## Core Idea

For a given network, define K as the sparsity ratio of the prunned network. The performance recovery curve is fit to a **logistic sigmoid**:

$$A(K) = A_0 + \frac{A_\infty - A_0}{1 + e^{-\beta(K - K_0)}}$$

| Parameter | Meaning |
|---|---|
| **K₀** | Critical sparsity threshold — the "phase transition point" |
| **β** | Inverse correlation length / steepness of the transition |
| **A₀** | Accuracy at full pruning (chance level) |
| **A∞** | Accuracy at full network (unpruned) |

If the pruning-recovery curve is a true phase transition, the **susceptibility** (variance of the loss across diverse prompts) should peak sharply at K₀ — analogous to diverging susceptibility at a critical point in a magnetic system. This prediction is tested experimentally in `LLM_pruning_test.ipynb`.

**Relevant literature referenced in the notebooks:**
- [Information Flow Through Neural Networks (2017)](https://arxiv.org/pdf/1712.00003)
- [The Lottery Ticket Hypothesis (2018)](https://arxiv.org/abs/1803.03635)
- [Phase Transitions in LLMs and O(N) (2025)](https://arxiv.org/pdf/2501.16241)
- [Phase Transitions in Neural Network Pruning (2026)](https://arxiv.org/pdf/2602.15224)
- [Phase Transitions in LLMs (2026, Nature)](https://www.nature.com/articles/s44387-026-00072-8)

---

## Repository Structure

```
.
├── pruning.py              # Core library: FCNetwork, pruning methods, sigmoid fitting
├── mnist_scaling.py        # Scaling law experiments on MNIST (H × L architecture grid)
├── cifar_scaling.py        # Same experiments extended to CIFAR-10
├── Pythia_test.ipynb       # Pythia LLM family (14M–6.9B), WANDA pruning, scaling laws
└── LLM_pruning_test.ipynb  # Mixed LLMs + susceptibility / phase transition analysis
```

### `pruning.py` — Core Library

The heart of the repository. Implements:

- **`FCNetwork`** — pure NumPy fully-connected ReLU network with Adam optimiser and He initialisation. Safe for parallel multi-config use (seed passed per instance).
- **Five pruning methods**, precomputed as column masks for efficiency:
  - `signal` — Activates paths by signal magnitude `|W·x|`
  - `weight` — Weight magnitude pruning
  - `wanda` — WANDA (Sun et al., 2023): `|W| × ‖x‖`, recommended for LLM-scale comparisons
  - `taylor` — Taylor sensitivity (gradient × activation)
  - `random` — Random baseline
- **`precompute_pruning_scores`** — Vectorised batch computation of all scores
- **`evaluate_path_accuracy`** — Batched K-sweep with memory budget control
- **`sigmoid_fn` / `fit_sigmoid`** — Sigmoid model and scipy curve-fit wrapper

Can be used as a **standalone script** or **imported as a module**:
```python
from pruning import (
    FCNetwork, relu, softmax, accuracy,
    precompute_pruning_scores, evaluate_path_accuracy,
    sigmoid_fn, fit_sigmoid,
    PRUNING_METHODS, METHOD_STYLE,
)
```

### `mnist_scaling.py` — MNIST Scaling Laws

Trains FC networks across an architecture grid `H ∈ {32, 64, 128, 256}` × `L ∈ {2, 3, 5, 7, 10}` on the sklearn Digits (8×8 MNIST-like) dataset and compares all five pruning methods. Outputs:

```
pruning_coupling/mnist_figures/
  ├── mnist_scaling_curves.png     — sigmoid fits per architecture
  ├── mnist_scaling_laws.png       — K₀/H, g_eff vs L, compressibility
  ├── mnist_parameter_heatmaps.png — K₀, β, g_eff heatmaps over (H, L)
  ├── mnist_scaling_results.json
  └── mnist_scaling_laws.json
```

### `cifar_scaling.py` — CIFAR-10 Scaling Laws

Extends the MNIST analysis to CIFAR-10 using **WANDA pruning only** (recommended for LLM comparisons). CIFAR's 3072-dim inputs are first reduced to 200 dimensions via PCA (retaining ~X% variance). Tests whether the scaling laws discovered on MNIST transfer to a harder dataset. Outputs:

```
pruning_coupling/cifar_figures/
  ├── cifar_scaling_curves.png
  ├── cifar_scaling_laws.png
  ├── cifar_parameter_heatmaps.png
  ├── cifar_scaling_results.json
  └── cifar_scaling_laws.json
```

### `Pythia_test.ipynb` — LLM Scaling Laws (Pythia Family)

Applies the same framework to **transformer MLP layers** across the full Pythia model family. For each model the pipeline is:

1. Collect MLP activation statistics over 128 C4 calibration samples
2. Compute per-neuron WANDA scores (combined up-projection and down-projection signals)
3. Sweep K = fraction of `d_ff` neurons kept (1% → 100%), measuring **perplexity recovery** on WikiText-2
4. Fit the sigmoid and extract `(K₀, β)`
5. Fit joint power-law scaling across the model family: `K₀ ~ d_ff^α × L^γ`

Designed to run on **Google Colab T4** for models up to 2.8B; an A100 is needed for 6.9B.

### `LLM_pruning_test.ipynb` — Phase Transition Analysis

Exploratory notebook testing the phase transition hypothesis on mixed open-source LLMs (TinyLlama-1.1B, Qwen-0.5B, SmolLM2-1.7B). Contains two main experiments:

**Experiment 1 — Basic sigmoid recovery**: Applies top-K activation sparsity hooks to MLP down-projection layers and fits the sigmoid curve.

**Experiment 2 — Susceptibility divergence**: Uses 10 diverse prompts to measure the *variance* of the loss (susceptibility χ) across the K-sweep. A sharp peak in χ at K₀ is the signature of a genuine second-order phase transition, distinguishing the result from a mere empirical S-curve.

Also includes a discussion of **finite-size scaling / data collapse**: if LLM pruning is a true phase transition, recovery curves for models of different sizes should collapse onto a universal curve when plotted against the rescaled variable `(K − K₀) × H^{1/ν}`.

---

## Scaling Laws

The main empirical finding is that sigmoid parameters scale as **power laws** in architecture:

$$K_0 \approx a \cdot H^\alpha \cdot L^\gamma$$
$$\beta \approx a \cdot H^\alpha \cdot L^\gamma$$

where H is hidden width and L is depth.

---

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib tensorflow
# For LLM experiments:
pip install torch transformers datasets accelerate
```

For `Pythia_test.ipynb` and `LLM_pruning_test.ipynb`, a **GPU runtime** (Google Colab T4 or better) is strongly recommended.

---

## Usage

**Run MNIST scaling laws:**
```bash
python mnist_scaling.py
```

**Run CIFAR-10 scaling laws:**
```bash
python cifar_scaling.py
```

**Use `pruning.py` as a library:**
```python
from pruning import FCNetwork, precompute_pruning_scores, evaluate_path_accuracy, fit_sigmoid

model = FCNetwork(input_size=64, hidden_size=128, num_hidden_layers=3, num_classes=10)
model.train(X_tr, y_tr, X_val, y_val, epochs=300)

scores = precompute_pruning_scores(model, X_tr, y_tr, methods=['wanda'])
k_values = list(range(1, 129))
accs, normal_acc = evaluate_path_accuracy(model, X_te, y_te, k_values, scores['wanda'])

popt, perr, r2 = fit_sigmoid(k_values, accs, normal_acc)
K_0, beta = popt[2], popt[3]
g_eff = np.exp(-beta)
```

**Pythia experiments** — open `Pythia_test.ipynb` in Colab with a GPU runtime. Select models via the `--models` argument or `DEFAULT_MODELS` list. A HuggingFace token is required for model downloads.

---

## Output Figures

Each script/notebook produces three canonical figures:

| Figure | Contents |
|---|---|
| `*_scaling_curves.png` | Scatter + sigmoid fit for each (H, L) configuration |
| `*_scaling_laws.png` | K₀/H ratio, g_eff vs depth, compressibility scatter |
| `*_parameter_heatmaps.png` | K₀, β, g_eff as heatmaps over the architecture grid |

---

## Notes & Caveats

- All FC network experiments use pure **NumPy** — no deep learning framework required for `pruning.py` / `mnist_scaling.py` / `cifar_scaling.py`.
- CIFAR-10 loading supports three backends in order: `tensorflow.keras`, `torchvision`, raw download from the Toronto URL.
- The WANDA method is preferred for cross-scale comparisons as it accounts for both weight magnitude and activation statistics.
- LLM experiments expose a HuggingFace token in `Pythia_test.ipynb` — **replace or rotate this token before sharing the notebook publicly**.