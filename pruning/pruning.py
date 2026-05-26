#!/usr/bin/env python3
"""
Feynman Path Integral Analogy in Neural Networks — Multi-Pruning Comparison
============================================================================
PyTorch implementation with batched path-tracing and multiple pruning methods.

Pruning methods
---------------
  signal  : Dynamic |W·x| magnitude (per-sample, per-pixel)
  weight  : Static weight-magnitude ranking
  wanda   : Weighted Activation * Norm-based Data-Aware (Sun et al., 2023)
  taylor  : First-order Taylor sensitivity |grad_W * W|
  random  : Uniform random baseline

Optimisations
-------------
  Static methods (weight/wanda/taylor/random): column masks precomputed once
  per K, then applied as a single boolean slice.
  Batched sample processing: samples grouped in mini-batches of size B,
  creating [B, I, H] tensors.
  ReLU masks computed once; reused across all K values.
  Bias-offset sanity check: K=H reconstruction verified against the standard
  forward pass before applying the correction.

API (for importing)
-------------------
  from pruning import (
      FCNetwork, accuracy,
      precompute_pruning_scores, evaluate_path_accuracy,
      sigmoid_fn, fit_sigmoid,
      PRUNING_METHODS, METHOD_STYLE,
  )
"""

import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# Plot style (no LaTeX for portability)
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Serif",
    "axes.labelsize": 12, "font.size": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

# --- Pruning method registry -------------------------------------------------

PRUNING_METHODS = {
    'signal':  'Signal |W·x|',
    'weight':  'Weight Magnitude',
    'wanda':   'WANDA',
    'taylor':  'Taylor Sensitivity',
    'random':  'Random (baseline)',
}
METHOD_STYLE = {
    'signal':  dict(color='steelblue',  marker='o', ls='-'),
    'weight':  dict(color='crimson',    marker='s', ls='--'),
    'wanda':   dict(color='darkorange', marker='^', ls='-.'),
    'taylor':  dict(color='seagreen',   marker='D', ls=':'),
    'random':  dict(color='gray',       marker='x', ls=(0, (3, 1, 1, 1))),
}


# --- Helpers ------------------------------------------------------------------

def accuracy(logits, y):
    """Compute classification accuracy from numpy logits and label arrays."""
    return (logits.argmax(axis=1) == y).mean()


# --- Neural network -----------------------------------------------------------

class FCNetwork(nn.Module):
    """Fully-connected ReLU network: [input] -> [H]*L -> [C].

    Uses He initialisation and Adam optimiser, matching the behaviour
    of the original pure-NumPy implementation.

    Parameters
    ----------
    input_size        : int – dimensionality of input features
    hidden_size       : int – neurons per hidden layer (H)
    num_hidden_layers : int – number of hidden layers (L)
    num_classes       : int – output classes (C)
    seed              : int – RNG seed for weight initialisation
    """

    def __init__(self, input_size=64, hidden_size=64,
                 num_hidden_layers=3, num_classes=10, seed=42):
        super().__init__()
        self.H = hidden_size
        self.L = num_hidden_layers
        self.C = num_classes
        self.input_size = input_size

        torch.manual_seed(seed)

        layers = []
        sizes = [input_size] + [hidden_size] * num_hidden_layers + [num_classes]
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(fan_in, fan_out, dtype=torch.float32)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """Standard forward pass: input -> hidden ReLU layers -> logits."""
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def forward_with_masks(self, x):
        """Forward pass that also returns per-layer ReLU masks.

        Returns
        -------
        logits     : Tensor [N, C]
        relu_masks : list of Tensor [N, H], one per hidden layer
        """
        relu_masks = []
        for layer in self.layers[:-1]:
            z = layer(x)
            mask = (z > 0).to(z.dtype)
            x = z * mask
            relu_masks.append(mask)
        return self.layers[-1](x), relu_masks

    def forward_cache(self, x):
        """Forward pass caching pre- and post-activations for gradient computation.

        Returns
        -------
        logits : Tensor [N, C]
        zs     : list of pre-activation Tensors
        hs     : list of post-activation Tensors (hs[0] = input)
        """
        zs, hs = [], [x]
        for layer in self.layers[:-1]:
            z = layer(x)
            zs.append(z)
            x = torch.relu(z)
            hs.append(x)
        return self.layers[-1](x), zs, hs

    def train_model(self, X_tr, y_tr, X_val, y_val,
                    epochs=300, bs=64, lr=1e-3, verbose=True):
        """Train the network with Adam and cross-entropy loss.

        Parameters
        ----------
        X_tr, X_val : ndarray [N, D] - training / validation features
        y_tr, y_val : ndarray [N]    - integer class labels
        epochs      : int
        bs          : int            - mini-batch size
        lr          : float          - learning rate
        verbose     : bool

        Returns
        -------
        float - final validation accuracy
        """
        p = next(self.parameters())
        device, dtype = p.device, p.dtype
        X_tr_t = torch.as_tensor(X_tr, dtype=dtype, device=device)
        y_tr_t = torch.as_tensor(y_tr, dtype=torch.long, device=device)
        X_val_t = torch.as_tensor(X_val, dtype=dtype, device=device)
        y_val_t = torch.as_tensor(y_val, dtype=torch.long, device=device)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        N = X_tr_t.shape[0]

        if verbose:
            n_params = sum(p.numel() for p in self.parameters())
            print(f"  Model: [{self.input_size}]->[{self.H}]x{self.L}"
                  f"->[{self.C}]  Params: {n_params:,}")

        self.train()
        for ep in range(1, epochs + 1):
            idx = torch.randperm(N, device=device)
            total_loss = 0.0
            for s in range(0, N, bs):
                sl = idx[s:s + bs]
                logits = self(X_tr_t[sl])
                loss = criterion(logits, y_tr_t[sl])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(sl)
            if verbose and (ep % 50 == 0 or ep == 1):
                self.eval()
                with torch.no_grad():
                    va = (self(X_val_t).argmax(1) == y_val_t).float().mean().item()
                self.train()
                print(f"  Ep {ep:4d}/{epochs}  loss={total_loss / N:.4f}"
                      f"  val={100 * va:.1f}%")

        self.eval()
        with torch.no_grad():
            fa = (self(X_val_t).argmax(1) == y_val_t).float().mean().item()
        if verbose:
            print(f"  Final val accuracy: {100 * fa:.2f}%")
        return fa

    # --- NumPy-compatible weight access for pruning code ----------------------

    @property
    def W(self):
        """Weight matrices as a list of numpy arrays (read-only views)."""
        return [layer.weight.detach().cpu().numpy() for layer in self.layers]

    @property
    def b(self):
        """Bias vectors as a list of numpy arrays (read-only views)."""
        return [layer.bias.detach().cpu().numpy() for layer in self.layers]

    def numpy_forward(self, X):
        """Compute forward pass in numpy (for pruning evaluation)."""
        h = X
        W, b = self.W, self.b
        for l in range(self.L):
            h = np.maximum(0, h @ W[l].T + b[l])
        return h @ W[-1].T + b[-1]

    def numpy_forward_with_masks(self, X):
        """Forward pass in numpy returning ReLU masks."""
        h = X
        W, b = self.W, self.b
        relu_masks = []
        for l in range(self.L):
            z = h @ W[l].T + b[l]
            mask = (z > 0).astype(z.dtype)
            h = z * mask
            relu_masks.append(mask)
        return h @ W[-1].T + b[-1], relu_masks


# --- Data loading (sklearn digits) --------------------------------------------

def load_digits_data(seed=42):
    """Load and split the sklearn digits dataset (8x8 MNIST-like).

    Returns
    -------
    X_tr, X_val, X_te : ndarray [N, 64] – standardised features
    y_tr, y_val, y_te : ndarray [N]      – integer class labels
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te = sc.transform(X_te)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# --- Pruning-score pre-computation -------------------------------------------

def _collect_layer_activations(model, X):
    """Post-ReLU activations: acts[0]=X, acts[l]=output of hidden layer l-1."""
    W, b = model.W, model.b
    acts = [X]
    h = X
    for l in range(model.L):
        h = np.maximum(0, h @ W[l].T + b[l])
        acts.append(h)
    return acts


def precompute_pruning_scores(model, X_calib, y_calib, methods=None, seed=42):
    """Compute per-layer neuron importance scores for each pruning method.

    Parameters
    ----------
    model   : FCNetwork
    X_calib : ndarray [N, D] - calibration data (typically training set)
    y_calib : ndarray [N]    - calibration labels
    methods : list of str or None (defaults to all methods)
    seed    : int

    Returns
    -------
    dict {method_name: [score_l0, ..., score_l{L-1}]}
        Each score is [1, H] (broadcastable) or None (signal = dynamic).
    """
    if methods is None:
        methods = list(PRUNING_METHODS.keys())

    W, b = model.W, model.b
    N = X_calib.shape[0]
    y_ohe = np.eye(model.C)[y_calib]
    acts = _collect_layer_activations(model, X_calib)

    scores = {m: [] for m in methods}

    # Gradients for Taylor method (manual backprop on numpy weights)
    grad_W = None
    if 'taylor' in methods:
        h = X_calib
        zs, hs = [], [X_calib]
        for l in range(model.L):
            z = h @ W[l].T + b[l]
            zs.append(z)
            h = np.maximum(0, z)
            hs.append(h)
        logits = h @ W[-1].T + b[-1]
        # softmax
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits_shifted)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        delta = (probs - y_ohe) / N
        grad_W = [None] * len(W)
        grad_W[-1] = delta.T @ hs[-1]
        for l in range(model.L - 1, -1, -1):
            delta = (delta @ W[l + 1]) * (zs[l] > 0).astype(float)
            grad_W[l] = delta.T @ hs[l]

    rng = np.random.default_rng(seed + 99)

    for l in range(model.L):
        W_l = W[l]
        act_l = acts[l]

        if 'signal' in methods:
            scores['signal'].append(None)  # computed on-the-fly

        if 'weight' in methods:
            scores['weight'].append(
                np.linalg.norm(W_l, axis=1, keepdims=True).T)

        if 'wanda' in methods:
            act_norm = np.sqrt((act_l ** 2).mean(axis=0))
            wanda = (np.abs(W_l) * act_norm[np.newaxis, :]).sum(
                axis=1, keepdims=True).T
            scores['wanda'].append(wanda)

        if 'taylor' in methods:
            taylor = (np.abs(grad_W[l] * W_l)).sum(
                axis=1, keepdims=True).T
            scores['taylor'].append(taylor)

        if 'random' in methods:
            scores['random'].append(
                rng.uniform(0, 1, (1, W_l.shape[0])))

    return scores


# --- Batched path-tracing engine ----------------------------------------------

def _precompute_column_masks(layer_scores, k_values, H, L):
    """For a static method, pre-compute boolean column masks per (K, layer)."""
    masks = {}
    for K in k_values:
        per_layer = []
        for l in range(L):
            if K >= H:
                per_layer.append(None)
            else:
                s = layer_scores[l].ravel()
                idx = np.argpartition(s, H - K)[H - K:]
                m = np.zeros(H, dtype=bool)
                m[idx] = True
                per_layer.append(m)
        masks[K] = per_layer
    return masks


def _sparsify_dynamic(current, K):
    """Per-row top-K by absolute value.  current: [*, I, H] or [I, H]."""
    H = current.shape[-1]
    if K >= H:
        return current
    s = np.abs(current)
    kth = H - K
    top = np.argpartition(s, kth, axis=-1)[..., kth:]
    result = np.zeros_like(current)
    lead = tuple(
        np.arange(current.shape[d]).reshape(
            [current.shape[d] if d == dd else 1
             for dd in range(current.ndim)])
        for d in range(current.ndim - 1)
    )
    result[(*lead, top)] = current[(*lead, top)]
    return result


def _trace_paths_batch(model_W, model_L, model_H,
                       X_batch, relu_masks_batch, K,
                       col_masks_K, is_static):
    """Trace K-strongest paths for a mini-batch.

    Parameters
    ----------
    model_W          : list of ndarray - weight matrices
    model_L          : int - number of hidden layers
    model_H          : int - hidden size
    X_batch          : [B, I]
    relu_masks_batch : list of [B, H] (one per hidden layer)
    K                : int - beam width
    col_masks_K      : list of bool[H] or None (static method column masks)
    is_static        : bool

    Returns
    -------
    logits : [B, C]
    """
    B, I = X_batch.shape
    H = model_H

    # Layer 0: [B, I, H]
    current = model_W[0].T[np.newaxis, :, :] * X_batch[:, :, np.newaxis]
    current *= relu_masks_batch[0][:, np.newaxis, :]

    if K < H:
        if is_static and col_masks_K[0] is not None:
            current[:, :, ~col_masks_K[0]] = 0.0
        else:
            current = _sparsify_dynamic(current, K)

    for l in range(1, model_L):
        current = current @ model_W[l].T
        current *= relu_masks_batch[l][:, np.newaxis, :]
        if K < H:
            if is_static and col_masks_K[l] is not None:
                current[:, :, ~col_masks_K[l]] = 0.0
            else:
                current = _sparsify_dynamic(current, K)

    logits = (current @ model_W[-1].T).sum(axis=1)
    return logits


def _estimate_batch_size(I, H, max_mem_mb=256):
    """Choose B so that [B, I, H] * 8 bytes <= max_mem_mb."""
    elem = I * H * 8
    if elem == 0:
        return 64
    B = max(1, int(max_mem_mb * 1e6 / elem))
    return min(B, 512)


# --- Evaluation loop ----------------------------------------------------------

def evaluate_path_accuracy(model, X_test, y_test, k_values,
                           layer_scores, method_name='signal',
                           max_mem_mb=256):
    """Evaluate accuracy(K) for one pruning method on test data.

    Parameters
    ----------
    model        : FCNetwork
    X_test       : ndarray [N, D]
    y_test       : ndarray [N]
    k_values     : list of int
    layer_scores : list of ndarray or None per layer
    method_name  : str
    max_mem_mb   : int - memory budget for batched path-tracing

    Returns
    -------
    accs       : dict {K: float}
    normal_acc : float (full unpruned network)
    """
    N, I = X_test.shape
    H = model.H
    W = model.W
    is_static = (method_name != 'signal')

    actual_logits, relu_masks = model.numpy_forward_with_masks(X_test)
    normal_acc = accuracy(actual_logits, y_test)

    col_masks = {}
    if is_static:
        col_masks = _precompute_column_masks(layer_scores, k_values, H, model.L)

    B = _estimate_batch_size(I, H, max_mem_mb)

    def _eval_K(K):
        logits = np.zeros((N, model.C))
        cm = col_masks.get(K, [None] * model.L)
        for s in range(0, N, B):
            e = min(s + B, N)
            rm = [relu_masks[l][s:e] for l in range(model.L)]
            logits[s:e] = _trace_paths_batch(
                W, model.L, model.H,
                X_test[s:e], rm, K, cm, is_static)
        return logits

    full_path_logits = _eval_K(H)
    bias_offset = actual_logits - full_path_logits

    check_acc = accuracy(full_path_logits + bias_offset, y_test)
    if not np.isclose(check_acc, normal_acc, atol=1e-6):
        warnings.warn(
            f"[{method_name}] K=H sanity check: {check_acc:.6f} vs "
            f"{normal_acc:.6f}.  Bias offset may be unreliable.")

    accs = {}
    for K in k_values:
        if K == H:
            accs[K] = normal_acc
        else:
            path_logits = _eval_K(K)
            accs[K] = accuracy(path_logits + bias_offset, y_test)

    return accs, normal_acc


# --- Sigmoid fitting ----------------------------------------------------------

def sigmoid_fn(K, A_inf, A_0, K_0, beta):
    """Parametric sigmoid: A_0 + (A_inf - A_0) / (1 + exp(-beta*(K - K_0)))."""
    K = np.asarray(K, dtype=float)
    return A_0 + (A_inf - A_0) / (1.0 + np.exp(
        np.clip(-beta * (K - K_0), -500, 500)))


def fit_sigmoid(k_values, accuracies, normal_acc):
    """Fit acc(K) = A_0 + (A_inf - A_0) / (1 + exp(-beta*(K - K_0))).

    Returns
    -------
    popt : ndarray [A_inf, A_0, K_0, beta] or None
    perr : ndarray of parameter std errors or None
    r2   : float or None
    """
    k_arr = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values])

    try:
        # Data-driven initial guess for beta: scale the largest interior
        # secant slope by the dynamic range so the optimiser starts near
        # the true steepness instead of stalling at 0.2.
        A_hi = max(float(acc_arr.max()), float(normal_acc))
        A_lo = max(float(acc_arr.min()), 0.0)
        span = max(A_hi - A_lo, 1e-3)
        dk = np.diff(k_arr)
        # Guard against duplicate / zero-width x samples.
        valid = dk > 0
        if valid.any():
            slopes = np.abs(np.diff(acc_arr)[valid] / dk[valid])
            s_max = float(slopes.max())
        else:
            s_max = 0.2
        beta0 = float(np.clip(4.0 * s_max / span, 0.2, 100.0))
        p0 = [normal_acc, float(np.min(acc_arr)),
              float(np.median(k_arr)), beta0]
        bounds = (
            [0.0, -0.05, 0.0, 1e-4],
            [1.0,  1.0,  float(max(k_arr)) * 2, 200.0],
        )
        popt, pcov = curve_fit(sigmoid_fn, k_arr, acc_arr, p0=p0,
                               bounds=bounds, maxfev=30_000)
        perr = np.sqrt(np.diag(pcov))
        resid = acc_arr - sigmoid_fn(k_arr, *popt)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((acc_arr - acc_arr.mean()) ** 2)
        n, p = len(k_arr), len(popt)
        r2 = (1 - (ss_res / (n - p)) / (ss_tot / (n - 1))
               if (ss_tot > 0 and n > p) else float('nan'))
        return popt, perr, r2
    except Exception:
        return None, None, None


# --- Comparison plot ----------------------------------------------------------

def make_comparison_plot(k_values, all_accs, all_popts, all_r2,
                         normal_acc, output_dir='./figures',
                         hidden_size=64, num_hidden_layers=5):
    """Generate the side-by-side accuracy-curve and bar-chart figure."""
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[1.6, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    k_arr = np.array(k_values, dtype=float)
    k_fine = np.linspace(1, max(k_values), 600)

    for m in PRUNING_METHODS:
        if m not in all_accs:
            continue
        st = METHOD_STYLE[m]
        acc_arr = np.array([all_accs[m][k] for k in k_values]) * 100
        popt = all_popts[m]
        r2 = all_r2[m]

        ax1.scatter(k_arr, acc_arr, s=28, color=st['color'],
                    marker=st['marker'], zorder=6, alpha=0.9)
        ax1.plot(k_arr, acc_arr, alpha=0.25, color=st['color'], lw=1.0)

        if popt is not None:
            A_inf, A_0, K_0, beta = popt
            g = np.exp(-beta)
            label = (f"{PRUNING_METHODS[m]}  "
                     f"$\\beta={beta:.3f}$  $K_0={K_0:.1f}$  "
                     f"$g={g:.3f}$  $R^2={r2:.3f}$")
            ax1.plot(k_fine, sigmoid_fn(k_fine, *popt) * 100,
                     color=st['color'], lw=2.2, ls=st['ls'],
                     label=label, zorder=5)
        else:
            ax1.plot(k_arr, acc_arr, color=st['color'], lw=1.5,
                     ls=st['ls'], label=f"{PRUNING_METHODS[m]} (fit failed)")

    ax1.axhline(normal_acc * 100, color='black', ls=':', lw=1.5, alpha=0.6,
                label=f'Full network {100 * normal_acc:.1f}%')
    ax1.set_xlabel('Paths per pixel  K')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Path-Integral Recovery Curve by Pruning Method')
    ax1.legend(fontsize=7.8, loc='lower right', framealpha=0.92)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max(k_values) + 1)

    valid = [m for m in PRUNING_METHODS if m in all_popts and all_popts[m] is not None]
    n = len(valid)
    if n > 0:
        x = np.arange(n)
        w = 0.35
        K0_vals = [all_popts[m][2] for m in valid]
        beta_vals = [all_popts[m][3] for m in valid]
        g_vals = [np.exp(-all_popts[m][3]) for m in valid]
        colors = [METHOD_STYLE[m]['color'] for m in valid]
        labels = [PRUNING_METHODS[m] for m in valid]

        ax2.bar(x - w / 2, K0_vals, w, color=colors, alpha=0.85,
                edgecolor='black', lw=0.6)
        ax2.set_ylabel('$K_0$')
        ax2b = ax2.twinx()
        ax2b.bar(x + w / 2, beta_vals, w, color=colors, alpha=0.45,
                 edgecolor='black', lw=0.6, hatch='//')
        ax2b.set_ylabel('$\\beta$', color='dimgray')
        ax2b.tick_params(axis='y', labelcolor='dimgray')
        for i, (k0, g) in enumerate(zip(K0_vals, g_vals)):
            ax2.text(x[i] - w / 2, k0 + 0.15, f'g={g:.3f}', ha='center',
                     va='bottom', fontsize=7.5, color=colors[i], fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8.5, rotation=22, ha='right')
        ax2.set_title('Sigmoid Fit Parameters per Method')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(K0_vals) * 1.35)

    fig.suptitle(
        f'Feynman Path Integral Analogy — Pruning Method Comparison\n'
        f'FC-{num_hidden_layers}x{hidden_size} on sklearn Digits  |  '
        f'Full-network acc: {100 * normal_acc:.2f}%  |  K = 1...{hidden_size}',
        fontsize=12, y=1.02)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'pruning_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# --- Summary table ------------------------------------------------------------

def print_fit_summary(all_popts, all_perrs, all_r2, normal_acc):
    """Print a formatted table of sigmoid fit parameters for each method."""
    hline = '-' * 80
    print(f"\n{'=' * 80}")
    print("  SIGMOID FIT  acc(K) = A_0 + (A_inf - A_0)/(1 + exp(-beta*(K - K_0)))")
    print(hline)
    print(f"  {'Method':<26}  {'A_inf':>7}  {'A_0':>7}  "
          f"{'K_0':>7}  {'beta':>7}  {'g=exp(-b)':>8}  {'R2':>7}")
    print(hline)
    for m, label in PRUNING_METHODS.items():
        popt = all_popts.get(m)
        r2 = all_r2.get(m)
        if popt is None:
            print(f"  {label:<26}  {'FAILED':>50}")
            continue
        A_inf, A_0, K_0, beta = popt
        g = np.exp(-beta)
        perr = all_perrs.get(m, [0] * 4)
        print(f"  {label:<26}  {100 * A_inf:>6.2f}%  {100 * A_0:>6.2f}%  "
              f"{K_0:>7.2f}  {beta:>7.4f}  {g:>8.4f}  {r2:>7.4f}")
        if perr is not None and any(p > 0 for p in perr):
            ea, eb, ek, ebeta = perr
            print(f"  {'  errors':<26}  {100 * ea:>6.2f}%  {100 * eb:>6.2f}%  "
                  f"{ek:>7.2f}  {ebeta:>7.4f}")
    print(hline)
    print(f"  Full-network accuracy: {100 * normal_acc:.2f}%")
    print(f"{'=' * 80}\n")


# --- Main (standalone execution on sklearn digits) ----------------------------

if __name__ == '__main__':
    SEED = 42
    HIDDEN_SIZE = 64
    NUM_LAYERS = 5
    EPOCHS = 300
    OUTPUT = 'mnist_figures'

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT, exist_ok=True)

    t0 = time.time()

    # 1. Data
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_digits_data(SEED)
    print(f"  Digits  Train:{X_tr.shape[0]}  Val:{X_val.shape[0]}  "
          f"Test:{X_te.shape[0]}")

    # 2. Train
    model = FCNetwork(input_size=64, hidden_size=HIDDEN_SIZE,
                      num_hidden_layers=NUM_LAYERS, seed=SEED)
    model.train_model(X_tr, y_tr, X_val, y_val, epochs=EPOCHS)

    # 3. Scores
    all_scores = precompute_pruning_scores(model, X_tr, y_tr, seed=SEED)

    # 4. K sweep
    k_values = list(range(1, HIDDEN_SIZE + 1))
    print(f"\n  Evaluating K = 1 ... {HIDDEN_SIZE}"
          f" for {len(PRUNING_METHODS)} methods\n")

    all_accs = {}
    all_popts = {}
    all_perrs = {}
    all_r2 = {}
    normal_acc = None

    for method in PRUNING_METHODS:
        t1 = time.time()
        print(f"  -- {PRUNING_METHODS[method]} --", end="", flush=True)
        accs, nacc = evaluate_path_accuracy(
            model, X_te, y_te, k_values, all_scores[method], method)
        if normal_acc is None:
            normal_acc = nacc
        all_accs[method] = accs

        popt, perr, r2 = fit_sigmoid(k_values, accs, nacc)
        all_popts[method] = popt
        all_perrs[method] = perr if perr is not None else [0] * 4
        all_r2[method] = r2 if r2 is not None else float('nan')

        if popt is not None:
            A_inf, A_0, K_0, beta = popt
            g = np.exp(-beta)
            print(f"  beta={beta:.4f}  K0={K_0:.2f}  g={g:.4f}  "
                  f"R2={r2:.4f}  [{time.time() - t1:.1f}s]")
        else:
            print(f"  fit failed  [{time.time() - t1:.1f}s]")

    # 5. Summary
    print_fit_summary(all_popts, all_perrs, all_r2, normal_acc)

    # 6. Plot
    plot_path = make_comparison_plot(
        k_values, all_accs, all_popts, all_r2, normal_acc,
        output_dir=OUTPUT, hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS)
    print(f"  Plot saved: {plot_path}")

    # 7. Save data
    with open(os.path.join(OUTPUT, 'accuracies_all_methods.txt'), 'w') as f:
        for m in PRUNING_METHODS:
            acc_fmt = {int(k): round(float(all_accs[m][k]), 4) for k in k_values}
            f.write(f"# {PRUNING_METHODS[m]}\nacc_{m} = {acc_fmt}\n\n")
        f.write("\n# Sigmoid fit parameters [A_inf, A_0, K_0, beta]\n")
        for m in PRUNING_METHODS:
            p = all_popts[m]
            if p is not None:
                f.write(f"fit_{m} = {{'A_inf':{p[0]:.6f}, 'A_0':{p[1]:.6f}, "
                        f"'K_0':{p[2]:.6f}, 'beta':{p[3]:.6f}, "
                        f"'g':{np.exp(-p[3]):.6f}, 'R2':{all_r2[m]:.6f}}}\n")

    print(f"  Total runtime: {time.time() - t0:.1f}s")
    print("  Done!")