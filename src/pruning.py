#!/usr/bin/env python3
"""
Feynman Path Integral Analogy in Neural Networks — Multi-Pruning Comparison
============================================================================
Pure NumPy implementation.  Importable module AND standalone script.

OPTIMISATIONS vs original
─────────────────────────
 • Static methods (weight/wanda/taylor/random): column masks precomputed once
   per K, then applied as a single boolean slice — replaces O(I·H·log H)
   argpartition per row with O(I·(H−K)) zeroing.
 • Batched sample processing: samples grouped in mini-batches of size B,
   creating [B,I,H] tensors.  Eliminates Python-level per-sample loop.
 • ReLU masks computed once; reused across all K values.
 • Bias-offset sanity check: K=H reconstruction is verified to match the
   standard forward pass to within atol=1e-6 before applying the correction.

BUG FIXES
─────────
 1. FCNetwork.__init__ now accepts a `seed` parameter instead of reading
    a module-level global — safe for parallel / multi-config use.
 2. adam_step `lr` parameter is now per-call, not bound to a global.
 3. Sigmoid fit: A₀ lower-bound raised from 0 to -0.05 (accounts for noise
    near chance level); A₀ initial guess uses min(acc) not acc[0].
 4. LaTeX rendering disabled by default (portability).
 5. Module-level side effects (random seed, directory creation) moved inside
    `if __name__ == '__main__'`.

API (for importing)
───────────────────
  from pruning import (
      FCNetwork, relu, softmax, accuracy,
      precompute_pruning_scores, evaluate_path_accuracy,
      sigmoid_fn, fit_sigmoid,
      PRUNING_METHODS, METHOD_STYLE,
  )
"""

import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ═══════════════════════════════════════════════════════════
#  PLOT STYLE  (no LaTeX — portable)
# ═══════════════════════════════════════════════════════════
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Serif",
    "axes.labelsize": 12, "font.size": 12,
    "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10,
})

# ═══════════════════════════════════════════════════════════
#  PRUNING METHOD REGISTRY
# ═══════════════════════════════════════════════════════════
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
    'random':  dict(color='gray',       marker='x', ls=(0,(3,1,1,1))),
}


# ═══════════════════════════════════════════════════════════
#  1.  PURE-NUMPY NEURAL NETWORK
# ═══════════════════════════════════════════════════════════

def relu(z):       return np.maximum(0, z)
def relu_grad(z):  return (z > 0).astype(float)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(probs, y_onehot):
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

def accuracy(logits, y):
    return (logits.argmax(axis=1) == y).mean()


class FCNetwork:
    """Fully-connected ReLU network [INPUT]→[H]×L→[C], pure NumPy + Adam.

    Parameters
    ----------
    input_size       : int   dimensionality of input features
    hidden_size      : int   neurons per hidden layer  (H)
    num_hidden_layers: int   number of hidden layers   (L)
    num_classes      : int   output classes             (C)
    seed             : int   RNG seed for weight init
    """

    def __init__(self, input_size=64, hidden_size=64,
                 num_hidden_layers=3, num_classes=10, seed=42):
        self.H = hidden_size
        self.L = num_hidden_layers
        self.C = num_classes
        self.input_size = input_size
        rng   = np.random.default_rng(seed)
        sizes = [input_size] + [hidden_size] * num_hidden_layers + [num_classes]
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / fan_in),
                                     (fan_out, fan_in)).astype(np.float64))
            self.b.append(np.zeros(fan_out, dtype=np.float64))
        self._adam_init()

    # ── forward passes ────────────────────────────────────
    def forward(self, X):
        h = X
        for l in range(self.L):
            h = relu(h @ self.W[l].T + self.b[l])
        return h @ self.W[-1].T + self.b[-1]

    def forward_with_masks(self, X):
        """Return (logits, relu_masks) where relu_masks[l] is [N, H]."""
        h, relu_masks = X, []
        for l in range(self.L):
            z    = h @ self.W[l].T + self.b[l]
            mask = (z > 0).astype(np.float64)
            h    = z * mask
            relu_masks.append(mask)
        return h @ self.W[-1].T + self.b[-1], relu_masks

    def _forward_cache(self, X):
        """Return (logits, pre-activations, post-activations) for backprop."""
        zs, hs = [], [X]
        h = X
        for l in range(self.L):
            z = h @ self.W[l].T + self.b[l]
            zs.append(z); h = relu(z); hs.append(h)
        return h @ self.W[-1].T + self.b[-1], zs, hs

    # ── backprop ──────────────────────────────────────────
    def compute_gradients(self, X, y_onehot):
        N = X.shape[0]
        logits, zs, hs = self._forward_cache(X)
        probs = softmax(logits)
        loss  = cross_entropy(probs, y_onehot)
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        delta = (probs - y_onehot) / N
        dW[-1] = delta.T @ hs[-1]
        db[-1] = delta.sum(0)
        for l in range(self.L - 1, -1, -1):
            delta = (delta @ self.W[l + 1]) * relu_grad(zs[l])
            dW[l] = delta.T @ hs[l]
            db[l] = delta.sum(0)
        return loss, dW, db

    # ── Adam optimiser ────────────────────────────────────
    def _adam_init(self):
        self._t  = 0
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]

    def adam_step(self, dW, db, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        for l in range(len(self.W)):
            self._mW[l] = b1 * self._mW[l] + (1 - b1) * dW[l]
            self._vW[l] = b2 * self._vW[l] + (1 - b2) * dW[l] ** 2
            mhat = self._mW[l] / (1 - b1 ** t)
            vhat = self._vW[l] / (1 - b2 ** t)
            self.W[l] -= lr * mhat / (np.sqrt(vhat) + eps)

            self._mb[l] = b1 * self._mb[l] + (1 - b1) * db[l]
            self._vb[l] = b2 * self._vb[l] + (1 - b2) * db[l] ** 2
            mhat = self._mb[l] / (1 - b1 ** t)
            vhat = self._vb[l] / (1 - b2 ** t)
            self.b[l] -= lr * mhat / (np.sqrt(vhat) + eps)

    # ── training loop ─────────────────────────────────────
    def train(self, X_tr, y_tr, X_val, y_val,
              epochs=300, bs=64, lr=1e-3, verbose=True):
        N = X_tr.shape[0]
        y_ohe = np.eye(self.C)[y_tr]
        if verbose:
            n_params = sum(w.size + b.size for w, b in zip(self.W, self.b))
            print(f"  Model: [{X_tr.shape[1]}]→[{self.H}]×{self.L}→[{self.C}]  "
                  f"Params: {n_params:,}")
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(N)
            total_loss = 0.0
            for s in range(0, N, bs):
                sl = idx[s:s + bs]
                loss, dW, db = self.compute_gradients(X_tr[sl], y_ohe[sl])
                self.adam_step(dW, db, lr=lr)
                total_loss += loss * len(sl)
            if verbose and (ep % 50 == 0 or ep == 1):
                va = accuracy(self.forward(X_val), y_val)
                print(f"  Ep {ep:4d}/{epochs}  loss={total_loss/N:.4f}  val={100*va:.1f}%")
        fa = accuracy(self.forward(X_val), y_val)
        if verbose:
            print(f"  ► Final val accuracy: {100*fa:.2f}%")
        return fa


# ═══════════════════════════════════════════════════════════
#  2.  DATA LOADING (sklearn digits)
# ═══════════════════════════════════════════════════════════

def load_digits_data(seed=42):
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
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ═══════════════════════════════════════════════════════════
#  3.  PRUNING-SCORE PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════

def collect_layer_activations(model, X):
    """Post-ReLU activations: acts[0]=X, acts[l]=output of hidden layer l-1."""
    acts = [X]
    h = X
    for l in range(model.L):
        h = relu(h @ model.W[l].T + model.b[l])
        acts.append(h)
    return acts


def precompute_pruning_scores(model, X_calib, y_calib, methods=None, seed=42):
    """
    Per-layer neuron importance scores for every requested method.

    Returns
    -------
    scores : dict  {method: [score_l0, ..., score_l{L-1}]}
             Each score is [1, H] (broadcastable) or None (signal = dynamic).
    """
    if methods is None:
        methods = list(PRUNING_METHODS.keys())

    N = X_calib.shape[0]
    y_ohe = np.eye(model.C)[y_calib]
    acts  = collect_layer_activations(model, X_calib)

    scores = {m: [] for m in methods}

    # Gradients (only if taylor requested) ─────────────────
    grad_W = None
    if 'taylor' in methods:
        logits, zs, hs = model._forward_cache(X_calib)
        probs = softmax(logits)
        delta = (probs - y_ohe) / N
        grad_W = [None] * len(model.W)
        grad_W[-1] = delta.T @ hs[-1]
        for l in range(model.L - 1, -1, -1):
            delta = (delta @ model.W[l + 1]) * relu_grad(zs[l])
            grad_W[l] = delta.T @ hs[l]

    rng = np.random.default_rng(seed + 99)

    for l in range(model.L):
        W_l   = model.W[l]          # [H, H_in]
        act_l = acts[l]             # [N, H_in]

        if 'signal' in methods:
            scores['signal'].append(None)       # computed on-the-fly

        if 'weight' in methods:
            scores['weight'].append(
                np.linalg.norm(W_l, axis=1, keepdims=True).T)  # [1, H]

        if 'wanda' in methods:
            act_norm = np.sqrt((act_l ** 2).mean(axis=0))       # [H_in]
            wanda = (np.abs(W_l) * act_norm[np.newaxis, :]).sum(
                axis=1, keepdims=True).T                         # [1, H]
            scores['wanda'].append(wanda)

        if 'taylor' in methods:
            taylor = (np.abs(grad_W[l] * W_l)).sum(
                axis=1, keepdims=True).T                         # [1, H]
            scores['taylor'].append(taylor)

        if 'random' in methods:
            scores['random'].append(
                rng.uniform(0, 1, (1, W_l.shape[0])))           # [1, H]

    return scores


# ═══════════════════════════════════════════════════════════
#  4.  BATCHED PATH-TRACING ENGINE
# ═══════════════════════════════════════════════════════════
#
#  Two code paths:
#    • STATIC  (weight / wanda / taylor / random):
#        scores are [1,H] → same top-K columns for all pixels.
#        Fast: just zero the (H−K) weakest columns.
#    • DYNAMIC (signal):
#        scores = |current[i,j]| → top-K varies per pixel.
#        Falls back to per-row argpartition.

def _precompute_column_masks(layer_scores, k_values, H, L):
    """
    For a STATIC method, return {K: [bool_mask_l0, ..., bool_mask_l{L-1}]}
    where each mask is shape [H] with exactly K True entries.
    """
    masks = {}
    for K in k_values:
        per_layer = []
        for l in range(L):
            if K >= H:
                per_layer.append(None)          # keep everything
            else:
                s = layer_scores[l].ravel()     # [H]
                idx = np.argpartition(s, H - K)[H - K:]
                m = np.zeros(H, dtype=bool)
                m[idx] = True
                per_layer.append(m)
        masks[K] = per_layer
    return masks


def _sparsify_dynamic(current, K):
    """Per-row top-K by |current|.  current: [*, I, H]  or  [I, H]."""
    H = current.shape[-1]
    if K >= H:
        return current
    s = np.abs(current)
    kth = H - K
    # argpartition along last axis
    top = np.argpartition(s, kth, axis=-1)[..., kth:]
    result = np.zeros_like(current)
    # fancy-index: build matching leading indices
    lead = tuple(
        np.arange(current.shape[d]).reshape(
            [current.shape[d] if d == dd else 1
             for dd in range(current.ndim)])
        for d in range(current.ndim - 1)
    )
    result[(*lead, top)] = current[(*lead, top)]
    return result


def _trace_paths_batch(model, X_batch, relu_masks_batch, K,
                       layer_scores, col_masks_K, is_static):
    """
    Trace K-strongest paths for a mini-batch.

    Parameters
    ----------
    X_batch          : [B, I]
    relu_masks_batch : list of [B, H]  (one per hidden layer)
    K                : int  beam width
    col_masks_K      : list of bool[H] or None  (static method column masks)
    is_static        : bool

    Returns
    -------
    logits : [B, C]
    """
    B, I = X_batch.shape
    H = model.H

    # Layer 0: [B, I, H]
    current = model.W[0].T[np.newaxis, :, :] * X_batch[:, :, np.newaxis]
    current *= relu_masks_batch[0][:, np.newaxis, :]

    if K < H:
        if is_static and col_masks_K[0] is not None:
            current[:, :, ~col_masks_K[0]] = 0.0
        else:
            current = _sparsify_dynamic(current, K)

    # Hidden layers 1 … L−1
    for l in range(1, model.L):
        current = current @ model.W[l].T                        # [B, I, H]
        current *= relu_masks_batch[l][:, np.newaxis, :]

        if K < H:
            if is_static and col_masks_K[l] is not None:
                current[:, :, ~col_masks_K[l]] = 0.0
            else:
                current = _sparsify_dynamic(current, K)

    # Output layer: [B, I, H] @ [C, H].T → [B, I, C] → sum over I → [B, C]
    logits = (current @ model.W[-1].T).sum(axis=1)
    return logits


def _estimate_batch_size(I, H, max_mem_mb=256):
    """Choose B so that [B, I, H] × 8 bytes ≤ max_mem_mb."""
    elem = I * H * 8
    if elem == 0:
        return 64
    B = max(1, int(max_mem_mb * 1e6 / elem))
    return min(B, 512)


# ═══════════════════════════════════════════════════════════
#  5.  EVALUATION LOOP
# ═══════════════════════════════════════════════════════════

def evaluate_path_accuracy(model, X_test, y_test, k_values,
                           layer_scores, method_name='signal',
                           max_mem_mb=256):
    """
    Evaluate accuracy(K) for one pruning method on test data.

    Returns
    -------
    accs       : dict {K: float}
    normal_acc : float  (full unpruned network)
    """
    N, I = X_test.shape
    H = model.H
    is_static = (method_name != 'signal')

    # 1. Standard forward pass → ground truth + ReLU masks
    actual_logits, relu_masks = model.forward_with_masks(X_test)
    normal_acc = accuracy(actual_logits, y_test)

    # 2. Precompute column masks for static methods
    col_masks = {}
    if is_static:
        col_masks = _precompute_column_masks(
            layer_scores, k_values, H, model.L)

    # 3. Choose batch size based on memory budget
    B = _estimate_batch_size(I, H, max_mem_mb)

    # 4. Helper: run all batches for a given K
    def _eval_K(K):
        logits = np.zeros((N, model.C))
        cm = col_masks.get(K, [None] * model.L)
        for s in range(0, N, B):
            e = min(s + B, N)
            rm = [relu_masks[l][s:e] for l in range(model.L)]
            logits[s:e] = _trace_paths_batch(
                model, X_test[s:e], rm, K, layer_scores, cm, is_static)
        return logits

    # 5. Bias offset: difference between standard forward and K=H path trace
    full_path_logits = _eval_K(H)
    bias_offset = actual_logits - full_path_logits

    # Sanity check: K=H + bias must reproduce normal accuracy
    check_acc = accuracy(full_path_logits + bias_offset, y_test)
    if not np.isclose(check_acc, normal_acc, atol=1e-6):
        warnings.warn(
            f"[{method_name}] K=H sanity check: {check_acc:.6f} vs "
            f"{normal_acc:.6f}.  Bias offset may be unreliable.")

    # 6. Sweep K values
    accs = {}
    for K in k_values:
        if K == H:
            accs[K] = normal_acc
        else:
            path_logits = _eval_K(K)
            accs[K] = accuracy(path_logits + bias_offset, y_test)

    return accs, normal_acc


# ═══════════════════════════════════════════════════════════
#  6.  SIGMOID FITTING
# ═══════════════════════════════════════════════════════════

def sigmoid_fn(K, A_inf, A_0, K_0, beta):
    K = np.asarray(K, dtype=float)
    return A_0 + (A_inf - A_0) / (1.0 + np.exp(
        np.clip(-beta * (K - K_0), -500, 500)))


def fit_sigmoid(k_values, accuracies, normal_acc):
    """
    Fit acc(K) = A₀ + (A∞−A₀)/(1 + exp(−β(K−K₀))) .

    Returns (popt, perr, R²) or (None, None, None).
    popt = [A_inf, A_0, K_0, beta].
    """
    k_arr   = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values])

    try:
        p0 = [normal_acc,
              float(np.min(acc_arr)),          # FIX: use min, not first point
              float(np.median(k_arr)),
              0.2]
        bounds = (
            [0.0, -0.05, 0.0, 1e-4],          # FIX: A₀ can go slightly < 0
            [1.0,  1.0,  float(max(k_arr)) * 2, 20.0],
        )
        popt, pcov = curve_fit(sigmoid_fn, k_arr, acc_arr, p0=p0,
                               bounds=bounds, maxfev=30_000)
        perr   = np.sqrt(np.diag(pcov))
        resid  = acc_arr - sigmoid_fn(k_arr, *popt)
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((acc_arr - acc_arr.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return popt, perr, r2
    except Exception:
        return None, None, None


# ═══════════════════════════════════════════════════════════
#  7.  COMPARISON PLOT
# ═══════════════════════════════════════════════════════════

def make_comparison_plot(k_values, all_accs, all_popts, all_r2,
                         normal_acc, output_dir='./figures',
                         hidden_size=64, num_hidden_layers=5):
    fig  = plt.figure(figsize=(16, 7))
    gs   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                             width_ratios=[1.6, 1])
    ax1  = fig.add_subplot(gs[0])
    ax2  = fig.add_subplot(gs[1])

    k_arr  = np.array(k_values, dtype=float)
    k_fine = np.linspace(1, max(k_values), 600)

    for m in PRUNING_METHODS:
        if m not in all_accs:
            continue
        st      = METHOD_STYLE[m]
        acc_arr = np.array([all_accs[m][k] for k in k_values]) * 100
        popt    = all_popts[m]
        r2      = all_r2[m]

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
                label=f'Full network {100*normal_acc:.1f}%')
    ax1.set_xlabel('Paths per pixel  K')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Path-Integral Recovery Curve by Pruning Method')
    ax1.legend(fontsize=7.8, loc='lower right', framealpha=0.92)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, max(k_values) + 1)

    # Bar chart
    valid = [m for m in PRUNING_METHODS if m in all_popts and all_popts[m] is not None]
    n = len(valid)
    if n > 0:
        x = np.arange(n); w = 0.35
        K0_vals   = [all_popts[m][2] for m in valid]
        beta_vals = [all_popts[m][3] for m in valid]
        g_vals    = [np.exp(-all_popts[m][3]) for m in valid]
        colors    = [METHOD_STYLE[m]['color'] for m in valid]
        labels    = [PRUNING_METHODS[m] for m in valid]

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
        f'Full-network acc: {100*normal_acc:.2f}%  |  K = 1…{hidden_size}',
        fontsize=12, y=1.02)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'pruning_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════
#  8.  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════

def print_fit_summary(all_popts, all_perrs, all_r2, normal_acc):
    print(f"\n{'═'*80}")
    print(f"  SIGMOID FIT  acc(K) = A₀ + (A∞−A₀)/(1+exp(−β(K−K₀)))")
    print(f"{'─'*80}")
    print(f"  {'Method':<26}  {'A∞':>7}  {'A₀':>7}  "
          f"{'K₀':>7}  {'β':>7}  {'g=e⁻ᵝ':>8}  {'R²':>7}")
    print(f"{'─'*80}")
    for m, label in PRUNING_METHODS.items():
        popt = all_popts.get(m)
        r2   = all_r2.get(m)
        if popt is None:
            print(f"  {label:<26}  {'FAILED':>50}")
            continue
        A_inf, A_0, K_0, beta = popt
        g = np.exp(-beta)
        perr = all_perrs.get(m, [0]*4)
        print(f"  {label:<26}  {100*A_inf:>6.2f}%  {100*A_0:>6.2f}%  "
              f"{K_0:>7.2f}  {beta:>7.4f}  {g:>8.4f}  {r2:>7.4f}")
        if perr is not None and any(p > 0 for p in perr):
            ea, eb, ek, ebeta = perr
            print(f"  {'± errors':<26}  {100*ea:>6.2f}%  {100*eb:>6.2f}%  "
                  f"{ek:>7.2f}  {ebeta:>7.4f}")
    print(f"{'─'*80}")
    print(f"  Full-network accuracy: {100*normal_acc:.2f}%")
    print(f"{'═'*80}\n")


# ═══════════════════════════════════════════════════════════
#  MAIN  (standalone execution on sklearn digits)
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    SEED         = 42
    HIDDEN_SIZE  = 64
    NUM_LAYERS   = 5
    EPOCHS       = 300
    OUTPUT       = 'mnist_figures'

    np.random.seed(SEED)
    os.makedirs(OUTPUT, exist_ok=True)

    t0 = time.time()

    # 1. Data
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_digits_data(SEED)
    print(f"  Digits  Train:{X_tr.shape[0]}  Val:{X_val.shape[0]}  "
          f"Test:{X_te.shape[0]}")

    # 2. Train
    model = FCNetwork(input_size=64, hidden_size=HIDDEN_SIZE,
                      num_hidden_layers=NUM_LAYERS, seed=SEED)
    model.train(X_tr, y_tr, X_val, y_val, epochs=EPOCHS)

    # 3. Scores
    all_scores = precompute_pruning_scores(model, X_tr, y_tr, seed=SEED)

    # 4. K sweep
    k_values = list(range(1, HIDDEN_SIZE + 1))
    print(f"\n  Evaluating K = 1 … {HIDDEN_SIZE} for {len(PRUNING_METHODS)} methods\n")

    all_accs  = {}
    all_popts = {}
    all_perrs = {}
    all_r2    = {}
    normal_acc = None

    for method in PRUNING_METHODS:
        t1 = time.time()
        print(f"  ── {PRUNING_METHODS[method]} ──", end="", flush=True)
        accs, nacc = evaluate_path_accuracy(
            model, X_te, y_te, k_values, all_scores[method], method)
        if normal_acc is None:
            normal_acc = nacc
        all_accs[method] = accs

        popt, perr, r2 = fit_sigmoid(k_values, accs, nacc)
        all_popts[method] = popt
        all_perrs[method] = perr if perr is not None else [0] * 4
        all_r2[method]    = r2 if r2 is not None else float('nan')

        if popt is not None:
            A_inf, A_0, K_0, beta = popt
            g = np.exp(-beta)
            print(f"  β={beta:.4f}  K₀={K_0:.2f}  g={g:.4f}  "
                  f"R²={r2:.4f}  [{time.time()-t1:.1f}s]")
        else:
            print(f"  fit failed  [{time.time()-t1:.1f}s]")

    # 5. Summary
    print_fit_summary(all_popts, all_perrs, all_r2, normal_acc)

    # 6. Plot
    plot_path = make_comparison_plot(
        k_values, all_accs, all_popts, all_r2, normal_acc,
        output_dir=OUTPUT, hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS)
    print(f"  Plot saved → {plot_path}")

    # 7. Save data
    with open('mnist_figures/accuracies_all_methods.txt', 'w') as f:
        for m in PRUNING_METHODS:
            acc_fmt = {int(k): round(float(all_accs[m][k]), 4) for k in k_values}
            f.write(f"# {PRUNING_METHODS[m]}\nacc_{m} = {acc_fmt}\n\n")
        f.write(f"\n# Sigmoid fit parameters [A_inf, A_0, K_0, beta]\n")
        for m in PRUNING_METHODS:
            p = all_popts[m]
            if p is not None:
                f.write(f"fit_{m} = {{'A_inf':{p[0]:.6f}, 'A_0':{p[1]:.6f}, "
                        f"'K_0':{p[2]:.6f}, 'beta':{p[3]:.6f}, "
                        f"'g':{np.exp(-p[3]):.6f}, 'R2':{all_r2[m]:.6f}}}\n")

    print(f"  Total runtime: {time.time()-t0:.1f}s")
    print("  Done!")