#!/usr/bin/env python3
"""
Scaling Laws for Neural Network Effective Coupling Constants
=============================================================

Tasks 3 & 4: Train networks with varying width H and depth L,
extract sigmoid parameters (K0, beta, g_eff), and search for
scaling laws relating these to architecture.

Also implements improvements from Task 1 (bug fixes).
"""

import os, time, warnings, copy, json, itertools
warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import curve_fit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
OUTPUT_DIR = './pruning_coupling/mnist_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
#  NEURAL NETWORK (with bug fixes from Task 1)
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
    """Fully-connected ReLU network, pure NumPy + Adam."""

    def __init__(self, input_size=64, hidden_size=64,
                 num_hidden_layers=3, num_classes=10, seed=42):
        self.H = hidden_size
        self.L = num_hidden_layers
        self.C = num_classes
        rng = np.random.default_rng(seed)
        sizes = [input_size] + [hidden_size] * num_hidden_layers + [num_classes]
        self.W = []
        self.b = []
        for i_in, i_out in zip(sizes[:-1], sizes[1:]):
            # He initialization
            self.W.append(rng.normal(0, np.sqrt(2.0/i_in), (i_out, i_in)).astype(np.float64))
            self.b.append(np.zeros(i_out, dtype=np.float64))
        self._adam_init()

    def forward(self, X):
        h = X
        for l in range(self.L):
            h = relu(h @ self.W[l].T + self.b[l])
        return h @ self.W[-1].T + self.b[-1]

    def forward_with_masks(self, X):
        h, relu_masks = X, []
        for l in range(self.L):
            z = h @ self.W[l].T + self.b[l]
            mask = (z > 0).astype(np.float64)
            h = z * mask
            relu_masks.append(mask)
        return h @ self.W[-1].T + self.b[-1], relu_masks

    def _forward_cache(self, X):
        zs, hs = [], [X]
        h = X
        for l in range(self.L):
            z = h @ self.W[l].T + self.b[l]
            zs.append(z); h = relu(z); hs.append(h)
        return h @ self.W[-1].T + self.b[-1], zs, hs

    def compute_gradients(self, X, y_onehot):
        N = X.shape[0]
        logits, zs, hs = self._forward_cache(X)
        probs = softmax(logits)
        loss = cross_entropy(probs, y_onehot)
        dW, db = [None]*len(self.W), [None]*len(self.b)
        delta = (probs - y_onehot) / N
        dW[-1] = delta.T @ hs[-1]; db[-1] = delta.sum(0)
        for l in range(self.L - 1, -1, -1):
            delta = (delta @ self.W[l+1]) * relu_grad(zs[l])
            dW[l] = delta.T @ hs[l]; db[l] = delta.sum(0)
        return loss, dW, db

    def _adam_init(self):
        self._t = 0
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]

    def adam_step(self, dW, db, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        for l in range(len(self.W)):
            self._mW[l] = b1*self._mW[l] + (1-b1)*dW[l]
            self._vW[l] = b2*self._vW[l] + (1-b2)*dW[l]**2
            self.W[l] -= lr*(self._mW[l]/(1-b1**t))/(np.sqrt(self._vW[l]/(1-b2**t))+eps)
            self._mb[l] = b1*self._mb[l] + (1-b1)*db[l]
            self._vb[l] = b2*self._vb[l] + (1-b2)*db[l]**2
            self.b[l] -= lr*(self._mb[l]/(1-b1**t))/(np.sqrt(self._vb[l]/(1-b2**t))+eps)

    def train(self, X_tr, y_tr, X_val, y_val, epochs=300, bs=64, verbose=False):
        N = X_tr.shape[0]
        y_ohe = np.eye(self.C)[y_tr]
        for ep in range(1, epochs+1):
            idx = np.random.permutation(N)
            for s in range(0, N, bs):
                sl = idx[s:s+bs]
                loss, dW, db = self.compute_gradients(X_tr[sl], y_ohe[sl])
                self.adam_step(dW, db)
            if verbose and (ep % 100 == 0 or ep == 1):
                va = accuracy(self.forward(X_val), y_val)
                print(f"    Ep {ep:4d}  val={100*va:.1f}%")
        return accuracy(self.forward(X_val), y_val)


# ═══════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════

def load_data():
    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                  random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                  random_state=42, stratify=y_tmp)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te = sc.transform(X_te)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ═══════════════════════════════════════════════════════════
#  PATH PRUNING (vectorized - BUG FIX: much faster)
# ═══════════════════════════════════════════════════════════

def sparsify(current, k, H):
    """Keep top-K entries per row by absolute value."""
    if k >= H:
        return current
    s = np.abs(current)
    kth = H - k
    top_idx = np.argpartition(s, kth, axis=1)[:, kth:]
    result = np.zeros_like(current)
    rows = np.arange(current.shape[0])[:, np.newaxis]
    result[rows, top_idx] = current[rows, top_idx]
    return result


def evaluate_pruned_accuracy(model, X_test, y_test, k_values):
    """Evaluate accuracy for each K using signal-based (magnitude) pruning."""
    N = X_test.shape[0]
    H = model.H
    actual_logits, relu_masks_list = model.forward_with_masks(X_test)
    normal_acc = accuracy(actual_logits, y_test)

    accs = {}
    for K in k_values:
        # Vectorized: process all samples at once via batch path computation
        # For each sample, trace through the network keeping top-K activations
        pl = np.zeros((N, model.C))
        for n in range(N):
            x_n = X_test[n]
            # Layer 0
            current = model.W[0].T * x_n[:, np.newaxis]  # [I, H]
            current *= relu_masks_list[0][n][np.newaxis, :]
            current = sparsify(current, K, H)

            # Hidden layers 1..L-1
            for l in range(1, model.L):
                current = current @ model.W[l].T
                current *= relu_masks_list[l][n][np.newaxis, :]
                current = sparsify(current, K, H)

            pl[n] = (current @ model.W[-1].T).sum(axis=0)

        # Bias correction
        if K == H:
            bias_offset = actual_logits - pl
        accs[K] = accuracy(pl, y_test)

    return accs, normal_acc


# ═══════════════════════════════════════════════════════════
#  SIGMOID FIT
# ═══════════════════════════════════════════════════════════

def sigmoid_fn(K, A_inf, A_0, K_0, beta):
    K = np.asarray(K, dtype=float)
    return A_0 + (A_inf - A_0) / (1.0 + np.exp(np.clip(-beta*(K - K_0), -500, 500)))


def fit_sigmoid(k_values, accuracies, normal_acc):
    """Fit sigmoid and return (popt, perr, r2) or (None, None, None)."""
    k_arr = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values])
    try:
        p0 = [normal_acc, acc_arr[0], np.median(k_arr), 0.2]
        bounds = ([0, 0, 0, 1e-4], [1.0, 1.0, float(max(k_arr))*2, 20.0])
        popt, pcov = curve_fit(sigmoid_fn, k_arr, acc_arr, p0=p0,
                                bounds=bounds, maxfev=30000)
        perr = np.sqrt(np.diag(pcov))
        resid = acc_arr - sigmoid_fn(k_arr, *popt)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((acc_arr - acc_arr.mean())**2)
        n = len(k_arr)
        p = len(popt)
        r2 = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else float('nan')
        return popt, perr, r2
    except Exception as e:
        return None, None, None


# Also fit pure exponential for comparison (original paper eq 1)
def exp_fn(K, A_inf, A_0, tau):
    K = np.asarray(K, dtype=float)
    return A_inf - (A_inf - A_0) * np.exp(-K / tau)

def fit_exponential(k_values, accuracies, normal_acc):
    k_arr = np.array(k_values, dtype=float)
    acc_arr = np.array([accuracies[k] for k in k_values])
    try:
        p0 = [normal_acc, acc_arr[0], np.median(k_arr)]
        bounds = ([0, 0, 0.1], [1.0, 1.0, float(max(k_arr))*5])
        popt, pcov = curve_fit(exp_fn, k_arr, acc_arr, p0=p0,
                                bounds=bounds, maxfev=30000)
        resid = acc_arr - exp_fn(k_arr, *popt)
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((acc_arr - acc_arr.mean())**2)
        n = len(k_arr)
        p = len(popt)
        r2 = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else float('nan')
        return popt, r2
    except:
        return None, None


# ═══════════════════════════════════════════════════════════
#  TASK 4: SCALING LAW SCAN
# ═══════════════════════════════════════════════════════════

def run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """
    Train networks for a grid of (H, L) and extract sigmoid parameters.
    Returns a list of result dicts.
    """
    # Grid of architectures
    H_values = [16, 24, 32, 48, 56, 64, 96, 128]
    L_values = [2, 3, 4, 5, 7, 8, 10, 12]

    results = []
    total = len(H_values) * len(L_values)
    count = 0

    for H in H_values:
        for L in L_values:
            count += 1
            t0 = time.time()
            print(f"\n  [{count}/{total}] H={H}, L={L}", end="", flush=True)

            # Train
            model = FCNetwork(input_size=64, hidden_size=H,
                              num_hidden_layers=L, num_classes=10, seed=42)
            # More epochs for deeper/narrower networks
            epochs = 300 if H >= 32 else 500
            val_acc = model.train(X_tr, y_tr, X_val, y_val, epochs=epochs)
            print(f"  val={100*val_acc:.1f}%", end="", flush=True)

            # Skip if model didn't train well enough
            if val_acc < 0.20:
                print(f"  SKIP (val too low)")
                continue

            # Evaluate pruned accuracy for K = 1..H
            k_values = list(range(1, H + 1))
            accs, normal_acc = evaluate_pruned_accuracy(model, X_te, y_te, k_values)

            # Fit sigmoid
            popt_sig, perr_sig, r2_sig = fit_sigmoid(k_values, accs, normal_acc)

            # Fit exponential for comparison
            popt_exp, r2_exp = fit_exponential(k_values, accs, normal_acc)

            res = {
                'H': H, 'L': L,
                'val_acc': float(val_acc),
                'normal_acc': float(normal_acc),
                'n_params': sum(w.size + b.size for w, b in zip(model.W, model.b)),
                'accs': {int(k): float(v) for k, v in accs.items()},
            }

            if popt_sig is not None:
                A_inf, A_0, K_0, beta = popt_sig
                g_eff = np.exp(-beta)
                res.update({
                    'sigmoid_A_inf': float(A_inf),
                    'sigmoid_A_0': float(A_0),
                    'sigmoid_K_0': float(K_0),
                    'sigmoid_beta': float(beta),
                    'sigmoid_g_eff': float(g_eff),
                    'sigmoid_R2': float(r2_sig),
                })
                print(f"  K0={K_0:.1f} β={beta:.3f} g={g_eff:.3f} R²={r2_sig:.3f}", end="")
            else:
                res['sigmoid_R2'] = None
                print(f"  sigmoid fit FAILED", end="")

            if popt_exp is not None:
                res.update({
                    'exp_A_inf': float(popt_exp[0]),
                    'exp_A_0': float(popt_exp[1]),
                    'exp_tau': float(popt_exp[2]),
                    'exp_R2': float(r2_exp),
                })

            dt = time.time() - t0
            print(f"  [{dt:.0f}s]")
            results.append(res)

    return results


# ═══════════════════════════════════════════════════════════
#  SCALING LAW FITTING
# ═══════════════════════════════════════════════════════════

def power_law(x, a, b):
    return a * np.power(x, b)

def power_law_2d(HL, a, alpha, gamma):
    """K0 = a * H^alpha * L^gamma"""
    H, L = HL
    return a * np.power(H, alpha) * np.power(L, gamma)

def log_linear_beta(HL, a, alpha, gamma):
    """beta = a * H^alpha * L^gamma"""
    H, L = HL
    return a * np.power(H, alpha) * np.power(L, gamma)

def fit_scaling_laws(results):
    """
    Attempt to find scaling laws for K0 and beta as functions of H and L.
    """
    # Filter to good sigmoid fits
    good = [r for r in results if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 5:
        print("  Not enough good fits for scaling law analysis")
        return None

    H_arr = np.array([r['H'] for r in good], dtype=float)
    L_arr = np.array([r['L'] for r in good], dtype=float)
    K0_arr = np.array([r['sigmoid_K_0'] for r in good])
    beta_arr = np.array([r['sigmoid_beta'] for r in good])
    g_arr = np.array([r['sigmoid_g_eff'] for r in good])
    A_inf_arr = np.array([r['sigmoid_A_inf'] for r in good])

    scaling_results = {}

    # ── K0 scaling ──────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  SCALING LAW ANALYSIS ({len(good)} good fits, R² > 0.80)")
    print(f"{'═'*70}")

    # Try K0 = a * H^alpha * L^gamma
    try:
        popt, pcov = curve_fit(power_law_2d, (H_arr, L_arr), K0_arr,
                                p0=[1.0, 0.5, 0.5], maxfev=10000,
                                bounds=([0, -3, -3], [1000, 3, 3]))
        K0_pred = power_law_2d((H_arr, L_arr), *popt)
        ss_res = np.sum((K0_arr - K0_pred)**2)
        ss_tot = np.sum((K0_arr - K0_arr.mean())**2)
        n = len(K0_arr); p = len(popt)
        r2_K0 = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else 0
        perr = np.sqrt(np.diag(pcov))
        a_K0, alpha_K0, gamma_K0 = popt
        scaling_results['K0'] = {
            'a': float(a_K0), 'alpha': float(alpha_K0), 'gamma': float(gamma_K0),
            'R2': float(r2_K0), 'formula': f'K0 = {a_K0:.3f} * H^{alpha_K0:.3f} * L^{gamma_K0:.3f}'
        }
        print(f"\n  K₀ = {a_K0:.3f} × H^{alpha_K0:.3f} × L^{gamma_K0:.3f}")
        print(f"       ± ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R² = {r2_K0:.4f}")
    except Exception as e:
        print(f"  K₀ power-law fit failed: {e}")

    # Try beta = a * H^alpha * L^gamma
    try:
        popt, pcov = curve_fit(log_linear_beta, (H_arr, L_arr), beta_arr,
                                p0=[1.0, -0.5, -0.5], maxfev=10000,
                                bounds=([0, -3, -3], [100, 3, 3]))
        beta_pred = log_linear_beta((H_arr, L_arr), *popt)
        ss_res = np.sum((beta_arr - beta_pred)**2)
        ss_tot = np.sum((beta_arr - beta_arr.mean())**2)
        n = len(beta_arr); p = len(popt)
        r2_beta = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else 0
        perr = np.sqrt(np.diag(pcov))
        a_b, alpha_b, gamma_b = popt
        scaling_results['beta'] = {
            'a': float(a_b), 'alpha': float(alpha_b), 'gamma': float(gamma_b),
            'R2': float(r2_beta), 'formula': f'β = {a_b:.3f} * H^{alpha_b:.3f} * L^{gamma_b:.3f}'
        }
        print(f"\n  β = {a_b:.3f} × H^{alpha_b:.3f} × L^{gamma_b:.3f}")
        print(f"       ± ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R² = {r2_beta:.4f}")
    except Exception as e:
        print(f"  β power-law fit failed: {e}")

    # Try g_eff = a * H^alpha * L^gamma
    try:
        popt, pcov = curve_fit(log_linear_beta, (H_arr, L_arr), g_arr,
                                p0=[0.5, 0.1, 0.1], maxfev=10000,
                                bounds=([0, -3, -3], [2, 3, 3]))
        g_pred = log_linear_beta((H_arr, L_arr), *popt)
        ss_res = np.sum((g_arr - g_pred)**2)
        ss_tot = np.sum((g_arr - g_arr.mean())**2)
        n = len(g_arr); p = len(popt)
        r2_g = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else 0
        perr = np.sqrt(np.diag(pcov))
        a_g, alpha_g, gamma_g = popt
        scaling_results['g_eff'] = {
            'a': float(a_g), 'alpha': float(alpha_g), 'gamma': float(gamma_g),
            'R2': float(r2_g), 'formula': f'g = {a_g:.3f} * H^{alpha_g:.3f} * L^{gamma_g:.3f}'
        }
        print(f"\n  g_eff = {a_g:.3f} × H^{alpha_g:.3f} × L^{gamma_g:.3f}")
        print(f"       ± ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R² = {r2_g:.4f}")
    except Exception as e:
        print(f"  g_eff power-law fit failed: {e}")

    # ── Additional: K0/H ratio analysis ──────────────────
    K0_over_H = K0_arr / H_arr
    print(f"\n  K₀/H ratio statistics:")
    print(f"    mean = {K0_over_H.mean():.3f} ± {K0_over_H.std():.3f}")
    print(f"    range = [{K0_over_H.min():.3f}, {K0_over_H.max():.3f}]")
    scaling_results['K0_over_H'] = {
        'mean': float(K0_over_H.mean()), 'std': float(K0_over_H.std())
    }

    # ── Fixed-L slices: K0 vs H for each L ──────────────
    print(f"\n  Fixed-L slices (K₀ vs H):")
    unique_L = sorted(set(r['L'] for r in good))
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if len(subset) >= 3:
            Hs = np.array([r['H'] for r in subset], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in subset])
            try:
                popt, _ = curve_fit(power_law, Hs, K0s, p0=[1, 0.5], maxfev=5000)
                K0_pred = power_law(Hs, *popt)
                ss_res = np.sum((K0s - K0_pred)**2)
                ss_tot = np.sum((K0s - K0s.mean())**2)
                n = len(K0s); p = len(popt)
                r2 = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else 0
                print(f"    L={L_val}: K₀ = {popt[0]:.3f} x H^{popt[1]:.3f}  R²={r2:.3f}")
            except:
                print(f"    L={L_val}: fit failed")

    # ── Fixed-H slices: K0 vs L ──────────────────────────
    print(f"\n  Fixed-H slices (K₀ vs L):")
    unique_H = sorted(set(r['H'] for r in good))
    for H_val in unique_H:
        subset = [r for r in good if r['H'] == H_val]
        if len(subset) >= 3:
            Ls = np.array([r['L'] for r in subset], dtype=float)
            K0s = np.array([r['sigmoid_K_0'] for r in subset])
            try:
                popt, _ = curve_fit(power_law, Ls, K0s, p0=[1, 0.5], maxfev=5000)
                K0_pred = power_law(Ls, *popt)
                ss_res = np.sum((K0s - K0_pred)**2)
                ss_tot = np.sum((K0s - K0s.mean())**2)
                n = len(K0s); p = len(popt)
                r2 = 1 - (ss_res / (n - p)) / (ss_tot / (n - 1)) if (ss_tot > 0 and n > p) else 0
                print(f"    H={H_val}: K₀ = {popt[0]:.3f} × L^{popt[1]:.3f}  R²={r2:.3f}")
            except:
                print(f"    H={H_val}: fit failed")

    print(f"{'═'*70}")
    return scaling_results


# ═══════════════════════════════════════════════════════════
#  COMPREHENSIVE VISUALIZATION
# ═══════════════════════════════════════════════════════════

def make_scaling_plots(results, scaling_results):
    """Create comprehensive scaling law visualizations."""
    good = [r for r in results if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 3:
        print("  Not enough data for plots")
        return

    H_arr = np.array([r['H'] for r in good], dtype=float)
    L_arr = np.array([r['L'] for r in good], dtype=float)
    K0_arr = np.array([r['sigmoid_K_0'] for r in good])
    beta_arr = np.array([r['sigmoid_beta'] for r in good])
    g_arr = np.array([r['sigmoid_g_eff'] for r in good])
    A_inf_arr = np.array([r['sigmoid_A_inf'] for r in good])
    normal_arr = np.array([r['normal_acc'] for r in good])

    # ═══ Figure 1: Accuracy curves by architecture ═══
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Path-Pruning Accuracy Curves by Architecture\n'
                 'Sigmoid Fit: $A(K) = A_0 + (A_\\infty - A_0)/(1 + e^{-\\beta(K-K_0)})$',
                 fontsize=13, y=1.02)

    unique_L = sorted(set(r['L'] for r in good))
    unique_H = sorted(set(r['H'] for r in good))
    colors_L = plt.cm.viridis(np.linspace(0.15, 0.85, len(unique_L)))
    colors_H = plt.cm.plasma(np.linspace(0.15, 0.85, len(unique_H)))
    L_color = {L: c for L, c in zip(unique_L, colors_L)}
    H_color = {H: c for H, c in zip(unique_H, colors_H)}

    # Top row: accuracy curves grouped by L
    for idx, L_val in enumerate(unique_L[:3]):
        ax = axes[0, idx]
        subset = [r for r in good if r['L'] == L_val]
        for r in sorted(subset, key=lambda x: x['H']):
            k_vals = sorted(r['accs'].keys())
            acc_vals = [r['accs'][k]*100 for k in k_vals]
            k_fine = np.linspace(1, max(k_vals), 300)
            ax.scatter(k_vals, acc_vals, s=12, color=H_color[r['H']], alpha=0.7)
            if r.get('sigmoid_R2') and r['sigmoid_R2'] > 0.80:
                fit_line = sigmoid_fn(k_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                       r['sigmoid_K_0'], r['sigmoid_beta']) * 100
                ax.plot(k_fine, fit_line, color=H_color[r['H']], lw=1.5,
                        label=f'H={r["H"]} (g={r["sigmoid_g_eff"]:.2f})')
            ax.axhline(r['normal_acc']*100, color=H_color[r['H']], ls=':', lw=0.5, alpha=0.4)
        ax.set_title(f'L = {L_val} layers', fontsize=11)
        ax.set_xlabel('K (paths per pixel)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    # Bottom-left: K0 vs H colored by L
    ax = axes[1, 0]
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if subset:
            Hs = [r['H'] for r in subset]
            K0s = [r['sigmoid_K_0'] for r in subset]
            ax.scatter(Hs, K0s, s=80, color=L_color[L_val], edgecolors='black',
                       lw=0.5, label=f'L={L_val}', zorder=5)
    # Add scaling law fit line if available
    if scaling_results and 'K0' in scaling_results:
        sr = scaling_results['K0']
        for L_val in unique_L:
            H_fine = np.linspace(min(H_arr), max(H_arr), 100)
            K0_pred = sr['a'] * H_fine**sr['alpha'] * L_val**sr['gamma']
            ax.plot(H_fine, K0_pred, '--', color=L_color[L_val], alpha=0.5, lw=1)
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$K_0$ (inflection point)')
    ax.set_title(f'$K_0$ vs Width H')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-center: beta vs H colored by L
    ax = axes[1, 1]
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if subset:
            Hs = [r['H'] for r in subset]
            betas = [r['sigmoid_beta'] for r in subset]
            ax.scatter(Hs, betas, s=80, color=L_color[L_val], edgecolors='black',
                       lw=0.5, label=f'L={L_val}', zorder=5)
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$\\beta$ (growth rate)')
    ax.set_title('$\\beta$ vs Width H')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: g_eff vs H colored by L
    ax = axes[1, 2]
    for L_val in unique_L:
        subset = [r for r in good if r['L'] == L_val]
        if subset:
            Hs = [r['H'] for r in subset]
            gs = [r['sigmoid_g_eff'] for r in subset]
            ax.scatter(Hs, gs, s=80, color=L_color[L_val], edgecolors='black',
                       lw=0.5, label=f'L={L_val}', zorder=5)
    ax.axhline(1.0, color='red', ls=':', lw=1.5, alpha=0.5, label='$g=1$ (strongly coupled)')
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$g_{eff} = e^{-\\beta}$')
    ax.set_title('Effective Coupling vs Width')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'scaling_curves.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path1}")

    # ═══ Figure 2: Scaling law summary ═══
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('Scaling Laws for Effective Coupling Parameters',
                 fontsize=13, y=1.03)

    # K0/H ratio
    ax = axes[0]
    K0_over_H = K0_arr / H_arr
    for L_val in unique_L:
        mask = L_arr == L_val
        ax.scatter(H_arr[mask], K0_over_H[mask], s=80, color=L_color[L_val],
                   edgecolors='black', lw=0.5, label=f'L={L_val}', zorder=5)
    ax.axhline(K0_over_H.mean(), color='gray', ls='--', lw=1.5, alpha=0.6,
               label=f'mean={K0_over_H.mean():.2f}')
    ax.set_xlabel('H (hidden size)')
    ax.set_ylabel('$K_0 / H$')
    ax.set_title('Critical Path Fraction $K_0/H$')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # g_eff vs L for different H
    ax = axes[1]
    for H_val in unique_H:
        subset = [r for r in good if r['H'] == H_val]
        if len(subset) >= 2:
            Ls = sorted([r['L'] for r in subset])
            gs = [next(r['sigmoid_g_eff'] for r in subset if r['L'] == L) for L in Ls]
            ax.plot(Ls, gs, 'o-', color=H_color[H_val], lw=1.5, ms=7,
                    label=f'H={H_val}')
    ax.axhline(1.0, color='red', ls=':', lw=1.5, alpha=0.5)
    ax.set_xlabel('L (number of hidden layers)')
    ax.set_ylabel('$g_{eff}$')
    ax.set_title('Coupling Strength vs Depth')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Compressibility: accuracy at K=K0/2 vs g_eff
    ax = axes[2]
    for r in good:
        K_half = max(1, int(r['sigmoid_K_0'] / 2))
        K_half = min(K_half, max(r['accs'].keys()))
        acc_half = r['accs'].get(K_half, list(r['accs'].values())[0])
        ax.scatter(r['sigmoid_g_eff'], acc_half * 100, s=80,
                   color=L_color[r['L']], edgecolors='black', lw=0.5, zorder=5)
    ax.set_xlabel('$g_{eff} = e^{-\\beta}$')
    ax.set_ylabel('Accuracy at $K = K_0/2$ (%)')
    ax.set_title('Compressibility vs Coupling')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'scaling_laws.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path2}")

    # ═══ Figure 3: Heatmaps ═══
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Parameter Heatmaps Across Architecture Grid', fontsize=13, y=1.03)

    for idx, (param, label, cmap) in enumerate([
        ('sigmoid_K_0', '$K_0$', 'YlOrRd'),
        ('sigmoid_beta', '$\\beta$', 'YlGnBu'),
        ('sigmoid_g_eff', '$g_{eff}$', 'RdYlGn_r')
    ]):
        ax = axes[idx]
        # Build grid
        H_grid = sorted(set(r['H'] for r in good))
        L_grid = sorted(set(r['L'] for r in good))
        data = np.full((len(L_grid), len(H_grid)), np.nan)
        for r in good:
            i = L_grid.index(r['L'])
            j = H_grid.index(r['H'])
            data[i, j] = r[param]

        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks(range(len(H_grid)))
        ax.set_xticklabels(H_grid)
        ax.set_yticks(range(len(L_grid)))
        ax.set_yticklabels(L_grid)
        ax.set_xlabel('H (hidden size)')
        ax.set_ylabel('L (layers)')
        ax.set_title(label)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(len(L_grid)):
            for j in range(len(H_grid)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if data[i,j] > np.nanmean(data)*1.3 else 'black')

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, 'parameter_heatmaps.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path3}")

    return [path1, path2, path3]


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_total = time.time()

    print("=" * 70)
    print("  SCALING LAW ANALYSIS FOR NEURAL NETWORK EFFECTIVE COUPLING")
    print("=" * 70)

    # Load data
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_data()
    print(f"  Data: Train={X_tr.shape[0]}, Val={X_val.shape[0]}, Test={X_te.shape[0]}")

    # Run the scaling scan
    print(f"\n  Starting architecture scan...")
    results = run_scaling_scan(X_tr, X_val, X_te, y_tr, y_val, y_te)

    # Save raw results
    with open(os.path.join(OUTPUT_DIR, 'scaling_results.json'), 'w') as f:
        # Convert numpy types for JSON serialization
        clean_results = []
        for r in results:
            cr = {}
            for k, v in r.items():
                if k == 'accs':
                    cr[k] = {str(kk): vv for kk, vv in v.items()}
                elif isinstance(v, (np.floating, np.integer)):
                    cr[k] = float(v)
                else:
                    cr[k] = v
            clean_results.append(cr)
        json.dump(clean_results, f, indent=2)

    # Fit scaling laws
    scaling_results = fit_scaling_laws(results)

    # Print comprehensive results table
    print(f"\n{'═'*100}")
    print(f"  {'H':>4}  {'L':>3}  {'Params':>8}  {'ValAcc':>7}  "
          f"{'A∞':>6}  {'A₀':>6}  {'K₀':>6}  {'β':>7}  {'g_eff':>7}  {'R²':>6}  {'K₀/H':>5}")
    print(f"{'─'*100}")
    for r in sorted(results, key=lambda x: (x['L'], x['H'])):
        if r.get('sigmoid_R2') is not None:
            K0_H = r['sigmoid_K_0'] / r['H']
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  {100*r['val_acc']:>6.1f}%  "
                  f"{100*r['sigmoid_A_inf']:>5.1f}%  {100*r['sigmoid_A_0']:>5.1f}%  "
                  f"{r['sigmoid_K_0']:>6.1f}  {r['sigmoid_beta']:>7.4f}  "
                  f"{r['sigmoid_g_eff']:>7.4f}  {r['sigmoid_R2']:>6.3f}  {K0_H:>5.2f}")
        else:
            print(f"  {r['H']:>4}  {r['L']:>3}  {r['n_params']:>8,}  {100*r['val_acc']:>6.1f}%  "
                  f"{'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>6}  {'—':>5}")
    print(f"{'═'*100}")

    # Generate plots
    print(f"\n  Generating visualizations...")
    plot_paths = make_scaling_plots(results, scaling_results)

    # Save scaling results
    if scaling_results:
        with open(os.path.join(OUTPUT_DIR, 'scaling_laws.json'), 'w') as f:
            json.dump(scaling_results, f, indent=2)

    dt = time.time() - t_total
    print(f"\n  Total runtime: {dt:.0f}s")
    print("  Done!")
