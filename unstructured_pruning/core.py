"""Shared utilities for unstructured pruning: mask application, evaluation, power-law fits."""

from __future__ import annotations

import copy
import json
import os
import time
import warnings

import numpy as np
import torch
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def apply_mask(model, masks):
    """Return a deep copy of ``model`` with hidden-layer weights multiplied by ``masks``.

    Parameters
    ----------
    model : FCNetwork
    masks : list of Tensor, one per hidden layer (same shape as ``layer.weight``)

    Returns
    -------
    FCNetwork (independent copy; caller's model untouched)
    """
    pruned = copy.deepcopy(model)
    with torch.no_grad():
        for layer, mask in zip(pruned.layers[:-1], masks):
            layer.weight.data.mul_(mask.to(layer.weight.dtype).to(layer.weight.device))
    return pruned


def evaluate_masked_accuracy(model, X_test, y_test, mask_sets):
    """Evaluate test accuracy for each density ``s``, averaging over mask realisations.

    Parameters
    ----------
    model     : FCNetwork (unpruned reference)
    X_test    : ndarray [N, D]
    y_test    : ndarray [N]
    mask_sets : dict {s: [mask_set_seed_0, mask_set_seed_1, ...]}
                where each mask_set is a list of per-hidden-layer tensors.

    Returns
    -------
    accs       : dict {s: (mean_acc, std_acc)}
    normal_acc : float  (unpruned reference accuracy)
    """
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    X_t = torch.as_tensor(X_test, dtype=dtype, device=device)
    y_np = np.asarray(y_test)

    model.eval()
    with torch.no_grad():
        normal_acc = float((model(X_t).argmax(1).cpu().numpy() == y_np).mean())

    accs = {}
    for s, seed_masks in mask_sets.items():
        per_seed = []
        for masks in seed_masks:
            pruned = apply_mask(model, masks)
            pruned.eval()
            with torch.no_grad():
                pred = pruned(X_t).argmax(1).cpu().numpy()
            per_seed.append(float((pred == y_np).mean()))
        accs[float(s)] = (float(np.mean(per_seed)), float(np.std(per_seed)))
    return accs, normal_acc


# --- Scaling law fitting (factored from pruning/mnist_scaling.py) --------------

def _power_law_2d(HL, a, alpha, gamma):
    H, L = HL
    return a * np.power(H, alpha) * np.power(L, gamma)


def _adj_r2(y, y_pred, n_params):
    n = len(y)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 0 or n <= n_params:
        return float('nan')
    return 1 - (ss_res / (n - n_params)) / (ss_tot / (n - 1))


def _fit_power_law_2d(H_arr, L_arr, y_arr, p0, bounds, label):
    try:
        popt, pcov = curve_fit(_power_law_2d, (H_arr, L_arr), y_arr,
                               p0=p0, bounds=bounds, maxfev=10_000)
        a, alpha, gamma = popt
        r2 = _adj_r2(y_arr, _power_law_2d((H_arr, L_arr), *popt), 3)
        perr = np.sqrt(np.diag(pcov))
        print(f"\n  {label} = {a:.4f} * H^{alpha:.3f} * L^{gamma:.3f}")
        print(f"       +/- ({perr[0]:.3f}, {perr[1]:.3f}, {perr[2]:.3f})")
        print(f"       R2_adj = {r2:.4f}")
        return {
            'a': float(a), 'alpha': float(alpha), 'gamma': float(gamma),
            'R2': float(r2),
            'formula': f'{label} = {a:.4f} * H^{alpha:.3f} * L^{gamma:.3f}',
        }
    except Exception as e:
        warnings.warn(f"{label} power-law fit failed: {e}")
        return None


def fit_scaling_laws(results, min_r2=0.80):
    """Fit s_0 / beta / g_eff as power laws in (H, L)."""
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > min_r2]
    if len(good) < 5:
        print(f"  Not enough good fits ({len(good)}) for scaling law analysis")
        return None

    H_arr = np.array([r['H'] for r in good], dtype=float)
    L_arr = np.array([r['L'] for r in good], dtype=float)
    s0_arr = np.array([r['sigmoid_s_0'] for r in good])
    beta_arr = np.array([r['sigmoid_beta'] for r in good])
    g_arr = np.array([r['sigmoid_g_eff'] for r in good])

    print(f"\n{'=' * 70}")
    print(f"  SCALING LAW ANALYSIS ({len(good)} good fits, R2_adj > {min_r2})")
    print(f"{'=' * 70}")

    out = {}
    out['s0'] = _fit_power_law_2d(
        H_arr, L_arr, s0_arr, [0.1, -0.5, -0.5],
        ([0, -3, -3], [10, 3, 3]), 's_0')
    out['beta'] = _fit_power_law_2d(
        H_arr, L_arr, beta_arr, [10.0, 0.0, 0.0],
        ([0, -3, -3], [1000, 3, 3]), 'beta')
    out['g_eff'] = _fit_power_law_2d(
        H_arr, L_arr, g_arr, [0.5, 0.0, 0.0],
        ([0, -3, -3], [2, 3, 3]), 'g_eff')

    print(f"{'=' * 70}")
    return {k: v for k, v in out.items() if v is not None}


# --- Shared experiment runner -------------------------------------------------

DEFAULT_DENSITIES = [0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
                     0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def _build_masks(method, model, densities, n_seeds, base_seed, X_calib):
    from .methods import random_masks, magnitude_masks, wanda_masks
    if method == 'random':
        return random_masks(model, densities, n_seeds=n_seeds, base_seed=base_seed)
    if method == 'magnitude':
        return magnitude_masks(model, densities, n_seeds=1, base_seed=base_seed)
    if method == 'wanda':
        return wanda_masks(model, densities, X_calib, n_seeds=1, base_seed=base_seed)
    raise ValueError(f"unknown method: {method}")


def run_scaling_experiment(
    data,                       # (X_tr, X_val, X_te, y_tr, y_val, y_te)
    *,
    input_size,
    h_values,
    l_values,
    method,
    output_dir,
    dataset_label,              # human-readable for plot titles
    densities=None,
    epochs_fn=None,             # (H, L) -> epochs; default 300
    bs=64,
    lr=1e-3,
    n_seeds=3,                  # only used for random
    seed=42,
    val_acc_floor=0.15,
    device=None,
):
    """Train (H, L) grid, apply ``method`` masks, fit sigmoid + scaling laws, plot.

    Writes ``scaling_results.json``, ``scaling_laws.json``, ``scaling_curves.png``
    and ``s0_scaling.png`` into ``output_dir``.  Returns ``(results, scaling)``.
    """
    # Local import to avoid circularity at module load time
    from pruning.pruning import FCNetwork, fit_sigmoid

    if densities is None:
        densities = DEFAULT_DENSITIES
    if epochs_fn is None:
        epochs_fn = lambda H, L: 300
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)
    X_tr, X_val, X_te, y_tr, y_val, y_te = data

    results = []
    total = len(h_values) * len(l_values)
    count = 0
    t_total = time.time()

    for H in h_values:
        for L in l_values:
            count += 1
            t0 = time.time()
            print(f"\n  [{count}/{total}] H={H}, L={L}", end='', flush=True)

            model = FCNetwork(input_size=input_size, hidden_size=H,
                              num_hidden_layers=L, num_classes=10, seed=seed)
            model = model.to(device)
            val_acc = model.train_model(
                X_tr, y_tr, X_val, y_val,
                epochs=epochs_fn(H, L), bs=bs, lr=lr, verbose=False)
            print(f"  val={100 * val_acc:.1f}%", end='', flush=True)
            if val_acc < val_acc_floor:
                print("  SKIP")
                continue

            # Masks are built & evaluated on CPU to keep memory predictable
            model_cpu = model.to('cpu')
            mask_sets = _build_masks(method, model_cpu, densities,
                                     n_seeds=n_seeds, base_seed=seed,
                                     X_calib=X_tr)
            accs_stats, normal_acc = evaluate_masked_accuracy(
                model_cpu, X_te, y_te, mask_sets)

            s_values = sorted(accs_stats.keys())
            accs_mean = {s: accs_stats[s][0] for s in s_values}
            popt, perr, r2 = fit_sigmoid(s_values, accs_mean, normal_acc)

            res = {
                'H': int(H), 'L': int(L),
                'val_acc': float(val_acc),
                'normal_acc': float(normal_acc),
                'n_params': sum(p.numel() for p in model.parameters()),
                'densities': [float(s) for s in s_values],
                'accs_mean': [float(accs_stats[s][0]) for s in s_values],
                'accs_std':  [float(accs_stats[s][1]) for s in s_values],
            }
            if popt is not None:
                A_inf, A_0, s_0, beta = popt
                res.update({
                    'sigmoid_A_inf': float(A_inf),
                    'sigmoid_A_0':   float(A_0),
                    'sigmoid_s_0':   float(s_0),
                    'sigmoid_beta':  float(beta),
                    'sigmoid_g_eff': float(np.exp(-beta)),
                    'sigmoid_R2':    float(r2),
                })
                print(f"  s0={s_0:.3f} beta={beta:.2f} R2={r2:.3f}"
                      f"  [{time.time() - t0:.0f}s]")
            else:
                res['sigmoid_R2'] = None
                print(f"  sigmoid FAILED  [{time.time() - t0:.0f}s]")
            results.append(res)

    # Save
    with open(os.path.join(output_dir, 'scaling_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_dir}/scaling_results.json")

    scaling = fit_scaling_laws(results)
    if scaling:
        with open(os.path.join(output_dir, 'scaling_laws.json'), 'w') as f:
            json.dump(scaling, f, indent=2)
        print(f"  Saved: {output_dir}/scaling_laws.json")

    print("\n  Generating plots...")
    make_plots(results, scaling, output_dir,
               title_prefix=f"{dataset_label} — {method}")

    print(f"\n  Total runtime: {time.time() - t_total:.0f}s")
    return results, scaling


# --- Plotting -----------------------------------------------------------------

def make_plots(results, scaling, output_dir, title_prefix=''):
    """Two figures: recovery curves (top panels per L) + s_0 scaling (dot+line | heatmap)."""
    from pruning.pruning import sigmoid_fn  # lazy import

    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 3:
        print("  Not enough good fits for plots")
        return []

    unique_L = sorted(set(r['L'] for r in good))
    unique_H = sorted(set(r['H'] for r in good))
    L_col = dict(zip(unique_L,
                     plt.cm.viridis(np.linspace(0.15, 0.85, max(len(unique_L), 1)))))
    H_col = dict(zip(unique_H,
                     plt.cm.plasma(np.linspace(0.15, 0.85, max(len(unique_H), 1)))))

    # Figure 1: recovery curves per L + s_0 vs H ---------------------------
    n_panels = min(len(unique_L), 5)
    fig = plt.figure(figsize=(6 * n_panels, 11))
    top_axes = [fig.add_subplot(2, n_panels, i + 1) for i in range(n_panels)]
    ax_s0 = fig.add_subplot(2, 1, 2)
    fig.suptitle(f'{title_prefix} — Recovery & $s_0$ scaling\n'
                 r'$A(s) = A_0 + (A_\infty - A_0)/(1 + e^{-\beta(s-s_0)})$',
                 fontsize=13, y=1.02)

    for idx, L_val in enumerate(unique_L[:n_panels]):
        ax = top_axes[idx]
        for r in sorted([r for r in good if r['L'] == L_val],
                        key=lambda x: x['H']):
            s_vals = np.array(r['densities'])
            a_mean = np.array(r['accs_mean']) * 100
            a_std = np.array(r['accs_std']) * 100
            ax.errorbar(s_vals, a_mean, yerr=a_std, fmt='o', ms=4,
                        color=H_col[r['H']], alpha=0.7, lw=0.8)
            s_fine = np.linspace(min(s_vals), max(s_vals), 300)
            fit_line = sigmoid_fn(s_fine, r['sigmoid_A_inf'], r['sigmoid_A_0'],
                                  r['sigmoid_s_0'], r['sigmoid_beta']) * 100
            ax.plot(s_fine, fit_line, color=H_col[r['H']], lw=1.5,
                    label=f'H={r["H"]}  $s_0$={r["sigmoid_s_0"]:.3f}')
            ax.axhline(r['normal_acc'] * 100, color=H_col[r['H']],
                       ls=':', lw=0.5, alpha=0.4)
        ax.set_title(f'L = {L_val} layers')
        ax.set_xlabel('Density $s$ (weights kept)')
        ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(alpha=0.3)

    for L_val in unique_L:
        sub = sorted([r for r in good if r['L'] == L_val], key=lambda x: x['H'])
        if sub:
            ax_s0.plot([r['H'] for r in sub],
                       [r['sigmoid_s_0'] for r in sub],
                       'o-', color=L_col[L_val], lw=1.8, ms=8,
                       markeredgecolor='black', markeredgewidth=0.5,
                       label=f'L={L_val}', zorder=5)
    if scaling and 's0' in scaling:
        sr = scaling['s0']
        H_fine = np.linspace(min(unique_H), max(unique_H), 200)
        for L_val in unique_L:
            ax_s0.plot(H_fine,
                       sr['a'] * H_fine ** sr['alpha'] * L_val ** sr['gamma'],
                       '--', color=L_col[L_val], alpha=0.45, lw=1.2)
        formula = (f"$s_0 = {sr['a']:.4f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$   $R^2={sr['R2']:.3f}$")
        ax_s0.text(0.05, 0.95, formula, transform=ax_s0.transAxes, fontsize=9,
                   va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax_s0.set_xlabel('H (hidden size)')
    ax_s0.set_ylabel('$s_0$ (inflection density)')
    ax_s0.set_title('$s_0$ vs Width H')
    ax_s0.legend(fontsize=8)
    ax_s0.grid(alpha=0.3)

    plt.tight_layout()
    p1 = os.path.join(output_dir, 'scaling_curves.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p1}")

    # Figure 2: s_0 dot+line | heatmap -------------------------------------
    fig, (ax_dot, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title_prefix} — $s_0$ scaling law', fontsize=13, y=1.02)

    for L_val in unique_L:
        sub = sorted([r for r in good if r['L'] == L_val], key=lambda x: x['H'])
        if sub:
            Hs = np.array([r['H'] for r in sub], dtype=float)
            s0s = np.array([r['sigmoid_s_0'] for r in sub])
            ax_dot.plot(Hs, s0s, 'o-', color=L_col[L_val], lw=2, ms=9,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=f'L={L_val}', zorder=5)
    if scaling and 's0' in scaling:
        sr = scaling['s0']
        H_fine = np.linspace(min(unique_H), max(unique_H), 200)
        for L_val in unique_L:
            ax_dot.plot(H_fine,
                        sr['a'] * H_fine ** sr['alpha'] * L_val ** sr['gamma'],
                        '--', color=L_col[L_val], alpha=0.45, lw=1.2)
        formula = (f"$s_0 = {sr['a']:.4f}\\,H^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$   $R^2={sr['R2']:.3f}$")
        ax_dot.text(0.05, 0.95, formula, transform=ax_dot.transAxes, fontsize=9,
                    va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    ax_dot.set_xlabel('H (hidden size)')
    ax_dot.set_ylabel('$s_0$')
    ax_dot.set_title('$s_0$ vs Width (lines per L)')
    ax_dot.legend(fontsize=8)
    ax_dot.grid(alpha=0.3)

    H_grid = sorted(set(r['H'] for r in good))
    L_grid = sorted(set(r['L'] for r in good))
    data = np.full((len(L_grid), len(H_grid)), np.nan)
    for r in good:
        data[L_grid.index(r['L']), H_grid.index(r['H'])] = r['sigmoid_s_0']
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('lightgray')
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax_heat.imshow(masked, aspect='auto', cmap=cmap, origin='lower')
    ax_heat.set_xticks(range(len(H_grid))); ax_heat.set_xticklabels(H_grid)
    ax_heat.set_yticks(range(len(L_grid))); ax_heat.set_yticklabels(L_grid)
    ax_heat.set_xlabel('H (hidden size)')
    ax_heat.set_ylabel('L (layers)')
    ax_heat.set_title('$s_0$ heatmap over $(H, L)$')
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label='$s_0$')
    for i in range(len(L_grid)):
        for j in range(len(H_grid)):
            if not np.isnan(data[i, j]):
                v = data[i, j]
                c = 'white' if v > np.nanmean(data) * 1.3 else 'black'
                ax_heat.text(j, i, f'{v:.3f}', ha='center', va='center',
                             fontsize=8, fontweight='bold', color=c)

    plt.tight_layout()
    p2 = os.path.join(output_dir, 's0_scaling.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p2}")

    return [p1, p2]
