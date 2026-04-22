"""Plotting utilities for Pythia family pruning results."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def sigmoid_fn(K, A_inf, A_0, K_0, beta):
    K = np.asarray(K, dtype=float)
    return A_0 + (A_inf - A_0) / (1.0 + np.exp(
        np.clip(-beta * (K - K_0), -500, 500)))


def _fit_sigmoid(k_fracs, recoveries):
    k_arr = np.array(k_fracs)
    r_arr = np.array(recoveries)
    try:
        p0 = [max(r_arr), min(r_arr), np.median(k_arr), 10.0]
        bounds = ([0.0, -0.1, 0.0, 0.1], [1.5, 1.0, 1.0, 200.0])
        popt, _ = curve_fit(sigmoid_fn, k_arr, r_arr, p0=p0,
                            bounds=bounds, maxfev=30000)
        return popt
    except Exception:
        return None


def make_sigmoid_curves_plot(results, output_dir):
    """All 7 Pythia recovery curves + sigmoid fits in one figure."""
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if not good:
        print("  No good fits"); return

    good = sorted(good, key=lambda x: x['n_params_M'])
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(good)))

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle('Pythia Family — WANDA Pruning Recovery Curves', fontsize=13)

    for i, r in enumerate(good):
        kf = np.array(r['k_fracs'])
        rec = np.array(r['recoveries'])
        col = colors[i]

        ax.scatter(kf * 100, rec, s=22, color=col, alpha=0.75, zorder=5)

        popt = _fit_sigmoid(r['k_fracs'], r['recoveries'])
        if popt is not None:
            kf_fine = np.linspace(0.01, 1.0, 400)
            fit = sigmoid_fn(kf_fine, *popt)
            label = (f'{r["model"]}  '
                     f'$K_0$={r["sigmoid_K_0"]*100:.1f}%  '
                     f'$R^2$={r["sigmoid_R2"]:.3f}')
            ax.plot(kf_fine * 100, fit, color=col, lw=2, label=label)
            ax.axvline(r['sigmoid_K_0'] * 100, color=col,
                       lw=0.8, ls=':', alpha=0.5)

    ax.set_xlabel('MLP neurons kept (%)', fontsize=12)
    ax.set_ylabel('Loss recovery  ($\\log$ PPL$_{base}$ / $\\log$ PPL$_{sparse}$)', fontsize=11)
    ax.set_title('Sigmoid phase transition across model scales', fontsize=11)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.85)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pythia_sigmoid_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def make_k0_scaling_plot(results, scaling, output_dir):
    """2-panel K₀ scaling figure matching mnist_figures/k0_scaling.png style."""
    good = [r for r in results
            if r.get('sigmoid_R2') is not None and r['sigmoid_R2'] > 0.80]
    if len(good) < 2:
        print("  Not enough data for K₀ scaling plot"); return

    unique_L = sorted(set(r['L'] for r in good))
    unique_dff = sorted(set(r['d_ff'] for r in good))
    L_cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(unique_L)))
    L_color = {L: L_cmap[i] for i, L in enumerate(unique_L)}

    fig, (ax_dot, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Pythia Family — $K_0$ Scaling Law', fontsize=13, y=1.02)

    # Left: K₀_abs vs d_ff, log-log, lines per L
    for L_val in unique_L:
        sub = sorted([r for r in good if r['L'] == L_val], key=lambda x: x['d_ff'])
        if sub:
            xs = np.array([r['d_ff'] for r in sub], dtype=float)
            ys = np.array([r['sigmoid_K_0_abs'] for r in sub])
            ax_dot.plot(xs, ys, 'o-', color=L_color[L_val], lw=2, ms=9,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=f'L={L_val}', zorder=5)
            for r in sub:
                ax_dot.annotate(r['model'],
                                (r['d_ff'], r['sigmoid_K_0_abs']),
                                fontsize=7, ha='left', va='bottom',
                                xytext=(4, 4), textcoords='offset points')

    if scaling and 'K0_abs' in scaling:
        sr = scaling['K0_abs']
        dff_fine = np.geomspace(min(unique_dff) * 0.9, max(unique_dff) * 1.1, 300)
        for L_val in unique_L:
            ax_dot.plot(dff_fine,
                        sr['a'] * dff_fine ** sr['alpha'] * L_val ** sr['gamma'],
                        '--', color=L_color[L_val], alpha=0.40, lw=1.2)
        formula = (f"$K_0 = {sr['a']:.3f}\\,d_{{ff}}^{{{sr['alpha']:.3f}}}"
                   f"\\,L^{{{sr['gamma']:.3f}}}$  $R^2={sr['R2']:.3f}$")
        ax_dot.text(0.05, 0.95, formula, transform=ax_dot.transAxes, fontsize=9,
                    va='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax_dot.set_xscale('log')
    ax_dot.set_yscale('log')
    ax_dot.set_xlabel('$d_{ff}$ (MLP intermediate dim)', fontsize=11)
    ax_dot.set_ylabel('$K_0$ (neurons)', fontsize=11)
    ax_dot.set_title('$K_0$ vs Width  (lines per L)', fontsize=11)
    ax_dot.legend(fontsize=8)
    ax_dot.grid(alpha=0.3, which='both')

    # Right: K₀_frac (%) heatmap over sparse (d_ff × L) grid
    data = np.full((len(unique_L), len(unique_dff)), np.nan)
    for r in good:
        data[unique_L.index(r['L']), unique_dff.index(r['d_ff'])] = r['sigmoid_K_0'] * 100

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color='lightgray')
    masked = np.ma.array(data, mask=np.isnan(data))
    im = ax_heat.imshow(masked, aspect='auto', cmap=cmap, origin='lower',
                        vmin=80, vmax=100)
    dff_labels = [str(v) for v in unique_dff]
    ax_heat.set_xticks(range(len(unique_dff)))
    ax_heat.set_xticklabels(dff_labels, rotation=30, ha='right')
    ax_heat.set_yticks(range(len(unique_L)))
    ax_heat.set_yticklabels(unique_L)
    ax_heat.set_xlabel('$d_{ff}$ (MLP intermediate dim)', fontsize=11)
    ax_heat.set_ylabel('L (transformer layers)', fontsize=11)
    ax_heat.set_title('$K_0$ Heatmap over $(d_{ff}, L)$ Grid  (%)', fontsize=11)
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label='$K_0$ (%)')

    for i in range(len(unique_L)):
        for j in range(len(unique_dff)):
            if not np.isnan(data[i, j]):
                val = data[i, j]
                c = 'white' if val > 95 else 'black'
                ax_heat.text(j, i, f'{val:.1f}%', ha='center', va='center',
                             fontsize=9, fontweight='bold', color=c)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pythia_k0_scaling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path
