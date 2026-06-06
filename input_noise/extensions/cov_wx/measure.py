#!/usr/bin/env python3
"""Probe the Cov(W, x) hypothesis from docs/input_noise.md S6.2(b).

For each of the 5 pilot cells, compute a single-number summary of how far
the trained first-layer weights W^(1) deviate from the factorisation
assumption <W^2 x^2> = <W^2><x^2> made in S3 of
pruning_sigmoid_derivation.md.  Then correlate that with the empirical
1-D-noise-sweep discrepancy dR^2 = R^2(sigma) - R^2(sigma^2).

The hypothesis predicts: cells with larger Cov(W, x) should show larger
dR^2 (more departure from the pure-second-cumulant prediction that
SNR depends on sigma^2, not sigma).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pruning.mnist_scaling import load_data as load_digits      # noqa: E402
from pruning.mnist28_scaling import load_mnist28                # noqa: E402
from pruning.pruning import FCNetwork                           # noqa: E402

CELLS = [
    ('digits',  64,  2),
    ('digits', 128,  2),
    ('digits', 128,  4),
    ('mnist28', 128, 2),
    ('mnist28', 256, 2),
]
CKPT_DIR    = os.path.join(_ROOT, 'input_noise', 'checkpoints')
RESULTS_JSON = os.path.join(_ROOT, 'input_noise', 'results.json')
EXT_JSON     = os.path.join(_ROOT, 'input_noise', 'extensions',
                            'depth_cells', 'results.json')
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

LOADERS = {'digits': load_digits, 'mnist28': load_mnist28}


def load_cell(dataset, H, L, X_te_cache):
    """Return (W1, X_te) for one cell.  Re-uses X_te per dataset."""
    if dataset not in X_te_cache:
        _, _, X_te, _, _, _ = LOADERS[dataset]()
        X_te_cache[dataset] = np.asarray(X_te, dtype=np.float64)
    X_te = X_te_cache[dataset]
    D = X_te.shape[1]
    model = FCNetwork(input_size=D, hidden_size=H, num_hidden_layers=L,
                      num_classes=10, seed=42)
    path = os.path.join(CKPT_DIR, f'{dataset}_H{H}_L{L}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f'checkpoint missing: {path} '
                                '(re-run input_noise/run_experiment.py first)')
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['state_dict'])
    W1 = model.layers[0].weight.detach().cpu().numpy().astype(np.float64)
    return W1, X_te


def cov_statistic(W1, X_te):
    """Factorisation-error statistic for Cov(W^2, x^2).

    The S3 partition-function derivation assumes
        <W_{hj}^2 x_j^2>_{h,j} = <W^2><x^2>.
    Return the relative deviation
        delta_factor = <W^2 x^2> / (<W^2><x^2>) - 1
    where averages run over both hidden index h and input index j.
    The hypothesis Cov(W, x) != 0 predicts |delta_factor| > 0.
    """
    H, D = W1.shape
    x2_j = (X_te ** 2).mean(axis=0)          # [D]
    w2 = W1 ** 2                              # [H, D]
    Ew2x2 = float((w2 * x2_j[None, :]).mean())
    Ew2   = float(w2.mean())
    Ex2   = float(x2_j.mean())
    return Ew2x2 / (Ew2 * Ex2) - 1.0


def load_extension_cells():
    """Optional: subagent S1 may have written extra cells to depth_cells/."""
    if not os.path.exists(EXT_JSON):
        return []
    with open(EXT_JSON) as f:
        return json.load(f)


def main():
    # 1. baseline 5 cells from assets/input_noise/pilot/results.json
    with open(RESULTS_JSON) as f:
        baseline = json.load(f)
    extras = load_extension_cells()

    X_te_cache = {}
    rows = []
    for record in baseline + extras:
        dataset, H, L = record['dataset'], record['H'], record['L']
        # Tolerate extension cells that may not be in CELLS list.
        try:
            W1, X_te = load_cell(dataset, H, L, X_te_cache)
        except FileNotFoundError as e:
            print(f'  skip {dataset} H={H} L={L}: {e}')
            continue
        delta = cov_statistic(W1, X_te)
        r2_s  = record['noise']['fit_in_sigma']['R2']
        r2_s2 = record['noise']['fit_in_sigma2']['R2']
        dR2 = r2_s - r2_s2
        label = f'{dataset} H={H} L={L}'
        rows.append((label, delta, abs(delta), dR2))
        print(f'  {label:24s}  delta_factor={delta:+.4f}  |delta|={abs(delta):.4f}'
              f'  dR2={dR2:+.4f}')

    if len(rows) < 3:
        raise RuntimeError(f'too few cells ({len(rows)}) for any correlation')

    deltas      = np.array([r[1] for r in rows])
    abs_deltas  = np.array([r[2] for r in rows])
    dR2s        = np.array([r[3] for r in rows])

    # Correlations: signed and absolute. The hypothesis predicts a positive
    # relationship between Cov(W, x) magnitude and dR^2.
    rho_s, p_rho_s = spearmanr(deltas, dR2s)
    r_p,   p_p     = pearsonr(deltas, dR2s)
    rho_abs, p_rho_abs = spearmanr(abs_deltas, dR2s)
    r_abs,   p_r_abs   = pearsonr(abs_deltas, dR2s)

    N = len(rows)
    out = {
        'N': N,
        'rows': [
            {'cell': lab, 'delta_factor': d, 'abs_delta_factor': ad,
             'dR2': dr}
            for (lab, d, ad, dr) in rows
        ],
        'signed': {
            'pearson_r': float(r_p),   'pearson_p': float(p_p),
            'spearman_rho': float(rho_s), 'spearman_p': float(p_rho_s),
        },
        'absolute': {
            'pearson_r': float(r_abs), 'pearson_p': float(p_r_abs),
            'spearman_rho': float(rho_abs), 'spearman_p': float(p_rho_abs),
        },
    }
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\n  N = {N}')
    print(f'  signed   delta_factor vs dR^2 : Pearson r={r_p:+.3f} (p={p_p:.3f})'
          f'   Spearman rho={rho_s:+.3f} (p={p_rho_s:.3f})')
    print(f'  absolute |delta|     vs dR^2 : Pearson r={r_abs:+.3f} (p={p_r_abs:.3f})'
          f'   Spearman rho={rho_abs:+.3f} (p={p_rho_abs:.3f})')

    # Optional scatter plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.scatter(abs_deltas, dR2s, s=40, color='steelblue', zorder=3)
        for (lab, _, ad, dr) in rows:
            ax.annotate(lab.replace('mnist28', 'M').replace('digits', 'D'),
                        (ad, dr), fontsize=7, xytext=(4, 4),
                        textcoords='offset points')
        ax.set_xlabel(r'$|\delta_{\rm fact}| = |\langle W^2 x^2\rangle/'
                      r'(\langle W^2\rangle\langle x^2\rangle) - 1|$')
        ax.set_ylabel(r'$\Delta R^2 = R^2(\sigma) - R^2(\sigma^2)$')
        ax.set_title(f'Cov(W, x) hypothesis test  (N={N})\n'
                     fr'Spearman $\rho$ = {rho_abs:+.2f}, '
                     fr'Pearson r = {r_abs:+.2f}')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'results_scatter.png'), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f'  scatter skipped: {e}')


if __name__ == '__main__':
    main()
