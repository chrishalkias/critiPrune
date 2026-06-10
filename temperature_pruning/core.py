"""(temperature, density) sweep on already-trained FCNetwork checkpoints.

For each (H, L, repeat) cell on disk the runner:
    1. Loads the checkpoint via ``unstructured_pruning.core.load_fc_checkpoint``.
    2. For each sigma in the temperature grid:
         For each noise seed:
             - Builds a noisy copy via ``temperature_pruning.noise.add_weight_noise``.
             - Builds Bernoulli random masks at every density.
             - Evaluates test accuracy per (density, mask seed).
       Aggregates mean / std over the (n_noise_seeds * n_mask_seeds) accuracies.
    3. Fits a logistic A(s) sigmoid per (H, L, repeat, sigma) and stores
       ``s_0(sigma) === p_c(sigma)`` for the downstream linear-in-sigma test.

Output: ``output_dir/results.json`` -- one row per (H, L, repeat, sigma).
The schema mirrors ``unstructured_pruning/scaling_results.json`` with one
extra ``sigma`` column so existing plotting helpers can be re-used.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from unstructured_pruning.core import (
    DEFAULT_DENSITIES,
    evaluate_masked_accuracy,
    load_fc_checkpoint,
)
from unstructured_pruning.methods import random_masks
from unstructured_pruning.base.pruning import fit_sigmoid

from .noise import add_weight_noise, layer_weight_rms


DEFAULT_SIGMAS = np.linspace(0.0, 1.0, 100).tolist()


def _aggregate_accs(per_run_accs):
    """Aggregate {s: [acc, acc, ...]} -> {s: (mean, std)}."""
    return {s: (float(np.mean(v)), float(np.std(v))) for s, v in per_run_accs.items()}


def _single_trial(model, sigma, X_te, y_te, densities,
                  n_mask_seeds, n_noise_seeds, trial_seed):
    """One bundle of (n_noise_seeds * n_mask_seeds) reps for a single sigma.

    Returns the per-density mean accuracy curve (across the inner bundle)
    and the resulting sigmoid fit. This is the unit treated as one
    "trial" when computing error bars across many such bundles.
    """
    per_run = {float(s): [] for s in densities}
    normal_accs = []

    for k in range(n_noise_seeds):
        rng = np.random.default_rng(trial_seed + 7919 * k)
        noisy = add_weight_noise(model, sigma, rng)
        mask_sets = random_masks(noisy, densities,
                                 n_seeds=n_mask_seeds,
                                 base_seed=trial_seed + 1009 * k)
        accs_stats, normal_acc = evaluate_masked_accuracy(
            noisy, X_te, y_te, mask_sets)
        normal_accs.append(normal_acc)
        for s, (m, _) in accs_stats.items():
            per_run[float(s)].append(m)

    s_values = sorted(per_run)
    accs_mean_curve = {s: float(np.mean(per_run[s])) for s in s_values}
    na_mean = float(np.mean(normal_accs))
    popt, _perr, r2 = fit_sigmoid(s_values, accs_mean_curve, na_mean)
    return {
        'accs_mean': accs_mean_curve,
        'normal_acc': na_mean,
        'popt': None if popt is None else [float(x) for x in popt],
        'R2': None if r2 is None else float(r2),
    }


def _evaluate_one_sigma(model, sigma, X_te, y_te, densities,
                       n_mask_seeds, n_noise_seeds, base_seed,
                       n_trials=1):
    """Run ``n_trials`` independent bundles for one sigma.

    Each trial is a self-contained (noise, mask) experiment with its own
    seed, sigmoid fit and per-density accuracy curve. With ``n_trials=1``
    the behaviour is equivalent to the original single-bundle evaluation.

    Returns
    -------
    accs : dict {s: (mean, std)}
        Per-density mean and std *across trials* (the std is what shows
        up as the inner uncertainty of the accuracy curve).
    na_mean, na_std : float
        Unpruned accuracy of the noisy model, averaged across trials.
    s_0_trials : list of float
        Per-trial sigmoid inflection (NaN for trials whose fit failed).
    R2_trials : list of float
        Per-trial sigmoid R^2.
    popt_grand : list of float or None
        ``[A_inf, A_0, s_0, beta]`` from fitting the trial-pooled mean
        accuracy curve (used as the "display" sigmoid for plotting).
    R2_grand : float or None
    """
    trials = []
    for t in range(n_trials):
        trial_seed = base_seed + 1_000_003 * t
        trials.append(_single_trial(
            model, sigma, X_te, y_te, densities,
            n_mask_seeds, n_noise_seeds, trial_seed))

    # Per-density distribution across trials.
    pool = {float(s): [] for s in densities}
    nas = []
    for tr in trials:
        nas.append(tr['normal_acc'])
        for s, m in tr['accs_mean'].items():
            pool[float(s)].append(m)
    accs = {s: (float(np.mean(v)), float(np.std(v))) for s, v in pool.items()}
    na_mean = float(np.mean(nas))
    na_std = float(np.std(nas))

    s_0_trials = []
    R2_trials = []
    for tr in trials:
        if tr['popt'] is None:
            s_0_trials.append(float('nan'))
            R2_trials.append(float('nan'))
        else:
            s_0_trials.append(float(tr['popt'][2]))
            R2_trials.append(float(tr['R2']) if tr['R2'] is not None
                             else float('nan'))

    # "Display" sigmoid: refit on the trial-pooled mean accuracy curve.
    s_values = sorted(pool)
    grand_mean_curve = {s: float(np.mean(pool[s])) for s in s_values}
    popt_grand, _perr, R2_grand = fit_sigmoid(
        s_values, grand_mean_curve, na_mean)
    popt_grand_list = (None if popt_grand is None
                       else [float(x) for x in popt_grand])
    R2_grand = None if R2_grand is None else float(R2_grand)

    return accs, na_mean, na_std, s_0_trials, R2_trials, popt_grand_list, R2_grand


def run_temperature_pruning_experiment(
    data,
    *,
    h_values,
    l_values,
    sigmas=None,
    densities=None,
    ckpt_dir,
    output_dir,
    repeat_ids=(0,),
    n_mask_seeds=3,
    n_noise_seeds=3,
    n_trials=1,
    seed=42,
    noise_scale='rms',
):
    """Run the (sigma, density) sweep and persist results.

    Parameters
    ----------
    data : tuple
        ``(X_tr, X_val, X_te, y_tr, y_val, y_te)`` -- only the test split is used.
    h_values, l_values : iterable of int
        Architecture cells to evaluate. Checkpoints must exist on disk under
        ``ckpt_dir/H{H}_L{L}_r{r}.pt`` for every chosen ``r in repeat_ids``.
    sigmas : list of float, optional
        Temperature grid. Defaults to ``DEFAULT_SIGMAS``.
    densities : list of float, optional
        Pruning density grid. Defaults to ``unstructured_pruning.core.DEFAULT_DENSITIES``.
    ckpt_dir : str
        Directory of saved checkpoints (e.g. the existing
        ``checkpoints/sklearn_random``).
    output_dir : str
        Where to write ``results.json`` (and later figures).
    repeat_ids : tuple of int
        Which checkpoint repeats (``r0``, ``r1``, ...) to load.
    n_mask_seeds, n_noise_seeds : int
    seed : int
        Master RNG anchor; per-cell seeds are derived deterministically.
    noise_scale : {'rms', 'absolute'}
        Forwarded to ``add_weight_noise``.

    Returns
    -------
    list of dict -- one row per (H, L, repeat, sigma).
    """
    if sigmas is None:
        sigmas = list(DEFAULT_SIGMAS)
    if densities is None:
        densities = list(DEFAULT_DENSITIES)

    _, _, X_te, _, _, y_te = data

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'results.json')

    results = []
    finished = set()  # (H, L, repeat, sigma_key) tuples already on disk
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        for r in results:
            finished.add((int(r['H']), int(r['L']), int(r['repeat']),
                          round(float(r['sigma']), 6)))
        print(f"  Resume: loaded {len(results)} existing rows")

    total = len(h_values) * len(l_values) * len(repeat_ids) * len(sigmas)
    count = 0
    t_total = time.time()

    for H in h_values:
        for L in l_values:
            for rep in repeat_ids:
                ckpt_path = os.path.join(ckpt_dir, f'H{H}_L{L}_r{rep}.pt')
                if not os.path.exists(ckpt_path):
                    print(f"  [skip] missing checkpoint: {ckpt_path}")
                    count += len(sigmas)
                    continue

                model, ckpt = load_fc_checkpoint(ckpt_path)
                rms = layer_weight_rms(model)
                base_seed = int(seed + 1000 * rep + 31 * H + 17 * L)

                for sigma in sigmas:
                    count += 1
                    sigma_key = round(float(sigma), 6)
                    if (int(H), int(L), int(rep), sigma_key) in finished:
                        print(f"  [{count}/{total}] H={H} L={L} r={rep} "
                              f"sigma={sigma:.3f}  done -- skip")
                        continue

                    t0 = time.time()
                    (accs_stats, na_mean, na_std,
                     s_0_trials, R2_trials,
                     popt_grand, R2_grand) = _evaluate_one_sigma(
                        model, sigma, X_te, y_te, densities,
                        n_mask_seeds=n_mask_seeds,
                        n_noise_seeds=n_noise_seeds,
                        base_seed=base_seed,
                        n_trials=n_trials,
                    )

                    s_values = sorted(accs_stats.keys())

                    # Per-trial summary statistics (NaN-safe).
                    s_0_arr = np.array([x for x in s_0_trials
                                        if x == x])  # drop NaNs
                    s_0_mean = (float(np.mean(s_0_arr))
                                if s_0_arr.size else None)
                    s_0_std = (float(np.std(s_0_arr))
                               if s_0_arr.size > 1 else 0.0)
                    R2_arr = np.array([x for x in R2_trials if x == x])
                    R2_mean = (float(np.mean(R2_arr))
                               if R2_arr.size else None)

                    row = {
                        'H': int(H), 'L': int(L),
                        'repeat': int(rep),
                        'sigma': float(sigma),
                        'noise_scale': noise_scale,
                        'val_acc': float(ckpt.get('val_acc', float('nan'))),
                        'normal_acc': na_mean,
                        'normal_acc_std': na_std,
                        'layer_rms': rms,
                        'densities': [float(s) for s in s_values],
                        'accs_mean': [float(accs_stats[s][0]) for s in s_values],
                        'accs_std':  [float(accs_stats[s][1]) for s in s_values],
                        'n_mask_seeds': int(n_mask_seeds),
                        'n_noise_seeds': int(n_noise_seeds),
                        'n_trials': int(n_trials),
                        'sigmoid_s_0_trials': s_0_trials,
                        'sigmoid_s_0_std': s_0_std,
                        'sigmoid_R2_trials': R2_trials,
                    }
                    # The "display" sigmoid is the fit on the trial-pooled
                    # mean accuracy curve. Headline s_0 reported here is
                    # the per-trial mean (more robust than the display fit's
                    # s_0 because each trial contributes one independent
                    # estimate).
                    if popt_grand is not None and s_0_mean is not None:
                        A_inf, A_0, _s0_grand, beta = popt_grand
                        row.update({
                            'sigmoid_A_inf': float(A_inf),
                            'sigmoid_A_0':   float(A_0),
                            'sigmoid_s_0':   float(s_0_mean),
                            'sigmoid_beta':  float(beta),
                            'sigmoid_R2':    float(R2_grand) if R2_grand is not None
                                             else (R2_mean or float('nan')),
                        })
                        print(f"  [{count}/{total}] H={H} L={L} r={rep} "
                              f"sigma={sigma:.3f}  "
                              f"s0={s_0_mean:.3f}+/-{s_0_std:.3f} "
                              f"R2={R2_grand or R2_mean or float('nan'):.3f}  "
                              f"na={na_mean:.3f}  "
                              f"[{time.time() - t0:.1f}s, {n_trials} trial(s)]")
                    else:
                        row['sigmoid_R2'] = None
                        print(f"  [{count}/{total}] H={H} L={L} r={rep} "
                              f"sigma={sigma:.3f}  sigmoid FAILED  "
                              f"[{time.time() - t0:.1f}s]")

                    results.append(row)
                    finished.add((int(H), int(L), int(rep), sigma_key))

                # Persist after each cell so a kill loses at most one cell.
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)

    print(f"\n  Saved: {results_path}  ({len(results)} rows)")
    print(f"  Total runtime: {time.time() - t_total:.0f}s")
    return results
