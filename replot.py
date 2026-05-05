#!/usr/bin/env python3
"""Regenerate all figures from existing JSON result files."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pruning'))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fix_accs(results):
    """JSON stores accs keys as strings; convert back to int."""
    for r in results:
        if 'accs' in r:
            r['accs'] = {int(k): v for k, v in r['accs'].items()}
    return results


# --- sklearn digits (mnist_figures) ---
print("\n=== sklearn digits ===")
from mnist_scaling import make_scaling_plots  # noqa: E402
results = fix_accs(load_json('mnist_figures/scaling_results.json'))
scaling = load_json('mnist_figures/scaling_laws.json')
make_scaling_plots(results, scaling, output_dir='assets/mnist_figures')

# --- MNIST 28x28 (mnist28_figures) ---
print("\n=== MNIST 28×28 ===")
from mnist28_scaling import make_plots as make_plots_28  # noqa: E402
results28 = fix_accs(load_json('assets/mnist28_figures/mnist28_scaling_results.json'))
scaling28 = load_json('assets/mnist28_figures/mnist28_scaling_laws.json')
make_plots_28(results28, scaling28, 'assets/mnist28_figures')

# --- CIFAR-10 (cifar_figures) ---
print("\n=== CIFAR-10 ===")
from cifar_scaling import make_plots as make_plots_cifar  # noqa: E402
results_c = fix_accs(load_json('assets/cifar_figures/cifar_scaling_results.json'))
scaling_c = load_json('assets/cifar_figures/cifar_scaling_laws.json')
make_plots_cifar(results_c, scaling_c, 'assets/cifar_figures')

# --- Pythia (pythia_figures) ---
if os.path.exists('pythia_figures/pythia_results.json'):
    print("\n=== Pythia ===")
    from pythia_scaling import make_sigmoid_curves_plot, make_k0_scaling_plot  # noqa: E402
    p_results = load_json('pythia_figures/pythia_results.json')
    p_scaling = load_json('pythia_figures/pythia_scaling_laws.json')
    os.makedirs('assets/pythia_figures', exist_ok=True)
    make_sigmoid_curves_plot(p_results, 'assets/pythia_figures')
    make_k0_scaling_plot(p_results, p_scaling, 'assets/pythia_figures')

print("\nDone.")
