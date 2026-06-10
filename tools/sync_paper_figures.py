#!/usr/bin/env python3
"""Refresh the manuscript's figure copies from the generated assets/.

The paper figures under ``.docs/new_paper/manuscript/figures/`` used to be
symlinks into the experiment output dirs. Those outputs now live under
``assets/`` (which is gitignored / local-only), and the manuscript copies are
real committed-on-disk PNGs so the paper compiles self-contained. After
regenerating a figure in ``assets/``, run this to copy the current versions
into the manuscript dir.

    python tools/sync_paper_figures.py            # copy all mapped figures
    python tools/sync_paper_figures.py --check    # report stale/missing only
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, '.docs/new_paper/manuscript/figures')

# manuscript filename -> source under assets/
MAPPING = {
    'input_noise_collapse_all.png':        'assets/input_noise/cluster/collapse_all.png',
    'input_noise_collapse_cluster.png':    'assets/input_noise/cluster/collapse_all.png',
    'input_noise_collapse_by_method.png':  'assets/input_noise/cluster/collapse_by_method.png',
    'input_noise_r2_vs_HL.png':            'assets/input_noise/cluster/r2_vs_HL.png',
    'input_noise_sigma2_1_scaling.png':    'assets/input_noise/cluster/sigma2_1_scaling.png',
    'collapse_multi_iso.png':              'assets/input_noise/extensions/collapse_multi_iso.png',
    'depth_r2.png':                        'assets/input_noise/extensions/depth_r2.png',
    'falsifiability_prior_prediction.png': 'assets/input_noise/extensions/falsifiability/prior_prediction.png',
    'falsifiability_null_control.png':     'assets/input_noise/extensions/falsifiability/null_control.png',
    'falsifiability_signed_residual.png':  'assets/input_noise/extensions/falsifiability/signed_residual_L2.png',
    'beta_vs_H_FSS.png':                   'assets/temperature_pruning/fss_check/beta_vs_H.png',
    's0_manifold_mnist.png':               'assets/unstructured_pruning/mnist28_wanda/s0_3d_v2.png',
    's0_manifold_digits.png':              'assets/unstructured_pruning/sklearn_wanda/s0_3d_v2.png',
    'sigmoid_overlay.png':                 'assets/unstructured_pruning/sigmoid_overlay.png',
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true',
                    help='only report stale/missing, do not copy')
    args = ap.parse_args()

    copied = stale = missing = 0
    for name, rel in sorted(MAPPING.items()):
        src = os.path.join(ROOT, rel)
        dst = os.path.join(FIG, name)
        if not os.path.exists(src):
            print(f'  MISSING SOURCE  {rel}  (regenerate it first)')
            missing += 1
            continue
        same = os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False)
        if same:
            continue
        stale += 1
        if args.check:
            print(f'  STALE  {name}  <-  {rel}')
        else:
            shutil.copyfile(src, dst)
            print(f'  copied {name}  <-  {rel}')
            copied += 1

    if args.check:
        print(f'\n  {stale} stale, {missing} missing source(s).')
    else:
        print(f'\n  {copied} copied, {missing} missing source(s).')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
