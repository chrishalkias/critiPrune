#!/usr/bin/env python3
"""Regenerate ``r2_vs_HL.png`` and ``sigma2_1_scaling.png`` from the
already-fitted ``per_cell_fits.json`` (no re-run of the cluster sweep).

The two plotting functions in ``cluster_analyze.py`` were updated to
drop the ``cifar_pca`` panel (already excluded upstream from analysis)
and lay the remaining three datasets out 1x3. This driver re-invokes
just those two functions and overwrites the two PNGs in place.

Run::

    .venv/bin/python input_noise/figures_cluster/_regen_no_cifar_pca.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_INPUT_NOISE = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_INPUT_NOISE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from input_noise.cluster_analyze import (
    _style,
    plot_r2_vs_HL,
    plot_sigma2_1_scaling,
)


def main() -> None:
    have_latex = _style()
    records_path = os.path.join(_HERE, 'per_cell_fits.json')
    with open(records_path) as f:
        records = json.load(f)
    print(f'  loaded {len(records)} per-cell records from {records_path}')

    r2_path = os.path.join(_HERE, 'r2_vs_HL.png')
    plot_r2_vs_HL(records, r2_path, have_latex)
    print(f'  wrote {r2_path}')

    sig_path = os.path.join(_HERE, 'sigma2_1_scaling.png')
    plot_sigma2_1_scaling(records, sig_path, have_latex)
    print(f'  wrote {sig_path}')


if __name__ == '__main__':
    main()
