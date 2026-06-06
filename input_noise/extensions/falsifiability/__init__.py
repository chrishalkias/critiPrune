"""Falsifiability re-analysis of the input-noise iso-accuracy claim.

Three analyses, all driven by ``analysis.py``:

  A1. Prior-prediction reframe — fit ``sigma^2(1)_prior`` from the s=1 column
      of the joint grid alone, then predict the full contour and compare to
      the measured iso-A = 0.5 contour. Removes self-referentiality of the
      original procedure.

  A2. Signed mean residual ``<eta - (1 - xi)>`` at L = 2, cell-by-cell and
      cell-count-weighted, including a middle-range restriction
      ``xi in [0.3, 0.7]`` that excludes the geometrically-constrained
      endpoints.

  A3. Baseline-monotone null control — replace the measured contour with
      surrogates ``sigma^2_null(s) = sigma^2(1) * s^k`` for ``k in
      {0.5, 1, 2}`` and recompute the same residual.

Reads only ``input_noise/results_cluster_all.json`` and the existing
helper module ``input_noise.extensions._analysis``; writes
``results.json``, three PNGs, and ``REPORT.md``.
"""
