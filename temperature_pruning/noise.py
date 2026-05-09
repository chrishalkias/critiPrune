"""Temperature knob: additive Gaussian noise on hidden-layer weights.

The diluted Curie-Weiss bond-disorder extension (IsingPruning.pdf, sec. 1.8.2)
treats Gaussian-disordered couplings as the natural finite-temperature analogue
of the uniform ferromagnet. Adding ``N(0, sigma^2)`` to each retained weight
provides a single scalar knob ``sigma`` that maps onto ``J_1`` in the paper's
notation; the trained weights stay as the ``J_0`` mean field.

The noise std is scaled per-layer by the layer's weight RMS so that the same
``sigma`` corresponds to the same fractional perturbation across the (H, L)
architecture grid (where weight magnitudes differ with fan-in).
"""

from __future__ import annotations

import copy

import numpy as np
import torch


def add_weight_noise(model, sigma, rng, scale='rms'):
    """Return a deep copy of ``model`` with Gaussian noise added to hidden weights.

    Only hidden layers (``model.layers[:-1]``) are perturbed, matching the
    convention used by ``unstructured_pruning.methods.random_masks``: the
    read-out layer is left untouched so accuracy is well-defined and the
    temperature knob acts on the same parameters as the pruning mask.

    Parameters
    ----------
    model : FCNetwork
        Loaded checkpoint. Not mutated.
    sigma : float
        Noise standard deviation. ``sigma <= 0`` returns a clean copy.
    rng : numpy.random.Generator
        Source of randomness for reproducibility across (cell, sigma, repeat).
    scale : {'rms', 'absolute'}
        ``'rms'`` (default): per-layer std = ``sigma * rms(W_layer)`` so
        ``sigma`` is a fractional perturbation.
        ``'absolute'``: per-layer std = ``sigma`` (unscaled).

    Returns
    -------
    FCNetwork
        Independent copy with noisy hidden-layer weights. Biases and the
        read-out layer are unchanged.
    """
    noisy = copy.deepcopy(model)
    if sigma is None or sigma <= 0:
        return noisy

    with torch.no_grad():
        for layer in noisy.layers[:-1]:
            W = layer.weight
            if scale == 'rms':
                rms = float(W.detach().pow(2).mean().sqrt().item())
                std = float(sigma) * rms
            elif scale == 'absolute':
                std = float(sigma)
            else:
                raise ValueError(f"unknown scale mode: {scale!r}")
            if std <= 0:
                continue
            noise_np = rng.normal(loc=0.0, scale=std, size=tuple(W.shape))
            noise = torch.from_numpy(noise_np).to(dtype=W.dtype, device=W.device)
            W.add_(noise)
    return noisy


def layer_weight_rms(model):
    """Return list of per-hidden-layer weight RMS values (for diagnostics)."""
    out = []
    for layer in model.layers[:-1]:
        W = layer.weight.detach()
        out.append(float(W.pow(2).mean().sqrt().item()))
    return out
