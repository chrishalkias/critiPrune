"""Unstructured pruning strategies.

Each strategy returns a dict ``{s: [mask_set_seed_0, mask_set_seed_1, ...]}``
where each *mask set* is a list of one float32 tensor per hidden layer, with
shape matching ``layer.weight`` and values in {0, 1}.

Only hidden layers are masked; the final classifier layer is kept intact,
matching the convention used in the structured pipeline.
"""

from __future__ import annotations

import numpy as np
import torch


UNSTRUCTURED_METHODS = {
    'random':    'Random (unstructured)',
    'magnitude': 'Weight magnitude (unstructured)',
    'wanda':     'WANDA (unstructured)',
}


def _replicate(masks, n_seeds):
    """Deterministic methods: return the same mask-set ``n_seeds`` times."""
    return [masks for _ in range(n_seeds)]


def _topk_mask(score, k):
    """Return a 0/1 mask of the same shape as ``score`` keeping its top-k entries
    (over the last axis if 1D; globally if 2D)."""
    if score.ndim == 1:
        if k >= score.numel():
            return torch.ones_like(score)
        thresh = torch.kthvalue(score, score.numel() - k + 1).values
        return (score >= thresh).to(score.dtype)
    raise ValueError("_topk_mask expects 1-D score (use row-wise topk for 2-D)")


def random_masks(model, densities, n_seeds=3, base_seed=42):
    """Per-layer Bernoulli masks with keep-probability ``s``.

    Parameters
    ----------
    model     : FCNetwork
    densities : iterable of float in (0, 1]
    n_seeds   : int - number of random mask realisations to draw per density
    base_seed : int

    Returns
    -------
    dict {s: [list of N_SEEDS mask-sets]}
    """
    out = {}
    for s in densities:
        seed_masks = []
        for k in range(n_seeds):
            g = torch.Generator().manual_seed(base_seed + 1000 * k + int(1e6 * s))
            masks = []
            for layer in model.layers[:-1]:
                shape = layer.weight.shape
                # Bernoulli keep mask with probability s
                mask = (torch.rand(shape, generator=g) < float(s)).to(torch.float32)
                masks.append(mask)
            seed_masks.append(masks)
        out[float(s)] = seed_masks
    return out


def magnitude_masks(model, densities, n_seeds=1, base_seed=42):
    """Global-per-layer magnitude pruning: keep top ``s·|W|`` weights of each hidden layer.

    Deterministic; ``n_seeds`` is accepted for API parity with ``random_masks``
    and simply duplicates the (identical) mask set.
    """
    out = {}
    for s in densities:
        masks = []
        for layer in model.layers[:-1]:
            W_abs = layer.weight.detach().abs().cpu()
            N = W_abs.numel()
            k = max(1, int(round(float(s) * N)))
            mask = _topk_mask(W_abs.flatten(), k).reshape(W_abs.shape)
            masks.append(mask)
        out[float(s)] = _replicate(masks, n_seeds)
    return out


def wanda_masks(model, densities, X_calib, n_seeds=1, base_seed=42):
    """Original WANDA (Sun et al., 2023) adapted to FC layers.

    Per output neuron (row of W), score each incoming weight j by
    ``|W_ij| * ||X_j||_2``, where ``X_j`` is that input feature's activation
    across the calibration set.  Keep the top ``s * fan_in`` entries per row.

    Parameters
    ----------
    model     : FCNetwork
    densities : iterable of float in (0, 1]
    X_calib   : ndarray or Tensor [N_calib, D]  (no labels needed)
    """
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    X = torch.as_tensor(np.asarray(X_calib), dtype=dtype, device=device)

    # Per-hidden-layer input activations: layer l's input is activations[l].
    activations = [X]
    h = X
    model.eval()
    with torch.no_grad():
        for layer in model.layers[:-1]:
            h = torch.relu(layer(h))
            activations.append(h)

    out = {}
    for s in densities:
        masks = []
        for l, layer in enumerate(model.layers[:-1]):
            W_abs = layer.weight.detach().abs()                       # [fan_out, fan_in]
            x_norm = activations[l].norm(p=2, dim=0)                  # [fan_in]
            score = W_abs * x_norm.unsqueeze(0)                       # [fan_out, fan_in]
            fan_in = score.shape[1]
            k = max(1, int(round(float(s) * fan_in)))
            if k >= fan_in:
                mask = torch.ones_like(score)
            else:
                _, idx = score.topk(k, dim=1)
                mask = torch.zeros_like(score)
                mask.scatter_(1, idx, 1.0)
            masks.append(mask.cpu())
        out[float(s)] = _replicate(masks, n_seeds)
    return out


def taylor_masks(model, densities, X_calib, y_calib, n_seeds=1, base_seed=42):
    raise NotImplementedError("taylor unstructured pruning not yet implemented")
