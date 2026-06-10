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
    'random':         'Random (unstructured)',
    'magnitude':      'Weight magnitude (unstructured)',
    'wanda':          'WANDA (input-norm, per-row)',
    # --- added saliency suite ------------------------------------------
    'snip':           'SNIP (|w·g|, global)',
    'grasp':          'GraSP (Hessian–gradient, global)',
    'synflow':        'SynFlow (data-free path-norm, global)',
    'gradient':       'Gradient magnitude (|g|, global)',
    'lamp':           'LAMP (layer-adaptive magnitude, global)',
    'wanda_output':   'WANDA (output-norm, per-layer)',
    'wanda_global':   'WANDA (input-norm, global)',
    'random_er':      'Random Erdős–Rényi (layer-scaled)',
    'anti_magnitude': 'Anti-magnitude (keep smallest, control)',
    # --- this project's design (see .docs/lessons_about-pruning.md) --------
    'basp':           'BASP (bidirectional activation saliency, per-layer)',
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


# ---------------------------------------------------------------------------
# Saliency-suite helpers
# ---------------------------------------------------------------------------
def _calib_xy(model, X_calib, y_calib):
    """Tensors for a calibration batch on the model's device/dtype."""
    p = next(model.parameters())
    X = torch.as_tensor(np.asarray(X_calib), dtype=p.dtype, device=p.device)
    y = torch.as_tensor(np.asarray(y_calib), dtype=torch.long, device=p.device)
    return X, y


def _hidden_weights(model):
    """The weight tensors that pruning targets: hidden layers only."""
    return [layer.weight for layer in model.layers[:-1]]


def _hidden_weight_grads(model, X, y, create_graph=False):
    """``(weights, grads)`` of cross-entropy loss w.r.t. each hidden weight."""
    import torch.nn.functional as F
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(X)
    loss = F.cross_entropy(logits, y)
    weights = _hidden_weights(model)
    grads = torch.autograd.grad(loss, weights, create_graph=create_graph)
    return weights, grads


def _per_layer_topk_masks(scores, s):
    """Keep the top ``s`` fraction of entries *within each layer*."""
    masks = []
    for sc in scores:
        flat = sc.flatten()
        N = flat.numel()
        k = max(1, int(round(float(s) * N)))
        if k >= N:
            masks.append(torch.ones_like(sc, dtype=torch.float32).cpu())
            continue
        thresh = torch.kthvalue(flat, N - k + 1).values
        masks.append((sc >= thresh).to(torch.float32).cpu())
    return masks


def _global_topk_masks(scores, s):
    """Keep the top ``s`` fraction of entries *pooled across all layers*.

    This is the single-shot global threshold used by SNIP / GraSP / SynFlow /
    LAMP: one cut over every prunable weight, so the per-layer sparsity is
    decided by the scores rather than fixed a priori.
    """
    flat_all = torch.cat([sc.flatten() for sc in scores])
    N = flat_all.numel()
    k = max(1, int(round(float(s) * N)))
    if k >= N:
        return [torch.ones_like(sc, dtype=torch.float32).cpu() for sc in scores]
    thresh = torch.kthvalue(flat_all, N - k + 1).values
    return [(sc >= thresh).to(torch.float32).cpu() for sc in scores]


# ---------------------------------------------------------------------------
# Saliency-suite methods (all one-shot on the trained checkpoint)
# ---------------------------------------------------------------------------
def snip_masks(model, densities, X_calib, y_calib, n_seeds=1, base_seed=42):
    """SNIP connection sensitivity: score = |w · ∂L/∂w|, global top-k.

    Lee et al. 2019. Keeps the connections whose removal most perturbs the
    loss on the calibration batch.
    """
    X, y = _calib_xy(model, X_calib, y_calib)
    weights, grads = _hidden_weight_grads(model, X, y)
    scores = [(w.detach() * g).abs() for w, g in zip(weights, grads)]
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_global_topk_masks(scores, s), n_seeds)
    return out


def gradient_masks(model, densities, X_calib, y_calib, n_seeds=1, base_seed=42):
    """Pure gradient-magnitude saliency: score = |∂L/∂w|, global top-k."""
    X, y = _calib_xy(model, X_calib, y_calib)
    _weights, grads = _hidden_weight_grads(model, X, y)
    scores = [g.abs() for g in grads]
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_global_topk_masks(scores, s), n_seeds)
    return out


def grasp_masks(model, densities, X_calib, y_calib, n_seeds=1, base_seed=42):
    """GraSP: gradient-signal preservation via the Hessian-gradient product.

    Wang et al. 2020. Removal saliency is ``-(w ⊙ Hg)``; GraSP prunes the
    largest removal saliency, i.e. *keeps* the largest ``w ⊙ Hg``. The
    Hessian-vector product ``Hg`` is obtained by double backprop:
    ``Hg = ∂/∂w ⟨g, g_detached⟩``. Global top-k.
    """
    X, y = _calib_xy(model, X_calib, y_calib)
    weights, grads = _hidden_weight_grads(model, X, y, create_graph=True)
    gv = sum((g * g.detach()).sum() for g in grads)
    Hg = torch.autograd.grad(gv, weights)
    keep_scores = [(w.detach() * hg).detach() for w, hg in zip(weights, Hg)]
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_global_topk_masks(keep_scores, s), n_seeds)
    return out


def synflow_masks(model, densities, n_seeds=1, base_seed=42):
    """SynFlow synaptic-flow saliency (data-free), single-iteration.

    Tanaka et al. 2020. Linearise the network (abs weights, no bias, no
    nonlinearity), push an all-ones input through, sum the output to a scalar
    ``R``, and score each weight by ``|∂R/∂w · w|``. The product runs through
    the read-out layer too so gradient flow to every hidden weight is counted.
    Global top-k.
    """
    p = next(model.parameters())
    layers = list(model.layers)                 # include read-out for the flow
    abs_W = [layer.weight.detach().abs().clone().requires_grad_(True)
             for layer in layers]
    h = torch.ones(1, abs_W[0].shape[1], dtype=p.dtype, device=p.device)
    for W in abs_W:
        h = h @ W.t()                           # linear, no bias, no relu
    R = h.sum()
    grads = torch.autograd.grad(R, abs_W)
    # Hidden layers only (drop the read-out score).
    scores = [(g * W).abs().detach()
              for g, W in zip(grads[:-1], abs_W[:-1])]
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_global_topk_masks(scores, s), n_seeds)
    return out


def lamp_masks(model, densities, n_seeds=1, base_seed=42):
    """LAMP: Layer-Adaptive Magnitude Pruning score, global top-k.

    Lee et al. 2021. Within each layer, sort weights by magnitude; the LAMP
    score of a weight is ``w² / Σ_{j ≥ r} w_j²`` (tail-normalised square),
    which makes magnitudes comparable across layers of different scale. A
    single global threshold then allocates sparsity per layer automatically.
    """
    scores = []
    for W in _hidden_weights(model):
        w2 = (W.detach() ** 2).flatten()
        order = torch.argsort(w2, descending=True)
        sorted_w2 = w2[order]
        tail = torch.flip(torch.cumsum(torch.flip(sorted_w2, [0]), 0), [0])
        lamp_sorted = sorted_w2 / tail.clamp_min(1e-12)
        lamp = torch.empty_like(w2)
        lamp[order] = lamp_sorted
        scores.append(lamp.reshape(W.shape))
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_global_topk_masks(scores, s), n_seeds)
    return out


def anti_magnitude_masks(model, densities, n_seeds=1, base_seed=42):
    """Control baseline: keep the *smallest*-magnitude weights per layer.

    Deliberately adversarial ordering (the opposite of magnitude pruning); a
    lower bound that any sensible saliency should beat.
    """
    scores = [-W.detach().abs() for W in _hidden_weights(model)]
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_per_layer_topk_masks(scores, s), n_seeds)
    return out


def _layer_activations(model, X_calib):
    """Per-hidden-layer input activations on a calibration batch (no grad)."""
    p = next(model.parameters())
    X = torch.as_tensor(np.asarray(X_calib), dtype=p.dtype, device=p.device)
    activations = [X]
    h = X
    model.eval()
    with torch.no_grad():
        for layer in model.layers[:-1]:
            h = torch.relu(layer(h))
            activations.append(h)
    return activations


def wanda_global_masks(model, densities, X_calib, n_seeds=1, base_seed=42):
    """WANDA input-norm score ``|W_ij|·‖x_j‖`` but with a *global* top-k.

    Same per-element score as the per-row WANDA, pooled across all hidden
    layers for a single threshold (so layers compete for the budget).
    """
    activations = _layer_activations(model, X_calib)
    scores = []
    for l, layer in enumerate(model.layers[:-1]):
        W_abs = layer.weight.detach().abs()
        x_norm = activations[l].norm(p=2, dim=0)             # [fan_in]
        scores.append(W_abs * x_norm.unsqueeze(0))
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_global_topk_masks(scores, s), n_seeds)
    return out


def wanda_output_masks(model, densities, X_calib, n_seeds=1, base_seed=42):
    """WANDA-style score using the *output* activation norm: ``|W_ij|·‖a_i‖``.

    ``a_i`` is output neuron ``i``'s post-ReLU activation across the
    calibration set. Per-layer top-k (output scaling is constant within a row,
    so a global threshold is used across the layer's matrix).
    """
    activations = _layer_activations(model, X_calib)
    scores = []
    for l, layer in enumerate(model.layers[:-1]):
        W_abs = layer.weight.detach().abs()                  # [fan_out, fan_in]
        a_norm = activations[l + 1].norm(p=2, dim=0)         # [fan_out]
        scores.append(W_abs * a_norm.unsqueeze(1))
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_per_layer_topk_masks(scores, s), n_seeds)
    return out


def random_er_masks(model, densities, n_seeds=3, base_seed=42):
    """Random pruning with Erdős–Rényi layer-scaled keep probabilities.

    Mocanu et al. 2018 / Evci et al. 2020 (ERK, kernel term dropped for FC).
    Layer ``l`` gets raw weight ``(fan_in+fan_out)/(fan_in·fan_out)``; a single
    scale ``ε`` is chosen so the overall kept fraction matches ``s``, then each
    weight is kept by an independent Bernoulli draw. Larger layers are pruned
    harder, unlike uniform random.
    """
    hidden = _hidden_weights(model)
    shapes = [tuple(W.shape) for W in hidden]
    raw = np.array([(fo + fi) / (fo * fi) for (fo, fi) in shapes], dtype=float)
    params = np.array([fo * fi for (fo, fi) in shapes], dtype=float)
    total = params.sum()
    out = {}
    for s in densities:
        eps = float(s) * total / float((raw * params).sum())
        probs = np.minimum(1.0, eps * raw)
        seed_masks = []
        for k in range(n_seeds):
            masks = []
            for li, (fo, fi) in enumerate(shapes):
                g = torch.Generator().manual_seed(
                    base_seed + 1000 * k + 7 * li + int(1e6 * float(s)))
                m = (torch.rand((fo, fi), generator=g) < probs[li])
                masks.append(m.to(torch.float32))
            seed_masks.append(masks)
        out[float(s)] = seed_masks
    return out


# ---------------------------------------------------------------------------
# BASP — Bidirectional Activation-Saliency Pruning (this project's design)
# ---------------------------------------------------------------------------
# The algorithm lives in its own package, unstructured_pruning/BASP/. The
# dispatcher below imports it lazily so methods.py stays import-cycle free.


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------
def build_masks(method, model, densities, *, X_calib=None, y_calib=None,
                n_seeds=3, base_seed=42):
    """Return ``{s: [mask_set, ...]}`` for any registered method.

    Gradient methods (snip, grasp, gradient) require ``X_calib`` and
    ``y_calib``; the activation methods (wanda*) require ``X_calib`` only;
    the rest are data-free.
    """
    if method == 'random':
        return random_masks(model, densities, n_seeds=n_seeds, base_seed=base_seed)
    if method == 'random_er':
        return random_er_masks(model, densities, n_seeds=n_seeds, base_seed=base_seed)
    if method == 'magnitude':
        return magnitude_masks(model, densities, n_seeds=1, base_seed=base_seed)
    if method == 'anti_magnitude':
        return anti_magnitude_masks(model, densities, n_seeds=1)
    if method == 'lamp':
        return lamp_masks(model, densities, n_seeds=1)
    if method == 'synflow':
        return synflow_masks(model, densities, n_seeds=1)
    if method == 'wanda':
        return wanda_masks(model, densities, X_calib, n_seeds=1, base_seed=base_seed)
    if method == 'wanda_global':
        return wanda_global_masks(model, densities, X_calib, n_seeds=1)
    if method == 'wanda_output':
        return wanda_output_masks(model, densities, X_calib, n_seeds=1)
    if method == 'snip':
        return snip_masks(model, densities, X_calib, y_calib, n_seeds=1)
    if method == 'gradient':
        return gradient_masks(model, densities, X_calib, y_calib, n_seeds=1)
    if method == 'grasp':
        return grasp_masks(model, densities, X_calib, y_calib, n_seeds=1)
    if method == 'basp':
        from .BASP import basp_masks  # lazy: BASP imports helpers from here
        return basp_masks(model, densities, X_calib, n_seeds=1)
    raise ValueError(f"unknown method: {method}")
