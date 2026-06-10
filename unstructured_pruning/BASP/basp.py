"""BASP — Bidirectional Activation-Saliency Pruning (this project's design).

Rationale (docs/lessons_about-pruning.md): the accuracy of a frozen pruned net
is ``A(s) = Phi(SNR(s))`` with ``SNR(s) = (J0 / sqrt(V)) * sqrt(s / (1 - s))``.
A pruning method cannot change the sigmoid's *shape* — only *where* it sits (the
critical density ``s_0``). The job of a good method is therefore to keep the
subnetwork that maximises the surviving signal ``J0``, i.e. to push ``s_0`` left.

BASP keeps exactly the two levers that an in-repo design sweep
(``unstructured_pruning.BASP.design_probe``) found to move ``s_0`` on these FC
nets, and drops the ones that did not:

  (1) a BIDIRECTIONAL activation-weighted signal score ``|W_ij|*||x_j||*||a_i||``.
      The output-relevance factor ``||a_i||`` is the term that beats plain WANDA.
  (2) PER-LAYER (uniform-s) allocation, not per-row and not a single global
      threshold. Per-layer lets every output neuron's weights compete for the
      layer budget (per-row WANDA cannot, and the ``||a_i||`` factor cancels
      within a row); a LAMP-style global threshold under-performed badly here.

Iteration and global/LAMP allocation were measured to *hurt* and are omitted:
BASP is one-shot, label-free, and one forward pass. The advantage over WANDA and
magnitude grows with depth ``L``, the regime where the depth exponent
``gamma > 0`` (per-layer variance accumulation) makes the surviving-signal choice
hardest.

The shared helpers (``_layer_activations``, ``_per_layer_topk_masks``,
``_replicate``) live in :mod:`unstructured_pruning.methods`; the dispatcher
``unstructured_pruning.methods.build_masks`` routes ``method='basp'`` here.
"""

from __future__ import annotations

from ..methods import _layer_activations, _per_layer_topk_masks, _replicate


def bidirectional_saliency(model, X_calib):
    """Per-hidden-layer signal saliency ``|W_ij| * ||x_j||_2 * ||a_i||_2``.

    ``x_j`` is the input-feature activation feeding the weight (WANDA's input
    drive) and ``a_i`` is the post-ReLU activation of the output neuron it feeds
    (its downstream relevance). The product is the first-order contribution of
    the weight to the propagated signal J0: a large weight matters only if it is
    both *driven* on the calibration data and *read out* by an active neuron.
    """
    activations = _layer_activations(model, X_calib)          # [X, a1, a2, ...]
    scores = []
    for l, layer in enumerate(model.layers[:-1]):
        W_abs = layer.weight.detach().abs()                   # [fan_out, fan_in]
        x_norm = activations[l].norm(p=2, dim=0)              # [fan_in]
        a_norm = activations[l + 1].norm(p=2, dim=0)          # [fan_out]
        scores.append(W_abs * x_norm.unsqueeze(0) * a_norm.unsqueeze(1))
    return scores


def basp_masks(model, densities, X_calib, n_seeds=1, base_seed=42):
    """BASP masks: bidirectional saliency, kept top-``s`` *within each layer*.

    One-shot and data-only (a calibration batch, no labels, no backprop).

    Parameters
    ----------
    model      : FCNetwork
    densities  : iterable of keep-fractions ``s in (0, 1]``
    X_calib    : calibration inputs
    """
    scores = bidirectional_saliency(model, X_calib)
    out = {}
    for s in densities:
        out[float(s)] = _replicate(_per_layer_topk_masks(scores, float(s)),
                                   n_seeds)
    return out
