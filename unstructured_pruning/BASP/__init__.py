"""BASP — Bidirectional Activation-Saliency Pruning.

This project's one-shot unstructured pruning algorithm. See
``.docs/lessons_about-pruning.md`` for the design guide and derivation.
"""

from .basp import basp_masks, bidirectional_saliency

__all__ = ['basp_masks', 'bidirectional_saliency']
