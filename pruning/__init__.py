"""Public API for the FC path-tracing pruning library (re-exports from
:mod:`pruning.pruning`)."""
from .pruning import (  # noqa: F401
    FCNetwork,
    accuracy,
    sigmoid_fn,
    fit_sigmoid,
    precompute_pruning_scores,
    evaluate_path_accuracy,
    _sparsify_dynamic,
    _precompute_column_masks,
    _estimate_batch_size,
    _trace_paths_batch,
    load_digits_data,
    make_comparison_plot,
    print_fit_summary,
    PRUNING_METHODS,
    METHOD_STYLE,
)

__all__ = [
    "FCNetwork",
    "accuracy",
    "sigmoid_fn",
    "fit_sigmoid",
    "precompute_pruning_scores",
    "evaluate_path_accuracy",
    "_sparsify_dynamic",
    "_precompute_column_masks",
    "_estimate_batch_size",
    "_trace_paths_batch",
    "load_digits_data",
    "make_comparison_plot",
    "print_fit_summary",
    "PRUNING_METHODS",
    "METHOD_STYLE",
]
