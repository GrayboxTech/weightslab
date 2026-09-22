"""Live low-dimensional projections of a model's representation.

Currently one implementation: parametric UMAP (:mod:`parametric_umap`), which
trains a small ``features -> R^3`` encoder alongside the model and writes each
sample's coordinates back as ordinary per-sample signals.
"""

from weightslab.projection.parametric_umap import (  # noqa: F401
    ENV_ENABLED,
    ENV_DIM,
    ENV_EVERY,
    ENV_NEIGHBORS,
    ProjectionEncoder,
    ProjectionTracker,
    attach_projection,
    detach_projection,
    find_ab_params,
    first_tensor,
    flatten_features,
    checkpoint_path,
    get_tracker,
    load_projection,
    membership_high_dim,
    observe_batch,
    pick_embedding_layer,
    project_dataset,
    projection_enabled,
    resolve_module,
    save_projection,
)

__all__ = [
    "ENV_ENABLED",
    "ENV_DIM",
    "ENV_EVERY",
    "ENV_NEIGHBORS",
    "ProjectionEncoder",
    "ProjectionTracker",
    "attach_projection",
    "detach_projection",
    "find_ab_params",
    "first_tensor",
    "flatten_features",
    "checkpoint_path",
    "get_tracker",
    "load_projection",
    "membership_high_dim",
    "observe_batch",
    "pick_embedding_layer",
    "project_dataset",
    "projection_enabled",
    "resolve_module",
    "save_projection",
]
