"""Live low-dimensional projections of a model's representation.

Built in: parametric UMAP (:mod:`parametric_umap`), which trains a small
``features -> R^3`` encoder alongside the model and writes each sample's
coordinates back as ordinary per-sample signals.

Your own: any projection you compute (t-SNE, umap-learn, PCA...) plugs into the
same board through :func:`save_projection_coords`, or through
``project_dataset(..., method=...)`` (see :mod:`custom`).
"""

from weightslab.projection.parametric_umap import (  # noqa: F401
    ENV_ENABLED,
    ENV_DIM,
    ENV_EVERY,
    ENV_GRAPH,
    ENV_NEIGHBORS,
    ProjectionEncoder,
    ProjectionTracker,
    attach_projection,
    clear_pending_restore,
    detach_projection,
    encoder_state,
    find_ab_params,
    first_tensor,
    flatten_features,
    checkpoint_path,
    get_tracker,
    layer_detail,
    load_projection,
    membership_high_dim,
    mirror_projection,
    observe_batch,
    pick_embedding_layer,
    project_dataset,
    projection_enabled,
    reattach_projection,
    resolve_module,
    restore_encoder,
    restore_from_checkpoint,
    save_projection,
)
from weightslab.projection.custom import save_projection_coords  # noqa: F401

__all__ = [
    "ENV_ENABLED",
    "ENV_DIM",
    "ENV_EVERY",
    "ENV_GRAPH",
    "ENV_NEIGHBORS",
    "ProjectionEncoder",
    "ProjectionTracker",
    "attach_projection",
    "clear_pending_restore",
    "detach_projection",
    "encoder_state",
    "find_ab_params",
    "first_tensor",
    "flatten_features",
    "checkpoint_path",
    "get_tracker",
    "layer_detail",
    "load_projection",
    "membership_high_dim",
    "mirror_projection",
    "observe_batch",
    "pick_embedding_layer",
    "project_dataset",
    "projection_enabled",
    "reattach_projection",
    "resolve_module",
    "restore_encoder",
    "restore_from_checkpoint",
    "save_projection",
    "save_projection_coords",
]
