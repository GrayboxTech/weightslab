"""Projections you compute yourself: t-SNE, umap-learn, PCA, anything.

The Projection Board draws any per-sample ``signals//<prefix>_x`` /
``_y`` (/ ``_z``) columns, whatever wrote them, so plugging your own algorithm
in is a matter of writing coordinates under a prefix of your choosing:

* :func:`save_projection_coords` -- you have the coordinates (and the sample
  ids they belong to); WeightsLab stores them.
* ``wl.project_dataset(model, loader, method=...)`` -- WeightsLab collects the
  features from the model for you and hands them to your ``method``.

Either way the result sits beside the built-in parametric UMAP (``umap``), and
the board's picker switches between them.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

# Becomes part of a column name (`signals//<prefix>_x`), so keep it to what
# survives every storage layer and query: no "/" (the signals// separator), no
# spaces, nothing that needs quoting in a query.
_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
AXES = ("x", "y", "z")


def validate_prefix(prefix) -> str:
    """*prefix*, checked. Raises ``ValueError`` with what is allowed."""
    name = str(prefix or "").strip()
    if not _PREFIX_RE.match(name):
        raise ValueError(
            f"projection prefix {prefix!r} is not usable as a column name: use "
            f"letters, digits, '_', '.' or '-', starting with a letter or digit "
            f"(e.g. 'tsne', 'umap_learn', 'pca.layer3').")
    return name


def as_coordinates(coords, n: int | None = None) -> np.ndarray:
    """Coerce *coords* to a float32 ``(N, 2)`` or ``(N, 3)`` array.

    Accepts a torch tensor, a numpy array, or a nested list. Rows that are
    NaN/inf are allowed -- they mean "not placed" and the board skips them.
    """
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    array = np.asarray(coords, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] not in (2, 3):
        raise ValueError(
            f"projection coordinates must be shaped (N, 2) or (N, 3); got "
            f"{tuple(array.shape)}. Reduce to 2 or 3 components first "
            f"(e.g. TSNE(n_components=3)).")
    if n is not None and array.shape[0] != n:
        raise ValueError(
            f"projection coordinates have {array.shape[0]} rows but there are "
            f"{n} sample ids; they must be aligned one to one.")
    return array


def method_name(method) -> str:
    """A prefix for *method* when the caller did not name one.

    ``TSNE(...)`` -> ``"tsne"``, ``def my_pca(x)`` -> ``"my_pca"``; a lambda,
    having no useful name, -> ``"custom"``.
    """
    name = getattr(method, "__name__", None) or type(method).__name__
    if name.startswith("<"):
        return "custom"
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name.lower()).strip("_.-")
    return name or "custom"


def run_method(method, features: np.ndarray) -> np.ndarray:
    """Run a user projection on ``(N, F)`` features, return ``(N, 2|3)``.

    *method* is anything with a scikit-learn style ``fit_transform`` (``TSNE``,
    ``PCA``, ``umap.UMAP``) or a plain callable ``features -> coords``.
    """
    fit_transform = getattr(method, "fit_transform", None)
    if callable(fit_transform):
        out = fit_transform(features)
    elif callable(method):
        out = method(features)
    else:
        raise TypeError(
            f"method must be 'umap', a callable features -> coords, or an object "
            f"with fit_transform (e.g. sklearn's TSNE); got {type(method).__name__}.")
    return as_coordinates(out, n=features.shape[0])


def check_not_live_prefix(prefix: str) -> None:
    """Refuse to write under the prefix the live parametric UMAP owns.

    The live projection rewrites its prefix on every fit, a few hundred samples
    at a time, so a user projection stored there would be overwritten
    piecemeal: one cloud holding two unrelated layouts. Turn the built-in off
    (``projection=False``) to reuse its name.
    """
    from weightslab.projection.parametric_umap import get_tracker
    live = get_tracker()
    if live is not None and live._handle is not None and live.signal_prefix == prefix:
        raise ValueError(
            f"prefix {prefix!r} belongs to the live parametric UMAP, which keeps "
            f"rewriting it during training. Pick another prefix (e.g. 'tsne'), or "
            f"turn the built-in off with wl.watch_or_edit(model, flag='model', "
            f"projection=False).")


def save_projection_coords(coords, batch_ids, prefix: str) -> list:
    """Store a projection you computed yourself, for the Projection Board.

    Writes ``signals//<prefix>_x``, ``_y`` (and ``_z`` for 3-D) onto each
    sample's row -- ordinary per-sample signals, so the grid can sort, filter
    and export them too -- and the board lists *prefix* in its picker beside
    the built-in ``umap``.

    The samples are NOT counted as seen: placing a sample in a picture is not
    the model training on it, so ``nb_seen`` / ``last_seen`` are left alone.

    Call it whenever you like -- at step 500, once per epoch, after training.
    Writing the same prefix again replaces the coordinates of the samples you
    pass; samples you leave out keep theirs.

    Args:
        coords: ``(N, 2)`` or ``(N, 3)`` -- tensor, array or nested list. A 2-D
            projection is drawn on the z = 0 plane. NaN rows are skipped.
        batch_ids: the ``N`` sample ids the rows belong to, in the same order
            (the ids the WeightsLab loaders yield).
        prefix: the projection's name, e.g. ``"tsne"``. Not ``"umap"`` while the
            built-in live projection is running (it would overwrite yours).

    Returns:
        list: the column names written.

    Example:
        >>> from sklearn.manifold import TSNE
        >>> coords = TSNE(n_components=3).fit_transform(features)   # (N, 3)
        >>> wl.save_projection_coords(coords, batch_ids=ids, prefix="tsne")
    """
    from weightslab import src as _src

    prefix = validate_prefix(prefix)
    check_not_live_prefix(prefix)
    if hasattr(batch_ids, "detach"):
        ids = [str(i) for i in batch_ids.detach().cpu().tolist()]
    else:
        ids = [str(i) for i in batch_ids]
    array = as_coordinates(coords, n=len(ids))
    if not ids:
        return []

    signals = {f"{prefix}_{axis}": array[:, i] for i, axis in enumerate(AXES[:array.shape[1]])}
    if array.shape[1] == 2:
        # A prefix that once held a 3-D projection still has its _z column, and
        # the board reads x, y AND z when all three exist -- new x/y would be
        # drawn against stale depth. Flatten it rather than leave it.
        try:
            # The handle save_signals writes through, so both agree on which
            # dataframe "already has a _z column" refers to.
            frame_m = _src.DATAFRAME_M if _src.DATAFRAME_M is not None else _src.get_dataframe()
            if f"signals//{prefix}_z" in frame_m.get_df_view(limit=1).columns:
                signals[f"{prefix}_z"] = np.zeros(len(ids), dtype=np.float32)
        except Exception:
            pass

    _src.save_signals(signals=signals, batch_ids=ids, log=False, _seen=False)
    from weightslab.projection.registry import register_prefix
    register_prefix(prefix)
    # Flush now. save_signals only buffers, and the board reads flushed rows:
    # with the usual 30 s flush interval, the cell that just "plugged in" a
    # t-SNE was followed by half a minute of "No projection found". The live
    # projection writes every step and rightly stays buffered; this is one
    # explicit write the user is about to go and look at.
    try:
        frame_m = _src.DATAFRAME_M if _src.DATAFRAME_M is not None else _src.get_dataframe()
        if hasattr(frame_m, "flush"):
            frame_m.flush()
    except Exception as exc:
        logger.debug(f"[projection] '{prefix}': immediate flush failed ({exc}); "
                     f"the board shows it at the next scheduled flush")
    written = [f"signals//{name}" for name in signals]
    placed = int(np.isfinite(array).all(axis=1).sum())
    logger.info(f"[projection] '{prefix}': {placed} of {len(ids)} samples placed "
                f"({array.shape[1]}-D) -> {written}")
    return written
