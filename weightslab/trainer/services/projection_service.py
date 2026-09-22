"""Serving for the live 3-D projection (see :mod:`weightslab.projection`).

The coordinates themselves are written during training as ordinary per-sample
signals (``signals//umap_x`` and friends), so this module is purely a read
path: it slices the current data view down to what one camera can actually
draw and hands back packed arrays.

Two things here are worth knowing about:

**Level of detail is server-side, driven by the client's view box.** The client
never asks for "the dataset"; it asks for "what is inside the box my camera is
looking at, at most N points". Zooming in shrinks the box, so the same N buys
finer detail. A projection over millions of samples therefore never crosses the
wire in full, and the renderer's budget is a constant.

**The subsample is stable, not random.** Points are ranked by a deterministic
hash of their sample id and the lowest-ranked N are kept. Because that rank
does not depend on the box, a point visible while zoomed out is still visible
after you zoom in -- zooming reveals additional points rather than resampling
the cloud. A fresh ``random.sample`` per request would make the scene boil
under the smallest camera move, which reads as noise and destroys any sense
that you are looking at a stable structure.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from weightslab.proto import experiment_service_pb2 as pb2

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "umap"
DEFAULT_MAX_POINTS = 50_000
# Above this the browser's main thread spends longer packing the typed arrays
# than the GPU spends drawing them; the client can ask for less, never more.
HARD_MAX_POINTS = 400_000
AXES = ("x", "y", "z")
# Every cluster keeps at least this share of its own points, whatever the
# budget -- see stratified_keep for why a flat global sample is not enough.
MIN_GROUP_FRACTION = 0.10


def coordinate_columns(frame: pd.DataFrame, prefix: str = DEFAULT_PREFIX) -> list:
    """The projection columns present in *frame*, in x, y, z order.

    Accepts both the ``signals//`` spelling the training write-back produces and
    a bare ``umap_x``, so a projection written by hand (or restored from an H5
    that flattened the prefix) still renders.
    """
    found = []
    for axis in AXES:
        for candidate in (f"signals//{prefix}_{axis}", f"{prefix}_{axis}"):
            if candidate in frame.columns:
                found.append(candidate)
                break
        else:
            break  # axes must be contiguous: x, then y, then z
    return found


def available_prefixes(frame: pd.DataFrame) -> list:
    """Every projection stored on *frame*, by prefix.

    A prefix counts as present only when its ``_x`` AND ``_y`` columns both
    exist -- a half-written projection would otherwise show up in the picker and
    fail the moment it was selected. ``"umap"`` (the live one) is listed first
    when present; the rest are alphabetical.
    """
    found = set()
    for column in frame.columns:
        name = str(column)
        base = name[len("signals//"):] if name.startswith("signals//") else name
        if not base.endswith("_x"):
            continue
        prefix = base[:-2]
        if not prefix:
            continue
        # >= 2, not merely truthy: a prefix with only an _x column would be
        # offered in the picker and then render nothing when selected.
        if len(coordinate_columns(frame, prefix)) >= 2:
            found.add(prefix)
    ordered = sorted(found - {DEFAULT_PREFIX})
    return ([DEFAULT_PREFIX] if DEFAULT_PREFIX in found else []) + ordered


def grouping_key(frame: pd.DataFrame, coords: np.ndarray, color_column: str,
                 extent: tuple | None = None) -> np.ndarray:
    """The "cluster" each point belongs to, for stratified decimation.

    Preference order, most to least meaningful:

    1. the categorical column the user is colouring by -- if they are looking
       at classes, classes are the clusters;
    2. ``target`` when it is low-cardinality (the usual supervised label);
    3. spatial cells of the cloud itself -- a label-free fallback that still
       guarantees no *visible* region of the projection can vanish, which is
       the property that actually matters on screen.
    """
    for candidate in (color_column, "target", "signals//target"):
        name = (candidate or "").strip()
        if not name:
            continue
        series = _as_series(frame, name)
        if series is None:
            continue
        values = series.astype(str).to_numpy()
        distinct = len(np.unique(values))
        if 1 < distinct <= 256:
            return values

    # Spatial fallback: an 8^3 grid.
    #
    # Anchored to the WHOLE cloud's extent, never to the current box. A grid
    # recomputed per request re-labels every point as the camera moves, so the
    # chosen subset changes wholesale and "zooming in only adds points" stops
    # holding -- the cloud reshuffles under the user instead of gaining detail.
    cells = 8
    if extent is not None:
        lo, hi = np.asarray(extent[0], dtype=float), np.asarray(extent[1], dtype=float)
    else:
        lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    idx = np.clip(((coords - lo) / span * cells).astype(int), 0, cells - 1)
    return np.array(["c%d_%d_%d" % tuple(row) if len(row) == 3 else "c%d_%d" % tuple(row)
                     for row in idx])


def stratified_keep(ids: np.ndarray, groups: np.ndarray, budget: int,
                    min_fraction: float = MIN_GROUP_FRACTION) -> np.ndarray:
    """Indices to draw: proportional, but with a floor per group.

    A single global decimation starves small clusters. At a 50k budget over 5M
    points a 2k-point cluster gets ~20 dots -- it reads as noise, or disappears,
    and the user concludes the model has no such cluster. Guaranteeing every
    group at least *min_fraction* of its own points keeps each cluster legible
    at every zoom level, which is the entire reason to look at a projection.

    Within a group, points are taken in the deterministic rank order
    (:func:`_stable_rank`), so the two properties the viewer depends on still
    hold: the same request returns the same points, and zooming in only ever
    adds points to the ones already on screen.
    """
    n = len(ids)
    if n <= budget:
        return np.arange(n)

    rank = _stable_rank(ids)
    ordered: dict = {}
    for group in np.unique(groups):
        member = np.flatnonzero(groups == group)
        ordered[group] = member[np.argsort(rank[member], kind="stable")]

    # SMALLEST groups first. They are the ones a flat sample erases, and paying
    # their floor costs the big groups almost nothing. Serving the big ones
    # first would spend the whole budget before reaching the clusters that
    # needed protecting -- which is the bug this ordering exists to avoid.
    by_size = sorted(ordered, key=lambda g: (len(ordered[g]), str(g)))

    # More groups than the budget has points: not every group can be drawn.
    # Take a deterministic prefix rather than overshooting the budget the
    # client said it could render.
    if len(by_size) > budget:
        by_size = by_size[:budget]

    quota: dict = {}
    remaining = budget
    for group in by_size:
        size = len(ordered[group])
        want = min(size, max(1, int(np.ceil(min_fraction * size))))
        give = min(want, remaining)
        quota[group] = give
        remaining -= give
        if remaining <= 0:
            break

    # Whatever is left goes out in proportion to each group's remaining points.
    if remaining > 0:
        headroom = {g: len(ordered[g]) - quota.get(g, 0) for g in by_size}
        total_headroom = sum(headroom.values())
        if total_headroom > 0:
            for group in by_size:
                room = headroom[group]
                if room <= 0:
                    continue
                add = min(room, int(remaining * room / total_headroom))
                quota[group] = quota.get(group, 0) + add

    chosen = [ordered[g][:quota[g]] for g in by_size if quota.get(g, 0) > 0]
    if not chosen:
        return np.arange(min(n, budget))
    keep = np.concatenate(chosen)
    keep.sort()          # input order, so ids/colours stay aligned
    return keep


def _stable_rank(sample_ids: np.ndarray) -> np.ndarray:
    """Deterministic [0, 1) rank per sample id -- the stable-subsample key.

    md5 of the id rather than Python's ``hash``: the latter is salted per
    process, so the cloud would resample itself every time the server restarted.
    """
    ids = np.asarray(sample_ids)

    # Fast path: WeightsLab sample ids are integers (carried as strings). A
    # vectorised integer mix is ~100x quicker than a per-id md5, and the md5
    # loop is the single dominant cost of this RPC at scale -- 2.1s of a 5.9s
    # response over 1M rows, which is the difference between a board that
    # refreshes and one that stalls.
    try:
        as_int = ids.astype(np.int64, copy=False)
    except (TypeError, ValueError):
        as_int = None
    if as_int is None:
        try:
            as_int = ids.astype(str).astype(np.int64)
        except (TypeError, ValueError):
            as_int = None

    if as_int is not None:
        # splitmix64-style finaliser: deterministic, process-independent, and
        # well spread for the sequential ids this actually gets.
        x = as_int.astype(np.uint64, copy=True)
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
        return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    # Non-numeric ids (uuids, paths): md5 per id, as before. Correct, slower,
    # and unavoidable without assuming a format.
    return np.array(
        [int(hashlib.md5(str(s).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
         for s in ids],
        dtype=np.float64,
    )


def _as_series(frame: pd.DataFrame, name: str):
    """Series for *name* whether it is a column or an index level (same helper
    GetHistogram uses, for the same reason: ``origin`` may live in either)."""
    if name in frame.columns:
        return frame[name]
    names = list(getattr(frame.index, "names", []) or [])
    if name in names:
        return pd.Series(frame.index.get_level_values(name), index=frame.index)
    if getattr(frame.index, "name", None) == name:
        return pd.Series(frame.index, index=frame.index)
    return None


def _index_level(frame: pd.DataFrame, name: str):
    """Index level *name*, or ``None``.

    Levels must be addressed BY NAME, never by position. The ledger frame is
    indexed ``(sample_id, annotation_id)`` but the data service's view is
    indexed ``(origin, sample_id)`` -- so "level 0" is the sample id in one and
    the split name in the other. Assuming a position silently hands back
    ``'train_loader'`` as a sample id and filters every row away.
    """
    names = list(getattr(frame.index, "names", []) or [])
    if name in names:
        return frame.index.get_level_values(name)
    return None


def _sample_ids(frame: pd.DataFrame) -> np.ndarray:
    """Sample ids for *frame*, whatever its index layout."""
    level = _index_level(frame, "sample_id")
    if level is not None:
        return level.to_numpy()
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(0).to_numpy()
    return index.to_numpy()


def build_projection_response(frame, request) -> pb2.ProjectionResponse:
    """Answer one :class:`ProjectionRequest` against the dataset.

    *frame* is the WHOLE dataset, not the data board's current subview (see
    ``DataService.GetProjection``): selecting in the projection filters that
    board, so following the filter here would collapse the cloud to the points
    just selected and leave nothing to select from next.

    Index layout is not assumed. The ledger frame is indexed
    ``(sample_id, annotation_id)`` and the data service's view is indexed
    ``(origin, sample_id)``, so every level is resolved by NAME -- reading
    "level 0" gives a split name on one of them.
    """
    prefix = (request.prefix or DEFAULT_PREFIX).strip() or DEFAULT_PREFIX
    budget = int(request.max_points) or DEFAULT_MAX_POINTS
    budget = max(1, min(budget, HARD_MAX_POINTS))

    if frame is None or len(frame) == 0:
        return pb2.ProjectionResponse(
            success=False, message="no data view available")

    # Listed even on the failure paths below, so a user who asked for a prefix
    # that isn't there is told which ones ARE -- the common case right after an
    # offline re-projection under a new name.
    prefixes = available_prefixes(frame)

    columns = coordinate_columns(frame, prefix)
    if len(columns) < 2:
        hint = (f" available: {', '.join(prefixes)}" if prefixes else
                " - train with WEIGHTSLAB_PROJECTION enabled, or wait for the "
                "first fit")
        return pb2.ProjectionResponse(
            success=False,
            message=f"no projection for prefix '{prefix}'.{hint}",
            available_prefixes=prefixes,
        )

    # Sample-level rows only: on a detection/segmentation frame the instance
    # rows (annotation_id >= 1) carry no coordinates and would come back as a
    # wall of NaN.
    #
    # Addressed by NAME: the data service's view has no annotation level at all
    # (it is indexed by origin/sample_id), so a positional "level 1 == 0" test
    # there compares sample ids against 0 and throws every row away.
    #
    # Values also arrive as int64 on some frames and object/strings on others,
    # so coerce before comparing; and never let this filter empty the frame,
    # which would report "no finite coordinates yet" on a working projection.
    annotations = _index_level(frame, "annotation_id")
    if annotations is not None:
        try:
            level = pd.to_numeric(
                pd.Series(annotations), errors="coerce").to_numpy()
            keep = level == 0
            if keep.any():
                frame = frame[keep]
        except Exception:
            pass

    coords = np.column_stack(
        [pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64)
         for c in columns])
    finite = np.isfinite(coords).all(axis=1)
    if not finite.any():
        return pb2.ProjectionResponse(
            success=False,
            message="projection columns exist but hold no finite coordinates yet",
            total_available=0,
            available_prefixes=prefixes,
        )

    coords = coords[finite]
    sub = frame[finite]
    ids = _sample_ids(sub).astype(str)
    total_available = int(coords.shape[0])

    # Extent of the WHOLE projection, not of the box: the client frames the
    # full cloud from this on first load and draws the zoom indicator from it.
    extent_min = coords.min(axis=0)
    extent_max = coords.max(axis=0)

    # --- view box ---------------------------------------------------------
    if request.has_bounds:
        lo = np.array([request.min_x, request.min_y, request.min_z][:coords.shape[1]])
        hi = np.array([request.max_x, request.max_y, request.max_z][:coords.shape[1]])
        inside = ((coords >= lo) & (coords <= hi)).all(axis=1)
        coords = coords[inside]
        sub = sub[inside]
        ids = ids[inside]
    total_in_view = int(coords.shape[0])

    # --- stable, stratified decimation -------------------------------------
    if total_in_view > budget:
        groups = grouping_key(sub, coords, request.color_column,
                              extent=(extent_min, extent_max))
        keep = stratified_keep(ids, groups, budget)
        coords = coords[keep]
        sub = sub.iloc[keep]
        ids = ids[keep]

    response = pb2.ProjectionResponse(
        success=True,
        message=f"{len(ids)} of {total_in_view} points in view",
        sample_ids=ids.tolist(),
        coords=coords.astype(np.float32).reshape(-1).tolist(),
        returned=len(ids),
        total_in_view=total_in_view,
        total_available=total_available,
        dims=int(coords.shape[1]),
        available_prefixes=prefixes,
    )
    response.extent_min_x, response.extent_min_y = float(extent_min[0]), float(extent_min[1])
    response.extent_max_x, response.extent_max_y = float(extent_max[0]), float(extent_max[1])
    if coords.shape[1] > 2:
        response.extent_min_z = float(extent_min[2])
        response.extent_max_z = float(extent_max[2])

    _fill_origins(response, sub)
    _fill_colors(response, sub, request.color_column)
    _fill_discarded(response, sub)

    try:
        from weightslab.projection import get_tracker
        tracker = get_tracker()
        if tracker is not None:
            response.fits = int(tracker.steps_trained)
    except Exception:
        pass

    return response


def _fill_origins(response, frame) -> None:
    """Split membership, as an id per point plus the name table.

    Interned rather than repeated per point: at a 50k budget the literal string
    "train_loader" 50,000 times is most of the payload.
    """
    series = _as_series(frame, "origin")
    if series is None:
        return
    values = series.astype(str).to_numpy()
    names, ids = np.unique(values, return_inverse=True)
    response.origins.extend(names.astype(str).tolist())
    # .tolist() rather than a comprehension: it converts the whole array in C,
    # and these run once per POINT on every refresh of the board.
    response.origin_ids.extend(np.asarray(ids, dtype=np.int32).tolist())


def _fill_discarded(response, frame) -> None:
    """Per-point discarded flag, so the viewer can grey those samples out."""
    series = _as_series(frame, "discarded")
    if series is None:
        return
    try:
        from weightslab.trainer.services.data_service import set_flag_mask
        mask = set_flag_mask(series)
    except Exception:
        # NaN in a nullable flag is NOT "discarded" -- the same trap the
        # histogram path documents.
        mask = series.fillna(False).astype(bool).to_numpy()
    response.discarded.extend(np.asarray(mask, dtype=bool).tolist())


def _fill_colors(response, frame, column: str) -> None:
    """Optional per-point colour source.

    Numeric columns fill ``color_values`` (the client ramps them); anything else
    fills ``color_labels`` (the client assigns a categorical palette). Same
    numeric-or-not question GetHistogram answers, kept deliberately simple here:
    a column that coerces to any finite number is numeric.
    """
    column = (column or "").strip()

    # "Colour by split" is the CLIENT's job: it holds the split palette, so a
    # sample is the same colour here as in the data grid. Filling labels for it
    # would hand the viewer a second, different palette for the same thing.
    if column in ("origin", "split"):
        return

    if not column:
        # Default to the ground-truth label when there is one: on a
        # classification set the classes are what the clusters MEAN, and
        # colouring by split paints the entire train set one colour, which
        # says nothing about the structure on screen.
        for candidate in ("target", "signals//target"):
            series = _as_series(frame, candidate)
            if series is None:
                continue
            distinct = len(pd.unique(series.astype(str).to_numpy()))
            if 1 < distinct <= 64:
                column = candidate
                break
    if not column:
        return
    series = _as_series(frame, column)
    if series is None:
        return

    # A class id is a label, not a measurement. Running 0..9 through the
    # continuous ramp places class 4 "between" 3 and 5 and paints neighbouring
    # classes near-identical shades -- the one thing a projection coloured by
    # class must not say. A regression target (many distinct values) still
    # ramps, which is what it should do.
    if column in ("target", "signals//target"):
        labels = series.astype(str)
        distinct = len(pd.unique(labels.to_numpy()))
        if 1 < distinct <= 64:
            response.color_labels.extend(labels.tolist())
            return

    numeric = pd.to_numeric(series, errors="coerce")
    if np.isfinite(numeric.to_numpy(dtype=np.float64)).any():
        response.color_values.extend(
            numeric.fillna(np.nan).to_numpy(dtype=np.float64).tolist())
    else:
        response.color_labels.extend(series.astype(str).tolist())
