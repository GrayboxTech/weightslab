"""Serving for the live 3-D projection (see :mod:`weightslab.projection`).

The coordinates themselves are written during training as ordinary per-sample
signals (``signals//umap_x`` and friends), so this module is purely a read
path: it slices the current data view down to what one camera can actually
draw and hands back packed arrays.

Three things here are worth knowing about:

**Level of detail is server-side, driven by the client's view.** The client
never asks for "the dataset"; it asks for "what my camera can see, at most N
points" -- a box, plus the camera's frustum planes. Zooming in shrinks that
region, so the same N buys finer detail. A projection over millions of samples
therefore never crosses the wire in full, and the renderer's budget is a
constant.

**The subsample is stable, not random.** Every point has one fixed place in a
priority order (a deterministic hash of its sample id, stratified by cluster),
and a request returns the first N points of that order inside its region. So
the same request returns the same points, and zooming in only ever ADDS points:
a point drawn while zoomed out is still drawn after you zoom into it. A fresh
``random.sample`` per request would make the scene boil under the smallest
camera move, which reads as noise and destroys any sense that you are looking at
a stable structure.

**A request costs about what it returns, not the size of the dataset.** That
order, and a spatial grid over it, are built once per data change into a
:class:`ProjectionIndex` -- on a worker thread, while the previous one keeps
answering (see :class:`ProjectionCache`). A view query then reads a few cells or
a short prefix of the order instead of every sample. Recomputing the sample
over the whole frame on every request took 18-45 s at 3M samples, which is
what this layout exists to avoid.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid

import numpy as np
import pandas as pd

from weightslab.proto import experiment_service_pb2 as pb2

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "umap"
# How many points one view returns when the client does not ask for a number
# (the board does not, unless a browser overrides it): see default_max_points.
ENV_MAX_POINTS = "WEIGHTSLAB_PROJECTION_MAX_POINTS"
DEFAULT_MAX_POINTS = 70_000
# Above this the browser's main thread spends longer packing the typed arrays
# than the GPU spends drawing them; the client can ask for less, never more.
HARD_MAX_POINTS = 400_000
_WARNED_MAX_POINTS: set = set()
AXES = ("x", "y", "z")
# The most distinct values a label column may have and still count as "the
# clusters" for stratification.
MAX_GROUPS = 256
# Groups no bigger than this are ordered exactly within themselves (see
# priority_order). Bigger ones use the hash rank directly: as good for them, and
# it spares sorting millions of points twice.
SMALL_GROUP = 4096

# Below this many points a query is one vectorised pass over all of them --
# exact, and faster than consulting the cell grid.
CELL_INDEX_MIN_POINTS = 200_000
# Cells per axis of the spatial grid. Both keep a cell id in 16 bits, which is
# what lets numpy sort by cell with its linear-time radix sort.
GRID_CELLS = {2: 256, 3: 32}
# A query whose candidate cells hold at most this share of the cloud reads those
# cells; a bigger one walks the priority order and stops at the budget, which
# for a region that big is the shorter walk.
CELL_PATH_SHARE = 1 / 8


def default_max_points() -> int:
    """How many points a view returns when the client leaves it to the server.

    ``WEIGHTSLAB_PROJECTION_MAX_POINTS`` if set to a positive integer, else
    ``DEFAULT_MAX_POINTS`` (70,000); capped at ``HARD_MAX_POINTS``. Read on every
    request rather than once, so setting it from the Studio's notebook
    (``os.environ[...] = "150000"``) applies at the next camera move.
    """
    raw = os.environ.get(ENV_MAX_POINTS, "").strip()
    if not raw:
        return DEFAULT_MAX_POINTS
    try:
        value = int(raw)                      # "150000" and "150_000" both parse
    except ValueError:
        value = 0
    if value <= 0:
        if raw not in _WARNED_MAX_POINTS:     # once per value, not per camera move
            _WARNED_MAX_POINTS.add(raw)
            logger.warning(f"[projection] {ENV_MAX_POINTS}={raw!r} is not a positive "
                           f"integer; drawing {DEFAULT_MAX_POINTS:,} points per view")
        return DEFAULT_MAX_POINTS
    return min(value, HARD_MAX_POINTS)


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

    And only when something registered it as a projection (see
    :mod:`weightslab.projection.registry`): a column pair alone is not enough,
    or ``center_x`` / ``center_y`` signals turn up as a projection called
    ``center``. A run with no registry at all predates it, and keeps the old
    column-pair discovery.
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
    try:
        from weightslab.projection.registry import known_prefixes
        registered = known_prefixes()
    except Exception:
        registered = None
    if registered is not None:
        found = {p for p in found if p in registered or p == DEFAULT_PREFIX}
    ordered = sorted(found - {DEFAULT_PREFIX})
    return ([DEFAULT_PREFIX] if DEFAULT_PREFIX in found else []) + ordered


def grouping_key(frame: pd.DataFrame, coords: np.ndarray, color_column: str,
                 extent: tuple | None = None) -> np.ndarray:
    """The "cluster" each point belongs to, as integer codes, for stratification.

    Preference order, most to least meaningful:

    1. the categorical column the user is colouring by -- if they are looking
       at classes, classes are the clusters;
    2. ``target`` when it is low-cardinality (the usual supervised label);
    3. spatial cells of the cloud itself -- a label-free fallback that still
       guarantees no *visible* region of the projection can vanish, which is
       the property that actually matters on screen.

    *frame*'s rows must line up with *coords*.
    """
    return _group_codes(frame, None, coords, color_column, extent)[1]


def _group_codes(frame, rows, coords, color_column, extent=None):
    """``(column, codes)`` for :func:`grouping_key`, over *rows* of *frame* only.

    *column* is the label column used, or ``""`` for the spatial fallback.
    """
    for candidate in (color_column, "target", "signals//target"):
        name = (candidate or "").strip()
        if not name:
            continue
        series = _as_series(frame, name)
        if series is None:
            continue
        values = series.to_numpy()
        if rows is not None:
            values = values[rows]
        codes = _label_codes(values)
        if codes is not None:
            return name, codes
    return "", _spatial_codes(coords, extent)


def _label_codes(values) -> np.ndarray | None:
    """Integer codes for a label column, or None unless 1 < distinct <= MAX_GROUPS.

    Probed on a sample first: factorising ten million free-text values only to
    learn there are millions of them is the expensive way to say no. NaN is a
    group of its own, as the string conversion this replaces made it.
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return None
    try:
        if n > 100_000:
            probe = values[:: n // 100_000]
            if len(pd.unique(probe)) > MAX_GROUPS:
                return None
        codes, uniques = pd.factorize(values, use_na_sentinel=False)
    except TypeError:
        # Unhashable cells (lists, dicts): compare them by their text.
        try:
            codes, uniques = pd.factorize(values.astype(str), use_na_sentinel=False)
        except Exception:
            return None
    if not 1 < len(uniques) <= MAX_GROUPS:
        return None
    return codes.astype(np.int32, copy=False)


def _spatial_codes(coords, extent=None) -> np.ndarray:
    """Cell of an 8^d grid for each point: the label-free grouping.

    Anchored to the WHOLE cloud's extent, never to the current box. A grid
    recomputed per request re-labels every point as the camera moves, so the
    chosen subset changes wholesale and "zooming in only adds points" stops
    holding -- the cloud reshuffles under the user instead of gaining detail.
    """
    cells = 8
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)
    if extent is not None:
        lo, hi = np.asarray(extent[0], dtype=float), np.asarray(extent[1], dtype=float)
    else:
        lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    idx = np.clip(((coords - lo) / span * cells).astype(np.int64), 0, cells - 1)
    weights = cells ** np.arange(idx.shape[1] - 1, -1, -1)
    return (idx * weights).sum(axis=1).astype(np.int32)


def priority_order(rank: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Every point's index, in the order points are drawn.

    A request draws the first N points of this order that are inside its
    region, which is what makes the sample stable: a fixed order means the same
    request returns the same points, a bigger budget only adds points, and so
    does a smaller region (zooming in).

    The order is stratified by group, by the square root of the group's size.
    A flat sample of 50k points out of 10M gives a 2k-point cluster ~10 dots --
    it reads as noise, or vanishes, and the user concludes the model has no such
    cluster. Ordering each point by ``position_in_its_group / sqrt(group_size)``
    draws a group of n points at a density proportional to ``1/sqrt(n)``: a
    cluster 100x smaller is drawn 10x denser than a flat sample would, and every
    group, however small, has its first point near the very front.

    This replaces a fixed 10% floor per group. That floor was paid smallest
    group first until the budget ran out, so as soon as 10% of the data
    exceeded the budget -- 500k samples at the default 50k -- it spent the whole
    budget on the two smallest classes and drew the others not at all.

    *rank* is each point's stable hash in [0, 1) (see :func:`_stable_rank`).
    Groups up to :data:`SMALL_GROUP` points are ordered exactly by it; for
    bigger ones the rank times the group size IS the position, to within noise
    that does not matter at that size.
    """
    rank = np.asarray(rank, dtype=np.float64)
    codes = np.asarray(groups)
    if codes.dtype.kind not in "iu":
        codes, _ = pd.factorize(codes, use_na_sentinel=False)
    sizes = np.bincount(codes)
    root = np.sqrt(sizes, dtype=np.float64)
    key = rank * root[codes]
    small = sizes <= SMALL_GROUP
    if small.any():
        members = np.flatnonzero(small[codes])
        order = members[np.lexsort((rank[members], codes[members]))]
        group = codes[order]
        position = np.arange(len(order)) - np.searchsorted(group, group, side="left")
        # The rank term only breaks ties between groups' equal positions (every
        # group's first point sits at 0); it is far below the gap between two
        # positions of one group.
        key[order] = position / root[group] + rank[order] * 1e-6
    # Sorting the keys was the most expensive step of a build: a comparison
    # sort of every point, ~2 s at 10M. Quantised to 16 bits they radix-sort in
    # linear time instead, points sharing a bucket keeping their row order. Any
    # FIXED order keeps the properties above; this one only blurs the
    # stratification by 1/65536 of the key range.
    top = float(key.max()) if len(key) else 0.0
    if top <= 0:
        return np.arange(len(key))
    buckets = np.minimum(key * (65535.0 / top), 65535.0).astype(np.uint16)
    return np.argsort(buckets, kind="stable")


def stratified_keep(ids: np.ndarray, groups: np.ndarray, budget: int) -> np.ndarray:
    """Indices to draw out of *ids*: the first *budget* of :func:`priority_order`,
    in input order (so ids and colours stay aligned)."""
    n = len(ids)
    if n <= budget:
        return np.arange(n)
    keep = priority_order(_stable_rank(ids), groups)[:budget]
    keep.sort()
    return keep


def _mix_to_unit(x: np.ndarray) -> np.ndarray:
    """splitmix64-style finaliser, mapped to [0, 1): deterministic,
    process-independent, and well spread for sequential inputs."""
    x = x.astype(np.uint64, copy=True)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _stable_rank(sample_ids: np.ndarray) -> np.ndarray:
    """Deterministic [0, 1) rank per sample id -- the stable-subsample key.

    Never Python's ``hash``: it is salted per process, so the cloud would
    resample itself every time the server restarted.
    """
    ids = np.asarray(sample_ids)

    # Fast path: WeightsLab sample ids are integers (often carried as strings).
    # A vectorised integer mix is ~100x quicker than hashing text.
    as_int = None
    if ids.dtype.kind in "iu":
        as_int = ids.astype(np.int64, copy=False)
    elif ids.dtype.kind == "f":
        if np.isfinite(ids).all():
            as_int = ids.astype(np.int64)
    else:
        try:
            as_int = ids.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            as_int = None
    if as_int is not None:
        return _mix_to_unit(as_int)

    # Non-numeric ids (uuids, paths): FNV-1a over the code points, vectorised
    # across ids. An md5 per id was correct but a Python loop -- ~20 s at 10M.
    text = ids.astype(str)
    width = text.dtype.itemsize // 4
    points = text.view(np.uint32).reshape(len(text), width)
    h = np.full(len(text), 0xCBF29CE484222325, dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    for j in range(width):
        code = points[:, j].astype(np.uint64)
        # Skip the zero padding: a hash that depended on the array's width
        # would change the moment a longer id joined the dataset.
        h = np.where(code != 0, (h ^ code) * prime, h)
    return _mix_to_unit(h)


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


def _sample_ids_at(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Sample ids of *rows* only -- :func:`_sample_ids` without materialising
    the level for every row of a 10M-row frame to read a few thousand."""
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        names = list(index.names)
        level = names.index("sample_id") if "sample_id" in names else 0
        codes = np.asarray(index.codes[level])[rows]
        values = index.levels[level].take(np.where(codes < 0, 0, codes)).to_numpy()
        if (codes < 0).any():
            values = values.astype(object)
            values[codes < 0] = np.nan
        return values
    return index.take(rows).to_numpy()


def restrict_to_sample_ids(frame: pd.DataFrame, sample_ids) -> pd.DataFrame:
    """Rows of *frame* whose sample id is in *sample_ids* (compared as strings).

    An empty result is returned as such (the caller reports it): silently
    serving the whole frame when the ids do not match made a broken restriction
    look like a restriction that was never applied.

    Converts every id in *frame* on each call; :class:`SampleIdLookup` is the
    cached form the data service uses.
    """
    wanted = {str(s) for s in sample_ids}
    if not wanted:
        return frame
    keep = pd.Series(_sample_ids(frame)).astype(str).isin(wanted).to_numpy()
    return frame[keep]


def _plain_ids(values: pd.Index) -> bool:
    """Whether *values* hold plain integer or string ids, which pandas can
    hash as they are (anything else is compared as text)."""
    return values.dtype.kind in "iu" or (
        values.dtype.kind == "O" and values.inferred_type == "string")


def _keys_like(values: pd.Index, sample_ids) -> np.ndarray:
    """*sample_ids* as the dtype *values* holds -- the grid sends strings,
    a frame may hold integers."""
    ids = np.asarray(sample_ids)
    if values.dtype.kind in "iu":
        if ids.dtype.kind in "iu":
            return ids.astype(np.int64, copy=False)
        numbers = pd.to_numeric(pd.Series(ids.astype(str)), errors="coerce").dropna()
        return numbers[numbers == numbers.round()].astype(np.int64).to_numpy()
    if ids.dtype.kind == "O" and len(ids) and isinstance(ids[0], str):
        return ids
    return ids.astype(str).astype(object)


class SampleIdLookup:
    """Rows of a frame by sample id, through the index's own hash tables.

    A grid page is a hundred-odd samples asked for on every poll and scroll; a
    lasso selection can be millions. Converting every id of a 10M-row frame to
    a string to find them took ~4 s a time. On the data service's
    ``(origin, sample_id)`` view the ids are looked up in the sample_id LEVEL
    instead -- each distinct id once, through the hash table pandas builds once
    and keeps on the level, which every filtered slice of the view shares --
    and the rows are then found by integer level code. A frame indexed by
    sample id alone uses its index's table the same way. Only ids that are not
    plain integers or strings are converted to text, once per index.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._index_obj = None
        self._lookup: pd.Index | None = None

    def rows(self, frame: pd.DataFrame, sample_ids) -> np.ndarray:
        """Sorted row positions of *frame* holding any of *sample_ids*."""
        index = frame.index
        if isinstance(index, pd.MultiIndex):
            names = list(index.names)
            level = names.index("sample_id") if "sample_id" in names else 0
            values = index.levels[level]
            if _plain_ids(values):
                wanted = values.get_indexer(_keys_like(values, sample_ids))   # levels are unique
                wanted = wanted[wanted >= 0]
                if not len(wanted):
                    return np.zeros(0, dtype=np.int64)
                return np.flatnonzero(np.isin(np.asarray(index.codes[level]), wanted))
        elif _plain_ids(index):
            keys = _keys_like(index, sample_ids)
            found = (index.get_indexer(keys) if index.is_unique
                     else index.get_indexer_non_unique(keys)[0])
            return _sorted_unique(found[found >= 0])
        with self._lock:
            if self._index_obj is not index:
                self._lookup = pd.Index(_sample_ids(frame).astype(str))
                self._index_obj = index
            lookup = self._lookup
        keys = np.asarray(sample_ids).astype(str).astype(object)
        found = (lookup.get_indexer(keys) if lookup.is_unique
                 else lookup.get_indexer_non_unique(keys)[0])
        return _sorted_unique(found[found >= 0])


def _unique_ids_at(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """The distinct sample ids of *rows* -- de-duplicated by integer level code
    rather than by hashing millions of id strings."""
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        names = list(index.names)
        level = names.index("sample_id") if "sample_id" in names else 0
        codes = _sorted_unique(np.asarray(index.codes[level])[rows])
        return index.levels[level].take(codes[codes >= 0]).to_numpy()
    return pd.unique(index.take(rows).to_numpy())


def _sorted_unique(values: np.ndarray) -> np.ndarray:
    """``np.unique`` for integers, by sort-and-compare: numpy 2's hash-based
    unique took 2.2 s on the 1.5M row numbers of one lasso, this ~50 ms."""
    values = np.sort(values)
    if len(values) < 2:
        return values
    return values[np.concatenate(([True], values[1:] != values[:-1]))]


def missing_reason(prefix: str, prefixes: list) -> str:
    """Why there is nothing to draw for *prefix* -- the text the board's
    "No projection found" ribbon shows, so it has to say what to DO: wait for
    the first fit, look at the live projection's failure, or plug one in.
    """
    head = f"no projection for prefix '{prefix}'"
    if prefixes:
        return f"{head}. Available: {', '.join(prefixes)}."
    # Registered but not on the board yet: the data view is rebuilt in the
    # background (at most every few seconds), so a projection written a moment
    # ago -- the user's own t-SNE, just plugged in -- is not in it yet. Saying
    # "plug one in" right after they did would send them looking for a bug.
    try:
        from weightslab.projection.registry import known_prefixes
        just_written = sorted(known_prefixes() or ())
    except Exception:
        just_written = []
    if just_written:
        were = "was" if len(just_written) == 1 else "were"
        return (f"{head} on the board yet: {', '.join(just_written)} {were} just "
                f"written and should appear within a few seconds.")
    try:
        from weightslab.projection import get_tracker
        tracker = get_tracker()
    except Exception:
        tracker = None
    if tracker is not None and tracker.signal_prefix == prefix:
        if tracker.disabled_reason:
            return f"{head}: the live projection stopped after {tracker.disabled_reason}."
        if tracker._handle is not None:
            return (f"{head} yet: the live UMAP places its first points after its "
                    f"first fit (every {tracker.every_n_steps} training steps).")
    return (f"{head}: the built-in projection is off for this run. Plug in your "
            f"own with wl.save_projection_coords(coords, batch_ids, prefix=...).")


# --- the view index ------------------------------------------------------------


class ViewRegion:
    """The part of the cloud one request asks for: a box, optionally cut by the
    camera's frustum planes (a point is inside when ``n.p + d >= 0`` for each).
    """

    def __init__(self, lo, hi, normals=None, offsets=None):
        # float32, like the coordinates: a float64 bound would widen every
        # coordinate it is compared with.
        self.lo = np.asarray(lo, dtype=np.float32)
        self.hi = np.asarray(hi, dtype=np.float32)
        self.normals = None if normals is None else np.asarray(normals, dtype=np.float32)
        self.offsets = None if offsets is None else np.asarray(offsets, dtype=np.float32)

    @classmethod
    def from_request(cls, request, dims: int) -> "ViewRegion | None":
        """The request's region, or None for the whole cloud."""
        planes = np.asarray(list(getattr(request, "frustum_planes", []) or []),
                            dtype=np.float64)
        normals = offsets = None
        if planes.size and planes.size % 4 == 0:
            planes = planes.reshape(-1, 4)
            planes = planes[np.isfinite(planes).all(axis=1)]
            if len(planes):
                # A 2-D projection lies on z = 0, so the plane's z term drops.
                normals, offsets = planes[:, :dims], planes[:, 3]
        if not request.has_bounds and normals is None:
            return None
        if request.has_bounds:
            lo = np.array([request.min_x, request.min_y, request.min_z][:dims], dtype=np.float64)
            hi = np.array([request.max_x, request.max_y, request.max_z][:dims], dtype=np.float64)
            # A NaN bound is no bound (and must mean the same on every path).
            lo = np.where(np.isnan(lo), -np.inf, lo)
            hi = np.where(np.isnan(hi), np.inf, hi)
        else:
            lo = np.full(dims, -np.inf)
            hi = np.full(dims, np.inf)
        return cls(lo, hi, normals, offsets)

    def contains(self, xyz: np.ndarray) -> np.ndarray:
        """Boolean mask of the rows of *xyz* inside the region."""
        inside = np.ones(len(xyz), dtype=bool)
        for axis in range(xyz.shape[1]):
            if np.isfinite(self.lo[axis]):
                inside &= xyz[:, axis] >= self.lo[axis]
            if np.isfinite(self.hi[axis]):
                inside &= xyz[:, axis] <= self.hi[axis]
        if self.normals is not None and inside.any():
            # Planes only on what the box kept: one small matmul instead of six
            # passes over every candidate.
            idx = np.flatnonzero(inside)
            ok = (xyz[idx] @ self.normals.T + self.offsets >= 0).all(axis=1)
            inside[idx[~ok]] = False
        return inside


# The lasso is rasterised at canvas resolution, capped to this many pixels on
# the long side (a 4K canvas is the most anyone draws a loop on).
MAX_LASSO_RASTER = 4096


def points_in_lasso(xyz: np.ndarray, view_projection, polygon: np.ndarray,
                    viewport) -> np.ndarray:
    """Which rows of *xyz* land inside *polygon* on screen.

    The same projection as the viewer's own ``idsInPolygon``: *view_projection*
    is three.js's ``projectionMatrix * matrixWorldInverse`` (column-major, as
    its ``elements``), *polygon* is in canvas pixels with y down, and *viewport*
    is the canvas size. Points behind the camera are never inside.

    The loop is rasterised once with PIL and each point is then a single mask
    lookup -- an even-odd test against every edge would cost points x edges,
    which for a few million points and a hand-drawn loop is seconds.
    """
    from PIL import Image, ImageDraw

    result = np.zeros(len(xyz), dtype=bool)
    width, height = float(viewport[0]), float(viewport[1])
    if width <= 0 or height <= 0 or len(polygon) < 3 or not len(xyz):
        return result
    matrix = np.asarray(view_projection, dtype=np.float64).reshape(4, 4).T
    dims = xyz.shape[1]
    # A 2-D projection lies on z = 0, so the matrix's z column drops out.
    clip = np.asarray(xyz, dtype=np.float64) @ matrix[:, :dims].T + matrix[:, 3]
    front = np.flatnonzero(clip[:, 3] > 1e-9)
    if not len(front):
        return result
    w = clip[front, 3]
    scale = min(1.0, MAX_LASSO_RASTER / max(width, height))
    cols, rows = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    px = (clip[front, 0] / w + 1.0) * 0.5 * cols
    py = (1.0 - clip[front, 1] / w) * 0.5 * rows
    on_canvas = np.flatnonzero((px >= 0) & (px < cols) & (py >= 0) & (py < rows))

    image = Image.new("1", (cols, rows), 0)
    ImageDraw.Draw(image).polygon(
        [(float(x) * scale, float(y) * scale) for x, y in polygon], fill=1, outline=1)
    mask = np.asarray(image, dtype=bool)
    hit = mask[py[on_canvas].astype(np.int64), px[on_canvas].astype(np.int64)]
    result[front[on_canvas[hit]]] = True
    return result


class _Cells:
    """A uniform grid over the cloud: each cell's points, in priority order."""

    __slots__ = ("g", "lo", "size", "start", "prio", "xyz")


def _build_cells(xyz: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> _Cells:
    """Group *xyz* (already in priority order) by grid cell.

    A stable sort by cell keeps each cell's points in priority order, so a cell
    is a sorted run of priorities -- the property the cell path of
    :meth:`ProjectionIndex.select` relies on.
    """
    dims = xyz.shape[1]
    g = GRID_CELLS[dims]
    lo = np.asarray(lo, dtype=np.float64)
    span = np.where((np.asarray(hi) - lo) > 0, np.asarray(hi) - lo, 1.0)
    size = span / g
    cell = np.zeros(len(xyz), dtype=np.uint16)     # 32^3 and 256^2 both fit: radix sort
    for axis in range(dims):
        part = ((xyz[:, axis] - np.float32(lo[axis])) * np.float32(1.0 / size[axis])).astype(np.int32)
        np.clip(part, 0, g - 1, out=part)
        cell *= g
        cell += part.astype(np.uint16)
    order = np.argsort(cell, kind="stable")
    start = np.zeros(g ** dims + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=g ** dims), out=start[1:])
    cells = _Cells()
    cells.g, cells.lo, cells.size, cells.start = g, lo, size, start
    cells.prio = order.astype(np.int32 if len(order) < 2 ** 31 else np.int64)
    cells.xyz = np.take(xyz, order, axis=0)
    return cells


def _column_values(frame: pd.DataFrame, name: str) -> np.ndarray:
    """*name* as a float array; NaN where it is not a number. No copy when the
    column already is float (the signals columns are, see DataService)."""
    column = frame[name]
    values = column.to_numpy()
    if values.dtype.kind == "f":
        return values
    return pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)


def _column_extent(columns):
    """``(lo, hi)`` over 1-D coordinate columns. Reduced per column: the same
    min/max along axis 0 of the stacked (n, 3) array took ~1 s at 10M rows,
    contiguous columns ~60 ms."""
    lo = np.array([float(c.min()) for c in columns], dtype=np.float32)
    hi = np.array([float(c.max()) for c in columns], dtype=np.float32)
    return lo, hi


def _sample_level_mask(frame: pd.DataFrame):
    """Sample-level rows only, or None for "all of them".

    On a detection/segmentation frame the instance rows (annotation_id >= 1)
    carry no coordinates and would come back as a wall of NaN.

    Addressed by NAME: the data service's view has no annotation level at all
    (it is indexed by origin/sample_id), so a positional "level 1 == 0" test
    there compares sample ids against 0 and throws every row away.

    Values also arrive as int64 on some frames and object/strings on others,
    so coerce before comparing; and never let this filter empty the frame,
    which would report "no finite coordinates yet" on a working projection.
    """
    annotations = _index_level(frame, "annotation_id")
    if annotations is None:
        return None
    try:
        level = pd.to_numeric(pd.Series(annotations), errors="coerce").to_numpy()
        keep = level == 0
        return keep if keep.any() else None
    except Exception:
        return None


def _frame_rows(frame: pd.DataFrame, rows: np.ndarray, names) -> pd.DataFrame:
    """*rows* of *frame*, with only the columns in *names* -- the colour, split
    and discarded fills read a few columns of the drawn points, and slicing a
    wide frame for them cost more than the rest of the response. Built column by
    column (each keeps its dtype) rather than through ``iloc``."""
    data = {}
    for name in dict.fromkeys(n for n in names if n):
        if name in frame.columns:
            column = frame[name]
            if isinstance(column, pd.Series):
                data[name] = column.array.take(rows)
    return pd.DataFrame(data, index=frame.index.take(rows))


def _fill_origins_from_index(response, frame: pd.DataFrame, rows: np.ndarray) -> bool:
    """:func:`_fill_origins` straight from the index's level codes, when the
    split is an index level (the data service's view) -- no string per point.
    False when it is not, for the general path to handle."""
    index = frame.index
    if "origin" in frame.columns or not isinstance(index, pd.MultiIndex) \
            or "origin" not in index.names:
        return False
    level = list(index.names).index("origin")
    codes = np.asarray(index.codes[level])[rows]
    used, inverse = np.unique(codes, return_inverse=True)
    names = index.levels[level]
    response.origins.extend([str(names[c]) if c >= 0 else "nan" for c in used])
    response.origin_ids.extend(np.asarray(inverse, dtype=np.int32).tolist())
    return True


class ProjectionIndex:
    """One projection of one data frame, laid out so a view query costs about
    what it returns.

    Holds every projected point in :func:`priority_order` (``rows`` are their
    rows in *frame*, ``xyz`` their coordinates when the index was built), and,
    from :data:`CELL_INDEX_MIN_POINTS` points up, the same points grouped by a
    uniform grid. :meth:`select` answers "the first N points of the order inside
    this region" from whichever of the two reads less.

    What is drawn is read from *frame* at answer time -- coordinates, colours,
    split, discarded -- so a point's position and colour are live even while
    the index that chose it is a few seconds old. Only WHICH points are chosen
    waits for the next build.
    """

    def __init__(self, *, frame, prefix, columns, rows, xyz, lo, hi, cells,
                 color_column, group_column, fingerprint, build_seconds):
        self.frame = frame
        self.prefix = prefix
        self.columns = columns
        self.dims = len(columns)
        self.rows = rows
        self.xyz = xyz
        self.lo = lo
        self.hi = hi
        self.cells = cells
        self.color_column = color_column
        self.group_column = group_column
        self.fingerprint = fingerprint
        self.build_seconds = build_seconds

    @property
    def n(self) -> int:
        return len(self.rows)

    def ensure_cells(self) -> None:
        """Build the cell grid if it was deferred (see :meth:`build`)."""
        if self.cells is None and self.n >= CELL_INDEX_MIN_POINTS:
            self.cells = _build_cells(self.xyz, self.lo, self.hi)

    @classmethod
    def build(cls, frame, prefix: str = "", color_column: str = "", previous=None,
              defer_cells: bool = False):
        """``(index, None)``, or ``(None, failure_response)`` when there is
        nothing to index.

        With *previous* (an index of the same frame), unchanged data returns
        *previous* itself, and moved coordinates reuse its priority order --
        the hash, grouping and sort are what a full build spends its time on,
        and none of them depend on where the points are.

        *defer_cells* leaves the cell grid to :meth:`ensure_cells`: queries
        are exact without it, only slower, so the first answer need not wait.
        """
        started = time.perf_counter()
        if frame is None or len(frame) == 0:
            return None, pb2.ProjectionResponse(
                success=False, message="no data view available")

        # Listed even on the failure paths below, so a user who asked for a prefix
        # that isn't there is told which ones ARE -- the common case right after an
        # offline re-projection under a new name.
        prefixes = available_prefixes(frame)
        # No preference from the client means "whatever there is": the live
        # projection when it exists, else the first stored one. Defaulting blindly
        # to "umap" failed the first request of every run that turned the built-in
        # off and brought its own t-SNE, and the board only recovered a poll later.
        asked = (prefix or "").strip()
        prefix = asked or (DEFAULT_PREFIX if DEFAULT_PREFIX in prefixes or not prefixes
                           else prefixes[0])

        columns = coordinate_columns(frame, prefix)
        if len(columns) < 2:
            return None, pb2.ProjectionResponse(
                success=False,
                message=missing_reason(prefix, prefixes),
                available_prefixes=prefixes,
            )

        raw = [_column_values(frame, c) for c in columns]
        keep = np.ones(len(frame), dtype=bool)
        for values in raw:
            keep &= np.isfinite(values)
        level = _sample_level_mask(frame)
        if level is not None:
            keep &= level
        if not keep.any():
            return None, pb2.ProjectionResponse(
                success=False,
                message=f"projection '{prefix}' has no coordinates yet",
                total_available=0,
                available_prefixes=prefixes,
            )

        color_column = (color_column or "").strip()
        fingerprint = (
            len(frame),
            hashlib.blake2b(np.packbits(keep).tobytes(), digest_size=16).digest(),
            tuple(float(np.nansum(v, dtype=np.float64)) for v in raw),
        )
        rows = group_column = None
        if (previous is not None and previous.frame is frame and previous.prefix == prefix
                and previous.columns == columns and previous.color_column == color_column):
            if previous.fingerprint == fingerprint:
                return previous, None
            if previous.fingerprint[:2] == fingerprint[:2]:
                # Same samples, new coordinates.
                rows, group_column = previous.rows, previous.group_column

        row_type = np.int32 if len(frame) < 2 ** 31 else np.int64
        if rows is None:
            # Every row projected is the common case: skip the gathers.
            everything = bool(keep.all())
            rows_asc = np.arange(len(frame), dtype=row_type) if everything else np.flatnonzero(keep)
            columns_xyz = [values if everything else values[rows_asc] for values in raw]
            lo, hi = _column_extent(columns_xyz)
            xyz = np.stack(columns_xyz, axis=1).astype(np.float32, copy=False)
            ids = _sample_ids(frame)
            group_column, codes = _group_codes(frame, None if everything else rows_asc,
                                               xyz, color_column, (lo, hi))
            order = priority_order(_stable_rank(ids if everything else ids[rows_asc]), codes)
            rows = rows_asc[order].astype(row_type, copy=False)
            xyz = np.take(xyz, order, axis=0)
        else:
            columns_xyz = [values[rows] for values in raw]
            lo, hi = _column_extent(columns_xyz)
            xyz = np.stack(columns_xyz, axis=1).astype(np.float32, copy=False)

        cells = (_build_cells(xyz, lo, hi)
                 if len(rows) >= CELL_INDEX_MIN_POINTS and not defer_cells else None)
        return cls(
            frame=frame, prefix=prefix, columns=columns, rows=rows, xyz=xyz,
            lo=lo.astype(np.float64), hi=hi.astype(np.float64), cells=cells,
            color_column=color_column, group_column=group_column,
            fingerprint=fingerprint, build_seconds=time.perf_counter() - started,
        ), None

    # ------------------------------------------------------------------ query
    def select(self, region: ViewRegion | None, budget: int):
        """``(positions, in_view)``: the first *budget* points of the priority
        order inside *region*, as ascending positions in that order, and how
        many points the region holds (estimated when the walk stopped early).

        Every path returns exactly the same points for the same region; they
        differ only in how much they read.
        """
        n = self.n
        if region is None:
            return np.arange(min(n, budget)), n
        if self.cells is None:
            hits = np.flatnonzero(region.contains(self.xyz))
            return hits[:budget], len(hits)
        starts, ends = self._cell_runs(region)
        candidates = int((ends - starts).sum())
        if candidates == 0:
            return np.zeros(0, dtype=np.int64), 0
        if candidates <= max(budget, n * CELL_PATH_SHARE):
            return self._select_from_cells(region, starts, ends, budget)
        return self._select_by_walk(region, budget)

    def _cell_runs(self, region: ViewRegion):
        """Start/end offsets of the grid cells the region can reach: those its
        box overlaps, minus those wholly outside one of its planes.

        Culling cells against the frustum is what keeps an oblique view cheap:
        its bounding box takes in most of the cloud, the frustum itself a
        slice of it."""
        cells = self.cells
        g = cells.g
        axes = []
        for axis in range(self.dims):
            lo, hi = float(region.lo[axis]), float(region.hi[axis])
            if lo == np.inf or hi == -np.inf:
                empty = np.zeros(0, dtype=np.int64)
                return empty, empty
            first = 0 if lo == -np.inf else int(np.floor((lo - cells.lo[axis]) / cells.size[axis]))
            last = g - 1 if hi == np.inf else int(np.floor((hi - cells.lo[axis]) / cells.size[axis]))
            first, last = max(first, 0), min(last, g - 1)
            if first > last:
                empty = np.zeros(0, dtype=np.int64)
                return empty, empty
            axes.append(np.arange(first, last + 1))
        index = [grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")]
        ids = index[0]
        for part in index[1:]:
            ids = ids * g + part
        if region.normals is not None:
            keep = np.ones(len(ids), dtype=bool)
            corner = [cells.lo[a] + index[a] * cells.size[a] for a in range(self.dims)]
            for normal, offset in zip(region.normals.astype(np.float64), region.offsets):
                # The cell's corner furthest along the normal: if even that one
                # is outside the plane, the whole cell is.
                reach = float(offset) + sum(
                    normal[a] * (corner[a] + (cells.size[a] if normal[a] > 0 else 0.0))
                    for a in range(self.dims))
                keep &= reach >= 0
            ids = ids[keep]
        return cells.start[ids], cells.start[ids + 1]

    @staticmethod
    def _cell_positions(starts, ends) -> np.ndarray:
        """Every offset of the runs ``[start, end)``, concatenated."""
        lengths = ends - starts
        live = lengths > 0
        starts, lengths = starts[live], lengths[live]
        total = int(lengths.sum())
        before = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        return np.repeat(starts - before, lengths) + np.arange(total)

    def select_all(self, region: ViewRegion | None) -> np.ndarray:
        """Ascending positions of EVERY point inside *region* -- no budget.
        What a selection needs: the drawn points are only a subsample."""
        n = self.n
        if region is None:
            return np.arange(n)
        if self.cells is not None:
            starts, ends = self._cell_runs(region)
            candidates = int((ends - starts).sum())
            if candidates == 0:
                return np.zeros(0, dtype=np.int64)
            if candidates <= n // 2:
                positions = self._cell_positions(starts, ends)
                inside = region.contains(self.cells.xyz[positions])
                found = self.cells.prio[positions[inside]].astype(np.int64)
                found.sort()
                return found
        return np.flatnonzero(region.contains(self.xyz))

    def lasso(self, request) -> np.ndarray:
        """Ascending positions of every point whose screen position, under the
        request's camera, falls inside its lasso polygon -- whether or not the
        view ever drew it. The request's box and planes (the loop's own
        sub-frustum) narrow the candidates first."""
        polygon = np.asarray(list(request.lasso_px), dtype=np.float64)
        viewport = list(request.lasso_viewport)
        matrix = list(request.view_projection)
        if polygon.size < 6 or polygon.size % 2 or len(viewport) != 2 or len(matrix) != 16:
            return np.zeros(0, dtype=np.int64)
        candidates = self.select_all(ViewRegion.from_request(request, self.dims))
        if not len(candidates):
            return candidates
        inside = points_in_lasso(self.xyz[candidates], matrix, polygon.reshape(-1, 2), viewport)
        return candidates[inside]

    def _select_from_cells(self, region, starts, ends, budget):
        cells = self.cells
        positions = self._cell_positions(starts, ends)
        inside = region.contains(cells.xyz[positions])
        prios = cells.prio[positions[inside]].astype(np.int64)
        in_view = len(prios)
        if in_view > budget:
            prios = np.partition(prios, budget - 1)[:budget]
        prios.sort()
        return prios, in_view

    def _select_by_walk(self, region, budget):
        """Walk the priority order in growing chunks until *budget* points of
        the region have been seen. For a region holding a share s of the cloud
        that reads about budget / s points."""
        n = self.n
        found, hits = 0, []
        start, step = 0, max(4 * budget, 65_536)
        while start < n and found < budget:
            stop = min(n, start + step)
            idx = np.flatnonzero(region.contains(self.xyz[start:stop])) + start
            hits.append(idx)
            found += len(idx)
            start, step = stop, step * 2
        positions = np.concatenate(hits)[:budget] if hits else np.zeros(0, dtype=np.int64)
        in_view = found if start >= n else max(found, int(round(found * n / start)))
        return positions, in_view

    # --------------------------------------------------------------- response
    def _coords_at(self, rows: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Live coordinates of *rows*, falling back to the indexed ones for any
        row that no longer reads as a number."""
        indexed = self.xyz[positions]
        try:
            live = np.stack([np.asarray(_column_values(self.frame, c)[rows], dtype=np.float32)
                             for c in self.columns], axis=1)
        except Exception:
            return indexed
        stale = ~np.isfinite(live).all(axis=1)
        if stale.any():
            live[stale] = indexed[stale]
        return live

    def respond(self, request, selection: "_Selection | None" = None) -> pb2.ProjectionResponse:
        """Answer *request* from this index. *selection* (the current lasso
        selection) flags the returned points that belong to it."""
        budget = int(request.max_points) or default_max_points()
        budget = max(1, min(budget, HARD_MAX_POINTS))
        positions, in_view = self.select(ViewRegion.from_request(request, self.dims), budget)
        rows = self.rows[positions]
        frame = self.frame
        ids = np.asarray(_sample_ids_at(frame, rows))

        response = pb2.ProjectionResponse(
            success=True,
            message=f"{len(rows)} of {in_view} points in view",
            sample_ids=ids.astype(str).tolist(),
            coords=self._coords_at(rows, positions).reshape(-1).tolist(),
            returned=len(rows),
            total_in_view=in_view,
            total_available=self.n,
            dims=self.dims,
            available_prefixes=available_prefixes(frame),
        )
        # Extent of the WHOLE projection, not of the region: the client frames
        # the full cloud from this on first load and fits its grid walls to it.
        response.extent_min_x, response.extent_min_y = float(self.lo[0]), float(self.lo[1])
        response.extent_max_x, response.extent_max_y = float(self.hi[0]), float(self.hi[1])
        if self.dims > 2:
            response.extent_min_z = float(self.lo[2])
            response.extent_max_z = float(self.hi[2])

        sub = _frame_rows(frame, rows, ("origin", "discarded", request.color_column,
                                        "target", "signals//target"))
        if not _fill_origins_from_index(response, frame, rows):
            _fill_origins(response, sub)
        _fill_colors(response, sub, request.color_column)
        _fill_discarded(response, sub)
        if selection is not None and len(selection.ids) and len(ids):
            response.in_selection.extend(selection.flags(frame, rows, ids).tolist())

        try:
            from weightslab.projection import get_tracker
            tracker = get_tracker()
            # Only the projection the live encoder writes has fits. A user's own
            # t-SNE shown with the encoder's count would claim training it never had.
            if tracker is not None and tracker.signal_prefix == self.prefix:
                response.fits = int(tracker.steps_trained)
        except Exception:
            pass
        response.layer, response.layer_detail = projection_layer(self.prefix)
        return response


def projection_layer(prefix: str) -> tuple:
    """``(layer, layer_detail)`` that *prefix*'s coordinates were computed from,
    ``("", "")`` when unknown (a projection the user computed themselves).

    The live tracker answers for its own prefix -- it may have re-attached to
    another layer since it registered; the registry for everything else.
    """
    try:
        from weightslab.projection import get_tracker
        tracker = get_tracker()
        if (tracker is not None and tracker.signal_prefix == prefix
                and tracker.layer_name):
            return tracker.layer_name, tracker.layer_detail or ""
    except Exception:
        pass
    try:
        from weightslab.projection.registry import prefix_info
        info = prefix_info(prefix)
    except Exception:
        info = {}
    return info.get("layer", ""), info.get("layer_detail", "")


def build_projection_response(frame, request) -> pb2.ProjectionResponse:
    """Answer one :class:`ProjectionRequest` against *frame*, indexing it on
    the spot.

    The data service keeps the index between requests instead (see
    :class:`ProjectionCache`); this uncached form is for tests, examples and
    one-off callers, and returns exactly what the cached one would.

    Index layout is not assumed. The ledger frame is indexed
    ``(sample_id, annotation_id)`` and the data service's view is indexed
    ``(origin, sample_id)``, so every level is resolved by NAME -- reading
    "level 0" gives a split name on one of them.
    """
    index, failure = ProjectionIndex.build(frame, request.prefix, request.color_column)
    if failure is not None:
        return failure
    return index.respond(request)


def _respond_in_full(frame, request) -> pb2.ProjectionResponse:
    """Every point of *frame*: no region, no budget. For a grid page, which is
    small enough to draw whole and is exactly what the user asked to see."""
    plain = pb2.ProjectionRequest()
    plain.CopyFrom(request)
    plain.has_bounds = False
    plain.ClearField("frustum_planes")
    plain.max_points = HARD_MAX_POINTS
    return build_projection_response(frame, plain)


class _Selection:
    """A lasso selection: its distinct sample ids, and the rows of the frame it
    was drawn on. Rows make the common cases integer work -- applying it to
    that same frame, flagging that frame's drawn points, combining it with the
    next loop -- and the ids carry it to any other frame."""

    __slots__ = ("ids", "frame", "rows", "_index")

    def __init__(self, ids: np.ndarray, frame, rows: np.ndarray):
        self.ids, self.frame, self.rows = ids, frame, rows
        self._index = None

    def rows_in(self, frame, lookup: SampleIdLookup) -> np.ndarray:
        """Sorted rows of *frame* holding the selection."""
        if frame is self.frame:
            return self.rows
        return lookup.rows(frame, self.ids) if len(self.ids) else np.zeros(0, dtype=np.int64)

    def flags(self, frame, rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Per point (given by its row of *frame* and its id): selected?"""
        if frame is self.frame:
            return np.isin(rows, self.rows)
        if self._index is None:
            self._index = pd.Index(self.ids)
        return self._index.get_indexer(ids) >= 0


class _Entry:
    __slots__ = ("index", "kind", "checked_at", "pulled_at")

    def __init__(self, index, kind, pulled_at=0.0):
        self.index = index
        self.kind = kind
        self.checked_at = time.monotonic()
        self.pulled_at = pulled_at


class ProjectionCache:
    """The :class:`ProjectionIndex` of each projection being looked at, kept
    fresh in the background.

    The first request for a projection builds its index -- the one wait anybody
    sees, a few seconds at 10M samples. From then on a request only QUERIES an
    index. When the data has moved on (training writes coordinates all the
    time), a newer index is built on a worker thread and swapped in while the
    old one keeps answering, so no camera move ever waits for a rebuild and the
    choice of points is at most one rebuild behind.

    The heavy work also happens outside the data service's lock, which the old
    path held for the whole computation -- stalling the grid and the plots for
    as long as a projection request took.
    """

    #: Least seconds between two freshness checks of one index -- more when a
    #: build takes longer, so a worker spends at most a fifth of its time
    #: rebuilding (training keeps the rest).
    REFRESH_SECONDS = 5.0
    #: Least seconds between two full pulls of the dataset -- needed only while
    #: the data view is filtered and the board still shows everything.
    FULL_PULL_SECONDS = 60.0

    #: How many lasso selections are kept for the data board to apply.
    KEPT_SELECTIONS = 4

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: dict = {}
        self._busy: set = set()
        self._ids = SampleIdLookup()
        # Selections, by token, oldest first; and the latest, which ADD and
        # REFINE combine with.
        self._selections: dict = {}
        self._selection: _Selection | None = None
        self._selection_lookup = SampleIdLookup()

    def note_edit(self) -> None:
        """Samples were edited (discarded, tagged): what a pulled copy of the
        dataset says about them is now stale.

        While the data view is filtered or sorted, the whole-dataset cloud is
        drawn from such a copy, re-pulled at most every FULL_PULL_SECONDS --
        so a sample discarded in the grid stayed un-greyed in the projection
        for up to a minute. The next request refreshes instead; the refresh
        still runs in the background, so the request itself does not wait.
        """
        with self._lock:
            for entry in self._entries.values():
                entry.pulled_at = 0.0
                entry.checked_at = 0.0

    def serve(self, request, *, view, is_filtered, refresh_view=None, pull_full=None):
        """Answer *request*.

        *view* returns the data board's current frame as it stands;
        *refresh_view* brings it up to date first (and may take a lock);
        *pull_full* builds the whole, unfiltered dataset (expensive);
        *is_filtered* says whether the view is narrowed by a filter.
        """
        follow = bool(request.follow_view)
        grid_ids = list(request.restrict_sample_ids) if follow else []
        if grid_ids:
            return self._serve_grid(view(), request, grid_ids)
        entry, failure = self._entry_for(request, (view, is_filtered, refresh_view, pull_full))
        if failure is not None:
            return failure
        if getattr(request, "lasso_mode", 0):
            return self._select(entry.index, request)
        with self._lock:
            selection = self._selection
        return entry.index.respond(request, selection)

    def _entry_for(self, request, sources):
        """``(entry, None)`` for the projection *request* looks at, or
        ``(None, failure)``. Builds it the first time; refreshes it in the
        background after that."""
        view, is_filtered, _refresh_view, _pull_full = sources
        # The whole dataset, unless the board follows a FILTERED view: then the
        # cloud is that subset (see DataService.GetProjection for why).
        kind = "view" if (request.follow_view and is_filtered()) else "full"
        key = (kind, (request.prefix or "").strip())
        color = (request.color_column or "").strip()

        with self._lock:
            entry = self._entries.get(key)
        # A filtered view the index was not built from is a different set of
        # samples: answering from it would show the wrong cloud, so rebuild.
        if entry is not None and kind == "view" and entry.index.frame is not view():
            entry = None
        if entry is None:
            frame, pulled = self._source(kind, sources)
            # The first answer is the one the user waits for: the grid only
            # speeds queries up, so it is built right after, on a worker.
            index, failure = ProjectionIndex.build(frame, key[1], color, defer_cells=True)
            if failure is not None:
                return None, failure
            if index.cells is None and index.n >= CELL_INDEX_MIN_POINTS:
                threading.Thread(target=index.ensure_cells, name="WL-ProjectionCells",
                                 daemon=True).start()
            entry = _Entry(index, kind, pulled_at=time.monotonic() if pulled else 0.0)
            with self._lock:
                self._entries[key] = entry
        else:
            self._maybe_refresh(key, entry, color, sources)
        return entry, None

    # -------------------------------------------------------------- selection
    def _select(self, index: ProjectionIndex, request) -> pb2.ProjectionResponse:
        """A lasso: select on the server, keep it, hand back a token.

        Every sample inside the loop is selected, not just the drawn ones -- at
        10M samples the view draws one in two hundred, and a selection of the
        drawn points only would be a sample of a selection. That can be
        millions of ids, so they stay here and the data board applies them by
        token ("@projection_selection <token>").
        """
        frame = index.frame
        rows = np.sort(index.rows[index.lasso(request)]).astype(np.int64)
        with self._lock:
            previous = self._selection
        mode = int(request.lasso_mode)
        if previous is not None and len(previous.ids) and mode in (
                pb2.PROJECTION_LASSO_ADD, pb2.PROJECTION_LASSO_REFINE):
            # Combined as row numbers -- integers, not millions of id strings.
            before = previous.rows_in(frame, self._selection_lookup)
            rows = (np.union1d(before, rows) if mode == pb2.PROJECTION_LASSO_ADD
                    else np.intersect1d(before, rows, assume_unique=True))
        # One sample under two splits is one sample selected.
        selection = _Selection(_unique_ids_at(frame, rows), frame, rows)
        token = uuid.uuid4().hex[:16]
        with self._lock:
            self._selection = selection
            self._selections[token] = selection
            while len(self._selections) > self.KEPT_SELECTIONS:
                self._selections.pop(next(iter(self._selections)))
        ids = selection.ids
        return pb2.ProjectionResponse(
            success=True,
            message=f"{len(ids)} selected",
            selected=len(ids),
            selection_token=token,
            total_available=index.n,
            dims=index.dims,
            available_prefixes=available_prefixes(index.frame),
        )

    def selection(self, token: str):
        """The sample ids a lasso selected, or None once it has expired."""
        with self._lock:
            selection = self._selections.get(token)
        return None if selection is None else selection.ids

    def selection_rows(self, token: str, frame) -> np.ndarray | None:
        """Rows of *frame* a lasso selected, or None once it has expired. Free
        when *frame* is the one the lasso was drawn on -- the usual case."""
        with self._lock:
            selection = self._selections.get(token)
        return None if selection is None else selection.rows_in(frame, self._selection_lookup)

    def clear_selection(self) -> None:
        """Forget the selection ADD and REFINE build on (the board was reset)."""
        with self._lock:
            self._selection = None

    def full_frame(self):
        """The frame the whole dataset was last indexed from, or None."""
        with self._lock:
            for (kind, _prefix), entry in self._entries.items():
                if kind == "full":
                    return entry.index.frame
        return None

    def rows_of(self, frame, sample_ids) -> np.ndarray:
        """Rows of *frame* holding *sample_ids* (see :class:`SampleIdLookup`)."""
        return self._selection_lookup.rows(frame, sample_ids)

    def _serve_grid(self, frame, request, grid_ids):
        if frame is None or len(frame) == 0:
            return pb2.ProjectionResponse(success=False, message="no data view available")
        rows = self._ids.rows(frame, grid_ids)
        if not len(rows):
            logger.warning(
                "GetProjection: none of the grid's %d sample ids (e.g. %s) are in "
                "the data view", len(grid_ids), grid_ids[:3])
            return pb2.ProjectionResponse(
                success=False,
                message=(f"None of the {len(grid_ids)} samples in "
                         "the grid are in the projection data."))
        return _respond_in_full(frame.iloc[rows], request)

    @staticmethod
    def _source(kind, sources):
        """``(frame, pulled)``: the frame an index of *kind* is built from."""
        view, is_filtered, refresh_view, pull_full = sources
        if kind == "full" and is_filtered() and pull_full is not None:
            pulled = pull_full()
            if pulled is not None and not getattr(pulled, "empty", True):
                return pulled, True
        if refresh_view is not None:
            return refresh_view(), False
        return view(), False

    def _maybe_refresh(self, key, entry, color, sources):
        wait = max(self.REFRESH_SECONDS, 4.0 * entry.index.build_seconds)
        if (time.monotonic() - entry.checked_at < wait
                and entry.index.color_column == color):
            return
        with self._lock:
            if key in self._busy:
                return
            self._busy.add(key)
        threading.Thread(
            target=self._refresh, args=(key, entry, color, sources),
            name="WL-ProjectionIndex", daemon=True,
        ).start()

    def _refresh(self, key, entry, color, sources):
        try:
            view, is_filtered, refresh_view, pull_full = sources
            now = time.monotonic()
            if entry.kind == "full" and is_filtered():
                # The whole dataset is not the view right now; re-pulling it
                # costs seconds and a full copy, so it is done rarely.
                if (now - entry.pulled_at < self.FULL_PULL_SECONDS
                        and entry.index.color_column == color):
                    entry.checked_at = now
                    return
            frame, pulled = self._source(entry.kind, sources)
            index, failure = ProjectionIndex.build(frame, key[1], color, previous=entry.index)
            with self._lock:
                if failure is not None:
                    # Gone (prefix removed, data reset): the next request
                    # rebuilds in the open and reports why.
                    self._entries.pop(key, None)
                elif index is entry.index:
                    entry.checked_at = time.monotonic()
                else:
                    self._entries[key] = _Entry(
                        index, entry.kind,
                        pulled_at=time.monotonic() if pulled else entry.pulled_at)
        except Exception:
            logger.exception("[projection] background index refresh failed")
            entry.checked_at = time.monotonic()
        finally:
            with self._lock:
                self._busy.discard(key)


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


def _text_labels(series: pd.Series):
    """``(codes, distinct_texts)`` for *series* -- its values as the text
    ``astype(str)`` gives, without converting every value to a string: each
    distinct value is converted once."""
    try:
        codes, uniques = pd.factorize(series.to_numpy(), use_na_sentinel=False)
        texts = pd.Series(uniques).astype(str).to_numpy()
    except TypeError:
        codes, texts = pd.factorize(series.astype(str).to_numpy(), use_na_sentinel=False)
        texts = np.asarray(texts)
    # Distinct values may share a text (1 and "1"): merge them as the old
    # astype(str)-then-unique did.
    merged, remap = np.unique(texts, return_inverse=True)
    return remap[codes], merged


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
            if 1 < len(_text_labels(series)[1]) <= 64:
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
        codes, uniques = _text_labels(series)
        if 1 < len(uniques) <= 64:
            response.color_labels.extend(uniques[codes].tolist())
            return

    numeric = pd.to_numeric(series, errors="coerce")
    if np.isfinite(numeric.to_numpy(dtype=np.float64)).any():
        response.color_values.extend(
            numeric.fillna(np.nan).to_numpy(dtype=np.float64).tolist())
    else:
        response.color_labels.extend(series.astype(str).tolist())
