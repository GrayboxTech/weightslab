"""Process-local store for per-sample media attached to metadata columns.

Any metadata field can carry media — an image, a mask, a video clip, or a point
cloud — not just the sample's own input. That is what makes a *generated* clip
viewable next to the clip it was trained against: the training script attaches
it with ``wl.save_media(field="pred_video", ...)`` and the studio then shows a
thumbnail in the list and opens the full viewer on click.

Split of responsibilities:

  * the dataframe column ``media:<field>`` holds only a small JSON **descriptor**
    (kind, mime, geometry, fps, duration) — cheap to query, sort and ship in a
    metadata response.
  * the encoded **bytes** live here, keyed by ``(field, sample_id)``, and are
    streamed on demand by the GetMedia RPC.

Bytes are held in memory rather than on disk on purpose: attached media is
regenerated every epoch, so persisting it would mean writing (and garbage
collecting) thousands of files nobody asks for. The store is capped by total
bytes and evicts least-recently-used entries, so a long run cannot grow without
bound — attaching media to more samples than the cap holds simply means the
oldest thumbnails stop being openable, which degrades gracefully.
"""
import json
import logging
import os
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Dataframe column prefix. Mirrors the existing "tag:" convention.
MEDIA_COLUMN_PREFIX = "media:"

# Media kinds a descriptor may declare.
KIND_IMAGE = "image"
KIND_MASK = "mask"
KIND_VIDEO = "video"
KIND_AUDIO = "audio"
KIND_POINTCLOUD = "pointcloud"

_VALID_KINDS = (KIND_IMAGE, KIND_MASK, KIND_VIDEO, KIND_AUDIO, KIND_POINTCLOUD)

_DEFAULT_MAX_BYTES = 256 * 1024 * 1024 # 256 MiB

_lock = threading.RLock()
# (field, sample_id) -> entry dict. Insertion-ordered => LRU.
_entries: "OrderedDict[tuple, dict]" = OrderedDict()
_total_bytes = 0


def media_column(field: str) -> str:
    """Dataframe column name for a media field."""
    return f"{MEDIA_COLUMN_PREFIX}{field}"


def field_from_column(column: str) -> str:
    """Inverse of :func:`media_column` ("" when not a media column)."""
    text = str(column or "")
    if text.startswith(MEDIA_COLUMN_PREFIX):
        return text[len(MEDIA_COLUMN_PREFIX):]
    return ""


def is_media_column(column: str) -> bool:
    return str(column or "").startswith(MEDIA_COLUMN_PREFIX)


def max_bytes() -> int:
    """Byte cap for the store (env WL_MEDIA_STORE_MAX_BYTES)."""
    raw = os.environ.get("WL_MEDIA_STORE_MAX_BYTES")
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "WL_MEDIA_STORE_MAX_BYTES=%r is not an integer — using default %d",
            raw, _DEFAULT_MAX_BYTES)
        return _DEFAULT_MAX_BYTES
    return value if value > 0 else _DEFAULT_MAX_BYTES


def put(field: str, sample_id, data: bytes, mime: str, kind: str,
        poster: bytes = b"", meta: dict = None) -> dict:
    """Store one sample's media and return its descriptor dict.

    ``poster`` is the still shown in the grid/list; for an image kind it may be
    the image itself. ``meta`` carries kind-specific extras (fps, frame_count,
    has_audio, num_points, ...) and is merged into the descriptor.
    """
    global _total_bytes

    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    entry = {
        "field": str(field),
        "sample_id": str(sample_id),
        "data": data or b"",
        "mime": str(mime or ""),
        "kind": str(kind),
        "poster": poster or b"",
        "meta": dict(meta or {}),
    }
    key = (str(field), str(sample_id))
    size = len(entry["data"]) + len(entry["poster"])

    with _lock:
        previous = _entries.pop(key, None)
        if previous is not None:
            _total_bytes -= len(previous["data"]) + len(previous["poster"])
        _entries[key] = entry
        _total_bytes += size

        cap = max_bytes()
        while _total_bytes > cap and len(_entries) > 1:
            _, evicted = _entries.popitem(last=False)
            _total_bytes -= len(evicted["data"]) + len(evicted["poster"])

    return descriptor(entry)


def descriptor(entry: dict) -> dict:
    """The small JSON-able summary that goes into the dataframe column."""
    out = {
        "kind": entry["kind"],
        "mime": entry["mime"],
        "bytes": len(entry.get("data") or b""),
    }
    out.update(entry.get("meta") or {})
    return out


def descriptor_json(entry: dict) -> str:
    try:
        return json.dumps(descriptor(entry))
    except (TypeError, ValueError):
        return json.dumps({"kind": entry.get("kind", "")})


def get(field: str, sample_id) -> dict:
    """Fetch an entry, refreshing its LRU position. None when absent."""
    key = (str(field), str(sample_id))
    with _lock:
        entry = _entries.pop(key, None)
        if entry is None:
            return None
        _entries[key] = entry # refresh recency
        return entry


def get_poster(field: str, sample_id) -> bytes:
    """Poster bytes for a stored entry (b"" when absent)."""
    entry = get(field, sample_id)
    return (entry or {}).get("poster") or b""


def fields() -> list:
    """Every media field currently holding at least one entry."""
    with _lock:
        return sorted({key[0] for key in _entries})


def stats() -> dict:
    """Store occupancy, for diagnostics and tests."""
    with _lock:
        return {"entries": len(_entries), "bytes": _total_bytes, "cap": max_bytes()}


def clear() -> None:
    """Drop everything (used between runs and by tests)."""
    global _total_bytes
    with _lock:
        _entries.clear()
        _total_bytes = 0
