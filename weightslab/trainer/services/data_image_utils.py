"""
data_image_utils — Image encoding, mask compression, and proto helpers for gRPC data serving.

Extracted from data_service.py to keep image-specific logic separate from the
DataService orchestration class. All functions here are pure (stateless) and
safe to call from any thread.
"""

import io
import logging

import numpy as np

from PIL import Image
import weightslab.proto.experiment_service_pb2 as pb2

logger = logging.getLogger(__name__)


# =============================================================================
# RLE Encoding for sparse masks (uint8 class-ID masks)
# =============================================================================
# Format: sequence of (value: uint8, run_length: uint16-LE) pairs = 3 bytes each.
# For a 720×1280 mask with ~6 classes the output is typically 1-10 KB vs 921 KB raw.
# The frontend decodes this back into a flat Uint8Array for the WebGL renderer.

def rle_encode_mask(mask_flat: np.ndarray) -> bytes:
    """Run-length encode a flat uint8 mask array.

    Returns bytes in the format: [value(1B), run_length(2B little-endian)] repeating.
    Maximum run length per segment is 65535 (uint16 max).

    Uses NumPy vectorised operations for ~100x speedup over the Python loop version
    (0.5ms vs 400ms for a 720×1280 mask).
    """
    if mask_flat.size == 0:
        return b""

    # Find where values change (vectorised)
    diff = np.diff(mask_flat)
    change_indices = np.flatnonzero(diff)
    # Run starts: [0, change_idx+1, change_idx+1, ...]
    starts = np.empty(len(change_indices) + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = change_indices + 1
    # Run lengths
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = mask_flat.size
    lengths = ends - starts # numpy int array
    values = mask_flat[starts] # numpy uint8 array

    # Split any runs > 65535 into multiple segments
    out_vals: list[int] = []
    out_lens: list[int] = []
    for v, ln in zip(values, lengths):
        v_int = int(v)
        ln_int = int(ln)
        while ln_int > 65535:
            out_vals.append(v_int)
            out_lens.append(65535)
            ln_int -= 65535
        out_vals.append(v_int)
        out_lens.append(ln_int)

    # Pack all segments at once using a structured array (zero Python-loop packing)
    n = len(out_vals)
    buf = np.empty(n, dtype=np.dtype([('val', 'u1'), ('run', '<u2')]))
    buf['val'] = out_vals
    buf['run'] = out_lens
    return buf.tobytes()


def rle_decode_mask(data: bytes) -> np.ndarray:
    """Decode a RLE-encoded mask back to a flat uint8 numpy array.

    Expects bytes in the format: [value(1B), run_length(2B little-endian)] repeating,
    as produced by rle_encode_mask.
    """
    if not data:
        return np.empty(0, dtype=np.uint8)

    n = len(data) // 3
    buf = np.frombuffer(data[:n * 3], dtype=np.dtype([('val', 'u1'), ('run', '<u2')]))
    # np.repeat handles large masks correctly (no uint16 overflow) and is fully vectorised.
    return np.repeat(buf['val'], buf['run'].astype(np.intp))


def create_data_stat(name, stat_type, shape=None, value=None, value_string="", thumbnail=b""):
    """Helper to create a DataStat proto with all fields properly initialized.

    Args:
        name: Stat name
        stat_type: Type string (scalar, array, string, bytes, rle_mask, etc.)
        shape: List of shape dimensions
        value: List of float values
        value_string: String value
        thumbnail: Bytes object for thumbnail / encoded data (default empty bytes)

    Returns:
        pb2.DataStat: Properly initialized DataStat
    """
    return pb2.DataStat(
        name=name,
        type=stat_type,
        shape=shape or [],
        value=value or [],
        value_string=value_string,
        thumbnail=thumbnail,
    )


def generate_thumbnail(pil_image, max_size=(128, 128), quality=85):
    """Generate a JPEG thumbnail from a PIL image.

    Args:
        pil_image: PIL Image object
        max_size: Max dimensions (width, height) for thumbnail
        quality: JPEG quality (1-95)

    Returns:
        bytes: JPEG thumbnail as bytes
    """
    try:
        thumb = pil_image.copy()
        thumb.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed (JPEG doesn't support RGBA)
        if thumb.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', thumb.size, (255, 255, 255))
            if thumb.mode == 'P':
                thumb = thumb.convert('RGBA')
            if thumb.mode in ('RGBA', 'LA'):
                background.paste(thumb, mask=thumb.split()[-1])
                thumb = background
        elif thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')

        buffer = io.BytesIO()
        thumb.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Failed to generate thumbnail: {e}")
        return b""


def encode_image_webp(pil_image, quality=70, method=2):
    """Encode a PIL image as WebP bytes.

    Args:
        pil_image: PIL Image object (any mode)
        quality: WebP quality (1-100)
        method: WebP encoding method (0=fast, 6=slow/best)

    Returns:
        bytes: WebP-encoded image
    """
    try:
        img = pil_image.convert('RGB') if pil_image.mode not in ('RGB', 'RGBA') else pil_image
        buf = io.BytesIO()
        img.save(buf, format='WEBP', quality=quality, method=method)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Failed to encode WebP: {e}")
        return b""


# =============================================================================
# Tabular inputs (1-D feature vectors)
# =============================================================================
# Tabular samples have no image — the model input IS a 1-D feature vector. We
# transmit the actual feature values losslessly in the DataStat ``value`` field
# (type ``"vector"``), and attach a small heatmap so image-only grids still show
# a cell. The List Exploration view reads the per-feature metadata columns; a
# tabular-aware grid can read ``value`` directly.

def looks_like_tabular(np_img) -> bool:
    """True when the sample input is a 1-D feature vector (no spatial dims)."""
    return np_img is not None and getattr(np_img, "ndim", None) == 1


def render_tabular_heatmap(vec: np.ndarray, cell: int = 16, quality: int = 60):
    """Render a 1-D feature vector as a small square grayscale heatmap.

    Returns ``(webp_bytes, [h, w, c])``. Values are min-max normalised for
    display only; the exact values travel in the DataStat ``value`` field.
    """
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = int(v.size)
    if n == 0:
        return b"", [0, 0, 0]
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side, dtype=np.float32)
    padded[:n] = v
    grid = padded.reshape(side, side)
    mn, mx = float(np.nanmin(grid)), float(np.nanmax(grid))
    norm = (grid - mn) / (mx - mn + 1e-8)
    img8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    try:
        pil = Image.fromarray(img8, mode="L").resize(
            (side * cell, side * cell), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="WEBP", quality=quality, method=1)
        data = buf.getvalue()
        buf.close()
        return data, [side * cell, side * cell, 3]
    except Exception as e:  # pragma: no cover - display-only fallback
        logger.warning(f"Failed to render tabular heatmap: {e}")
        return b"", [side, side, 1]


def build_tabular_raw_data_stat(vec: np.ndarray):
    """Build the ``raw_data`` DataStat for a tabular (1-D) input.

    The feature values are carried losslessly in ``value`` (type ``"vector"``,
    ``shape=[N]``); a heatmap rides along in ``thumbnail`` for display only.
    """
    v = np.asarray(vec, dtype=np.float32).ravel()
    values = [float(x) for x in v]
    thumb, _shape = render_tabular_heatmap(v)
    return create_data_stat(
        "raw_data", "vector", shape=[len(values)], value=values, thumbnail=thumb)


def resize_mask_nearest(mask_arr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a 2D+ mask array using nearest-neighbour (preserves class IDs).

    Args:
        mask_arr: numpy array with shape (H, W, ...) or (H, W)
        target_w: target width
        target_h: target height

    Returns:
        numpy uint8 array of shape (target_h, target_w)
    """
    src = mask_arr.astype(np.uint8) if mask_arr.ndim == 2 else mask_arr.astype(np.uint8)[:, :, 0]
    pil = Image.fromarray(src)
    pil = pil.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.asarray(pil, dtype=np.uint8)
