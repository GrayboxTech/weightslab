import re
import numpy as np
import torch as th
from typing import Any

__all__ = [
    "_detect_dataset_split",
    "_is_scalarish",
    "_is_dense_array",
    "_to_numpy_safe",
    "_downsample_nn",
    "_filter_columns_by_patterns",
    "_matches_pattern",
]


# Pre-compile regex patterns for stats matching
_PATTERN_CACHE = {}

def _get_compiled_pattern(pattern: str):
    """Get or compile regex pattern from cache (avoid recompiling on every call)."""
    if pattern not in _PATTERN_CACHE:
        try:
            _PATTERN_CACHE[pattern] = re.compile(pattern)
        except re.error:
            _PATTERN_CACHE[pattern] = None
    return _PATTERN_CACHE[pattern]


def _filter_columns_by_patterns(columns: list, patterns: list) -> list:
    """
    Filter columns by matching against patterns.
    Patterns can be exact strings or regex patterns.
    Optimized with pattern caching to avoid recompiling regex on every call.
    """
    matched_cols = []
    for col in columns:
        for pattern in patterns:
            # Try exact match first (fastest)
            if col == pattern:
                matched_cols.append(col)
                break
            compiled = _get_compiled_pattern(pattern)
            if compiled and compiled.search(col):
                matched_cols.append(col)
                break
    return matched_cols


def _matches_pattern(name: str, patterns: list) -> bool:
    """Check if a name matches any pattern (exact or regex, cached)."""
    for pattern in patterns:
        if name == pattern:
            return True
        compiled = _get_compiled_pattern(pattern)
        if compiled and compiled.search(name):
            return True
    return False


def _detect_dataset_split(ds) -> str:
    """Best-effort split detection for common datasets. Returns actual split name or 'unknown'."""
    train_attr = getattr(ds, 'train', None)
    if train_attr is True:
        return 'train'
    if train_attr is False:
        split = getattr(ds, 'split', None)
        if isinstance(split, str) and split.strip():
            return split.strip().lower()
        return 'test'

    split = getattr(ds, 'split', None)
    if isinstance(split, str) and split.strip():
        return split.strip().lower()

    for attr_name in ['mode', 'subset', 'dataset_type']:
        val = getattr(ds, attr_name, None)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()

    return 'unknown'


def _is_scalarish(x) -> bool:
    if isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_)):
        return True
    if isinstance(x, str):
        return len(x) <= 256
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _is_dense_array(x) -> bool:
    return isinstance(x, np.ndarray) and x.ndim >= 2


def _to_numpy_safe(x: Any):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x)
        except Exception:
            return None
    try:
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return None


def _downsample_nn(arr: np.ndarray, max_hw: int = 96) -> np.ndarray:
    """Downsample 2D/3D arrays using simple striding (nearest-neighbor-like)."""
    if arr.ndim == 2:
        H, W = arr.shape
        scale = max(1, int(np.ceil(max(H, W) / max_hw)))
        return arr[::scale, ::scale]
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[1]:
            C, H, W = arr.shape
            scale = max(1, int(np.ceil(max(H, W) / max_hw)))
            return arr[:, ::scale, ::scale]
        else:
            H, W, C = arr.shape
            scale = max(1, int(np.ceil(max(H, W) / max_hw)))
            return arr[::scale, ::scale, :]
    return arr
