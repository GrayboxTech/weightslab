import re
import numpy as np
import torch as th
import logging

from typing import Any
from PIL import Image


__all__ = [
    "_detect_dataset_split",
    "_is_scalarish",
    "_is_dense_array",
    "_to_numpy_safe",
    "_downsample_nn",
    "_filter_columns_by_patterns",
    "_matches_pattern",
]


# load global logger
logger = logging.getLogger(__name__)


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


def _to_numpy_safe(x):
    if isinstance(x, (int, float)):
        return np.array([x])

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


def get_mask(raw, dataset=None, dataset_index=None, raw_data=None):
    # Check if prediction_raw is a numpy array (could be bboxes)
    if isinstance(raw, np.ndarray) and (raw.ndim == 2 or raw.ndim == 3) and raw.shape[-1] >= 4:
        # raw appears to be bboxes (N, 4+) format
        # Get the item (image) to determine mask dimensions
        raw_data = dataset[dataset_index] if dataset is not None and dataset_index is not None else raw_data
        if raw_data is None:
            return raw

        # Extract the item (first element of the tuple)
        if isinstance(raw_data, tuple):
            item = raw_data[0]
        else:
            item = raw_data

        # Convert item to numpy to get shape
        item_np = _to_numpy_safe(item)
        if item_np is not None:
            # Determine height and width from item
            if item_np.ndim == 3:
                # Channels-first format: (C, H, W)
                if item_np.shape[0] < item_np.shape[1]:
                    _, height, width = item_np.shape
                else:
                    # Channels-last format: (H, W, C)
                    height, width, _ = item_np.shape
            elif item_np.ndim == 2:
                # Grayscale: (H, W)
                height, width = item_np.shape
            else:
                # Cannot determine dimensions
                return raw

            # Generate segmentation map from bboxes
            segmentation_map = np.zeros((height, width), dtype=np.int64)

            # Return segmentation map directly if it matches raw shape
            if segmentation_map.shape == raw.shape[-2:]:  # B, C, H, W
                return raw

            # Generate segmentation map from bboxes
            raw = raw[0] if raw.ndim == 3 else raw  # Handle batch dimension if present
            for bbox_data in raw:
                x1, y1, x2, y2 = bbox_data[:4].astype(int)
                # Extract class id if available, otherwise use 1
                class_id = int(bbox_data[4]) if len(bbox_data) > 4 else 1

                # Clip to valid image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width))
                y2 = max(0, min(y2, height))

                # Fill the bounding box region
                if x2 > x1 and y2 > y1:
                    segmentation_map[y1:y2, x1:x2] = class_id

            return segmentation_map

    return raw


def load_label(dataset, sample_id):
    """Load label/target from dataset at given index.

    Arguments:
        dataset: The dataset object to load from.
        sample_id: The sample ID to load the label for.
    
    Expected dataset patterns:
    - dataset[index] -> (data, label)
    - dataset[index] -> (data, uids, label)
    - dataset[index] -> (data, uids, label, metadata) with metadata containing


    Returns the label in its native format (int, array, etc.).
    """
    # Get index from sample_id
    try:
        index = dataset.get_index_from_sample_id(sample_id)
    except (KeyError, ValueError, IndexError):
        logger.debug(f"Sample ID {sample_id} not found in current dataset. Likely a ghost record from a previous run.")
        return None

    # Get dataset wrapper if exists
    wrapped = getattr(dataset, "wrapped_dataset", dataset)

    # Try common dataset patterns first
    if hasattr(wrapped, '__getitem__'):
        data = wrapped[index]

        if isinstance(data, (list, tuple)):
            if len(data) == 1:
                return None  # Only data, no label
            elif len(data) == 2:  # if len==2, only data and uid, no extra info
                return None  # No label, only data and uid
            elif len(data) == 3:  # if len==3, data, uids, label, no extra info
                label = _to_numpy_safe(data[2])  # Third element is typically the label
                label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=data)
            elif len(data) > 3:  # if len>3, data, uids, label, classes, extra info
                if len(data) == 4:
                    label = _to_numpy_safe(data[2])  # Third element is typically the label
                    label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=data)
                    metadata = data[3]
                    classes = _to_numpy_safe(metadata['classes']) if isinstance(metadata, dict) and 'classes' in metadata else None
                    if classes is not None:
                        label = _to_numpy_safe(data[2])  # Second element is typically the label
                        # Concat label with classes if available (bbox detection, i.e., (4,) -> (5,) with class id)
                        label = np.concatenate([label, classes[..., None]], axis=1)
                    else:
                        label = _to_numpy_safe(data[2])  # Second element is typically the label
                else:
                    label = _to_numpy_safe(data[2])  # Third element is typically the label
                    label = get_mask(label, dataset=wrapped, dataset_index=index, raw_data=data)
                    metadata = data[3:]
            return label[0] if label.ndim == 1 and label.shape[0] == 1 else label
    return None


def load_metadata(dataset, sample_id):
    """Load metadata from dataset at given index.

    Arguments:
        dataset: The dataset object to load from.
        sample_id: The sample ID to load the label for.
    
    Expected dataset patterns:
    - dataset[index] -> (data, label)
    - dataset[index] -> (data, uids, label)
    - dataset[index] -> (data, uids, label, metadata) with metadata containing


    Returns the metadata in its native format (dict, etc.).
    """
    # Get index from sample_id
    try:
        index = dataset.get_index_from_sample_id(sample_id)
    except (KeyError, ValueError, IndexError):
        logger.debug(f"Sample ID {sample_id} not found in current dataset. Likely a ghost record from a previous run.")
        return None

    # Get dataset wrapper if exists
    wrapped = getattr(dataset, "wrapped_dataset", dataset)

    # Try common dataset patterns first
    if hasattr(wrapped, '__getitem__'):
        data = wrapped[index]

        if isinstance(data, (list, tuple)):
            if len(data) == 1:
                return None  # Only data, no metadata
            elif len(data) == 2:  # if len==2, only data and uid, no extra info
                return None  # No metadata, only data and uid
            elif len(data) == 3:  # if len==3, data, uids, label, no extra info
                return None  # No metadata, only data, uid, and label
            elif len(data) > 3:  # if len>3, data, uids, label, classes, extra info
                if len(data) == 4:
                    metadata = data[3]
                else:
                    metadata = data[3:]
            return metadata
    return None


def load_raw_image(dataset, index) -> Image.Image:
    """Load raw image from dataset at given index."""

    def to_uint8(np_img: np.ndarray) -> np.ndarray:
        """Convert an array to uint8 safely for PIL.
        - Floats in [0,1] -> scale by 255
        - Values outside [0,255] -> clip
        - Cast to uint8
        """
        if not isinstance(np_img, np.ndarray):
            np_img = np.array(np_img)

        if np_img.dtype == np.uint8:
            return np_img

        if np.issubdtype(np_img.dtype, np.floating):
            min_v = float(np.nanmin(np_img)) if np_img.size else 0.0
            max_v = float(np.nanmax(np_img)) if np_img.size else 1.0
            if max_v <= 128.0:  # TODO fix convert image type
                np_img = (np_img - min_v) / (max_v - min_v) * 255.0
        # Clip to valid byte range then cast
        np_img = np.clip(np_img, 0, 255)
        return np_img.astype(np.uint8)

    # Get dataset wrapper if exists
    wrapped = getattr(dataset, "wrapped_dataset", dataset)

    if hasattr(wrapped, "images") and isinstance(wrapped.images, list):
        img_path = wrapped.images[index]
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, "files") and isinstance(wrapped.files, list):
        img_path = wrapped.files[index]
        img = Image.open(img_path)
        return img.convert("RGB")
    elif hasattr(wrapped, '__getitem__') or hasattr(wrapped, "data") or hasattr(wrapped, "dataset"):
        np_img = wrapped[index]
        if isinstance(np_img, (list, tuple)):
            np_img = np_img[0]
        if hasattr(np_img, 'numpy'):
            np_img = np_img.numpy()
        if np_img.ndim == 2:
            np_img = to_uint8(np_img)
            return Image.fromarray(np_img, mode="L")
        elif np_img.ndim == 3:
            # Convert from channel-first (C, H, W) to channel-last (H, W, C) for PIL
            if np_img.shape[0] in [1, 3, 4] and np_img.shape[0] != np_img.shape[-1]:
                np_img = np.transpose(np_img, (1, 2, 0))
            np_img = to_uint8(np_img)
            # Choose mode based on channels
            if np_img.shape[-1] == 1:
                return Image.fromarray(np_img[..., 0], mode="L")
            if np_img.shape[-1] == 3:
                return Image.fromarray(np_img, mode="RGB")
            if np_img.shape[-1] == 4:
                return Image.fromarray(np_img, mode="RGBA")
            raise ValueError(f"Unsupported channel count: {np_img.shape[-1]}")
        else:
            raise ValueError(f"Unsupported image shape: {np_img.shape}")
    elif hasattr(wrapped, "samples") or hasattr(wrapped, "imgs"):
        if hasattr(wrapped, "samples"):
            img_path, _ = wrapped.samples[index]
        else:
            img_path, _ = wrapped.imgs[index]
        img = Image.open(img_path)
        return img.convert("L") if img.mode in ["1", "L", "I;16", "I"] else img.convert("RGB")
    else:
        raise ValueError("Dataset type not supported for raw image extraction.")


def load_uid(dataset, sample_id):
    """Load uid from dataset at given index.

    Arguments:
        dataset: The dataset object to load from.
        sample_id: The sample ID to load the label for.
    
    Expected dataset patterns:
    - dataset[index] -> (data, label)
    - dataset[index] -> (data, uids, label)
    - dataset[index] -> (data, uids, label, metadata) with metadata containing


    Returns the metadata in its native format (dict, etc.).
    """
    # Get index from sample_id
    try:
        index = dataset.get_index_from_sample_id(sample_id)
    except (KeyError, ValueError, IndexError):
        logger.debug(f"Sample ID {sample_id} not found in current dataset. Likely a ghost record from a previous run.")
        return None

    # Get dataset wrapper if exists
    wrapped = getattr(dataset, "wrapped_dataset", dataset)

    # Try common dataset patterns first
    if hasattr(wrapped, '__getitem__'):
        data = wrapped[index]

        if isinstance(data, (list, tuple)):
            if len(data) == 1:
                return None  # Only data, no metadata
            elif len(data) >= 2:  # if len==2, only data and uid, no extra info
                return data[1]  # Second element is typically the uid 
    return None