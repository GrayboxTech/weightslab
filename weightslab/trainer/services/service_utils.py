import torch as th
import numpy as np

from PIL import Image


def _to_numpy_safe(x):
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


def get_mask(raw, dataset, dataset_index):
    # Check if prediction_raw is a numpy array (could be bboxes)
    if isinstance(raw, np.ndarray) and (raw.ndim == 2 or raw.ndim == 3) and raw.shape[-1] >= 4:
        # raw appears to be bboxes (N, 4+) format
        # Get the item (image) to determine mask dimensions
        raw_data = dataset[dataset_index]

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

    # Not bounding boxes, return as is
    return raw


def load_label(dataset, sample_id):
    """Load label/target from dataset at given index.

    Returns the label in its native format (int, array, etc.).
    """
    # Get dataset wrapper if exists
    wrapped = getattr(dataset, "wrapped_dataset", dataset)
    index = dataset.get_index_from_sample_id(sample_id)

    # Try common dataset patterns
    if hasattr(wrapped, '__getitem__'):
        data = wrapped[index]
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            classes = _to_numpy_safe(data[3]) if len(data) >= 4 else None
            if classes is not None:
                label = _to_numpy_safe(data[2])  # Second element is typically the label

                # Concat label with classes if available (detection)
                if classes is not None:
                    label = np.concatenate([label, classes[..., None]], axis=1)
            else:
                label = _to_numpy_safe(data[1])  # Second element is typically the label
            return label

    # Try targets/labels attribute
    if hasattr(wrapped, "targets"):
        label = wrapped.targets[index]
        if hasattr(label, 'numpy'):
            label = label.numpy()
        if hasattr(label, 'item') and hasattr(label, 'shape') and label.shape == ():
            label = label.item()
        return label

    if hasattr(wrapped, "labels"):
        label = wrapped.labels[index]
        if hasattr(label, 'numpy'):
            label = label.numpy()
        if hasattr(label, 'item') and hasattr(label, 'shape') and label.shape == ():
            label = label.item()
        return label

    # Try samples/imgs pattern (returns tuple of path, label)
    if hasattr(wrapped, "samples"):
        _, label = wrapped.samples[index]
        return label

    if hasattr(wrapped, "imgs"):
        _, label = wrapped.imgs[index]
        return label

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
            if max_v <= 1.0 and min_v >= 0.0:
                np_img = np_img * 255.0
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
        if hasattr(wrapped, "dataset"):
            wrapped_data = wrapped.dataset.base.data if hasattr(wrapped.dataset, "base") else wrapped.dataset
        elif hasattr(wrapped, "data"):
            wrapped_data = wrapped.data
        else:
            wrapped_data = wrapped
        np_img = wrapped_data[index]
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
