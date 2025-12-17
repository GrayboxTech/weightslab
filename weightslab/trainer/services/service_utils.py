import io
import numpy as np
from PIL import Image


def load_raw_image(dataset, index):
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
    elif hasattr(wrapped, "data") or hasattr(wrapped, "dataset"):
        if hasattr(wrapped, "dataset"):
            wrapped_data = wrapped.dataset.base.data if hasattr(wrapped.dataset, "base") else wrapped.dataset
        else:
            wrapped_data = wrapped.data
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
