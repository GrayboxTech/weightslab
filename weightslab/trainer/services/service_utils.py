import io
import numpy as np
from PIL import Image


def load_raw_image(dataset, index):
    """Load raw image from dataset at given index."""
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
            return Image.fromarray(np_img.astype(np.uint8), mode="L")
        elif np_img.ndim == 3:
            # Convert from channel-first (C, H, W) to channel-last (H, W, C) for PIL
            if np_img.shape[0] in [1, 3, 4]:  # Likely channel-first
                np_img = np.transpose(np_img, (1, 2, 0))
            return Image.fromarray(np_img.astype(np.uint8), mode="RGB")
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
