"""
FaceDataset — toy face recognition dataset.

Supported back-ends
-------------------
"olivetti"  Olivetti Faces from sklearn (40 identities, 400 images, 64×64 grey).
            Self-contained; requires only scikit-learn.  Good default toy set.
"lfw"       Labeled Faces in the Wild (LFW) via torchvision.  Downloaded on
            first use to *root*.  Much larger but realistic.
"folder"    Generic ImageFolder layout:  root/{split}/{class_name}/*.jpg

Every sample is returned as:
    (image_tensor: Tensor[C,H,W],
     uid:          str,
     label:        int,
     metadata:     dict)

These map directly onto the (data, uid, target, metadata) convention used
throughout the WeightsLAB kitchen examples so the same training loop works
without modification.
"""

import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple

import torchvision.transforms as T

logger = logging.getLogger(__name__)


# ============================================================
# Dataset
# ============================================================

class FaceDataset(Dataset):
    """Unified face recognition dataset wrapper.

    Args:
        root:                  Download / data root (only used for lfw / folder).
        dataset_type:          One of "olivetti", "lfw", "folder".
        split:                 "train" or "test" (ignored for pre-split sources).
        image_size:            Spatial size; images are resized to (image_size, image_size).
        train_ratio:           Fraction of per-class samples used for training
                               (Olivetti only).
        min_images_per_class:  Classes with fewer samples are discarded.
        transform:             Optional torchvision transform; defaults to
                               Resize → ToTensor → Normalize([0.5], [0.5]).
        seed:                  RNG seed for reproducible train/test splits.
    """

    def __init__(
        self,
        root: str = ".",
        dataset_type: str = "olivetti",
        split: str = "train",
        image_size: int = 64,
        train_ratio: float = 0.8,
        min_images_per_class: int = 2,
        transform=None,
        seed: int = 42,
    ):
        self.dataset_type = dataset_type
        self.split        = split
        self.image_size   = image_size
        self.transform    = transform or self._default_transform(image_size)

        # These are populated by each loader
        self.images: Optional[np.ndarray] = None   # (N, H, W) float [0,1] — Olivetti only
        self.img_paths: Optional[np.ndarray] = None
        self.labels: np.ndarray = np.array([], dtype=np.int64)
        self.num_classes: int = 0

        if dataset_type == "olivetti":
            self._load_olivetti(train_ratio, min_images_per_class, seed)
        elif dataset_type == "lfw":
            self._load_lfw(root, min_images_per_class, split)
        elif dataset_type == "folder":
            self._load_folder(root, split)
        else:
            raise ValueError(
                f"Unknown dataset_type {dataset_type!r}. "
                "Choose from 'olivetti', 'lfw', 'folder'."
            )

        logger.info(
            f"FaceDataset [{dataset_type} / {split}] "
            f"| samples={len(self)} | classes={self.num_classes}"
        )

    # ----------------------------------------------------------
    # Loaders
    # ----------------------------------------------------------

    @staticmethod
    def _default_transform(image_size: int):
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _load_olivetti(self, train_ratio: float, min_images: int, seed: int):
        """Load and split the Olivetti Faces dataset (sklearn)."""
        from sklearn.datasets import fetch_olivetti_faces

        data   = fetch_olivetti_faces(shuffle=True, random_state=seed)
        images = data.images                           # (400, 64, 64) float [0,1]
        labels = data.target.astype(np.int64)

        # Drop classes with insufficient samples
        unique, counts = np.unique(labels, return_counts=True)
        valid_classes  = unique[counts >= min_images]
        mask           = np.isin(labels, valid_classes)
        images, labels = images[mask], labels[mask]

        # Remap labels to a contiguous 0…N-1 range
        mapping = {int(c): i for i, c in enumerate(sorted(valid_classes.tolist()))}
        labels  = np.array([mapping[int(l)] for l in labels], dtype=np.int64)

        # Per-class stratified train/test split
        rng = np.random.RandomState(seed)
        train_idx, test_idx = [], []
        for cls in np.unique(labels):
            idx    = np.where(labels == cls)[0]
            n_train = max(1, int(len(idx) * train_ratio))
            perm   = rng.permutation(len(idx))
            train_idx.extend(idx[perm[:n_train]].tolist())
            test_idx.extend(idx[perm[n_train:]].tolist())

        indices      = train_idx if self.split == "train" else test_idx
        self.images  = images[indices]
        self.labels  = labels[indices]
        self.num_classes = len(mapping)

    def _load_lfw(self, root: str, min_images: int, split: str):
        """Load LFW People via torchvision (downloads on first call)."""
        from torchvision.datasets import LFWPeople

        split_map  = {"train": "train", "test": "test", "val": "10fold"}
        lfw_split  = split_map.get(split, "train")
        ds         = LFWPeople(root=root, split=lfw_split, download=True, transform=None)

        paths, lbls = zip(*ds.imgs)
        lbls        = np.array(lbls, dtype=np.int64)

        # Filter low-shot identities
        unique, counts = np.unique(lbls, return_counts=True)
        valid          = set(unique[counts >= min_images].tolist())
        mask           = np.array([int(l) in valid for l in lbls])

        self.img_paths = np.array(paths)[mask]
        lbls           = lbls[mask]

        mapping        = {int(c): i for i, c in enumerate(sorted(valid))}
        self.labels    = np.array([mapping[int(l)] for l in lbls], dtype=np.int64)
        self.num_classes = len(mapping)

    def _load_folder(self, root: str, split: str):
        """Load from a torchvision ImageFolder directory."""
        from torchvision.datasets import ImageFolder

        split_dir      = os.path.join(root, split)
        ds             = ImageFolder(split_dir)
        paths, lbls    = zip(*ds.imgs)
        self.img_paths = list(paths)
        self.labels    = np.array(lbls, dtype=np.int64)
        self.num_classes = len(ds.classes)

    # ----------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, Dict]:
        """Return (image_tensor, uid, label_id, metadata).

        The (data, uid, target, metadata) signature mirrors the WeightsLAB
        kitchen convention used in the VLA training example.
        """
        label = int(self.labels[idx])

        if self.dataset_type == "olivetti":
            from PIL import Image as PILImage
            img_np  = self.images[idx]              # (H, W) float [0,1]
            img_pil = PILImage.fromarray(
                (img_np * 255).astype(np.uint8), mode="L"
            ).convert("RGB")
        else:
            from PIL import Image as PILImage
            img_pil = PILImage.open(self.img_paths[idx]).convert("RGB")

        image_tensor = self.transform(img_pil)

        uid = f"{self.split}_cls{label:04d}_idx{idx:06d}"
        metadata = {
            "split":        self.split,
            "label_id":     label,
            "idx":          idx,
            "dataset_type": self.dataset_type,
        }
        return image_tensor, uid, label, metadata
