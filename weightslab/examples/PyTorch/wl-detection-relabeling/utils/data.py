import os
import ssl
import zipfile
import urllib.request

import numpy as np
import torch

from torchvision import transforms

from torch.utils.data import Dataset

from PIL import Image


# =============================================================================
# Penn-Fudan Pedestrian detection dataset
# =============================================================================
# A small, real object-detection dataset (~170 photos, one class: "person").
# It ships per-instance segmentation masks; we derive an axis-aligned bounding
# box per pedestrian from each mask. Downloaded + extracted on first use.
#
# On-disk layout after extraction:
# <root>/PennFudanPed/
# PNGImages/FudanPed00001.png ...
# PedMasks/FudanPed00001_mask.png ... # pixel value k = k-th pedestrian, 0 = bg
#
# WL renders detection targets/predictions from a per-sample [N, 6] array
# ``[x1, y1, x2, y2, class_id, confidence]`` normalized to [0, 1] (GT conf = 1.0)
# — see ``get_items`` below and ``data_service.py`` (task_type == "detection").

CLASS_NAMES = ["person"]

_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

# ImageNet statistics — the MobileNetV3 backbone is pretrained with these, so we
# normalize model inputs the same way. (The UI still shows the original PNG.)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def download_penn_fudan(root):
    """Download + extract Penn-Fudan into <root>/PennFudanPed (idempotent)."""
    base = os.path.join(root, "PennFudanPed")
    if os.path.isdir(os.path.join(base, "PNGImages")):
        return base

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "PennFudanPed.zip")

    if not os.path.exists(zip_path):
        print(f"[data] Downloading Penn-Fudan dataset to {zip_path} ...", flush=True)
        try:
            urllib.request.urlretrieve(_URL, zip_path)
        except Exception as e:
            # Some corporate environments break TLS verification - retry unverified.
            print(f"[data] TLS verification failed ({e}); retrying without verification.", flush=True)
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(_URL, context=ctx) as resp, open(zip_path, "wb") as fh:
                fh.write(resp.read())

    print("[data] Extracting Penn-Fudan ...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return base


def _boxes_from_mask(mask_path):
    """Derive one bbox per pedestrian from a Penn-Fudan instance mask.

    Returns (boxes_px [N, 4] int xyxy, height, width). Background (0) skipped.
    """
    mask = np.array(Image.open(mask_path))
    h, w = mask.shape[:2]
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0] # drop background

    boxes = []
    for oid in obj_ids:
        ys, xs = np.where(mask == oid)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), h, w


class PennFudanDetectionDataset(Dataset):
    """Pedestrian bounding-box detection over the Penn-Fudan images.

    Args:
        root: directory to download/extract the dataset into.
        split: "train" or "val" (deterministic split of the 170 images).
        image_size: square resize fed to the model.
        val_fraction: fraction of images held out for validation.
        max_samples: optional cap on the split size (for quick runs).
    """

    def __init__(
        self,
        root,
        split="train",
        num_classes=1,
        image_size=256,
        val_fraction=0.2,
        max_samples=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.image_size = image_size
        # Explicit task type; bypasses WL's label-shape heuristic so bboxes are
        # rendered as detection overlays (not mistaken for classification).
        self.task_type = "detection"
        self.class_names = CLASS_NAMES[:num_classes]

        base = download_penn_fudan(root)
        img_dir = os.path.join(base, "PNGImages")
        mask_dir = os.path.join(base, "PedMasks")

        all_imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))

        # Deterministic train/val split: every k-th image goes to val.
        k = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
        if split == "val":
            selected = all_imgs[::k]
        else:
            val_set = set(all_imgs[::k])
            selected = [f for f in all_imgs if f not in val_set]

        selected = selected[:max_samples] if max_samples != None else selected

        self.images = []
        self.masks = []
        for fname in selected:
            base_name, _ = os.path.splitext(fname)
            mask_path = os.path.join(mask_dir, base_name + "_mask.png")
            if os.path.exists(mask_path):
                self.images.append(os.path.join(img_dir, fname))
                self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(f"No image/mask pairs found under {base}")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.images)

    def _load_boxes(self, mask_path):
        """Read mask → [N, 6] float32 = [x1, y1, x2, y2, cls=0, conf=1.0] (normalized)."""
        boxes_px, h, w = _boxes_from_mask(mask_path)
        if boxes_px.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)
        norm = boxes_px.copy()
        norm[:, [0, 2]] /= float(w)
        norm[:, [1, 3]] /= float(h)
        n = norm.shape[0]
        cls = np.zeros((n, 1), dtype=np.float32) # single class: person
        conf = np.ones((n, 1), dtype=np.float32)
        return np.concatenate([norm, cls, conf], axis=1).astype(np.float32)

    def __getitem__(self, idx):
        """Returns (item, uid, target, metadata).

        - item: normalized image tensor [C, H, W]
        - uid: unique sample id (string)
        - target: [N, 6] float32 = [x1, y1, x2, y2, class_id, confidence]
        - metadata: dict with source paths
        """
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        uid = os.path.splitext(os.path.basename(img_path))[0]

        metadata = {
            "img_path": img_path,
            "mask_path": mask_path,
        } if include_metadata else None

        img_t = None
        if include_images:
            img = Image.open(img_path).convert("RGB")
            img_t = self.image_transform(img)

        target = None
        if include_labels:
            target = self._load_boxes(mask_path)

        return img_t, uid, target, metadata


def det_collate(batch):
    """Collate WL per-sample tuples for object detection.

    A sample owns a variable number of boxes, so targets cannot be stacked. We
    keep them as a Python list (one [N_i, 6] tensor per sample), exactly the
    layout WL's per-instance helpers expect (``targets[s]`` iterates that
    sample's boxes in annotation order).

    Returns:
        images: FloatTensor [B, C, H, W]
        ids: list[str] of length B
        targets: list[B] of [N_i, 6] float tensors ([x1, y1, x2, y2, cls, conf])
        metas: list[B] of metadata dicts
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    targets = [
        torch.as_tensor(b[2], dtype=torch.float32)
        if not isinstance(b[2], torch.Tensor) else b[2].float()
        for b in batch
    ]
    metas = [b[3] if len(b) > 3 else None for b in batch]
    return images, ids, targets, metas
