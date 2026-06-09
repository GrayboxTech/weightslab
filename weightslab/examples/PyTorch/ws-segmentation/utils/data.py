import os
import torch
import numpy as np

from torchvision import transforms

from torch.utils.data import Dataset

from PIL import Image


# =============================================================================
# BDD100k segmentation dataset
# =============================================================================
class BDD100kSegDataset(Dataset):
    """
    Uses your existing layout:

      data/BDD100k_reduced/
        images_1280x720/
          train/
          val/
        bdd100k_labels_dac_daa_lls_lld_curbs/
          train/
          val/

    Assumes image & label share basename (e.g. 0001.jpg / 0001.png).
    """

    def __init__(
        self,
        root,
        split="train",
        num_classes=6,
        ignore_index=255,
        image_size=256,
        max_samples=None
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.task_type = "segmentation"

        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        image_files = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_files = sorted(set(image_files))[:max_samples] if max_samples is not None else sorted(set(image_files))  # Optionally limit number of samples for faster testing

        self.images = []
        self.masks = []
        for fname in image_files:
            img_path = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)
            lbl_name = base + ".png"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(lbl_path):
                self.images.append(img_path)
                self.masks.append(lbl_path)

        if len(self.images) == 0:
            raise RuntimeError(f"No image/label pairs found in {img_dir} / {lbl_dir}")

        # This is used by load_raw_image in DataService / trainer_tools
        # so exposing .images is enough.
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=image_size,
                    interpolation=Image.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize = transforms.Resize(
            size=image_size,
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
            IMPORTANT: returns (item, uid, target) only.
                - item: transformed input (e.g. image tensor)
                - uid: unique identifier for the sample (e.g. filename)
                - target: transformed label/target (e.g. mask tensor)
                - metadata: optional dict with any additional info (e.g. original paths)
        """
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):

        img_path = self.images[idx]
        mask_path = self.masks[idx]
        uid = os.path.basename(img_path)

        metadata = {
            'img_path': img_path,
            'mask_path': mask_path,
        } if include_metadata else None

        # Process images
        img_t = None
        if include_images:
            img = Image.open(img_path).convert("RGB")
            img_t = self.image_transform(img)

        # Process labels/masks
        mask_t_instances = list()
        mask_t = None
        if include_labels:
            mask = Image.open(mask_path)
            mask_r = self.mask_resize(mask)
            mask_np = np.array(mask_r, dtype=np.int64)
            mask_t = torch.from_numpy(mask_np)  # [H, W] int64

            # Format labels to register multiple instance_ids
            lbl_max = mask_t.max().item()
            for i in range(1, lbl_max + 1):
                m = torch.zeros_like(mask_t)
                m[mask_t == i] = i  # Assign class ID as instance ID for simplicity; if set to 1, all instances of the same class would be merged...
                mask_t_instances.append(m)
        return img_t, uid, mask_t_instances, metadata


def seg_collate(batch):
    """Collate WL per-sample tuples for instance-segmentation.

    Each item is ``(img, uid, instances, metadata)`` where ``instances`` is a
    LIST of per-instance mask tensors ([H, W], pixel value = class id). The
    default torch collate cannot batch variable-length instance lists, so we
    keep them as a Python list (one entry per sample). Empty instances (all
    background) are filtered out so every kept instance is a real annotation.

    Returns:
        images:  FloatTensor [B, C, H, W]
        ids:     list[str] of length B
        labels:  list[B] where labels[s] is a list of instance mask tensors
        metas:   list[B] of metadata dicts
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    labels = [
        [m for m in (b[2] or []) if torch.as_tensor(m).max() > 0] if isinstance(b[2], list) else b[2]
        for b in batch
    ]
    metas = [b[3] if len(b) > 3 else None for b in batch]
    return images, ids, labels, metas
