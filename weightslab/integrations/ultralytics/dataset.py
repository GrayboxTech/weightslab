"""WL-aware Ultralytics YOLODataset.

Subclass `WLAwareDataset` of UL's `YOLODataset` that:
  * Returns the WL 4-tuple `(image, str(idx), labels, metadata)` from
    `__getitem__` so the WL `DataSampleTrackingWrapper` can wrap it and
    replace `str(idx)` with the wrapper's globally-unique uid.
  * Exposes `class_names` / `num_classes` for studio's bbox renderer.
  * Sets `task_type = "detection"` explicitly so the WL ledger init doesn't
    misclassify zero-GT samples as classification.
  * Provides `fast_get_label(i)` so WL's ledger init skips image decode —
    with a PIL fallback when `lab["shape"]` is missing (val sets whose
    labels.cache predates YOLO's shape population).
"""
from __future__ import annotations

import numpy as np

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.ops import xywh2xyxy, xywhn2xyxy


class WLAwareDataset(YOLODataset):
    """YOLODataset that also speaks WL's preview protocol via `get_items()`."""

    task_type = "detection"

    @property
    def class_names(self):
        return self.data.get("names")

    @property
    def num_classes(self):
        return len(self.data.get("names") or {})

    def __getitem__(self, idx):
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def fast_get_label(self, i):
        """No-image-decode label access for WL ledger init.

        PIL fallback when `lab["shape"]` is missing (memoized for next call)."""
        lab = self.labels[i]
        shp = lab.get("shape")
        if shp is None:
            from PIL import Image as _PIL
            with _PIL.open(lab["im_file"]) as im:
                w0, h0 = im.size  # PIL: (w, h)
                shp = (h0, w0)
                lab["shape"] = shp  # memoize
        h0, w0 = shp
        new = self.imgsz
        r = min(new / h0, new / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        padw, padh = (new - nw) / 2, (new - nh) / 2
        bboxes_lb = xywhn2xyxy(lab["bboxes"], w=nw, h=nh, padw=padw, padh=padh) / float(new)

        # Unified 6-col bbox: [x1, y1, x2, y2, class_id, confidence=1.0].
        n = bboxes_lb.shape[0]
        cls = lab["cls"].reshape(-1, 1).astype(np.float32)
        target = (
            np.concatenate([bboxes_lb.astype(np.float32), cls, np.ones((n, 1), dtype=np.float32)], axis=1)
            if n > 0 else np.zeros((0, 6), dtype=np.float32)
        )
        return None, str(i), target, {"img_path": lab["im_file"], "cls": lab["cls"]}

    def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
        data = super().__getitem__(i)
        image = data["img"] if include_images else None
        metadata = (
            {"img_path": data["im_file"], "cls": data["cls"], "batch": data}
            if include_metadata else {}
        )
        labels = None

        if include_labels:
            xyxy = xywh2xyxy(data["bboxes"])
            xyxy_np = (
                xyxy.detach().cpu().numpy().astype(np.float32)
                if hasattr(xyxy, "detach")
                else np.asarray(xyxy, dtype=np.float32)
            )
            cls = np.asarray(data["cls"]).reshape(-1, 1).astype(np.float32)
            n = xyxy_np.shape[0]
            labels = (
                np.concatenate([xyxy_np, cls, np.ones((n, 1), dtype=np.float32)], axis=1)
                if n > 0 else np.zeros((0, 6), dtype=np.float32)
            )

        return image, str(i), labels, metadata
