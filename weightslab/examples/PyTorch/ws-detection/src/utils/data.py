import numpy as np
import torch as th

from copy import deepcopy
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.ops import xywh2xyxy, xywhn2xyxy


class YOLODatasetWL(YOLODataset):
    """YOLODataset that also speaks WL's preview protocol via get_items()."""

    @property
    def class_names(self):
        return self.data.get("names")

    @property
    def num_classes(self):
        return len(self.data.get("names") or {})

    # Explicit task; bypasses WL's label-shape heuristic which falls back to
    # 'classification' for images with zero GT boxes.
    task_type = "detection"

    def __getitem__(self, idx):
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def fast_get_label(self, i):
        """No image decode label access for WL ledger init."""
        lab = self.labels[i]
        h0, w0 = lab["shape"]
        new = self.imgsz
        r = min(new / h0, new / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        padw, padh = (new - nw) / 2, (new - nh) / 2
        bboxes_lb = xywhn2xyxy(lab["bboxes"], w=nw, h=nh, padw=padw, padh=padh) / float(new)

        # Unified 6-col bbox: [x1, y1, x2, y2, class_id, confidence]; GT confidence = 1.0.
        n = bboxes_lb.shape[0]
        cls = lab["cls"].reshape(-1, 1).astype(np.float32)
        target = list(
            np.concatenate([bboxes_lb.astype(np.float32), cls, np.ones((n, 1), dtype=np.float32)], axis=1)
            if n > 0 else np.zeros((0, 6), dtype=np.float32)
        )
        brightness = th.rand(1).item()  # Dummy brightness for demo purposes.
        metad = {
            "img_path": lab["im_file"],
            "cls": lab["cls"],
            "weather": "sunny" if brightness > 0.5 else "cloudy",
        }
        return None, str(i), target, metad

    def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
        data = super().__getitem__(i)
        image = data['img'] if include_images else None
        metadata = (
            {
                'img_path': data['im_file'],
                'cls': data['cls'],
                'batch': data,
                'weather': "sunny" if th.rand(1).item() > 0.5 else "cloudy"
            }
            if include_metadata else {}
        )
        labels = None

        if include_labels:
            # Same 6-col schema as fast_get_label.
            xyxy = xywh2xyxy(data['bboxes'])
            xyxy_np = (xyxy.detach().cpu().numpy().astype(np.float32)
                       if hasattr(xyxy, 'detach') else np.asarray(xyxy, dtype=np.float32))
            cls = np.asarray(data['cls']).reshape(-1, 1).astype(np.float32)
            n = xyxy_np.shape[0]
            labels = (
                np.concatenate([xyxy_np, cls, np.ones((n, 1), dtype=np.float32)], axis=1)
                if n > 0 else np.zeros((0, 6), dtype=np.float32)
            )

        return image, str(i), labels, metadata


def _wl_yolo_collate(batchs):
    """Repack WL's per-sample tuples into a YOLO batch."""
    imgs = None
    labels, uids, im_files = [], [], []
    all_cls, all_bboxes, all_batch_idx = [], [], []
    meta = None

    for n, (img, uid, label, item_meta) in enumerate(batchs):
        imgs = img[None] if imgs is None else th.cat([imgs, img[None]])
        labels.append(label[None])
        uids.append(uid)

        if meta is None:
            meta = deepcopy(item_meta)
            meta['img_path'] = [meta['img_path']]
        else:
            meta['img_path'].append(item_meta['img_path'])

        if 'batch' in item_meta:
            buf = item_meta['batch']
            im_files.append(buf['im_file'])
            bboxes_i = buf['bboxes'].reshape(-1, 4)
            all_cls.append(buf['cls'].reshape(-1, 1))
            all_bboxes.append(bboxes_i)
            all_batch_idx.append(th.full((bboxes_i.shape[0], 1), n, dtype=th.float32))

    if meta is not None and 'batch' in meta:
        meta['batch'] = {
            'im_file': im_files,
            'img': imgs,
            'cls': th.cat(all_cls, dim=0),
            'bboxes': th.cat(all_bboxes, dim=0),
            'batch_idx': th.cat(all_batch_idx, dim=0),
        }

    return imgs, uids, labels, meta
