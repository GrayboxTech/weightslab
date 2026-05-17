import numpy as np
import torch as th
import weightslab as wl

from copy import deepcopy
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.ops import xywhn2xyxy


class YOLODatasetWL(YOLODataset):
    """YOLODataset that also speaks WL's preview protocol.

    YOLO's __getitem__ returns a training-shaped dict; WL's preview/UI path
    expects an image-shaped item. `get_items` is the hook WL feature-detects
    and prefers over __getitem__ when present.
    """
    def __init__(
        self,
        cfg: IterableSimpleNamespace,
        img_path: str,
        batch: int,
        data: dict,
        mode: str = "train",
        rect: bool = False,
        stride: int = 32
    ):
        super().__init__(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg, 
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=stride,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

        # Preload labels for WL preview access; super().__getitem__ won't be called in preview mode
        self.get_labels()

    def get_labels(self):
        return super().get_labels()

    @property
    def class_names(self):
        return self.data.get("names")

    @property
    def num_classes(self):
        names = self.data.get("names") or {}
        return len(names)

    # Explicit task declaration; bypasses WL's label-shape heuristic which
    # falls back to 'classification' for images with zero GT boxes.
    task_type = "detection"
    
    def __getitem__(self, idx):
        """Override to return dicts as (dict, uid, _target) tuples for WL."""
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def fast_get_label(self, i):
        """Cheap, no-decode label access (WL ledger contract).

        Reads ultralytics' cached label entry (no pixel I/O) and emits GT in
        letterboxed pixel-xyxy — same coord system criterions.py uses at
        training time. Used by WL during ledger init to avoid the ~10 it/s
        decode path; returns (data, uid, target, metadata) per the contract
        in weightslab/data/data_samples_with_ops.py:315.
        """
        lab = self.labels[i]
        h0, w0 = lab["shape"]
        new = self.imgsz
        r = min(new / h0, new / w0)
        nw, nh = round(w0 * r), round(h0 * r)
        padw, padh = (new - nw) / 2, (new - nh) / 2
        bboxes_lb = xywhn2xyxy(lab["bboxes"], w=nw, h=nh, padw=padw, padh=padh) / float(new)

        # Unified 6-col bbox tensor: [x1, y1, x2, y2, class_id, confidence].
        # For GT, confidence is 1.0 (ground-truth is certain). Predictions populate
        # col 5 with the model's max class probability. Studio's serializer (patched
        # at site-packages data_service.py) extracts class_ids from col 4 and scores
        # from col 5 when shape[-1] >= 6.
        n = bboxes_lb.shape[0]
        cls = lab["cls"].reshape(-1, 1).astype(np.float32)
        conf = np.ones((n, 1), dtype=np.float32)
        target = np.concatenate(
            [bboxes_lb.astype(np.float32), cls, conf], axis=1
        ) if n > 0 else np.zeros((0, 6), dtype=np.float32)

        # Only keys the UI / downstream consumers actually use go in metadata.
        # ori_shape / resized_shape / ratio_pad are dropped here — they fluffed
        # the studio's per-sample stats panel without anyone reading them.
        metadata = {
            "img_path": lab["im_file"],
            "cls": lab["cls"],
        }
        return None, str(i), target, metadata

    
    def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
        data = super().__getitem__(i)
        
        # WL preview expects (img, uid, labels, metadata) tuple; we'll pack everything into metadata and return labels and image optionally to avoid unnecessary overhead in the training loop
        image = None
        labels = None
        metadata = {}

        if include_metadata:
            # ori_shape / resized_shape / num_classes / class_names dropped — UI clutter,
            # not consumed downstream in this project. `batch` is the full ultralytics
            # dict and is kept because the collate function reads from it.
            metadata = {
                'img_path': data['im_file'],
                'cls': data['cls'],
                'batch': data
            }
        if include_images:
            image = data['img'] 
        if include_labels:
            # Unified 6-col bbox tensor: [x1, y1, x2, y2, class_id, confidence].
            # GT confidence is 1.0; matches fast_get_label and the prediction packing
            # in main.py so every storage site shares one schema.
            from ultralytics.utils.ops import xywh2xyxy
            xyxy = xywh2xyxy(data['bboxes'])
            if hasattr(xyxy, 'detach'):
                xyxy_np = xyxy.detach().cpu().numpy().astype(np.float32)
            else:
                xyxy_np = np.asarray(xyxy, dtype=np.float32)
            cls = np.asarray(data['cls']).reshape(-1, 1).astype(np.float32)
            n = xyxy_np.shape[0]
            conf = np.ones((n, 1), dtype=np.float32)
            labels = (np.concatenate([xyxy_np, cls, conf], axis=1)
                      if n > 0 else np.zeros((0, 6), dtype=np.float32))

        # Img, uid, lbls, meta
        return image, str(i), labels, metadata


def _wl_yolo_collate(batchs):
    """Repack WL's per-sample (yolo_dict, uid, _target) tuples into a YOLO batch.

    WL stores UIDs as strings internally; cast to int for the tensor.
    """
    try:
        imgs = None
        labels = list()
        uids = list()
        meta = None
        all_cls = []
        all_bboxes = []
        all_batch_idx = []
        im_files = []

        for n, items in enumerate(batchs):
            img, uid, label, item_meta = items

            # Stack images
            imgs = img[None] if imgs is None else th.cat([imgs, img[None]])
            labels.append(label[None])
            uids.append(uid)

            if meta is None:
                meta = deepcopy(item_meta)
                meta['img_path'] = [meta['img_path']]
            else:
                ditem = deepcopy(item_meta)
                meta['img_path'].append(ditem['img_path'])

            if 'batch' in item_meta:
                buf = item_meta['batch']
                im_files.append(buf['im_file'])

                cls_i = buf['cls'].reshape(-1, 1)       # (n_boxes, 1)
                bboxes_i = buf['bboxes'].reshape(-1, 4) # (n_boxes, 4)
                n_boxes = bboxes_i.shape[0]

                all_cls.append(cls_i)
                all_bboxes.append(bboxes_i)
                all_batch_idx.append(th.full((n_boxes, 1), n, dtype=th.float32))

        if meta is not None and 'batch' in meta:
            meta['batch'] = {
                'im_file': im_files,
                'img': imgs,                                        # (B, C, H, W)
                'cls': th.cat(all_cls, dim=0),                     # (N, 1)
                'bboxes': th.cat(all_bboxes, dim=0),               # (N, 4)
                'batch_idx': th.cat(all_batch_idx, dim=0),         # (N, 1)
            }

    except Exception as e:
        print(f'Error collate function {e}')

    return imgs, uids, labels, meta
