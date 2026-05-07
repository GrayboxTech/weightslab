import torch as th
import weightslab as wl

from copy import deepcopy
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import RANK, colorstr


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

    def __getitem__(self, idx):
        """Override to return dicts as (dict, uid, _target) tuples for WL."""
        return self.get_items(idx, include_metadata=True, include_labels=True, include_images=True)

    def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
        data = super().__getitem__(i)

        # WL preview expects (img, uid, labels, metadata) tuple; we'll pack everything into metadata and return labels and image optionally to avoid unnecessary overhead in the training loop
        image = None
        labels = None
        metadata = {}

        if include_metadata:
            metadata = {
                'img_path': data['im_file'],
                'ori_shape': data['ori_shape'],
                'resized_shape': data['resized_shape'],
                'num_classes': 1,  # Cls stands for classes ?
                'cls': data['cls'],
                'batch': data
            }
        if include_images:
            image = data['img']
        if include_labels:
            labels = data['bboxes']
            # BBx GT are x1 y1 len_x1 len_y1...
            for b in labels:
                b[2:] = b[:2]+b[2:]

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
                meta['ori_shape'] = ditem['ori_shape']
                meta['resized_shape'] = ditem['resized_shape']

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
