"""Instance-level projection for Ultralytics YOLO -- one point per box.

The adapter :func:`weightslab.project_instances` uses for a YOLO detect /
segment model (see :mod:`weightslab.projection.instances` for what is
computed). What it knows about Ultralytics:

* **Features**: the maps the ``Detect`` head reads (P3/P4/P5), captured by a
  forward PRE-hook -- the list is copied, since older heads overwrite its
  items in place. Strides come from ``Detect.stride``.
* **Predictions**: the eval output, ``(B, 4 + nc, anchors)`` (xywh unless the
  head decodes xyxy), or ``(B, max_det, 6)`` for an end-to-end head. Filtered
  and NMS-ed here with ``torchvision.ops.batched_nms`` rather than through
  ``ultralytics.utils.nms``, whose signature and return type have moved
  between releases.
* **Data**: a clean pass over each split. Training batches are mosaics of
  four images, which would pin three images' boxes on the fourth's sample id;
  so the dataset is read through its validation transforms (letterbox only,
  same square size as training), item by item, in its own order. The labelled
  boxes keep their order, so box *i* is annotation ``i + 1`` -- the instance
  row the ledger already holds for it.

Convenience: :func:`project_trainer_instances` projects a
``WLAwareTrainer``'s train + val splits with its EMA model, and the trainer
runs it every N epochs when the hyperparameters carry an
``instance_projection`` section (see :func:`instance_projection_config`).
"""
from __future__ import annotations

import contextlib
import copy
import logging
from typing import Iterator

import numpy as np
import torch as th

from weightslab.projection.instances import (
    InstanceAdapter,
    InstanceBatch,
    InstanceForward,
    _named_loaders,
    project_instances,
)

logger = logging.getLogger(__name__)

#: Keys of the ``instance_projection`` hyperparameter section, with defaults.
INSTANCE_PROJECTION_DEFAULTS = {
    "enabled": False,
    "every_n_epochs": 1,
    "prefix": "instances",
    "method": "umap",
    "conf": 0.25,
    "match_iou": 0.5,
    "include_predictions": True,
    "max_instances": 50000,
    "epochs": 10,
}


def _detect_head(module):
    """The detection head of a YOLO model (its last ``Detect`` subclass)."""
    try:
        from ultralytics.nn.modules.head import Detect
    except Exception:  # pragma: no cover - ultralytics missing
        Detect = None
    found = None
    for sub in module.modules():
        if (Detect is not None and isinstance(sub, Detect)) or type(sub).__name__ == "Detect":
            found = sub
    return found


def _tracked_dataset(loader):
    """``(tracking wrapper, Ultralytics dataset)`` behind a WeightsLab loader,
    a plain DataLoader or a dataset."""
    tracked = None
    for attr in ("tracked_dataset", "dataset"):
        try:
            tracked = getattr(loader, attr, None)
        except Exception:
            tracked = None
        if tracked is not None:
            break
    if tracked is None:
        tracked = loader
    inner = getattr(tracked, "wrapped_dataset", tracked)
    return tracked, inner


@contextlib.contextmanager
def _eval_transforms(dataset):
    """*dataset* reads through its validation transforms (letterbox, no
    mosaic / flips / colour jitter) inside the block."""
    saved = {k: getattr(dataset, k) for k in ("augment", "rect", "transforms")
             if hasattr(dataset, k)}
    try:
        if hasattr(dataset, "build_transforms") and "transforms" in saved:
            from ultralytics.cfg import DEFAULT_CFG
            dataset.augment = False
            dataset.rect = False
            dataset.transforms = dataset.build_transforms(hyp=copy.copy(DEFAULT_CFG))
        yield dataset
    finally:
        for key, value in saved.items():
            try:
                setattr(dataset, key, value)
            except Exception:
                pass


def _xywh_to_xyxy(boxes: th.Tensor) -> th.Tensor:
    xy, wh = boxes[..., :2], boxes[..., 2:4] / 2
    return th.cat([xy - wh, xy + wh], dim=-1)


class UltralyticsInstanceAdapter(InstanceAdapter):
    """:class:`InstanceAdapter` for Ultralytics YOLO detect/segment models."""

    def __init__(self, batch_size: int = 16, nms_iou: float = 0.7, max_det: int = 300):
        self.batch_size = int(batch_size)
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)

    # ------------------------------------------------------------------ data
    def batches(self, loader) -> Iterator[InstanceBatch]:
        from .collate import wl_ul_dict_collate

        tracked, dataset = _tracked_dataset(loader)
        n = len(tracked)
        with _eval_transforms(dataset):
            for start in range(0, n, self.batch_size):
                items = [tracked[i] for i in range(start, min(n, start + self.batch_size))]
                items = [it for it in items if isinstance(it, (tuple, list)) and len(it) >= 4]
                if not items:
                    continue
                batch = wl_ul_dict_collate(items)
                yield self._to_instance_batch(batch)

    @staticmethod
    def _to_instance_batch(batch) -> InstanceBatch:
        img = batch["img"]
        height, width = int(img.shape[-2]), int(img.shape[-1])
        ids = batch["ids"].tolist() if hasattr(batch["ids"], "tolist") else list(batch["ids"])
        bidx = batch["batch_idx"].view(-1).long()
        cls = batch["cls"].view(-1).long()
        boxes = _xywh_to_xyxy(batch["bboxes"].float()) * th.tensor(
            [width, height, width, height], dtype=th.float32)
        gt = []
        for b in range(img.shape[0]):
            mine = (bidx == b).nonzero(as_tuple=True)[0]
            gt.append((boxes[mine].numpy(), cls[mine].numpy(),
                       np.arange(1, len(mine) + 1, dtype=np.int64)))
        return InstanceBatch(inputs=img, sample_ids=[str(i) for i in ids], gt=gt,
                             image_size=(height, width))

    # ----------------------------------------------------------------- model
    def forward(self, model, batch: InstanceBatch, *, conf: float) -> InstanceForward:
        from weightslab.projection.parametric_umap import resolve_module

        module = resolve_module(model) or model
        head = _detect_head(module)
        if head is None:
            raise TypeError("UltralyticsInstanceAdapter: the model has no Detect head")
        captured = {}

        def _grab(_m, args):
            # Copied: older heads assign x[i] = ... in place.
            captured["feats"] = list(args[0])

        handle = head.register_forward_pre_hook(_grab)
        try:
            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype
        except StopIteration:
            device, dtype = th.device("cpu"), th.float32
        inputs = batch.inputs.to(device)
        inputs = (inputs.float() / 255.0) if not inputs.is_floating_point() else inputs.float()
        try:
            out = model(inputs.to(dtype))
        finally:
            handle.remove()
        feats = captured.get("feats")
        if not feats:
            raise RuntimeError("UltralyticsInstanceAdapter: the Detect head did not run")
        strides = [float(s) for s in getattr(head, "stride", [])]
        if len(strides) != len(feats):
            strides = [float(inputs.shape[-1]) / float(f.shape[-1]) for f in feats]
        preds = self._predictions(head, out, conf)
        return InstanceForward(feature_maps=feats, strides=strides, predictions=preds)

    def _predictions(self, head, out, conf: float) -> list:
        """Per image ``(boxes xyxy px, confidences, classes)`` after NMS."""
        from torchvision.ops import batched_nms

        y = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(y, dict):  # training-mode output: nothing decoded
            y = head._inference(y.get("one2many", y))
        y = y.detach().float().cpu()
        results = []
        if y.ndim == 3 and y.shape[-1] == 6 and getattr(head, "end2end", False):
            # End-to-end head: (B, max_det, 6) x1, y1, x2, y2, score, class.
            for det in y:
                det = det[det[:, 4] > conf]
                results.append((det[:, :4].numpy(), det[:, 4].numpy(), det[:, 5].long().numpy()))
            return results
        nc = int(getattr(head, "nc", y.shape[1] - 4))
        xyxy_head = bool(getattr(head, "xyxy", False))
        for det in y:  # (4 + nc + extra, anchors)
            boxes = det[:4].T
            scores = det[4:4 + nc].T
            best, cls = scores.max(dim=1)
            keep = best > conf
            boxes, best, cls = boxes[keep], best[keep], cls[keep]
            if not xyxy_head:
                boxes = _xywh_to_xyxy(boxes)
            if boxes.shape[0]:
                kept = batched_nms(boxes, best, cls, self.nms_iou)[: self.max_det]
                boxes, best, cls = boxes[kept], best[kept], cls[kept]
            results.append((boxes.numpy(), best.numpy(), cls.long().numpy()))
        return results

    # -------------------------------------------------------------- metadata
    def class_names(self, model, loaders) -> list:
        from weightslab.projection.parametric_umap import resolve_module

        module = resolve_module(model) or model
        names = getattr(module, "names", None)
        if not names:
            for _name, loader in _named_loaders(loaders):
                _, dataset = _tracked_dataset(loader)
                data = getattr(dataset, "data", None)
                names = data.get("names") if isinstance(data, dict) else None
                if names:
                    break
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        return [str(n) for n in (names or [])]

    def layer(self, model) -> tuple:
        from weightslab.projection.parametric_umap import resolve_module

        module = resolve_module(model) or model
        head = _detect_head(module)
        name = next((n for n, m in module.named_modules() if m is head), "") if head is not None else ""
        return name, "Detect input (P3/P4/P5, RoI-pooled per box)"


def instance_projection_config(hyperparams) -> dict:
    """The ``instance_projection`` hyperparameter section merged over the
    defaults; ``enabled`` False when it is absent."""
    try:
        section = hyperparams.get("instance_projection") if hyperparams is not None else None
    except Exception:
        section = None
    cfg = dict(INSTANCE_PROJECTION_DEFAULTS)
    if hasattr(section, "items"):
        # A section that is there means "on", unless it says otherwise.
        cfg.update({"enabled": True, **dict(section)})
    elif section is True:
        cfg["enabled"] = True
    return cfg


def project_trainer_instances(trainer, **overrides) -> dict:
    """Project every box of a ``WLAwareTrainer``'s train and val splits, with
    its EMA model (the one validation scores) when it has one."""
    model = trainer.ema.ema if getattr(trainer, "ema", None) is not None else trainer.model
    # A list, not a dict: each loader names its own split (its ledger name).
    loaders = [loader for loader in (getattr(trainer, "train_loader", None),
                                     getattr(trainer, "test_loader", None))
               if loader is not None]
    kwargs = {k: v for k, v in overrides.items() if v is not None}
    return project_instances(model, loaders, adapter=UltralyticsInstanceAdapter(), **kwargs)
