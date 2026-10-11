"""Teach ``wl.project_instances`` about a detection model it has never seen.

``wl.project_instances`` knows Ultralytics out of the box. For anything else it
needs an :class:`~weightslab.projection.instances.InstanceAdapter`, three
questions about your model and data:

* ``batches(loader)``: the images and their labelled boxes, in input pixels,
  with each box's annotation id (1-based, its place in the target list);
* ``forward(model, batch, conf=...)``: the feature map(s) to pool each box
  from (with their strides) and the model's predictions after NMS;
* ``class_names`` / ``layer``: for the legend and the board's caption.

Everything else (RoI pooling, matching predictions to labels, the
consistent / wrong class / no prediction / missing label categories, fitting
the layout, storing it for the Studio) is WeightsLab's.

This one is for the SmallDetector of ``examples/PyTorch/wl-detection``: a
MobileNetV3 backbone, a small conv neck, and one box per cell of an S x S grid.
"""
from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms

from weightslab.projection import resolve_module
from weightslab.projection.instances import (
    InstanceAdapter,
    InstanceBatch,
    InstanceForward,
    tracked_batches,
)


class SmallDetectorInstanceAdapter(InstanceAdapter):
    """:class:`InstanceAdapter` for ``SmallDetector`` (one box per grid cell)."""

    def __init__(self, grid_size: int, collate_fn, decode_predictions,
                 batch_size: int = 16, nms_iou: float = 0.5, max_det: int = 20):
        self.grid_size = int(grid_size)
        self.collate_fn = collate_fn
        self.decode_predictions = decode_predictions
        self.batch_size = int(batch_size)
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)

    # ------------------------------------------------------------------ data
    def batches(self, loader):
        # tracked_batches reads the WHOLE split in dataset order, without
        # touching the loader's own iterator (the training loop keeps its place).
        for images, ids, targets, _ in tracked_batches(loader, self.collate_fn, self.batch_size):
            height, width = int(images.shape[-2]), int(images.shape[-1])
            scale = torch.tensor([width, height, width, height], dtype=torch.float32)
            gt = []
            for target in targets:
                target = torch.as_tensor(target, dtype=torch.float32).reshape(-1, 6)
                gt.append((
                    (target[:, :4] * scale).numpy(),        # normalised xyxy -> pixels
                    target[:, 4].long().numpy(),             # class ids
                    np.arange(1, len(target) + 1),           # annotation ids, target order
                ))
            yield InstanceBatch(inputs=images, sample_ids=[str(i) for i in ids],
                                gt=gt, image_size=(height, width))

    # ----------------------------------------------------------------- model
    def forward(self, model, batch, *, conf):
        module = resolve_module(model) or model
        grabbed = {}
        # The neck's output is the representation the head reads: pool boxes there.
        handle = module.neck.register_forward_hook(
            lambda _m, _inputs, output: grabbed.__setitem__("fmap", output))
        try:
            device = next(module.parameters()).device
            outputs = model(batch.inputs.to(device))      # [B, S, S, 5 + classes]
        finally:
            handle.remove()
        fmap = grabbed["fmap"]
        stride = batch.image_size[1] / fmap.shape[-1]

        height, width = batch.image_size
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        predictions = []
        for det in self.decode_predictions(outputs.detach(), self.grid_size,
                                           conf_thresh=conf, max_det=100):
            det = det.float()
            boxes, classes, scores = det[:, :4] * scale, det[:, 4].long(), det[:, 5]
            if len(det):
                # Neighbouring cells can predict the same pedestrian.
                keep = batched_nms(boxes, scores, classes, self.nms_iou)[: self.max_det]
                boxes, classes, scores = boxes[keep], classes[keep], scores[keep]
            predictions.append((boxes.numpy(), scores.numpy(), classes.numpy()))
        return InstanceForward(feature_maps=[fmap], strides=[stride], predictions=predictions)

    # -------------------------------------------------------------- metadata
    def class_names(self, model, loaders):
        module = resolve_module(model) or model
        return list(getattr(module, "class_names", []) or [])

    def layer(self, model):
        return "neck", "neck output (RoI-pooled per box)"
