"""wl_ultralytics — the WL ↔ Ultralytics integration shim.

Everything that doesn't belong in a clean entrypoint lives here:
  * Per-sample loss / IoU / detection-metric wrappers around UL primitives.
  * Dataset wrapper (`YOLODatasetWL`) that speaks WL's preview protocol.
  * Collate fn that repacks WL per-sample tuples into UL batches.
  * `load_config` (YAML + device + log_dir defaults).
  * `attach` — installs UL callbacks that wire datasets / optimizer / model /
    losses / metrics into WL.
  * Dispatch — patches `wl.watch_or_edit` so YOLO instances route through
    `attach` symmetrically with the other registrations.
  * Env defaults — set BEFORE weightslab is imported by this module.

The destination is that this file eventually folds into weightslab's
integrations module; for now it's the single self-contained helper.

Importing this module is a side-effecting operation:
  * Sets `WL_PRELOAD_IMAGE_OVERVIEW=0` and `WEIGHTSLAB_LOG_LEVEL=WARNING`.
  * Monkey-patches `wl.watch_or_edit` to dispatch YOLO models to `attach`.
  * On first `attach(...)`, registers `wl.keep_serving` as an atexit hook
    so the studio backend stays alive after training ends.
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import atexit
import tempfile
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch as th
import torch.nn as nn
import yaml

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss
from ultralytics.utils.nms import non_max_suppression, box_iou
from ultralytics.utils.ops import xywh2xyxy, xywhn2xyxy
from ultralytics.utils.tal import make_anchors, dist2bbox

import weightslab as wl
from weightslab.components.global_monitoring import pause_controller


# =============================================================================
# Helpers — prediction decoding, GT↔pred matching, mini-AP.
# =============================================================================

def _decode_predictions(pred, img_h, img_w, conf=0.25, iou_thres=0.5):
    """Decode raw model preds [batch, 64+nc, 8400] → per-sample (boxes, [score, cls]) after NMS."""
    pred = pred.detach().cpu()
    pred = pred.permute(0, 2, 1)  # [batch, 8400, 64+nc]
    pred_boxes = pred[..., :64]   # DFL distributions
    pred_cls = pred[..., 64:]     # class logits

    # make_anchors returns grid-coord points + stride_tensor to scale to pixels.
    # Dropping stride_tensor → boxes off by 8/16/32×.
    strides = [8, 16, 32]
    anchors, stride_tensor = make_anchors(
        [th.zeros(1, 1, img_h // s, img_w // s) for s in strides],
        strides, grid_cell_offset=0.5,
    )

    # Emit xywh (not xyxy): non_max_suppression unconditionally xywh2xyxys its input.
    pred_boxes = pred_boxes.view(*pred_boxes.shape[:2], 4, 16)
    pred_boxes = pred_boxes.softmax(-1) @ th.arange(16).float()
    pred_boxes = dist2bbox(pred_boxes, anchors.unsqueeze(0), xywh=True) * stride_tensor

    # NMS expects [B, 4+nc, N].
    pred_scores = pred_cls.sigmoid()
    pred_combined = th.cat([pred_boxes, pred_scores], dim=-1).permute(0, 2, 1)
    preds_nms = non_max_suppression(
        pred_combined, conf_thres=conf, iou_thres=iou_thres, max_det=300)

    # Per sample: (N_i, 6) = [x1, y1, x2, y2, conf, cls] in pixels.
    return ([i[:, :-2] if i.ndim > 1 else i for i in preds_nms],
            [i[:, -2:] if i.ndim > 1 else i for i in preds_nms])


def _greedy_match(pred_boxes, pred_conf, gt_boxes, iou_thr):
    """Greedy GT-to-prediction matching by descending confidence.
    Returns (tp, fp, fn). Each prediction matches at most one GT (IoU >= iou_thr)."""
    P, G = pred_boxes.shape[0], gt_boxes.shape[0]
    if P == 0 and G == 0:
        return 0, 0, 0
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0
    iou = box_iou(pred_boxes, gt_boxes)            # [P, G]
    order = th.argsort(pred_conf, descending=True)
    matched = th.zeros(G, dtype=th.bool)
    tp = 0
    for p in order.tolist():
        avail = (~matched) & (iou[p] >= iou_thr)
        if avail.any():
            best = th.where(avail, iou[p], th.full_like(iou[p], -1.0)).argmax().item()
            matched[best] = True
            tp += 1
    return tp, P - tp, G - int(matched.sum().item())


def _mini_ap(pred_boxes, pred_conf, gt_boxes, iou_thr):
    """Per-sample mini Average Precision at one IoU threshold.
    11-point interpolation (VOC2007-style)."""
    P, G = pred_boxes.shape[0], gt_boxes.shape[0]
    if P == 0 and G == 0:
        return 1.0
    if P == 0 or G == 0:
        return 0.0
    iou = box_iou(pred_boxes, gt_boxes)
    order = th.argsort(pred_conf, descending=True)
    matched = th.zeros(G, dtype=th.bool)
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for p in order.tolist():
        avail = (~matched) & (iou[p] >= iou_thr)
        if avail.any():
            best = th.where(avail, iou[p], th.full_like(iou[p], -1.0)).argmax().item()
            matched[best] = True
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / G)
    ap = 0.0
    for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_at_r = max([p for p, rr in zip(precisions, recalls) if rr >= r], default=0.0)
        ap += p_at_r / 11.0
    return ap


# =============================================================================
# Per-sample loss / metric wrappers.
# =============================================================================

class PerSampleDetectionLoss(nn.Module):
    """Per-sample wrapper around v8DetectionLoss; loss_type picks [0=box, 1=cls, 2=dfl] or None=sum."""

    def __init__(self, model, loss_type=None):
        super().__init__()
        self.base_loss = DetectionLoss(model)
        self.loss_type = loss_type
        self.model = model

    def _pick(self, loss):
        # v8DetectionLoss returns [box, cls, dfl]. Pick this criterion's component; .mean() would
        # give (box+cls+dfl)/3 across all three → 3× gradient inflation when summed.
        if loss.numel() > 1:
            return loss[self.loss_type] if self.loss_type is not None else loss.sum()
        return loss.squeeze() if loss.ndim > 0 else loss

    def forward(self, pred, batch):
        batch_idx = batch.get('batch_idx', None)

        if batch_idx is None or batch_idx.numel() == 0:
            out = self.base_loss(pred, batch)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            return self._pick(loss).unsqueeze(0)

        if isinstance(pred, dict):
            first = next(iter(pred.values()))
            pred_bs = first.shape[0]
        elif isinstance(pred, (tuple, list)):
            pred_bs = pred[0].shape[0]
        else:
            pred_bs = pred.shape[0]

        batch_idx_flat = batch_idx.flatten().long()
        sample_losses = []

        for s in range(pred_bs):
            box_mask = batch_idx_flat == s
            sample_batch = {
                k: (th.zeros_like(v[box_mask]) if k == 'batch_idx' else v[box_mask])
                if isinstance(v, th.Tensor) and v.shape[0] == len(batch_idx_flat) else v
                for k, v in batch.items()
            }
            if isinstance(pred, dict):
                sample_pred = {k: v[s:s+1] if isinstance(v, th.Tensor) else v for k, v in pred.items()}
            elif isinstance(pred, (tuple, list)):
                sample_pred = tuple(p[s:s+1] if isinstance(p, th.Tensor) else p for p in pred)
            else:
                sample_pred = pred[s:s+1]
            out = self.base_loss(sample_pred, sample_batch)
            sample_loss = out[0] if isinstance(out, (tuple, list)) else out
            sample_losses.append(self._pick(sample_loss))

        return th.stack(sample_losses)


class PerSampleIoU(nn.Module):
    """Per-sample IoU between GT and NMS-filtered predictions."""

    def __init__(self, conf: float = 0.25, iou_thres: float = 0.5):
        super().__init__()
        self.conf = conf
        self.iou_thres = iou_thres

    def forward(self, pred, batch):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        elif isinstance(pred, dict) and 'boxes' in pred and 'scores' in pred:
            pred = th.cat([pred['boxes'], pred['scores']], dim=1)

        img, batch_idx, boxes_norm = batch['img'], batch['batch_idx'], batch['bboxes']
        img_h, img_w = img.shape[-2:]
        preds_nms, _ = _decode_predictions(pred, img_h, img_w, self.conf, self.iou_thres)

        scale = th.tensor([img_w, img_h, img_w, img_h], dtype=boxes_norm.dtype)
        gt_xyxy = xywh2xyxy(boxes_norm.detach().cpu()) * scale

        out = th.full((img.shape[0],), float("nan"))
        for i in range(img.shape[0]):
            gi = (batch_idx == i).nonzero(as_tuple=True)[0]
            if preds_nms[i].shape[0] > 0 and gi.numel() > 0:
                out[i] = box_iou(gt_xyxy[gi], preds_nms[i][:, :4]).max(dim=1).values.mean()
        return th.nan_to_num(out, nan=0.0)


class PerSampleDetMetric(nn.Module):
    """Per-sample detection metric — precision / recall / mAP at IoU=0.5 or
    averaged over IoU=0.5:0.05:0.95.

    For empty (preds, GT) the convention is: both empty → 1.0; one empty → 0.0.
    """

    METRICS = (
        "precision@0.5",   "recall@0.5",   "mAP@0.5",
        "precision@0.5:0.95", "recall@0.5:0.95", "mAP@0.5:0.95",
    )
    _IOU_RANGE = th.linspace(0.5, 0.95, 10).tolist()

    def __init__(self, metric: str, conf: float = 0.25, iou_thres_nms: float = 0.5):
        super().__init__()
        if metric not in self.METRICS:
            raise ValueError(f"unknown metric {metric!r}; choose from {self.METRICS}")
        self.metric = metric
        self.conf = conf
        self.iou_thres_nms = iou_thres_nms

    def _per_sample_value(self, pred_boxes, pred_conf, gt_boxes):
        kind, _, thr_spec = self.metric.partition("@")
        thresholds = self._IOU_RANGE if thr_spec == "0.5:0.95" else [float(thr_spec)]

        if kind == "mAP":
            vals = [_mini_ap(pred_boxes, pred_conf, gt_boxes, t) for t in thresholds]
            return float(sum(vals) / len(vals))

        vals = []
        for t in thresholds:
            tp, fp, fn = _greedy_match(pred_boxes, pred_conf, gt_boxes, t)
            denom = (tp + fp) if kind == "precision" else (tp + fn)
            vals.append(1.0 if (tp == 0 and denom == 0) else (tp / denom if denom > 0 else 0.0))
        return float(sum(vals) / len(vals))

    def forward(self, pred, batch):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        elif isinstance(pred, dict) and 'boxes' in pred and 'scores' in pred:
            pred = th.cat([pred['boxes'], pred['scores']], dim=1)

        img, batch_idx, boxes_norm = batch['img'], batch['batch_idx'], batch['bboxes']
        img_h, img_w = img.shape[-2:]
        preds_nms, scores_cls = _decode_predictions(pred, img_h, img_w, self.conf, self.iou_thres_nms)

        scale = th.tensor([img_w, img_h, img_w, img_h], dtype=boxes_norm.dtype)
        gt_xyxy = xywh2xyxy(boxes_norm.detach().cpu()) * scale

        out = th.zeros(img.shape[0])
        for i in range(img.shape[0]):
            gi = (batch_idx == i).nonzero(as_tuple=True)[0]
            gt = gt_xyxy[gi]
            pb = preds_nms[i][:, :4] if preds_nms[i].numel() > 0 else preds_nms[i]
            pc = scores_cls[i][:, 0] if scores_cls[i].numel() > 0 else scores_cls[i]
            out[i] = self._per_sample_value(pb, pc, gt)
        return out


# =============================================================================
# Dataset — YOLODataset adapter + WL preview protocol + collate.
# =============================================================================

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
        target = (
            np.concatenate([bboxes_lb.astype(np.float32), cls, np.ones((n, 1), dtype=np.float32)], axis=1)
            if n > 0 else np.zeros((0, 6), dtype=np.float32)
        )
        return None, str(i), target, {"img_path": lab["im_file"], "cls": lab["cls"]}

    def get_items(self, i, include_metadata=False, include_labels=False, include_images=False):
        data = super().__getitem__(i)
        image = data['img'] if include_images else None
        metadata = (
            {'img_path': data['im_file'], 'cls': data['cls'], 'batch': data}
            if include_metadata else {}
        )
        labels = None

        if include_labels:
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


# =============================================================================
# Config loading.
# =============================================================================

def load_config(path):
    cfg = yaml.safe_load(open(path)) if os.path.exists(path) else {}
    if cfg.get("device", "auto") == "auto":
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("root_log_dir"):
        cfg["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(cfg["root_log_dir"], exist_ok=True)
    return cfg


# =============================================================================
# Integration: attach UL callbacks for monitoring + steering.
# =============================================================================

_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]
_atexit_registered = False


def attach(model, cfg=None):
    """Install UL callbacks. `cfg` kept for backward compat; device is
    auto-inferred from the wrapped model in `on_train_start`."""
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(wl.keep_serving)
        _atexit_registered = True

    losses = {"train": {}, "val": {}}
    ious = {}
    val_metrics = {}            # per-sample detection metrics (val-only)

    def _on_train_start(trainer):
        for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
            loader.dataset.__class__ = YOLODatasetWL
            wl.watch_or_edit(
                loader.dataset, flag="data", loader_name=f"{split}_loader",
                batch_size=loader.batch_size, shuffle=(split == "train"),
                num_workers=0, drop_last=False, compute_hash=False,
                is_training=(split == "train"),
                collate_fn=_wl_yolo_collate,
                preload_labels=True, preload_metadata=True,
            )

        # UL builds the optimizer by iterating model.modules() with isinstance
        # checks — wrap optimizer first, then model, otherwise the checks fail.
        trainer.optimizer = wl.watch_or_edit(trainer.optimizer, flag="optimizer")
        # `light=True` is the ModelInterface default (dev branch); pass
        # `light=False` to opt into model surgery + checkpoint auto-load.
        trainer.model = wl.watch_or_edit(trainer.model, flag="model")

        for split in ("train", "val"):
            for t, n in _LOSS_PARTS:
                losses[split][n] = wl.watch_or_edit(
                    PerSampleDetectionLoss(trainer.model, loss_type=t),
                    flag="loss", name=f"{split}/{n}", per_sample=True, log=True,
                )
            ious[split] = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name=f"miou/{split}", per_sample=True, log=True,
            )

        # Per-sample detection metrics — val only. UL's aggregated mAP /
        # precision / recall would belong at on_val_end via results_dict
        # traversal; not wired here yet (scalar-emit idiom in WL is TBD).
        for m in PerSampleDetMetric.METRICS:
            val_metrics[m] = wl.watch_or_edit(
                PerSampleDetMetric(metric=m, conf=0.25, iou_thres_nms=0.5),
                flag="metric", name=f"val/{m}", per_sample=True, log=True,
            )

        @wl.eval_fn
        def _validate(loader):
            trainer.validator(model=(trainer.ema.ema if trainer.ema else trainer.model))

        pause_controller.resume(force=True)

    def _on_train_batch_start(trainer):
        wl.guard_training_context.__enter__()

    def _on_train_batch_end(trainer):
        _emit_losses(losses, ious, "train", trainer.preds, trainer.batch)
        wl.guard_training_context.__exit__(None, None, None)

    def _on_val_batch_start(validator):
        wl.guard_testing_context.__enter__()

    def _on_val_batch_end(validator):
        _emit_losses(losses, ious, "val", validator.preds, validator.batch)
        bs = validator.batch["batch_idx"]
        for ch in val_metrics.values():
            ch(validator.preds, validator.batch, batch_ids=bs)
        wl.guard_testing_context.__exit__(None, None, None)

    model.add_callback("on_train_start",       _on_train_start)
    model.add_callback("on_train_batch_start", _on_train_batch_start)
    model.add_callback("on_train_batch_end",   _on_train_batch_end)
    model.add_callback("on_val_batch_start",   _on_val_batch_start)
    model.add_callback("on_val_batch_end",     _on_val_batch_end)


def _emit_losses(losses, ious, split, preds, batch):
    bs = batch["batch_idx"]
    for n in ("bbxs", "clsf", "dfl"):
        losses[split][n](preds, batch, batch_ids=bs)
    ious[split](preds, batch, batch_ids=bs)


# =============================================================================
# Dispatch — route YOLO objects through `attach` so callers can write
# `model = wl.watch_or_edit(model)` symmetrically with the other registrations.
# =============================================================================

def _is_ultralytics_model(obj):
    return obj.__class__.__module__.startswith("ultralytics") and hasattr(obj, "add_callback")


_orig_watch_or_edit = wl.watch_or_edit


def _dispatch_watch_or_edit(obj, *args, **kwargs):
    if _is_ultralytics_model(obj):
        attach(obj)
        return obj
    return _orig_watch_or_edit(obj, *args, **kwargs)


wl.watch_or_edit = _dispatch_watch_or_edit


# =============================================================================
# Misc.
# =============================================================================

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)
