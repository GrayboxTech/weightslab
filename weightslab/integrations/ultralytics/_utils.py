"""Internal helpers — Sink channel sink, per-image scatter, post-NMS
normalization, per-sample IoU/mAP@0.5, and studio-overlay formatting."""
from __future__ import annotations

import torch
import torch as th
import torch.nn as nn

from ultralytics.utils.nms import box_iou
from ultralytics.utils.ops import xywh2xyxy


class Sink(nn.Module):
    """Pass-through so an already-computed scalar can ride WL's signal pipe as a
    curve via `wl.watch_or_edit(Sink(), flag="loss"|"metric", ...)`. Used for
    aggregate train-loss / val-metric curves."""

    def forward(self, x):
        return x


def _scatter_per_image(per_fg, img_of_fg, bs):
    """Sum per-foreground-anchor terms into a per-image (bs,) vector.
    Background images (no fg anchor) stay 0."""
    out = torch.zeros(bs, device=per_fg.device, dtype=per_fg.dtype)
    out.scatter_add_(0, img_of_fg, per_fg.reshape(-1))
    return out


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


def normalize_post_nms_preds(preds_list):
    """UL 8.4.51 returns post-NMS as `list[dict]` per image with keys
    `bboxes` (Ni, 4), `conf` (Ni,), `cls` (Ni,), `extra` (ignored). Older UL
    returns `list[Tensor(Ni, 6)] = [x1,y1,x2,y2,conf,cls]`. Normalize to the
    older shape so downstream helpers stay version-agnostic."""
    out = []
    for p in preds_list:
        if isinstance(p, dict):
            bb = p.get("bboxes")
            cf = p.get("conf")
            cl = p.get("cls")
            if bb is None or bb.numel() == 0:
                out.append(th.zeros((0, 6)))
                continue
            cf = cf.view(-1, 1).float()
            cl = cl.view(-1, 1).float()
            out.append(th.cat([bb.float(), cf, cl], dim=-1))
        else:
            out.append(p if p.numel() > 0 else th.zeros((0, 6)))
    return out


def per_sample_iou_post_nms(preds_list, batch):
    """preds_list: list[Tensor(Ni, 6)] post-NMS in pixel coords (UL convention:
    [x1,y1,x2,y2,conf,cls]). batch: UL batch dict — boxes are normalized xywh,
    batch_idx maps each box to its image. Returns Tensor(bs,) — mean of
    per-GT max IoU for each image."""
    img = batch["img"]
    bs, _, h, w = img.shape
    batch_idx = batch["batch_idx"].detach().cpu()
    gt_norm = batch["bboxes"].detach().cpu()
    out = th.zeros(bs)
    if gt_norm.numel() == 0:
        return out
    scale = th.tensor([w, h, w, h], dtype=gt_norm.dtype)
    gt_xyxy = xywh2xyxy(gt_norm) * scale
    for i in range(bs):
        gi = (batch_idx == i).nonzero(as_tuple=True)[0]
        if gi.numel() == 0 or preds_list[i].shape[0] == 0:
            continue
        out[i] = box_iou(gt_xyxy[gi], preds_list[i][:, :4].detach().cpu()).max(dim=1).values.mean()
    return out


def per_sample_map50_post_nms(preds_list, batch):
    """Per-image AP@0.5 from UL post-NMS preds. Empty preds AND empty GT → 1.0
    (vacuously correct); one empty → 0.0."""
    img = batch["img"]
    bs, _, h, w = img.shape
    batch_idx = batch["batch_idx"].detach().cpu()
    gt_norm = batch["bboxes"].detach().cpu()
    out = th.zeros(bs)
    scale = th.tensor([w, h, w, h], dtype=gt_norm.dtype) if gt_norm.numel() > 0 else None
    gt_xyxy = xywh2xyxy(gt_norm) * scale if scale is not None else None
    for i in range(bs):
        gi = (batch_idx == i).nonzero(as_tuple=True)[0] if gt_xyxy is not None else th.empty(0, dtype=th.long)
        gt = gt_xyxy[gi] if gi.numel() > 0 else th.zeros((0, 4))
        p = preds_list[i].detach().cpu()
        if p.shape[0] == 0 and gt.shape[0] == 0:
            out[i] = 1.0
            continue
        if p.shape[0] == 0 or gt.shape[0] == 0:
            continue  # one empty — score 0
        out[i] = _mini_ap(p[:, :4], p[:, 4], gt, 0.5)
    return out


def preds_for_overlay(preds_list, batch):
    """Convert UL post-NMS preds (list[Tensor(Ni, 6)] = [x1,y1,x2,y2,conf,cls]
    in pixels) to studio's expected per-image format:
    [Ni, 6] = [x1,y1,x2,y2,class_id,score] in NORMALIZED [0,1] coords.
    Empty images get a (0, 6) tensor."""
    img = batch["img"]
    _, _, h, w = img.shape
    scale = th.tensor([w, h, w, h], dtype=th.float32)
    out = []
    for p in preds_list:
        if p.numel() == 0:
            out.append(th.zeros((0, 6)))
            continue
        p_cpu = p.detach().cpu().float()
        boxes_norm = p_cpu[:, :4] / scale
        # UL: [..., conf, cls] → studio: [..., cls, score]
        out.append(th.cat([boxes_norm, p_cpu[:, 5:6], p_cpu[:, 4:5]], dim=-1))
    return out
