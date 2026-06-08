"""Per-sample TRAIN + VAL signal installers — pure taps over UL, no math
reimplementation. Every value shipped is captured from UL's existing
computation (forward hooks, pre-hooks, or a thin tap that replaces a
function reference with one that records its output before returning it).

Train signals (all routed through `get_assigned_targets_and_loss` after UL
ran the step):
  * train/cls_per_sample — `crit.bce` forward hook captures (bs, na, nc)
    BCE-per-anchor; sum(1,2) → per-image.
  * train/box_per_sample — `bbox_loss` pre-hook captures fg_mask +
    target_scores; a tap on `ul_loss.bbox_iou` captures the per-fg-anchor
    IoU UL computes once; we form `(1-iou)*weight` (UL's exact box-term
    formula) and scatter per image.
  * train/dfl_per_sample — `bbox_loss.dfl_loss` forward hook captures the
    per-fg-anchor DFL tensor; multiply by weight (the same one UL uses)
    and scatter per image.
  * train preds overlay — `Detect` head forward hook captures the raw
    training-mode preds dict (UL training skips decode). We then call
    UL's own `Detect._inference` + `non_max_suppression` for decode/NMS.

Val signals:
  * val preds overlay — wrap `validator.update_metrics`; preds already
    NMS'd by UL. Pure tap.
  * val/iou_per_sample — wrap `validator._process_batch` (UL calls it per
    image inside update_metrics) and use UL's own `box_iou` once to
    derive a per-image mean-of-max-IoU-per-GT scalar. Reduction only.

Discarding is enforced at the data sampler (deny-aware sampler excludes
discarded samples from batches), so signals always log the true loss of
whatever samples are actually in the batch.
"""
from __future__ import annotations

import torch as th
from torch.nn import Identity

from ultralytics.nn.modules.head import Detect
from ultralytics.utils import loss as ul_loss
from ultralytics.utils.metrics import box_iou as ul_box_iou
from ultralytics.utils.nms import non_max_suppression

import weightslab as wl

from ._utils import normalize_post_nms_preds


_NMS_CONF = 1e-4
_NMS_IOU = 0.45


def _scatter(per_fg, img_of_fg, bs):
    out = th.zeros(bs, device=per_fg.device, dtype=per_fg.dtype)
    out.scatter_add_(0, img_of_fg, per_fg.reshape(-1))
    return out


def _overlay_from_nms(nms_preds, img_shape):
    """UL post-NMS → studio overlay (xyxy normalized + cls + score)."""
    _, _, h, w = img_shape
    scale = th.tensor([w, h, w, h], dtype=th.float32)
    return [
        th.cat([(p[:, :4] / scale).cpu().float(), p[:, 5:6].cpu(), p[:, 4:5].cpu()], -1)
        if p.numel() else th.zeros((0, 6))
        for p in normalize_post_nms_preds(nms_preds)
    ]


def install_per_sample_signals(model):
    """Install per-sample TRAIN signals and live train preds overlay."""
    crit = (model.criterion if getattr(model, "criterion", None) is not None
            else model.init_criterion())
    bl = crit.bbox_loss
    detect_head = next((m for m in model.modules() if isinstance(m, Detect)), None)
    ch = {n: wl.watch_or_edit(Identity(), flag="loss",
                              name=f"train/{n}_per_sample",
                              per_sample=True, log=True)
          for n in ("cls", "box", "dfl")}

    # --- taps: capture UL's intermediates without rerunning the math ---
    def _bce_hook(m, inp, out): crit._wl_bce = out.detach()
    crit.bce.register_forward_hook(_bce_hook)

    def _bl_pre(m, args):
        # args = (pred_dist, pred_bboxes, anchor_points, target_bboxes,
        #         target_scores, target_scores_sum, fg_mask, imgsz, stride)
        crit._wl_fg_mask = args[6]
        crit._wl_target_scores = args[4]
    bl.register_forward_pre_hook(_bl_pre)

    def _dfl_hook(m, inp, out): crit._wl_dfl_per_fg = out.detach()
    bl.dfl_loss.register_forward_hook(_dfl_hook)

    # Tap UL's bbox_iou (used inside bbox_loss.forward) — capture the per-fg
    # IoU UL computes once; do NOT recompute. Restored on first replacement
    # if already tapped to keep idempotency.
    _orig_bbox_iou = getattr(ul_loss, "_wl_orig_bbox_iou", ul_loss.bbox_iou)
    ul_loss._wl_orig_bbox_iou = _orig_bbox_iou
    def _bbox_iou_tap(*a, **kw):
        out = _orig_bbox_iou(*a, **kw)
        crit._wl_iou_per_fg = out.detach()
        return out
    ul_loss.bbox_iou = _bbox_iou_tap

    if detect_head is not None:
        def _det_hook(m, inp, out):
            crit._wl_raw_preds = out if m.training else None
        detect_head.register_forward_hook(_det_hook)

    _orig = crit.get_assigned_targets_and_loss
    def _ship(preds, batch):
        for k in ("_wl_bce", "_wl_fg_mask", "_wl_target_scores",
                  "_wl_dfl_per_fg", "_wl_iou_per_fg", "_wl_raw_preds"):
            setattr(crit, k, None)
        res = _orig(preds, batch)
        if not model.training or crit._wl_bce is None:
            return res

        ids = batch["ids"]
        bs = batch["img"].shape[0]

        cls_kwargs = {"batch_ids": ids}
        if detect_head is not None and crit._wl_raw_preds is not None:
            try:
                y = detect_head._inference(crit._wl_raw_preds)
                nms = non_max_suppression(y, conf_thres=_NMS_CONF, iou_thres=_NMS_IOU)
                cls_kwargs["preds"] = {"bboxes": _overlay_from_nms(nms, batch["img"].shape)}
            except Exception:
                pass
        ch["cls"](crit._wl_bce.sum(dim=(1, 2)), **cls_kwargs)

        # per-image box/dfl: scatter UL's per-fg terms into a (bs,) vector.
        fg = crit._wl_fg_mask
        if fg is not None and fg.any() and crit._wl_iou_per_fg is not None:
            img_of_fg = fg.nonzero(as_tuple=False)[:, 0]
            weight = crit._wl_target_scores.sum(-1)[fg].unsqueeze(-1)
            box_term = ((1.0 - crit._wl_iou_per_fg) * weight).detach()
            ch["box"](_scatter(box_term, img_of_fg, bs), batch_ids=ids)
            if crit._wl_dfl_per_fg is not None:
                dfl_term = (crit._wl_dfl_per_fg * weight).detach()
                ch["dfl"](_scatter(dfl_term, img_of_fg, bs), batch_ids=ids)
        return res
    crit.get_assigned_targets_and_loss = _ship


def install_per_sample_val_signals(validator):
    """Install per-sample VAL signals (preds overlay + per-image IoU)."""
    preds_ch = wl.watch_or_edit(Identity(), flag="metric",
                                name="val/preds_per_sample",
                                per_sample=True, log=True)
    iou_ch = wl.watch_or_edit(Identity(), flag="metric",
                              name="val/iou_per_sample",
                              per_sample=True, log=True)

    # _process_batch runs per-image inside update_metrics; tap to capture
    # one per-image IoU scalar (uses UL's box_iou — pure reduction).
    state = {"per_img_iou": []}
    _orig_proc = validator._process_batch
    def _proc_tap(predn, pbatch):
        out = _orig_proc(predn, pbatch)
        gt = pbatch["bboxes"]
        pb = predn["bboxes"]
        if gt.numel() > 0 and pb.numel() > 0:
            iou = ul_box_iou(gt, pb)
            state["per_img_iou"].append(float(iou.max(dim=1).values.mean()))
        elif gt.numel() == 0 and pb.numel() == 0:
            state["per_img_iou"].append(1.0)  # vacuously correct
        else:
            state["per_img_iou"].append(0.0)
        return out
    validator._process_batch = _proc_tap

    _orig_update = validator.update_metrics
    def _tap(preds, batch):
        state["per_img_iou"].clear()
        _orig_update(preds, batch)
        ids = batch["ids"]
        preds_ch(
            th.zeros(batch["img"].shape[0]),
            batch_ids=ids,
            preds={"bboxes": _overlay_from_nms(preds, batch["img"].shape)},
        )
        if len(state["per_img_iou"]) == batch["img"].shape[0]:
            iou_ch(th.tensor(state["per_img_iou"]), batch_ids=ids)
    validator.update_metrics = _tap
