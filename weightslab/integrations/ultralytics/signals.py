"""Per-sample TRAIN (cls/box/dfl + live preds overlay) and VAL (IoU, mAP@0.5
+ post-NMS preds overlay) signal installers."""
from __future__ import annotations

import torch
import torch as th

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import bbox2dist

import weightslab as wl

from ._utils import (
    Sink,
    _scatter_per_image,
    normalize_post_nms_preds,
    per_sample_iou_post_nms,
    per_sample_map50_post_nms,
    preds_for_overlay,
)


def install_per_sample_signals(model, *, nms_conf_thres: float = 1e-4, nms_iou_thres: float = 0.45):
    """Make a YOLO detection criterion emit per-sample cls/box/dfl losses
    through WL loss-wrapper channels, keyed by canonical wrapper uids read
    from `batch['ids']`. Also runs NMS on the criterion's pre-NMS
    pred_bboxes/pred_scores once per step and ships the decoded predictions
    via `preds={'bboxes': ...}` on the cls channel so studio renders a live
    train-prediction overlay.

    Discarding is NOT done here — it is enforced at the data plane (the
    deny-aware sampler excludes discarded samples from the batch entirely).
    Signals therefore log the true loss of whatever samples are actually in
    the batch.

    Call once in `on_train_start`; emission happens inline inside every
    train loss call.
    """
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    crit = model.criterion
    device = next(model.parameters()).device

    # cls: stash the (bs, na, nc) BCE output AND the pre-sigmoid pred_scores
    # logits (BCE's first input) so the overlay path can NMS them.
    def _bce_hook(m, inp, out):
        crit._wl_bce = out
        crit._wl_pred_scores = inp[0].detach()
    crit.bce.register_forward_hook(_bce_hook)

    # box/dfl: faithful bbox_loss.forward drop-in that ALSO stashes per-image
    # box/dfl terms and decoded pred_bboxes/stride for the overlay path.
    bl = crit.bbox_loss

    def _bbox_forward(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                      target_scores, target_scores_sum, fg_mask, imgsz, stride):
        bs = fg_mask.shape[0]
        img_of_fg = fg_mask.nonzero(as_tuple=False)[:, 0]
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        box_term = (1.0 - iou) * weight
        target_ltrb = bbox2dist(anchor_points, target_bboxes, bl.dfl_loss.reg_max - 1)
        dfl_term = bl.dfl_loss(
            pred_dist[fg_mask].view(-1, bl.dfl_loss.reg_max), target_ltrb[fg_mask]
        ) * weight
        crit._wl_box = _scatter_per_image(box_term.detach(), img_of_fg, bs)
        crit._wl_dfl = _scatter_per_image(dfl_term.detach(), img_of_fg, bs)
        crit._wl_pred_bboxes = pred_bboxes.detach()
        crit._wl_stride = stride
        crit._wl_imgsz = imgsz
        loss_iou = box_term.sum() / target_scores_sum
        loss_dfl = dfl_term.sum() / target_scores_sum
        return loss_iou, loss_dfl
    bl.forward = _bbox_forward

    ch = {
        sig: wl.watch_or_edit(Sink(), flag="loss", name=f"train/{sig}_per_sample",
                              per_sample=True, log=True)
        for sig in ("cls", "box", "dfl")
    }

    _assign_orig = crit.get_assigned_targets_and_loss

    def _assign(preds, batch):
        crit._wl_box = crit._wl_dfl = None
        crit._wl_pred_bboxes = crit._wl_stride = crit._wl_imgsz = None
        crit._wl_pred_scores = None
        res = _assign_orig(preds, batch)
        if model.training:
            ids = batch["ids"].to(device)
            train_overlay = None
            if (crit._wl_pred_bboxes is not None
                    and crit._wl_pred_scores is not None
                    and crit._wl_stride is not None):
                try:
                    pred_bb_px = crit._wl_pred_bboxes * crit._wl_stride
                    pred_sc = crit._wl_pred_scores.sigmoid()
                    nms_input = th.cat(
                        [pred_bb_px.float(), pred_sc.float()], dim=-1
                    ).permute(0, 2, 1)
                    nms_out = non_max_suppression(
                        nms_input, conf_thres=nms_conf_thres, iou_thres=nms_iou_thres,
                    )
                    train_overlay = preds_for_overlay(
                        normalize_post_nms_preds(nms_out), batch,
                    )
                except Exception as e:
                    import sys
                    import traceback
                    if not hasattr(crit, "_wl_overlay_err_printed"):
                        crit._wl_overlay_err_printed = True
                        print(f"[WL train overlay] failed: {e}", file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                    train_overlay = None
            cls_kwargs = {"batch_ids": ids}
            if train_overlay is not None:
                cls_kwargs["preds"] = {"bboxes": train_overlay}
            ch["cls"](crit._wl_bce.sum(dim=(1, 2)).detach(), **cls_kwargs)
            if crit._wl_box is not None:
                ch["box"](crit._wl_box, batch_ids=ids)
                ch["dfl"](crit._wl_dfl, batch_ids=ids)
        return res
    crit.get_assigned_targets_and_loss = _assign


def install_per_sample_val_signals(validator):
    """Wrap `validator.update_metrics(preds, batch)` to emit per-image IoU,
    AP@0.5, and post-NMS predictions overlay via WL channels. Requires
    `batch['ids']` (provided by `wl_ul_dict_collate`)."""
    iou_ch = wl.watch_or_edit(
        Sink(), flag="metric", name="val/iou_per_sample",
        per_sample=True, log=True,
    )
    map_ch = wl.watch_or_edit(
        Sink(), flag="metric", name="val/map50_per_sample",
        per_sample=True, log=True,
    )

    _orig_update = validator.update_metrics

    def _wrapped_update(preds, batch):
        try:
            uids = batch.get("ids", None)
            if (uids is not None
                    and isinstance(preds, (list, tuple))
                    and len(preds) == batch["img"].shape[0]):
                norm_preds = normalize_post_nms_preds(preds)
                iou_per = per_sample_iou_post_nms(norm_preds, batch)
                map_per = per_sample_map50_post_nms(norm_preds, batch)
                overlay = preds_for_overlay(norm_preds, batch)
                iou_ch(iou_per, batch_ids=uids, preds={"bboxes": overlay})
                map_ch(map_per, batch_ids=uids)
        except Exception as e:
            import sys
            import traceback
            print(f"[WL val per-sample] skipped batch: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
        return _orig_update(preds, batch)

    validator.update_metrics = _wrapped_update
