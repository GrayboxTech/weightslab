"""Per-sample TRAIN + VAL signal installers — pure taps over UL, no math
reimplementation. Decode/NMS used in train come from UL's own code paths.

  * train/cls_per_sample: forward-hook on `crit.bce` captures the
    (bs, na, nc) BCE-per-anchor tensor UL already computes; reduce to
    per-image with `.sum(dim=(1,2))`.
  * train preds overlay: forward-hook on the `Detect` head captures the
    raw training-mode preds dict (UL training skips decoding). We then
    call UL's own `Detect._inference(preds)` + UL's `non_max_suppression`
    to get NMS'd preds — UL's decode/NMS code, no rewrite.
  * val preds overlay: wrap `validator.update_metrics(preds, batch)` —
    `preds` is already NMS'd by UL.

Discarding is enforced at the data sampler (deny-aware sampler excludes
discarded samples from batches), so signals always log the true loss of
whatever samples are actually in the batch.
"""
from __future__ import annotations

import torch as th
from torch.nn import Identity

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.nms import non_max_suppression

import weightslab as wl

from ._utils import normalize_post_nms_preds


_NMS_CONF = 1e-4   # tiny so early-training overlays aren't empty
_NMS_IOU = 0.45


def _overlay_from_nms(nms_preds, img_shape):
    """UL post-NMS (`list[dict]` or `list[Tensor]`) → studio overlay format
    (`list[Tensor(Ni, 6) = xyxy + cls + score]` in normalized [0,1] coords)."""
    _, _, h, w = img_shape
    scale = th.tensor([w, h, w, h], dtype=th.float32)
    return [
        th.cat([(p[:, :4] / scale).cpu().float(), p[:, 5:6].cpu(), p[:, 4:5].cpu()], -1)
        if p.numel() else th.zeros((0, 6))
        for p in normalize_post_nms_preds(nms_preds)
    ]


def install_per_sample_signals(model):
    """Train per-sample cls + train preds overlay via UL hooks.

    cls: `crit.bce` forward hook → (bs, na, nc) → sum(1,2) → per-image.
    overlay: forward hook on the `Detect` head captures raw training-mode
    preds; we feed them through UL's own `_inference` + NMS for decode."""
    crit = (model.criterion if getattr(model, "criterion", None) is not None
            else model.init_criterion())
    cls_ch = wl.watch_or_edit(
        Identity(), flag="loss", name="train/cls_per_sample",
        per_sample=True, log=True,
    )

    detect_head = next((m for m in model.modules() if isinstance(m, Detect)), None)

    def _bce_hook(m, inp, out): crit._wl_bce = out.detach()
    crit.bce.register_forward_hook(_bce_hook)

    if detect_head is not None:
        def _det_hook(m, inp, out):
            crit._wl_raw_preds = out if m.training else None
        detect_head.register_forward_hook(_det_hook)

    _orig = crit.get_assigned_targets_and_loss
    def _ship(preds, batch):
        crit._wl_bce = None
        crit._wl_raw_preds = None
        res = _orig(preds, batch)
        if model.training and crit._wl_bce is not None:
            kwargs = {"batch_ids": batch["ids"]}
            if detect_head is not None and crit._wl_raw_preds is not None:
                try:
                    y = detect_head._inference(crit._wl_raw_preds)
                    nms = non_max_suppression(y, conf_thres=_NMS_CONF, iou_thres=_NMS_IOU)
                    kwargs["preds"] = {"bboxes": _overlay_from_nms(nms, batch["img"].shape)}
                except Exception:
                    pass
            cls_ch(crit._wl_bce.sum(dim=(1, 2)), **kwargs)
        return res
    crit.get_assigned_targets_and_loss = _ship


def install_per_sample_val_signals(validator):
    """Val: wrap `validator.update_metrics` to tap UL's already-NMS'd preds
    and ship the overlay. No NMS or decode here — pure tap."""
    ch = wl.watch_or_edit(
        Identity(), flag="metric", name="val/preds_per_sample",
        per_sample=True, log=True,
    )
    _orig = validator.update_metrics
    def _tap(preds, batch):
        ch(
            th.zeros(batch["img"].shape[0]),
            batch_ids=batch["ids"],
            preds={"bboxes": _overlay_from_nms(preds, batch["img"].shape)},
        )
        return _orig(preds, batch)
    validator.update_metrics = _tap
