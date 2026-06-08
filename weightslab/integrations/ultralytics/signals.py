"""Per-sample signal pipelines for UL detection — pure taps over UL, no math
reimplementation.

Each signal is a `Signal(name, flag, reduce, preds=None)` record. Capture
primitives (`fwd_hook`, `pre_hook`, `fn_tap`, `per_call_buffer`) install a
hook on a UL internal and return a zero-arg getter for the captured state;
closures bind the getter into each signal's `reduce` / `preds` function.

A *pipeline* (`install_train_pipeline`, `install_val_pipeline`) is the one
imperative bit: it wraps a UL sync method, runs each signal's reduce after
the original ran, and ships to a WL channel keyed by `batch["ids"]`.

Adding signal N+1 is appending one `Signal(...)` record to a list. No edits
to the orchestrator, no new hook wiring unless the new signal taps a place
no existing signal taps yet.

Public:
    Signal, fwd_hook, pre_hook, fn_tap, per_call_buffer,
    install_train_pipeline, install_val_pipeline,
    default_train_signals, default_val_signals,
    install_per_sample_signals, install_per_sample_val_signals,
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch as th
from torch.nn import Identity

from ultralytics.nn.modules.head import Detect
from ultralytics.utils import loss as ul_loss
from ultralytics.utils.metrics import box_iou as ul_box_iou
from ultralytics.utils.nms import non_max_suppression

import weightslab as wl

from ._utils import normalize_post_nms_preds


# ─── capture primitives ────────────────────────────────────────────────
# Each returns a zero-arg getter for the captured value. The hook itself
# fires inside UL's call; the getter is read inside our reduce/preds.

def fwd_hook(module) -> Callable[[], Any]:
    """Capture the forward-pass output of `module`."""
    box = {"v": None}
    def _h(_m, _i, out): box["v"] = out.detach() if hasattr(out, "detach") else out
    module.register_forward_hook(_h)
    return lambda: box["v"]


def pre_hook(module) -> Callable[[], Any]:
    """Capture the positional args passed to `module.forward`."""
    box = {"v": None}
    def _h(_m, args): box["v"] = args
    module.register_forward_pre_hook(_h)
    return lambda: box["v"]


def fn_tap(namespace, name: str) -> Callable[[], Any]:
    """Tap a function reference in a module namespace (e.g.
    `ul_loss.bbox_iou`). UL still does the call; we record the result."""
    box = {"v": None}
    _orig = getattr(namespace, name)
    def _t(*a, **kw):
        out = _orig(*a, **kw)
        box["v"] = out.detach() if hasattr(out, "detach") else out
        return out
    setattr(namespace, name, _t)
    return lambda: box["v"]


def per_call_buffer(obj, name: str, mapper: Callable[[tuple, Any], Any]) -> Callable[[], list]:
    """Accumulate one entry per call of `obj.name` via `mapper(args, ret)`.
    Returns a getter for the accumulated list — caller is responsible for
    clearing between rounds."""
    buf: list = []
    _orig = getattr(obj, name)
    def _t(*a, **kw):
        out = _orig(*a, **kw)
        buf.append(mapper(a, out))
        return out
    setattr(obj, name, _t)
    return lambda: buf


# ─── signal record ─────────────────────────────────────────────────────

@dataclass
class Signal:
    """A per-sample signal.

      * `reduce(batch)` returns a (B,) tensor of per-image scalars, or
        `None` to skip this round (e.g. no fg-anchors this step).
      * `preds(batch)` optionally returns the studio-overlay dict
        `{"bboxes": list_of_tensors}` to attach via the channel's
        `preds=` kwarg.
    """
    name: str
    flag: str  # "loss" | "metric"
    reduce: Callable[[dict], Optional[th.Tensor]]
    preds: Optional[Callable[[dict], Optional[dict]]] = None


# ─── orchestrators ─────────────────────────────────────────────────────

def _make_channels(signals):
    return {s.name: wl.watch_or_edit(
                Identity(), flag=s.flag, name=s.name, per_sample=True, log=True)
            for s in signals}


def _ship_round(signals, channels, batch):
    ids = batch["ids"]
    # Wrap reduce/preds in no_grad — signal capture is observational; running
    # `Detect._inference` + NMS inside the train autograd graph creates tensors
    # the graph keeps alive until backward, which compounds across steps and
    # OOMs at batch sizes the model alone would fit.
    with th.no_grad():
        for s in signals:
            v = s.reduce(batch)
            if v is None:
                continue
            kw = {"batch_ids": ids}
            if s.preds is not None:
                p = s.preds(batch)
                if p is not None:
                    kw["preds"] = p
            channels[s.name](v, **kw)


def install_train_pipeline(model, signals: list[Signal]):
    """Sync point: `criterion.get_assigned_targets_and_loss(preds, batch)`."""
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    crit = model.criterion
    channels = _make_channels(signals)
    _orig = crit.get_assigned_targets_and_loss
    def _ship(preds, batch):
        res = _orig(preds, batch)
        if model.training:
            _ship_round(signals, channels, batch)
        return res
    crit.get_assigned_targets_and_loss = _ship


def install_val_pipeline(validator, signals: list[Signal]):
    """Sync point: `validator.update_metrics(preds, batch)`.
    Note: `preds` is exposed to signal reducers by stashing on the validator
    before the original call — signals that need it read `validator._wl_preds`."""
    channels = _make_channels(signals)
    _orig = validator.update_metrics
    def _ship(preds, batch):
        validator._wl_preds = preds  # exposed to signal reducers/predsers
        _ship_round(signals, channels, batch)
        return _orig(preds, batch)
    validator.update_metrics = _ship


# ─── helpers used by the default packs ─────────────────────────────────

def _scatter(per_fg, img_of_fg, bs):
    out = th.zeros(bs, device=per_fg.device, dtype=per_fg.dtype)
    out.scatter_add_(0, img_of_fg, per_fg.reshape(-1))
    return out


def _overlay_dict(nms_preds, img_hw):
    h, w = img_hw
    scale = th.tensor([w, h, w, h], dtype=th.float32)
    out = []
    for p in normalize_post_nms_preds(nms_preds):
        if p.numel() == 0:
            out.append(th.zeros((0, 6)))
            continue
        pc = p.detach().cpu().float()
        out.append(th.cat([pc[:, :4] / scale, pc[:, 5:6], pc[:, 4:5]], -1))
    return {"bboxes": out}


# ─── default signal packs ──────────────────────────────────────────────

def default_train_signals(model) -> list[Signal]:
    """The 4 default UL-detection train signals: per-sample cls/box/dfl +
    live preds overlay. Compose with user signals: `default_train_signals(m) + [Signal(...)]`."""
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    crit = model.criterion
    bl = crit.bbox_loss
    detect_head = next((m for m in model.modules() if isinstance(m, Detect)), None)

    get_bce = fwd_hook(crit.bce)
    get_dfl = fwd_hook(bl.dfl_loss)
    get_iou = fn_tap(ul_loss, "bbox_iou")
    get_bl_args = pre_hook(bl)
    get_det = fwd_hook(detect_head) if detect_head is not None else None

    def _fg_state():
        args = get_bl_args()
        if args is None:
            return None
        fg = args[6]
        if not fg.any():
            return None
        weight = args[4].sum(-1)[fg].unsqueeze(-1)
        return fg, weight, fg.nonzero(as_tuple=False)[:, 0]

    def cls_r(batch):
        bce = get_bce()
        return bce.sum(dim=(1, 2)) if bce is not None else None

    def box_r(batch):
        st = _fg_state()
        iou = get_iou()
        if st is None or iou is None:
            return None
        fg, w, img_of_fg = st
        return _scatter(((1.0 - iou) * w).detach(), img_of_fg, fg.shape[0])

    def dfl_r(batch):
        st = _fg_state()
        dfl = get_dfl()
        if st is None or dfl is None:
            return None
        fg, w, img_of_fg = st
        return _scatter((dfl * w).detach(), img_of_fg, fg.shape[0])

    def overlay_p(batch):
        if get_det is None:
            return None
        raw = get_det()
        if raw is None:
            return None
        try:
            y = detect_head._inference(raw)
            nms = non_max_suppression(y, conf_thres=1e-4, iou_thres=0.45)
            return _overlay_dict(nms, batch["img"].shape[-2:])
        except Exception:
            return None

    return [
        Signal("train/cls_per_sample", "loss", reduce=cls_r, preds=overlay_p),
        Signal("train/box_per_sample", "loss", reduce=box_r),
        Signal("train/dfl_per_sample", "loss", reduce=dfl_r),
    ]


def default_val_signals(validator) -> list[Signal]:
    """The 2 default UL-detection val signals: per-image IoU + preds overlay."""
    def _iou_mapper(args, _out):
        predn, pbatch = args
        gt, pb = pbatch["bboxes"], predn["bboxes"]
        if gt.numel() > 0 and pb.numel() > 0:
            return float(ul_box_iou(gt, pb).max(dim=1).values.mean())
        if gt.numel() == 0 and pb.numel() == 0:
            return 1.0
        return 0.0

    get_iou_buf = per_call_buffer(validator, "_process_batch", _iou_mapper)

    def iou_r(batch):
        buf = get_iou_buf()
        if len(buf) != batch["img"].shape[0]:
            buf.clear()
            return None
        v = th.tensor(buf)
        buf.clear()
        return v

    def overlay_p(batch):
        preds = getattr(validator, "_wl_preds", None)
        return _overlay_dict(preds, batch["img"].shape[-2:]) if preds is not None else None

    def zero_r(batch):
        return th.zeros(batch["img"].shape[0])

    return [
        Signal("val/iou_per_sample",   "metric", reduce=iou_r),
        Signal("val/preds_per_sample", "metric", reduce=zero_r, preds=overlay_p),
    ]


# ─── top-level API (back-compat with the existing trainer.py calls) ────

def install_per_sample_signals(model):
    """Default train pipeline. Equivalent to:
        install_train_pipeline(model, default_train_signals(model))"""
    install_train_pipeline(model, default_train_signals(model))


def install_per_sample_val_signals(validator):
    """Default val pipeline. Equivalent to:
        install_val_pipeline(validator, default_val_signals(validator))"""
    install_val_pipeline(validator, default_val_signals(validator))
