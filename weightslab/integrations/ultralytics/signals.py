"""Per-sample signal pipelines for UL detection — pure taps over UL, no math
reimplementation.

Each signal is a `Signal(name, flag, reduce, preds=None)` record. Capture
primitives (`fwd_hook`, `pre_hook`, `fn_tap`, `method_call_tap`,
`per_call_buffer`) install a hook on a UL internal and return a zero-arg
getter for the captured state; closures bind the getter into each
signal's `reduce` / `preds` function.

A *pipeline* (`install_train_pipeline`, `install_val_pipeline`) is the
one imperative bit: it wraps a UL sync method, runs each signal's reduce
after the original ran, and ships to a WL channel keyed by `batch["ids"]`.

Adding signal N+1 is appending one `Signal(...)` record to a list. No
edits to the orchestrator, no new hook wiring unless the new signal taps
a place no existing signal taps yet.

Public:
    Signal, fwd_hook, pre_hook, fn_tap, method_call_tap, per_call_buffer,
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

from weightslab.utils.tools import filter_kwargs_for_callable

from ._utils import normalize_post_nms_preds


# ─── overlay fallback tunables ─────────────────────────────────────────
# Train overlay's NMS pulls thresholds from `model.args.{conf,iou}` (the
# values you'd pass through `YOLO.train(conf=..., iou=...)`). These
# fallbacks kick in only if `model.args.{conf,iou}` is `None` — which
# happens for training because UL only auto-populates those for predict.
#
#   OVERLAY_CONF_FALLBACK — tiny so early-epoch overlays aren't empty.
#                           UL's predict default of 0.25 would hide the
#                           model entirely while it's still learning.
#   OVERLAY_IOU_FALLBACK  — matches UL's default inference IoU.
#   OVERLAY_MAX_DETS      — readability cap; UL's NMS otherwise produces
#                           up to 300 boxes per image, flooding the studio.
OVERLAY_CONF_FALLBACK = 1e-4
OVERLAY_IOU_FALLBACK = 0.45
OVERLAY_MAX_DETS = 50


def _overlay_nms_thresholds(model):
    """Read `(conf, iou)` from `model.args` with module fallback."""
    args = getattr(model, "args", None)
    conf = getattr(args, "conf", None) if args is not None else None
    iou = getattr(args, "iou", None) if args is not None else None
    return (
        conf if conf is not None else OVERLAY_CONF_FALLBACK,
        iou if iou is not None else OVERLAY_IOU_FALLBACK,
    )


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


def method_call_tap(obj, attr: str) -> Callable[[], Any]:
    """Capture the return value of `obj.attr(...)` when `attr` is a callable
    whose `__call__` is NOT routed through `nn.Module.__call__` (so
    `register_forward_hook` never fires on it — UL's `DFLoss` is the case
    in point). Replace `obj.attr` with a proxy that forwards attribute
    access to the original and intercepts the call.

    If `obj` is an `nn.Module` and `attr` is a registered child module,
    `setattr` would reject our non-Module proxy; pop the child from
    `_modules` first and then write through `__dict__` directly."""
    inner = getattr(obj, attr)
    box: dict = {"v": None}

    class _Tap:
        def __getattr__(self, n): return getattr(inner, n)
        def __call__(self, *a, **kw):
            out = inner(*a, **kw)
            box["v"] = out.detach() if hasattr(out, "detach") else out
            return out

    if isinstance(obj, th.nn.Module) and attr in obj._modules:
        del obj._modules[attr]
        obj.__dict__[attr] = _Tap()
    else:
        setattr(obj, attr, _Tap())
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
    """Iterate the signal list, ship anything the reducers produce. Wrapped
    in `no_grad` because signal capture is observational — running e.g.
    `Detect._inference` + NMS inside the train autograd graph would keep
    those tensors alive until backward and compound across steps."""
    ids = batch["ids"]
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
        validator._wl_preds = preds   # exposed to signal reducers/predsers
        res = _orig(preds, batch)     # runs first — fills _process_batch buf
        _ship_round(signals, channels, batch)
        return res
    validator.update_metrics = _ship


# ─── helpers used by the default packs ─────────────────────────────────

def _scatter(per_fg, img_of_fg, bs):
    out = th.zeros(bs, device=per_fg.device, dtype=per_fg.dtype)
    out.scatter_add_(0, img_of_fg, per_fg.reshape(-1))
    return out


def _overlay_dict(nms_preds, img_hw):
    """UL post-NMS list → studio overlay dict, capped at `OVERLAY_MAX_DETS`
    per image by confidence, coords normalised to [0,1]."""
    h, w = img_hw
    scale = th.tensor([w, h, w, h], dtype=th.float32)
    out = []
    for p in normalize_post_nms_preds(nms_preds):
        if p.numel() == 0:
            out.append(th.zeros((0, 6)))
            continue
        pc = p.detach().cpu().float()
        if pc.shape[0] > OVERLAY_MAX_DETS:
            pc = pc[pc[:, 4].argsort(descending=True)[:OVERLAY_MAX_DETS]]
        out.append(th.cat([pc[:, :4] / scale, pc[:, 5:6], pc[:, 4:5]], -1))
    return {"bboxes": out}


# ─── default signal packs ──────────────────────────────────────────────

def default_train_signals(model, signals_cfg: dict = {}) -> list[Signal]:
    """Default UL-detection train signals: per-sample cls/box/dfl. If the
    model exposes a `Detect` head, a live preds overlay rides on the cls
    signal. Compose with user signals: `default_train_signals(m) + [...]`."""
    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()
    crit = model.criterion
    bl = crit.bbox_loss
    detect_head = next((m for m in model.modules() if isinstance(m, Detect)), None)

    get_bce = fwd_hook(crit.bce)                # bce is a plain nn.Module
    get_iou = fn_tap(ul_loss, "bbox_iou")       # bbox_iou is a plain function
    get_dfl = method_call_tap(bl, "dfl_loss")   # DFLoss overrides __call__
    get_bl_args = pre_hook(bl)

    def _fg_state():
        # bbox_loss isn't called if every anchor is background; the
        # downstream getters then have no fresh value, so callers must
        # treat None as "skip this round".
        args = get_bl_args()
        if args is None:
            return None
        fg = args[6]
        if not fg.any():
            return None
        weight = args[4].sum(-1)[fg].unsqueeze(-1)
        return fg, weight, fg.nonzero(as_tuple=False)[:, 0]

    def cls_r(_batch):
        return get_bce().sum(dim=(1, 2))

    def _fresh(val, w):
        # A tapped value is only this step's if it pairs 1:1 with the fg
        # weights captured by the bbox_loss pre-hook. When this step assigns
        # zero foreground, bbox_loss is skipped (loss.py: `if fg_mask.sum()`)
        # so the taps stay stale — and `get_iou` in particular may hold an
        # EMA-validation value (validator.py runs `model.loss` on the EMA
        # model, refreshing the *global* bbox_iou tap but not this train
        # instance's pre-hook). Mismatched lengths ⇒ stale ⇒ skip the round.
        return val is not None and val.shape[0] == w.shape[0]

    def box_r(_batch):
        st = _fg_state()
        if st is None:
            return None
        fg, w, img_of_fg = st
        iou = get_iou()
        if not _fresh(iou, w):
            return None
        return _scatter(((1.0 - iou) * w).detach(), img_of_fg, fg.shape[0])

    def dfl_r(_batch):
        st = _fg_state()
        if st is None:
            return None
        fg, w, img_of_fg = st
        dfl = get_dfl()
        if not _fresh(dfl, w):
            return None
        return _scatter((dfl * w).detach(), img_of_fg, fg.shape[0])

    signals = [
        Signal("train/cls_per_sample", "loss", reduce=cls_r),
        Signal("train/box_per_sample", "loss", reduce=box_r),
        Signal("train/dfl_per_sample", "loss", reduce=dfl_r),
    ]

    # Live preds overlay on the cls signal — only if there's a Detect head.
    # Reuses UL's own `_inference` + `non_max_suppression` (no math rewrite).
    if detect_head is not None:
        get_det = fwd_hook(detect_head)
        nms_cfg = filter_kwargs_for_callable(non_max_suppression, signals_cfg.get('nms', signals_cfg.get('train_nms', {})))
        conf_thres, iou_thres = _overlay_nms_thresholds(model)
        conf_thres = conf_thres if nms_cfg.get('conf_thres') is None else nms_cfg.get('conf_thres')
        iou_thres = iou_thres if nms_cfg.get('iou_thres') is None else nms_cfg.get('iou_thres')
        nms_cfg['conf_thres'] = conf_thres
        nms_cfg['iou_thres'] = iou_thres

        def overlay_p(batch):
            y = detect_head._inference(get_det())
            nms = non_max_suppression(
                y,
                **nms_cfg
            )
            return _overlay_dict(nms, batch["img"].shape[-2:])
        signals[0].preds = overlay_p

    return signals


def default_val_signals(validator, signals_cfg: dict = {}) -> list[Signal]:
    """Default UL-detection val signals: per-image IoU with live preds
    overlay riding on the IoU signal."""
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
        preds = validator._wl_preds
        return _overlay_dict(preds, batch["img"].shape[-2:])

    return [
        Signal("val/iou_per_sample", "metric", reduce=iou_r, preds=overlay_p),
    ]


# ─── top-level API (back-compat with the existing trainer.py calls) ────

def install_per_sample_signals(model, signals_cfg: dict = {}):
    """Default train pipeline. Equivalent to:
        install_train_pipeline(model, default_train_signals(model))"""
    install_train_pipeline(model, default_train_signals(model, signals_cfg=signals_cfg))


def install_per_sample_val_signals(validator, signals_cfg: dict = {}):
    """Default val pipeline. Equivalent to:
        install_val_pipeline(validator, default_val_signals(validator))"""
    install_val_pipeline(validator, default_val_signals(validator, signals_cfg=signals_cfg))
