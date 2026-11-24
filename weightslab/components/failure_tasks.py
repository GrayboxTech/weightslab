"""How "Why do these fail?" judges a sample, per task.

Every task turns a model output and its targets into one record per sample:

- ``score``: higher is better, in [0, 1]: 1 / 0 (classification: right /
  wrong), the image's mean IoU (segmentation), the F1 of its boxes
  (detection), the share of tokens predicted right (language model), or the
  share of train / val samples with a higher loss (your own per-sample loss);
- ``ok``: ``score >= threshold`` (classification: predicted right);
- ``value``: what the gate shows (the score, or the loss itself);
- ``caption``: one line for the cards ("3 → 7 (92%)", "IoU 0.31", ...);
- ``group`` / ``pred_group``: the true / predicted class, classification only
  (look-alikes of the same class, mislabeled suspects);
- ``parts``: per class, (numerator, denominator) summed over a set of samples
  for the per-class gate check (right / 1, intersection / union, found / boxes).

``better`` / ``worse`` compare two records of one sample (before / after a
mask or a repair): classification flips right <-> wrong; the other tasks
need a change of at least ``delta`` (crossing the threshold counts from a
quarter of it), so a sample sitting on the threshold does not flip on noise.

A task also knows the repair loss (``loss``) and what to draw on a thumbnail
(``marks``: the wrong pixels, the true / predicted boxes).

Which task: ``set_failure_task`` (or ``model.wl_failure_task``), else the
``task_type`` of the model / dataset, else the shapes of the output and the
targets. Detection needs boxes: a native output (torchvision-style dicts,
per-image ``[x1, y1, x2, y2, class, conf]`` rows), or
``set_failure_boxes_fn`` / ``model.wl_failure_boxes(output)``; without them,
any task falls back to the per-sample loss registered with
``wl.watch_or_edit(criterion, flag="loss")``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TASK: Any = None
_BOXES_FN: Optional[Callable] = None
_LOSS_FN: Optional[Callable] = None

_CLS_HINTS = {"classification", "image_classification", "text_classification", "tabular"}
_SEG_HINTS = {"segmentation", "semantic_segmentation", "seg"}
_DET_HINTS = {"detection", "object_detection", "det"}
_LM_HINTS = {"language_model", "language_modeling", "language_modelling", "lm", "llm", "causal_lm",
             "text_generation", "next_token", "text"}


class NotSupported(ValueError):
    """This model / task can't be analysed (the message says why)."""


def set_failure_task(task: Any) -> None:
    """Force the task: "classification", "segmentation", "detection",
    "language_model", "loss", or a :class:`Task` instance. ``None`` = detect."""
    global _TASK
    _TASK = task


def set_failure_boxes_fn(fn: Optional[Callable]) -> None:
    """Detection: ``fn(output)`` -> one ``[N, 6]`` array per image,
    ``[x1, y1, x2, y2, class, confidence]`` (WeightsLab's box rows), in the
    targets' coordinates (or normalised to [0, 1])."""
    global _BOXES_FN
    _BOXES_FN = fn


def set_failure_loss_fn(fn: Optional[Callable]) -> None:
    """``fn(output, targets)`` -> one loss per sample ``[B]``: scores the
    samples when no built-in task fits, and trains the repairs."""
    global _LOSS_FN
    _LOSS_FN = fn


# =============================================================================
# Helpers
# =============================================================================
def logits_of(output) -> Optional[th.Tensor]:
    """The prediction tensor of a model output (a tensor, ``.logits``, a dict)."""
    if isinstance(output, th.Tensor):
        return output
    value = getattr(output, "logits", None)
    if isinstance(value, th.Tensor):
        return value
    if isinstance(output, Mapping):
        for key in ("logits", "out", "output", "pred", "preds", "scores"):
            if isinstance(output.get(key), th.Tensor):
                return output[key]
    from weightslab.components.model_representation import _first_tensor
    return _first_tensor(output)


def _as_tensor(value) -> Optional[th.Tensor]:
    if value is None:
        return None
    if isinstance(value, th.Tensor):
        return value.detach()
    try:
        return th.as_tensor(np.asarray(value))
    except Exception:
        return None


def _int_like(t: Optional[th.Tensor]) -> bool:
    if t is None or t.numel() == 0:
        return False
    if not t.is_floating_point():
        return True
    sample = t.reshape(-1)[:4096].float()
    return bool(th.all(sample == sample.round()))


def _hint_of(*objs) -> str:
    for obj in objs:
        value = getattr(obj, "task_type", None) if obj is not None else None
        if isinstance(value, str) and value.strip() and value.strip().lower() != "unknown":
            return value.strip().lower()
    return ""


def _input_tensor(inputs) -> Optional[th.Tensor]:
    from weightslab.components.model_representation import _first_tensor
    if isinstance(inputs, Mapping):
        for key in ("input_ids", "img", "image", "images", "pixel_values", "x"):
            if isinstance(inputs.get(key), th.Tensor):
                return inputs[key]
    return _first_tensor(inputs)


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


# =============================================================================
# Tasks
# =============================================================================
class Task:
    name = "task"
    title = "task"              # for the user: "segmentation"
    metric = "score"            # what the gate shows
    grouped = False             # classes per sample (look-alikes of the same class)
    continuous = True           # the score is a level, not right / wrong
    delta = 0.1
    comfortable = 0.1           # look-alikes: at least this far above the threshold
    threshold = 0.5
    can_mark = False

    def __init__(self, inner: Optional[nn.Module] = None):
        self.inner = inner
        self.repair_note = ""

    # ---- per sample ---------------------------------------------------------
    def evaluate(self, output, targets, n: int, inputs=None) -> List[dict]:
        raise NotImplementedError

    def finalize(self, ev: dict) -> dict:
        ev["ok"] = bool(ev["score"] >= self.threshold)
        ev["margin"] = float(ev["score"] - self.threshold)
        ev.setdefault("conf", float(ev["score"]))
        ev.setdefault("value", float(ev["score"]))
        ev["caption"] = self.caption(ev)
        return ev

    def caption(self, ev: dict) -> str:
        return f"{self.metric} {ev['score']:.2f}"

    def better(self, before: dict, after: dict) -> bool:
        gain = float(after["score"]) - float(before["score"])
        return gain >= self.delta or (after["ok"] and not before["ok"] and gain >= self.delta / 4)

    def worse(self, before: dict, after: dict) -> bool:
        drop = float(before["score"]) - float(after["score"])
        return drop >= self.delta or (before["ok"] and not after["ok"] and drop >= self.delta / 4)

    # ---- wording --------------------------------------------------------------
    def failing_text(self) -> str:
        return f"{self.metric} below {self.threshold:.2f}"

    def fmt(self, value: float) -> str:
        return f"{value:.3f}"

    def fmt_delta(self, value: float) -> str:
        return f"{value:+.3f}"

    # ---- training / drawing -----------------------------------------------------
    def loss(self, output, targets, inputs=None) -> th.Tensor:
        raise NotSupported(f"I don't know how to train a {self.title} model")

    @property
    def can_train(self) -> bool:
        return True

    def marks(self, output, targets, n: int, inputs=None) -> List[Optional[dict]]:
        return [None] * n


class ClassificationTask(Task):
    name = "classification"
    title = "classification"
    metric = "accuracy"
    grouped = True
    continuous = False
    delta = 1.0
    comfortable = 0.2
    threshold = 0.5

    @staticmethod
    def labels(labels, batch: int) -> Optional[np.ndarray]:
        """(B,) int64 class indices from labels as ints, one-hot rows or (B, 1)."""
        if labels is None:
            return None
        if isinstance(labels, (list, tuple)):
            try:
                labels = th.as_tensor(np.asarray([np.asarray(v).reshape(-1)[0] for v in labels]))
            except Exception:
                return None
        labels = _as_tensor(labels)
        if labels is None:
            return None
        labels = labels.detach().cpu()
        if labels.ndim == 2 and labels.shape[0] == batch and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)
        labels = labels.reshape(-1)
        if labels.numel() != batch:
            return None
        if labels.is_floating_point():
            labels = labels.round()
        return labels.long().numpy()

    def _logits(self, output, batch: int) -> th.Tensor:
        logits = logits_of(output)
        if logits is None:
            raise NotSupported("the model returned no tensor")
        logits = logits.float()
        if logits.ndim != 2 or logits.shape[0] != batch:
            raise NotSupported(
                f"its output has shape {tuple(logits.shape)}; a classification output is "
                "(batch, classes)")
        if logits.shape[1] == 1:   # binary logit
            logits = th.cat([th.zeros_like(logits), logits], dim=1)
        return logits

    def classify(self, output, labels, batch: int) -> Dict[str, np.ndarray]:
        logits = self._logits(output, batch).detach()
        y = self.labels(labels, batch)
        if y is None:
            raise NotSupported("no class labels found in the batch")
        n_classes = int(logits.shape[1])
        if (y < 0).any() or (y >= n_classes).any():
            raise NotSupported(f"labels outside the {n_classes} classes")
        probs = th.softmax(logits, dim=1).cpu()
        pred = probs.argmax(dim=1)
        yt = th.as_tensor(y)
        p_true = probs.gather(1, yt.view(-1, 1)).view(-1)
        others = probs.clone()
        others.scatter_(1, yt.view(-1, 1), -1.0)
        return {
            "label": y,
            "pred": pred.numpy().astype(np.int64),
            "conf": probs.max(dim=1).values.numpy(),
            "p_true": p_true.numpy(),
            "margin": (p_true - others.max(dim=1).values).numpy(),
            "n_classes": np.asarray(n_classes),
        }

    def evaluate(self, output, targets, n: int, inputs=None) -> List[dict]:
        res = self.classify(output, targets, n)
        out = []
        for i in range(n):
            label, pred = int(res["label"][i]), int(res["pred"][i])
            right = label == pred
            out.append(self.finalize({
                "score": 1.0 if right else 0.0, "label": label, "pred": pred,
                "group": label, "pred_group": pred, "conf": float(res["conf"][i]),
                "p_true": float(res["p_true"][i]), "cls_margin": float(res["margin"][i]),
                "parts": {label: (1.0 if right else 0.0, 1.0)}}))
        return out

    def finalize(self, ev: dict) -> dict:
        ev["ok"] = ev["pred"] == ev["label"]
        ev["margin"] = ev["cls_margin"]
        ev["value"] = 1.0 if ev["ok"] else 0.0
        ev["caption"] = self.caption(ev)
        return ev

    def caption(self, ev: dict) -> str:
        if ev["ok"]:
            return f"{ev['label']} ({ev['p_true'] * 100:.0f}%)"
        return f"{ev['label']} → {ev['pred']} ({ev['conf'] * 100:.0f}%)"

    def failing_text(self) -> str:
        return "predicted wrong"

    def fmt(self, value: float) -> str:
        return f"{value * 100:.1f}%"

    def fmt_delta(self, value: float) -> str:
        return f"{value * 100:+.1f}"

    def loss(self, output, targets, inputs=None) -> th.Tensor:
        logits = logits_of(output)
        n = int(logits.shape[0])
        logits = self._logits(output, n)
        y = self.labels(targets, n)
        if y is None:
            raise NotSupported("no class labels found in the batch")
        return F.cross_entropy(logits, th.as_tensor(y, device=logits.device))


class SegmentationTask(Task):
    """(B, K, *S) logits against (B, *S) class maps: the image's mean IoU
    over the classes in its target or its prediction."""
    name = "segmentation"
    title = "segmentation"
    metric = "IoU"
    can_mark = True

    def __init__(self, inner=None, ignore_index: Optional[int] = None):
        super().__init__(inner)
        self.ignore_index = ignore_index

    def _prepare(self, output, targets, n: int, train: bool = False):
        logits = logits_of(output)
        if logits is None or logits.ndim < 3 or logits.shape[0] != n:
            raise NotSupported(f"a segmentation output is (batch, classes, H, W); got "
                               f"{tuple(logits.shape) if logits is not None else 'nothing'}")
        logits = logits.float() if train else logits.detach().float()
        if logits.shape[1] == 1:
            logits = th.cat([th.zeros_like(logits), logits], dim=1)
        k = int(logits.shape[1])
        y = _as_tensor(targets)
        if y is None:
            raise NotSupported("no target masks found in the batch")
        y = y.to(logits.device)
        if y.ndim == logits.ndim and y.shape[1] == k and y.is_floating_point() and k > 1:
            y = y.argmax(dim=1)                       # one-hot masks
        elif y.ndim == logits.ndim and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != logits.ndim - 1 or y.shape[0] != n:
            raise NotSupported(f"target masks of shape {tuple(y.shape)} don't match the output "
                               f"{tuple(logits.shape)}")
        if y.is_floating_point():
            y = (y >= 0.5) if (k == 2 and not _int_like(y)) else y.round()
        y = y.long()
        if tuple(y.shape[1:]) != tuple(logits.shape[2:]):
            mode = {1: "linear", 2: "bilinear", 3: "trilinear"}.get(y.ndim - 1, "nearest")
            logits = F.interpolate(logits, size=tuple(y.shape[1:]), mode=mode,
                                   **({} if mode == "nearest" else {"align_corners": False}))
        valid = (y >= 0) & (y < k)
        if self.ignore_index is not None and 0 <= int(self.ignore_index) < k:
            valid &= y != int(self.ignore_index)
        return logits, y, valid, k

    def evaluate(self, output, targets, n: int, inputs=None) -> List[dict]:
        logits, y, valid, k = self._prepare(output, targets, n)
        pred = logits.argmax(dim=1)
        idx = y.clamp(0, k - 1) * k + pred
        out = []
        for b in range(n):
            v = valid[b].reshape(-1)
            cm = th.bincount(idx[b].reshape(-1)[v], minlength=k * k).reshape(k, k).double().cpu().numpy()
            inter = np.diag(cm)
            union = cm.sum(axis=0) + cm.sum(axis=1) - inter
            present = union > 0
            ious = np.where(present, inter / np.maximum(union, 1), np.nan)
            score = float(np.nanmean(ious)) if present.any() else 1.0
            pixels = float(inter.sum() / max(cm.sum(), 1))
            worst = int(np.nanargmin(ious)) if present.sum() > 1 else -1
            out.append(self.finalize({
                "score": score, "pixels": pixels,
                "worst": (worst, float(ious[worst])) if worst >= 0 else None,
                "parts": {int(c): (float(inter[c]), float(union[c])) for c in np.flatnonzero(present)}}))
        return out

    def caption(self, ev: dict) -> str:
        text = f"IoU {ev['score']:.2f}"
        worst = ev.get("worst")
        if worst is not None and not ev.get("ok", True):
            text += f" · class {worst[0]}: {worst[1]:.2f}"
        return text

    def loss(self, output, targets, inputs=None) -> th.Tensor:
        logits = logits_of(output)
        logits, y, valid, _ = self._prepare(output, targets, int(logits.shape[0]), train=True)
        return F.cross_entropy(logits, th.where(valid, y, th.full_like(y, -100)), ignore_index=-100)

    def marks(self, output, targets, n: int, inputs=None) -> List[Optional[dict]]:
        logits, y, valid, _ = self._prepare(output, targets, n)
        wrong = ((logits.argmax(dim=1) != y) & valid)
        out = []
        for b in range(n):
            err = wrong[b]
            while err.ndim > 2:                       # a volume: its middle slice
                err = err[err.shape[0] // 2]
            if err.ndim == 1:
                err = err.unsqueeze(0)
            err = err.float()[None, None]
            if max(err.shape[-2:]) > 128:
                scale = 128.0 / max(err.shape[-2:])
                err = F.interpolate(err, size=(max(1, int(err.shape[-2] * scale)),
                                               max(1, int(err.shape[-1] * scale))), mode="nearest")
            out.append({"kind": "mask", "error": err[0, 0].cpu().numpy() > 0.5})
        return out


class LanguageModelTask(Task):
    """(B, T, V) logits against (B, T) next tokens: the share of tokens
    predicted right (teacher forcing). Labels equal to the inputs (the
    Hugging Face convention) are shifted by one."""
    name = "language_model"
    title = "language model"
    metric = "token accuracy"

    def __init__(self, inner=None):
        super().__init__(inner)
        self.shift: Optional[bool] = None

    def _prepare(self, output, targets, n: int, inputs=None, train: bool = False):
        logits = logits_of(output)
        if logits is None or logits.ndim != 3 or logits.shape[0] != n:
            raise NotSupported(f"a language-model output is (batch, positions, vocabulary); got "
                               f"{tuple(logits.shape) if logits is not None else 'nothing'}")
        logits = logits if train else logits.detach()
        y = _as_tensor(targets)
        if y is None or y.ndim != 2 or y.shape[0] != n:
            raise NotSupported("no next-token targets (batch, positions) found in the batch")
        y = y.to(logits.device).long()
        if self.shift is None:
            x = _input_tensor(inputs)
            self.shift = bool(x is not None and tuple(x.shape) == tuple(y.shape)
                              and th.equal(x.to(y.device).long(), y))
        if self.shift:
            logits, y = logits[:, :-1], y[:, 1:]
        if logits.shape[1] != y.shape[1]:
            t = min(int(logits.shape[1]), int(y.shape[1]))
            logits, y = logits[:, -t:], y[:, -t:]
        v = int(logits.shape[2])
        valid = (y >= 0) & (y < v)
        return logits, y, valid, v

    def _text(self, ids: Sequence[int]) -> List[str]:
        from weightslab.components.model_representation import _token_text
        text = _token_text(self.inner, list(ids))
        return text if len(text) == len(ids) else [f"#{int(i)}" for i in ids]

    def evaluate(self, output, targets, n: int, inputs=None) -> List[dict]:
        logits, y, valid, v = self._prepare(output, targets, n, inputs)
        out = []
        for b in range(n):
            lp = F.log_softmax(logits[b].float(), dim=-1)           # (T, V), one sample at a time
            yb, vb = y[b], valid[b]
            top = lp.max(dim=-1)
            true_lp = lp.gather(-1, yb.clamp(0, v - 1).unsqueeze(-1)).squeeze(-1)
            right = (top.indices == yb) & vb
            count = int(vb.sum())
            acc = float(right.sum()) / max(count, 1) if count else 1.0
            nll = float(-(true_lp * vb).sum()) / max(count, 1)
            miss = None
            wrong = vb & ~right
            if bool(wrong.any()):
                gap = th.where(wrong, top.values - true_lp, th.full_like(true_lp, -1e9))
                j = int(gap.argmax())
                want, said = self._text([int(yb[j]), int(top.indices[j])])
                miss = (want, said)
            out.append(self.finalize({"score": acc, "loss": nll, "miss": miss, "tokens": count}))
        return out

    def caption(self, ev: dict) -> str:
        text = f"{_pct(ev['score'])} tokens right · loss {ev['loss']:.2f}"
        if ev.get("miss") and not ev.get("ok", True):
            want, said = ev["miss"]
            text += f" · “{want}” → “{said}”"
        return text

    def fmt(self, value: float) -> str:
        return f"{value * 100:.1f}%"

    def fmt_delta(self, value: float) -> str:
        return f"{value * 100:+.1f}"

    def failing_text(self) -> str:
        return f"under {_pct(self.threshold)} of the tokens predicted right"

    def loss(self, output, targets, inputs=None) -> th.Tensor:
        logits = logits_of(output)
        logits, y, valid, v = self._prepare(output, targets, int(logits.shape[0]), inputs, train=True)
        return F.cross_entropy(logits.float().reshape(-1, v),
                               th.where(valid, y, th.full_like(y, -100)).reshape(-1), ignore_index=-100)


# ---- detection ------------------------------------------------------------------
def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _np(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, th.Tensor):
        return value.detach().float().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except Exception:
        return None


def gt_boxes(targets, n: int) -> Optional[List[np.ndarray]]:
    """Per image ``[M, 5]`` rows ``x1, y1, x2, y2, class`` from WeightsLab
    box rows (``[x1, y1, x2, y2, class(, conf)]``), torchvision-style dicts,
    or an Ultralytics batch (``bboxes`` xywh, ``cls``, ``batch_idx``)."""
    if isinstance(targets, Mapping) and "bboxes" in targets and "batch_idx" in targets:
        boxes, cls = _np(targets["bboxes"]), _np(targets.get("cls"))
        idx = _np(targets["batch_idx"])
        if boxes is None or idx is None:
            return None
        boxes = boxes.reshape(-1, 4)
        cls = cls.reshape(-1) if cls is not None else np.zeros(len(boxes))
        idx = idx.reshape(-1).astype(int)
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2, cls], axis=1)
        return [xyxy[idx == b] for b in range(n)]
    if not isinstance(targets, (list, tuple)) or len(targets) != n:
        return None
    out = []
    for t in targets:
        if isinstance(t, Mapping):
            boxes = _np(t.get("boxes", t.get("bboxes")))
            if boxes is None:
                return None
            boxes = boxes.reshape(-1, 4)
            labels = _np(t.get("labels", t.get("cls")))
            labels = labels.reshape(-1) if labels is not None and labels.size == len(boxes) \
                else np.zeros(len(boxes))
            out.append(np.concatenate([boxes, labels[:, None]], axis=1))
            continue
        arr = _np(t)
        if arr is None:
            return None
        if arr.size == 0:
            out.append(np.zeros((0, 5)))
            continue
        if arr.ndim == 1 and arr.shape[0] in (4, 5, 6):
            arr = arr[None]
        if arr.ndim != 2 or arr.shape[1] < 4 or arr.shape[1] > 6:
            return None
        cls = arr[:, 4] if arr.shape[1] >= 5 else np.zeros(len(arr))
        out.append(np.concatenate([arr[:, :4], cls[:, None]], axis=1))
    return out


def _looks_like_boxes(targets, n: int) -> bool:
    """Box targets without a hint: an Ultralytics batch, or one dict / 2-D
    ``[M, 4..6]`` array per image (never a tensor of labels or masks)."""
    if isinstance(targets, Mapping):
        return gt_boxes(targets, n) is not None
    if not isinstance(targets, (list, tuple)) or len(targets) != n or not n:
        return False
    for t in targets:
        if isinstance(t, Mapping):
            continue
        arr = _np(t)
        if arr is None or not ((arr.ndim == 2 and 4 <= arr.shape[-1] <= 6) or arr.size == 0):
            return False
    return gt_boxes(targets, n) is not None


def native_boxes(output, n: int) -> Optional[List[np.ndarray]]:
    """Per image ``[N, 6]`` rows ``x1, y1, x2, y2, class, conf`` from a
    torchvision-style output (dicts of boxes / labels / scores) or a list of
    WeightsLab box rows, else None."""
    if not isinstance(output, (list, tuple)) or len(output) != n:
        return None
    out = []
    for item in output:
        if isinstance(item, Mapping) and "boxes" in item:
            boxes = _np(item["boxes"]).reshape(-1, 4)
            labels = _np(item.get("labels"))
            scores = _np(item.get("scores"))
            labels = labels.reshape(-1) if labels is not None and labels.size == len(boxes) \
                else np.zeros(len(boxes))
            scores = scores.reshape(-1) if scores is not None and scores.size == len(boxes) \
                else np.ones(len(boxes))
            out.append(np.concatenate([boxes, labels[:, None], scores[:, None]], axis=1))
            continue
        arr = _np(item)
        if arr is None or (arr.size and (arr.ndim != 2 or arr.shape[1] not in (5, 6))):
            return None
        if arr.size == 0:
            out.append(np.zeros((0, 6)))
        elif arr.shape[1] == 5:
            out.append(np.concatenate([arr, np.ones((len(arr), 1))], axis=1))
        else:
            out.append(arr)
    return out


def _image_size(inputs) -> Tuple[float, float]:
    x = _input_tensor(inputs)
    if x is not None and x.ndim >= 3:
        return float(x.shape[-1]), float(x.shape[-2])
    return 1.0, 1.0


def _normalized(rows: np.ndarray, w: float, h: float, pixels: bool) -> np.ndarray:
    if not pixels or rows.size == 0:
        return rows
    rows = rows.copy()
    rows[:, [0, 2]] /= max(w, 1.0)
    rows[:, [1, 3]] /= max(h, 1.0)
    return rows


class DetectionTask(Task):
    """The F1 of an image's boxes at IoU >= ``iou`` (same class), counting the
    predictions with confidence >= ``conf``."""
    name = "detection"
    title = "detection"
    metric = "F1"
    can_mark = True

    def __init__(self, inner=None, boxes_fn: Optional[Callable] = None, iou: float = 0.5,
                 conf: float = 0.25, loss_fn: Optional[Callable] = None, loss_name: str = ""):
        super().__init__(inner)
        self.boxes_fn = boxes_fn
        self.iou = float(iou)
        self.conf = float(conf)
        self.loss_fn = loss_fn
        self.loss_name = loss_name

    def _boxes(self, output, targets, n, inputs):
        if self.boxes_fn is not None:
            try:
                raw = self.boxes_fn(output)
            except TypeError:
                raw = self.boxes_fn(output, inputs)
            preds = []
            for p in raw or []:
                arr = _np(p)
                if arr is None or arr.size == 0:
                    preds.append(np.zeros((0, 6)))
                    continue
                arr = arr.reshape(-1, arr.shape[-1])
                if arr.shape[1] == 5:
                    arr = np.concatenate([arr, np.ones((len(arr), 1))], axis=1)
                preds.append(arr[:, :6])
        else:
            preds = native_boxes(output, n)
        gts = gt_boxes(targets, n)
        if preds is None or len(preds) != n:
            raise NotSupported("the detection output isn't a list of boxes per image "
                               "(see set_failure_boxes_fn)")
        if gts is None:
            raise NotSupported("no target boxes per image found in the batch")
        w, h = _image_size(inputs)
        p_pix = any(p.size and float(np.max(p[:, :4])) > 1.5 for p in preds)
        g_pix = any(g.size and float(np.max(g[:, :4])) > 1.5 for g in gts)
        if p_pix != g_pix:                     # one side normalised: normalise both
            preds = [_normalized(p, w, h, p_pix) for p in preds]
            gts = [_normalized(g, w, h, g_pix) for g in gts]
        return preds, gts, w, h, (p_pix and g_pix)

    def evaluate(self, output, targets, n: int, inputs=None) -> List[dict]:
        preds, gts, *_ = self._boxes(output, targets, n, inputs)
        out = []
        for p, g in zip(preds, gts):
            p = p[p[:, 5] >= self.conf] if p.size else p
            p = p[np.argsort(-p[:, 5])] if p.size else p
            used = np.zeros(len(g), dtype=bool)
            iou = _box_iou(p[:, :4], g[:, :4]) if p.size and g.size else np.zeros((len(p), len(g)))
            tp = 0
            found: Dict[int, float] = {}
            for i in range(len(p)):
                if not len(g):
                    break
                cand = np.where(~used & (g[:, 4].round() == round(float(p[i, 4]))), iou[i], -1.0)
                j = int(np.argmax(cand))
                if cand[j] >= self.iou:
                    used[j] = True
                    tp += 1
                    c = int(round(float(g[j, 4])))
                    found[c] = found.get(c, 0.0) + 1.0
            fp, fn = len(p) - tp, len(g) - tp
            score = 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
            parts: Dict[int, Tuple[float, float]] = {}
            for c in np.unique(g[:, 4].round()).astype(int) if len(g) else []:
                parts[int(c)] = (found.get(int(c), 0.0), float(np.sum(g[:, 4].round() == c)))
            out.append(self.finalize({"score": float(score), "tp": tp, "fp": fp, "fn": fn,
                                      "n_gt": int(len(g)), "parts": parts}))
        return out

    def caption(self, ev: dict) -> str:
        bits = [f"F1 {ev['score']:.2f}"]
        if ev["fn"]:
            bits.append(f"{ev['fn']} missed")
        if ev["fp"]:
            bits.append(f"{ev['fp']} extra")
        if not ev["fn"] and not ev["fp"]:
            bits.append(f"all {ev['n_gt']} found" if ev["n_gt"] else "nothing to find")
        return " · ".join(bits)

    @property
    def can_train(self) -> bool:
        return self.loss_fn is not None

    def loss(self, output, targets, inputs=None) -> th.Tensor:
        if self.loss_fn is None:
            raise NotSupported("a detection repair needs your loss: register it with "
                               "wl.watch_or_edit(criterion, flag='loss') or set_failure_loss_fn")
        return _per_sample(self.loss_fn(output, targets), None).mean()

    def marks(self, output, targets, n: int, inputs=None) -> List[Optional[dict]]:
        preds, gts, w, h, both_pix = self._boxes(output, targets, n, inputs)
        out = []
        for p, g in zip(preds, gts):
            p = p[p[:, 5] >= self.conf] if p.size else p
            if both_pix:
                p, g = _normalized(p, w, h, True), _normalized(g, w, h, True)
            out.append({"kind": "boxes", "gt": g[:, :4].tolist(), "pred": p[:, :4].tolist()})
        return out


# ---- your own loss ----------------------------------------------------------------
def _per_sample(values, n: Optional[int]) -> th.Tensor:
    t = values if isinstance(values, th.Tensor) else logits_of(values)
    if t is None:
        raise NotSupported("the loss returned no tensor")
    if t.ndim == 0:
        raise NotSupported("the loss returns one value per batch; it needs reduction='none'")
    if n is not None and t.shape[0] != n:
        raise NotSupported(f"the loss returned {t.shape[0]} values for {n} samples")
    return t.float().reshape(t.shape[0], -1).mean(dim=1)


class LossTask(Task):
    """Your per-sample loss, ranked against train / val samples: the score is
    the share of reference samples with a higher loss. A sample fails when
    its loss is higher than 75% of them (``threshold`` 0.25)."""
    name = "loss"
    metric = "loss"
    threshold = 0.25

    def __init__(self, inner, fn: Callable, source: str, title: str = "model", boxes: bool = False):
        super().__init__(inner)
        self.fn = fn
        self.source = source
        self.title = title
        self.boxes = boxes
        self.ref: Optional[np.ndarray] = None
        self.can_mark = boxes

    def set_reference(self, losses: Sequence[float]) -> None:
        ref = np.sort(np.asarray([v for v in losses if np.isfinite(v)], dtype=np.float64))
        self.ref = ref if ref.size else None

    def rank(self, loss: float) -> float:
        if self.ref is None or not np.isfinite(loss):
            return 0.0 if not np.isfinite(loss) else 0.5
        return float(1.0 - np.searchsorted(self.ref, loss, side="left") / len(self.ref))

    def evaluate(self, output, targets, n: int, inputs=None) -> List[dict]:
        with th.no_grad():
            losses = _per_sample(self.fn(output, targets), n).cpu().numpy()
        return [self.finalize({"loss": float(v), "score": self.rank(float(v))}) for v in losses]

    def finalize(self, ev: dict) -> dict:
        ev["score"] = self.rank(ev["loss"])
        ev["value"] = ev["loss"]
        ev["conf"] = ev["score"]
        return super().finalize(ev)

    def caption(self, ev: dict) -> str:
        if self.ref is None:
            return f"loss {ev['loss']:.3g}"
        return f"loss {ev['loss']:.3g} · higher than {_pct(1.0 - ev['score'])} of train / val"

    def failing_text(self) -> str:
        return f"a loss higher than {_pct(1.0 - self.threshold)} of train / val"

    def fmt(self, value: float) -> str:
        return f"{value:.4g}"

    def fmt_delta(self, value: float) -> str:
        return f"{value:+.3g}"

    def loss(self, output, targets, inputs=None) -> th.Tensor:
        return _per_sample(self.fn(output, targets), None).mean()

    def marks(self, output, targets, n: int, inputs=None) -> List[Optional[dict]]:
        gts = gt_boxes(targets, n) if self.boxes else None
        if gts is None:
            return [None] * n
        w, h = _image_size(inputs)
        return [{"kind": "boxes", "gt": _normalized(g, w, h, bool(g.size and np.max(g[:, :4]) > 1.5))[:, :4]
                 .tolist(), "pred": []} for g in gts]


def user_loss(inner: Optional[nn.Module]) -> Tuple[Optional[Callable], str]:
    """The per-sample loss to score / train with: ``set_failure_loss_fn``,
    ``model.wl_failure_loss``, else a criterion registered with
    ``flag="loss"`` (its original forward: no signal is logged)."""
    fn = _LOSS_FN or getattr(inner, "wl_failure_loss", None)
    if callable(fn):
        return fn, "your failure loss"
    names: List[str] = []
    try:
        from weightslab import src
        names = list(src.auto_loss_shape_signal_names())
    except Exception:
        pass
    try:
        from weightslab.backend.ledgers import get_signal, list_signals
        names += [n for n in list_signals() if "loss" in str(n).lower() and n not in names]
    except Exception:
        return None, ""
    names.sort(key=lambda n: (0 if "train" in str(n).lower() else 1, str(n)))
    for name in names:
        try:
            obj = get_signal(name)
            forward = getattr(obj, "forward", None)
            raw = getattr(forward, "__wrapped__", None)
        except Exception:
            continue
        if callable(raw):
            return raw, f"your loss `{name}`"
    return None, ""


# =============================================================================
# Which task
# =============================================================================
def detect_task(output, targets, n: int, inputs, inner: Optional[nn.Module],
                datasets: Sequence[Any] = (), options: Optional[Mapping[str, Any]] = None) -> Task:
    """The task of this model, from a first batch: forced, hinted, or guessed
    from the shapes; the registered per-sample loss when nothing fits."""
    opts = dict(options or {})
    forced = _TASK if _TASK is not None else getattr(inner, "wl_failure_task", None)
    if isinstance(forced, Task):
        forced.inner = forced.inner or inner
        return forced
    hint = str(forced).strip().lower() if isinstance(forced, str) and forced else \
        _hint_of(inner, *datasets)
    ignore = next((getattr(d, "ignore_index", None) for d in datasets
                   if getattr(d, "ignore_index", None) is not None), None)
    boxes_fn = _BOXES_FN or getattr(inner, "wl_failure_boxes", None)
    loss_fn, loss_name = user_loss(inner)
    det_iou, det_conf = float(opts.get("det_iou", 0.5)), float(opts.get("det_conf", 0.25))

    def fallback(why: str, boxes: bool = False, title: str = "model") -> Task:
        if loss_fn is not None:
            task = LossTask(inner, loss_fn, loss_name, title=title, boxes=boxes)
            try:
                task.evaluate(output, targets, n, inputs)
                task.repair_note = why
                return task
            except Exception as exc:
                why = f"{why}; {loss_name} didn't give one value per sample ({exc})"
        raise NotSupported(why)

    def first_fit(tasks: Sequence[Task]) -> Tuple[Optional[Task], str]:
        why = ""
        for task in tasks:
            try:
                task.evaluate(output, targets, n, inputs)   # raises NotSupported with the reason
                return task, ""
            except NotSupported as exc:
                why = why or str(exc)
            except Exception as exc:
                why = why or f"{type(exc).__name__}: {exc}"
        return None, why

    if hint == "loss":
        return fallback("you asked for the loss")
    looks_det = _looks_like_boxes(targets, n)
    if hint.startswith("detection") or hint in _DET_HINTS or (looks_det and hint not in _CLS_HINTS):
        if hint == "detection_pointcloud":
            return fallback("3-D boxes aren't scored yet", title="3-D detection")
        if boxes_fn is not None or native_boxes(output, n) is not None:
            task, why = first_fit([DetectionTask(inner, boxes_fn, det_iou, det_conf, loss_fn, loss_name)])
            return task or fallback(why, boxes=True, title="detection")
        return fallback("its detection output needs decoding into boxes (set_failure_boxes_fn)",
                        boxes=True, title="detection")

    # The hinted task first, then what the shapes suggest.
    candidates: List[Task] = []
    if hint in _SEG_HINTS:
        candidates.append(SegmentationTask(inner, ignore))
    elif hint in _LM_HINTS:
        candidates.append(LanguageModelTask(inner))
    elif hint in _CLS_HINTS:
        candidates.append(ClassificationTask(inner))
    logits = logits_of(output)
    y = _as_tensor(targets) if isinstance(targets, (th.Tensor, np.ndarray)) or (
        isinstance(targets, (list, tuple)) and targets and all(np.ndim(_np(t)) == 0 for t in targets)) else None
    if logits is not None and y is not None and logits.shape[0] == n:
        x = _input_tensor(inputs)
        tokens = x is not None and not x.is_floating_point() and x.ndim == 2
        if logits.ndim == 2:
            candidates.append(ClassificationTask(inner))
        elif logits.ndim == 3 and y.ndim == 2 and _int_like(y):
            lm_fit = y.shape[1] in (logits.shape[1], logits.shape[1] + 1) or \
                (tokens and tuple(x.shape) == tuple(y.shape))
            seg_fit = y.shape[1] == logits.shape[2]
            pair = [LanguageModelTask(inner), SegmentationTask(inner, ignore)]
            candidates.extend(pair if lm_fit and (tokens or not seg_fit) else pair[::-1])
        elif logits.ndim >= 3 and (y.ndim == logits.ndim - 1 or
                                   (y.ndim == logits.ndim and y.shape[1] in (1, logits.shape[1]))):
            candidates.append(SegmentationTask(inner, ignore))
    task, why = first_fit(candidates)
    if task is not None:
        return task
    shape = tuple(logits.shape) if logits is not None else None
    return fallback(why or f"I couldn't tell the task from an output of shape {shape} "
                    "(set_failure_task)", title=candidates[0].title if candidates else "model")


# =============================================================================
# Drawing on thumbnails
# =============================================================================
def draw_mark(img, mark: Optional[dict]):
    """Draw *mark* on a PIL RGB image: wrong pixels in red, true boxes in
    green, predicted boxes in orange."""
    if not mark:
        return img
    try:
        from PIL import Image, ImageDraw
        w, h = img.size
        if mark.get("kind") == "mask" and mark.get("error") is not None:
            err = np.asarray(mark["error"], dtype=np.uint8) * 255
            if err.ndim == 2 and err.any():
                mask = Image.fromarray(err).resize((w, h), Image.NEAREST)
                red = Image.new("RGB", (w, h), (235, 40, 40))
                img = Image.composite(Image.blend(img, red, 0.6), img, mask)
        elif mark.get("kind") == "boxes":
            draw = ImageDraw.Draw(img)
            for rows, colour in ((mark.get("gt") or [], (40, 200, 90)), (mark.get("pred") or [], (255, 140, 0))):
                for x1, y1, x2, y2 in rows[:50]:
                    box = [x1 * (w - 1), y1 * (h - 1), x2 * (w - 1), y2 * (h - 1)]
                    if box[2] > box[0] and box[3] > box[1]:
                        draw.rectangle(box, outline=colour, width=1)
    except Exception as exc:
        logger.debug(f"[failures] could not draw on a thumbnail: {exc}")
    return img
