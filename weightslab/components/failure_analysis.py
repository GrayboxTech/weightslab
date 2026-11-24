"""Why do these samples fail? -- a guided failure analysis and repair.

A user picks samples the model gets wrong (a tag, a grid selection) and an
agent-like session walks through six steps, asking before each one unless it
runs automatically:

1. **check** -- run the model on them: which are wrong now (true -> predicted)?
2. **lookalikes** -- for each failure, the most similar samples of the same
   class that the model gets right, similar in the model's OWN features (the
   input of its last layer), searched in train / val. They are the control
   group: failures vs look-alikes isolates the failure, not the class. The
   same neighbours flag failures that look mislabeled (their neighbours carry
   the predicted label, the model is sure) and failures unlike any training
   sample (out of distribution).
3. **compare** -- per layer, Cohen's d of every neuron, failures vs
   look-alikes; the layer where the difference starts (the first jump; later
   layers mostly inherit it); per chosen layer, the suspects: neurons whose
   |d| (of their mean OR their max -- a feature on a small region is averaged
   away otherwise) beats a shuffled-label baseline.
4. **test** -- the causal check: replace the suspects by their look-alike
   average during an eval only (forward hooks, nothing written), count the
   failures fixed and the healthy samples broken; narrow down by halves.
5. **explain** -- the samples the confirmed neurons respond to most; one
   category per failure: label suspect / spurious feature / unlike training /
   boundary.
6. **fix** -- tag samples for data work, and / or repair: fine-tune candidates
   from the current weights (control, replay, targeted, reset), each checked
   by a gate (held-out failures fixed, validation accuracy kept, few flips);
   the model changes only when the best one is accepted.

Small or big models and datasets: every pass runs chunk by chunk under the
training lock (training and the Studio interleave), can be cancelled, and is
capped -- at most ``max_failures`` failures, a ``pool`` of candidates
(per class when the dataframe knows targets), ``max_lookalikes``, a guard set
of ``guard_max`` samples -- and the caps shrink when one forward is slow
(``*_seconds`` budgets), so a big model gets a smaller pool, never an hour.

What "failing" means depends on the task (``failure_tasks``): predicted
wrong (classification), an image's IoU (segmentation), the F1 of its boxes
(detection), the share of next tokens predicted right (language models), or
your registered per-sample loss ranked against train / val for anything else.
Continuous scores ask where the line goes (step 1); "fixed" / "broken" then
mean a change of at least ``delta``, so samples on the line don't flip on
noise. Only classification has look-alikes "of the same class" and the
mislabeled check.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from weightslab.components.model_representation import (
    _CallTracker,
    _call_model,
    _Cancelled,
    _channel_axis,
    _first_tensor,
    _image_preview,
    _image_view,
    _inference_mode,
    _inputs_from_batch,
    _loader_entries,
    _locate,
    _model_device,
    _to_device,
    _token_text,
    _training_lock,
    build_batch,
    resolve_inner,
    resolve_samples,
    samples_with_tag,
)
from weightslab.components.failure_tasks import (
    ClassificationTask,
    LossTask,
    NotSupported,
    Task,
    detect_task,
    draw_mark,
    set_failure_boxes_fn,
    set_failure_loss_fn,
    set_failure_task,
)

logger = logging.getLogger(__name__)

NotClassification = NotSupported        # the earlier name


def _tensors_of(module: nn.Module):
    yield from module.named_parameters()
    yield from module.named_buffers()


def _weights_of(module: nn.Module) -> Dict[str, th.Tensor]:
    """A CPU copy of the parameters and buffers. Not ``state_dict()``: WeightsLab
    layers add tracker entries to it that ``load_state_dict`` then refuses."""
    return {k: v.detach().to("cpu", copy=True) for k, v in _tensors_of(module)}


def _load_weights(module: nn.Module, state: Mapping[str, th.Tensor]) -> None:
    """Copy *state* back in place (the optimizer keeps the same tensors)."""
    with th.no_grad():
        for k, t in _tensors_of(module):
            v = state.get(k)
            if v is None:
                continue
            if v.shape == t.shape:
                t.copy_(v)
            else:
                t.data = v.to(device=t.device, dtype=t.dtype).clone()

DEFAULTS: Dict[str, Any] = {
    "max_failures": 256,       # failures analysed (a seeded subset beyond)
    "pool": 4096,              # candidates scanned for look-alikes (train / val)
    "per_class_pool": 1024,    # ...at most this many per class of interest
    "pool_seconds": 90.0,      # the pool shrinks so its pass takes about this
    "k_lookalikes": 8,         # look-alikes per failure
    "max_lookalikes": 256,
    "k_confusers": 4,          # samples of the predicted class, per failure
    "max_controls": 256,       # other healthy samples, for the damage check
    "neighbors_k": 10,         # label-suspect / out-of-distribution neighbours
    "label_suspect_share": 0.7,
    "label_suspect_conf": 0.8,
    "ood_quantile": 0.99,
    "feature_dim": 256,        # features randomly projected to at most this
    "batch": 32,               # forward chunk when the loader has no batch size
    "max_suspect_layers": 3,
    "max_suspects": 32,
    "shuffles": 200,
    "null_quantile": 0.95,
    "confirm_share": 0.2,      # a mask "explains" >= this share of the failures...
    "confirm_damage": 0.1,     # ...breaking <= this share of the look-alikes
    "explain_neurons": 4,
    "top_samples": 6,
    "top_pool": 2048,
    "repair_steps": 0,         # 0 = about 3 passes over the focus samples
    "repair_min_steps": 30,
    "repair_max_steps": 200,
    "repair_seconds": 120.0,   # per candidate
    "repair_lr_scale": 0.5,    # x the user's optimizer learning rate
    "repair_lr": 1e-4,         # when there is no user optimizer to copy
    "anchor": 1e-3,            # L2 pull toward the starting weights
    "focus_share": 0.25,       # share of a repair batch from the failures + neighbours
    "guard_max": 1000,         # samples the gate is measured on
    "guard_seconds": 60.0,
    "gate_acc_drop": 0.002,    # validation accuracy may drop by at most 0.2 pt...
    "gate_flip_rate": 0.005,   # ...flip at most 0.5% right -> wrong (or a quarter of the gains)...
    "gate_class_drop": 0.01,   # ...and no class drops more than 1 pt
    "ref_samples": 256,        # train / val samples a continuous score is compared with
    "ok_threshold": -1.0,      # >= 0: where "failing" starts (else the task's default)
    "det_iou": 0.5,            # detection: a box is found at this IoU...
    "det_conf": 0.25,          # ...by a prediction this confident
    "thumb": 64,
    "max_thumbs": 48,
    "auto_delay": 1.0,         # an automatic answer waits this long (readable)
    "seed": 0,
}

STEPS: List[Tuple[str, str]] = [
    ("check", "Check the failures"),
    ("lookalikes", "Find similar healthy samples"),
    ("compare", "Compare"),
    ("test", "Test the suspects"),
    ("explain", "Explain"),
    ("fix", "Fix"),
]

TAGS = {
    "failures": "wopt_failures",
    "lookalikes": "wopt_lookalikes",
    "label": "wopt_label_review",
    "ood": "wopt_unlike_training",
    "top": "wopt_neuron_top",
    "fixed": "wopt_fixed",
}

_TARGET_KEYS = ("label", "labels", "target", "targets", "y")
_TARGETS_FN: Optional[Callable] = None

Key = Tuple[str, str]          # (sample_id, split)


def set_failure_targets_fn(fn: Optional[Callable]) -> None:
    """How to read the labels from a collated batch (default: the third item
    of WeightsLab's ``(inputs, ids, labels)`` batches, or a dict's
    label / target key). A model can also define ``wl_failure_targets(batch)``."""
    global _TARGETS_FN
    _TARGETS_FN = fn


class _Stop(Exception):
    """The user (or the flow) ended the session normally."""


# =============================================================================
# Small helpers
# =============================================================================
def _targets_from_batch(batch, inner: Optional[nn.Module]):
    fn = _TARGETS_FN or getattr(inner, "wl_failure_targets", None)
    if callable(fn):
        return fn(batch)
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            return batch[2]
        if len(batch) == 2:
            return batch[1]
        return None
    if isinstance(batch, Mapping):
        for key in batch:
            if str(key).lower() in _TARGET_KEYS:
                return batch[key]
    return None


def _labels_array(labels, batch: int) -> Optional[np.ndarray]:
    """(B,) int64 class indices from labels as ints, one-hot rows or (B, 1)."""
    return ClassificationTask.labels(labels, batch)


def classify(output, labels, batch: int) -> Dict[str, np.ndarray]:
    """Per-sample prediction, confidence and margin of a classification output."""
    return ClassificationTask().classify(output, labels, batch)


def _has_own_params(module: nn.Module) -> bool:
    return any(True for _ in module.parameters(recurse=False))


def _png_url(x: th.Tensor, size: int, mark: Optional[dict] = None) -> str:
    view = _image_view(x.detach().float().cpu())
    if view is None:
        return ""
    try:
        from PIL import Image
        prev = _image_preview(view, list(view.shape), size)
        c, h, w = len(prev["channel_indices"]), prev["height"], prev["width"]
        data = np.frombuffer(prev["data"], dtype=np.uint8).reshape(c, h, w)
        if c == 1:
            img = Image.fromarray(np.ascontiguousarray(data[0]))
        else:
            img = Image.fromarray(np.ascontiguousarray(data[:3].transpose(1, 2, 0)))
        if mark:
            img = draw_mark(img.convert("RGB"), mark)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.debug(f"[failures] thumbnail failed: {exc}")
        return ""


def _cohen_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-column Cohen's d of rows *a* against rows *b* (> 0: larger on a)."""
    na, nb = a.shape[0], b.shape[0]
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    va = a.var(axis=0, ddof=1) if na > 1 else np.zeros_like(ma)
    vb = b.var(axis=0, ddof=1) if nb > 1 else np.zeros_like(mb)
    dof = max(na + nb - 2, 1)
    sd = np.sqrt(np.maximum(((na - 1) * va + (nb - 1) * vb) / dof, 0.0))
    scale = float(np.mean(sd)) if sd.size else 0.0
    peak = float(np.max(np.abs(np.concatenate([ma, mb])))) if ma.size else 0.0
    floor = max(1e-3 * scale, 1e-6 * peak, 1e-12)
    return np.clip((ma - mb) / np.maximum(sd, floor), -10.0, 10.0)


def null_threshold(a: np.ndarray, b: np.ndarray, shuffles: int, quantile: float,
                   rng: np.random.Generator) -> float:
    """The *quantile* of max |d| over neurons when A / B labels are shuffled:
    a neuron above it differs more than chance would explain."""
    x = np.concatenate([a, b], axis=0)
    na = a.shape[0]
    n = x.shape[0]
    if na < 1 or n - na < 1:
        return float("inf")
    peaks = []
    for _ in range(max(int(shuffles), 1)):
        perm = rng.permutation(n)
        d = _cohen_d(x[perm[:na]], x[perm[na:]])
        peaks.append(float(np.max(np.abs(d))) if d.size else 0.0)
    return float(np.quantile(peaks, quantile))


def first_jump(scores: Sequence[float], share: float = 0.6, floor: float = 0.3) -> int:
    """Index of the first layer whose score reaches *share* of the largest
    (and at least *floor*): where the difference starts. -1 if none."""
    if not len(scores):
        return -1
    top = float(max(scores))
    if top < floor:
        return -1
    for i, s in enumerate(scores):
        if s >= max(share * top, floor):
            return i
    return -1


def _dataset_ids(dataset) -> List[str]:
    """Every sample id of a tracked dataset (a list or a numpy array), as str."""
    ids = getattr(dataset, "unique_ids", None)
    if ids is None:
        try:
            return [str(i) for i in range(len(dataset))]
        except Exception:
            return []
    try:
        return [str(i) for i in list(ids)]
    except Exception:
        return []


def _n(count: int, one: str, many: str) -> str:
    """"**1** is" / "**3** are" -- the count in bold, the verb agreeing."""
    return f"**{count}** {one if count == 1 else many}"


def _split_kind(entry) -> str:
    text = f"{entry[0]} {entry[3]}".lower()
    if "train" in text:
        return "train"
    if "val" in text:
        return "val"
    if "test" in text:
        return "test"
    return "other"


def _batch_size_of_loader(loader, default: int) -> int:
    try:
        dl = loader.get_dataloader() if callable(getattr(loader, "get_dataloader", None)) else None
        size = int(getattr(dl, "batch_size", 0) or 0)
        if size > 0:
            return min(size, 256)
    except Exception:
        pass
    return default


def _cat_batches(batches: List[Any], sizes: Optional[Sequence[int]] = None, _depth: int = 0):
    """Concatenate collated batches of one structure along the batch dim.

    *sizes* (samples per batch) tells a per-sample list -- one box array per
    image, one id per sample -- from a structure like ``(inputs, ids,
    targets)``; the top level is always a structure."""
    first = batches[0]
    if len(batches) == 1:
        return first
    if isinstance(first, th.Tensor):
        if first.ndim == 0:
            return first
        try:
            return th.cat([b for b in batches], dim=0)
        except RuntimeError:              # variable trailing sizes: one tensor per sample
            return [row for b in batches for row in b]
    if isinstance(first, (list, tuple)):
        scalars = bool(first) and not isinstance(first[0], (th.Tensor, list, tuple, Mapping))
        lists = isinstance(first, list) and sizes is not None and all(
            isinstance(b, list) and len(b) == s for b, s in zip(batches, sizes))
        if _depth > 0 and (scalars or lists):     # ids (a tuple of str), one box array per image
            return [v for b in batches for v in b]
        parts = [_cat_batches([b[i] for b in batches], sizes, _depth + 1) for i in range(len(first))]
        return type(first)(parts) if isinstance(first, tuple) else parts
    if isinstance(first, Mapping):
        if "batch_idx" in first and sizes is not None:   # an Ultralytics batch: renumber the images
            offset, shifted = 0, []
            for b, s in zip(batches, sizes):
                shifted.append({**b, "batch_idx": b["batch_idx"] + offset})
                offset += s
            batches = shifted
        return {k: _cat_batches([b[k] for b in batches], sizes, _depth + 1) for k in first}
    return first


# =============================================================================
# The live model, sample by sample
# =============================================================================
class _Runner:
    """Forwards samples through the live model chunk by chunk, under the
    training lock, in eval mode, without counting a step."""

    def __init__(self, model, loaders, cancel: threading.Event, batch: int):
        self.model = model
        self.inner = resolve_inner(model)
        self.entries = _loader_entries(loaders)
        self.by_split = {e[3]: e for e in self.entries}
        self.cancel = cancel
        self.batch = batch
        self.device = _model_device(self.inner)

    def entry(self, split: str):
        return self.by_split.get(split)

    def chunk_size(self, entry) -> int:
        return _batch_size_of_loader(entry[1], self.batch)

    def items(self, keys: Sequence[Key]):
        return [(sid, split, self.by_split[split]) for sid, split in keys if split in self.by_split]

    def batch_of(self, keys: Sequence[Key]):
        """One collated batch of *keys* (several splits are concatenated)."""
        groups: Dict[str, List[str]] = {}
        for sid, split in keys:
            groups.setdefault(split, []).append(sid)
        parts = [build_batch(self.by_split[s], ids) for s, ids in groups.items()]
        order = [(sid, s) for s, ids in groups.items() for sid in ids]
        return _cat_batches(parts, [len(ids) for ids in groups.values()]), order

    def run(self, keys: Sequence[Key], on_chunk: Callable, hooks: Sequence[Callable] = (),
            progress: Optional[Callable[[int], None]] = None, chunk: Optional[int] = None) -> None:
        """on_chunk(keys, output, labels, inputs) for every chunk of *keys*."""
        groups: Dict[str, List[str]] = {}
        for sid, split in keys:
            if split in self.by_split:
                groups.setdefault(split, []).append(sid)
        done = 0
        for split, ids in groups.items():
            entry = self.by_split[split]
            size = chunk or self.chunk_size(entry)
            for start in range(0, len(ids), size):
                if self.cancel.is_set():
                    raise _Cancelled()
                part = ids[start:start + size]
                with _training_lock(cancel=self.cancel):
                    batch = build_batch(entry, part)
                    inputs = _to_device(_inputs_from_batch(batch, self.inner), self.device)
                    labels = _targets_from_batch(batch, self.inner)
                    with _inference_mode(self.model, self.inner), ExitStack() as stack:
                        for hook in hooks:
                            stack.enter_context(hook())
                        output = _call_model(self.model, inputs)
                    on_chunk([(sid, split) for sid in part], output, labels, inputs)
                done += len(part)
                if progress is not None:
                    progress(done)


@contextmanager
def _module_hook(module: nn.Module, fn, pre: bool = False):
    handle = module.register_forward_pre_hook(fn) if pre else module.register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def _pool_positions(x: th.Tensor, axis: Optional[int], cells: int = 16,
                    max_dims: int = 65536) -> th.Tensor:
    """(B, C * P): the channels at P <= *cells* positions (a 4 x 4 grid of an
    image, 16 spans of a sequence), so a big feature map or a long sequence
    stays a few thousand numbers per sample."""
    b = int(x.shape[0])
    if x.ndim <= 2 or axis is None or axis == 0:
        return x.reshape(b, -1)
    x = x.movedim(axis, 1)
    c = int(x.shape[1])
    cells = max(1, min(cells, max_dims // max(c, 1)))
    spatial = tuple(int(s) for s in x.shape[2:])
    if int(np.prod(spatial)) <= cells:
        return x.reshape(b, -1)
    if len(spatial) == 2:
        side = max(1, int(math.sqrt(cells)))
        x = F.adaptive_avg_pool2d(x, (min(side, spatial[0]), min(side, spatial[1])))
    else:
        x = F.adaptive_avg_pool1d(x.reshape(b, c, -1), cells)
    return x.reshape(b, -1)


def _pooled_values(tensor: th.Tensor, axis: Optional[int], batch: int):
    """(mean, max |.|) per sample and neuron, (B, C) each, or None."""
    if not isinstance(tensor, th.Tensor) or tensor.ndim == 0 or tensor.shape[0] != batch:
        return None
    x = th.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if axis is None or x.ndim < 2 or axis == 0:
        return None
    flat = x.movedim(axis, 1).reshape(batch, x.shape[axis], -1)
    if flat.shape[2] == 0:
        return None
    return flat.mean(dim=2).cpu().numpy(), flat.abs().amax(dim=2).cpu().numpy()


# =============================================================================
# The session: a conversation that runs the steps
# =============================================================================
class FailureSession:
    def __init__(self, model, loaders, keys: List[Key], label: str, automatic: bool,
                 options: Dict[str, Any]):
        self.id = uuid.uuid4().hex[:12]
        self.opts = dict(DEFAULTS)
        self.opts.update({k: v for k, v in (options or {}).items() if k in DEFAULTS})
        self.rng = np.random.default_rng(int(self.opts["seed"]))
        self.model = model
        self.loaders = loaders
        self.requested = keys
        self.label = label
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.answered = threading.Event()
        self.state = "running"
        self.message = ""
        self.step = ""
        self.step_label = ""
        self.done = 0
        self.total = 0
        self.automatic = bool(automatic)
        self._auto_since = time.monotonic()
        self.messages: List[dict] = []
        self.seq = 0
        self.question: Optional[dict] = None
        self._qn = 0
        self._answer: Optional[Tuple[str, List[str]]] = None
        self.title = ""
        self.thread: Optional[threading.Thread] = None
        # analysis state
        self.runner: Optional[_Runner] = None
        self.task: Optional[Task] = None
        self.info: Dict[Key, dict] = {}         # per sample, before any change: score, ok, caption...
        self.reference: List[Key] = []         # train / val samples a continuous score is read against
        self.failures: List[Key] = []
        self.analysed: List[Key] = []          # failures the comparison uses
        self.label_suspects: List[Key] = []
        self.near_pred: List[Key] = []         # among the predicted class, confident
        self.ood: List[Key] = []
        self.lookalikes: List[Key] = []
        self.confusers: List[Key] = []
        self.controls: List[Key] = []
        self.pool: List[Key] = []
        self.pairs: Dict[Key, List[Key]] = {}
        self.head_name = ""
        self.sec_per_sample = 0.0
        self.layers: List[dict] = []
        self.candidates: List[dict] = []       # chosen layers with their suspects
        self.confirmed: Optional[dict] = None  # {"node", "neurons", "fixed": [...], ...}
        self.categories: Dict[str, List[Key]] = {}
        self.tagged: Dict[str, int] = {}
        self.repair_results: List[dict] = []
        self._best_state: Optional[Dict[str, th.Tensor]] = None
        self._snapshot: Optional[Dict[str, th.Tensor]] = None
        self._held = False
        self._mark_cache: Dict[Key, Optional[dict]] = {}   # thumbnails' overlays, current weights

    # ---- conversation --------------------------------------------------------
    def say(self, text: str, card: Optional[dict] = None, role: str = "agent") -> None:
        with self.lock:
            self.seq += 1
            self.messages.append({"seq": self.seq, "role": role, "step": self.step,
                                  "text": text, "card_json": json.dumps(card) if card else ""})

    def set_step(self, key: str) -> None:
        with self.lock:
            self.step = key
            self.step_label = dict(STEPS).get(key, key)
            self.done = self.total = 0

    def progress(self, label: str, done: int, total: int) -> None:
        with self.lock:
            self.step_label = label
            self.done = int(done)
            self.total = int(total)

    def set_automatic(self, value: bool) -> None:
        with self.lock:
            if value and not self.automatic:
                self._auto_since = time.monotonic()
            self.automatic = bool(value)
        self.answered.set()

    def ask(self, key: str, text: str, choices: List[dict], multiple: bool = False,
            defaults: Optional[List[str]] = None, confirm: str = "Continue"):
        """Post a question and wait for its answer (or the default when
        automatic). Returns the choice id, or the list of ids (multiple)."""
        if not choices:
            return [] if multiple else ""
        primary = next((c["id"] for c in choices if c.get("primary")), choices[0]["id"])
        defaults = list(defaults) if defaults is not None else ([primary] if multiple else [])
        with self.lock:
            self._qn += 1
            qid = f"{key}-{self._qn}"
            self.question = {"id": qid, "text": text, "choices": choices, "multiple": multiple,
                             "defaults": defaults, "confirm_label": confirm}
            self.state = "waiting"
            self._answer = None
            asked_at = time.monotonic()
        self.answered.clear()
        delay = float(self.opts["auto_delay"])
        while True:
            if self.cancel.is_set():
                raise _Cancelled()
            with self.lock:
                answer = self._answer
                auto = self.automatic and \
                    time.monotonic() - max(asked_at, self._auto_since) >= delay
            if answer is not None:
                role = "user"
                ids = answer[1] if multiple else [answer[0]]
                break
            if auto:
                role = "auto"
                ids = defaults if multiple else [primary]
                break
            self.answered.wait(timeout=0.1)
            self.answered.clear()
        labels = {c["id"]: c["label"] for c in choices}
        shown = ", ".join(labels.get(i, i) for i in ids) or "(none)"
        self.say(shown if role == "user" else f"Automatic: {shown}", role=role)
        with self.lock:
            self.question = None
            self.state = "running"
        return list(ids) if multiple else ids[0]

    def answer(self, question_id: str, choice: str, choices: Sequence[str]) -> Tuple[bool, str]:
        with self.lock:
            if self.question is None:
                return False, "no question is pending"
            if question_id and question_id != self.question["id"]:
                return False, "that question was already answered"
            valid = {c["id"] for c in self.question["choices"]}
            if self.question["multiple"]:
                picked = [c for c in choices if c in valid]
                self._answer = ("", picked)
            else:
                if choice not in valid:
                    return False, f"unknown choice {choice!r}"
                self._answer = (choice, [choice])
        self.answered.set()
        return True, ""

    def status(self, since: int = 0) -> dict:
        with self.lock:
            msgs = [dict(m) for m in self.messages if m["seq"] > since]
            q = dict(self.question) if self.question else None
            step_keys = [k for k, _ in STEPS]
            return {
                "session_id": self.id, "state": self.state, "message": self.message,
                "step": self.step, "step_label": self.step_label,
                "done": self.done, "total": self.total, "automatic": self.automatic,
                "messages": msgs, "question": q, "last_seq": self.seq, "title": self.title,
                "steps": step_keys, "step_titles": [t for _, t in STEPS],
                "step_index": step_keys.index(self.step) if self.step in step_keys else -1,
            }

    # ---- samples on cards ------------------------------------------------------
    def _caption(self, key: Key) -> str:
        info = self.info.get(key)
        return str(info.get("caption", "")) if info else ""

    def _marks(self, keys: Sequence[Key]) -> Dict[Key, dict]:
        """What to draw on each thumbnail (wrong pixels, true / predicted
        boxes): one more forward of these few samples, under the lock."""
        task = self.task
        if task is None or not task.can_mark or not keys:
            return {}
        cache = self._mark_cache
        todo = [k for k in keys if k not in cache]

        def on_chunk(part, output, labels, inputs):
            for key, mark in zip(part, task.marks(output, labels, len(part), inputs)):
                cache[key] = mark or None

        if todo:
            try:
                self.runner.run(todo, on_chunk)
            except _Cancelled:
                raise
            except Exception as exc:
                logger.debug(f"[failures] could not mark the thumbnails: {exc}")
        return {k: cache[k] for k in keys if cache.get(k)}

    def sample_items(self, keys: Sequence[Key], captions: Optional[Dict[Key, str]] = None) -> List[dict]:
        """Card items with a thumbnail (images, with the errors drawn on them)
        or text (token ids)."""
        keys = list(keys)[: int(self.opts["max_thumbs"])]
        items = [{"id": sid, "origin": split, "thumb": "", "text": "",
                  "caption": (captions or {}).get((sid, split), self._caption((sid, split)))}
                 for sid, split in keys]
        if not keys or self.runner is None:
            return items
        size = int(self.opts["thumb"])
        index = {k: i for i, k in enumerate(keys)}
        try:
            marks = self._marks(keys)
            with _training_lock(cancel=self.cancel):
                batch, order = self.runner.batch_of(keys)
            inputs = _inputs_from_batch(batch, self.runner.inner)
            x = _first_tensor(inputs)
            if x is None:
                return items
            for row, key in enumerate(order):
                item = items[index[key]]
                sample = x[row]
                if sample.dtype in (th.int64, th.int32) and sample.ndim == 1:
                    text = "".join(_token_text(self.runner.inner, sample.tolist()))
                    item["text"] = (text or " ".join(str(int(v)) for v in sample[:24].tolist()))[:120]
                else:
                    item["thumb"] = _png_url(sample, size, marks.get(key))
        except _Cancelled:
            raise
        except Exception as exc:
            logger.debug(f"[failures] could not build thumbnails: {exc}")
        return items

    # ---- the flow ------------------------------------------------------------
    def run(self) -> None:
        try:
            self._intro()
            self._check()
            self._find_lookalikes()
            self._compare()
            self._test()
            self._explain()
            self._fix()
            with self.lock:
                self.state = "done"
                self.step_label = "Done"
        except _Stop:
            with self.lock:
                self.state = "done"
                self.step_label = "Stopped"
        except _Cancelled:
            self._rollback_repair()
            with self.lock:
                self.state = "cancelled"
                self.step_label = "Stopped"
                self.question = None
            self.say("Stopped.")
        except NotSupported as exc:
            self._rollback_repair()
            with self.lock:
                self.state = "done"
                self.message = str(exc)
                self.question = None
            self.say(f"I can't analyse this model: {exc}. I read classification, segmentation, "
                     "detection and language-model outputs, or any per-sample loss registered with "
                     "`wl.watch_or_edit(criterion, flag=\"loss\")`; `set_failure_task` / "
                     "`set_failure_boxes_fn` tell me more.")
        except Exception as exc:
            logger.warning(f"[failures] analysis failed: {exc}", exc_info=True)
            self._rollback_repair()
            with self.lock:
                self.state = "error"
                self.message = f"{type(exc).__name__}: {exc}"
                self.question = None
            self.say(f"Something went wrong: `{type(exc).__name__}: {exc}`. The model is unchanged.")
        finally:
            self._release_hold()

    def _ask_continue(self, key: str, text: str, go: str, hint: str = "") -> None:
        choice = self.ask(key, text, [{"id": "go", "label": go, "primary": True, "hint": hint},
                                      {"id": "stop", "label": "Stop here"}])
        if choice == "stop":
            self.say("OK, stopping here. Everything found so far stays tagged.")
            raise _Stop()

    def _intro(self) -> None:
        n = len(self.requested)
        self.title = f"Why do these {n} samples fail?" if n != 1 else "Why does this sample fail?"
        self.say(
            f"I'll look into why {self.label} fail{'s' if n == 1 else ''}, step by step: "
            "**check** them, find **similar healthy samples**, **compare** the two, "
            "**test** the suspect neurons by silencing them, **explain** what they respond to, "
            "then propose a **fix**. I'll ask before each step; tick **Automatic** (top right) "
            "to let me run them all.")
        self._ask_continue("start", "Start the analysis?", "Start")

    # ---- 1. check ----------------------------------------------------------------
    def _check(self) -> None:
        self.set_step("check")
        keys = list(self.requested)
        cap = int(self.opts["max_failures"])
        if len(keys) > cap:
            keep = sorted(self.rng.choice(len(keys), cap, replace=False))
            keys = [keys[i] for i in keep]
            self.say(f"{len(self.requested)} samples is a lot: I'll work on an even random "
                     f"subset of {cap}.")
        inner = self.runner.inner
        head = {"name": "", "module": None}
        names = {id(m): n for n, m in inner.named_modules()}

        def recorder():
            handles = []

            def hook(module, args, output):
                head["module"] = module
                head["name"] = names.get(id(module), "")

            for module in inner.modules():
                if _has_own_params(module):
                    handles.append(module.register_forward_hook(hook))

            @contextmanager
            def ctx():
                try:
                    yield
                finally:
                    for h in handles:
                        h.remove()
            return ctx()

        found: Dict[Key, dict] = {}
        datasets = [e[2] for e in self.runner.entries]

        def on_chunk(part, output, labels, inputs):
            if self.task is None:
                self.task = detect_task(output, labels, len(part), inputs, inner, datasets, self.opts)
            for key, ev in zip(part, self.task.evaluate(output, labels, len(part), inputs)):
                found[key] = ev

        t0 = time.monotonic()
        self.progress(f"Running the model on {len(keys)} samples", 0, len(keys))
        self.runner.run(keys, on_chunk, hooks=[recorder],
                        progress=lambda d: self.progress(f"Running the model on {len(keys)} samples",
                                                         d, len(keys)))
        self.sec_per_sample = (time.monotonic() - t0) / max(len(found), 1)
        self.head_name = head["name"]
        self.head_module = head["module"]
        self.info.update(found)
        task = self.task
        if task is None:
            raise NotSupported("none of these samples could be loaded")
        if float(self.opts["ok_threshold"]) >= 0 and task.continuous:
            task.threshold = float(self.opts["ok_threshold"])
        missing = len(keys) - len(found)
        checked = [k for k in keys if k in found]
        if task.continuous:
            self._describe_task(checked)
            wrong = self._choose_failures(checked)
        else:
            if task.name != "classification":
                self._describe_task(checked)
            wrong = [k for k in checked if not self.info[k]["ok"]]
        right = [k for k in checked if k not in set(wrong)]
        self.failures = wrong
        if task.continuous:
            lines = [f"**{len(wrong)}** of the {len(checked)} samples fail now."]
        else:
            lines = [f"**{len(wrong)}** of the {len(checked)} samples are wrong now (true → predicted)."]
        if right:
            lines.append(f"{len(right)} {'are predicted right' if not task.continuous else 'do well enough'} "
                         "at the current weights: I'll leave them out.")
        if missing:
            lines.append(f"{missing} could not be loaded.")
        self.say(" ".join(lines), {"kind": "samples", "title": "Failures",
                                   "items": self.sample_items(wrong[:24]),
                                   "more": max(len(wrong) - 24, 0)})
        if not wrong:
            self.say("Nothing to analyse: none of these samples fails now.")
            raise _Stop()
        self._tag(TAGS["failures"], wrong)
        hint = ("Same class, predicted right, closest in the model's own features" if task.grouped else
                "Samples the model does well on, closest in its own features")
        self._ask_continue("check", "Find similar healthy samples?", "Find similar healthy samples", hint)

    def _eval_keys(self, keys: Sequence[Key], label: str, hooks: Sequence[Callable] = ()) -> Dict[Key, dict]:
        """The task's record of every key (score, ok, caption...), with *hooks* on."""
        out: Dict[Key, dict] = {}
        task = self.task

        def on_chunk(part, output, labels, inputs):
            for key, ev in zip(part, task.evaluate(output, labels, len(part), inputs)):
                out[key] = ev

        total = len(keys)
        self.runner.run(keys, on_chunk, hooks=hooks, progress=lambda d: self.progress(label, d, total))
        return out

    def _describe_task(self, checked: Sequence[Key]) -> None:
        task = self.task
        if task.name == "segmentation":
            text = ("This is a **segmentation** model: I judge each image by its **IoU** (the mean over "
                    "the classes in its mask or its prediction). Red pixels on the thumbnails are wrong.")
        elif task.name == "detection":
            text = (f"This is a **detection** model: I judge each image by the **F1** of its boxes (a true "
                    f"box is found by a prediction of its class with IoU ≥ {task.iou:.2f} and confidence "
                    f"≥ {task.conf:.2f}). Green: true boxes; orange: predicted ones.")
            if not task.can_train:
                text += (" No loss is registered, so I can analyse but not repair: register it with "
                         "`wl.watch_or_edit(criterion, flag=\"loss\")`.")
        elif task.name == "language_model":
            text = ("This is a **language model**: I judge each sample by the share of its next tokens "
                    "it predicts right (teacher forcing), the miss I show being its most confident one.")
            if getattr(task, "shift", False):
                text += " Its labels are its inputs, so I compare each position with the next token."
        elif isinstance(task, LossTask):
            why = f" ({task.repair_note})" if task.repair_note else ""
            text = (f"I can't read this {task.title}'s output directly{why}, so I judge each sample by "
                    f"{task.source}: the share of train / val samples with a higher loss.")
            if task.boxes:
                text += " Green: the true boxes."
        else:
            text = f"I judge each sample by its {task.metric}."
        self.say(text)

    def _reference(self) -> List[Key]:
        """A seeded sample of train / val, scored: what "doing well" looks like."""
        runner = self.runner
        entries = [e for e in runner.entries if _split_kind(e) in ("train", "val")] or list(runner.entries)
        failing = set(self.requested)
        keys = [(sid, e[3]) for e in entries for sid in _dataset_ids(e[2])]
        keys = [k for k in keys if k not in failing]
        budget = int(self.opts["ref_samples"])
        if self.sec_per_sample > 0:
            budget = min(budget, max(32, int(float(self.opts["pool_seconds"]) / 4 / self.sec_per_sample)))
        if len(keys) > budget:
            keys = [keys[i] for i in sorted(self.rng.choice(len(keys), budget, replace=False))]
        if not keys:
            return []
        evs = self._eval_keys(keys, f"Scoring {len(keys)} samples of train / val for comparison")
        self.info.update(evs)
        self.reference = [k for k in keys if k in evs]
        return self.reference

    def _choose_failures(self, checked: Sequence[Key]) -> List[Key]:
        """Continuous scores: where does failing start? Asked, with the
        counts each choice gives; the default is the task's threshold."""
        task = self.task
        ref = self._reference()
        if isinstance(task, LossTask):
            task.set_reference([self.info[k]["loss"] for k in ref])
            for ev in self.info.values():
                task.finalize(ev)
        sel = np.array([self.info[k]["score"] for k in checked], dtype=np.float64)
        ref_scores = np.array([self.info[k]["score"] for k in ref], dtype=np.float64)
        options: List[Tuple[str, float, str]] = [("default", float(task.threshold), task.failing_text())]
        if isinstance(task, LossTask):
            options.append(("strict", 0.10, "a loss higher than 90% of train / val"))
        elif ref_scores.size >= 8:
            q = float(np.quantile(ref_scores, 0.25))
            if abs(q - task.threshold) > 0.02:
                options.append(("ref", q, f"{task.metric} below {q:.2f}, the worst quarter of train / val"))
        choices = []
        for cid, thr, text in options:
            n_fail = int(np.sum(sel < thr))
            choices.append({"id": cid, "label": f"{text[0].upper()}{text[1:]}: {n_fail} of {len(checked)}",
                            "hint": "", "_n": n_fail, "_t": thr})
        choices.append({"id": "all", "label": f"All {len(checked)} selected samples",
                        "hint": "Analyse them all, whatever their score", "_n": len(checked), "_t": None})
        primary = next((c for c in choices if c["_n"] > 0), choices[-1])
        primary["primary"] = True
        values = [self.info[k]["value"] for k in checked]
        rows = [{"label": f"Selected ({len(checked)}): mean {task.metric}",
                 "value": task.fmt(float(np.mean(values))), "good": False}]
        if ref:
            rows.append({"label": f"Train / val ({len(ref)}): mean {task.metric}",
                         "value": task.fmt(float(np.mean([self.info[k]["value"] for k in ref]))),
                         "good": True})
            if not isinstance(task, LossTask):
                rows.append({"label": f"Train / val with {task.failing_text()}",
                             "value": f"{int(np.sum(ref_scores < task.threshold))}/{len(ref)}",
                             "good": True})
        self.say("How these samples compare with train / val:", {"kind": "metrics", "rows": rows})
        picked = self.ask("threshold", "Which samples count as failing?",
                          [{k: v for k, v in c.items() if not k.startswith("_")} for c in choices])
        choice = next(c for c in choices if c["id"] == picked)
        if choice["_t"] is None:
            return list(checked)
        task.threshold = float(choice["_t"])
        for ev in self.info.values():
            task.finalize(ev)
        return [k for k in checked if not self.info[k]["ok"]]

    # ---- 2. look-alikes -------------------------------------------------------
    def _pool_keys(self) -> List[Key]:
        """Candidates in train / val (or every other split), capped -- per
        class of interest first when the dataframe knows the targets."""
        runner = self.runner
        entries = [e for e in runner.entries if _split_kind(e) in ("train", "val")] or \
            [e for e in runner.entries]
        failing = set(self.requested)
        budget = int(self.opts["pool"])
        if self.sec_per_sample > 0:
            budget = min(budget, max(256, int(float(self.opts["pool_seconds"]) / self.sec_per_sample)))
        all_keys: List[Key] = []
        for e in entries:
            all_keys.extend((sid, e[3]) for sid in _dataset_ids(e[2]))
        all_keys = [k for k in all_keys if k not in failing]
        targets, discarded = self._dataframe_targets()
        if discarded:
            all_keys = [k for k in all_keys if k not in discarded and k[0] not in discarded]
        if len(all_keys) <= budget:
            return all_keys
        picked: List[Key] = []
        if targets and self.task.grouped:
            classes = {self.info[k]["group"] for k in self.failures} | \
                      {self.info[k]["pred_group"] for k in self.failures}
            by_class: Dict[int, List[Key]] = {}
            for k in all_keys:
                t = targets.get(k, targets.get(k[0]))
                if t in classes:
                    by_class.setdefault(t, []).append(k)
            per = max(int(self.opts["per_class_pool"]), 1)
            per = min(per, max(budget // max(len(classes), 1), 16))
            for c, ks in by_class.items():
                if len(ks) > per:
                    ks = [ks[i] for i in sorted(self.rng.choice(len(ks), per, replace=False))]
                picked.extend(ks)
        chosen = set(picked)
        rest = [k for k in all_keys if k not in chosen]
        extra = max(budget - len(picked), min(len(rest), budget // 4))
        if rest and extra > 0:
            idx = self.rng.choice(len(rest), min(extra, len(rest)), replace=False)
            picked.extend(rest[i] for i in sorted(idx))
        return picked

    @staticmethod
    def _dataframe_targets() -> Tuple[Dict[Any, int], set]:
        try:
            from weightslab.backend.ledgers import get_dataframe
            manager = get_dataframe()
            df = manager.get_df_view() if manager is not None else None
        except Exception:
            df = None
        if df is None or len(df) == 0:
            return {}, set()
        targets: Dict[Any, int] = {}
        discarded: set = set()
        index = df.index
        ids = index.get_level_values(0) if getattr(index, "nlevels", 1) > 1 else index
        origins = df["origin"].astype(str).tolist() if "origin" in df.columns else [""] * len(df)
        if "target" in df.columns:
            for sid, org, value in zip(ids, origins, df["target"].tolist()):
                try:
                    arr = np.asarray(value).reshape(-1)
                    if arr.size == 1 and np.isfinite(float(arr[0])):
                        targets[(str(sid), org)] = int(round(float(arr[0])))
                        targets.setdefault(str(sid), int(round(float(arr[0]))))
                except Exception:
                    continue
        if "discarded" in df.columns:
            for sid, org, flag in zip(ids, origins, df["discarded"].tolist()):
                if bool(flag) and flag == flag:
                    discarded.add((str(sid), org))
        return targets, discarded

    def _features_hook(self, store: Dict[str, Any]):
        """Pre-hook on the model's last layer: its input, pooled over
        positions (pixels, tokens) down to a few, flattened and randomly
        projected to at most feature_dim dims."""
        module = self.head_module
        dim = int(self.opts["feature_dim"])
        hints: Dict[Tuple[int, ...], int] = {}

        def hook(mod, args):
            x = _first_tensor(args)
            if x is None or x.ndim < 1:
                return None
            x = _pool_positions(x.detach().float(), _channel_axis(mod, x, hints))
            if x.shape[1] > dim:
                proj = store.get("proj")
                if proj is None or proj.shape[0] != x.shape[1]:
                    g = th.Generator().manual_seed(int(self.opts["seed"]))
                    proj = th.randn(x.shape[1], dim, generator=g) / math.sqrt(dim)
                    store["proj"] = proj
                x = x.cpu() @ proj
            store["last"] = x.cpu()
            return None

        @contextmanager
        def ctx():
            if module is None:
                yield
                return
            with _module_hook(module, hook, pre=True):
                yield
        return ctx

    def _embed(self, keys: Sequence[Key], label: str) -> Tuple[List[Key], np.ndarray]:
        """Features (L2-normalised) and classification info of *keys*."""
        store: Dict[str, Any] = {}
        order: List[Key] = []
        feats: List[np.ndarray] = []

        def on_chunk(part, output, labels, inputs):
            evs = self.task.evaluate(output, labels, len(part), inputs)
            f = store.get("last")
            if f is None or f.shape[0] != len(part):
                f = _pool_positions(_first_tensor(output).detach().float(), 1).cpu()
            store["last"] = None
            for key, ev in zip(part, evs):
                self.info[key] = ev
            order.extend(part)
            feats.append(F.normalize(f, dim=1).numpy())

        total = len(keys)
        self.runner.run(keys, on_chunk, hooks=[self._features_hook(store)],
                        progress=lambda d: self.progress(label, d, total))
        return order, (np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), np.float32))

    def _find_lookalikes(self) -> None:
        self.set_step("lookalikes")
        pool = self._pool_keys()
        fail_order, fail_feats = self._embed(self.failures, f"Reading the {len(self.failures)} failures")
        pool_order, pool_feats = self._embed(pool, f"Scanning {len(pool)} samples of train / val")
        self.pool = pool_order
        if not pool_order:
            self.say("I found no other samples to compare with (no train / val split).")
            raise _Stop()
        opts = self.opts
        k_ok, k_conf, k_nb = int(opts["k_lookalikes"]), int(opts["k_confusers"]), int(opts["neighbors_k"])
        task = self.task
        grouped = task.grouped
        labels = np.array([self.info[k].get("group", -1) if grouped else 0 for k in pool_order])
        margins = np.array([self.info[k]["margin"] for k in pool_order])
        correct = np.array([bool(self.info[k]["ok"]) for k in pool_order])
        comfortable = float(task.comfortable)
        sims = fail_feats @ pool_feats.T                       # (F, P)
        # Out of distribution: the k-th neighbour distance against the pool's own.
        k_ref = min(k_nb, max(len(pool_order) - 1, 1))
        ref_n = min(512, len(pool_order))
        ref_idx = self.rng.choice(len(pool_order), ref_n, replace=False)
        ref_sims = pool_feats[ref_idx] @ pool_feats.T
        ref_sims[np.arange(ref_n), ref_idx] = -np.inf          # not itself
        ref_kth = 1.0 - np.sort(ref_sims, axis=1)[:, ::-1][:, k_ref - 1]
        ood_cut = float(np.quantile(ref_kth, float(opts["ood_quantile"]))) if ref_n > 1 else np.inf
        look: List[Key] = []
        conf_keys: List[Key] = []
        per_failure: Dict[Key, List[Key]] = {}
        self.near_pred, self.ood = [], []
        for i, key in enumerate(fail_order):
            info = self.info[key]
            order = np.argsort(-sims[i])
            group = info.get("group", -1) if grouped else 0
            same_ok = [j for j in order if labels[j] == group and correct[j]
                       and margins[j] >= comfortable][:k_ok]
            if len(same_ok) < k_ok:
                same_ok += [j for j in order if labels[j] == group and correct[j]
                            and j not in same_ok][: k_ok - len(same_ok)]
            per_failure[key] = [pool_order[j] for j in same_ok]
            if grouped:
                confusers = [j for j in order if labels[j] == info["pred_group"] and correct[j]][:k_conf]
                conf_keys.extend(pool_order[j] for j in confusers)
                neighbours = order[:k_nb]
                share_pred = float(np.mean(labels[neighbours] == info["pred_group"])) if len(neighbours) else 0.0
                if share_pred >= float(opts["label_suspect_share"]) and                         info["conf"] >= float(opts["label_suspect_conf"]):
                    self.near_pred.append(key)
            kth = 1.0 - float(sims[i, order[min(k_ref, len(order)) - 1]])
            if kth > ood_cut:
                self.ood.append(key)
        # Round-robin so every failure gets its share under the cap.
        cap = int(opts["max_lookalikes"])
        seen = set()
        for r in range(k_ok):
            for key in fail_order:
                lst = per_failure.get(key, [])
                if r < len(lst) and lst[r] not in seen and len(look) < cap:
                    seen.add(lst[r])
                    look.append(lst[r])
        self.lookalikes = look
        self.pairs = per_failure
        self.confusers = list(dict.fromkeys(k for k in conf_keys if k not in seen))
        used = seen | set(self.confusers)
        healthy = [k for k, ok in zip(pool_order, correct) if ok and k not in used]
        n_ctrl = min(int(opts["max_controls"]), len(healthy))
        self.controls = [healthy[i] for i in sorted(self.rng.choice(len(healthy), n_ctrl, replace=False))] \
            if n_ctrl else []
        if not look:
            self.say("I found no sample of the failures' classes that the model gets right: "
                     "nothing to compare with." if grouped else
                     "I found no sample the model does well on: nothing to compare with.")
            raise _Stop()
        head = f"`{self.head_name}`" if self.head_name else "its last layer"
        what = ("samples of the **same class** that the model gets **right**" if grouped else
                f"samples the model does **well** on ({task.metric} ≥ {task.threshold:.2f})")
        unit = "not in pixels" if task.name in ("segmentation", "detection") or task.can_mark else             "not in the raw input"
        self.say(
            f"For each failure, the {k_ok} most similar {what}: similar in the model's own features "
            f"(the input of {head}), {unit}. Searched among {len(pool_order)} samples of train / val; "
            f"**{len(look)}** look-alikes (tagged `{TAGS['lookalikes']}`).",
            {"kind": "pairs", "rows": [
                {"failure": self.sample_items([key])[0],
                 "lookalikes": self.sample_items(per_failure.get(key, [])[:4])}
                for key in fail_order[:6]], "more": max(len(fail_order) - 6, 0)})
        self._tag(TAGS["lookalikes"], look)
        notes = []
        if self.near_pred:
            notes.append(f"{_n(len(self.near_pred), 'sits', 'sit')} among samples of the *predicted* class in the "
                         "model's features, with a confident prediction: either they are **mislabeled**, "
                         "or the model **reacts to something misleading** in them. Silencing the suspects "
                         "(step 4) tells which.")
        if self.ood:
            self._tag(TAGS["ood"], self.ood)
            notes.append(f"{_n(len(self.ood), 'is', 'are')} unlike anything in train / val: farther from their "
                         f"neighbours than {float(opts['ood_quantile']) * 100:.0f}% of training samples "
                         f"(tagged `{TAGS['ood']}`).")
        if notes:
            self.say(" ".join(notes))
        self._ask_continue("lookalikes", "Compare the failures with their look-alikes, layer by layer?",
                           "Compare failures with look-alikes",
                           "Per layer and per neuron: where do they part ways?")
        self.analysed = list(fail_order)

    # ---- 3. compare -------------------------------------------------------------
    def _layer_stats(self, keys: Sequence[Key], side: int, stats: Dict[str, Any], label: str) -> None:
        """Running per-neuron sums of every leaf layer with weights."""
        from weightslab.components.model_comparison import SideStats
        runner = self.runner
        head = self.head_module
        hints: Dict[Tuple[int, ...], int] = {}
        order = stats.setdefault("_order", [])
        modules = stats.setdefault("_modules", {})
        counts: Dict[str, int] = {}

        def on_post(frame, output):
            module = frame.module
            if not frame.name or frame.had_child or module is head or not _has_own_params(module):
                return
            if "#" in frame.node_id:
                counts[frame.name] = counts.get(frame.name, 1) + 1
                return
            tensor = _first_tensor(output)
            if tensor is None:
                return
            axis = _channel_axis(module, tensor, hints)
            pooled = _pooled_values(tensor, axis, int(tensor.shape[0]) if tensor.ndim else 0)
            if pooled is None:
                return
            mean, _ = pooled
            pair = stats.get(frame.node_id)
            if pair is None:
                pair = (SideStats(), SideStats())
                stats[frame.node_id] = pair
                order.append(frame.node_id)
                modules[frame.node_id] = module
            values = th.from_numpy(mean)
            pair[side].update(values, values.abs(), float(values.abs().sum()), int(values.numel()))

        def tracker():
            return _CallTracker(runner.inner, on_post=on_post)

        total = len(keys)
        runner.run(keys, lambda *a: None, hooks=[tracker],
                   progress=lambda d: self.progress(label, d, total))
        stats.setdefault("_shared", set()).update(counts)

    def _layer_samples(self, nodes: Dict[str, nn.Module], keys: Sequence[Key], label: str,
                       keep_maps: bool = False) -> Dict[str, dict]:
        """Per-sample (mean, max) of *nodes*' neurons, and the sum of their maps."""
        out: Dict[str, dict] = {n: {"mean": [], "max": [], "sum": None, "count": 0, "axis": None,
                                    "shape": None} for n in nodes}
        hints: Dict[Tuple[int, ...], int] = {}

        def make(node, module):
            def hook(mod, args, output):
                tensor = _first_tensor(output)
                if tensor is None:
                    return None
                axis = _channel_axis(mod, tensor, hints)
                pooled = _pooled_values(tensor, axis, int(tensor.shape[0]))
                if pooled is None:
                    return None
                rec = out[node]
                rec["mean"].append(pooled[0])
                rec["max"].append(pooled[1])
                rec["axis"] = axis
                if keep_maps:
                    x = tensor.detach().float().movedim(axis, 1)
                    total = x.sum(dim=0).cpu()
                    if rec["sum"] is None:
                        rec["sum"] = total
                        rec["shape"] = tuple(total.shape)
                    elif tuple(total.shape) == rec["shape"]:
                        rec["sum"] = rec["sum"] + total
                    else:
                        rec["shape"] = None     # variable sizes: no per-position mean
                    rec["count"] += int(x.shape[0])
                return None

            @contextmanager
            def ctx():
                with _module_hook(module, hook):
                    yield
            return ctx

        total = len(keys)
        self.runner.run(keys, lambda *a: None, hooks=[make(n, m) for n, m in nodes.items()],
                        progress=lambda d: self.progress(label, d, total))
        for rec in out.values():
            rec["mean"] = np.concatenate(rec["mean"], axis=0) if rec["mean"] else np.zeros((0, 0))
            rec["max"] = np.concatenate(rec["max"], axis=0) if rec["max"] else np.zeros((0, 0))
        return out

    def _compare(self) -> None:
        from weightslab.components.model_comparison import neuron_effects
        self.set_step("compare")
        a, b = self.analysed, self.lookalikes
        stats: Dict[str, Any] = {}
        self._layer_stats(a, 0, stats, f"Failures through every layer ({len(a)})")
        self._layer_stats(b, 1, stats, f"Look-alikes through every layer ({len(b)})")
        order: List[str] = stats["_order"]
        layers = []
        for node in order:
            sa, sb = stats[node]
            if not sa.n or not sb.n or sa.channels != sb.channels:
                continue
            effect = neuron_effects(sa, sb, "group")
            layers.append({"node": node, "score": float(np.sqrt(np.mean(effect * effect))),
                           "channels": int(sa.channels), "strong": int((np.abs(effect) >= 0.8).sum())})
        if not layers:
            self.say("No layer with weights produced activations I could compare.")
            raise _Stop()
        scores = [l["score"] for l in layers]
        jump = first_jump(scores)
        by_score = sorted(range(len(layers)), key=lambda i: -scores[i])
        chosen_idx = ([jump] if jump >= 0 else []) + [i for i in by_score if i != jump]
        chosen_idx = chosen_idx[: int(self.opts["max_suspect_layers"])]
        for i, layer in enumerate(layers):
            layer["jump"] = i == jump
            layer["chosen"] = i in chosen_idx
        self.layers = layers
        # Per chosen layer: per-sample values -> shuffled baseline -> suspects.
        modules = stats["_modules"]
        nodes = {layers[i]["node"]: modules[layers[i]["node"]] for i in chosen_idx}
        va = self._layer_samples(nodes, a, f"Failures in {len(nodes)} layers")
        vb = self._layer_samples(nodes, b, f"Look-alikes in {len(nodes)} layers", keep_maps=True)
        self.candidates = []
        for i in chosen_idx:
            node = layers[i]["node"]
            ra, rb = va[node], vb[node]
            if ra["mean"].size == 0 or rb["mean"].size == 0 or ra["mean"].shape[1] != rb["mean"].shape[1]:
                continue
            n, c = ra["mean"].shape[0] + rb["mean"].shape[0], ra["mean"].shape[1]
            shuffles = int(min(int(self.opts["shuffles"]), max(50, int(2e8 / max(n * c, 1)))))
            q = float(self.opts["null_quantile"])
            d_mean = _cohen_d(ra["mean"], rb["mean"])
            d_max = _cohen_d(ra["max"], rb["max"])
            t_mean = null_threshold(ra["mean"], rb["mean"], shuffles, q, self.rng)
            t_max = null_threshold(ra["max"], rb["max"], shuffles, q, self.rng)
            strength = np.maximum(np.abs(d_mean) / max(t_mean, 1e-9), np.abs(d_max) / max(t_max, 1e-9))
            idx = [int(j) for j in np.argsort(-strength) if strength[j] > 1.0][: int(self.opts["max_suspects"])]
            mean_b = rb["mean"].mean(axis=0)
            maps = (rb["sum"] / max(rb["count"], 1)) if rb["sum"] is not None and rb["shape"] else None
            self.candidates.append({
                "node": node, "module": modules[node], "axis": ra["axis"], "score": layers[i]["score"],
                "jump": layers[i]["jump"], "suspects": idx, "d_mean": d_mean, "d_max": d_max,
                "t_mean": t_mean, "t_max": t_max, "strength": strength, "mean_b": mean_b, "map_b": maps,
                "avg_a": ra["mean"].mean(axis=0), "avg_b": mean_b, "channels": c})
        jump_node = layers[jump]["node"] if jump >= 0 else ""
        card_layers = {"kind": "layers", "rows": [
            {"node": l["node"], "score": round(l["score"], 3), "jump": l["jump"], "chosen": l["chosen"],
             "channels": l["channels"], "strong": l["strong"]} for l in layers],
            "a_ids": [k[0] for k in a], "a_origins": [k[1] for k in a],
            "b_ids": [k[0] for k in b], "b_origins": [k[1] for k in b]}
        intro = ("Where the failures and their look-alikes part ways. A layer's score is the RMS of its "
                 "neurons' Cohen's d: about 0 means the layer sees both alike, 0.8 or more is a large "
                 "difference.")
        if jump_node:
            intro += (f" The difference **starts at `{jump_node}`**; later layers mostly inherit it, so "
                      "that's where I look first.")
        else:
            intro += " No layer stands out: the failures look like their look-alikes all the way through."
        self.say(intro, card_layers)
        with_suspects = [c for c in self.candidates if c["suspects"]]
        if not with_suspects:
            kind = ("**boundary cases** (close to another class)" if self.task.grouped else
                    "**hard cases** (spread over many features)")
            self.say("No neuron differs between failures and look-alikes more than shuffling the two "
                     f"groups does by chance. These failures look like {kind}, not a single feature the "
                     "model reacts to.")
            self._ask_continue("compare", "Go to the fix?", "Go to the fix")
            self.candidates = []
            return
        rows = []
        for cand in with_suspects:
            for j in cand["suspects"][:8]:
                rows.append([cand["node"], j, round(float(cand["d_mean"][j]), 2),
                             round(float(cand["d_max"][j]), 2),
                             round(float(cand["avg_a"][j]), 3), round(float(cand["avg_b"][j]), 3)])
        first = with_suspects[0]
        skipped = "" if first["node"] == jump_node or not jump_node else (
            f"No neuron of `{jump_node}` alone beats chance (the difference there is spread thin), so "
            f"I'll look at `{first['node']}` first. ")
        self.say(
            f"Suspects: neurons whose difference beats a **shuffled baseline** (the 95th percentile of "
            f"the largest |d| when failures and look-alikes are shuffled together). {skipped}"
            f"`{first['node']}` has **{len(first['suspects'])}** (|d| > {first['t_mean']:.2f} on the "
            f"mean, > {first['t_max']:.2f} on the max). d > 0: more active on the failures; d < 0: a "
            "feature missing on them.",
            {"kind": "table", "columns": ["layer", "neuron", "d (mean)", "d (max)", "failures (avg)",
                                          "look-alikes (avg)"], "rows": rows[:24]})
        self.candidates = with_suspects
        self._ask_continue("compare", "Test the suspects by silencing them (during an eval only)?",
                           "Test the suspects",
                           "Replace them by their look-alike average and count the failures fixed")

    # ---- 4. test ----------------------------------------------------------------
    def _mask_hook(self, cand: dict, neurons: Sequence[int]):
        module = cand["module"]
        axis = cand["axis"]
        idx = th.as_tensor(list(neurons), dtype=th.long)
        map_b = cand["map_b"]
        mean_b = th.as_tensor(cand["mean_b"], dtype=th.float32)

        def hook(mod, args, output):
            if not isinstance(output, th.Tensor):
                return None
            x = output.clone().movedim(axis, 1)
            sel = idx.to(x.device)
            if map_b is not None and tuple(map_b.shape) == tuple(x.shape[1:]):
                value = map_b.to(x.device, x.dtype)[sel]
            else:
                shape = [len(neurons)] + [1] * (x.ndim - 2)
                value = mean_b.to(x.device, x.dtype)[sel].view(shape)
            x[:, sel] = value.expand(x.shape[0], *x[:, sel].shape[1:]) if value.ndim == x.ndim - 1 \
                else value
            return x.movedim(1, axis)

        @contextmanager
        def ctx():
            with _module_hook(module, hook):
                yield
        return ctx

    def _changes(self, evs: Dict[Key, dict]) -> Tuple[List[Key], List[Key], float, float]:
        """(better, worse, mean value before, mean value after) against the
        records taken before any change."""
        task = self.task
        keys = [k for k in evs if k in self.info]
        better = [k for k in keys if task.better(self.info[k], evs[k])]
        worse = [k for k in keys if task.worse(self.info[k], evs[k])]
        before = float(np.mean([self.info[k]["value"] for k in keys])) if keys else 0.0
        after = float(np.mean([evs[k]["value"] for k in keys])) if keys else 0.0
        return better, worse, before, after

    def _try_mask(self, cand: dict, neurons: Sequence[int], with_controls: bool = False,
                  failures: Optional[Sequence[Key]] = None, with_look: bool = True) -> dict:
        hook = self._mask_hook(cand, neurons)
        label = f"Silencing {len(neurons)} neurons of {cand['node']}"
        fails = self._eval_keys(self.analysed if failures is None else failures, label + " (failures)", [hook])
        looks = self._eval_keys(self.lookalikes, label + " (look-alikes)", [hook]) if with_look else {}
        fixed, _, f_before, f_after = self._changes(fails)
        _, broken, l_before, l_after = self._changes(looks)
        res = {"neurons": list(neurons), "fixed": fixed, "broken_look": broken,
               "n_fail": len(fails), "n_look": len(looks),
               "fail_values": (f_before, f_after), "look_values": (l_before, l_after)}
        if with_controls and self.controls:
            ctrl = self._eval_keys(self.controls, label + " (other healthy samples)", [hook])
            res["broken_ctrl"] = self._changes(ctrl)[1]
            res["n_ctrl"] = len(ctrl)
        return res

    def _test(self) -> None:
        self.set_step("test")
        if not self.candidates:
            return
        sub = self.analysed
        if len(sub) > 64:
            sub = [sub[i] for i in sorted(self.rng.choice(len(sub), 64, replace=False))]
        for n, cand in enumerate(self.candidates):
            neurons = cand["suspects"]
            res = self._try_mask(cand, neurons)
            first = self._try_mask(cand, neurons, failures=sub, with_look=False)
            target = max(1, int(math.ceil(0.9 * len(first["fixed"]))))
            narrowed = list(neurons)
            best = first
            while len(narrowed) > 1 and len(best["fixed"]) > 0:
                half = len(narrowed) // 2
                found = False
                for part in (narrowed[:half], narrowed[half:]):
                    r = self._try_mask(cand, part, failures=sub, with_look=False)
                    if len(r["fixed"]) >= target:
                        narrowed, best, found = part, r, True
                        break
                if not found:
                    break
            final = self._try_mask(cand, narrowed, with_controls=True)
            n_fail, n_look = max(final["n_fail"], 1), max(final["n_look"], 1)
            explains = len(final["fixed"]) >= max(1, math.ceil(float(self.opts["confirm_share"]) * n_fail))
            harmless = len(final["broken_look"]) <= float(self.opts["confirm_damage"]) * n_look
            task = self.task
            verb = ("fixed", "broken") if not task.continuous else ("better", "worse")
            rows = [
                {"label": (f"All {len(neurons)} suspects" if len(neurons) > 1 else "The suspect")
                 + f": failures {verb[0]}",
                 "value": f"{len(res['fixed'])}/{res['n_fail']}", "good": len(res["fixed"]) > 0},
                {"label": f"look-alikes {verb[1]}", "value": f"{len(res['broken_look'])}/{res['n_look']}",
                 "good": len(res["broken_look"]) == 0},
            ]
            if task.continuous:
                (fb, fa), (lb, la) = res["fail_values"], res["look_values"]
                rows.append({"label": f"failures: mean {task.metric}",
                             "value": f"{task.fmt(fb)} → {task.fmt(fa)}", "good": fa > fb})
                rows.append({"label": f"look-alikes: mean {task.metric}",
                             "value": f"{task.fmt(lb)} → {task.fmt(la)}",
                             "good": la >= lb - task.delta / 2})
            if len(narrowed) < len(neurons):
                rows.append({"label": f"Narrowed to {len(narrowed)}: failures {verb[0]}",
                             "value": f"{len(final['fixed'])}/{final['n_fail']}", "good": explains})
                rows.append({"label": f"look-alikes {verb[1]}",
                             "value": f"{len(final['broken_look'])}/{final['n_look']}", "good": harmless})
            if "n_ctrl" in final:
                rows.append({"label": f"other healthy samples {verb[1]}",
                             "value": f"{len(final['broken_ctrl'])}/{final['n_ctrl']}",
                             "good": len(final["broken_ctrl"]) <= 0.05 * max(final["n_ctrl"], 1)})
            text = (f"Silencing the suspects of `{cand['node']}`, each replaced by its average on the "
                    "look-alikes, during an eval only (the weights are untouched):")
            self.say(text, {"kind": "metrics", "rows": rows,
                            "neurons": [int(j) for j in narrowed], "node": cand["node"]})
            if explains and harmless:
                self.confirmed = {"cand": cand, "neurons": narrowed, "result": final}
                fixed = len(final["fixed"])
                self.say(f"**Found a mechanism:** {len(narrowed)} neuron(s) of `{cand['node']}` "
                         f"({', '.join(str(j) for j in narrowed[:12])}) explain {fixed} of "
                         f"{final['n_fail']} failures without "
                         f"{'breaking' if not task.continuous else 'degrading'} the look-alikes.")
                self._ask_continue("test", "See what these neurons respond to?", "Explain these neurons")
                return
            nxt = self.candidates[n + 1]["node"] if n + 1 < len(self.candidates) else ""
            if nxt:
                choice = self.ask("test", f"Silencing `{cand['node']}`'s suspects doesn't explain the failures "
                                  f"without damage. Try the next layer, `{nxt}`?",
                                  [{"id": "next", "label": f"Try `{nxt}`", "primary": True},
                                   {"id": "skip", "label": "Go to the fix"},
                                   {"id": "stop", "label": "Stop here"}])
                if choice == "stop":
                    raise _Stop()
                if choice == "skip":
                    break
            else:
                kind = "**boundary cases**" if self.task.grouped else "**hard cases**"
                self.say("No small set of neurons explains these failures without damage: they look "
                         f"like {kind} rather than one feature the model reacts to.")
                self._ask_continue("test", "Go to the fix?", "Go to the fix")

    # ---- 5. explain ---------------------------------------------------------------
    def _explain(self) -> None:
        self.set_step("explain")
        conf = self.confirmed
        if conf is not None:
            cand = conf["cand"]
            neurons = sorted(conf["neurons"], key=lambda j: -float(cand["strength"][j]))[
                : int(self.opts["explain_neurons"])]
            budget = int(self.opts["top_pool"])
            if self.sec_per_sample > 0:
                budget = min(budget, max(256, int(float(self.opts["pool_seconds"]) / self.sec_per_sample)))
            pool = self.pool if len(self.pool) <= budget else \
                [self.pool[i] for i in sorted(self.rng.choice(len(self.pool), budget, replace=False))]
            values = self._layer_samples({cand["node"]: cand["module"]}, pool,
                                         f"What {len(neurons)} neurons respond to ({len(pool)} samples)")
            vals = values[cand["node"]]["mean"]
            k = int(self.opts["top_samples"])
            rows, top_keys = [], []
            for j in neurons:
                if vals.size == 0:
                    break
                order = np.argsort(-vals[:, j])[:k]
                keys = [pool[i] for i in order]
                top_keys.extend(keys)
                rows.append({"neuron": int(j), "effect": round(float(cand["d_mean"][j]), 2),
                             "samples": self.sample_items(keys)})
            self._tag(TAGS["top"], top_keys)
            self.say(f"The samples of train / val that each neuron of `{cand['node']}` responds to most "
                     f"(tagged `{TAGS['top']}`). If they share something (a background, a mark, a "
                     "lighting), that's what the model wrongly relies on for the failures.",
                     {"kind": "neurons", "node": cand["node"], "rows": rows})
        # One category per failure.
        fixed = set(conf["result"]["fixed"]) if conf else set()
        near = set(self.near_pred)
        task = self.task
        cats: Dict[str, List[Key]] = {"label": [], "spurious": [], "ood": [], "boundary": []}
        for key in self.failures:
            if key in fixed:
                cats["spurious"].append(key)
            elif key in near:
                cats["label"].append(key)
            elif key in set(self.ood):
                cats["ood"].append(key)
            else:
                cats["boundary"].append(key)
        self.categories = cats
        self.label_suspects = list(cats["label"])
        self._tag(TAGS["label"], self.label_suspects)
        meta = {
            "label": ("Look mislabeled", "confident, among the predicted class, not explained by "
                                         "any neuron: review their labels", TAGS["label"]),
            "spurious": ("Spurious feature", "the neurons above explain them", ""),
            "ood": ("Unlike training", "collect more data like them", TAGS["ood"]),
            "boundary": ("Boundary cases", "close to another class; fine-tuning with their "
                                           "look-alikes helps", "") if task.grouped else
                        ("Hard cases", "no single feature explains them; fine-tuning with their "
                                       "look-alikes helps", ""),
        }
        rows = [{"key": key, "label": meta[key][0], "hint": meta[key][1], "tag": meta[key][2],
                 "count": len(ks), "items": self.sample_items(ks[:6])} for key, ks in cats.items() if ks]
        self.say("One category per failure:", {"kind": "categories", "rows": rows})
        if task.can_train:
            options = [
                {"id": "both", "label": "Tag for data work + repair the model", "primary": True,
                 "hint": "Tags first, then fine-tune candidates checked by a gate"},
                {"id": "repair", "label": "Repair the model",
                 "hint": "Fine-tune from the current weights; kept only if it passes the checks"},
                {"id": "tags", "label": "Tag samples for data work only"},
                {"id": "stop", "label": "Stop here"},
            ]
        else:
            self.say("I can't repair this model by myself: I have no loss to train it with. Register "
                     "yours with `wl.watch_or_edit(criterion, flag=\"loss\")` (one value per sample) "
                     "and start again to try repairs.")
            options = [{"id": "tags", "label": "Tag samples for data work", "primary": True},
                       {"id": "stop", "label": "Stop here"}]
        choice = self.ask("explain", "How do you want to fix it?", options)
        if choice == "stop":
            raise _Stop()
        self._fix_choice = choice

    # ---- 6. fix --------------------------------------------------------------------
    def _fix(self) -> None:
        self.set_step("fix")
        choice = getattr(self, "_fix_choice", "both")
        if choice in ("tags", "both"):
            rows = [{"tag": t, "count": n} for t, n in self.tagged.items()]
            if rows:
                todo = {
                    TAGS["label"]: f"review the labels of `{TAGS['label']}`",
                    TAGS["ood"]: f"collect more samples like `{TAGS['ood']}`",
                    TAGS["top"]: f"check what `{TAGS['top']}` share",
                    TAGS["lookalikes"]: f"`{TAGS['lookalikes']}` are the healthy references",
                }
                steps = [text for tag, text in todo.items() if tag in self.tagged]
                self.say("Tagged for data work" + (": " + "; ".join(steps) if steps else "") +
                         ". Filter the grid by these tags.", {"kind": "tags", "rows": rows})
            else:
                self.say("There's no sample table to tag in (no registered dataframe).")
        if choice in ("repair", "both"):
            self._repair()
        paused = ("Training is paused (I paused it for the repairs): press Play to continue. "
                  if getattr(self, "_paused_by_me", False) else "")
        self.say("Done. " + ("The tags stay in the grid. " if self.tagged else "") + paused
                 + "Start a new analysis from any tag or selection.")

    # ---- repair ------------------------------------------------------------------
    def _hold_training(self) -> bool:
        from weightslab.components.global_monitoring import pause_controller
        if not pause_controller.is_paused():
            choice = self.ask("pause", "Training is running. Pause it while I try the repairs?",
                              [{"id": "pause", "label": "Pause training and repair", "primary": True},
                               {"id": "skip", "label": "Skip the repair"}])
            if choice != "pause":
                return False
            try:
                pause_controller.pause()
                self._paused_by_me = True
            except Exception as exc:
                logger.debug(f"[failures] pause failed: {exc}")
        try:
            pause_controller.hold("a repair is running in the failure analysis")
            self._held = True
        except Exception as exc:
            logger.debug(f"[failures] hold failed: {exc}")
        return True

    def _release_hold(self) -> None:
        if not self._held:
            return
        try:
            from weightslab.components.global_monitoring import pause_controller
            pause_controller.release()
        except Exception:
            pass
        self._held = False

    def _rollback_repair(self) -> None:
        if self._snapshot is None:
            return
        try:
            with _training_lock(timeout_s=120.0):
                _load_weights(self.runner.inner, self._snapshot)
        except Exception as exc:
            logger.error(f"[failures] could not roll the repair back: {exc}")
        self._snapshot = None

    def _guard_keys(self) -> Tuple[List[Key], str]:
        runner = self.runner
        vals = [e for e in runner.entries if _split_kind(e) == "val"]
        failing = set(self.requested)
        if vals:
            entries, where = vals, "the validation split"
        else:
            splits = {k[1] for k in self.failures}
            entries = [e for e in runner.entries if e[3] in splits] or \
                [e for e in runner.entries if _split_kind(e) == "test"]
            where = (f"`{entries[0][3]}` without the analysed samples (no validation split)"
                     if entries else "")
        keys = [(sid, e[3]) for e in entries for sid in _dataset_ids(e[2])]
        keys = [k for k in keys if k not in failing]
        if len(keys) < 20:
            # Every sample of that split is under analysis: guard on training
            # samples the repairs do not focus on (they still see them in replay).
            focus = failing | set(self.lookalikes) | set(self.confusers)
            trains = [e for e in runner.entries if _split_kind(e) == "train"]
            spare = [(sid, e[3]) for e in trains for sid in _dataset_ids(e[2])]
            spare = [k for k in spare if k not in focus]
            if len(spare) > len(keys):
                keys = spare
                where = (f"`{trains[0][3]}`, minus the failures and look-alikes "
                         "(the other splits hold only analysed samples)")
        budget = int(self.opts["guard_max"])
        if self.sec_per_sample > 0:
            budget = min(budget, max(200, int(float(self.opts["guard_seconds"]) / self.sec_per_sample)))
        if len(keys) > budget:
            keys = [keys[i] for i in sorted(self.rng.choice(len(keys), budget, replace=False))]
        return keys, where

    def _optimizer_for(self, params: List[nn.Parameter]):
        lr = float(self.opts["repair_lr"])
        cls, defaults = th.optim.Adam, {}
        try:
            from weightslab.backend.ledgers import get_optimizer
            user = get_optimizer()
            groups = getattr(user, "param_groups", None)
            if groups:
                lr = float(groups[0].get("lr", lr)) * float(self.opts["repair_lr_scale"])
                base = getattr(user, "optimizer", user)
                if isinstance(base, th.optim.Optimizer):
                    cls = type(base)
                    defaults = {k: v for k, v in base.defaults.items()
                                if k not in ("lr", "params", "foreach", "fused", "capturable",
                                             "differentiable")}
        except Exception:
            pass
        try:
            return cls(params, lr=lr, **defaults), lr, cls.__name__
        except Exception:
            return th.optim.Adam(params, lr=lr), lr, "Adam"

    def _repair(self) -> None:
        if not self.task.can_train:
            self.say("No repair: I have no loss to train this model with.")
            return
        if not self._hold_training():
            self.say("Repair skipped; training was left running.")
            return
        runner = self.runner
        inner = runner.inner
        usable = [k for k in self.failures if k not in set(self.label_suspects)]
        if not usable:
            self.say("Every failure looks mislabeled: a repair would teach the wrong labels. Review them first.")
            return
        order = [usable[i] for i in self.rng.permutation(len(usable))]
        if len(order) >= 4:
            probe, fix = order[::2], order[1::2]
            probe_note = (f"{len(fix)} failures to train on and {len(probe)} held out, to see whether "
                          "the fix carries over to failures it never saw")
        else:
            probe, fix = [], order
            probe_note = (f"only {len(order)} failure(s): none can be held out, so a fix measured on them "
                          "may just be memorised")
        trains = {e[3] for e in runner.entries if _split_kind(e) == "train"}
        look_tr = [k for k in self.lookalikes if k[1] in trains] or list(self.lookalikes)
        conf_tr = [k for k in self.confusers if k[1] in trains] or list(self.confusers)
        replay = [k for k in self.pool if k[1] in trains] or list(self.pool)
        guard, where = self._guard_keys()
        cand = self.confirmed["cand"] if self.confirmed else (self.candidates[0] if self.candidates else None)
        neurons = self.confirmed["neurons"] if self.confirmed else []
        options = [{"id": "control", "label": "Control: keep training on normal data",
                    "hint": "The reference: without it, 'better' may just mean 'trained longer'"},
                   {"id": "replay", "label": "Fine-tune with failures + look-alikes + replay",
                    "hint": "Everything trainable, pulled back toward the current weights"}]
        if cand is not None:
            options.append({"id": "targeted", "label": f"Train only `{cand['node']}`",
                            "hint": "Everything else frozen: the smallest change"})
        if cand is not None and neurons:
            options.append({"id": "reset", "label": f"Reset the {len(neurons)} neuron(s), then fine-tune",
                            "hint": "Forget what they learned, relearn it with the look-alikes"})
        mix = "failures + look-alikes + confusers" if self.task.grouped else "failures + look-alikes"
        self.say(f"Repair plan: {probe_note}. Every candidate starts from the current weights, mixes "
                 f"{int(float(self.opts['focus_share']) * 100)}% {mix} with "
                 f"{100 - int(float(self.opts['focus_share']) * 100)}% normal training samples, and is "
                 f"checked on {len(guard)} samples of {where}. The model only changes if you accept.")
        picked = self.ask("repair", "Which repairs should I try?", options, multiple=True,
                          defaults=[o["id"] for o in options], confirm="Run the repairs")
        if not picked:
            self.say("No repair selected.")
            return
        if "control" not in picked:
            picked = ["control"] + picked
        # Baseline (the current weights) on every evaluated set.
        base_guard = self._eval_keys(guard, "Current model on the gate samples")
        self._snapshot = _weights_of(inner)
        start = {n: p.detach().clone() for n, p in inner.named_parameters()}
        focus = fix * 2 + look_tr + conf_tr
        train_entry = next((e for e in runner.entries if _split_kind(e) == "train"), runner.entries[0])
        bsz = runner.chunk_size(train_entry)
        n_focus = max(1, int(round(bsz * float(self.opts["focus_share"]))))
        steps = int(self.opts["repair_steps"]) or max(int(self.opts["repair_min_steps"]),
                                                      int(math.ceil(3 * len(focus) / n_focus)))
        self._steps = min(steps, int(self.opts["repair_max_steps"]))
        results = []
        for i, name in enumerate(picked):
            if self.cancel.is_set():
                raise _Cancelled()
            with _training_lock(timeout_s=120.0, cancel=self.cancel):
                _load_weights(inner, self._snapshot)
            if name == "reset":
                self._reset_rows(cand, neurons)
            trainable = self._trainable(name, cand)
            res = self._finetune(name, i, len(picked), trainable,
                                 [] if name == "control" else focus, replay, start, bsz, n_focus)
            res.update(self._evaluate(name, fix, probe, guard, base_guard))
            res["state"] = _weights_of(inner)
            results.append(res)
        with _training_lock(timeout_s=120.0):
            _load_weights(inner, self._snapshot)
        self._gate(results, bool(probe))
        best = max((r for r in results if r["pass"] and r["name"] != "control"),
                   key=lambda r: r["score"], default=None)
        labels = {o["id"]: o["label"] for o in options}
        task = self.task
        verb = "fixed" if not task.continuous else "better"
        table = {"kind": "repair", "columns": ["repair", f"failures {verb} (trained)",
                                               f"held-out {verb}", f"gate {task.metric}",
                                               "flips" if not task.continuous else "worse", "passes"],
                 "rows": [{"name": r["name"], "label": labels.get(r["name"], r["name"]),
                           "fix": f"{r['fix_fixed']}/{r['n_fix']}",
                           "probe": f"{r['probe_fixed']}/{r['n_probe']}" if r["n_probe"] else "-",
                           "acc": f"{task.fmt(r['value'])} ({task.fmt_delta(r['value'] - r['base_value'])})",
                           "flips": r["flips"], "pass": r["pass"], "best": best is r,
                           "note": r.get("why", ""), "steps": r["steps"]} for r in results],
                 "baseline_acc": results[0]["base_acc"] if results else 0.0,
                 "baseline_text": (f"Current model on the gate samples: {task.metric} "
                                   f"{task.fmt(results[0]['base_value'])}.") if results else ""}
        self.repair_results = [{k: v for k, v in r.items() if k != "state"} for r in results]
        if best is None:
            # The closest one: fixes more than the control, fails only on the guard.
            control_ref = next((r["probe_fixed"] if probe else r["fix_fixed"]
                                for r in results if r["name"] == "control"), 0)
            near = max((r for r in results if r["name"] != "control" and
                        (r["probe_fixed"] if probe else r["fix_fixed"]) > control_ref),
                       key=lambda r: r["score"], default=None)
            self.say(f"No repair passed the checks (failures fixed beyond the control, {self.task.metric} "
                     "kept, few flips): the model stays as it was. The tags point at the data that would "
                     "help.", table)
            if near is None:
                self._snapshot = None
                self._release_hold()
                return
            name = labels.get(near["name"], near["name"])
            choice = self.ask("accept", f"The closest is \"{name}\" ({near.get('why', '')}). Apply it anyway?",
                              [{"id": "keep", "label": "Keep the original model", "primary": True},
                               {"id": "accept", "label": "Apply it anyway",
                                "hint": "Your call: it fails the checks above"}])
            best = near if choice == "accept" else None
            if best is None:
                self.say("Kept the original model.")
                self._snapshot = None
                self._release_hold()
                return
            choice = "accept"
        else:
            self.say(f"Best: **{labels.get(best['name'], best['name'])}**. It passes every check.", table)
            choice = self.ask("accept", f"Apply \"{labels.get(best['name'], best['name'])}\" to the live model?",
                              [{"id": "accept", "label": "Accept the repair", "primary": True},
                               {"id": "keep", "label": "Keep the original model"}])
        if choice == "accept":
            with _training_lock(timeout_s=120.0):
                _load_weights(inner, best["state"])
            self._mark_cache.clear()
            fixed = best.get("fixed_keys", [])
            self._tag(TAGS["fixed"], fixed)
            self.say(f"Applied. {len(fixed)} failure(s) are now {'right' if not self.task.continuous else 'better'} "
                     f"(tagged `{TAGS['fixed']}`). Training "
                     "stays paused: press Play to continue; the new weights are saved with the next "
                     "checkpoint.")
        else:
            self.say("Kept the original model.")
        self._snapshot = None
        self._release_hold()

    def _trainable(self, name: str, cand: Optional[dict]) -> List[nn.Parameter]:
        inner = self.runner.inner
        if name == "targeted" and cand is not None:
            params = [p for p in cand["module"].parameters() if p.requires_grad]
            if params:
                return params
        return [p for p in inner.parameters() if p.requires_grad]

    def _reset_rows(self, cand: dict, neurons: Sequence[int]) -> None:
        module = cand["module"]
        weight = getattr(module, "weight", None)
        if not isinstance(weight, th.Tensor) or weight.ndim < 1:
            return
        from weightslab.utils.neuron_graph import neuron_axis
        axis = neuron_axis(module)
        idx = th.as_tensor(list(neurons), dtype=th.long, device=weight.device)
        with _training_lock(timeout_s=120.0), th.no_grad():
            std = float(weight.detach().float().std()) or 0.02
            rows = weight.index_select(axis, idx)
            weight.index_copy_(axis, idx, th.randn_like(rows) * std)
            bias = getattr(module, "bias", None)
            if isinstance(bias, th.Tensor) and axis == 0:
                bias.index_fill_(0, idx, 0.0)

    def _finetune(self, name: str, i: int, n: int, params: List[nn.Parameter], focus: List[Key],
                  replay: List[Key], start: Dict[str, th.Tensor], bsz: int, n_focus: int) -> dict:
        """Fine-tune the live weights for self._steps steps (the same for every
        candidate; the first one shrinks it when a step is slow)."""
        runner = self.runner
        inner = runner.inner
        opt, lr, opt_name = self._optimizer_for(params)
        n_focus = n_focus if focus else 0
        steps = self._steps
        anchor = float(self.opts["anchor"])
        names = {id(p): nm for nm, p in inner.named_parameters()}
        anchored = [(p, start[names[id(p)]]) for p in params if id(p) in names]
        from weightslab.projection.parametric_umap import _frozen_age
        label = f"Repair {i + 1}/{n}: {name}"
        t0 = time.monotonic()
        done = 0
        for step in range(steps):
            if self.cancel.is_set():
                raise _Cancelled()
            picks = []
            if n_focus:
                picks += [focus[j] for j in self.rng.integers(0, len(focus), n_focus)]
            if replay:
                picks += [replay[j] for j in self.rng.integers(0, len(replay), max(bsz - len(picks), 1))]
            with _training_lock(timeout_s=120.0, cancel=self.cancel):
                batch, _ = runner.batch_of(picks)
                inputs = _to_device(_inputs_from_batch(batch, inner), runner.device)
                targets = _to_device(_targets_from_batch(batch, inner), runner.device)
                was_training = inner.training
                inner.train()
                try:
                    with _frozen_age(self.model), th.enable_grad():
                        loss = self.task.loss(_call_model(inner, inputs), targets, inputs)
                        if anchor > 0 and anchored:
                            loss = loss + anchor * sum(((p - p0) ** 2).sum() for p, p0 in anchored)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                finally:
                    inner.train(was_training)
            done = step + 1
            self.progress(f"{label} — step {done}/{steps}", done, steps)
            elapsed = time.monotonic() - t0
            if i == 0 and done >= 5 and elapsed / done * steps > float(self.opts["repair_seconds"]) * 1.2:
                steps = max(done, int(float(self.opts["repair_seconds"]) / (elapsed / done)))
                self._steps = steps
            if done >= steps:
                break
        opt.zero_grad(set_to_none=True)
        return {"name": name, "steps": done, "lr": lr, "optimizer": opt_name}

    def _evaluate(self, name: str, fix: List[Key], probe: List[Key], guard: List[Key],
                  base_guard: Dict[Key, dict]) -> dict:
        """A candidate against the current weights: failures fixed (better),
        gate samples made worse / better, the gate's mean score and its
        worst per-class drop."""
        task = self.task
        fixres = self._eval_keys(fix, f"Checking {name}: trained failures")
        proberes = self._eval_keys(probe, f"Checking {name}: held-out failures") if probe else {}
        guardres = self._eval_keys(guard, f"Checking {name}: gate samples")
        keys = [k for k in guard if k in base_guard and k in guardres]
        n = max(len(keys), 1)
        flips = sum(1 for k in keys if task.worse(base_guard[k], guardres[k]))
        gains = sum(1 for k in keys if task.better(base_guard[k], guardres[k]))
        base_acc = sum(float(base_guard[k]["score"]) for k in keys) / n
        acc = sum(float(guardres[k]["score"]) for k in keys) / n
        base_value = sum(float(base_guard[k]["value"]) for k in keys) / n
        value = sum(float(guardres[k]["value"]) for k in keys) / n
        # Per class: summed parts (right / 1, intersection / union, found / boxes).
        sums: Dict[Any, List[float]] = {}
        counts: Dict[Any, int] = {}
        for k in keys:
            for c, (num, den) in (base_guard[k].get("parts") or {}).items():
                acc_c = sums.setdefault(c, [0.0, 0.0, 0.0, 0.0])
                acc_c[0] += num
                acc_c[1] += den
                counts[c] = counts.get(c, 0) + 1
            for c, (num, den) in (guardres[k].get("parts") or {}).items():
                acc_c = sums.setdefault(c, [0.0, 0.0, 0.0, 0.0])
                acc_c[2] += num
                acc_c[3] += den
        worst, worst_allowed = 0.0, 0.0
        for c, (bn, bd, nn_, nd) in sums.items():
            if bd <= 0 or counts.get(c, 0) == 0:
                continue
            drop = bn / bd - (nn_ / nd if nd > 0 else 0.0)
            allowed = max(float(self.opts["gate_class_drop"]), 1.0 / counts[c])
            if drop - allowed > worst - worst_allowed:
                worst, worst_allowed = drop, allowed

        def better(res):
            return [k for k, ev in res.items() if k in self.info and task.better(self.info[k], ev)]

        fixed_keys = better(fixres) + better(proberes)
        return {"n_fix": len(fixres), "fix_fixed": len(better(fixres)),
                "n_probe": len(proberes), "probe_fixed": len(better(proberes)),
                "n_guard": len(keys), "flips": flips, "gains": gains, "base_acc": base_acc, "acc": acc,
                "acc_delta": acc - base_acc, "base_value": base_value, "value": value,
                "class_drop": worst, "class_allowed": worst_allowed, "fixed_keys": fixed_keys}

    def _gate(self, results: List[dict], has_probe: bool) -> None:
        control = next((r for r in results if r["name"] == "control"), None)
        ref = (control["probe_fixed"] if has_probe else control["fix_fixed"]) if control else 0
        for r in results:
            n = max(r["n_guard"], 1)
            target = r["probe_fixed"] if has_probe else r["fix_fixed"]
            task = self.task
            checks = [
                (r["flips"] <= max(1, float(self.opts["gate_flip_rate"]) * n, 0.25 * r["gains"]),
                 f"{r['flips']} {'flips' if not task.continuous else 'worse'}"),
                (-r["acc_delta"] <= max(float(self.opts["gate_acc_drop"]), 1.0 / n),
                 f"{task.metric} {task.fmt_delta(r['value'] - r['base_value'])}"),
                (r["class_drop"] <= r["class_allowed"], f"a class drops {r['class_drop'] * 100:.1f} pt"),
                (target > ref if r["name"] != "control" else True, "no more failures fixed than the control"),
                (target >= 1 if r["name"] != "control" else True, "no failure fixed"),
            ]
            failed = [why for ok, why in checks if not ok]
            r["pass"] = not failed and r["name"] != "control"
            r["why"] = "reference" if r["name"] == "control" else ("; ".join(failed) if failed else "")
            r["score"] = target - r["flips"]

    # ---- tags ------------------------------------------------------------------------
    def _tag(self, tag: str, keys: Sequence[Key]) -> None:
        ids = list(dict.fromkeys(k[0] for k in keys))
        if not ids:
            return
        try:
            from weightslab.backend.ledgers import get_dataframe
            manager = get_dataframe()
            if manager is None or not hasattr(manager, "upsert_df"):
                return
            from weightslab.src import tag_samples
            if tag_samples(ids, tag, mode="add"):
                self.tagged[tag] = self.tagged.get(tag, 0) + len(ids)
        except Exception as exc:
            logger.debug(f"[failures] tagging {tag} failed: {exc}")


# =============================================================================
# Sessions
# =============================================================================
class FailureAnalysisManager:
    """Process-wide failure analyses: the latest few, one running at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: "Dict[str, FailureSession]" = {}
        self._latest = ""

    @staticmethod
    def _keys(tag: str, sample_ids: Sequence[str], origins: Sequence[str], origin: str,
              loaders) -> Tuple[List[Key], str]:
        if tag:
            ids, orgs = samples_with_tag(tag, origin)
            label = f"the samples tagged `{tag}`"
        else:
            ids, orgs = [str(s) for s in sample_ids], [str(o) for o in origins]
            label = "the selected samples" if len(ids) != 1 else f"sample `{ids[0]}`"
        resolved = resolve_samples(loaders, ids, orgs)
        return list(dict.fromkeys((sid, split) for sid, split, _ in resolved)), label

    def start(self, model, loaders, tag: str = "", sample_ids: Sequence[str] = (),
              origins: Sequence[str] = (), origin: str = "", automatic: bool = False,
              options: Optional[Dict[str, Any]] = None) -> FailureSession:
        keys, label = self._keys(tag, sample_ids, origins, origin, loaders)
        if not keys:
            raise ValueError("no sample found" + (f" with tag {tag!r}" if tag else ""))
        previous = self.get()
        if previous is not None and previous.state in ("running", "waiting"):
            previous.cancel.set()
            if previous.thread is not None:
                previous.thread.join(timeout=30)
        session = FailureSession(model, loaders, keys, label, automatic, options or {})
        session.runner = _Runner(model, loaders, session.cancel, int(session.opts["batch"]))
        with self._lock:
            self._sessions[session.id] = session
            self._latest = session.id
            for old in list(self._sessions)[:-4]:
                self._sessions.pop(old, None)
        session.thread = threading.Thread(target=session.run, name=f"wl-failures-{session.id}",
                                          daemon=True)
        session.thread.start()
        return session

    def get(self, session_id: str = "") -> Optional[FailureSession]:
        with self._lock:
            return self._sessions.get(session_id or self._latest)

    def wait(self, session_id: str, timeout: float = 120.0) -> Optional[FailureSession]:
        session = self.get(session_id)
        if session is not None and session.thread is not None:
            session.thread.join(timeout=timeout)
        return session


FAILURES = FailureAnalysisManager()
