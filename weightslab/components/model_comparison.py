"""Compare A vs B, layer by layer, for the Weights Studio "Modeling" tab.

Run the model on two sets of samples -- A and B: a tag, a grid selection, or
one sample each -- and tell, for every layer, how differently it represents
them. The question it answers: *where* in the network do A and B part ways?
(Failing vs passing samples of one class, a domain shift, a suspicious slice.)

Two modes, picked from the sides' sizes:

* **pair** -- one sample against one sample. A layer's score is the
  normalized squared distance of the two activation maps,
  ``|a - b|^2 / (|a|^2 + |b|^2)`` in [0, 2]: 0 = identical, 1 = unrelated,
  2 = opposite (it equals the cosine distance when both maps carry the same
  energy, and grows when one of them is weaker). The maps are kept whole --
  where things fire matters -- down-sampled only beyond ``MAP_BUDGET`` values.
* **group** -- sets of samples. Each sample's neurons are pooled (a channel's
  mean over its positions, a feature as is), and a neuron's *effect* is
  Cohen's d of A against B: difference of means over the pooled standard
  deviation, clipped to +-``EFFECT_CLIP``. A layer's score is the RMS effect
  over its neurons: ~0 means A and B look alike there, >= 0.8 is a large
  difference.

Either way, per neuron: the two means, the two firing rates, and the effect,
so a layer can be read as "these neurons respond to A, those to B, these only
to A". Nothing is stored per sample beyond running sums: a comparison costs
two representation passes' worth of forwards and O(neurons) memory per layer.

Samples in both A and B are left out of both (``overlap_dropped``): "the
failing ones of class 3" against "class 3" compares the failing ones with the
rest. The forwards reuse the representation machinery (same node ids, same
hooks, the training lock per chunk, eval mode, frozen age).
"""
from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F

from weightslab.components.model_representation import (
    INPUT_NODE,
    MANAGER,
    OUTPUT_NODE,
    ModelRepresentationManager,
    _CallTracker,
    _CHANNEL_FIRST_TYPES,
    _FEATURE_LAST_TYPES,
    _Cancelled,
    _call_model,
    _channel_axis,
    _firing,
    _first_tensor,
    _inference_mode,
    _inputs_from_batch,
    _model_device,
    _to_device,
    _training_lock,
    build_batch,
    resolve_inner,
    resolve_samples,
    samples_with_tag,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_SAMPLES = 256      # per side
EFFECT_CLIP = 10.0             # |Cohen's d| beyond this says nothing more
STRONG_EFFECT = 0.8            # "large" by Cohen's convention
ONLY_HIGH = 0.5                # "only A": fires on at least half of A...
ONLY_LOW = 0.1                 # ...and on at most a tenth of B
MAP_BUDGET = 1 << 18           # values kept per layer for the pair distance


# =============================================================================
# Statistics
# =============================================================================
def _pooled(tensor: th.Tensor, axis: Optional[int], batch: int) -> Optional[Tuple[th.Tensor, th.Tensor]]:
    """Per-sample neuron values (B, C) -- a channel's mean over its positions --
    and their largest |value| (B, C); None without a per-sample batch dim."""
    if tensor.ndim == 0 or tensor.shape[0] != batch:
        return None
    x = th.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if axis is None or x.ndim < 2 or axis == 0:
        flat = x.reshape(batch, 1, -1)              # the whole output as one neuron
    else:
        flat = x.movedim(axis, 1).reshape(batch, x.shape[axis], -1)
    if flat.shape[2] == 0:
        return None
    return flat.mean(dim=2), flat.abs().amax(dim=2)


def _map_of(tensor: th.Tensor, batch: int) -> Optional[th.Tensor]:
    """The (single) sample's whole activation, flat, float32 on the CPU --
    average-pooled down to MAP_BUDGET values if it is larger. Two samples'
    maps of one layer have the same shape, so they are pooled alike."""
    if tensor.ndim == 0 or tensor.shape[0] != batch or batch != 1:
        return None
    m = th.nan_to_num(tensor.detach()[0].float(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if m.numel() > MAP_BUDGET:
        m = F.adaptive_avg_pool1d(m.view(1, 1, -1), MAP_BUDGET).view(-1)
    return m.cpu()


def _normalized_sq_distance(a, b) -> float:
    """|a - b|^2 / (|a|^2 + |b|^2), in [0, 2]; 0 when both are zero."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    energy = float(a @ a + b @ b)
    if energy <= 1e-30:
        return 0.0
    d = a - b
    return float(min(max(d @ d / energy, 0.0), 2.0))


def _cosine_distance(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-30 and nb <= 1e-30:
        return 0.0
    if na <= 1e-30 or nb <= 1e-30:
        return 1.0
    return float(min(max(1.0 - (a @ b) / (na * nb), 0.0), 2.0))


class SideStats:
    """Running per-neuron sums of one side (A or B) at one node."""

    __slots__ = ("n", "channels", "s1", "s2", "fired", "abs_sum", "abs_count")

    def __init__(self):
        self.n = 0
        self.channels: Optional[int] = None
        self.s1: Optional[np.ndarray] = None
        self.s2: Optional[np.ndarray] = None
        self.fired: Optional[np.ndarray] = None
        self.abs_sum = 0.0
        self.abs_count = 0

    def update(self, values: th.Tensor, absmax: th.Tensor, abs_sum: float, count: int) -> None:
        channels = int(values.shape[1])
        if self.channels is None:
            self.channels = channels
            self.s1 = np.zeros(channels, dtype=np.float64)
            self.s2 = np.zeros(channels, dtype=np.float64)
            self.fired = np.zeros(channels, dtype=np.int64)
        elif channels != self.channels:
            return
        v = values.double()
        self.s1 += v.sum(dim=0).cpu().numpy()
        self.s2 += (v * v).sum(dim=0).cpu().numpy()
        self.fired += _firing(absmax).sum(dim=0).cpu().numpy().astype(np.int64)
        self.n += int(values.shape[0])
        self.abs_sum += abs_sum
        self.abs_count += count

    def mean(self) -> np.ndarray:
        return self.s1 / max(self.n, 1)

    def var(self) -> np.ndarray:
        """Unbiased per-neuron variance (zeros for a single sample)."""
        if self.n < 2:
            return np.zeros_like(self.s1)
        mean = self.mean()
        return np.maximum(self.s2 / self.n - mean * mean, 0.0) * self.n / (self.n - 1)

    def rate(self) -> np.ndarray:
        return self.fired / max(self.n, 1)

    def mean_abs(self) -> float:
        return self.abs_sum / self.abs_count if self.abs_count else 0.0


def neuron_effects(a: SideStats, b: SideStats, mode: str) -> np.ndarray:
    """Per-neuron effect of A against B (> 0: stronger on A).

    group: Cohen's d with the pooled standard deviation, floored at a small
    share of the layer's typical spread (a neuron constant on both sides but
    at two different values is a real, large difference -- it clips).
    pair: (a - b) / the layer's RMS value, so neurons compare within a layer.
    """
    ma, mb = a.mean(), b.mean()
    if mode == "pair":
        scale = math.sqrt(float(np.mean((ma * ma + mb * mb) / 2.0))) if len(ma) else 0.0
        effect = (ma - mb) / max(scale, 1e-12)
    else:
        dof = a.n + b.n - 2
        pooled = ((a.n - 1) * a.var() + (b.n - 1) * b.var()) / dof if dof > 0 else np.zeros_like(ma)
        sd = np.sqrt(np.maximum(pooled, 0.0))
        floor = max(1e-3 * float(sd.mean()) if len(sd) else 0.0,
                    1e-6 * float(np.max(np.abs(np.concatenate([ma, mb])))) if len(ma) else 0.0,
                    1e-12)
        effect = (ma - mb) / np.maximum(sd, floor)
    return np.clip(effect, -EFFECT_CLIP, EFFECT_CLIP)


def node_diff(node_id: str, a: SideStats, b: SideStats, mode: str,
              map_score: Optional[float] = None) -> Optional[dict]:
    """The per-layer verdict (ModelNodeDiff), or None while a side is empty."""
    if not a.n or not b.n or a.channels is None or a.channels != b.channels:
        return None
    effect = neuron_effects(a, b, mode)
    ma, mb = a.mean(), b.mean()
    ra, rb = a.rate(), b.rate()
    if mode == "pair":
        score = map_score if map_score is not None else _normalized_sq_distance(ma, mb)
    else:
        score = float(np.sqrt(np.mean(effect * effect))) if len(effect) else 0.0
    return {
        "node_id": node_id,
        "score": float(score),
        "cosine_distance": _cosine_distance(ma, mb),
        "channels": int(a.channels),
        "strong_channels": int((np.abs(effect) >= STRONG_EFFECT).sum()),
        "only_a": int(((ra >= ONLY_HIGH) & (rb <= ONLY_LOW)).sum()),
        "only_b": int(((rb >= ONLY_HIGH) & (ra <= ONLY_LOW)).sum()),
        "rate_a": float(ra.mean()) if len(ra) else 0.0,
        "rate_b": float(rb.mean()) if len(rb) else 0.0,
        "mean_abs_a": a.mean_abs(),
        "mean_abs_b": b.mean_abs(),
    }


# =============================================================================
# Sessions
# =============================================================================
class ComparisonSide:
    def __init__(self, label: str, tag: str, requested: int, resolved):
        self.label = label
        self.tag = tag
        self.requested = requested
        self.resolved = resolved                   # [(sample_id, origin, loader entry)]

    def status(self) -> dict:
        return {
            "label": self.label, "tag": self.tag, "requested": self.requested,
            "count": len(self.resolved),
            "sample_ids": [r[0] for r in self.resolved],
            "origins": [r[1] for r in self.resolved],
        }


class ComparisonSession:
    def __init__(self, mode: str, a: ComparisonSide, b: ComparisonSide, overlap: int,
                 arch_version: int, edit_count: int, age: int):
        self.id = uuid.uuid4().hex[:12]
        self.state = "running"
        self.done = 0
        self.total = len(a.resolved) + len(b.resolved)
        self.message = ""
        self.mode = mode
        self.a = a
        self.b = b
        self.overlap_dropped = overlap
        self.architecture_version = arch_version
        self.edit_count = edit_count
        self.model_age = age
        self.stats: "OrderedDict[str, Tuple[SideStats, SideStats]]" = OrderedDict()
        self.axis: Dict[str, str] = {}             # node -> "channel" | "feature" | ""
        self.maps: Dict[str, th.Tensor] = {}       # pair: A's maps, until B's forward
        self.map_score: Dict[str, float] = {}      # pair: the whole-map distance
        self.cancel = threading.Event()
        self.created = time.time()

    def status(self, stale: bool) -> dict:
        return {
            "comparison_id": self.id, "state": self.state, "done": self.done,
            "total": self.total, "message": self.message, "mode": self.mode,
            "a": self.a.status(), "b": self.b.status(),
            "overlap_dropped": self.overlap_dropped,
            "architecture_version": self.architecture_version,
            "model_age": self.model_age, "stale": stale,
        }


class _CompareCapture:
    """Hook callbacks feeding one side of a comparison from a forward."""

    def __init__(self, session: ComparisonSession, side: int, batch: int):
        self.session = session
        self.side = side
        self.batch = batch
        self.shape_hints: Dict[Tuple[int, ...], int] = {}

    def _feed(self, node_id: str, tensor: th.Tensor, axis: Optional[int]) -> None:
        pooled = _pooled(tensor, axis, self.batch)
        if pooled is None:
            return
        values, absmax = pooled
        session = self.session
        pair = session.stats.get(node_id)
        if pair is None:
            pair = (SideStats(), SideStats())
            session.stats[node_id] = pair
            nd = tensor.ndim
            session.axis[node_id] = ("channel" if axis == 1 and nd >= 3 else
                                     "feature" if axis is not None and axis == nd - 1 else "")
        x = tensor.detach()
        pair[self.side].update(values, absmax, float(x.float().abs().sum()), int(x.numel()))
        if session.mode == "pair":
            m = _map_of(tensor, self.batch)
            if self.side == 0:
                if m is not None:
                    session.maps[node_id] = m
            else:
                ref = session.maps.pop(node_id, None)
                if ref is not None and m is not None and ref.shape == m.shape:
                    session.map_score[node_id] = _normalized_sq_distance(ref.numpy(), m.numpy())

    def on_pre(self, frame, args, kwargs):
        if frame.name != "":
            return
        tensor = _first_tensor((args, kwargs))
        if tensor is not None:
            self._feed(INPUT_NODE, tensor, _channel_axis(None, tensor, self.shape_hints))

    def on_post(self, frame, output):
        node_id = OUTPUT_NODE if frame.name == "" else (
            frame.name if frame.had_child else frame.node_id)
        tensor = _first_tensor(output)
        if tensor is None:
            return
        module = frame.module if frame.name else None
        axis = _channel_axis(module, tensor, self.shape_hints)
        if module is not None and isinstance(module, _CHANNEL_FIRST_TYPES + _FEATURE_LAST_TYPES) \
                and axis is not None:
            self.shape_hints[tuple(tensor.shape)] = axis
        self._feed(node_id, tensor, axis)


def _subset(ids: List[str], origins: List[str], cap: int) -> Tuple[List[str], List[str]]:
    """At most *cap* of them: an even, seeded random subset in their order
    (the first *cap* of a sorted tag would be a biased sample)."""
    if len(ids) <= cap:
        return ids, origins
    keep = sorted(random.Random(0).sample(range(len(ids)), cap))
    return [ids[i] for i in keep], [origins[i] if i < len(origins) else "" for i in keep]


class ComparisonManager:
    """Process-wide comparisons (the latest few), one running at a time."""

    MAX_SESSIONS = 4

    def __init__(self, representation: ModelRepresentationManager = MANAGER):
        self.representation = representation
        self._lock = threading.RLock()
        self._sessions: "OrderedDict[str, ComparisonSession]" = OrderedDict()
        self._threads: Dict[str, threading.Thread] = {}

    @staticmethod
    def _side_samples(side: dict) -> Tuple[List[str], List[str], str]:
        tag = str(side.get("tag") or "").strip()
        if tag:
            ids, origins = samples_with_tag(tag, str(side.get("origin") or ""))
        else:
            ids = [str(s) for s in side.get("sample_ids") or []]
            origins = [str(o) for o in side.get("origins") or []]
        return ids, origins, tag

    @staticmethod
    def _label(side: dict, tag: str, resolved) -> str:
        label = str(side.get("label") or "").strip()
        if label:
            return label
        if tag:
            return f"tag '{tag}'"
        if len(resolved) == 1:
            return f"sample {resolved[0][0]}"
        return f"{len(resolved)} samples"

    def start(self, model, loaders, a: dict, b: dict, max_samples: int = 0) -> dict:
        cap = int(max_samples) if max_samples and max_samples > 0 else DEFAULT_MAX_SAMPLES
        ids_a, org_a, tag_a = self._side_samples(a)
        ids_b, org_b, tag_b = self._side_samples(b)
        requested_a, requested_b = len(ids_a), len(ids_b)
        # The dataframe is indexed by sample id: one id is one sample.
        shared = set(ids_a) & set(ids_b)
        if shared:
            keep_a = [i for i, s in enumerate(ids_a) if s not in shared]
            keep_b = [i for i, s in enumerate(ids_b) if s not in shared]
            ids_a, org_a = [ids_a[i] for i in keep_a], [org_a[i] for i in keep_a if i < len(org_a)]
            ids_b, org_b = [ids_b[i] for i in keep_b], [org_b[i] for i in keep_b if i < len(org_b)]
        ids_a, org_a = _subset(ids_a, org_a, cap)
        ids_b, org_b = _subset(ids_b, org_b, cap)
        res_a = resolve_samples(loaders, ids_a, org_a)
        res_b = resolve_samples(loaders, ids_b, org_b)
        side_a = ComparisonSide(self._label(a, tag_a, res_a), tag_a, requested_a, res_a)
        side_b = ComparisonSide(self._label(b, tag_b, res_b), tag_b, requested_b, res_b)
        mode = "pair" if len(res_a) == 1 and len(res_b) == 1 else "group"
        try:
            age = int(model.get_age())
        except Exception:
            age = 0
        rep = self.representation
        with self._lock:
            for running in self._sessions.values():
                if running.state == "running":
                    running.cancel.set()
            session = ComparisonSession(mode, side_a, side_b, len(shared),
                                        rep.version, rep.edit_count, age)
            self._sessions[session.id] = session
            while len(self._sessions) > self.MAX_SESSIONS:
                self._sessions.popitem(last=False)
        error = ""
        if resolve_inner(model) is None:
            error = "No model registered"
        else:
            for name, side, requested in (("A", side_a, requested_a), ("B", side_b, requested_b)):
                if side.resolved:
                    continue
                if requested and shared and requested == len(shared):
                    error = f"every sample of {name} is also in the other side"
                elif requested:
                    error = f"none of {name}'s samples could be loaded"
                else:
                    error = f"{name} has no samples"
                break
        if error:
            session.state, session.message = "error", error
            return session.status(False)
        thread = threading.Thread(target=self._run, args=(session, model),
                                  name=f"wl-compare-{session.id}", daemon=True)
        self._threads[session.id] = thread
        thread.start()
        return session.status(False)

    def _forward(self, session: ComparisonSession, model, inner, entry, chunk, side: int) -> None:
        batch = build_batch(entry, [c[0] for c in chunk])
        inputs = _to_device(_inputs_from_batch(batch, inner), _model_device(inner))
        capture = _CompareCapture(session, side, len(chunk))
        with _inference_mode(model, inner), _CallTracker(inner, capture.on_pre, capture.on_post):
            _call_model(model, inputs)

    def _run(self, session: ComparisonSession, model) -> None:
        inner = resolve_inner(model)
        # One sample per forward in pair mode: each map is that sample's alone.
        chunk_size = 1 if session.mode == "pair" else max(
            1, int(os.environ.get("WEIGHTSLAB_MODELING_BATCH", "16") or 16))
        try:
            for side, resolved in enumerate((session.a.resolved, session.b.resolved)):
                groups: "OrderedDict[str, list]" = OrderedDict()
                for sid, origin, entry in resolved:
                    groups.setdefault(entry[0], []).append((sid, origin, entry))
                for items in groups.values():
                    entry = items[0][2]
                    for start in range(0, len(items), chunk_size):
                        if session.cancel.is_set():
                            raise _Cancelled()
                        chunk = items[start:start + chunk_size]
                        with _training_lock(cancel=session.cancel):
                            self._forward(session, model, inner, entry, chunk, side)
                        session.done += len(chunk)
            if session.cancel.is_set():
                raise _Cancelled()
            session.state = "done"
        except _Cancelled:
            session.state = "cancelled"
            session.message = "replaced by a newer comparison"
        except Exception as exc:
            logger.warning(f"[modeling] comparison failed: {exc}")
            logger.debug("[modeling] comparison failure", exc_info=True)
            session.state = "error"
            session.message = f"{type(exc).__name__}: {exc}"
        finally:
            session.maps.clear()
            self._threads.pop(session.id, None)

    # ---- accessors -----------------------------------------------------------
    def get(self, comparison_id: str = "") -> Optional[ComparisonSession]:
        with self._lock:
            if comparison_id:
                return self._sessions.get(comparison_id)
            for session in reversed(self._sessions.values()):
                if session.state != "cancelled":
                    return session
            return None

    def is_stale(self, session: ComparisonSession) -> bool:
        rep = self.representation
        return session.edit_count != rep.edit_count or session.architecture_version != rep.version

    def wait(self, comparison_id: str, timeout: float = 60.0) -> Optional[ComparisonSession]:
        thread = self._threads.get(comparison_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(comparison_id)

    def clear(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                if session.state == "running":
                    session.cancel.set()
            self._sessions.clear()

    def diffs(self, session: ComparisonSession) -> List[dict]:
        out = []
        for node_id, (a, b) in list(session.stats.items()):
            diff = node_diff(node_id, a, b, session.mode, session.map_score.get(node_id))
            if diff is not None:
                out.append(diff)
        return out

    def layer(self, comparison_id: str, node_id: str) -> dict:
        session = self.get(comparison_id)
        if session is None:
            return {"success": False, "message": "no comparison"}
        pair = session.stats.get(node_id)
        diff = node_diff(node_id, pair[0], pair[1], session.mode,
                         session.map_score.get(node_id)) if pair else None
        if diff is None:
            return {"success": False, "session": session,
                    "message": f"{node_id} has no activation to compare"
                    if session.state != "running" else "still running"}
        a, b = pair
        effect = neuron_effects(a, b, session.mode)
        ma, mb = a.mean(), b.mean()
        sa, sb = np.sqrt(a.var()), np.sqrt(b.var())
        ra, rb = a.rate(), b.rate()
        neurons = [{
            "index": i, "mean_a": float(ma[i]), "mean_b": float(mb[i]),
            "std_a": float(sa[i]), "std_b": float(sb[i]), "effect": float(effect[i]),
            "rate_a": float(ra[i]), "rate_b": float(rb[i]),
        } for i in range(len(effect))]
        return {"success": True, "message": "", "summary": diff, "neurons": neurons,
                "mode": session.mode, "neuron_axis": session.axis.get(node_id, ""),
                "session": session}


COMPARE = ComparisonManager()
