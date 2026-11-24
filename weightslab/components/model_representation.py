"""Model representation for the Weights Studio "Modeling" tab.

Three things live here, all model-agnostic (no ``compute_dependencies``, no
torch.fx -- which cannot trace most detection, LiDAR or Hugging Face models):

1. **The architecture graph.** Every module of the model becomes a node: a
   *group* (a module whose children run -- ``encoder``, ``down1``,
   ``layer3.0``) or a *layer* (a module that runs as one unit -- a conv, a
   norm, an ``nn.MultiheadAttention``). The edges between layers come from ONE
   real forward pass with tensor provenance: every tensor remembers which
   layer(s) produced it, through any functional op in between (``+``,
   ``torch.cat``, ``view``...), via a :class:`~torch.overrides.TorchFunctionMode`.
   So residual adds, U-Net skip concatenations and YOLO routing all show up as
   the edges they are, on any ``nn.Module``.

2. **Representation sessions.** Run the model on a set of samples (picked in
   the grid, or every sample of a tag) in a background thread, one small
   chunk at a time under the training lock, and keep, per node: running
   activation statistics, per-channel statistics (which neurons fire, which
   never do) and small input/output previews for the first few samples.

3. **Layer edits.** freeze / unfreeze / reset / perturb on any layer with
   weights; prune / add (and neuron-level freeze/reset/perturb) when the model
   was wrapped with ``compute_dependencies=True``, through the existing
   :class:`~weightslab.models.model_with_ops.NetworkWithOps` operations, which
   propagate the change to every dependent layer.

Node ids are ``named_modules()`` names. A module called more than once per
forward gets one node per call (``relu``, ``relu#2``), which keeps the graph a
DAG; edits always address the module (``module_id``).

Transformers. A residual add of two different branches becomes its own small
node (kind ``op``, ``<block>.add``, ``<block>.add#2``): the residual stream
reads input -> +attention -> +MLP -> ... instead of every later layer being
wired to every earlier one. Attention weights computed functionally inside an
attention module (``softmax`` over (B, heads, T, S), or
``scaled_dot_product_attention``) are captured as that module's "attention"
previews, one tile per head. The q / k / v projections' neurons are grouped
by head (``neuron_groups``), and token-id inputs are previewed as tokens.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from torch import nn
from torch.overrides import TorchFunctionMode

logger = logging.getLogger(__name__)

INPUT_NODE = "@input"
OUTPUT_NODE = "@output"

DEFAULT_MAX_SAMPLES = 256
PREVIEW_SAMPLES = 8
PREVIEW_CHANNELS = 16
PREVIEW_TILE = 32          # max side of a spatial (H x W) tile
PREVIEW_VECTOR = 512       # max length of a feature-vector tile
PREVIEW_SEQUENCE = 64      # max side of a (tokens x features) tile
PREVIEW_BUDGET_BYTES = 96 * 1024 * 1024
# A channel "fires" on a sample when its largest |value| exceeds ACTIVE_EPS
# times the layer's largest |value| on that sample. Relative, not absolute:
# an untrained network can carry its whole signal at 1e-6 scale (YOLO with
# untrained BatchNorm statistics) without a single neuron being dead, while a
# SiLU/GELU neuron stuck at -1e-8 next to channels at 1.0 is dead. Exact
# zeros (a dead ReLU) are silent at any scale.
ACTIVE_EPS = 1e-6
ACTIVE_FLOOR = 1e-30
_MAX_WALK = 256            # container items inspected when looking for tensors

EDIT_OPS = ("freeze", "unfreeze", "reset", "perturb", "prune", "add")

_LABEL_KEYS = {
    "label", "labels", "target", "targets", "y", "ids", "uid", "uids",
    "sample_id", "sample_ids", "index", "indices", "meta", "metadata",
}

_CHANNEL_FIRST_TYPES: Tuple[type, ...] = (
    nn.modules.conv._ConvNd,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.modules.pooling._MaxPoolNd,
    nn.modules.pooling._AvgPoolNd,
    nn.modules.pooling._AdaptiveMaxPoolNd,
    nn.modules.pooling._AdaptiveAvgPoolNd,
    nn.GroupNorm,
    nn.Upsample,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.PixelShuffle,
)
_FEATURE_LAST_TYPES: Tuple[type, ...] = (
    nn.Linear,
    nn.Bilinear,
    nn.Embedding,
    nn.LayerNorm,
    nn.MultiheadAttention,
    nn.modules.rnn.RNNBase,
)


# =============================================================================
# Small helpers
# =============================================================================
def resolve_inner(model) -> Optional[nn.Module]:
    """The user's ``nn.Module`` behind a ledger Proxy / ModelInterface."""
    from weightslab.projection.parametric_umap import resolve_module
    return resolve_module(model)


def _iter_tensors(obj, _depth: int = 0) -> Iterator[th.Tensor]:
    """Tensors inside *obj* (tensor, list/tuple/dict nests, ModelOutput)."""
    if isinstance(obj, th.Tensor):
        yield obj
        return
    if _depth > 4 or obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return
    if isinstance(obj, Mapping):
        for i, value in enumerate(obj.values()):
            if i >= _MAX_WALK:
                break
            yield from _iter_tensors(value, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            if i >= _MAX_WALK:
                break
            yield from _iter_tensors(value, _depth + 1)


def _first_tensor(obj) -> Optional[th.Tensor]:
    for tensor in _iter_tensors(obj):
        return tensor
    return None


def _shape(obj) -> List[int]:
    tensor = _first_tensor(obj)
    return [int(s) for s in tensor.shape] if tensor is not None else []


def _to_device(obj, device):
    if isinstance(obj, th.Tensor):
        return obj.to(device)
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(v, device) for v in obj)
    return obj


def _model_device(inner: nn.Module):
    try:
        return next(inner.parameters()).device
    except StopIteration:
        try:
            return next(inner.buffers()).device
        except StopIteration:
            return th.device("cpu")


def _is_internal(module: nn.Module) -> bool:
    """WeightsLab's own bookkeeping submodules (the neuron trigger trackers a
    patched layer carries) -- never part of the user's architecture."""
    return type(module).__module__ == "weightslab.components.tracking"


def _has_children(module: nn.Module) -> bool:
    return any(not _is_internal(child) for child in module.children())


def _tokens(name: str) -> List[str]:
    return re.findall(r"[a-z]+", name.lower())


def _count_params(module: nn.Module, recurse: bool = True) -> Tuple[int, int]:
    total = trainable = 0
    for param in module.parameters(recurse=recurse):
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def _extra_repr(module: nn.Module) -> str:
    try:
        return str(module.extra_repr())[:200]
    except Exception:
        return ""


def _module_type(module: nn.Module) -> str:
    # Patched modules are renamed "Conv2dWithNeuronOps"; show the torch name.
    name = getattr(module, "module_name", None) or module.__class__.__name__
    return str(name).replace("WithNeuronOps", "")


# ---- user override of "batch -> model inputs" ------------------------------
_INPUTS_FN: Optional[Callable] = None


def set_modeling_inputs_fn(fn: Optional[Callable]) -> None:
    """Tell the Modeling tab how to feed a collated batch to the model.

    ``fn(batch)`` returns what the model is called with: a tensor/list (called
    as ``model(x)``), a dict (``model(**x)``), or an ``(args, kwargs)`` pair.
    By default the first element of a tuple batch is the input (WeightsLab's
    ``(inputs, ids, labels, ...)`` convention), and a dict batch is passed as
    keyword arguments minus its label/id keys. A model can also define
    ``wl_modeling_inputs(self, batch)`` itself.
    """
    global _INPUTS_FN
    _INPUTS_FN = fn


def _inputs_from_batch(batch, inner: Optional[nn.Module]):
    fn = _INPUTS_FN or getattr(inner, "wl_modeling_inputs", None)
    if callable(fn):
        return fn(batch)
    if isinstance(batch, (tuple, list)):
        return batch[0] if batch else batch
    if isinstance(batch, Mapping):
        return {k: v for k, v in batch.items() if str(k).lower() not in _LABEL_KEYS}
    return batch


def _call_model(model, inputs):
    if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], dict) \
            and isinstance(inputs[0], (tuple, list)):
        return model(*inputs[0], **inputs[1])
    if isinstance(inputs, Mapping):
        try:
            return model(**inputs)
        except TypeError:
            return model(inputs)
    return model(inputs)


def _batch_size_of(inputs) -> int:
    if isinstance(inputs, (list, tuple)) and inputs and not isinstance(inputs[0], th.Tensor):
        return len(inputs)
    if isinstance(inputs, (list, tuple)) and inputs and all(isinstance(t, th.Tensor) for t in inputs) \
            and len({tuple(t.shape) for t in inputs}) > 1:
        # A list of variable-size tensors (point clouds): one per sample.
        return len(inputs)
    tensor = _first_tensor(inputs)
    if tensor is None or tensor.ndim == 0:
        return 1
    return int(tensor.shape[0])


@contextmanager
def _inference_mode(model, inner: nn.Module):
    """eval() + no_grad + frozen age; the previous train/eval mode comes back."""
    from weightslab.projection.parametric_umap import _frozen_age
    was_training = inner.training
    try:
        inner.eval()
        with th.no_grad(), _frozen_age(model):
            yield
    finally:
        try:
            inner.train(was_training)
        except Exception:
            pass


@contextmanager
def _training_lock(timeout_s: float = 60.0, cancel: Optional[threading.Event] = None):
    """The global training lock, so a forward never interleaves with a step."""
    from weightslab.components.global_monitoring import weightslab_rlock
    deadline = time.monotonic() + max(timeout_s, 1.0)
    acquired = False
    while not acquired:
        acquired = weightslab_rlock.acquire(timeout=1.0)
        if acquired:
            break
        if cancel is not None and cancel.is_set():
            raise _Cancelled()
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"training lock not acquired within {timeout_s:.0f}s "
                "(a training or evaluation step is holding it)")
    try:
        yield
    finally:
        weightslab_rlock.release()


class _Cancelled(Exception):
    pass


# =============================================================================
# Channel axis + previews
# =============================================================================
def _channel_axis(module: Optional[nn.Module], tensor: th.Tensor,
                  shape_hints: Optional[Dict[Tuple[int, ...], int]] = None) -> Optional[int]:
    """Which axis of *tensor* (batch first) holds the neurons, or None."""
    nd = tensor.ndim
    if nd < 2:
        return None
    if module is not None and isinstance(module, _FEATURE_LAST_TYPES):
        return nd - 1
    if module is not None and isinstance(module, _CHANNEL_FIRST_TYPES):
        return 1
    if nd >= 4 or nd == 2:
        return 1
    if shape_hints:
        hint = shape_hints.get(tuple(tensor.shape))
        if hint is not None:
            return hint
    return nd - 1


def _resize_2d(x: th.Tensor, max_h: int, max_w: int) -> th.Tensor:
    """(N, H, W) -> (N, h, w) with h <= max_h, w <= max_w, aspect kept."""
    _, h, w = x.shape
    scale = min(1.0, max_h / max(h, 1), max_w / max(w, 1))
    th_, tw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    if (th_, tw) == (h, w):
        return x
    return th.nn.functional.adaptive_avg_pool2d(x.unsqueeze(0), (th_, tw)).squeeze(0)


def _image_view(x: th.Tensor) -> Optional[th.Tensor]:
    """(C, H, W) with C in 1 / 3 / 4 when *x* looks like an image, else None.

    (C, H, W) and (H, W) qualify when both sides are >= 8 px; a clip or a
    volume (C, T, H, W) shows its middle frame / slice. Point clouds (N, 4),
    token ids and feature vectors never qualify.
    """
    if x.ndim == 4 and x.shape[0] in (1, 3, 4):
        x = x[:, x.shape[1] // 2]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[0] not in (1, 3, 4):
        return None
    if min(int(x.shape[1]), int(x.shape[2])) < 8:
        return None
    return x


def _image_preview(img: th.Tensor, shape: List[int], tile: int) -> dict:
    """The model input as a picture: RGB / RGBA / gray, not one heatmap per
    channel. Values already in [0, 1] or [0, 255] keep their colours; anything
    else (a normalised input) is stretched over the WHOLE image -- one range
    for all channels, so the colour balance survives. tile_min / tile_max hold
    that range: value = min + byte / 255 * (max - min)."""
    img = _resize_2d(img, tile, tile)
    lo, hi = float(img.min()), float(img.max())
    if lo >= 0.0 and hi <= 1.0:
        lo, hi = 0.0, 1.0
    elif lo >= 0.0 and hi <= 255.0:
        lo, hi = 0.0, 255.0
    span = max(hi - lo, 1e-12)
    scaled = ((img - lo) / span * 255.0).round().clamp(0, 255).to(th.uint8)
    return {
        "shape": shape,
        "tiles": 1,
        "height": int(img.shape[1]),
        "width": int(img.shape[2]),
        "data": scaled.contiguous().numpy().tobytes(),   # planar: C x H x W
        "tile_min": [lo],
        "tile_max": [hi],
        "channel_indices": list(range(int(img.shape[0]))),
        "layout": "image",
    }


TOKEN_PREVIEW_MAX = 512


def _token_preview(sample: th.Tensor) -> dict:
    """A sequence of token ids (a language model's input): the ids themselves
    (layout "tokens"), plus a one-row strip of them for a uniform payload."""
    ids = [int(v) for v in sample.detach().reshape(-1)[:TOKEN_PREVIEW_MAX].cpu().tolist()]
    x = th.tensor(ids or [0], dtype=th.float32)
    lo, hi = float(x.min()), float(x.max())
    data = ((x - lo) / max(hi - lo, 1e-12) * 255.0).round().clamp(0, 255).to(th.uint8)
    return {
        "shape": [int(s) for s in sample.shape], "tiles": 1, "height": 1, "width": int(x.numel()),
        "data": data.numpy().tobytes(), "tile_min": [lo], "tile_max": [hi],
        "channel_indices": [], "layout": "tokens", "token_ids": ids,
    }


def _token_text(inner: Optional[nn.Module], ids: Sequence[int]) -> List[str]:
    """The text of each token id when the model can tell (a ``decode_tokens``
    method, or an ``itos`` table as in nanoGPT's char-level ``meta.pkl``)."""
    if inner is None or not ids:
        return []
    decode = getattr(inner, "decode_tokens", None)
    try:
        if callable(decode):
            out = decode(list(ids))
            if isinstance(out, (list, tuple)) and len(out) == len(ids):
                return [str(t) for t in out]
        itos = getattr(inner, "itos", None)
        if isinstance(itos, Mapping):
            return [str(itos.get(int(i), "")) for i in ids]
        if isinstance(itos, (list, tuple)):
            return [str(itos[int(i)]) if 0 <= int(i) < len(itos) else "" for i in ids]
    except Exception:
        return []
    return []


def make_preview(sample: th.Tensor, channel_axis: Optional[int], max_channels: int,
                 tile: int = PREVIEW_TILE, as_image: bool = False) -> Optional[dict]:
    """A uint8 picture of one sample's tensor (batch dim already removed).

    ``channel_axis`` is in the sample's own coordinates: 0 = channel-first
    (conv maps: one tile per channel), -1 = feature-last (one tokens x features
    tile), None = no neuron axis (one vector/flattened tile). ``as_image``
    (the model input): an image-shaped tensor becomes ONE natural picture
    (layout "image", see _image_preview).
    """
    if not sample.is_floating_point() and not sample.is_complex() and sample.ndim == 1 \
            and sample.dtype != th.bool:
        return _token_preview(sample)
    try:
        x = sample.detach().float().cpu()
    except Exception:
        return None
    if not th.isfinite(x).all():
        x = th.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    shape = [int(s) for s in x.shape]
    if as_image:
        img = _image_view(x)
        if img is not None:
            return _image_preview(img, shape, tile)
    channel_indices: List[int] = []
    if x.ndim == 0:
        x = x.reshape(1)

    if channel_axis == 0 and x.ndim >= 2:
        count = min(int(x.shape[0]), max(1, max_channels))
        channel_indices = list(range(count))
        tiles = x[:count]
        while tiles.ndim > 3:              # (C, D, H, W) -> max over depth
            tiles = tiles.amax(dim=1)
        if tiles.ndim == 2:                # (C, L): 1-D maps
            tiles = tiles.unsqueeze(1)
            tiles = _resize_2d(tiles, 1, max(128, tile * 4))
        else:
            tiles = _resize_2d(tiles, tile, tile)
        layout = "channels"
    elif x.ndim == 1:
        tiles = x.reshape(1, 1, -1)
        tiles = _resize_2d(tiles, 1, max(PREVIEW_VECTOR, tile * 16))
        layout = "vector"
    else:
        tiles = x.reshape(-1, x.shape[-1]).unsqueeze(0)   # tokens x features
        side = max(PREVIEW_SEQUENCE, tile * 2)
        tiles = _resize_2d(tiles, side, side)
        layout = "sequence"

    flat = tiles.reshape(tiles.shape[0], -1)
    mins = flat.amin(dim=1)
    maxs = flat.amax(dim=1)
    span = (maxs - mins).clamp_min(1e-12).unsqueeze(1)
    scaled = ((flat - mins.unsqueeze(1)) / span * 255.0).round().clamp(0, 255).to(th.uint8)
    return {
        "shape": shape,
        "tiles": int(tiles.shape[0]),
        "height": int(tiles.shape[1]),
        "width": int(tiles.shape[2]),
        "data": scaled.numpy().tobytes(),
        "tile_min": [float(v) for v in mins.tolist()],
        "tile_max": [float(v) for v in maxs.tolist()],
        "channel_indices": channel_indices,
        "layout": layout,
    }


# =============================================================================
# Statistics accumulator (one per node per session)
# =============================================================================
def _firing(absmax: th.Tensor) -> th.Tensor:
    """Which channels fire, from their largest |value| (last dim = channels;
    a leading dim, if any, is samples). See ACTIVE_EPS."""
    scale = absmax.amax(dim=-1, keepdim=True)
    return absmax > th.clamp(scale * ACTIVE_EPS, min=ACTIVE_FLOOR)


class NodeAccumulator:
    """Running activation statistics of one node over every sample seen."""

    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.abssum = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.inactive = 0
        self.channels: Optional[int] = None
        self.ch_sum: Optional[np.ndarray] = None
        self.ch_abs: Optional[np.ndarray] = None
        self.ch_values = 0                     # values per channel seen
        self.ch_max: Optional[np.ndarray] = None
        self.ch_active: Optional[np.ndarray] = None
        self.ch_obs = 0                        # observations (samples)

    def update(self, tensor: th.Tensor, axis: Optional[int], batch: int) -> None:
        if not tensor.is_floating_point() and not tensor.is_complex():
            x = tensor.detach().float()
        else:
            x = tensor.detach().float()
        if x.numel() == 0:
            return
        x = th.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.count += x.numel()
        self.sum += float(x.sum())
        self.sumsq += float((x * x).sum())
        self.abssum += float(x.abs().sum())
        self.min = min(self.min, float(x.min()))
        self.max = max(self.max, float(x.max()))
        self.inactive += int((x <= 0).sum())

        if axis is None or x.ndim < 2:
            return
        channels = int(x.shape[axis])
        if self.channels is None:
            self.channels = channels
            self.ch_sum = np.zeros(channels, dtype=np.float64)
            self.ch_abs = np.zeros(channels, dtype=np.float64)
            self.ch_max = np.full(channels, -np.inf, dtype=np.float64)
            self.ch_active = np.zeros(channels, dtype=np.int64)
        elif channels != self.channels:
            return
        # A channel "fires" on a sample when it is non-zero anywhere on it
        # (relative to the layer's scale, see ACTIVE_EPS).
        # Non-zero, not positive: a conv or linear output is signed, and a
        # channel that is always negative still carries information -- it is
        # the ReLU after it that kills it, and that ReLU's own output (exactly
        # zero) is where the dead neuron shows (see layer_details).
        if x.shape[0] == batch and axis != 0:
            xc = x.movedim(axis, 1).reshape(batch, channels, -1)       # (B, C, P)
            per_sample_absmax = xc.abs().amax(dim=2)                     # (B, C)
            self.ch_sum += xc.sum(dim=(0, 2)).double().cpu().numpy()
            self.ch_abs += xc.abs().sum(dim=(0, 2)).double().cpu().numpy()
            self.ch_values += batch * xc.shape[2]
            self.ch_max = np.maximum(self.ch_max, xc.amax(dim=(0, 2)).double().cpu().numpy())
            self.ch_active += _firing(per_sample_absmax).sum(dim=0).cpu().numpy()
            self.ch_obs += batch
        else:
            xc = x.movedim(axis, 0).reshape(channels, -1)
            self.ch_sum += xc.sum(dim=1).double().cpu().numpy()
            self.ch_abs += xc.abs().sum(dim=1).double().cpu().numpy()
            self.ch_values += xc.shape[1]
            self.ch_max = np.maximum(self.ch_max, xc.amax(dim=1).double().cpu().numpy())
            self.ch_active += _firing(xc.abs().amax(dim=1)).cpu().numpy().astype(np.int64)
            self.ch_obs += 1

    def summary(self, node_id: str) -> dict:
        if self.count == 0:
            return {"node_id": node_id}
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        dead = int((self.ch_active == 0).sum()) if self.ch_active is not None and self.ch_obs else 0
        return {
            "node_id": node_id,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min,
            "max": self.max,
            "inactive_fraction": self.inactive / self.count,
            "channels": int(self.channels or 0),
            "dead_channels": dead,
            "mean_abs": self.abssum / self.count,
        }

    def channel_stats(self) -> Optional[dict]:
        if self.channels is None or not self.ch_obs:
            return None
        values = max(self.ch_values, 1)
        return {
            "mean": (self.ch_sum / values).tolist(),
            "mean_abs": (self.ch_abs / values).tolist(),
            "max": self.ch_max.tolist(),
            "active_rate": (self.ch_active / self.ch_obs).tolist(),
        }


# =============================================================================
# Hooking: the per-call machinery shared by tracing and capture
# =============================================================================
class _Frame:
    __slots__ = ("module", "name", "node_id", "had_child", "sources", "in_shape", "in_tensor")

    def __init__(self, module, name, node_id):
        self.module = module
        self.name = name
        self.node_id = node_id
        self.had_child = False
        self.sources: Dict[str, List[int]] = {}
        self.in_shape: List[int] = []
        self.in_tensor = None


class _Provenance:
    """Which node(s) produced each tensor alive during one traced forward."""

    def __init__(self):
        self._map: Dict[int, frozenset] = {}
        self._keep: List[th.Tensor] = []   # keeps ids from being reused

    def mark(self, obj, sources: frozenset) -> None:
        for tensor in _iter_tensors(obj):
            self._map[id(tensor)] = sources
            self._keep.append(tensor)

    def sources(self, obj) -> Dict[str, List[int]]:
        found: Dict[str, List[int]] = {}
        for tensor in _iter_tensors(obj):
            srcs = self._map.get(id(tensor))
            if srcs:
                for src in srcs:
                    found.setdefault(src, [int(s) for s in tensor.shape])
        return found


_ADD_NAMES = frozenset({"add", "add_", "__add__", "__radd__", "__iadd__"})


class _ProvenanceMode(TorchFunctionMode):
    """Propagates provenance through every torch function (add, cat, view...).

    ``on_merge``: an add of two tensors coming from different nodes (a
    residual / skip connection) is handed to it; it names a merge node, and
    the sum's provenance becomes that node alone -- so a transformer's
    residual stream is a chain of "+" nodes, not every later layer wired to
    every earlier one.
    """

    def __init__(self, provenance: _Provenance, on_merge: Optional[Callable] = None):
        super().__init__()
        self.provenance = provenance
        self.on_merge = on_merge

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            if self.on_merge is not None and len(args) >= 2 \
                    and getattr(func, "__name__", "") in _ADD_NAMES \
                    and isinstance(args[0], th.Tensor) and isinstance(args[1], th.Tensor):
                a = self.provenance._map.get(id(args[0]))
                b = self.provenance._map.get(id(args[1]))
                if a and b and a != b:
                    merged = self.on_merge(a | b, out)
                    if merged:
                        self.provenance.mark(out, frozenset({merged}))
                        return out
            srcs = set()
            for tensor in _iter_tensors((args, kwargs)):
                found = self.provenance._map.get(id(tensor))
                if found:
                    srcs |= found
            if srcs:
                self.provenance.mark(out, frozenset(srcs))
        except Exception:
            pass
        return out


class _CallTracker:
    """Pre/post hooks on every module; turns module calls into node ids.

    A call is a *leaf* call when no child module ran inside it -- which is
    what makes ``nn.MultiheadAttention`` (whose ``out_proj`` never runs as a
    module) a layer and ``nn.Sequential`` a group, without a type list.
    """

    def __init__(self, root: nn.Module, on_pre=None, on_post=None):
        self.root = root
        self.names: Dict[int, str] = {}
        for name, module in root.named_modules():
            self.names.setdefault(id(module), name)
        # A module registered under several paths (Ultralytics' Conv blocks
        # all hold ONE shared SiLU as `.act`) is named, per call, by the path
        # under the module calling it: model.6.cv1.act, not model.0.act#15.
        self.paths: Dict[int, List[str]] = {}
        for name, module in _all_named_modules(root):
            self.paths.setdefault(id(module), []).append(name)
        self.paths = {k: v for k, v in self.paths.items() if len(v) > 1}
        self.stack: List[_Frame] = []
        self.counts: Counter = Counter()
        self.on_pre = on_pre
        self.on_post = on_post
        self.handles = []

    def _node_id(self, name: str) -> str:
        k = self.counts[name]
        self.counts[name] += 1
        if name == "":
            return ""
        return name if k == 0 else f"{name}#{k + 1}"

    def __enter__(self):
        for module in self.root.modules():
            if _is_internal(module):
                continue
            try:
                self.handles.append(module.register_forward_pre_hook(self._pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(self._post, with_kwargs=True))
            except Exception as exc:
                logger.debug(f"[modeling] could not hook {type(module).__name__}: {exc}")
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles = []
        return False

    def reset(self) -> None:
        self.stack = []
        self.counts = Counter()

    def _pre(self, module, args, kwargs):
        try:
            name = self.names.get(id(module))
            if name is None:
                return None
            if name == "" and not self.stack:
                self.counts = Counter()
            if self.stack:
                self.stack[-1].had_child = True
                aliases = self.paths.get(id(module))
                if aliases:
                    caller = self.stack[-1].name
                    name = next((p for p in aliases if _parent_name(p) == caller), name)
            frame = _Frame(module, name, self._node_id(name))
            frame.in_shape = _shape((args, kwargs))
            self.stack.append(frame)
            if self.on_pre is not None:
                self.on_pre(frame, args, kwargs)
        except Exception as exc:
            logger.debug(f"[modeling] pre-hook failed: {exc}")
        return None

    def _post(self, module, args, kwargs, output):
        try:
            frame = None
            while self.stack:
                candidate = self.stack.pop()
                if candidate.module is module:
                    frame = candidate
                    break
            if frame is None:
                return None
            if self.on_post is not None:
                self.on_post(frame, output)
        except Exception as exc:
            logger.debug(f"[modeling] post-hook failed: {exc}")
        return None


def _parent_name(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _all_named_modules(root: nn.Module):
    """``named_modules`` without de-duplication: a module shared by several
    parents is listed under each of its paths (each can be its own node)."""
    seen = set()
    for name, module in root.named_modules(remove_duplicate=False):
        if name in seen:
            continue
        seen.add(name)
        yield name, module


# =============================================================================
# 1. Architecture trace
# =============================================================================
class ArchitectureTrace:
    def __init__(self):
        self.calls: "OrderedDict[str, dict]" = OrderedDict()   # node id -> call record
        self.containers: Dict[str, dict] = {}                   # module name -> first call
        self.merges: "OrderedDict[str, dict]" = OrderedDict()  # "+" node id -> record
        self.edges: Dict[Tuple[str, str], List[int]] = {}
        self.input_shape: List[int] = []
        self.output_shape: List[int] = []
        self.traced = False
        self.message = ""


def trace_architecture(model, inputs) -> ArchitectureTrace:
    """One forward of *inputs* with provenance; see the module docstring."""
    inner = resolve_inner(model)
    trace = ArchitectureTrace()
    if inner is None:
        trace.message = "no nn.Module found on the registered model"
        return trace
    provenance = _Provenance()
    order = [0]

    def on_pre(frame: _Frame, args, kwargs):
        if frame.name == "":
            return
        frame.sources = provenance.sources((args, kwargs))

    def on_post(frame: _Frame, output):
        if frame.name == "":
            return
        record = {
            "node_id": frame.node_id,
            "module_name": frame.name,
            "in_shape": frame.in_shape,
            "out_shape": _shape(output),
        }
        if frame.had_child:
            trace.containers.setdefault(frame.name, record)
            return
        record["order"] = order[0]
        order[0] += 1
        trace.calls[frame.node_id] = record
        for src, shape in frame.sources.items():
            if src != frame.node_id:
                trace.edges[(src, frame.node_id)] = shape
        provenance.mark(output, frozenset({frame.node_id}))

    tracker = _CallTracker(inner, on_pre, on_post)
    merge_counts: Counter = Counter()

    def on_merge(sources: frozenset, out) -> str:
        frame = tracker.stack[-1] if tracker.stack else None
        if frame is None:
            return ""
        # Inside a module with no submodule, the module's own output stands
        # for whatever it adds up.
        if frame.name and not _has_children(frame.module):
            return ""
        scope = frame.name
        merge_counts[scope] += 1
        k = merge_counts[scope]
        base = f"{scope}.add" if scope else "add"
        node_id = base if k == 1 else f"{base}#{k}"
        shape = _shape(out)
        trace.merges[node_id] = {"node_id": node_id, "scope": scope, "order": order[0],
                                 "out_shape": shape, "call_index": k}
        order[0] += 1
        for src in sources:
            if src != node_id:
                trace.edges[(src, node_id)] = shape
        return node_id

    device = _model_device(inner)
    inputs = _to_device(inputs, device)
    provenance.mark(inputs, frozenset({INPUT_NODE}))
    trace.input_shape = _shape(inputs)
    try:
        with _inference_mode(model, inner), tracker, _ProvenanceMode(provenance, on_merge):
            output = _call_model(model, inputs)
        trace.output_shape = _shape(output)
        for src, shape in provenance.sources(output).items():
            trace.edges[(src, OUTPUT_NODE)] = shape
        # A merge whose sum went nowhere happened inside a module that then
        # ran as one unit (its output stands for it): drop it.
        while True:
            used = {src for (src, _dst) in trace.edges}
            dead = [m for m in trace.merges if m not in used]
            if not dead:
                break
            for m in dead:
                del trace.merges[m]
                for key in [k for k in trace.edges if k[1] == m]:
                    del trace.edges[key]
        trace.traced = bool(trace.calls)
        if not trace.traced:
            trace.message = "the forward pass ran no module"
    except Exception as exc:
        logger.warning(f"[modeling] architecture trace failed: {exc}")
        logger.debug("[modeling] trace failure", exc_info=True)
        trace.message = f"forward failed: {type(exc).__name__}: {exc}"
        trace.calls.clear()
        trace.edges.clear()
    return trace


# ---- roles ------------------------------------------------------------------
_ROLE_TOKENS = (
    ("attention", {"attn", "attention", "mha", "selfattention", "crossattention"}),
    ("mlp", {"mlp", "ffn", "feedforward", "ff"}),
    ("backbone", {"transformer", "trunk"}),
    ("embedding", {"embed", "embeds", "embedding", "embeddings", "wte", "wpe", "emb"}),
    ("bottleneck", {"bottleneck", "bridge", "middle", "mid", "latent", "center", "bottom"}),
    ("neck", {"neck", "fpn", "pan"}),
    ("decoder", {"decoder", "decoders", "dec", "up", "ups", "expand", "expanding", "upsampling"}),
    ("encoder", {"encoder", "encoders", "enc", "down", "downs", "contract", "contracting",
                 "backbone", "features", "stem", "downsampling"}),
    ("head", {"head", "heads", "classifier", "cls", "fc", "out", "outc", "output", "outputs",
              "lm", "pooler", "detect", "predictor", "logits"}),
)


def guess_role(name: str) -> str:
    tokens = set(_tokens(name.rsplit(".", 1)[-1]))
    for role, words in _ROLE_TOKENS:
        if tokens & words:
            return role
    return "block"


# Roles that hold at any depth (a transformer's attention / MLP sit 3-4
# levels down): from the module's class, else its name.
_DEEP_ROLES = ("attention", "mlp")


def _deep_role(name: str, module: nn.Module) -> str:
    kind = type(module).__name__.lower()
    if "attention" in kind or kind.endswith("attn"):
        return "attention"
    if "mlp" in kind or "feedforward" in kind or kind.endswith("ffn"):
        return "mlp"
    role = guess_role(name)
    return role if role in _DEEP_ROLES else ""


def _declared_roles(inner: nn.Module) -> Dict[str, str]:
    """Roles a model declares itself, where names say nothing.

    Ultralytics models (YOLO detect / segment / pose / ...) are one
    ``model`` Sequential of numbered blocks; their ``yaml`` lists which are
    the ``backbone`` and which the ``head``. Backbone blocks -> "backbone",
    the last block (Detect, Segment, ...) -> "head", the head blocks before
    it (the FPN/PAN) -> "neck". Looked for on the model and one level down
    (a ``YOLO`` wrapper holds the DetectionModel as ``.model``).
    """
    candidates = [("", inner)]
    child = getattr(inner, "model", None)
    if isinstance(child, nn.Module):
        candidates.append(("model", child))
    for prefix, module in candidates:
        cfg = getattr(module, "yaml", None)
        seq = getattr(module, "model", None)
        if not isinstance(cfg, Mapping) or not isinstance(seq, nn.Sequential):
            continue
        backbone, head = cfg.get("backbone"), cfg.get("head")
        if not isinstance(backbone, (list, tuple)) or not isinstance(head, (list, tuple)):
            continue
        n_back, n = len(backbone), len(seq)
        if n != n_back + len(head) or n < 2:
            continue
        base = f"{prefix}.model" if prefix else "model"
        return {
            f"{base}.{i}": "backbone" if i < n_back else ("head" if i == n - 1 else "neck")
            for i in range(n)
        }
    return {}


# =============================================================================
# Architecture graph (JSON-able dicts; the service turns them into protos)
# =============================================================================
def _is_tracked(model, module: nn.Module) -> Tuple[bool, int]:
    getter = getattr(module, "get_module_id", None)
    if not callable(getter):
        return False, -1
    try:
        layer_id = int(getter())
    except Exception:
        return False, -1
    try:
        dep = getattr(model, "_dep_manager", None)
        registered = dep.id_2_layer.get(layer_id) if dep is not None else None
    except Exception:
        registered = None
    return registered is module, layer_id


def _weight_out_axis(module: nn.Module) -> int:
    if isinstance(module, nn.modules.conv._ConvTransposeNd) or isinstance(module, nn.Embedding):
        return 1
    return 0


def neuron_weight_stats(module: nn.Module) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(L2 norm, mean |w|) per output neuron, or None when there is no weight."""
    weight = getattr(module, "weight", None)
    if not isinstance(weight, th.Tensor) or weight.ndim == 0:
        return None
    with th.no_grad():
        w = weight.detach().float()
        if w.ndim == 1:
            return w.abs().cpu().numpy(), w.abs().cpu().numpy()
        axis = _weight_out_axis(module)
        wm = w.movedim(axis, 0).reshape(w.shape[axis], -1)
        return wm.norm(dim=1).cpu().numpy(), wm.abs().mean(dim=1).cpu().numpy()


class FrozenRows:
    """Some output neurons of a layer kept fixed, on any layer (no
    dependencies needed): a forward pre-hook puts their weights back before
    every forward, so neither gradients nor momentum / weight decay can move
    them. Picklable (checkpoints dump the model with its hooks)."""

    ATTR = "_wl_frozen_rows"

    def __init__(self):
        self.rows: Dict[str, Tuple[int, List[int], th.Tensor]] = {}   # param -> (axis, idx, values)

    @staticmethod
    def of(module: nn.Module, create: bool = False) -> Optional["FrozenRows"]:
        hook = getattr(module, FrozenRows.ATTR, None)
        if hook is None and create:
            hook = FrozenRows()
            object.__setattr__(module, FrozenRows.ATTR, hook)
            module.register_forward_pre_hook(hook)
        return hook

    def indices(self) -> set:
        entry = self.rows.get("weight")
        return set(entry[1]) if entry else set()

    def restore(self, module: nn.Module) -> None:
        with th.no_grad():
            for pname, (axis, idx, values) in self.rows.items():
                param = getattr(module, pname, None)
                if isinstance(param, th.Tensor) and idx and max(idx) < param.shape[axis]:
                    index = th.as_tensor(idx, device=param.device)
                    param.data.index_copy_(axis, index, values.to(param.device, param.dtype))

    def __call__(self, module, args):
        if self.rows:
            self.restore(module)
        return None

    def set(self, module: nn.Module, indices: Sequence[int]) -> None:
        """Freeze *indices* too (already frozen ones keep their values)."""
        self.restore(module)
        keep = sorted(self.indices() | {int(i) for i in indices})
        self._capture(module, keep)

    def unset(self, module: nn.Module, indices: Optional[Sequence[int]]) -> None:
        self.restore(module)
        keep = [] if indices is None else sorted(self.indices() - {int(i) for i in indices})
        self._capture(module, keep)

    def recapture(self, module: nn.Module) -> None:
        """After an explicit edit of frozen rows (reset / perturb): keep the new values."""
        self._capture(module, sorted(self.indices()))

    def _capture(self, module: nn.Module, keep: List[int]) -> None:
        self.rows = {}
        if not keep:
            return
        weight = getattr(module, "weight", None)
        if isinstance(weight, th.Tensor):
            axis = _weight_out_axis(module) if weight.ndim > 1 else 0
            self.rows["weight"] = (axis, keep, weight.detach().index_select(axis, th.as_tensor(
                keep, device=weight.device)).clone().cpu())
        bias = getattr(module, "bias", None)
        if isinstance(bias, th.Tensor) and bias.ndim == 1:
            self.rows["bias"] = (0, keep, bias.detach().index_select(0, th.as_tensor(
                keep, device=bias.device)).clone().cpu())


def _frozen_neurons(module: nn.Module, tracked: bool) -> Tuple[bool, int, set]:
    """(whole layer frozen, frozen neuron count, frozen indices)."""
    params = list(module.parameters(recurse=False))
    whole = bool(params) and not any(p.requires_grad for p in params)
    indices = set()
    if tracked:
        lrs = (getattr(module, "neuron_2_lr", {}) or {}).get("weight", {})
        indices = {int(i) for i, lr in dict(lrs).items() if float(lr) == 0.0}
    hook = FrozenRows.of(module)
    if hook is not None:
        indices |= hook.indices()
    return whole, len(indices), indices


def _layer_neurons(module: nn.Module, tracked: bool, out_shape: List[int]) -> Tuple[int, int]:
    """(input neurons, output neurons) -- 0 when unknown."""
    if tracked:
        getter = getattr(module, "get_neurons", None)
        try:
            n_in = getter("in_neurons") if callable(getter) else getattr(module, "in_neurons", 0)
            n_out = getter("out_neurons") if callable(getter) else getattr(module, "out_neurons", 0)
            return int(n_in or 0), int(n_out or 0)
        except Exception:
            pass
    for a_in, a_out in (("in_channels", "out_channels"), ("in_features", "out_features"),
                        ("num_embeddings", "embedding_dim")):
        if hasattr(module, a_out):
            try:
                return int(getattr(module, a_in, 0) or 0), int(getattr(module, a_out) or 0)
            except Exception:
                pass
    if hasattr(module, "num_features"):
        n = int(getattr(module, "num_features") or 0)
        return n, n
    stats = neuron_weight_stats(module)
    if stats is not None:
        return 0, int(len(stats[0]))
    return 0, 0


def supported_ops(module: nn.Module, tracked: bool) -> List[str]:
    params = list(module.parameters(recurse=False))
    if not params:
        return []
    ops = ["freeze", "unfreeze"]
    weight = getattr(module, "weight", None)
    if isinstance(weight, th.Tensor):
        if tracked or callable(getattr(module, "reset_parameters", None)):
            ops.append("reset")
        ops.append("perturb")
        if tracked:
            ops += ["prune", "add"]
    return ops


def build_architecture(model, trace: Optional[ArchitectureTrace], version: int = 0) -> dict:
    inner = resolve_inner(model)
    if inner is None:
        return {"success": False, "message": "No model registered", "nodes": [], "edges": []}

    # Every path, shared modules included: the tracer names each call of a
    # shared module by its path under the caller (see _CallTracker).
    modules = OrderedDict(_all_named_modules(inner))
    declared_roles = _declared_roles(inner)
    calls = trace.calls if trace is not None else OrderedDict()
    containers = trace.containers if trace is not None else {}
    leaf_modules = {rec["module_name"] for rec in calls.values()}
    calls_by_module: Dict[str, List[dict]] = {}
    for rec in calls.values():
        calls_by_module.setdefault(rec["module_name"], []).append(rec)

    # A module that ran as one unit hides its children (they never ran).
    hidden_prefixes = tuple(f"{name}." for name in leaf_modules if name in modules and _has_children(modules[name]))

    # A parameter held by several modules (a language model's lm_head tied to
    # its token embedding): editing one edits the others -- say so.
    owners: Dict[int, List[str]] = {}
    for name, module in modules.items():
        for param in module.parameters(recurse=False):
            if name not in owners.setdefault(id(param), []):
                owners[id(param)].append(name)

    def _tied_with(name: str, module: nn.Module) -> List[str]:
        return sorted({o for param in module.parameters(recurse=False)
                       for o in owners.get(id(param), []) if o != name})

    nodes: List[dict] = []
    group_ids: List[str] = []
    for name, module in modules.items():
        if name == "" or _is_internal(module) or (hidden_prefixes and name.startswith(hidden_prefixes)):
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        depth = name.count(".")
        total, trainable = _count_params(module)
        description = _extra_repr(module)
        tied = _tied_with(name, module)
        if tied:
            description = f"{description}; weights tied with {', '.join(tied)}" if description \
                else f"weights tied with {', '.join(tied)}"
        base = {
            "name": name.rsplit(".", 1)[-1],
            "parent_id": parent,
            "type": _module_type(module),
            "parameter_count": total,
            "trainable_parameter_count": trainable,
            "description": description,
            "depth": depth,
            "module_id": name,
        }
        is_group = _has_children(module) and name not in leaf_modules
        if is_group:
            rec = containers.get(name, {})
            role = declared_roles.get(name) or _deep_role(name, module) \
                or (guess_role(name) if depth <= 2 else "block")
            node = dict(base, id=name, kind="group", role=role,
                        layer_id=-1, input_shape=rec.get("in_shape", []),
                        output_shape=rec.get("out_shape", []), order=-1, call_index=1,
                        frozen=bool(total) and trainable == 0, frozen_neuron_count=0,
                        supported_ops=["freeze", "unfreeze", "reset", "perturb"] if total else [],
                        input_neurons=0, output_neurons=0, executed=False)
            nodes.append(node)
            group_ids.append(name)
            continue

        tracked, layer_id = _is_tracked(model, module)
        whole_frozen, frozen_count, _ = _frozen_neurons(module, tracked)
        ops = supported_ops(module, tracked)
        records = calls_by_module.get(name) or [None]
        for i, rec in enumerate(records):
            out_shape = rec["out_shape"] if rec else []
            n_in, n_out = _layer_neurons(module, tracked, out_shape)
            nodes.append(dict(
                base,
                id=rec["node_id"] if rec else name,
                kind="layer", role="",
                layer_id=layer_id,
                input_shape=rec["in_shape"] if rec else [],
                output_shape=out_shape,
                order=int(rec["order"]) if rec else -1,
                call_index=i + 1,
                frozen=whole_frozen,
                frozen_neuron_count=frozen_count,
                supported_ops=ops,
                input_neurons=n_in,
                output_neurons=n_out,
                executed=rec is not None,
            ))

    # Groups: executed = any descendant ran; order = first descendant's.
    by_id = {n["id"]: n for n in nodes}
    for node in nodes:
        if node["kind"] != "layer" or node["order"] < 0:
            continue
        parent = node["parent_id"]
        while parent:
            group = by_id.get(parent)
            if group is None:
                break
            group["executed"] = True
            if group["order"] < 0 or node["order"] < group["order"]:
                group["order"] = node["order"]
            parent = group["parent_id"]

    # Residual adds ("+" nodes), inside the group they happened in.
    group_set = set(group_ids)
    for rec in (trace.merges.values() if trace is not None else []):
        parent = rec["scope"]
        while parent and parent not in group_set:
            parent = _parent_name(parent)
        nodes.append({
            "id": rec["node_id"], "name": "add", "kind": "op", "parent_id": parent, "type": "Add",
            "role": "", "layer_id": -1, "parameter_count": 0, "trainable_parameter_count": 0,
            "description": "adds two branches (a residual / skip connection)",
            "depth": parent.count(".") + 1 if parent else 0, "module_id": "",
            "input_shape": rec["out_shape"], "output_shape": rec["out_shape"],
            "order": int(rec["order"]), "call_index": int(rec["call_index"]),
            "frozen": False, "frozen_neuron_count": 0, "supported_ops": [],
            "input_neurons": 0, "output_neurons": 0, "executed": True,
        })

    _assign_bottleneck(nodes)

    in_node = {"id": INPUT_NODE, "name": "input", "kind": "input", "parent_id": "", "type": "Input",
               "role": "", "layer_id": -1, "output_shape": trace.input_shape if trace else [],
               "order": -2, "depth": 0, "executed": True, "module_id": INPUT_NODE}
    out_node = {"id": OUTPUT_NODE, "name": "output", "kind": "output", "parent_id": "", "type": "Output",
                "role": "", "layer_id": -1, "input_shape": trace.output_shape if trace else [],
                "order": 10 ** 9, "depth": 0, "executed": True, "module_id": OUTPUT_NODE}

    if trace is not None and trace.traced:
        edges = [{"source": s, "target": t, "shape": shape} for (s, t), shape in trace.edges.items()]
        traced, message = True, ""
    else:
        layer_nodes = [n for n in nodes if n["kind"] == "layer"]
        chain = [INPUT_NODE] + [n["id"] for n in layer_nodes] + [OUTPUT_NODE]
        edges = [{"source": a, "target": b, "shape": []} for a, b in zip(chain, chain[1:])]
        traced = False
        message = (trace.message if trace is not None else "") or "no input to trace the model with"

    total, trainable = _count_params(inner)
    dep = getattr(model, "_dep_manager", None)
    try:
        deps_available = bool(dep is not None and dep.id_2_layer)
    except Exception:
        deps_available = False
    try:
        age = int(model.get_age())
    except Exception:
        age = 0
    return {
        "success": True,
        "message": "",
        "model_name": inner.__class__.__name__,
        "model_age": age,
        "task_type": str(getattr(model, "task_type", "") or getattr(inner, "task_type", "") or ""),
        "nodes": [in_node] + nodes + [out_node],
        "edges": edges,
        "architecture_version": version,
        "dependencies_available": deps_available,
        "traced": traced,
        "trace_message": message,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def _assign_bottleneck(nodes: List[dict]) -> None:
    """Encoder siblings followed by decoder siblings: the last encoder one is
    the bottleneck (U-Net's deepest down block), unless one is named so."""
    siblings: Dict[str, List[dict]] = {}
    for node in nodes:
        if node["kind"] == "group":
            siblings.setdefault(node["parent_id"], []).append(node)
    for group in siblings.values():
        roles = {g["role"] for g in group}
        if "bottleneck" in roles or not {"encoder", "decoder"} <= roles:
            continue
        ordered = sorted((g for g in group if g["order"] >= 0), key=lambda g: g["order"])
        for a, b in zip(ordered, ordered[1:]):
            if a["role"] == "encoder" and b["role"] == "decoder":
                a["role"] = "bottleneck"
                break


# =============================================================================
# Sample loading
# =============================================================================
def _loader_entries(loaders: Dict[str, Any]) -> List[Tuple[str, Any, Any, str]]:
    entries = []
    for name, loader in (loaders or {}).items():
        if loader is None:
            continue
        try:
            dataset = getattr(loader, "tracked_dataset", None)
        except Exception:
            dataset = None
        if dataset is None:
            continue
        split = str(getattr(dataset, "_dataset_split", "") or name)
        entries.append((str(name), loader, dataset, split))
    return entries


def _locate(dataset, sample_id: str) -> Optional[int]:
    """The (physical) dataset index of *sample_id*, without loading it."""
    candidates = [sample_id]
    try:
        candidates.append(int(sample_id))
    except (TypeError, ValueError):
        pass
    for sid in candidates:
        locate = getattr(dataset, "get_physical_location", None)
        if callable(locate):
            try:
                return int(locate(sid)[0])
            except Exception:
                pass
        index_of = getattr(dataset, "get_index_from_sample_id", None)
        if callable(index_of):
            try:
                return int(index_of(sid))
            except Exception:
                pass
    return None


def _item_for(dataset, sample_id: str):
    """The dataset item for *sample_id*, exactly as the DataLoader sees it."""
    index = _locate(dataset, sample_id)
    if index is None:
        raise KeyError(f"sample {sample_id!r} not found")
    return dataset[index]


def _collate_fn(loader):
    try:
        dataloader = loader.get_dataloader() if callable(getattr(loader, "get_dataloader", None)) else None
    except Exception:
        dataloader = None
    fn = getattr(dataloader, "collate_fn", None) if dataloader is not None else None
    if fn is None:
        from torch.utils.data import default_collate
        fn = default_collate
    return fn


def resolve_samples(loaders: Dict[str, Any], sample_ids: Sequence[str],
                    origins: Sequence[str]) -> List[Tuple[str, str, Any]]:
    """[(sample_id, origin, loader_entry)] for the ids that exist."""
    entries = _loader_entries(loaders)
    resolved = []
    for i, sid in enumerate(sample_ids):
        origin = origins[i] if i < len(origins) else ""
        chosen = [e for e in entries if origin and origin in (e[0], e[3])] or entries
        for entry in chosen:
            if _locate(entry[2], str(sid)) is None:
                continue
            resolved.append((str(sid), entry[3], entry))
            break
    return resolved


def build_batch(entry, sample_ids: Sequence[str]):
    _, loader, dataset, _ = entry
    items = [_item_for(dataset, sid) for sid in sample_ids]
    return _collate_fn(loader)(items)


def samples_with_tag(tag: str, origin: str = "") -> Tuple[List[str], List[str]]:
    """(sample ids, origins) of every sample carrying *tag*, in dataframe order."""
    from weightslab.backend.ledgers import get_dataframe
    tag = str(tag).strip()
    if tag.lower().startswith("tag:"):
        tag = tag[4:]
    manager = get_dataframe()
    if manager is None:
        return [], []
    try:
        df = manager.get_df_view(column="origin", value=origin) if origin else manager.get_df_view()
    except Exception:
        df = manager.get_df_view()
    column = next((c for c in df.columns if str(c).lower() == f"tag:{tag}".lower()), None)
    if column is None:
        return [], []
    values = df[column]
    if values.dtype == object or str(values.dtype) == "category":
        mask = values.astype(str).str.lower().isin(["true", "1", "1.0"])
    else:
        mask = values.fillna(0).astype(bool)
    tagged = df[mask]
    index = tagged.index
    ids = index.get_level_values(0) if getattr(index, "nlevels", 1) > 1 else index
    origins = list(tagged["origin"]) if "origin" in tagged.columns else [origin] * len(tagged)
    seen, out_ids, out_origins = set(), [], []
    for sid, org in zip(ids, origins):
        sid = str(sid)
        if sid in seen:
            continue
        seen.add(sid)
        out_ids.append(sid)
        out_origins.append(str(org or ""))
    return out_ids, out_origins


# =============================================================================
# 2. Representation sessions
# =============================================================================
class RepresentationSession:
    def __init__(self, sample_ids, origins, label, tag, requested, arch_version, edit_count, age):
        self.id = uuid.uuid4().hex[:12]
        self.state = "running"
        self.done = 0
        self.total = len(sample_ids)
        self.message = ""
        self.sample_ids: List[str] = list(sample_ids)
        self.origins: List[str] = list(origins)
        self.source_label = label
        self.tag = tag
        self.requested = requested
        self.architecture_version = arch_version
        self.edit_count = edit_count
        self.model_age = age
        self.accs: Dict[str, NodeAccumulator] = {}
        # node id -> {"output": [preview...], "input": [preview...]} in sample order
        self.previews: Dict[str, Dict[str, List[dict]]] = {}
        self.preview_ids: List[Tuple[str, str]] = []        # (sample_id, origin)
        self.preview_bytes = 0
        self.previews_complete = True
        self.cancel = threading.Event()
        self.created = time.time()

    def status(self, stale: bool) -> dict:
        return {
            "session_id": self.id, "state": self.state, "done": self.done, "total": self.total,
            "message": self.message, "source_label": self.source_label,
            "sample_ids": self.sample_ids, "origins": self.origins,
            "architecture_version": self.architecture_version, "model_age": self.model_age,
            "requested": self.requested, "stale": stale, "tag": self.tag,
        }


def _sdpa_weights(q, k, v=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                  enable_gqa=False, **_ignored) -> Optional[th.Tensor]:
    """The attention weights F.scaled_dot_product_attention used (it never
    returns them): softmax(q k^T * scale + mask), (B, heads, T, S)."""
    if not isinstance(q, th.Tensor) or not isinstance(k, th.Tensor) or q.ndim < 3:
        return None
    if q.ndim >= 4 and k.shape[-3] != q.shape[-3] and q.shape[-3] % k.shape[-3] == 0:
        k = k.repeat_interleave(q.shape[-3] // k.shape[-3], dim=-3)
    L, S = q.shape[-2], k.shape[-2]
    scale = float(scale) if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    att = (q.float() @ k.float().transpose(-2, -1)) * scale
    if is_causal:
        keep = th.ones(L, S, dtype=th.bool, device=att.device).tril()
        att = att.masked_fill(~keep, float("-inf"))
    if isinstance(attn_mask, th.Tensor):
        if attn_mask.dtype == th.bool:
            att = att.masked_fill(~attn_mask, float("-inf"))
        else:
            att = att + attn_mask.float()
    return th.nan_to_num(att.softmax(dim=-1), nan=0.0)


_ATTENTION_TOKENS = frozenset({"attn", "attention", "mha", "selfattention", "crossattention"})


def _is_attention_module(name: str, module: Optional[nn.Module]) -> bool:
    kind = type(module).__name__.lower() if module is not None else ""
    if "attention" in kind or kind.endswith("attn"):
        return True
    return bool(set(_tokens(name.rsplit(".", 1)[-1])) & _ATTENTION_TOKENS)


def _has_attention(inner: nn.Module) -> bool:
    """Any module that looks like attention (worth watching for weights)."""
    return any(_is_attention_module(name, m) for name, m in inner.named_modules() if name)


class _AttentionMode(TorchFunctionMode):
    """Catches attention weights computed functionally inside a module: a
    softmax over the last dim of a (B, heads, T, S) tensor in an attention
    module, or F.scaled_dot_product_attention (weights recomputed)."""

    def __init__(self, on_weights: Callable[[th.Tensor, bool], None]):
        super().__init__()
        self.on_weights = on_weights

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            name = getattr(func, "__name__", "")
            if name == "scaled_dot_product_attention":
                weights = _sdpa_weights(*args, **kwargs)
                if weights is not None:
                    self.on_weights(weights, True)
            elif name == "softmax" and isinstance(out, th.Tensor) and out.ndim == 4:
                dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
                if dim in (-1, 3):
                    self.on_weights(out, False)
        except Exception as exc:
            logger.debug(f"[modeling] attention capture failed: {exc}")
        return out


class _Capture:
    """Hook callbacks that feed a session from forward passes."""

    def __init__(self, session: RepresentationSession, batch_ids: List[Tuple[str, str]],
                 batch: int, preview_rows: Dict[int, int], only: Optional[set] = None,
                 max_channels: int = PREVIEW_CHANNELS, store_stats: bool = True,
                 tile: int = PREVIEW_TILE):
        self.session = session
        self.batch_ids = batch_ids
        self.batch = batch
        self.preview_rows = preview_rows        # row in batch -> preview slot
        self.only = only
        self.max_channels = max_channels
        self.store_stats = store_stats
        self.shape_hints: Dict[Tuple[int, ...], int] = {}
        self.tile = tile
        # The model input of the current forward: previewed as the picture it
        # is (RGB / gray), and so is every layer input that is this very
        # tensor (typically the first conv).
        self._model_input: Optional[th.Tensor] = None
        self.tracker: Optional["_CallTracker"] = None    # set by the forward
        self.inner: Optional[nn.Module] = None

    def on_attention(self, weights: th.Tensor, explicit: bool) -> None:
        """Attention weights (B, heads, T, S) seen inside the current module:
        its "attention" previews, one tile per head."""
        stack = self.tracker.stack if self.tracker is not None else []
        frame = stack[-1] if stack else None
        if frame is None or not frame.name:
            return
        if not explicit and not _is_attention_module(frame.name, frame.module):
            return
        node_id = frame.name if frame.had_child else frame.node_id
        if not self._wanted(node_id) or weights.ndim != 4:
            return
        self._store_previews(node_id, "attention", weights.detach(), 1, layout="attention")

    def _wanted(self, node_id: str) -> bool:
        return self.only is None or node_id in self.only

    def _store_previews(self, node_id: str, direction: str, tensor: th.Tensor, axis: Optional[int],
                        as_image: bool = False, layout: str = ""):
        if not self.preview_rows:
            return
        session = self.session
        if session.preview_bytes > PREVIEW_BUDGET_BYTES and self.only is None:
            session.previews_complete = False
            return
        per_sample = tensor.shape[0] == self.batch if tensor.ndim >= 1 else False
        if not per_sample and self.batch != 1:
            return
        slot_axis = None
        if axis is not None:
            slot_axis = (axis - 1) if per_sample else axis
            slot_axis = 0 if slot_axis == 0 else -1
        bucket = session.previews.setdefault(node_id, {}).setdefault(direction, [])
        for row, slot in self.preview_rows.items():
            sample = tensor[row] if per_sample else tensor
            preview = make_preview(sample, slot_axis, self.max_channels, self.tile, as_image=as_image)
            if preview is None:
                continue
            if layout:
                preview["layout"] = layout
            if preview.get("layout") == "tokens":
                text = _token_text(self.inner, preview.get("token_ids") or [])
                if text:
                    preview["token_text"] = text
            sid, origin = session.preview_ids[slot]
            preview.update(sample_id=sid, origin=origin, direction=direction)
            while len(bucket) <= slot:
                bucket.append(None)
            bucket[slot] = preview
            session.preview_bytes += len(preview["data"])

    def on_pre(self, frame: _Frame, args, kwargs):
        node_id = INPUT_NODE if frame.name == "" else frame.node_id
        if frame.name == "":
            # Recorded even when only one layer is wanted (the zoom).
            self._model_input = _first_tensor((args, kwargs))
        if not self._wanted(node_id):
            return
        tensor = _first_tensor((args, kwargs))
        if tensor is None:
            return
        axis = _channel_axis(frame.module if frame.name else None, tensor, self.shape_hints)
        if node_id == INPUT_NODE:
            if self.store_stats:
                self.session.accs.setdefault(INPUT_NODE, NodeAccumulator()).update(tensor, axis, self.batch)
            self._store_previews(INPUT_NODE, "output", tensor, axis, as_image=True)
        else:
            self._store_previews(node_id, "input", tensor, axis,
                                 as_image=self._model_input is not None and tensor is self._model_input)

    def on_post(self, frame: _Frame, output):
        node_id = OUTPUT_NODE if frame.name == "" else (
            frame.name if frame.had_child else frame.node_id)
        tensor = _first_tensor(output)
        if tensor is None:
            return
        axis = _channel_axis(frame.module if frame.name else None, tensor, self.shape_hints)
        if frame.name and isinstance(frame.module, _CHANNEL_FIRST_TYPES + _FEATURE_LAST_TYPES) \
                and axis is not None:
            self.shape_hints[tuple(tensor.shape)] = axis
        if not self._wanted(node_id):
            return
        if self.store_stats:
            self.session.accs.setdefault(node_id, NodeAccumulator()).update(tensor, axis, self.batch)
        self._store_previews(node_id, "output", tensor, axis)


class ModelRepresentationManager:
    """Process-wide state for the Modeling tab (sessions + architecture cache)."""

    MAX_SESSIONS = 4

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: "OrderedDict[str, RepresentationSession]" = OrderedDict()
        self._threads: Dict[str, threading.Thread] = {}
        self._trace: Optional[ArchitectureTrace] = None
        self._signature = None
        self.version = 0
        self.edit_count = 0
        self._trace_inputs = None     # a one-sample input the last trace used

    # ---- architecture ------------------------------------------------------
    @staticmethod
    def _structure_signature(inner: nn.Module):
        return tuple(
            (name, type(m).__name__, tuple(tuple(p.shape) for p in m.parameters(recurse=False)))
            for name, m in inner.named_modules()
        )

    def _default_trace_inputs(self, model, loaders):
        if self._trace_inputs is not None:
            return self._trace_inputs
        inner = resolve_inner(model)
        entries = _loader_entries(loaders)
        entries.sort(key=lambda e: 0 if "train" in e[0].lower() else 1)
        for entry in entries:
            dataset = entry[2]
            batch = None
            try:
                # unique_ids is often a numpy array: never test its truthiness.
                raw_ids = getattr(dataset, "unique_ids", None)
                if raw_ids is not None and len(raw_ids):
                    batch = build_batch(entry, [str(raw_ids[0])])
            except Exception as exc:
                logger.debug(f"[modeling] first sample of {entry[0]} by id failed: {exc}")
            try:
                if batch is None:
                    batch = _collate_fn(entry[1])([dataset[0]])
                return _inputs_from_batch(batch, inner)
            except Exception as exc:
                logger.debug(f"[modeling] no trace input from {entry[0]}: {exc}")
        dummy = getattr(model, "dummy_input", None)
        if isinstance(dummy, th.Tensor):
            return dummy[:1]
        shape = getattr(inner, "input_shape", None)
        if isinstance(shape, (tuple, list)) and shape and all(isinstance(s, int) for s in shape):
            vocab = getattr(inner, "vocab_size", None)
            if isinstance(vocab, int) and vocab > 0:     # a language model: token ids
                return th.randint(0, vocab, (1,) + tuple(shape[1:]))
            return th.randn((1,) + tuple(shape[1:]))
        return None

    def architecture(self, model, loaders, refresh: bool = False) -> dict:
        inner = resolve_inner(model)
        if inner is None:
            return {"success": False, "message": "No model registered", "nodes": [], "edges": []}
        with self._lock:
            signature = self._structure_signature(inner)
            if signature != self._signature:
                if self._signature is not None:
                    self.version += 1
                self._signature = signature
                self._trace = None
            if self._trace is None or refresh:
                inputs = self._default_trace_inputs(model, loaders)
                if inputs is None:
                    trace = ArchitectureTrace()
                    trace.message = "no sample or input_shape to trace the model with"
                else:
                    with _training_lock():
                        trace = trace_architecture(model, inputs)
                self._trace = trace
            return build_architecture(model, self._trace, self.version)

    def note_edit(self, structural: bool) -> None:
        with self._lock:
            self.edit_count += 1
            if structural:
                self.version += 1
                self._signature = None   # forces a re-trace (without a 2nd bump)
                self._trace_inputs = None

    # ---- sessions ------------------------------------------------------------
    def start(self, model, loaders, sample_ids: Sequence[str], origins: Sequence[str],
              tag: str = "", tag_origin: str = "", max_samples: int = 0) -> dict:
        max_samples = int(max_samples) if max_samples and max_samples > 0 else DEFAULT_MAX_SAMPLES
        ids = [str(s) for s in sample_ids]
        origins = [str(o) for o in origins]
        if tag:
            ids, origins = samples_with_tag(tag, tag_origin)
        requested = len(ids)
        resolved = resolve_samples(loaders, ids[:max_samples], origins[:max_samples])
        if tag:
            label = f"tag '{tag}' ({requested} sample{'s' if requested != 1 else ''})"
        else:
            label = f"{requested} sample{'s' if requested != 1 else ''}"

        inner = resolve_inner(model)
        try:
            age = int(model.get_age())
        except Exception:
            age = 0
        with self._lock:
            for running in self._sessions.values():
                if running.state == "running":
                    running.cancel.set()
            session = RepresentationSession(
                [r[0] for r in resolved], [r[1] for r in resolved], label, tag,
                requested, self.version, self.edit_count, age)
            session.preview_ids = [(r[0], r[1]) for r in resolved[:PREVIEW_SAMPLES]]
            self._sessions[session.id] = session
            while len(self._sessions) > self.MAX_SESSIONS:
                self._sessions.popitem(last=False)
        if inner is None:
            session.state, session.message = "error", "No model registered"
            return session.status(False)
        if not resolved:
            session.state = "error"
            session.message = ("none of the requested samples could be loaded"
                               if requested else "no samples to represent")
            return session.status(False)
        thread = threading.Thread(
            target=self._run, args=(session, model, resolved),
            name=f"wl-modeling-{session.id}", daemon=True)
        self._threads[session.id] = thread
        thread.start()
        return session.status(False)

    def _forward_chunk(self, session, model, inner, entry, chunk, slots, only=None,
                       max_channels=PREVIEW_CHANNELS, store_stats=True, tile=PREVIEW_TILE):
        batch = build_batch(entry, [c[0] for c in chunk])
        inputs = _to_device(_inputs_from_batch(batch, inner), _model_device(inner))
        size = _batch_size_of(inputs)
        if size != len(chunk):
            size = len(chunk)
        rows = {row: slot for row, slot in slots.items() if row < size}
        capture = _Capture(session, [(c[0], c[1]) for c in chunk], size, rows, only,
                           max_channels, store_stats, tile)
        tracker = _CallTracker(inner, capture.on_pre, capture.on_post)
        capture.tracker = tracker
        capture.inner = inner
        with _inference_mode(model, inner), tracker:
            if rows and self._watch_attention(inner):
                with _AttentionMode(capture.on_attention):
                    _call_model(model, inputs)
            else:
                _call_model(model, inputs)
        if self._trace_inputs is None:
            try:
                single = build_batch(entry, [chunk[0][0]])
                self._trace_inputs = _inputs_from_batch(single, inner)
            except Exception:
                pass

    def _watch_attention(self, inner: nn.Module) -> bool:
        """Whether forwards should catch attention weights: the model has
        attention-like modules (cached per structure)."""
        key = (id(inner), self.version)
        cached = getattr(self, "_attention_key", None)
        if cached != key:
            self._attention_key = key
            self._attention_on = _has_attention(inner)
        return self._attention_on

    def _run(self, session: RepresentationSession, model, resolved) -> None:
        inner = resolve_inner(model)
        chunk_size = max(1, int(os.environ.get("WEIGHTSLAB_MODELING_BATCH", "16") or 16))
        preview_slot = {(sid, origin): i for i, (sid, origin) in enumerate(session.preview_ids)}
        try:
            # Group by loader, keep order within a loader.
            groups: "OrderedDict[str, list]" = OrderedDict()
            for sid, origin, entry in resolved:
                groups.setdefault(entry[0], []).append((sid, origin, entry))
            for items in groups.values():
                entry = items[0][2]
                for start in range(0, len(items), chunk_size):
                    if session.cancel.is_set():
                        raise _Cancelled()
                    chunk = items[start:start + chunk_size]
                    slots = {row: preview_slot[(c[0], c[1])] for row, c in enumerate(chunk)
                             if (c[0], c[1]) in preview_slot}
                    with _training_lock(cancel=session.cancel):
                        self._forward_chunk(session, model, inner, entry, chunk, slots)
                    session.done += len(chunk)
            if session.cancel.is_set():
                # Replaced while its last chunk ran: what the user asked for
                # now is the newer session, never this one.
                raise _Cancelled()
            session.state = "done"
        except _Cancelled:
            session.state = "cancelled"
            session.message = "replaced by a newer representation"
        except Exception as exc:
            logger.warning(f"[modeling] representation failed: {exc}")
            logger.debug("[modeling] representation failure", exc_info=True)
            session.state = "error"
            session.message = f"{type(exc).__name__}: {exc}"
        finally:
            self._threads.pop(session.id, None)

    def get(self, session_id: str = "") -> Optional[RepresentationSession]:
        with self._lock:
            if session_id:
                return self._sessions.get(session_id)
            for session in reversed(self._sessions.values()):
                if session.state != "cancelled":
                    return session
            return None

    def is_stale(self, session: RepresentationSession) -> bool:
        return session.edit_count != self.edit_count or session.architecture_version != self.version

    def wait(self, session_id: str, timeout: float = 60.0) -> Optional[RepresentationSession]:
        thread = self._threads.get(session_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(session_id)

    def clear(self) -> None:
        """Forget every representation (a running one is cancelled)."""
        with self._lock:
            for session in self._sessions.values():
                if session.state == "running":
                    session.cancel.set()
            self._sessions.clear()

    def sample_previews(self, model, loaders, node_id: str, sample_id: str, origin: str = "",
                        max_channels: int = 64, tile: int = 96) -> List[dict]:
        """Input/output previews of ONE sample at *node_id*, at *tile* px.

        Recomputed by a forward of that sample alone (never stored): the
        activation zoom asks for one sample at a time, at a resolution the
        session's thumbnails were never meant to keep.
        """
        inner = resolve_inner(model)
        resolved = resolve_samples(loaders, [str(sample_id)], [origin] if origin else [])
        if inner is None or not resolved:
            return []
        sid, org, entry = resolved[0]
        scratch = RepresentationSession([sid], [org], "", "", 1, self.version, self.edit_count, 0)
        scratch.preview_ids = [(sid, org)]
        with _training_lock():
            self._forward_chunk(scratch, model, inner, entry, [(sid, org)], {0: 0}, only={node_id},
                                max_channels=max_channels, store_stats=False,
                                tile=max(8, min(int(tile), 256)))
        bucket = scratch.previews.get(node_id, {})
        return [p for direction in ("input", "output", "attention") for p in (bucket.get(direction) or []) if p]

    def summaries(self, session: RepresentationSession) -> List[dict]:
        return [acc.summary(node_id) for node_id, acc in list(session.accs.items())]

    # ---- layer details ---------------------------------------------------------
    def ensure_previews(self, session, model, loaders, node_id, max_channels) -> None:
        """Previews for *node_id* if the session ran out of preview budget."""
        bucket = session.previews.get(node_id)
        if bucket and bucket.get("output"):
            return
        if session.previews_complete or not session.preview_ids:
            return
        inner = resolve_inner(model)
        resolved = resolve_samples(loaders, [s for s, _ in session.preview_ids],
                                   [o for _, o in session.preview_ids])
        if not resolved:
            return
        entry = resolved[0][2]
        chunk = [(r[0], r[1]) for r in resolved if r[2][0] == entry[0]]
        slots = {row: row for row in range(len(chunk))}
        with _training_lock():
            self._forward_chunk(session, model, inner, entry, chunk, slots,
                                only={node_id}, max_channels=max_channels, store_stats=False)


MANAGER = ModelRepresentationManager()


# =============================================================================
# Layer details
# =============================================================================
def classify_health(weight_norm: Optional[float], active_rate: Optional[float],
                    mean_abs: Optional[float], median_norm: float, median_act: float) -> str:
    if active_rate is not None and active_rate <= 0.0:
        return "dead"
    if weight_norm is not None and weight_norm <= 1e-8:
        return "dead"
    weak = False
    if weight_norm is not None and median_norm > 0 and weight_norm < 0.25 * median_norm:
        weak = True
    if mean_abs is not None and median_act > 0 and mean_abs < 0.05 * median_act:
        weak = True
    if weak:
        return "weak"
    if weight_norm is None and active_rate is None:
        return "unknown"
    return "healthy"


def layer_details(model, loaders, node_id: str, session_id: str = "",
                  max_channels: int = 0, max_samples: int = 0,
                  manager: ModelRepresentationManager = MANAGER,
                  skip_previews: bool = False, sample_id: str = "", origin: str = "",
                  tile_size: int = 0) -> dict:
    max_channels = max_channels if max_channels > 0 else PREVIEW_CHANNELS
    max_samples = max_samples if max_samples > 0 else PREVIEW_SAMPLES
    arch = manager.architecture(model, loaders)
    if not arch.get("success"):
        return {"success": False, "message": arch.get("message", "")}
    node = next((n for n in arch["nodes"] if n["id"] == node_id), None)
    if node is None:
        return {"success": False, "message": f"unknown node {node_id!r}"}
    inner = resolve_inner(model)
    session = manager.get(session_id)
    module = None
    if node["kind"] in ("layer", "group"):
        try:
            module = inner.get_submodule(node["module_id"])
        except Exception:
            module = None

    neurons: List[dict] = []
    neuron_axis = ""
    health_source = ""
    histogram, hist_min, hist_max = [], 0.0, 0.0
    if module is not None and node["kind"] == "layer":
        tracked, _ = _is_tracked(model, module)
        whole_frozen, _, frozen_idx = _frozen_neurons(module, tracked)
        wstats = neuron_weight_stats(module)
        acc = session.accs.get(node_id) if session is not None else None
        cstats = acc.channel_stats() if acc is not None else None
        n = len(wstats[0]) if wstats is not None else (len(cstats["mean"]) if cstats else 0)
        if cstats is not None and len(cstats["mean"]) != n:
            cstats = None if wstats is not None else cstats
        if cstats is not None:
            health_source = node_id
        # A weighted layer's neurons are judged where they can die: at the
        # ReLU they feed (conv -> bn -> relu), when there is one.
        if wstats is not None and session is not None:
            relu = _downstream_relu(arch, inner, node_id)
            relu_acc = session.accs.get(relu) if relu else None
            relu_stats = relu_acc.channel_stats() if relu_acc is not None else None
            if relu_stats is not None and len(relu_stats["mean"]) == n:
                cstats = relu_stats
                relu_type = next((x["type"] for x in arch["nodes"] if x["id"] == relu), "ReLU")
                health_source = f"{relu} ({relu_type})"
        if n:
            neuron_axis = "channel" if isinstance(module, _CHANNEL_FIRST_TYPES) else "feature"
            norms = wstats[0] if wstats is not None else None
            median_norm = float(np.median(norms)) if norms is not None and len(norms) else 0.0
            acts = np.asarray(cstats["mean_abs"]) if cstats else None
            median_act = float(np.median(acts)) if acts is not None and len(acts) else 0.0
            for i in range(n):
                wn = float(norms[i]) if norms is not None else None
                rate = float(cstats["active_rate"][i]) if cstats else None
                mabs = float(cstats["mean_abs"][i]) if cstats else None
                neurons.append({
                    "index": i,
                    "weight_norm": wn or 0.0,
                    "weight_mean_abs": float(wstats[1][i]) if wstats is not None else 0.0,
                    "activation_mean": float(cstats["mean"][i]) if cstats else 0.0,
                    "activation_max": float(cstats["max"][i]) if cstats else 0.0,
                    "active_rate": rate or 0.0,
                    "frozen": whole_frozen or i in frozen_idx,
                    "health": classify_health(wn, rate, mabs, median_norm, median_act),
                    "has_activation": cstats is not None,
                })
    if module is not None and node["kind"] in ("layer", "group"):
        # A group's "weights graph" is every parameter it holds.
        weights = [p.detach().float().reshape(-1)
                   for p in module.parameters(recurse=node["kind"] == "group") if p.ndim >= 1]
        if weights:
            flat = th.cat(weights)
            if flat.numel() > 2_000_000:
                flat = flat[th.randperm(flat.numel())[:2_000_000]]
            hist_min, hist_max = float(flat.min()), float(flat.max())
            if hist_max <= hist_min:
                hist_max = hist_min + 1e-6
            histogram = th.histc(flat.cpu(), bins=40, min=hist_min, max=hist_max).tolist()

    previews: List[dict] = []
    activation = None
    if sample_id:
        # The activation zoom: one sample, recomputed at a higher resolution.
        try:
            previews = manager.sample_previews(
                model, loaders, node_id, sample_id, origin,
                max_channels=max_channels, tile=tile_size or 96)
        except Exception as exc:
            logger.debug(f"[modeling] sample previews failed for {node_id}/{sample_id}: {exc}")
    elif session is not None and not skip_previews:
        try:
            manager.ensure_previews(session, model, loaders, node_id, max_channels)
        except Exception as exc:
            logger.debug(f"[modeling] lazy previews failed for {node_id}: {exc}")
        bucket = session.previews.get(node_id, {})
        for direction in ("input", "output", "attention"):
            for preview in (bucket.get(direction) or [])[:max_samples]:
                if preview is None:
                    continue
                if preview["tiles"] > max_channels and preview["layout"] == "channels":
                    preview = _truncate_preview(preview, max_channels)
                previews.append(preview)
    if session is not None:
        acc = session.accs.get(node_id)
        if acc is not None:
            activation = acc.summary(node_id)

    children = [n["id"] for n in arch["nodes"] if n.get("parent_id") == node_id] \
        if node["kind"] == "group" else []
    return {
        "success": True, "message": "", "node": node, "neurons": neurons,
        "weight_histogram": histogram, "weight_hist_min": hist_min, "weight_hist_max": hist_max,
        "previews": previews, "activation": activation,
        "session_id": session.id if session is not None else "",
        "neuron_axis": neuron_axis, "children": children,
        "health_source": health_source,
        "model_age": _model_age(model),
        "neuron_groups": neuron_groups(inner, node_id, len(neurons)) if neurons else [],
    }


_HEAD_ATTRS = ("n_head", "num_heads", "n_heads", "nhead", "num_attention_heads", "heads")
_WIDTH_ATTRS = ("n_embd", "embed_dim", "hidden_size", "d_model", "dim", "embedding_dim")
# Output projections: their OUTPUT neurons are the residual width, not heads.
_OUT_PROJ_NAMES = frozenset({"wo", "o", "o_proj", "out_proj", "c_proj", "proj", "dense", "out"})


def neuron_groups(inner: Optional[nn.Module], node_id: str, n_neurons: int) -> List[dict]:
    """How a projection's output neurons split into attention heads (and, for
    a fused projection, into q / k / v): [{label, start, size, kind}]. Empty
    when the layer is not a per-head projection of an attention module."""
    if inner is None or n_neurons <= 0:
        return []
    module_id = node_id.split("#", 1)[0]
    leaf = module_id.rsplit(".", 1)[-1].lower()
    if leaf in _OUT_PROJ_NAMES:
        return []
    parent_id = _parent_name(module_id)
    try:
        parent = inner.get_submodule(parent_id) if parent_id else inner
    except Exception:
        return []
    if not _is_attention_module(parent_id, parent):
        return []
    heads = next((int(getattr(parent, a)) for a in _HEAD_ATTRS
                  if isinstance(getattr(parent, a, None), int) and int(getattr(parent, a)) > 0), 0)
    if heads <= 0:
        return []
    width = next((int(getattr(parent, a)) for a in _WIDTH_ATTRS
                  if isinstance(getattr(parent, a, None), int) and int(getattr(parent, a)) > 0), 0)
    if width and n_neurons % width == 0 and n_neurons // width in (1, 2, 3):
        parts = n_neurons // width
    else:
        fused = {"qkv", "c_attn", "in_proj", "wqkv"}
        parts = 3 if leaf in fused or "qkv" in leaf else (2 if leaf in {"kv", "wkv", "kv_proj"} else 1)
        width = n_neurons // parts
    if width % heads or parts * width != n_neurons:
        return []
    head_dim = width // heads
    names = {1: [""], 2: ["k", "v"], 3: ["q", "k", "v"]}[parts]
    return [{"label": f"{(names[p] + ' · ') if names[p] else ''}head {h}", "start": p * width + h * head_dim,
             "size": head_dim, "kind": names[p] or "head"}
            for p in range(parts) for h in range(heads)]


def _model_age(model) -> int:
    try:
        return int(model.get_age())
    except Exception:
        return 0


# A ReLU's zero output is a neuron that did not fire; these keep the channel
# (and the neuron index) unchanged on the way to it.
_RELU_TYPES: Tuple[type, ...] = (nn.ReLU, nn.ReLU6)
_CHANNEL_PRESERVING: Tuple[type, ...] = (
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.Identity,
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
    nn.modules.pooling._MaxPoolNd,
    nn.modules.pooling._AvgPoolNd,
    nn.modules.pooling._AdaptiveMaxPoolNd,
    nn.modules.pooling._AdaptiveAvgPoolNd,
)


def _downstream_relu(arch: dict, inner: nn.Module, node_id: str, max_hops: int = 6) -> str:
    """The ReLU node *node_id* feeds through channel-preserving layers only
    (a single consumer at each hop), or "" -- e.g. c1 -> b1 -> r1 gives "r1"."""
    successors: Dict[str, List[str]] = {}
    for edge in arch.get("edges", []):
        successors.setdefault(edge["source"], []).append(edge["target"])
    module_of = {n["id"]: n.get("module_id", n["id"]) for n in arch.get("nodes", [])}
    current = node_id
    for _ in range(max_hops):
        nxt = successors.get(current, [])
        if len(nxt) != 1 or nxt[0] in (OUTPUT_NODE, INPUT_NODE):
            return ""
        try:
            module = inner.get_submodule(module_of.get(nxt[0], nxt[0]))
        except Exception:
            return ""
        if isinstance(module, _RELU_TYPES):
            return nxt[0]
        if not isinstance(module, _CHANNEL_PRESERVING):
            return ""
        current = nxt[0]
    return ""


def _truncate_preview(preview: dict, tiles: int) -> dict:
    out = dict(preview)
    size = preview["height"] * preview["width"]
    out["tiles"] = tiles
    out["data"] = preview["data"][: tiles * size]
    out["tile_min"] = preview["tile_min"][:tiles]
    out["tile_max"] = preview["tile_max"][:tiles]
    out["channel_indices"] = preview["channel_indices"][:tiles]
    return out


# =============================================================================
# 3. Edits
# =============================================================================
def _rows(tensor: th.Tensor, axis: int, indices: Optional[Sequence[int]]):
    if indices is None:
        return slice(None)
    index = [slice(None)] * tensor.ndim
    index[axis] = list(indices)
    return tuple(index)


def edit_layer(model, node_id: str, op: str, neuron_indices: Sequence[int] = (),
               count: int = 1, ratio: float = 0.1,
               manager: ModelRepresentationManager = MANAGER) -> dict:
    """Apply *op* to the module behind *node_id*. Raises ValueError on bad input."""
    op = (op or "").strip().lower()
    if op not in EDIT_OPS:
        raise ValueError(f"unknown op {op!r}; expected one of {', '.join(EDIT_OPS)}")
    inner = resolve_inner(model)
    if inner is None:
        raise ValueError("No model registered")
    module_id = node_id.split("#", 1)[0]
    try:
        module = inner.get_submodule(module_id)
    except Exception:
        raise ValueError(f"unknown layer {node_id!r}")
    indices = sorted({int(i) for i in neuron_indices}) if neuron_indices else []
    tracked, layer_id = _is_tracked(model, module)
    is_group = _has_children(module)

    if op in ("prune", "add"):
        if not tracked:
            raise ValueError(
                f"{op} needs the model's layer dependencies: wrap it with "
                "wl.watch_or_edit(model, flag='model', compute_dependencies=True)")
        if op == "prune" and not indices:
            raise ValueError("prune needs at least one neuron index")
        # The architecture changes: checkpoints on BOTH sides of the edit must
        # stay restorable (architectures are not dumped by default). Nothing
        # is written now -- the checkpoint manager keeps the pre-edit model in
        # memory and saves both architectures when training resumes.
        _note_structural_edit(_checkpoint_manager(), model)
        if op == "prune":
            model.prune_neurons(layer_id, indices)
            message = f"pruned {len(indices)} neuron(s) from {module_id}"
        else:
            count = int(count or 1)
            model.add_neurons(layer_id, count=count)
            message = f"added {count} neuron(s) to {module_id}"
        manager.note_edit(structural=True)
        return {"message": message}

    if op in ("freeze", "unfreeze"):
        frozen = op == "freeze"
        if tracked and indices and not is_group:
            (model.freeze_neurons if frozen else model.unfreeze_neurons)(layer_id, indices)
            message = f"{op} {len(indices)} neuron(s) of {module_id}"
        elif indices and not is_group:
            # Any layer: the frozen rows are put back before every forward.
            weight = getattr(module, "weight", None)
            if not isinstance(weight, th.Tensor):
                raise ValueError(f"{module_id} has no weights")
            axis = _weight_out_axis(module) if weight.ndim > 1 else 0
            if max(indices) >= weight.shape[axis]:
                raise ValueError(f"neuron index {max(indices)} out of range for {module_id} "
                                 f"({weight.shape[axis]} neurons)")
            hook = FrozenRows.of(module, create=frozen)
            if hook is not None:
                (hook.set if frozen else hook.unset)(module, indices)
            message = f"{op} {len(indices)} neuron(s) of {module_id}"
        else:
            params = list(module.parameters())
            if not params:
                raise ValueError(f"{module_id} has no parameters")
            if not frozen and tracked:
                # Per-neuron freezes are learning rates of 0 -- lift them too.
                try:
                    model.unfreeze_neurons(layer_id, None)
                except Exception:
                    pass
            for param in params:
                param.requires_grad_(not frozen)
            if not frozen:
                # Unfreezing the layer lifts its per-neuron freezes too.
                for sub_module in ([module] if not is_group else list(module.modules())):
                    hook = FrozenRows.of(sub_module)
                    if hook is not None:
                        hook.unset(sub_module, None)
            message = f"{op} {module_id} ({sum(p.numel() for p in params):,} parameters)"
        manager.note_edit(structural=False)
        return {"message": message}

    # reset / perturb
    if is_group:
        targets = [m for m in module.modules() if isinstance(getattr(m, "weight", None), th.Tensor)
                   and not _has_children(m)]
        if indices:
            raise ValueError("neuron indices apply to a layer, not a group")
    else:
        targets = [module]
    if op == "perturb" and not (0.0 < float(ratio) < 1.0):
        raise ValueError("ratio must be strictly between 0 and 1")
    done = 0
    for target in targets:
        t_tracked, t_id = _is_tracked(model, target)
        if t_tracked:
            if op == "reset":
                model.reset_neurons(t_id, indices or None)
            else:
                model.perturb_neurons(t_id, indices or None, ratio=float(ratio))
            done += 1
            continue
        weight = getattr(target, "weight", None)
        if not isinstance(weight, th.Tensor):
            continue
        frozen_rows = FrozenRows.of(target)
        if frozen_rows is not None:
            frozen_rows.restore(target)       # edit the frozen values, not drifted ones
        axis = _weight_out_axis(target) if weight.ndim > 1 else 0
        if indices and max(indices) >= weight.shape[axis]:
            raise ValueError(f"neuron index {max(indices)} out of range for {module_id} "
                             f"({weight.shape[axis]} neurons)")
        with th.no_grad():
            if op == "reset":
                if not callable(getattr(target, "reset_parameters", None)):
                    raise ValueError(f"{type(target).__name__} cannot be reset (no reset_parameters)")
                if not indices:
                    target.reset_parameters()
                else:
                    fresh = copy.deepcopy(target)
                    fresh.reset_parameters()
                    weight[_rows(weight, axis, indices)] = fresh.weight.detach()[_rows(weight, axis, indices)]
                    bias = getattr(target, "bias", None)
                    if isinstance(bias, th.Tensor):
                        bias[indices] = fresh.bias.detach()[indices]
            else:
                sel = _rows(weight, axis, indices or None)
                w = weight[sel]
                weight[sel] = w + float(ratio) * w * th.randint_like(w, -1, 2).float()
                bias = getattr(target, "bias", None)
                if isinstance(bias, th.Tensor):
                    bsel = indices if indices else slice(None)
                    b = bias[bsel]
                    bias[bsel] = b + float(ratio) * b * th.randint_like(b, -1, 2).float()
        if frozen_rows is not None:
            frozen_rows.recapture(target)     # an explicit edit sticks, frozen or not
        done += 1
    if not done:
        raise ValueError(f"{module_id} has no weights to {op}")
    manager.note_edit(structural=False)
    what = f"{len(indices)} neuron(s) of {module_id}" if indices else module_id
    return {"message": f"{op} {what}" + (f" (ratio {ratio:g})" if op == "perturb" else "")}


def _checkpoint_manager():
    """The registered CheckpointManager, or None (tests, model-only scripts)."""
    try:
        from weightslab.backend import ledgers
        manager = ledgers.get_checkpoint_manager()
        if manager is None or not callable(getattr(manager, "save_model_architecture", None)):
            return None
        return manager
    except Exception:
        return None


def _note_structural_edit(checkpoints, model) -> None:
    """Tell the checkpoint manager a structural edit is about to happen."""
    if checkpoints is None:
        return
    try:
        note = getattr(checkpoints, "note_structural_edit", None)
        if callable(note):
            note(model)
        else:   # an older manager: ask for the dump at the next model change
            checkpoints.request_architecture_dump()
    except Exception as exc:
        logger.warning(f"[modeling] could not record the structural edit: {exc}")
