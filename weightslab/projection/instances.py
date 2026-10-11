"""Instance-level projections: one point per BOX, not one per image.

A per-image projection answers "which images look alike". On a detection set
the question is usually about objects: are the boxes labelled ``truck`` sitting
in the middle of the ``car`` cluster? This module places every box -- every
labelled one and, optionally, every confident prediction with no label under
it -- in the same 2-D/3-D map, train and validation together.

How a box becomes a point
-------------------------
One feature vector per box, read from the model's own multi-scale feature maps
(for a YOLO model: the P3/P4/P5 maps the detection head reads). The box is
pooled out of EVERY level with ``torchvision.ops.roi_align`` (no crops, no
second model), each level is L2-normalised and the levels are concatenated.

Pooling from every level, rather than only from the level that "fits" the box,
keeps the vectors in one space: with one level per box size, a 63-px car and a
65-px car would be read from two different feature maps and the map would
cluster by box size before class. Labelled and predicted boxes are pooled the
same way, so an unlabelled car the model found lands among the labelled cars.

Then predictions are matched to labels (IoU > ``match_iou``, one to one,
class-agnostic, best IoU first), giving each point a *match* category:

==================  ==============================================================
``consistent``      a label, and a prediction of the same class on it
``wrong class``     a label, and a prediction of ANOTHER class on it (a truck
                    labelled car, or a car predicted truck)
``no prediction``   a label nothing was predicted on (spurious, or shifted box)
``missing label``   a confident prediction with no label under it (point kind
                    ``prediction``; only with ``include_predictions``)
==================  ==============================================================

Matched predictions are not points of their own: they would sit on top of
their label. Their class, confidence and IoU are carried by the label's point.

Where it is stored
------------------
Not in the per-sample dataframe -- predictions have no row there, and an image
with 40 boxes would need 40 rows of coordinates. Each projection is a table
beside the run's logs, ``<root_log_dir>/projection/instances/<prefix>.npz``,
and its prefix is registered with ``level="instance"`` so the Projection Board
offers it beside the per-image ones and draws one point per box.

Use
---
>>> wl.project_instances(model, {"train": train_loader, "val": val_loader})
>>> # or bring your own coordinates:
>>> wl.save_instance_projection_coords(coords, sample_ids, annotation_ids,
...                                    prefix="boxes_tsne", class_ids=cls)

Use the features of a model that has learnt something (a pretrained or an
early-epoch checkpoint): a randomly initialised network places boxes by colour
and size, not by what they are.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)

LEVEL = "instance"
DEFAULT_PREFIX = "instances"
INSTANCE_DIR = "instances"

#: Point kinds.
LABEL = "label"
PREDICTION = "prediction"

#: Match categories (see the module docstring).
CONSISTENT = "consistent"
WRONG_CLASS = "wrong class"
NO_PREDICTION = "no prediction"
MISSING_LABEL = "missing label"
MATCHES = (CONSISTENT, WRONG_CLASS, NO_PREDICTION, MISSING_LABEL)


# --------------------------------------------------------------------- matching
def box_iou(a, b) -> np.ndarray:
    """IoU of every box in *a* ``(n, 4)`` with every box in *b* ``(m, 4)``,
    both ``x1, y1, x2, y2``. ``(n, m)``; empty when either is empty."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def match_boxes(gt_boxes, pred_boxes, iou_threshold: float = 0.5):
    """One-to-one matching of predictions to labels.

    Greedy on IoU, best pair first, and class-AGNOSTIC: a truck predicted on a
    box labelled car must still match it, or the "wrong class" case could never
    be seen. Pairs need IoU strictly above *iou_threshold*.

    Returns ``(gt_to_pred, pred_to_gt, gt_iou)``: the matched index on the
    other side or -1, and each label's IoU with its match (NaN when unmatched).
    """
    iou = box_iou(gt_boxes, pred_boxes)
    n, m = iou.shape
    gt_to_pred = np.full(n, -1, dtype=np.int64)
    pred_to_gt = np.full(m, -1, dtype=np.int64)
    gt_iou = np.full(n, np.nan, dtype=np.float32)
    if not n or not m:
        return gt_to_pred, pred_to_gt, gt_iou
    rows, cols = np.nonzero(iou > iou_threshold)
    if not len(rows):
        return gt_to_pred, pred_to_gt, gt_iou
    # Stable sort, so equal IoUs resolve in a fixed (label, prediction) order.
    order = np.argsort(-iou[rows, cols], kind="stable")
    for k in order:
        g, p = rows[k], cols[k]
        if gt_to_pred[g] >= 0 or pred_to_gt[p] >= 0:
            continue
        gt_to_pred[g] = p
        pred_to_gt[p] = g
        gt_iou[g] = iou[g, p]
    return gt_to_pred, pred_to_gt, gt_iou


def match_categories(gt_classes, pred_classes, gt_to_pred, pred_to_gt):
    """``(label_categories, prediction_categories)`` for one image.

    A label is ``consistent`` / ``wrong class`` / ``no prediction``; a
    prediction takes its label's category when matched, else ``missing label``.
    """
    gt_classes = np.asarray(gt_classes).reshape(-1)
    pred_classes = np.asarray(pred_classes).reshape(-1)
    labels = []
    for g, p in enumerate(np.asarray(gt_to_pred)):
        if p < 0:
            labels.append(NO_PREDICTION)
        elif int(pred_classes[p]) == int(gt_classes[g]):
            labels.append(CONSISTENT)
        else:
            labels.append(WRONG_CLASS)
    predictions = [labels[g] if g >= 0 else MISSING_LABEL for g in np.asarray(pred_to_gt)]
    return labels, predictions


# --------------------------------------------------------------------- features
def roi_features(feature_maps, strides, rois, output_size: int = 2):
    """One vector per box, pooled from every feature map.

    *feature_maps*: ``[(B, C_l, H_l, W_l), ...]``; *strides*: the input pixels
    per cell of each; *rois*: ``(K, 5)`` rows ``(batch_index, x1, y1, x2, y2)``
    in input pixels. Returns ``(K, sum C_l)``: each level average-pooled over
    the box with ``roi_align``, L2-normalised, and the levels concatenated --
    scaled by ``1/sqrt(levels)`` so the whole vector has unit norm.
    """
    import torch as th
    from torchvision.ops import roi_align

    if rois.shape[0] == 0:
        width = int(sum(int(f.shape[1]) for f in feature_maps))
        return th.zeros((0, width), dtype=th.float32)
    parts = []
    for fmap, stride in zip(feature_maps, strides):
        boxes = rois.to(device=fmap.device, dtype=th.float32)
        pooled = roi_align(fmap.float(), boxes, output_size=(output_size, output_size),
                           spatial_scale=1.0 / float(stride), sampling_ratio=2, aligned=True)
        vec = pooled.mean(dim=(2, 3))
        parts.append(th.nn.functional.normalize(vec, dim=1))
    out = th.cat(parts, dim=1) / float(np.sqrt(len(parts)))
    return out.detach().float().cpu()


# ----------------------------------------------------------------------- table
@dataclass
class InstanceTable:
    """Every point of one instance-level projection, as parallel arrays.

    ``instance_ids`` names a point inside its sample: a label is its
    ``annotation_id`` (1-based, the order of the sample's target boxes -- the
    same id its instance row in the dataframe carries); a prediction is
    ``-(k + 1)``, *k* its rank among the sample's kept predictions.
    """

    sample_ids: np.ndarray
    instance_ids: np.ndarray
    class_ids: np.ndarray
    pred_class_ids: np.ndarray
    confidences: np.ndarray
    ious: np.ndarray
    matches: np.ndarray
    boxes: np.ndarray
    origins: np.ndarray
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    class_names: list = field(default_factory=list)
    layer: str = ""
    layer_detail: str = ""

    def __len__(self) -> int:
        return int(len(self.sample_ids))

    @property
    def kinds(self) -> np.ndarray:
        return np.where(np.asarray(self.instance_ids) > 0, LABEL, PREDICTION).astype(object)

    @property
    def dims(self) -> int:
        return int(self.coords.shape[1]) if self.coords.ndim == 2 else 0

    def class_name(self, class_id) -> str:
        cid = int(class_id)
        if cid < 0:
            return ""
        if 0 <= cid < len(self.class_names) and self.class_names[cid]:
            return str(self.class_names[cid])
        return str(cid)

    def class_labels(self, ids) -> np.ndarray:
        """Class names for an array of class ids ("" for -1)."""
        ids = np.asarray(ids, dtype=np.int64)
        table = {int(c): self.class_name(c) for c in np.unique(ids)} if len(ids) else {}
        return np.array([table[int(c)] for c in ids], dtype=object)

    @classmethod
    def empty(cls) -> "InstanceTable":
        return cls.concat([])

    @classmethod
    def concat(cls, parts: list) -> "InstanceTable":
        """One table out of several (same class names assumed)."""
        def cat(name, dtype, shape=()):
            arrays = [np.asarray(getattr(p, name)) for p in parts]
            if not arrays:
                return np.zeros((0,) + shape, dtype=dtype)
            return np.concatenate(arrays).astype(dtype, copy=False)
        first = parts[0] if parts else None
        # Coordinates only when every part has them (collected parts have none yet).
        placed = bool(parts) and all(
            np.asarray(p.coords).ndim == 2 and len(p.coords) == len(p) for p in parts)
        coords = (np.concatenate([np.asarray(p.coords, np.float32) for p in parts])
                  if placed else np.zeros((0, 3), np.float32))
        return cls(
            sample_ids=cat("sample_ids", object),
            instance_ids=cat("instance_ids", np.int64),
            class_ids=cat("class_ids", np.int32),
            pred_class_ids=cat("pred_class_ids", np.int32),
            confidences=cat("confidences", np.float32),
            ious=cat("ious", np.float32),
            matches=cat("matches", object),
            boxes=cat("boxes", np.float32, (4,)).reshape(-1, 4),
            origins=cat("origins", object),
            coords=coords,
            class_names=list(first.class_names) if first is not None else [],
            layer=first.layer if first is not None else "",
            layer_detail=first.layer_detail if first is not None else "",
        )

    def take(self, keep) -> "InstanceTable":
        keep = np.asarray(keep)
        return InstanceTable(
            sample_ids=self.sample_ids[keep], instance_ids=self.instance_ids[keep],
            class_ids=self.class_ids[keep], pred_class_ids=self.pred_class_ids[keep],
            confidences=self.confidences[keep], ious=self.ious[keep],
            matches=self.matches[keep], boxes=self.boxes[keep], origins=self.origins[keep],
            coords=self.coords[keep] if len(self.coords) == len(self) else self.coords,
            class_names=list(self.class_names), layer=self.layer, layer_detail=self.layer_detail,
        )

    # ---------------------------------------------------------------- storage
    def save(self, path: str) -> None:
        """Write to *path* (``.npz``), atomically: the board may be reading it."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.npz"
        meta = {"class_names": [str(c) for c in self.class_names],
                "layer": self.layer or "", "layer_detail": self.layer_detail or ""}
        np.savez_compressed(
            tmp,
            sample_ids=np.asarray(self.sample_ids, dtype=str),
            instance_ids=np.asarray(self.instance_ids, dtype=np.int64),
            class_ids=np.asarray(self.class_ids, dtype=np.int32),
            pred_class_ids=np.asarray(self.pred_class_ids, dtype=np.int32),
            confidences=np.asarray(self.confidences, dtype=np.float32),
            ious=np.asarray(self.ious, dtype=np.float32),
            matches=np.asarray(self.matches, dtype=str),
            boxes=np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4),
            origins=np.asarray(self.origins, dtype=str),
            coords=np.asarray(self.coords, dtype=np.float32),
            meta=np.asarray(json.dumps(meta)),
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "InstanceTable":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
            return cls(
                sample_ids=data["sample_ids"].astype(object),
                instance_ids=data["instance_ids"].astype(np.int64),
                class_ids=data["class_ids"].astype(np.int32),
                pred_class_ids=data["pred_class_ids"].astype(np.int32),
                confidences=data["confidences"].astype(np.float32),
                ious=data["ious"].astype(np.float32),
                matches=data["matches"].astype(object),
                boxes=data["boxes"].astype(np.float32).reshape(-1, 4),
                origins=data["origins"].astype(object),
                coords=data["coords"].astype(np.float32),
                class_names=list(meta.get("class_names") or []),
                layer=str(meta.get("layer") or ""),
                layer_detail=str(meta.get("layer_detail") or ""),
            )


def table_path(prefix: str, root_log_dir=None) -> Optional[str]:
    """``<root_log_dir>/projection/instances/<prefix>.npz``, or None without a root."""
    from weightslab.projection.registry import projection_dir
    folder = projection_dir(root_log_dir)
    if not folder:
        return None
    return os.path.join(folder, INSTANCE_DIR, f"{prefix}.npz")


_CACHE_LOCK = threading.Lock()
# path -> ((mtime, size), table): the board polls every few seconds, and a
# table of a million boxes takes a moment to decompress.
_TABLE_CACHE: dict = {}


def load_instance_table(prefix: str, root_log_dir=None) -> Optional[InstanceTable]:
    """The stored table of *prefix*, or None when there is none. Cached by the
    file's (mtime, size), so an unchanged file is read once."""
    path = table_path(prefix, root_log_dir)
    if not path:
        return None
    try:
        stamp = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None
    with _CACHE_LOCK:
        cached = _TABLE_CACHE.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
    try:
        table = InstanceTable.load(path)
    except Exception as exc:
        logger.warning(f"[projection] unreadable instance projection {path}: {exc}")
        return None
    with _CACHE_LOCK:
        _TABLE_CACHE[path] = (stamp, table)
    return table


def table_stamp(prefix: str, root_log_dir=None):
    """``(mtime, size)`` of *prefix*'s table, or None: what changes when it is
    rewritten (the service rebuilds its frames on a new stamp)."""
    path = table_path(prefix, root_log_dir)
    if not path:
        return None
    try:
        return (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return None


def write_instance_table(table: InstanceTable, prefix: str, root_log_dir=None) -> str:
    """Store *table* under *prefix* and register the prefix as instance-level."""
    from weightslab.projection.custom import validate_prefix, check_not_live_prefix
    from weightslab.projection.registry import register_prefix

    prefix = validate_prefix(prefix)
    check_not_live_prefix(prefix)
    path = table_path(prefix, root_log_dir)
    if not path:
        raise RuntimeError(
            "save_instance_projection_coords: no root_log_dir to store the projection "
            "under -- register hyperparameters with root_log_dir (or set "
            "WEIGHTSLAB_ROOT_LOG_DIR) first.")
    table.save(path)
    register_prefix(prefix, root_log_dir=root_log_dir, layer=table.layer or None,
                    layer_detail=table.layer_detail or None, level=LEVEL)
    with _CACHE_LOCK:
        _TABLE_CACHE.pop(path, None)
    return path


def is_instance_prefix(prefix: str, root_log_dir=None) -> bool:
    """Whether *prefix* is an instance-level projection."""
    try:
        from weightslab.projection.registry import prefix_info
        return prefix_info(prefix, root_log_dir).get("level") == LEVEL
    except Exception:
        return False


def instance_prefixes(root_log_dir=None) -> list:
    """Every registered instance-level prefix whose table exists, sorted."""
    try:
        from weightslab.projection.registry import known_prefixes, prefix_info
        names = known_prefixes(root_log_dir) or set()
    except Exception:
        return []
    found = []
    for name in sorted(names):
        try:
            if prefix_info(name, root_log_dir).get("level") != LEVEL:
                continue
        except Exception:
            continue
        if table_stamp(name, root_log_dir) is not None:
            found.append(name)
    return found


def _as_list(values, n=None, name="values"):
    if values is None:
        return None
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if n is not None and len(array) != n:
        raise ValueError(f"{name} has {len(array)} entries but there are {n} points.")
    return array


def save_instance_projection_coords(coords, sample_ids, annotation_ids, prefix: str, *,
                                    class_ids=None, class_names=None, matches=None,
                                    boxes=None, confidences=None, pred_class_ids=None,
                                    ious=None, origins=None, root_log_dir=None) -> str:
    """Store an instance-level projection you computed yourself.

    One point per box: ``coords[i]`` places the box ``annotation_ids[i]`` of
    sample ``sample_ids[i]``. An annotation id is 1-based -- the box's position
    in the sample's target list, which is also the id of its instance row. Use
    a NEGATIVE id for a prediction with no label (``-1``, ``-2``... per sample).

    Everything else is optional and only feeds the board's colours, legend and
    box highlight: ``class_ids`` (+ ``class_names``), ``matches`` (one of
    ``"consistent"``, ``"wrong class"``, ``"no prediction"``, ``"missing
    label"``), ``boxes`` (``(N, 4)`` x1, y1, x2, y2 normalised to the image),
    ``confidences``, ``pred_class_ids``, ``ious``, ``origins`` (split names).

    Writing a prefix again replaces the whole projection.

    Returns:
        str: the file written.
    """
    ids = [str(s) for s in (sample_ids.detach().cpu().tolist()
                            if hasattr(sample_ids, "detach") else sample_ids)]
    n = len(ids)
    from weightslab.projection.custom import as_coordinates
    array = as_coordinates(coords, n=n)
    if array.shape[1] == 2:
        array = np.concatenate([array, np.zeros((n, 1), np.float32)], axis=1)
    ann = _as_list(annotation_ids, n, "annotation_ids").astype(np.int64)
    cls = _as_list(class_ids, n, "class_ids")
    pcls = _as_list(pred_class_ids, n, "pred_class_ids")
    table = InstanceTable(
        sample_ids=np.asarray(ids, dtype=object),
        instance_ids=ann,
        class_ids=(cls.astype(np.int32) if cls is not None else np.full(n, -1, np.int32)),
        pred_class_ids=(pcls.astype(np.int32) if pcls is not None else np.full(n, -1, np.int32)),
        confidences=(_as_list(confidences, n, "confidences").astype(np.float32)
                     if confidences is not None else np.full(n, np.nan, np.float32)),
        ious=(_as_list(ious, n, "ious").astype(np.float32)
              if ious is not None else np.full(n, np.nan, np.float32)),
        matches=(np.asarray([str(m) for m in _as_list(matches, n, "matches")], dtype=object)
                 if matches is not None else np.full(n, "", dtype=object)),
        boxes=(_as_list(boxes, n, "boxes").astype(np.float32).reshape(-1, 4)
               if boxes is not None else np.full((n, 4), np.nan, np.float32)),
        origins=(np.asarray([str(o) for o in _as_list(origins, n, "origins")], dtype=object)
                 if origins is not None else np.full(n, "", dtype=object)),
        coords=array,
        class_names=_normalise_class_names(class_names),
    )
    path = write_instance_table(table, prefix, root_log_dir=root_log_dir)
    placed = int(np.isfinite(array).all(axis=1).sum())
    logger.info(f"[projection] '{prefix}' (instances): {placed} of {n} boxes placed -> {path}")
    return path


def _normalise_class_names(names) -> list:
    """Class names as a list indexed by class id (dicts are {id: name})."""
    if names is None:
        return []
    if isinstance(names, dict) or hasattr(names, "items"):
        items = {int(k): str(v) for k, v in dict(names).items()}
        if not items:
            return []
        out = [""] * (max(items) + 1)
        for k, v in items.items():
            if k >= 0:
                out[k] = v
        return out
    return [str(n) for n in names]


# ------------------------------------------------------------------ collection
@dataclass
class InstanceBatch:
    """One batch as :func:`project_instances` needs it.

    *inputs* goes to the model as is. *gt* holds, per image, the labelled boxes
    in input pixels ``(n, 4)`` x1, y1, x2, y2, their class ids ``(n,)`` and
    their annotation ids ``(n,)`` (1-based target order).
    """

    inputs: Any
    sample_ids: list
    gt: list
    image_size: tuple  # (height, width) of the model input, in pixels


@dataclass
class InstanceForward:
    """What one forward pass hands back: the feature maps (and their strides)
    boxes are pooled from, and per image the kept predictions -- boxes in input
    pixels ``(m, 4)``, confidences ``(m,)`` and class ids ``(m,)``."""

    feature_maps: list
    strides: list
    predictions: list


class InstanceAdapter:
    """What :func:`project_instances` needs to know about a detection model and
    its data. Subclass for a framework it does not know; the Ultralytics one is
    :class:`weightslab.integrations.ultralytics.instances.UltralyticsInstanceAdapter`.
    """

    def batches(self, loader) -> Iterator[InstanceBatch]:
        raise NotImplementedError

    def forward(self, model, batch: InstanceBatch, *, conf: float) -> InstanceForward:
        raise NotImplementedError

    def class_names(self, model, loaders) -> list:
        return []

    def layer(self, model) -> tuple:
        """``(named_modules() name, detail)`` of where features are read."""
        return "", ""


def _origin_of(name, loader) -> str:
    """The split name a loader's samples belong to."""
    if name:
        return str(name)
    for attr in ("_ledger_name", "loader_name", "name"):
        try:
            value = getattr(loader, attr, None)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value
    return ""


def _named_loaders(loaders) -> list:
    """``[(origin, loader), ...]`` from a loader, a list of them or a dict."""
    if isinstance(loaders, dict):
        return [(str(k), v) for k, v in loaders.items() if v is not None]
    if isinstance(loaders, (list, tuple)):
        return [(None, v) for v in loaders if v is not None]
    return [(None, loaders)]


def collect_instances(model, loaders, adapter: InstanceAdapter, *, conf: float = 0.25,
                      match_iou: float = 0.5, include_predictions: bool = True,
                      max_instances: Optional[int] = None, verbose: bool = True):
    """Sweep *loaders* through *model*: ``(features (N, F) tensor, table)``.

    The table has no coordinates yet; the rows of the feature matrix are its
    points, in order.
    """
    import torch as th
    from weightslab.projection.parametric_umap import _frozen_age, resolve_module

    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - tqdm is a dependency
        tqdm = None

    module = resolve_module(model) or model
    was_training = bool(getattr(module, "training", False))
    if hasattr(module, "eval"):
        module.eval()
    feature_chunks, parts = [], []
    collected = 0
    try:
        with th.no_grad(), _frozen_age(model):
            for name, loader in _named_loaders(loaders):
                origin = _origin_of(name, loader)
                iterator = adapter.batches(loader)
                if tqdm is not None:
                    iterator = tqdm(iterator, desc=f"[projection] boxes {origin}".rstrip(),
                                    unit="batch", disable=not verbose, leave=False)
                for batch in iterator:
                    out = adapter.forward(model, batch, conf=conf)
                    feats, part = _instances_of_batch(batch, out, origin, match_iou,
                                                      include_predictions)
                    if part is None:
                        continue
                    feature_chunks.append(feats)
                    parts.append(part)
                    collected += len(part)
                    if max_instances is not None and collected >= max_instances:
                        break
                if max_instances is not None and collected >= max_instances:
                    break
    finally:
        if was_training and hasattr(module, "train"):
            module.train()
    if not parts:
        return th.zeros((0, 0)), InstanceTable.empty()
    feats = th.cat(feature_chunks, dim=0)
    table = InstanceTable.concat(parts)
    if max_instances is not None and len(table) > max_instances:
        feats = feats[:max_instances]
        table = table.take(np.arange(max_instances))
    names = adapter.class_names(model, loaders)
    table.class_names = _normalise_class_names(names)
    table.layer, table.layer_detail = adapter.layer(model)
    return feats, table


def _instances_of_batch(batch: InstanceBatch, out: InstanceForward, origin: str,
                        match_iou: float, include_predictions: bool):
    """Features and table rows of one batch."""
    import torch as th

    height, width = batch.image_size
    scale = np.array([width, height, width, height], dtype=np.float32)
    rois, rows = [], []
    for b, sid in enumerate(batch.sample_ids):
        gt_boxes, gt_cls, gt_ann = batch.gt[b]
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        gt_cls = np.asarray(gt_cls, dtype=np.int64).reshape(-1)
        gt_ann = np.asarray(gt_ann, dtype=np.int64).reshape(-1)
        p_boxes, p_conf, p_cls = out.predictions[b]
        p_boxes = np.asarray(p_boxes, dtype=np.float32).reshape(-1, 4)
        p_conf = np.asarray(p_conf, dtype=np.float32).reshape(-1)
        p_cls = np.asarray(p_cls, dtype=np.int64).reshape(-1)

        g2p, p2g, g_iou = match_boxes(gt_boxes, p_boxes, match_iou)
        label_cats, pred_cats = match_categories(gt_cls, p_cls, g2p, p2g)
        for g in range(len(gt_boxes)):
            p = g2p[g]
            rows.append((sid, int(gt_ann[g]), int(gt_cls[g]),
                         int(p_cls[p]) if p >= 0 else -1,
                         float(p_conf[p]) if p >= 0 else np.nan,
                         float(g_iou[g]), label_cats[g], gt_boxes[g]))
            rois.append([b, *gt_boxes[g].tolist()])
        if include_predictions:
            k = 0
            for p in range(len(p_boxes)):
                if p2g[p] >= 0:
                    continue
                k += 1
                rows.append((sid, -k, int(p_cls[p]), int(p_cls[p]), float(p_conf[p]),
                             np.nan, pred_cats[p], p_boxes[p]))
                rois.append([b, *p_boxes[p].tolist()])
    if not rows:
        return None, None
    roi_tensor = th.tensor(rois, dtype=th.float32)
    # Degenerate boxes (zero width/height) still pool a value with aligned=True.
    feats = roi_features(out.feature_maps, out.strides, roi_tensor)
    boxes = np.stack([r[7] for r in rows]).astype(np.float32) / scale
    table = InstanceTable(
        sample_ids=np.array([str(r[0]) for r in rows], dtype=object),
        instance_ids=np.array([r[1] for r in rows], dtype=np.int64),
        class_ids=np.array([r[2] for r in rows], dtype=np.int32),
        pred_class_ids=np.array([r[3] for r in rows], dtype=np.int32),
        confidences=np.array([r[4] for r in rows], dtype=np.float32),
        ious=np.array([r[5] for r in rows], dtype=np.float32),
        matches=np.array([r[6] for r in rows], dtype=object),
        boxes=np.clip(boxes, 0.0, 1.0),
        origins=np.full(len(rows), origin, dtype=object),
    )
    return feats, table


# ---------------------------------------------------------------------- fitting
def fit_coordinates(features, method="umap", *, out_dim: int = 3, epochs: int = 10,
                    fit_batch: int = 512, n_neighbors: int = 15, lr: float = 1e-3,
                    min_dist: float = 0.1, spread: float = 1.0, device=None,
                    verbose: bool = True) -> np.ndarray:
    """``(N, F)`` features -> ``(N, out_dim)`` coordinates.

    *method*: ``"umap"`` (WeightsLab's parametric UMAP, fitted offline),
    ``"pca"`` (fast, deterministic), or your own -- an object with
    ``fit_transform`` or a callable ``features -> coords``.
    """
    import torch as th
    from weightslab.projection import custom as _custom

    feats = features if isinstance(features, th.Tensor) else th.as_tensor(np.asarray(features))
    feats = feats.float()
    if isinstance(method, str):
        name = method.strip().lower()
        if name == "pca":
            return _pca(feats, out_dim)
        if name == "umap":
            from weightslab.projection.parametric_umap import fit_umap_offline
            return fit_umap_offline(feats, out_dim=out_dim, epochs=epochs, fit_batch=fit_batch,
                                    n_neighbors=n_neighbors, lr=lr, min_dist=min_dist,
                                    spread=spread, device=device, verbose=verbose)
        raise ValueError(
            f"method={method!r} is not built in: pass 'umap', 'pca', a callable "
            f"features -> coords, or an object with fit_transform (e.g. TSNE).")
    return _custom.run_method(method, feats.numpy())


def _pca(feats, out_dim: int) -> np.ndarray:
    import torch as th
    n, f = feats.shape
    k = max(1, min(out_dim, n, f))
    centred = feats - feats.mean(dim=0, keepdim=True)
    # Exact SVD: deterministic, and the matrix is (N, ~500) at most.
    _, _, v = th.linalg.svd(centred, full_matrices=False)
    coords = (centred @ v[:k].T).numpy().astype(np.float32)
    if coords.shape[1] < out_dim:
        coords = np.concatenate(
            [coords, np.zeros((n, out_dim - coords.shape[1]), np.float32)], axis=1)
    return coords


# ------------------------------------------------------------------------ API
def resolve_adapter(model, adapter=None) -> InstanceAdapter:
    """*adapter*, or the one that knows *model* (Ultralytics YOLO for now)."""
    if adapter is not None:
        return adapter
    from weightslab.projection.parametric_umap import resolve_module
    module = resolve_module(model) or model
    for sub in getattr(module, "modules", lambda: [])():
        if type(sub).__name__ in ("Detect", "Segment", "v10Detect", "YOLOEDetect"):
            from weightslab.integrations.ultralytics.instances import UltralyticsInstanceAdapter
            return UltralyticsInstanceAdapter()
    raise TypeError(
        "project_instances: no adapter knows this model. Pass adapter=<InstanceAdapter> "
        "(see weightslab.projection.instances), or compute the coordinates yourself and "
        "store them with wl.save_instance_projection_coords(...).")


def project_instances(model, loaders, *, prefix: str = DEFAULT_PREFIX, method="umap",
                      adapter: Optional[InstanceAdapter] = None, conf: float = 0.25,
                      match_iou: float = 0.5, include_predictions: bool = True,
                      max_instances: Optional[int] = None, out_dim: int = 3,
                      epochs: int = 10, fit_batch: int = 512, n_neighbors: int = 15,
                      lr: float = 1e-3, min_dist: float = 0.1, spread: float = 1.0,
                      device=None, root_log_dir=None, verbose: bool = True) -> dict:
    """Project every BOX of a detection set: one point per labelled box, and one
    per confident prediction with no label (see the module docstring).

    Args:
        model: the detection model (or the handle ``wl.watch_or_edit`` returned).
            Used in ``eval()`` under ``no_grad``; its weights and its age are
            left alone.
        loaders: a loader, a list of loaders, or ``{"train": ..., "val": ...}``
            -- train and validation boxes go on the same map. The dict keys
            name the splits when the loaders do not.
        prefix: name of the projection in the board's picker.
        method: ``"umap"`` (default), ``"pca"``, or your own (``TSNE(...)``,
            a callable ``features -> coords``).
        adapter: how to run the model and read its boxes; auto-detected for
            Ultralytics YOLO.
        conf: confidence a prediction needs to count.
        match_iou: IoU above which a prediction matches a label.
        include_predictions: also place predictions with no label under them
            (``missing label`` points).
        max_instances: stop collecting after this many boxes.

    Returns:
        dict: ``{"prefix", "instances", "labels", "predictions", "samples",
        "matches": {category: count}, "feature_dim", "method", "path"}``.
    """
    from weightslab.projection.custom import validate_prefix, check_not_live_prefix

    prefix = validate_prefix(prefix)
    check_not_live_prefix(prefix)
    adapter = resolve_adapter(model, adapter)
    feats, table = collect_instances(model, loaders, adapter, conf=conf, match_iou=match_iou,
                                     include_predictions=include_predictions,
                                     max_instances=max_instances, verbose=verbose)
    if len(table) < 4:
        raise RuntimeError(
            f"project_instances: only {len(table)} boxes collected; need at least 4. "
            f"Do the loaders yield labelled images, and is conf={conf} reachable?")
    name = method if isinstance(method, str) else getattr(method, "__name__", type(method).__name__)
    if verbose:
        logger.info(f"[projection] {name} on {len(table)} boxes x {feats.shape[1]} features "
                    f"-> '{prefix}'")
    coords = fit_coordinates(feats, method, out_dim=out_dim, epochs=epochs, fit_batch=fit_batch,
                             n_neighbors=n_neighbors, lr=lr, min_dist=min_dist, spread=spread,
                             device=device, verbose=verbose)
    if coords.shape[1] == 2:
        coords = np.concatenate([coords, np.zeros((len(coords), 1), np.float32)], axis=1)
    table.coords = coords.astype(np.float32)
    path = write_instance_table(table, prefix, root_log_dir=root_log_dir)
    kinds = table.kinds
    counts = {m: int((table.matches == m).sum()) for m in MATCHES}
    stats = {
        "prefix": prefix,
        "instances": len(table),
        "labels": int((kinds == LABEL).sum()),
        "predictions": int((kinds == PREDICTION).sum()),
        "samples": int(len(set(table.sample_ids.tolist()))),
        "matches": counts,
        "feature_dim": int(feats.shape[1]),
        "method": str(name),
        "path": path,
    }
    if verbose:
        logger.info(f"[projection] instance projection '{prefix}' written: "
                    f"{stats['labels']} labels + {stats['predictions']} unlabelled predictions "
                    f"over {stats['samples']} images; {counts}")
    return stats
