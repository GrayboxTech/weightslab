"""Collect bounding-box/segmentation annotations from the registered dataframe
into the format-agnostic ``ImageAnnotations`` intermediate representation.

Two real architectural gaps drive most of the "best effort" logic here (see
docs/export.rst for the user-facing caveat):

- There is no dedicated class-id -> class-name registry. We fall back to a
  ``dataset.class_names`` attribute (the same fallback ``data_service.py``
  already uses for the studio UI), or an explicit ``class_names`` override.
- There is no per-sample stored image path/dimensions. We best-effort resolve
  a path via a few common dataset attribute names, and fall back to a
  segmentation mask's own shape, or a caller-supplied default size.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from weightslab.data.sample_stats import SampleStats
from weightslab.export.models import BoxAnnotation, ImageAnnotations, PolygonAnnotation

logger = logging.getLogger(__name__)

SID = SampleStats.Ex.SAMPLE_ID.value
ORIGIN = SampleStats.Ex.ORIGIN.value
TARGET = SampleStats.Ex.TARGET.value
PREDICTION = SampleStats.Ex.PREDICTION.value

_PATH_ATTR_CANDIDATES = ("image_paths", "img_files", "images", "imgs", "files", "samples")

_warned_no_cv2 = False


def _is_bbox_array(value: Any) -> bool:
    """Mirrors ``LedgeredDataFrameManager._is_bbox_array``: a 2-D array whose
    last dim is 4..6 is bbox coordinates; dense (H, W) masks have a much
    larger trailing dim and never collide with this shape.
    """
    try:
        arr = np.asanyarray(value)
    except Exception:
        return False
    return arr.ndim == 2 and arr.shape[-1] in range(3, 10)


def _boxes_from_cell(value: Any) -> List[Tuple[float, float, float, float, Optional[int]]]:
    """Flatten one target/prediction cell into (x1, y1, x2, y2, cls_id) tuples.

    Handles a single box (1-D, len 4..6), or multiple boxes stacked in one
    cell (2-D, last dim 4..6). Row layout is ``(x1, y1, x2, y2[, conf][, cls])``
    -- class id is the last column when the row is 5 or 6 wide.
    """
    if value is None:
        return []
    try:
        arr = np.asanyarray(value)
    except Exception:
        return []
    if arr.size == 0:
        return []

    if arr.ndim == 1 and arr.shape[0] in (4, 5, 6):
        rows = [arr]
    elif arr.ndim == 2 and arr.shape[-1] in range(4, 7):
        rows = list(arr)
    else:
        return []

    boxes = []
    for row in rows:
        row = np.asanyarray(row, dtype=float)
        cls_col = 4 if row.shape[0] >= 6 else -1
        cls_id = int(round(row[cls_col])) if row.shape[0] >= 5 else None
        boxes.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), cls_id))
    return boxes


def _mask_to_polygons(mask: np.ndarray, class_id: int) -> List[List[Tuple[float, float]]]:
    """Extract polygon contours for one class id from a dense (H, W) mask.

    Requires OpenCV (lazily imported) -- install with ``pip install
    weightslab[export]``. Callers should catch ``ImportError`` and degrade to
    bbox-only export rather than let this abort the whole run.
    """
    import cv2 # raises ImportError if the [export] extra isn't installed

    binary = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygons.append([(float(pt[0][0]), float(pt[0][1])) for pt in contour])
    return polygons


def _polygons_from_mask_cell(value: Any, class_names) -> List[PolygonAnnotation]:
    global _warned_no_cv2
    try:
        arr = np.asanyarray(value)
    except Exception:
        return []
    if arr.ndim != 2 or arr.size == 0 or _is_bbox_array(arr):
        return []

    class_ids = sorted(int(c) for c in np.unique(arr) if c != 0)
    if not class_ids:
        return []

    polygons = []
    for cid in class_ids:
        label = _label_for_class(cid, class_names)
        try:
            for points in _mask_to_polygons(arr, cid):
                polygons.append(PolygonAnnotation(points=points, label=label))
        except ImportError:
            if not _warned_no_cv2:
                logger.warning(
                    "[export] OpenCV not installed -- skipping segmentation polygon "
                    "export (bboxes are unaffected). Install with `pip install "
                    "weightslab[export]` to enable polygon export."
                )
                _warned_no_cv2 = True
            return []
    return polygons


def _label_for_class(cls_id: Optional[int], class_names: Union[dict, list, tuple, None]) -> str:
    if cls_id is None:
        return "object"
    if class_names is not None:
        try:
            if isinstance(class_names, dict):
                if cls_id in class_names:
                    return str(class_names[cls_id])
                if str(cls_id) in class_names:
                    return str(class_names[str(cls_id)])
            elif isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
                return str(class_names[cls_id])
        except Exception:
            pass
    return f"class_{cls_id}"


def _resolve_dataset_for_origin(origin: Optional[str]):
    try:
        from weightslab.backend.ledgers import get_dataloader, list_dataloaders
        names = [origin] if origin else list_dataloaders()
        for name in names:
            loader = get_dataloader(name)
            dataset = getattr(loader, "dataset", None)
            if dataset is not None:
                return dataset
    except Exception:
        logger.debug("[export] Could not resolve dataset for origin=%r", origin, exc_info=True)
    return None


def _resolve_class_names(explicit, origin: Optional[str]):
    if explicit is not None:
        return explicit
    dataset = _resolve_dataset_for_origin(origin)
    class_names = getattr(dataset, "class_names", None) if dataset is not None else None
    if not class_names:
        logger.debug(
            "[export] No class_names resolved (no override, no dataset.class_names "
            "attribute) -- labels will fall back to 'class_<id>'."
        )
    return class_names


def _positional_index(dataset, sample_id) -> int:
    if dataset is not None and hasattr(dataset, "get_index_from_sample_id"):
        try:
            return dataset.get_index_from_sample_id(sample_id)
        except Exception:
            pass
    try:
        return int(sample_id)
    except (TypeError, ValueError):
        return 0


def _resolve_image_path(dataset, sample_id) -> Optional[str]:
    if dataset is None:
        return None
    idx = _positional_index(dataset, sample_id)
    for attr in _PATH_ATTR_CANDIDATES:
        seq = getattr(dataset, attr, None)
        if seq is None:
            continue
        try:
            item = seq[idx]
        except Exception:
            continue
        if isinstance(item, (tuple, list)) and item:
            item = item[0]
        if isinstance(item, (str, os.PathLike)):
            return str(item)
    return None


def _resolve_image_dims(path: Optional[str], mask_arr: Optional[np.ndarray], default_size: Tuple[int, int]) -> Tuple[int, int]:
    if mask_arr is not None:
        h, w = mask_arr.shape[0], mask_arr.shape[1]
        return int(w), int(h)
    if path:
        try:
            from PIL import Image
            with Image.open(path) as im:
                return int(im.width), int(im.height)
        except Exception:
            logger.debug("[export] Could not open %s to read dimensions", path, exc_info=True)
    return default_size


def _tag_column(tag: str) -> str:
    tag = tag.strip()
    prefix = f"{SampleStats.Ex.TAG.value}:"
    return tag if tag.startswith(prefix) else f"{prefix}{tag}"


def _filter_by_tags(df, tags: Optional[List[str]]):
    """Keep only rows carrying ANY of `tags` (boolean True, or a non-empty
    categorical value -- both kinds of tag reuse the same ``tag:<name>``
    column, see ``wl.tag_samples``/``wl.set_categorical_tag``).
    """
    if not tags:
        return df

    tag_cols = [col for col in (_tag_column(t) for t in tags) if col in df.columns]
    if not tag_cols:
        return df.iloc[0:0]

    mask = False
    for col in tag_cols:
        col_mask = df[col].notna() & (df[col] != False) & (df[col] != "")
        mask = col_mask if mask is False else (mask | col_mask)
    return df[mask]


def collect_image_annotations(
    origin: Optional[str] = None,
    class_names: Optional[Union[dict, list, tuple]] = None,
    use_predictions: bool = False,
    default_image_size: Tuple[int, int] = (0, 0),
    image_extension: str = ".jpg",
    tags: Optional[List[str]] = None,
) -> List[ImageAnnotations]:
    """Build the per-image annotation list from the registered dataframe.

    Args:
        origin: restrict to one split/loader name (e.g. "train_loader");
            ``None`` exports every registered split.
        class_names: explicit class-id -> name mapping (dict or list), wins
            over any auto-detected ``dataset.class_names``.
        use_predictions: export model predictions instead of ground-truth targets.
        default_image_size: (width, height) fallback when no dimensions can
            be resolved from a mask or the source image file.
        image_extension: extension used for the synthetic filename
            (``sample_<id><ext>``) when no real image path can be resolved.
        tags: restrict to samples carrying ANY of these tags (``tag:`` prefix
            optional, e.g. ``["ToReview"]``); ``None``/empty exports every sample.
    """
    from weightslab.backend.ledgers import get_dataframe

    dfm = get_dataframe()
    if dfm is None or not hasattr(dfm, "get_combined_df"):
        return []
    df = dfm.get_combined_df()
    if df is None or df.empty:
        return []

    df = df.reset_index()
    if origin:
        if ORIGIN not in df.columns:
            return []
        df = df[df[ORIGIN] == origin]
    if df.empty:
        return []

    df = _filter_by_tags(df, tags)
    if df.empty:
        return []

    value_col = PREDICTION if use_predictions else TARGET
    if value_col not in df.columns:
        return []

    resolved_class_names = _resolve_class_names(class_names, origin)
    dataset_cache: Dict[str, Any] = {}

    images: List[ImageAnnotations] = []
    for sample_id, group in df.groupby(SID, sort=False):
        sample_origin = str(group[ORIGIN].iloc[0]) if ORIGIN in group.columns else (origin or "")
        dataset = dataset_cache.setdefault(sample_origin, _resolve_dataset_for_origin(sample_origin or origin))
        path = _resolve_image_path(dataset, sample_id)
        filename = os.path.basename(path) if path else f"sample_{sample_id}{image_extension}"

        boxes: List[BoxAnnotation] = []
        polygons: List[PolygonAnnotation] = []
        mask_arr: Optional[np.ndarray] = None

        width, height = _resolve_image_dims(path, mask_arr, default_image_size)
        for cell in group[value_col]:
            if cell is None:
                continue

            # Det.
            for x1, y1, x2, y2, cls_id in _boxes_from_cell(cell):
                x1, y1, x2, y2 = int(x1*width), int(y1*height), int(x2*width), int(y2*height)  # Converts coords to pixel space
                boxes.append(
                    BoxAnnotation(
                        x1,
                        y1,
                        x2,
                        y2,
                        _label_for_class(cls_id, resolved_class_names)
                    )
                )

            # Seg.
            try:
                arr = np.asanyarray(cell)
            except Exception:
                continue
            if arr.ndim == 2 and arr.size > 0 and not _is_bbox_array(arr):
                mask_arr = arr
                polygons.extend(_polygons_from_mask_cell(arr, resolved_class_names))

        width, height = _resolve_image_dims(path, mask_arr, default_image_size)
        images.append(ImageAnnotations(
            sample_id=str(sample_id),
            filename=filename,
            width=width,
            height=height,
            origin=sample_origin,
            boxes=boxes,
            polygons=polygons,
        ))

    return images
