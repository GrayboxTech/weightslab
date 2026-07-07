import ast
import io
import time
import logging
import os
import traceback
import threading
import json
import re

import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.backend.audit_logger import AuditLogger

from PIL import Image
from tqdm import tqdm
from typing import List
from pathlib import Path
from datetime import datetime
from concurrent import futures

try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = dict # type: ignore

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.utils.tools import safe_reset_index
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.proto.experiment_service_pb2 import SampleEditType
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.backend.ledgers import get_dataloaders, get_dataframe, list_signals, Proxy
from weightslab.data.data_utils import load_label, load_raw_image, to_numpy_safe
from weightslab.data.point_cloud_utils import (
    POINT_CLOUD_DETECTION_TASK,
    is_point_cloud_detection_task,
)
from weightslab.trainer.trainer_tools import execute_df_operation, generate_overview, encode_image_to_raw_bytes
from weightslab.data.data_utils import load_raw_image_array

# Image encoding / mask compression / proto helpers (extracted)
from weightslab.trainer.services.data_image_utils import (
    rle_encode_mask,
    create_data_stat,
    resize_mask_nearest,
)


# Get global logger
logger = logging.getLogger(__name__)


# Streamed chunk size for GetPointCloud (raw float32 bytes per gRPC message).
# Larger chunks mean fewer messages but more memory per message. Override with
# the WL_POINT_CLOUD_CHUNK_BYTES env variable (see docs/configuration.rst).
_DEFAULT_POINT_CLOUD_CHUNK_BYTES = 1 << 20 # 1 MiB


def _point_cloud_chunk_bytes() -> int:
    """Read WL_POINT_CLOUD_CHUNK_BYTES; non-positive/invalid falls back to the default."""
    raw = os.getenv("WL_POINT_CLOUD_CHUNK_BYTES")
    if raw is None or raw == "":
        return _DEFAULT_POINT_CLOUD_CHUNK_BYTES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "WL_POINT_CLOUD_CHUNK_BYTES=%r is not an integer — using default %d",
            raw, _DEFAULT_POINT_CLOUD_CHUNK_BYTES,
        )
        return _DEFAULT_POINT_CLOUD_CHUNK_BYTES
    if val <= 0:
        logger.warning(
            "WL_POINT_CLOUD_CHUNK_BYTES=%r must be > 0 — using default %d",
            raw, _DEFAULT_POINT_CLOUD_CHUNK_BYTES,
        )
        return _DEFAULT_POINT_CLOUD_CHUNK_BYTES
    return val


def normalize_metadata_copy_source_name(source_name: str, experiment_hash: str = None) -> str:
    """Normalize a source metadata name for deterministic copied-column naming."""
    name = str(source_name or "").strip()
    if not name:
        return "metadata"
    if "@" in name:
        _, name = name.split("@", 1)
    exp_hash = str(experiment_hash or "").strip()
    if exp_hash and name.startswith(f"{exp_hash}_"):
        name = name[len(exp_hash) + 1:]
    name = name.replace("@", "_")
    # Remove trailing "_<int>" only when the suffix is numeric
    if "_" in name:
        base, suffix = name.rsplit("_", 1)
        if suffix.isdigit() and base:
            name = base
    name = re.sub(r"\s+", "_", name)
    return name


def build_metadata_copy_column_names(existing_columns, experiment_hash: str, source_name: str) -> str:
    """Build backend/ui copied metadata column names.

    All copies use ``{base}_{n}@{hash}`` starting from n=1. The base name is
    derived by normalizing the source (stripping any hash prefix and trailing
    numeric suffix) so cloning a clone produces a sibling, not a grandchild.
    """
    existing_iterable = [] if existing_columns is None else existing_columns
    existing = {str(col) for col in existing_iterable}

    exp_hash = str(experiment_hash or "current_experiment_hash").strip() or "current_experiment_hash"
    base = normalize_metadata_copy_source_name(source_name)

    n = 1
    while True:
        candidate = f"{base}_{n}@{exp_hash}"
        if candidate not in existing:
            return candidate
        n += 1


def duplicate_metadata_column_in_dataframe(df: pd.DataFrame, source_column: str, experiment_hash: str):
    """Return a dataframe copy with one duplicated metadata column using experiment-hash naming."""
    if source_column not in df.columns:
        raise KeyError(f"Source metadata column not found: {source_column}")

    copy_name = build_metadata_copy_column_names(df.columns, experiment_hash, source_column)
    df[copy_name] = df[source_column]
    return df, copy_name


def is_copy_metadata_column_name(column_name: str) -> bool:
    name = str(column_name or "").strip()
    # Copy columns always have the form {base}_{n}@{hash} — the numeric suffix
    # before '@' is mandatory; bare "name@hash" is not a copy column.
    return bool(re.match(r".+_\d+@.+", name))


def detect_bbox_format(bboxes: np.ndarray) -> str:
    """Detect bounding box format: 'xyxy' (x1,y1,x2,y2) or 'xywh' (x,y,w,h).

    Args:
        bboxes: numpy array of shape (N, 4) or (N, 5+) containing bbox data

    Returns:
        'xyxy' or 'xywh' format string. Defaults to 'xyxy' if ambiguous.
    """
    if bboxes is None or bboxes.size == 0 or bboxes.shape[-1] < 4:
        return 'xyxy'

    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.ndim < 2:
        return 'xyxy'

    coords = bboxes[..., :4] if bboxes.shape[-1] >= 4 else bboxes
    if coords.size == 0:
        return 'xyxy'

    x1_or_x = coords[..., 0]
    x2_or_w = coords[..., 2]

    x1_or_x_max = np.max(x1_or_x)
    x1_or_x_min = np.min(x1_or_x)
    x2_or_w_max = np.max(x2_or_w)
    x2_or_w_min = np.min(x2_or_w)

    if x1_or_x_max <= x2_or_w_max and x1_or_x_min <= x2_or_w_min:
        return 'xyxy'

    return 'xywh'


def bboxes_to_segmentation_mask(bboxes: np.ndarray, height: int, width: int,
                                 bbox_format: str = 'xyxy', class_ids: np.ndarray = None) -> np.ndarray:
    """Convert bounding boxes to a segmentation mask by rasterizing them.

    Args:
        bboxes: array of shape (N, 4) with bbox coordinates
        height: height of the output mask
        width: width of the output mask
        bbox_format: 'xyxy' or 'xywh' format
        class_ids: optional array of shape (N,) with class IDs for each bbox. If None, uses 1 for all.

    Returns:
        uint8 mask array of shape (height, width) with class IDs
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if bboxes is None or bboxes.size == 0:
        return mask

    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.ndim < 2:
        return mask

    if class_ids is not None:
        class_ids = np.asarray(class_ids, dtype=np.uint8)

    for idx, bbox in enumerate(bboxes):
        if len(bbox) < 4:
            continue

        if bbox_format == 'xywh':
            x, y, w, h = bbox[:4]
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1, y1, x2, y2 = bbox[:4]

        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))

        if x2 > x1 and y2 > y1:
            class_id = class_ids[idx] if class_ids is not None else 1
            mask[y1:y2, x1:x2] = int(class_id)

    return mask


def is_protected_metadata_name(column_name: str) -> bool:
    """
        Determine if a metadata column name is protected (cannot be edited by user). Exception if contains "@" to allow editing of copied columns.
    """
    name = str(column_name or "").strip()
    if not name:
        return False

    protected = set(SampleStatsEx.ALL())
    if name in protected:
        return True

    # Protect dynamic families too (e.g. tag:xxx, signal:xxx, signals//xxx)
    prefixed_families = [
        re.escape(SampleStatsEx.TAG.value),
        re.escape(SampleStatsEx.SIGNAL.value),
        r"signals",
    ]
    family_regex = rf"^({'|'.join(prefixed_families)})(?:$|[:/_\\-].*)"
    protected = bool(re.match(family_regex, name, flags=re.IGNORECASE))

    if protected and '@' not in name:
        return True

    return False


class _BoolOpToBitwise(ast.NodeTransformer):
    """AST transformer: rewrites `A and B` / `A or B` into `A & B` / `A | B`."""

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        op_cls = ast.BitAnd if isinstance(node.op, ast.And) else ast.BitOr
        result = node.values[0]
        for value in node.values[1:]:
            result = ast.BinOp(left=result, op=op_cls(), right=value)
        return ast.copy_location(result, node)


_AND_OR_KEYWORD_RE = re.compile(r"\b(and|or)\b")


def rewrite_boolean_keywords_to_bitwise(code: str) -> str:
    """
    Rewrites Python `and`/`or` boolean operators to their pandas-safe bitwise
    equivalents (`&`/`|`) in a single-expression code string.

    LLMs frequently write pandas boolean masks with Python's `and`/`or`
    keywords (e.g. `(df['a'] > 1) and (df['b'] < 2)`), which raises "The
    truth value of a Series/array is ambiguous" because `and`/`or` implicitly
    call `bool()` on each operand — something a multi-row Series/array can
    never satisfy. Rewriting via the AST (not string substitution) correctly
    preserves operator precedence/parenthesization. Falls back to the
    original code unchanged if it isn't a single parseable expression (e.g.
    multi-statement snippets), so this is always safe to call.
    """
    if not code or not _AND_OR_KEYWORD_RE.search(code):
        return code
    try:
        tree = ast.parse(code, mode="eval")
        tree = _BoolOpToBitwise().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return code


class DataService:

    """
    Data service helpers + RPCs (for weights_studio UI).

    Images are sent over gRPC as bytes (JPEG) for simplicity and correctness.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        # Set by ExperimentService after construction so the agent can invoke
        # architecture ops (freeze/reset) in-process. None in standalone/tests.
        self.model_service = None
        # Use MonitoredRLock so the gRPC watchdog can observe the holder thread
        # and how long the lock has been held, and interrupt it if needed.
        from weightslab.watchdog.lock_monitor import MonitoredRLock
        self._lock = MonitoredRLock()

        # -- _slowUpdateInternals concurrency primitives -----------------------
        # Only ONE gRPC worker performs the expensive pull+reindex at a time.
        # All other workers that arrive while an update is in progress WAIT for
        # that update to finish (via _update_done Event), then reuse its result
        # rather than queuing to redo the same work.
        #
        # Protocol:
        # 1. try_acquire(_update_lock, blocking=False)
        # → won: clear _update_done, do the update, release, set _update_done
        # → lost: _update_done.wait() then return (result already fresh)
        self._update_lock = threading.Lock()
        self._update_done = threading.Event()
        self._update_done.set() # "done" initially so the very first call proceeds
        # Guard so a non-force (reader-triggered) view refresh runs in the BACKGROUND
        # at most once at a time — readers never pay the rebuild cost (they read the
        # current snapshot; the bg thread swaps in fresh data when ready).
        self._refresh_in_flight = threading.Lock()
        self._df_manager = get_dataframe()

        # init references to the context components
        self._ctx.ensure_components()

        # Resolve root log directory and H5 path for data storage
        self._root_log_dir = self._resolve_root_log_dir()
        self._h5_path = self._resolve_h5_path()
        self._stats_store = H5DataFrameStore(self._h5_path) if self._h5_path else None

        # Initialize audit logger using root_log_dir
        self.audit_logger = None
        if self._root_log_dir:
            try:
                self.audit_logger = AuditLogger(self._root_log_dir, ctx.exp_name or "default")
            except Exception as e:
                logger.warning(f"Failed to initialize audit logger in DataService: {e}")

        # Check hyperparameters for compute_natural_sort flag (default: False)
        # Users can enable it by setting compute_natural_sort=True in their hyperparameters.
        hp = self._ctx.components.get("hyperparams") if self._ctx and self._ctx.components else None
        hp_dict = hp.get() if Proxy.is_proxy(hp) else (hp if isinstance(hp, dict) else {}) # is it already a proxy ?
        self._compute_natural_sort = bool((hp_dict or {}).get("compute_natural_sort", False))

        # How per-instance (per-annotation) numeric columns are folded to a single
        # per-sample scalar when collapsing the (sample_id, annotation_id) view.
        # Supported: "mean" (default) or "max". The full per-instance breakdown is
        # always preserved separately in the `_instance_signals` dict column.
        agg = str(
            (hp_dict or {}).get("instance_aggregation",
                                os.environ.get("WL_INSTANCE_AGGREGATION", "mean"))
        ).strip().lower()
        self._instance_aggregation = agg if agg in ("mean", "max") else "mean"

        # In-memory dataframe view of all datasets combined (streamed to UI)
        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()
        self._agent = DataManipulationAgent(self)
        try:
            import weightslab.backend.cli as cli_backend

            cli_backend.set_cli_data_service(self)
        except Exception:
            logger.debug("[DataService] Could not attach agent to CLI", exc_info=True)

        self._last_internals_update_time = 0.0

        # Shared thread pool for data processing (avoid thread explosion)
        # Size: min(CPU cores * 2, 16) to balance concurrency without excessive threading
        self._data_executor = futures.ThreadPoolExecutor(
            thread_name_prefix="WL-DataProcessing",
            max_workers=8
        )

        self._is_filtered = False # Track if the current view is filtered/modified by user
        # logger.info("[DataService] Skipping expensive startup computations (aspect ratio, natural sort, signals).")
        # These should be triggered on-demand or run in background to avoid blocking training start.

        if self._compute_natural_sort:
           self._compute_natural_sort_stats()

        self._is_filtered = False # Track if the current view is filtered/modified by user

        # =====================================================================
        # Preview cache: pre-generate 64×64 or less WebP thumbnails + RLE masks for
        # every sample in the dataset.
        # Enabled by default (auto-builds on background thread).
        # Disable with env var WL_PRELOAD_IMAGE_OVERVIEW=0.
        # Max entries controlled by WL_MAX_PREVIEW_CACHE_SIZE (default: 2000).
        # =====================================================================
        self._preview_cache: dict[int, pb2.DataRecord] = {}
        self._preview_cache_ready = threading.Event()
        self._preview_cache_max = int(os.environ.get("WL_MAX_PREVIEW_CACHE_SIZE", "2000"))
        preload_flag = os.environ.get("WL_PRELOAD_IMAGE_OVERVIEW", "1") != "0"
        # Also honour the legacy hp_dict key
        if not preload_flag:
            preload_flag = bool((hp_dict or {}).get("preload_image_overview", False))
        if preload_flag:
            threading.Thread(
                target=self._build_preview_cache,
                name="WL-PreviewCache",
                daemon=True,
            ).start()
        else:
            self._preview_cache_ready.set() # No preload → mark immediately ready

        logger.info("DataService initialized.")

    @staticmethod
    def _clamp_to_downscale_only(target_width: int, target_height: int, original_width: int, original_height: int):
        """Clamp target size so thumbnail paths never upscale beyond original size."""
        tw = max(1, int(target_width))
        th = max(1, int(target_height))
        ow = max(1, int(original_width))
        oh = max(1, int(original_height))

        scale = min(1.0, ow / float(tw), oh / float(th))
        if scale < 1.0:
            tw = max(1, int(tw * scale))
            th = max(1, int(th * scale))
        return tw, th

    # -----------------------------------------------------------------
    # Preview cache builder (runs on background thread)
    # -----------------------------------------------------------------
    def _build_preview_cache(self) -> None:
        """Pre-generate 64×64 or less or less thumbnail + RLE mask for every row in the DF.

                Each entry is a lightweight ``DataRecord`` containing only:
                    • raw_data (bytes) — 64×64 or less or less WebP thumbnail
                    • target (rle_mask) — RLE-encoded GT mask resized to 64×64 or less or less
                    • pred_mask (rle_mask) — RLE-encoded prediction mask resized to 64×64 or less or less
                    • origin, task_type, num_classes, class_names (metadata)
            Respects ``_preview_cache_max`` to cap memory usage.
        """
        try:
            df = self._all_datasets_df
            if df is None or df.empty:
                logger.info("[PreviewCache] Empty DF — nothing to cache.")
                self._preview_cache_ready.set()
                return

            total = min(len(df), self._preview_cache_max)
            logger.info("[PreviewCache] Building 64×64 or less or less preview cache for %d samples …", total)
            t0 = time.time()

            PREVIEW_SIZE = 64 # fixed low-res dimension
            built = 0

            index_names = list(getattr(df.index, "names", []) or [])
            origin_index_pos = index_names.index(SampleStatsEx.ORIGIN.value) if SampleStatsEx.ORIGIN.value in index_names else None
            sample_id_index_pos = index_names.index(SampleStatsEx.SAMPLE_ID.value) if SampleStatsEx.SAMPLE_ID.value in index_names else None

            for row_idx, (row_index, row) in enumerate(tqdm(df.iterrows(), total=total, desc="[PreviewCache]", unit="sample")):
                if row_idx >= total:
                    break

                sample_id = row.get(SampleStatsEx.SAMPLE_ID.value)
                origin = row.get(SampleStatsEx.ORIGIN.value)

                # In the internal dataframe view, origin/sample_id are often in the index
                # (MultiIndex: origin, sample_id), not in dataframe columns.
                if isinstance(row_index, tuple):
                    if origin is None and origin_index_pos is not None and origin_index_pos < len(row_index):
                        origin = row_index[origin_index_pos]
                    if sample_id is None and sample_id_index_pos is not None and sample_id_index_pos < len(row_index):
                        sample_id = row_index[sample_id_index_pos]
                    # Fallback for unnamed tuple index that still follows (origin, sample_id)
                    if origin is None and len(row_index) >= 1:
                        origin = row_index[0]
                    if sample_id is None and len(row_index) >= 2:
                        sample_id = row_index[1]
                else:
                    if sample_id is None and (sample_id_index_pos is not None or df.index.name == SampleStatsEx.SAMPLE_ID.value):
                        sample_id = row_index

                origin = str(origin) if origin is not None else 'unknown'

                dataset = self._get_dataset(origin)
                if dataset is None:
                    continue

                stats: list = []
                try:
                    ds = getattr(dataset, "wrapped_dataset", dataset)
                    # --- Thumbnail image ---
                    if hasattr(dataset, "get_physical_location"):
                        ds_idx, member_rank = dataset.get_physical_location(sample_id)
                    else:
                        ds_idx = dataset.get_index_from_sample_id(sample_id)
                        member_rank = 0

                    _, _, _, pil_img = load_raw_image_array(dataset, ds_idx, rank=member_rank)
                    if pil_img is not None:
                        ar = pil_img.size[0] / max(pil_img.size[1], 1)
                        if PREVIEW_SIZE >= max(pil_img.size):
                            # No resize if image size are smaller
                            tw, th = pil_img.size
                            pil_thumb = pil_img
                        else:
                            # Otherwise resize
                            tw = max(1, int(PREVIEW_SIZE * ar)) if ar >= 1 else PREVIEW_SIZE
                            th = PREVIEW_SIZE if ar >= 1 else max(1, int(PREVIEW_SIZE / ar))
                            tw, th = self._clamp_to_downscale_only(tw, th, pil_img.size[0], pil_img.size[1])
                            pil_thumb = pil_img.resize((tw, th), Image.Resampling.BILINEAR)

                        # Buffer thumbnail as WebP bytes (smaller than JPEG and supports lossless compression for very small images)
                        buf = io.BytesIO()
                        pil_save = pil_thumb.convert('RGB') if pil_thumb.mode not in ('RGB', 'RGBA') else pil_thumb
                        pil_save.save(buf, format='WEBP', quality=50, method=0)
                        thumb_bytes = buf.getvalue()
                        buf.close()
                        nc = len(pil_thumb.getbands())
                        stats.append(create_data_stat('raw_data', 'bytes', shape=[th, tw, nc], thumbnail=thumb_bytes))
                        del pil_thumb, pil_save, pil_img

                    # Resolve task type early: point-cloud detection labels are
                    # box rows, not rasterizable masks.
                    _early_task = row.get(SampleStatsEx.TASK_TYPE.value)
                    _early_task = str(_early_task).strip().lower() if _early_task else ""
                    if not _early_task or _early_task in ("none", "nan", "unknown"):
                        _early_task = str(getattr(ds, "task_type", "") or "").strip().lower()
                    _is_pc_detection = is_point_cloud_detection_task(_early_task)

                    # --- GT mask ---
                    label = load_label(dataset, sample_id)
                    if label is not None and _is_pc_detection:
                        # BEV-projected boxes as JSON (same payload as the full
                        # record path) so grid thumbnails get box overlays.
                        from weightslab.data.point_cloud_utils import serialize_pointcloud_box_payload
                        label_arr = to_numpy_safe(label)
                        if label_arr is not None and label_arr.size > 0:
                            try:
                                payload = serialize_pointcloud_box_payload(ds, label_arr)
                                if payload:
                                    stats.append(create_data_stat(
                                        'target', 'string', shape=[1],
                                        value_string=json.dumps(payload)))
                            except Exception as exc:
                                logger.debug("[PreviewCache] detection_pointcloud target skipped: %s", exc)
                    elif label is not None:
                        label_arr = to_numpy_safe(label)
                        if label_arr is None:
                            try:
                                label_arr = np.asarray(label)
                            except Exception:
                                label_arr = None
                        if label_arr is not None and label_arr.ndim >= 2:
                            # Resize mask to 64×64 or less using nearest-neighbour (class IDs must stay exact)
                            from PIL import Image as _PILImage
                            h, w = label_arr.shape[:2]
                            mask_pil = _PILImage.fromarray(label_arr.astype(np.uint8) if label_arr.ndim == 2 else label_arr.astype(np.uint8)[:, :, 0])
                            mask_pil = mask_pil.resize((tw if 'tw' in dir() else PREVIEW_SIZE, th if 'th' in dir() else PREVIEW_SIZE), _PILImage.Resampling.NEAREST)
                            mask_u8 = np.asarray(mask_pil, dtype=np.uint8)
                            rle = rle_encode_mask(mask_u8.ravel())
                            stats.append(create_data_stat('target', 'rle_mask', shape=list(mask_u8.shape), thumbnail=rle))
                            del mask_pil, mask_u8

                    # --- Prediction mask ---
                    pred = row.get(SampleStatsEx.PREDICTION.value)
                    if pred is not None and _is_pc_detection:
                        from weightslab.data.point_cloud_utils import serialize_pointcloud_box_payload
                        pred_arr = to_numpy_safe(pred)
                        if pred_arr is not None and pred_arr.size > 0 and pred_arr.ndim == 2:
                            try:
                                payload = serialize_pointcloud_box_payload(ds, pred_arr)
                                if payload:
                                    stats.append(create_data_stat(
                                        'pred', 'string', shape=[1],
                                        value_string=json.dumps(payload)))
                            except Exception as exc:
                                logger.debug("[PreviewCache] detection_pointcloud pred skipped: %s", exc)
                    elif pred is not None:
                        pred_arr = to_numpy_safe(pred)
                        if pred_arr is None:
                            try:
                                pred_arr = np.asarray(pred)
                            except Exception:
                                pred_arr = None
                        if pred_arr is not None and pred_arr.ndim >= 2:
                            from PIL import Image as _PILImage
                            pred_pil = _PILImage.fromarray(pred_arr.astype(np.uint8) if pred_arr.ndim == 2 else pred_arr.astype(np.uint8)[:, :, 0])
                            pred_pil = pred_pil.resize((tw if 'tw' in dir() else PREVIEW_SIZE, th if 'th' in dir() else PREVIEW_SIZE), _PILImage.Resampling.NEAREST)
                            pred_u8 = np.asarray(pred_pil, dtype=np.uint8)
                            pred_rle = rle_encode_mask(pred_u8.ravel())
                            stats.append(create_data_stat('pred_mask', 'rle_mask', shape=list(pred_u8.shape), thumbnail=pred_rle))
                            del pred_pil, pred_u8

                    # --- Metadata ---
                    # Detect task type
                    db_task = row.get(SampleStatsEx.TASK_TYPE.value)
                    task_type = str(db_task).strip().lower() if db_task and str(db_task).strip().lower() not in ("none", "nan", "unknown", "") else "unknown"
                    if task_type == "unknown" and _early_task:
                        task_type = _early_task
                    if task_type == "unknown" and label is not None:
                        l_arr = to_numpy_safe(label)
                        if l_arr is not None and l_arr.ndim >= 2 and l_arr.shape[-2] >= 16 and l_arr.shape[-1] >= 16:
                            task_type = "segmentation"
                    stats.append(create_data_stat('origin', 'string', shape=[1], value_string=origin))
                    stats.append(create_data_stat('task_type', 'string', shape=[1], value_string=task_type))

                    # num_classes — always include for segmentation so the frontend can build the
                    # colour palette on first preview render (before hi-res records arrive).
                    nc_val = row.get('num_classes') or getattr(ds, "num_classes", None)
                    if not nc_val and label is not None:
                        try:
                            _nc_arr = to_numpy_safe(label)
                            if _nc_arr is None:
                                _nc_arr = np.asarray(label)
                            if _nc_arr is not None and _nc_arr.size > 0:
                                nc_val = int(_nc_arr.max()) + 1
                        except Exception:
                            pass
                    if nc_val:
                        stats.append(create_data_stat('num_classes', 'scalar', shape=[1], value=[float(nc_val)]))

                    record = pb2.DataRecord(sample_id=str(sample_id), data_stats=stats)
                    self._preview_cache[sample_id] = record
                    built += 1
                except Exception as exc:
                    # traceback.print_exc()
                    logger.debug("[PreviewCache] Skipped sample %s: %s", sample_id, exc)
                    continue

            elapsed = time.time() - t0
            logger.info("[PreviewCache] Done: %d/%d cached in %.1fs (%.1f ms/sample)",
                        built, total, elapsed, elapsed / max(built, 1) * 1000)
        except Exception as exc:
            logger.error("[PreviewCache] Failed: %s", exc, exc_info=True)
        finally:
            self._preview_cache_ready.set()

    def _get_preview_record(self, sample_id: int) -> "pb2.DataRecord | None":
        """Return a cached 64×64 or less preview record, or None if not available."""
        return self._preview_cache.get(sample_id)

    def _refresh_preview_masks_from_row(self, preview_rec: "pb2.DataRecord", row: pd.Series) -> "pb2.DataRecord":
        """Return a copy of preview_rec with GT/PRED masks refreshed from the live dataframe row.

        This keeps `raw_data` served from the fast in-memory preview cache while ensuring
        segmentation masks reflect the latest row values (e.g. updated predictions during training).
        """
        rec = pb2.DataRecord(sample_id=preview_rec.sample_id)
        rec.data_stats.extend(preview_rec.data_stats)

        # Point-cloud detection records carry box JSON, not rasterizable masks:
        # refresh the GT/pred box payloads instead of RLE-encoding the rows.
        _task = next((st.value_string for st in rec.data_stats if st.name == "task_type"), "")
        if is_point_cloud_detection_task(_task):
            return self._refresh_preview_boxes_3d_from_row(rec, row)

        # Infer preview dimensions from cached raw_data shape; fallback to 64x64.
        target_h, target_w = 64, 64
        for st in rec.data_stats:
            if st.name == "raw_data" and len(st.shape) >= 2:
                try:
                    target_h = int(st.shape[0]) if int(st.shape[0]) > 0 else 64
                    target_w = int(st.shape[1]) if int(st.shape[1]) > 0 else 64
                except Exception:
                    target_h, target_w = 64, 64
                break

        def _encode_row_mask(mask_like) -> tuple[bytes | None, list[int] | None]:
            if mask_like is None:
                return None, None
            arr = to_numpy_safe(mask_like)
            if arr is None:
                try:
                    arr = np.asarray(mask_like)
                except Exception:
                    return None, None
            if arr is None or arr.ndim < 2:
                return None, None

            try:
                arr_u8 = arr.astype(np.uint8) if arr.ndim == 2 else arr.astype(np.uint8)[:, :, 0]
                mask_pil = Image.fromarray(arr_u8)
                if target_w > 0 and target_h > 0 and (mask_pil.size[0] != target_w or mask_pil.size[1] != target_h):
                    mask_pil = mask_pil.resize((target_w, target_h), Image.Resampling.NEAREST)
                mask_u8 = np.asarray(mask_pil, dtype=np.uint8)
                return rle_encode_mask(mask_u8.ravel()), list(mask_u8.shape)
            except Exception:
                return None, None

        def _upsert_mask_stat(stat_name: str, rle_bytes: bytes | None, shape: list[int] | None) -> None:
            if not rle_bytes or not shape:
                return
            new_stat = create_data_stat(stat_name, 'rle_mask', shape=shape, thumbnail=rle_bytes)
            for i, st in enumerate(rec.data_stats):
                if st.name == stat_name:
                    rec.data_stats[i].CopyFrom(new_stat)
                    return
            rec.data_stats.append(new_stat)

        gt_rle, gt_shape = _encode_row_mask(row.get(SampleStatsEx.TARGET.value))
        _upsert_mask_stat('target', gt_rle, gt_shape)

        pred_rle, pred_shape = _encode_row_mask(row.get(SampleStatsEx.PREDICTION.value))
        _upsert_mask_stat('pred_mask', pred_rle, pred_shape)

        return rec

    def _refresh_preview_boxes_3d_from_row(self, rec: "pb2.DataRecord", row: pd.Series) -> "pb2.DataRecord":
        """Refresh GT/pred box JSON stats of a detection_pointcloud preview record."""
        from weightslab.data.point_cloud_utils import serialize_pointcloud_box_payload

        origin = next((st.value_string for st in rec.data_stats if st.name == "origin"), None)
        dataset = self._get_dataset(origin) if origin else None
        ds = getattr(dataset, "wrapped_dataset", dataset) if dataset is not None else None

        def _upsert_box_stat(stat_name: str, value_like) -> None:
            if value_like is None or ds is None:
                return
            arr = to_numpy_safe(value_like)
            if arr is None or getattr(arr, "ndim", 0) != 2 or arr.size == 0:
                return
            try:
                payload = serialize_pointcloud_box_payload(ds, arr)
            except Exception:
                return
            if not payload:
                return
            new_stat = create_data_stat(stat_name, 'string', shape=[1], value_string=json.dumps(payload))
            for i, st in enumerate(rec.data_stats):
                if st.name == stat_name:
                    rec.data_stats[i].CopyFrom(new_stat)
                    return
            rec.data_stats.append(new_stat)

        _upsert_box_stat('target', row.get(SampleStatsEx.TARGET.value))
        _upsert_box_stat('pred', row.get(SampleStatsEx.PREDICTION.value))
        return rec

    def _get_loader_by_origin(self, origin: str):
        """Dynamically retrieve loader for a specific origin (on-demand).

        Avoids maintaining persistent _loaders state; fetches only when needed.
        """
        try:
            loader_names = get_dataloaders()
            for loader_name in loader_names:
                loader = self._ctx.components.get(loader_name)
                if loader is None:
                    continue
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == origin:
                        return loader
                # Fallback: match by loader name
                elif origin in loader_name:
                    return loader
        except Exception as e:
            logger.debug(f"[_get_loader_by_origin] Failed to retrieve loader for origin={origin}: {e}")
        return None

    def _initialize_data_service(self):
        """Recreate the in-memory dataframe view from the shared H5 store."""
        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()

    def _resolve_root_log_dir(self) -> Path:
        """Resolve root log directory from hyperparams/env, fallback to ./logs."""
        root = None
        try:
            hp = self._ctx.components.get("hyperparams")
            if hp is not None and hasattr(hp, "get"):
                hp_dict = hp.get() if Proxy.is_proxy(hp) else (hp if isinstance(hp, dict) else {})
                if isinstance(hp_dict, dict):
                    root = (
                        hp_dict.get("root_log_dir")
                        or hp_dict.get("root_directory")
                        or hp_dict.get("root")
                    )
        except Exception:
            root = None

        root = root or os.getenv("WEIGHTSLAB_ROOT_LOG_DIR")
        if root is None:
            root = Path("logs").absolute()
        return Path(root)

    def _resolve_h5_path(self) -> Path | None:
        """Return the H5 path used by tracked datasets and the streaming view."""
        if self._root_log_dir is None:
            return None
        data_dir = Path(self._root_log_dir) / "checkpoints" / "data"
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return data_dir / "data_with_ops.h5"

    def _log_audit(
        self,
        action_type: str,
        status: str,
        details: dict = None,
        error: str = None,
    ) -> None:
        """Helper to log audit events if logger is available."""
        if self.audit_logger:
            try:
                self.audit_logger.log_event(
                    action_type=action_type,
                    status=status,
                    details=details,
                    error=error,
                )
            except Exception as e:
                logger.debug(f"Failed to log audit event: {e}")

    def _is_agent_available(self) -> bool:
        """
        Check if the agent (Ollama) is available for natural language queries.

        Returns:
            bool: True if agent is available, False otherwise
        """
        if self._agent is None:
            return False
        try:
            return self._agent.is_available()
        except Exception as e:
            logger.debug(f"Error checking agent availability: {e}")
            return False

    def _is_training_active(self) -> bool:
        """Return True if training is currently running (not paused)."""
        try:
            hp = self._ctx.components.get("hyperparams")
            if hp is not None and hasattr(hp, "get"):
                hp_dict = hp.get() if Proxy.is_proxy(hp) else (hp if isinstance(hp, dict) else {})
                if isinstance(hp_dict, dict):
                    flag = hp_dict.get("is_training")
                    if flag is not None:
                        return bool(flag)
        except Exception:
            pass
        # Fall back to pause controller state if hyperparams missing
        try:
            return not pause_controller.is_paused()
        except Exception:
            return True

    def _pull_into_all_data_view_df(self):
        """Stream stats from the global in-memory dataframe (ledger manager).

        Uses the shared dataframe manager instead of the H5 store and avoids
        blocking on IO. Falls back to last snapshot if retrieval fails.
        """
        try:
            # Load dataframe from the shared dataframe manager with arrays autoloaded from h5 storage
            df = self._df_manager.get_combined_df() if self._df_manager is not None else pd.DataFrame()
            if df.empty:
                logger.debug(f"[DataService] Pull returned empty dataframe (manager: {self._df_manager is not None})")
                return df

            # The manager now expands samples into one row per (sample_id, annotation_id)
            # instance. Collapse back to one row per sample for the sample-centric UI/agent
            # view, nesting per-instance signals into a dict column.
            df = self._df_manager.get_collapse_annotations_to_samples_df()

            # Ensure sample_id is a column if it was the index
            df = safe_reset_index(df)

            # Ensure we have a unique index across all origins by using a MultiIndex (origin, sample_id)
            # This is CRITICAL for correctly applying reindex() in _slowUpdateInternals without
            # exploding the dataframe size due to duplicate sample_id index labels.
            if SampleStatsEx.ORIGIN.value in df.columns:
                # Use drop=True to ensure origin is NOT in both index and columns (avoids ambiguity)
                # GetDataSamples calls reset_index() before processing rows, which restores them as columns
                df = df.set_index([SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value], drop=True)
            else:
                # Fallback to single index if origin is missing, though manager should provide it
                df = df.set_index([SampleStatsEx.SAMPLE_ID.value], drop=True)

            # DEDUPLICATE: Ensure index is unique before returning.
            # If duplicates exist, reindex() will fail later.
            if df.index.has_duplicates:
                logger.debug(f"[DataService] Dropping {df.index.duplicated().sum()} duplicate index labels from data view.")
                df = df[~df.index.duplicated(keep='last')]

            return df
        except Exception as e:
            logger.debug(f"[DataService] Error pulling data view: {e}")
            # Use getattr to safely check for attribute during __init__
            current_df = getattr(self, "_all_datasets_df", None)
            return current_df if current_df is not None else pd.DataFrame()

    def _get_origin_filter(self, request):
        """Extract requested origins if present on request (backward compatible)."""
        origins = None
        for attr in ("sample_origins", "origins", "origin"):
            try:
                val = getattr(request, attr, None)
                if val:
                    # Normalize to list
                    if isinstance(val, str):
                        origins = [val] if val.strip() else [] # Filter empty strings
                    else:
                        # Filter out empty strings from list
                        origins = [o for o in list(val) if o and str(o).strip()]
                    break
            except Exception:
                continue
        return origins or []

    def _filter_df_by_origin(self, df: pd.DataFrame, origins: list[str]) -> pd.DataFrame:
        if df is None or df.empty or not origins:
            return df
        try:
            if isinstance(df.index, pd.MultiIndex):
                if "origin" in df.index.names:
                    mask = df.index.get_level_values("origin").isin(origins)
                    return df.loc[mask]
            if "origin" in df.columns:
                return df[df["origin"].isin(origins)]
        except Exception as e:
            logger.debug(f"[_filter_df_by_origin] failed to filter by origins {origins}: {e}")
        return df

    def _load_existing_tags(self):
        """Ensure tags and natural sort columns are present on the streamed dataframe."""
        if self._all_datasets_df is None or self._all_datasets_df.empty:
            return

        if SampleStatsEx.TAG.value not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df[SampleStatsEx.TAG.value] = ""
            except Exception:
                pass

        # Ensure signals.defaults.natural exists (init with NaN if missing)
        if self._compute_natural_sort and "signals.defaults.natural" not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df["signals.defaults.natural"] = np.nan
            except Exception:
                pass

    def _get_dataset(self, origin: str):
        loader = self._get_loader_by_origin(origin)
        if loader is not None:
            dataset = getattr(loader, "tracked_dataset", None)
            return dataset

    def _is_nan_value(self, value):
        """Check if a value is NaN, handling both scalars and arrays."""
        try:
            # For scalars, np.isnan works directly
            if np.isscalar(value):
                return np.isnan(value)
            return False
        except (TypeError, ValueError):
            # For arrays/tensors, check if it's a single NaN value
            try:
                value_arr = np.asarray(value)
                return value_arr.size == 1 and np.isnan(value_arr.item())
            except (TypeError, ValueError):
                return False

    def _compute_natural_sort_stats(self):
        """
        Compute hardcoded natural sort statistics (brightness, hue, saturation, entropy) for all samples
        and update the dataframe.

        Includes 'signals.defaults.natural' (the weighted composite) for sorting.
        """
        # --- CONFIGURATION: Define Natural Sort Cues & Weights ---
        # Weights should sum to 1.0 ideally, but relative magnitude matters most.
        # Strategies:
        # 1. "Day vs Night" focus: Brightness=0.8, Entropy=0.2
        # 2. "Complexity" focus: Brightness=0.2, Entropy=0.8
        # 3. "Balanced": Brightness=0.5, Entropy=0.5
        # 4. "Grouped" (Pseudo-primary key): Brightness=5.0, Entropy=1.0 (Forces clustering by light)

        SORT_WEIGHTS = {
            "brightness": 0.7, # Primary cue: Lighting conditions
            "entropy": 0.3, # Secondary cue: Texture/Scene complexity
            "hue": 0.0 # Optional: Color tint
        }

        logger.info(f"[DataService] Starting natural sort stats computation with weights: {SORT_WEIGHTS}")

        if self._all_datasets_df is None or self._all_datasets_df.empty:
             return "No data to process"

        # Helper: Calculate Shannon Entropy (Complexity).
        # 256-bin histogram of the 8-bit grayscale image via numpy (no OpenCV).
        def calc_entropy(img_gray):
            try:
                gray_u8 = np.clip(np.rint(img_gray), 0, 255).astype(np.uint8)
                counts = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
                total = counts.sum()
                if total <= 0:
                    return 0.0
                # Normalize to probabilities, dropping zeros to avoid log(0)
                p = counts / total
                p = p[p > 0]
                # Shannon Entropy in bits
                return float(-np.sum(p * np.log2(p)))
            except Exception:
                return 0.0

        # helper to process a single image
        def process_sample(args):
            idx, row = args
            try:
                origin = row.get(SampleStatsEx.ORIGIN.value, 'unknown')

                # Sample ID is either the index or in the columns — use as-is (may be string UID)
                if SampleStatsEx.SAMPLE_ID.value in row:
                    sample_id = row[SampleStatsEx.SAMPLE_ID.value]
                else:
                    sample_id = idx

                # Skip if already computed (optimization for blocking startup)
                if "signals.defaults.natural" in row and not pd.isna(row["signals.defaults.natural"]):
                    return None

                dataset = self._get_dataset(origin)
                if not dataset:
                    logger.warning(f"[Natural Sort] Dataset not found for origin: {origin}")
                    return None

                # Use dataset index, not dataframe index
                ds_idx = dataset.get_index_from_sample_id(sample_id)
                if ds_idx is None:
                     logger.warning(f"[Natural Sort] Could not map sample_id {sample_id} to dataset index")
                     return None

                pil_img = load_raw_image(dataset, ds_idx)

                if pil_img is None:
                    # silenced to avoid spam if images are genuinely missing for many
                    # logger.warning(f"[Natural Sort] Failed to load image for sample {sample_id} at index {ds_idx}")
                    return None

                # Convert to numpy (RGB)
                img_np = np.array(pil_img)

                # Brightness (mean pixel intensity). For color images, reduce to
                # luma using the ITU-R 601-2 transform — the same weights PIL's
                # "L" mode uses — with plain numpy.
                if img_np.ndim == 3 and img_np.shape[2] >= 3:
                    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)
                    gray = img_np[..., :3].astype(np.float32) @ luma
                elif img_np.ndim == 3:
                    gray = img_np[..., 0].astype(np.float32)
                else:
                    gray = img_np.astype(np.float32)

                brightness = float(np.mean(gray))
                entropy = calc_entropy(gray)

                # HSV stats (hue/saturation). Computed with Pillow's "HSV" mode
                # (H, S, V each in 0-255) to avoid an OpenCV dependency.
                if img_np.ndim == 3 and img_np.shape[2] >= 3:
                    try:
                        rgb_img = pil_img if pil_img.mode == "RGB" else pil_img.convert("RGB")
                        hsv = np.asarray(rgb_img.convert("HSV"))
                        hue = float(np.mean(hsv[:, :, 0]))
                        saturation = float(np.mean(hsv[:, :, 1]))
                    except Exception:
                        hue = 0.0
                        saturation = 0.0
                else:
                    hue = 0.0
                    saturation = 0.0

                # --- Compute Composite Score ---
                # Normalize robustly to 0-1 range for combination
                # Brightness: 0-255 typical for 8-bit
                norm_brightness = min(max(brightness / 255.0, 0.0), 1.0)

                # Entropy: 0-8 bits typical for 8-bit image
                norm_entropy = min(max(entropy / 8.0, 0.0), 1.0)

                # Hue: 0-255 in Pillow's HSV space
                norm_hue = min(max(hue / 255.0, 0.0), 1.0)

                score = (
                    SORT_WEIGHTS.get("brightness", 0) * norm_brightness +
                    SORT_WEIGHTS.get("entropy", 0) * norm_entropy +
                    SORT_WEIGHTS.get("hue", 0) * norm_hue
                )

                return {
                    "sample_id": sample_id,
                    "origin": origin,
                    "signals.defaults.brightness": brightness,
                    "signals.defaults.entropy": entropy,
                    "signals.defaults.hue": hue,
                    "signals.defaults.saturation": saturation,
                    "signals.defaults.natural": score,
                }
            except Exception as e:
                # Log only the first few errors globally (using a simple counter if we could, but here we can't share state easily)
                # Fallback: Just log warning. To avoid spam, we can check if it's one of the first few tasks?
                # No, standard logger is fine, just use debug for spammy ones or INFO for specific tracking.
                logger.warning(f"Failed to compute stats for sample {idx} (id={row.get(SampleStatsEx.SAMPLE_ID.value, 'unknown')}): {e}")
                return None

        # Prepare tasks
        tasks = []
        for idx, row in safe_reset_index(self._all_datasets_df).iterrows():
             tasks.append((idx, row))

        logger.info(f"[DataService] Computing sort stats for {len(tasks)} samples...")

        # Run in parallel
        # Use a smaller executor if needed, but self._data_executor is shared
        results = self._data_executor.map(process_sample, tasks)

        # Collect results
        updates_by_origin = {}
        processed_count = 0
        failure_count = 0

        # Wrap results with tqdm for progress bar
        for res in tqdm(results, total=len(tasks), desc="Computing Natural Sort Stats", unit="samples"):
            if res:
                processed_count += 1
                origin = res["origin"]
                updates_by_origin.setdefault(origin, []).append(res)
            else:
                failure_count += 1
                if failure_count <= 5:
                     # We can't easily get the specific error detail here because it was swallowed in process_sample
                     # But we rely on process_sample to log it.
                     pass

        if failure_count > 0:
             logger.warning(f"[DataService] Failed to compute stats for {failure_count} samples. Check logs for details.")

        # Upsert to storage
        if self._df_manager:
            for origin, rows in updates_by_origin.items():
                df_update = pd.DataFrame(rows).set_index("sample_id")
                # Drop origin column from update if it's in the index or redundant
                if "origin" in df_update.columns:
                     del df_update["origin"]

                self._df_manager.upsert_df(df_update, origin=origin, force_flush=True)

        # Force refresh internal view
        self._slowUpdateInternals(force=True)

        logger.info(f"[DataService] Completed stats computation for {processed_count} samples")
        print(f"\n\nNatural sort computation finished for {processed_count} samples\n\n")
        return f"Computed stats for {processed_count} samples"

    def _compute_custom_signals(self):
        """
        Discover and compute registered custom signals for all active dataloaders.
        This allows scripts to simply register @wl.signal and have them computed automatically.
        """
        # Local import to avoid circular dependency
        import weightslab.src as wl_src

        # Get all registered signal names
        signals = list_signals()
        if not signals:
            return

        # Get all active dataloaders
        # use get_dataloaders without args to get all registered ones
        loaders = get_dataloaders(None)

        logger.info(f"[DataService] Checking custom signals {signals} for {len(loaders)} loaders...")

        for loader_name, loader in loaders.items():
            if not loader or not hasattr(loader, "dataset"):
                continue

            try:
                # We need to determine the origin/split name for the compute_signals function
                # Try to inspect the loader/dataset
                origin = loader_name # Fallback
                ds = getattr(loader, "tracked_dataset", None)
                if ds and hasattr(ds, "_dataset_split"):
                    origin = ds._dataset_split
                elif hasattr(loader, "dataset") and hasattr(loader.dataset, "split"):
                    origin = loader.dataset.split

                # Run computation
                logger.info(f"[DataService] Computing signals {signals} for loader '{loader_name}' (origin={origin})")
                wl_src.compute_signals(loader, origin=origin, signals=signals)

            except Exception as e:
                logger.error(f"[DataService] Failed to compute signals for loader '{loader_name}': {e}")

        # Force view update
        self._slowUpdateInternals(force=True)

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        start_total = time.time()
        try:
            origin = row.get(SampleStatsEx.ORIGIN.value, 'unknown')
            sample_id = row.get(SampleStatsEx.SAMPLE_ID.value, 0)
            logger.debug(f"Processing sample_id={sample_id} from origin={origin} with request: {request}")

            # ===== Timing accumulators =====
            t_image_load = 0.0
            t_image_encode = 0.0
            t_mask_encode = 0.0
            t_label_convert = 0.0

            # ===== Step 0: Initialize Variables======
            raw_shape, data_stats = [], []
            raw_data_bytes = b""

            # ====== Step 1: Request mode ======
            metadata_only_request = self._is_metadata_only_request(request)

            skip_label_for_request = metadata_only_request
            skip_prediction_for_request = metadata_only_request

            # ====== Step 2: Load dataset lazily (avoid unnecessary IO for metadata-only) ======
            needs_dataset = bool(request.include_raw_data) or (not skip_label_for_request)
            dataset = self._get_dataset(origin) if needs_dataset else None

            # ====== Step 3: Determine task type ======
            label = row.get(SampleStatsEx.TARGET.value)
            is_label_empty = False
            if label is None:
                is_label_empty = True
            elif isinstance(label, list) and not label:
                is_label_empty = True
            elif isinstance(label, str) and isinstance(label, str) and '.' in label and not label.endswith('.') and label.rsplit('.', 1)[-1] != '':
                is_label_empty = True
            elif isinstance(label, float):
                import math
                if math.isnan(label):
                    is_label_empty = True

            if is_label_empty and dataset and not skip_label_for_request:
                label = load_label(dataset, sample_id)

            ds = getattr(dataset, "wrapped_dataset", dataset)
            model = self._ctx.components.get("model") if self._ctx else None

            # Robust Task Type Detection
            # 1. DB Row takes precedence if valid string
            db_task_type = row.get(SampleStatsEx.TASK_TYPE.value)
            if db_task_type and isinstance(db_task_type, str) and db_task_type.strip().lower() not in ["none", "nan", "unknown", ""]:
                task_type = db_task_type.strip().lower()
            # 2. Explicit property on Dataset
            elif hasattr(ds, "task_type") and getattr(ds, "task_type"):
                task_type = str(getattr(ds, "task_type")).strip().lower()
            # 3. Explicit property on Model
            elif model and hasattr(model, "task_type") and getattr(model, "task_type"):
                task_type = str(getattr(model, "task_type")).strip().lower()
            elif metadata_only_request:
                # Metadata-only: avoid expensive label-based heuristics.
                task_type = "unknown"
            else:
                # 4. Safe Heuristic evaluation
                task_type = "classification" # Default fallback
                if label is not None:
                    if isinstance(label, dict):
                        if ('boxes' in label or 'bboxes' in label or 'bb' in label):
                            task_type = 'detection'
                    else:
                        l_arr = to_numpy_safe(label)
                        if l_arr is not None:
                            try:
                                ndim = l_arr.ndim
                                shape = l_arr.shape
                                if ndim >= 2:
                                    # Detection: shape like (N, 4), (N, 5), (N, 6)
                                    if ndim == 2 and shape[-1] in [4, 5, 6] and shape[-2] > 0:
                                        task_type = 'detection'
                                    # 3D detection: (N, 7..9) metric box rows
                                    # [cx, cy, cz, dx, dy, dz, yaw, cls?, conf?]
                                    elif ndim == 2 and shape[-1] in [7, 8, 9] and shape[-2] > 0:
                                        task_type = POINT_CLOUD_DETECTION_TASK
                                    # Segmentation: check spatial dims
                                    elif len(shape) >= 2 and shape[-2] >= 16 and shape[-1] >= 16:
                                        task_type = 'segmentation'
                            except Exception:
                                pass

            # ====== Step 5a: Metadata stats — moved to GetMetaData ======
            # Generic dataframe metadata columns (signals, tags, custom fields, etc.)
            # are no longer returned by GetDataSamples; the dedicated GetMetaData RPC
            # serves them. GetDataSamples returns only the rendering flags
            # origin / task_type / discarded (needed for the split border, overlay
            # mode and gray-out) plus image / label / prediction data below.

            # ====== Step 6: Add origin, task_type and discarded rendering flags ======
            data_stats.append(
                create_data_stat(
                    "origin", 'string', shape=[1], value_string=origin, thumbnail=b""
                )
            )
            data_stats.append(
                create_data_stat(
                    "task_type", 'string', shape=[1], value_string=str(task_type), thumbnail=b""
                )
            )
            # 'discarded' drives the grayed-out cell rendering, so it rides with the
            # image data as "1"/"0" (not treated as analytical metadata). This keeps
            # the gray-out reliable on every grid (re)fetch / scroll.
            try:
                _discarded_str = "1" if bool(row.get(SampleStatsEx.DISCARDED.value)) else "0"
            except Exception:
                _discarded_str = "0"
            data_stats.append(
                create_data_stat(
                    SampleStatsEx.DISCARDED.value, 'string', shape=[1], value_string=_discarded_str, thumbnail=b""
                )
            )

            target_mask_stat_index = None
            pred_mask_stat_index = None
            target_mask_u8 = None
            pred_mask_u8 = None

            # ====== Step 7: Process labels ======
            t_label_convert = 0.0
            if not skip_label_for_request and task_type != "classification":
                t0_gt_conv = time.time()
                label_raw = row.get(SampleStatsEx.TARGET.value) if label is None else label
                if label_raw is None and dataset is not None:
                    label_raw = load_label(dataset, sample_id)

                # Handle 3D (point cloud) detection: metric boxes are sent both
                # as raw 3D rows (for the interactive viewer) and projected to
                # the BEV image frame (legacy 'bboxes' key, so the existing 2D
                # detection renderer overlays them unchanged).
                if is_point_cloud_detection_task(task_type):
                    from weightslab.data.point_cloud_utils import serialize_pointcloud_box_payload
                    bbox_data = {}
                    label_arr = to_numpy_safe(label_raw)
                    if label_arr is None:
                        try:
                            label_arr = np.asarray(label_raw, dtype=np.float32) if label_raw is not None else np.array([])
                        except Exception:
                            label_arr = np.array([])
                    if label_arr is not None and label_arr.size > 0:
                        try:
                            bbox_data = serialize_pointcloud_box_payload(ds, label_arr)
                        except Exception as exc:
                            logger.debug("detection_pointcloud target serialization failed: %s", exc)
                    t_label_convert = time.time() - t0_gt_conv

                    data_stats.append(
                        create_data_stat(
                            name='target',
                            stat_type='string',
                            shape=[1],
                            value_string=json.dumps(bbox_data) if bbox_data else "",
                            thumbnail=b""
                        )
                    )
                # Handle detection task type with bounding boxes
                elif task_type == "detection":
                    # Send raw bbox data as JSON
                    bbox_data = {}
                    if isinstance(label_raw, dict):
                        # Support multiple dict key names for bboxes: 'bboxes', 'boxes', 'bb'
                        bbox_key = next((k for k in ['bboxes', 'boxes', 'bb'] if k in label_raw), None)
                        if bbox_key:
                            try:
                                # Convert dict to arrays
                                bboxes = label_raw.get(bbox_key)
                                bboxes = np.asarray([bbox for bbox in bboxes if len(bbox)])
                            except (ValueError, TypeError):
                                bboxes = np.array([])

                            class_ids = label_raw.get('class_ids')
                            if bboxes.size > 0 and bboxes.ndim >= 2:
                                bbox_format = detect_bbox_format(bboxes)
                                bbox_data = {
                                    "bboxes": bboxes.tolist(),
                                    "class_ids": class_ids.tolist() if hasattr(class_ids, 'tolist') else (class_ids if class_ids is not None else 1),
                                    "format": bbox_format
                                }
                    if not bbox_data and label_raw is not None:
                        label_arr = to_numpy_safe(label_raw)
                        if label_arr is None:
                            try:
                                label_arr = np.asarray(label_raw, dtype=np.float32)
                            except Exception:
                                label_arr = np.array([])

                        if label_arr.ndim == 1 and label_arr.size >= 4:
                            label_arr = label_arr.reshape(1, -1)
                        if label_arr.size > 0 and label_arr.ndim == 2 and label_arr.shape[-1] >= 4:
                            # 6-col rows when class+score present: [x,y,x,y,class,score]
                            # (or [tl_x,tl_y,w,h,class,score] for the xywh-top-left flavor).
                            # The `format` string stays legacy-compatible ('xyxy' | 'xywh')
                            # so any deployed renderer reads cols 0-3 correctly and ignores
                            # extras. Renderers that know the new schema infer class/score
                            # by row length (bbox.length >= 6).
                            fmt = detect_bbox_format(label_arr[..., :4])
                            bbox_data = {
                                "bboxes": label_arr.tolist(),
                                "format": fmt,
                            }

                    t_label_convert = time.time() - t0_gt_conv

                    # Send bbox data as JSON string
                    bbox_json = json.dumps(bbox_data) if bbox_data else ""
                    data_stats.append(
                        create_data_stat(
                            name='target',
                            stat_type='string',
                            shape=[1],
                            value_string=bbox_json,
                            thumbnail=b""
                        )
                    )
                # Handle segmentation task type
                elif label_raw is not None:
                    label_arr = to_numpy_safe(label_raw)
                    if label_arr is not None:
                        label_u8 = label_arr.astype(np.uint8) if label_arr.size > 0 else label_arr
                        t_label_convert = time.time() - t0_gt_conv

                        t0_rle = time.time()
                        rle_bytes = rle_encode_mask(label_u8.ravel()) if label_u8.size > 0 else b""
                        t_mask_encode += time.time() - t0_rle

                        data_stats.append(
                            create_data_stat(
                                name='target',
                                stat_type='rle_mask',
                                shape=list(label_u8.shape) if label_u8.size > 0 else [],
                                value=[],
                                thumbnail=rle_bytes
                            )
                        )
                        target_mask_stat_index = len(data_stats) - 1
                        target_mask_u8 = label_u8

                # Prefer row attribute if available, fallback to dataset, then model
                num_classes = (
                    row.get('num_classes')
                    or (getattr(dataset, "num_classes", None) if dataset else None)
                )
                if num_classes is None and label_raw is not None:
                    # Fallback: infer from this label
                    if is_point_cloud_detection_task(task_type) and getattr(label_arr, "ndim", 0) == 2 and label_arr.shape[-1] >= 8:
                        # Class ids live in column 7 of [cx,cy,cz,dx,dy,dz,yaw,cls,conf]
                        num_classes = int(label_arr[:, 7].max()) + 1
                    elif label_arr.size > 0:
                        max_id = int(label_arr.max())
                        num_classes = max(1, max_id) + 1
                    else:
                        num_classes = 2 # Always at least 2 classes for segmentation (foreground/background)

                data_stats.append(
                    create_data_stat(
                        name="num_classes",
                        stat_type="scalar",
                        shape=[1],
                        value=[float(num_classes)],
                        thumbnail=b""
                    )
                )
                # Sanity check: warn if the inferred num_classes is smaller than
                # the largest class id present in the label. Guard against empty
                # or non-array labels (e.g. detection samples with no objects),
                # where .max() on a zero-size array raises ValueError.
                label_np = to_numpy_safe(label_raw) if label_raw is not None else None
                if (
                    num_classes is not None
                    and label_np is not None
                    and label_np.size > 0
                    and num_classes < label_np.max()
                ):
                    logger.warning(f'Be aware that the num_classes infered is inferior to max value in the label')

                # Per-sample class_names emission. KEPT because the studio has
                # no dataset-level RPC (no GetClassNames / GetDatasetMetadata
                # in experiment_service.proto), so this is the only path that
                # delivers the id→label mapping to the UI. See ISSUES O-17 —
                # the proper fix is a dedicated dataset RPC; this block goes
                # away once that lands.
                class_names = (
                    row.get('class_names')
                    or (getattr(dataset, "class_names", None) if dataset else None)
                )

                if class_names:
                    val_str = ""
                    try:
                        if isinstance(class_names, (list, tuple)):
                            val_str = json.dumps(list(class_names))
                        elif isinstance(class_names, (dict, DictConfig)):
                            # Convert int keys to string for JSON if all keys are ints
                            if all(isinstance(k, int) for k in class_names.keys()):
                                class_names = {str(k): v for k, v in class_names.items()}
                            val_str = json.dumps(class_names)
                        else:
                            logger.warning(f"Unsupported class_names type: {type(class_names)}. Expected list, tuple, or dict.")
                            val_str = str(class_names)
                    except Exception:
                        logger.error(f"Error serializing class_names: {class_names}")

                    data_stats.append(
                        create_data_stat(
                            name="class_names",
                            stat_type="string",
                            shape=[1],
                            value_string=val_str,
                            thumbnail=b""
                        )
                    )
            elif not skip_label_for_request and label is not None:
                # Classification / other scalar-like labels

                # Handle dictionary labels (e.g. detection targets)
                if isinstance(label, dict):
                    def _json_default(o):
                        if isinstance(o, np.ndarray):
                            return o.tolist()
                        if isinstance(o, (np.integer, np.floating)):
                            return o.item()
                        if hasattr(o, "detach") and hasattr(o, "cpu") and hasattr(o, "tolist"):
                            return o.detach().cpu().tolist()  # torch.Tensor, without a hard import
                        return str(o)
                    try:
                        # json.dumps (not str()) so the frontend's JSON.parse
                        # of this 'target' stat never sees Python-repr syntax
                        # (single quotes, tuples, numpy array reprs).
                        dict_value_string = json.dumps(label, default=_json_default)
                    except Exception:
                        dict_value_string = json.dumps({"repr": str(label)})
                    data_stats.append(
                        create_data_stat(
                            name='target',
                            stat_type='string',
                            shape=[1],
                            value_string=dict_value_string,
                            thumbnail=b""
                        )
                    )
                else:
                    # Check if label is NaN (handle both scalars and arrays)
                    if self._is_nan_value(label):
                        pass # Skip NaN labels

                    # Handle scalar labels
                    try:
                        data_stats.append(
                            create_data_stat(
                                name='target',
                                stat_type='scalar',
                                shape=[1],
                                value=[float(label)],
                                thumbnail=b""
                            )
                        )
                    except (ValueError, TypeError):
                        # Fallback for non-scalar labels in non-segmentation tasks
                        try:
                            label_arr = np.asanyarray(label)
                            data_stats.append(
                                create_data_stat(
                                    name='target',
                                    stat_type='array',
                                    shape=list(label_arr.shape),
                                    value=label_arr.astype(float).ravel().tolist(),
                                    thumbnail=b""
                                )
                            )
                        except (ValueError, TypeError) as e:
                            # Use logger.debug to avoid spamming console
                            logger.debug(f"Could not convert label to array: {label}, error: {e}")

            # ====== Step 8: Process predictions ======
            pred = row.get(SampleStatsEx.PREDICTION.value)
            if skip_prediction_for_request:
                pred = None

            if task_type != "classification" and pred is not None:
                t0_pmask = time.time()

                # 3D (point cloud) detection predictions: same dual payload as
                # the GT path (BEV-projected 'bboxes' + raw 'bboxes_3d').
                if is_point_cloud_detection_task(task_type):
                    from weightslab.data.point_cloud_utils import serialize_pointcloud_box_payload
                    pred_bbox_data = {}
                    pred_source = pred
                    if isinstance(pred, dict):
                        # Accept dict-shaped predictions ({'bboxes_3d': ...} or {'bboxes': ...})
                        pred_source = pred.get('bboxes_3d', pred.get('bboxes', pred.get('boxes')))
                    pred_arr = to_numpy_safe(pred_source)
                    if pred_arr is None:
                        try:
                            pred_arr = np.asarray(pred_source, dtype=np.float32)
                        except Exception:
                            pred_arr = np.array([])
                    if pred_arr is not None and pred_arr.size > 0 and pred_arr.ndim == 2:
                        try:
                            pred_bbox_data = serialize_pointcloud_box_payload(ds, pred_arr)
                        except Exception as exc:
                            logger.debug("detection_pointcloud pred serialization failed: %s", exc)
                    t_mask_encode += time.time() - t0_pmask

                    data_stats.append(
                        create_data_stat(
                            name='pred',
                            stat_type='string',
                            shape=[1],
                            value_string=json.dumps(pred_bbox_data) if pred_bbox_data else "",
                            thumbnail=b""
                        )
                    )
                # Handle detection task type with bounding boxes
                elif task_type == "detection":
                    # Send raw bbox data as JSON
                    pred_bbox_data = {}
                    if isinstance(pred, dict):
                        # Support multiple dict key names for bboxes: 'bboxes', 'boxes', 'bb'
                        bbox_key = next((k for k in ['bboxes', 'boxes', 'bb'] if k in pred), None)
                        if bbox_key:
                            try:
                                pred_bboxes = np.asarray(pred.get(bbox_key), dtype=np.float32)
                            except (ValueError, TypeError):
                                # Handle jagged arrays or inconsistent shapes - convert via object dtype
                                try:
                                    # Convert dict to arrays
                                    pred_bboxes = pred.get(bbox_key)
                                    pred_bboxes = np.asarray([bbox for bbox in pred_bboxes if len(bbox)])
                                except (ValueError, TypeError):
                                    pred_bboxes = np.array([])

                            pred_class_ids = pred.get('class_ids')
                            if pred_bboxes.size > 0 and pred_bboxes.ndim >= 2:
                                bbox_format = detect_bbox_format(pred_bboxes)
                                pred_bbox_data = {
                                    "bboxes": pred_bboxes.tolist(),
                                    "class_ids": pred_class_ids.tolist() if hasattr(pred_class_ids, 'tolist') else (pred_class_ids if pred_class_ids is not None else 1),
                                    "format": bbox_format
                                }
                    if not pred_bbox_data and pred is not None:
                        pred_arr = to_numpy_safe(pred)
                        if pred_arr is None:
                            try:
                                pred_arr = np.asarray(pred, dtype=np.float32)
                            except Exception:
                                pred_arr = np.array([])

                        if pred_arr.size > 0 and pred_arr.ndim == 2 and pred_arr.shape[-1] >= 4:
                            # 6-col rows when class+score present (see GT path above).
                            # Format string is legacy-compatible; readers detect extras
                            # by row length.
                            fmt = detect_bbox_format(pred_arr[..., :4])
                            pred_bbox_data = {
                                "bboxes": pred_arr.tolist(),
                                "format": fmt,
                            }

                    t_mask_encode += time.time() - t0_pmask

                    # Send bbox data as JSON string
                    bbox_json = json.dumps(pred_bbox_data) if pred_bbox_data else ""
                    data_stats.append(
                        create_data_stat(
                            name='pred',
                            stat_type='string',
                            shape=[1],
                            value_string=bbox_json,
                            thumbnail=b""
                        )
                    )
                # Handle segmentation task type
                else:
                    try:
                        pred_arr = np.asarray(pred, dtype=np.uint8) if not isinstance(pred, dict) else np.array([])
                        rle_bytes = rle_encode_mask(pred_arr.ravel()) if pred_arr.size > 0 else b""
                        t_mask_encode += time.time() - t0_pmask
                        data_stats.append(
                            create_data_stat(
                                name='pred_mask',
                                stat_type='rle_mask',
                                shape=list(pred_arr.shape) if pred_arr.size > 0 else [],
                                value=[],
                                thumbnail=rle_bytes
                            )
                        )
                        pred_mask_stat_index = len(data_stats) - 1
                        pred_mask_u8 = pred_arr
                    except Exception as e:
                        logger.debug(f"Error processing prediction mask: {e}")
            else:
                # Classification: get prediction from row or dataset
                if pred is None:
                    pass # No prediction to process

                else:
                    # Handle scalar predictions (int, float, or unwrapped from H5)
                    try:
                        pred_val = float(np.asanyarray(pred).item()) if hasattr(pred, '__iter__') else float(pred)
                        data_stats.append(
                            create_data_stat(
                                name='pred',
                                stat_type='scalar',
                                shape=[1],
                                value=[pred_val],
                                thumbnail=b""
                            )
                        )
                    except (ValueError, TypeError):
                        # Fallback for non-scalar predictions in non-segmentation tasks
                        try:
                            pred_ = np.array(pred)
                            data_stats.append(
                                create_data_stat(
                                    name='pred',
                                    stat_type='array',
                                    shape=list(pred_.shape),
                                    value=pred_.astype(float).ravel().tolist(),
                                    thumbnail=b""
                                )
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert prediction to array: {pred}, error: {e}")

            # ====== Step 9: Generate raw data bytes and thumbnail ======
            if request.include_raw_data and dataset is not None:
                try:
                    # New grouped-aware location resolution
                    if hasattr(dataset, "get_physical_location"):
                        ds_idx, member_rank = dataset.get_physical_location(sample_id)
                    else:
                        ds_idx = dataset.get_index_from_sample_id(sample_id)
                        member_rank = 0
                except (KeyError, ValueError, AttributeError):
                    ds_idx = None
                    member_rank = 0
                    logger.debug(f"Missing sample_id={sample_id} in dataset mapping.")

                if ds_idx is not None:
                    # Pass the rank to load_raw_image_array so it plucks the right pixels
                    t0 = time.time()
                    np_img, is_volumetric, original_shape, middle_pil = load_raw_image_array(dataset, ds_idx, rank=member_rank)
                    t_image_load = time.time() - t0
                else:
                    np_img, is_volumetric, original_shape, middle_pil = None, False, [], None

                if middle_pil is not None:
                    original_size = middle_pil.size
                    target_width = original_size[0]
                    target_height = original_size[1]

                    # Full-resolution modal requests use negative resize values (e.g. -100).
                    # We cap them to a configurable max height while preserving aspect ratio.
                    is_full_resolution = (
                        (request.resize_width < 0 and abs(request.resize_width) >= 100)
                        or (request.resize_height < 0 and abs(request.resize_height) >= 100)
                    )

                    # Check for explicit aspect ratio on dataset (favors true ratio over squashed model input)
                    aspect_ratio = original_size[0] / original_size[1] if original_size[1] > 0 else 1.0

                    # Resize logic:
                    if request.resize_width < 0 and request.resize_height < 0:
                        percent = abs(request.resize_width) / 100.0
                        target_width = int(target_width * percent)
                        target_height = int(target_height * percent)
                    elif request.resize_width > 0 and request.resize_height > 0:
                        w_limit, h_limit = request.resize_width, request.resize_height
                        if w_limit / h_limit > aspect_ratio:
                            target_height = h_limit
                            target_width = int(target_height * aspect_ratio)
                        else:
                            target_width = w_limit
                            target_height = int(target_width / aspect_ratio)
                    elif request.resize_width == 0 and request.resize_height == 0:
                        target_height = int(os.environ.get("WL_DEFAULT_THUMBNAIL_SIZE", 180)) # Default full resolution image is 180p on the longest side, but can be overridden by env var
                        target_width = int(target_height * aspect_ratio)

                    if is_full_resolution:
                        max_modal_height = int(
                            os.environ.get(
                                "WL_MODAL_MAX_RESOLUTION",
                                os.environ.get("WL_DEFAULT_THUMBNAIL_SIZE", 360),
                            )
                        )
                        if max_modal_height > 0 and target_height > max_modal_height:
                            scale = max_modal_height / float(target_height)
                            target_height = max_modal_height
                            target_width = int(target_width * scale)

                    # Ensure dimensions are at least 1x1
                    target_width = max(1, target_width)
                    target_height = max(1, target_height)

                    # For thumbnails, avoid backend upsampling. The browser can upscale
                    # to the grid cell size client-side with lower gRPC payload cost.
                    if not is_full_resolution:
                        target_width, target_height = self._clamp_to_downscale_only(
                            target_width,
                            target_height,
                            original_size[0],
                            original_size[1],
                        )

                    # Resize middle slice for thumbnail if requested (maintain aspect ratio)
                    if target_width != original_size[0] or target_height != original_size[1]:
                        # BILINEAR is ~2x faster than LANCZOS and sufficient for small thumbnails;
                        # use LANCZOS only for full-resolution modal views where quality matters.
                        _resample = Image.Resampling.LANCZOS if is_full_resolution else Image.Resampling.BILINEAR
                        middle_pil = middle_pil.resize((target_width, target_height), _resample)

                    # Keep masks in sync with thumbnail size, but only for thumbnail paths.
                    # Full-resolution modal requests keep original mask resolution.
                    if not is_full_resolution:
                        if target_mask_stat_index is not None and target_mask_u8 is not None and target_mask_u8.size > 0:
                            resized_target = resize_mask_nearest(target_mask_u8, target_width, target_height)
                            target_stat = data_stats[target_mask_stat_index]
                            target_stat.shape[:] = list(resized_target.shape)
                            target_stat.thumbnail = rle_encode_mask(resized_target.ravel())

                        if pred_mask_stat_index is not None and pred_mask_u8 is not None and pred_mask_u8.size > 0:
                            resized_pred = resize_mask_nearest(pred_mask_u8, target_width, target_height)
                            pred_stat = data_stats[pred_mask_stat_index]
                            pred_stat.shape[:] = list(resized_pred.shape)
                            pred_stat.thumbnail = rle_encode_mask(resized_pred.ravel())

                    raw_data_bytes, raw_shape, t_image_encode = encode_image_to_raw_bytes(
                        np_img=np_img,
                        middle_pil=middle_pil,
                        original_shape=original_shape,
                        is_volumetric=is_volumetric,
                        is_full_resolution=is_full_resolution,
                        target_width=target_width,
                        target_height=target_height,
                    )

                    data_stats.append(
                        create_data_stat(
                            name='raw_data',
                            stat_type='bytes',
                            thumbnail=raw_data_bytes,
                            shape=raw_shape,
                        )
                    )

                    # Eagerly release intermediate image data to reduce peak memory
                    # across concurrent thread-pool workers.
                    del raw_data_bytes, middle_pil
                    if np_img is not None:
                        del np_img

            # ====== Step 10: Create DataRecord ======
            total_time = time.time() - start_total

            # Attach server-side timing breakdown so the frontend can display E2E metrics
            timing_str = json.dumps({
                "server_total_ms": round(total_time * 1000, 1),
                "image_load_ms": round(t_image_load * 1000, 1),
                "image_encode_ms": round(t_image_encode * 1000, 1),
                "label_convert_ms": round(t_label_convert * 1000, 1),
                "mask_rle_ms": round(t_mask_encode * 1000, 1),
            })
            data_stats.append(
                create_data_stat(
                    name='_timing',
                    stat_type='string',
                    shape=[],
                    value_string=timing_str,
                    thumbnail=b""
                )
            )

            # ====== Step 11: Handle multi-instance signals ======
            instance_signals = row.get('_instance_signals')
            if instance_signals and isinstance(instance_signals, dict):
                # Convert signals dictionary to JSON for UI display
                # Format: {annotation_id: {signal_name: value, ...}}
                signals_json = json.dumps(instance_signals)
                data_stats.append(
                    create_data_stat(
                        name='_instance_signals',
                        stat_type='string',
                        shape=[1],
                        value_string=signals_json,
                        thumbnail=b""
                    )
                )

            record = pb2.DataRecord(sample_id=str(sample_id), data_stats=data_stats)

            return record

        except Exception as e:
            total_time = time.time() - start_total
            logger.error(f"[Sample {row.get(SampleStatsEx.SAMPLE_ID.value, -1)}] X Error after {total_time:.3f}s: {e}", exc_info=True)
            return None

    def _get_categorical_tag_names(self) -> set:
        """Return the set of registered categorical tag names (without prefix)."""
        try:
            if self._df_manager is not None:
                return set(self._df_manager.get_categorical_tags().keys())
        except Exception:
            pass
        return set()

    def _get_unique_tags(self) -> List[str]:
        """Collect unique BOOLEAN tag names present in the tracked datasets.

        Tags are stored as individual columns with prefix "tag:". Categorical
        (multi-value) tags are reported separately via ``categorical_tags`` and
        excluded here so the UI doesn't render them as boolean toggles.
        """
        tags = set()
        try:
            categorical = self._get_categorical_tag_names()
            if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                for col in self._all_datasets_df.columns:
                    if col.startswith(f"{SampleStatsEx.TAG.value}:"):
                        tag_name = col[len(f"{SampleStatsEx.TAG.value}:"):]
                        if tag_name not in categorical:
                            tags.add(tag_name)
        except Exception as e:
            logger.warning(f"Error collecting unique tags: {e}")
        return sorted(list(tags))

    def _get_categorical_tag_defs(self) -> List["pb2.CategoricalTagDef"]:
        """Build CategoricalTagDef protos from the manager's tag registry."""
        defs = []
        try:
            if self._df_manager is not None:
                for name, categories in self._df_manager.get_categorical_tags().items():
                    defs.append(pb2.CategoricalTagDef(name=str(name), categories=[str(c) for c in categories]))
        except Exception as e:
            logger.debug(f"Error building categorical tag defs: {e}")
        return defs

    def _build_success_response(
        self,
        df,
        message: str,
        intent_type=pb2.INTENT_FILTER,
        analysis_result=""
    ) -> pb2.DataQueryResponse:
        """
        Centralized helper so every code path reports counts consistently.

        - number_of_all_samples: all rows in df
        - number_of_discarded_samples: rows with discarded == True (if column exists)
        - number_of_samples_in_the_loop: rows not discarded
        """
        total_count = len(df)
        discarded_count = (
            len(df[df.get(SampleStatsEx.DISCARDED.value, False) == True]) # noqa: E712
            if df is not None and SampleStatsEx.DISCARDED.value in df.columns
            else 0
        )
        in_loop_count = total_count - discarded_count
        unique_tags = self._get_unique_tags()

        # Log query execution
        self._log_audit(
            "query_execute",
            "success",
            {
                "query_type": "pandas" if intent_type == pb2.INTENT_FILTER else "analysis",
                "results_count": in_loop_count,
                "all_samples_count": total_count,
                "discarded_count": discarded_count,
                "message": message[:100],
            },
        )

        return pb2.DataQueryResponse(
            success=True,
            message=message,
            number_of_all_samples=total_count,
            number_of_samples_in_the_loop=in_loop_count,
            number_of_discarded_samples=discarded_count,
            unique_tags=unique_tags,
            categorical_tags=self._get_categorical_tag_defs(),
            agent_intent_type=intent_type,
            analysis_result=analysis_result
        )

    def _parse_direct_query(self, query: str) -> list:
        """
        Parse a simple direct query string into operations list.
        Expected format: "col > val and col2 == val2 sortby col desc"
        This bypasses the LLM agent for deterministic filtering.
        """
        operations = []
        query = query.strip()

        logger.debug(f"[_parse_direct_query] Parsing query: {repr(query)}")

        if query.lower().startswith("sortby "):
            filter_part = None
            sort_part = query[7:].strip()
        elif " sortby " in query.lower():
            parts = query.split(" sortby ", 1)
            filter_part = parts[0].strip() if parts[0].strip() else None
            sort_part = parts[1].strip() if len(parts) > 1 else None
        else:
            filter_part = query
            sort_part = None

        logger.debug(f"[_parse_direct_query] filter_part: {repr(filter_part)}, sort_part: {repr(sort_part)}")

        # Parse filter part
        if filter_part:
            # For now, treat the entire filter as a pandas query expression
            operations.append({
                "function": "df.query",
                "params": {"expr": filter_part}
            })
        # Parse sort part
        if sort_part:
            # Check for Scope flag
            # Format: "col desc scope:global" or "col desc scope:subset"
            sort_scope = "subset" # Default to preserving current context
            view_start = 0
            view_count = 0

            if "scope:global" in sort_part:
                sort_scope = "global"
                sort_part = sort_part.replace("scope:global", "")
            elif "scope:subset" in sort_part:
                sort_scope = "subset"
                sort_part = sort_part.replace("scope:subset", "")
            elif "scope:view" in sort_part:
                sort_scope = "view"
                sort_part = sort_part.replace("scope:view", "")

                # Extract start/count if present
                # Split by space to find params, reassemble sort string
                tokens = sort_part.split()
                clean_tokens = []
                for t in tokens:
                    low_t = t.lower()
                    if low_t.startswith("start:"):
                        try: view_start = int(low_t.split(":")[1])
                        except: pass
                    elif low_t.startswith("count:"):
                        try: view_count = int(low_t.split(":")[1])
                        except: pass
                    else:
                        clean_tokens.append(t)
                sort_part = " ".join(clean_tokens)

                sort_part = " ".join(clean_tokens)

            # If Global scope, we map it to subset behavior (sort current filtered view)
            # We do NOT reset context anymore.
            if sort_scope == "global":
                sort_scope = "subset"

            sort_cols = []
            sort_ascs = []

            # Split by comma to support multiple columns: "tags asc, target desc"
            parts = [p.strip() for p in sort_part.split(',')]

            for p in parts:
                if not p:
                    continue

                ascending = True
                col_raw = p

                # Check for direction suffix
                lower_s = p.lower()
                if lower_s.endswith(" asc"):
                    ascending = True
                    col_raw = p[:-4].strip()
                elif lower_s.endswith(" desc"):
                    ascending = False
                    col_raw = p[:-5].strip()

                # Clean up quotes from column name
                col = col_raw.replace('`', '').strip()

                if col:
                    sort_cols.append(col)
                    sort_ascs.append(ascending)

            if sort_cols:
                logger.debug(f"[_parse_direct_query] Sort: cols={sort_cols}, asc={sort_ascs}, scope={sort_scope}")

                if sort_scope == "view":
                     operations.append({
                        "function": "df.sort_view_slice",
                        "params": {"by": sort_cols, "ascending": sort_ascs, "start": view_start, "count": view_count}
                    })
                elif len(sort_cols) == 1 and sort_cols[0].lower() == 'index':
                    # Optimization for single 'index' sort
                    operations.append({
                        "function": "df.sort_index",
                        "params": {"ascending": sort_ascs[0]}
                    })
                else:
                    # Standard sort
                    operations.append({
                        "function": "df.sort_values",
                        "params": {"by": sort_cols, "ascending": sort_ascs}
                    })
        logger.debug(f"[_parse_direct_query] Parsed into {len(operations)} operations: {operations}")
        return operations

    def _sort_includes_sample_id(self, by) -> bool:
        by_list = [by] if isinstance(by, str) else list(by or [])
        return SampleStatsEx.SAMPLE_ID.value in by_list

    def _sample_id_sortable_series(self, values):
        """Return numeric values for sorting when all sample_ids are integer-like, else string values."""
        numeric = pd.to_numeric(values, errors="coerce")
        arr = np.asarray(numeric)
        if (
            getattr(numeric, "notna", lambda: pd.Series(numeric).notna())().all()
            and np.isfinite(arr).all()
            and np.equal(np.mod(arr, 1), 0).all()
        ):
            return numeric
        return values.astype(str)

    def _sort_values_numeric_aware(self, df: pd.DataFrame, sort_params: dict) -> None:
        """Sort dataframe while treating sample_id as numeric when possible."""
        params = dict(sort_params)
        if params.get("key") is None and self._sort_includes_sample_id(params.get("by")):
            def _key(series: pd.Series):
                if str(getattr(series, "name", "")) == SampleStatsEx.SAMPLE_ID.value:
                    return self._sample_id_sortable_series(series)
                return series

            params["key"] = _key

        df.sort_values(inplace=True, **params)

    @staticmethod
    def _build_agent_eval_globals(df):
        """
        Builds the eval() globals used for agent-generated code (df.modify /
        df.analyze), exposing `origin`/`sample_id` as plain Series whenever
        they live in the index rather than as regular columns, plus a
        reset_index view (`df_reset`). Returns (eval_globals, origin_series)
        so callers can also patch a `df['origin']` backward-compat KeyError.
        """
        origin_series = None
        if SampleStatsEx.ORIGIN.value in df.columns:
            origin_series = df[SampleStatsEx.ORIGIN.value]
        elif isinstance(df.index, pd.MultiIndex) and SampleStatsEx.ORIGIN.value in df.index.names:
            origin_series = pd.Series(
                df.index.get_level_values(SampleStatsEx.ORIGIN.value),
                index=df.index,
            )
        elif df.index.name == SampleStatsEx.ORIGIN.value:
            origin_series = pd.Series(df.index, index=df.index)

        df_reset = safe_reset_index(df)
        eval_globals = {"df": df, "df_reset": df_reset, "np": np, "pd": pd}
        # Expose per-sample signal HISTORY (from the experiment logger's DuckDB
        # store) to agent code, so prompts like "tag samples that never had
        # train_loss below 0.5" work: signal_history('train_loss','min') >= 0.5.
        # Returns a Series aligned to df.index (NaN where a sample has no history).
        eval_globals["signal_history"] = (
            lambda metric, reduce="min": DataService._reduce_signal_history_series(df, metric, reduce)
        )
        if origin_series is not None:
            eval_globals[SampleStatsEx.ORIGIN.value] = origin_series

        sample_id_series = None
        if SampleStatsEx.SAMPLE_ID.value in df.columns:
            sample_id_series = df[SampleStatsEx.SAMPLE_ID.value]
        elif isinstance(df.index, pd.MultiIndex) and SampleStatsEx.SAMPLE_ID.value in df.index.names:
            sample_id_series = pd.Series(
                df.index.get_level_values(SampleStatsEx.SAMPLE_ID.value),
                index=df.index,
            )
        elif df.index.name == SampleStatsEx.SAMPLE_ID.value:
            sample_id_series = pd.Series(df.index, index=df.index)
        if sample_id_series is not None:
            eval_globals[SampleStatsEx.SAMPLE_ID.value] = sample_id_series

        return eval_globals, origin_series

    @staticmethod
    def _eval_agent_code(code: str, eval_globals: dict, origin_series):
        """
        Evaluates agent-generated code with a backward-compat fallback: the
        agent frequently emits `df['origin']` even when `origin` actually
        lives in the index (a MultiIndex level rather than a column), which
        raises a bare KeyError. When that specific KeyError occurs and an
        origin Series is available, retry with `df['origin']` rewritten to
        the plain `origin` name (exposed by `_build_agent_eval_globals`).
        """
        try:
            return eval(code, eval_globals)
        except KeyError as e:
            if str(e).strip("'\"") == SampleStatsEx.ORIGIN.value and origin_series is not None:
                patched_code = re.sub(
                    r"df\[\s*['\"]origin['\"]\s*\]",
                    SampleStatsEx.ORIGIN.value,
                    code,
                )
                return eval(patched_code, eval_globals)
            raise

    @staticmethod
    def _reduce_signal_history_series(df, metric: str, reduce: str = "min"):
        """Per-sample reduction of a signal's full HISTORY, aligned to df.index.

        Backs the ``signal_history(metric, reduce)`` helper exposed to agent code.
        Reads the experiment logger's DuckDB per-sample store (append-only
        ``(metric, hash, sample_id, step, value)``), reduces each sample's whole
        time series (min/max/mean/count), and maps it back onto the current
        dataframe rows by sample_id. Rows whose sample has no recorded history
        get NaN — so, e.g., ``signal_history('train_loss','min') >= 0.5`` is
        False for them (no evidence they stayed above 0.5), never a crash.

        Only populated when signals were logged with ``save_signals(..., log=True)``;
        with no logger / no history it returns an all-NaN Series (a safe no-op).
        """
        from weightslab.backend import ledgers

        # Resolve each row's sample_id (column or index level).
        sample_level = SampleStatsEx.SAMPLE_ID.value
        if sample_level in df.columns:
            sid_values = df[sample_level]
        elif isinstance(df.index, pd.MultiIndex) and sample_level in (df.index.names or []):
            sid_values = pd.Series(df.index.get_level_values(sample_level), index=df.index)
        else:
            sid_values = pd.Series(df.index, index=df.index)

        empty = pd.Series(np.nan, index=df.index)

        try:
            logger_q = ledgers.get_logger()
        except Exception:
            logger_q = None
        if logger_q is None or not hasattr(logger_q, "reduce_per_sample"):
            return empty

        try:
            resolved = logger_q.resolve_graph_name(metric) if hasattr(logger_q, "resolve_graph_name") else metric
        except Exception:
            resolved = metric
        if not resolved:
            return empty

        exp_hash = None
        try:
            cm = ledgers.get_checkpoint_manager()
            if cm is not None and hasattr(cm, "get_current_experiment_hash"):
                exp_hash = cm.get_current_experiment_hash()
        except Exception:
            exp_hash = None

        try:
            reduced = logger_q.reduce_per_sample(resolved, reduce=reduce, exp_hash=exp_hash)
        except Exception as exc:
            logger.debug(f"[Agent] signal_history('{metric}','{reduce}') failed: {exc}")
            return empty
        if not reduced:
            return empty

        mapped = sid_values.astype(str).map(reduced)
        return pd.Series(pd.to_numeric(mapped, errors="coerce").to_numpy(), index=df.index)

    def _resolve_checkpoint_manager(self):
        """Fetch the live CheckpointManager: prefer the experiment context's
        components, fall back to the global ledger. Returns None if unavailable."""
        cm = None
        try:
            if getattr(self, "_ctx", None) is not None:
                self._ctx.ensure_components()
                cm = self._ctx.components.get("checkpoint_manager")
        except Exception:
            cm = None
        if cm is None:
            try:
                from weightslab.backend.ledgers import get_checkpoint_manager
                cm = get_checkpoint_manager()
            except Exception:
                cm = None
        # A ledger miss yields a Proxy(None) placeholder; treat "no save method" as absent.
        if cm is None or not hasattr(cm, "save_model_checkpoint"):
            return None
        return cm

    def _agent_save_checkpoint(self, include_architecture: bool = False) -> str:
        """Agent action: dump a model-weights checkpoint (and, if requested, the
        model architecture) via the live CheckpointManager. Mirrors the manual
        save path: ensure an experiment hash exists first, since every
        ``save_*`` returns None when ``current_exp_hash`` is unset."""
        cm = self._resolve_checkpoint_manager()
        if cm is None:
            return "Action: no checkpoint manager available; nothing was saved."
        try:
            if not getattr(cm, "current_exp_hash", None) and hasattr(cm, "update_experiment_hash"):
                cm.update_experiment_hash()
            saved = []
            if cm.save_model_checkpoint(force_dump_pending=True, update_manifest=True):
                saved.append("weights")
            if include_architecture and hasattr(cm, "save_model_architecture"):
                model = None
                try:
                    model = self._ctx.components.get("model") if getattr(self, "_ctx", None) else None
                except Exception:
                    model = None
                if model is None:
                    try:
                        from weightslab.backend.ledgers import get_model
                        model = get_model()
                    except Exception:
                        model = None
                if model is not None and cm.save_model_architecture(model):
                    saved.append("architecture")
            if not saved:
                return "Action: checkpoint save produced no output (is an experiment hash set?)."
            return f"Action: saved model checkpoint ({' + '.join(saved)})."
        except Exception as e:
            return f"Action: failed to save checkpoint: {e}"

    def _agent_save_data_state(self) -> str:
        """Agent action: snapshot the current data state (per-sample tags +
        discard flags + RNG) via the live CheckpointManager."""
        cm = self._resolve_checkpoint_manager()
        if cm is None or not hasattr(cm, "save_data_snapshot"):
            return "Action: no checkpoint manager available; data state not saved."
        try:
            if not getattr(cm, "current_exp_hash", None) and hasattr(cm, "update_experiment_hash"):
                cm.update_experiment_hash()
            path = cm.save_data_snapshot(force_new_state=True)
            if path:
                return "Action: saved current data state (tags + discard flags)."
            return "Action: data-state save produced no output (is an experiment hash set?)."
        except Exception as e:
            return f"Action: failed to save data state: {e}"

    def _agent_load_experiment(self, exp_hash) -> str:
        """Agent action: load and apply a full experiment state (model +
        weights + data + config) by its experiment hash, via the live
        CheckpointManager's ``load_state`` (the same path the UI reload uses).
        ``load_config=True`` so hyperparameters are restored too."""
        if not exp_hash:
            return ("Action: no experiment hash given; specify which state to load "
                    "(e.g. 'load experiment state <hash>').")
        cm = self._resolve_checkpoint_manager()
        if cm is None or not hasattr(cm, "load_state"):
            return "Action: no checkpoint manager available; cannot load experiment state."
        exp_hash = str(exp_hash)
        try:
            ok = cm.load_state(exp_hash=exp_hash, load_config=True)
            if ok:
                return f"Action: loaded experiment state from {exp_hash[:16]} (model, weights, data, config)."
            return f"Action: could not load experiment state {exp_hash[:16]} (hash not found?)."
        except Exception as e:
            return f"Action: failed to load experiment state {exp_hash[:16]}: {e}"

    def _agent_load_weights(self, step=None, exp_hash=None) -> str:
        """Agent action: load ONLY model weights (no architecture/config/data
        change), optionally at a specific training ``step``. Defaults to the
        current experiment hash when none is given."""
        cm = self._resolve_checkpoint_manager()
        if cm is None or not hasattr(cm, "load_state"):
            return "Action: no checkpoint manager available; cannot load weights."

        # Resolve the hash: explicit arg, else the current experiment.
        if not exp_hash:
            try:
                exp_hash = (cm.get_current_experiment_hash()
                            if hasattr(cm, "get_current_experiment_hash")
                            else getattr(cm, "current_exp_hash", None))
            except Exception:
                exp_hash = getattr(cm, "current_exp_hash", None)
        if not exp_hash:
            return "Action: no experiment hash available to load weights from."
        exp_hash = str(exp_hash)

        target_step = None
        if step is not None:
            try:
                target_step = int(step)
            except (TypeError, ValueError):
                return f"Action: invalid step '{step}'; expected an integer."

        where = f"step {target_step}" if target_step is not None else "latest step"
        try:
            ok = cm.load_state(
                exp_hash=exp_hash,
                load_model=False, load_weights=True,
                load_config=False, load_data=False,
                target_step=target_step,
            )
            if ok:
                return f"Action: loaded model weights ({where}) from {exp_hash[:16]}."
            return f"Action: could not load weights ({where}) from {exp_hash[:16]}."
        except Exception as e:
            return f"Action: failed to load weights ({where}) from {exp_hash[:16]}: {e}"

    # ------------------------------------------------------------------
    # Hyperparameter tuning (agent action)
    # ------------------------------------------------------------------
    # Semantic name -> ordered candidate dotted paths in the HP config. The
    # first candidate that ALREADY EXISTS is used (set_hyperparam auto-creates
    # missing paths, so we must resolve to a real key rather than blindly set).
    _HP_ALIASES = {
        "batch_size": ["data.train_loader.batch_size", "data.batch_size", "batch_size"],
        "learning_rate": ["optimizer.lr", "lr", "optimizer.learning_rate"],
        "dump_ratio": ["experiment_dump_to_train_steps_ratio"],
        "eval_ratio": ["eval_full_to_train_steps_ratio"],
    }
    # Wording -> canonical semantic name.
    _HP_SYNONYMS = {
        "batch_size": "batch_size", "batchsize": "batch_size", "batch": "batch_size",
        "learning_rate": "learning_rate", "learningrate": "learning_rate",
        "lr": "learning_rate", "learning_rate_lr": "learning_rate",
        "dump_ratio": "dump_ratio", "dumping_model_ratio": "dump_ratio",
        "dump_model_ratio": "dump_ratio", "checkpoint_ratio": "dump_ratio",
        "model_dump_ratio": "dump_ratio",
        "eval_ratio": "eval_ratio", "evaluation_ratio": "eval_ratio",
        "eval_full_ratio": "eval_ratio",
    }

    @staticmethod
    def _hp_read_path(hp, dotted):
        """Read a dotted path from the HP Proxy/dict; None if absent."""
        cur = hp
        for part in str(dotted).split('.'):
            if cur is None:
                return None
            try:
                if hasattr(cur, "get"):
                    cur = cur.get(part, None)
                elif isinstance(cur, dict):
                    cur = cur.get(part, None)
                else:
                    return None
            except Exception:
                return None
        return cur

    @classmethod
    def _hp_path_exists(cls, hp, dotted):
        """True only if every segment of the dotted path is actually present
        (so we never resolve to a key set_hyperparam would have to invent)."""
        cur = hp
        for part in str(dotted).split('.'):
            if cur is None:
                return False
            try:
                present = part in cur
            except Exception:
                present = False
            if not present:
                return False
            cur = cls._hp_read_path(cur, part)
        return True

    @classmethod
    def _resolve_hp_path(cls, hp, param):
        """Map a user-facing HP name (semantic word or dotted path) to an
        existing dotted path in the live config, or None if it can't be found."""
        if not param:
            return None
        param = str(param).strip()
        # 1. An explicit dotted path that already exists wins.
        if cls._hp_path_exists(hp, param):
            return param
        # 2. Semantic alias -> first existing candidate path.
        key = param.lower().replace(" ", "_").replace("-", "_")
        canonical = cls._HP_SYNONYMS.get(key)
        for cand in cls._HP_ALIASES.get(canonical, []):
            if cls._hp_path_exists(hp, cand):
                return cand
        # 3. Fallback: search the config tree for a leaf key equal to the last token.
        leaf = key.split(".")[-1]
        return cls._search_hp_leaf(hp, leaf)

    @classmethod
    def _search_hp_leaf(cls, hp, leaf, _prefix=""):
        """Depth-first search for a leaf key exactly matching `leaf`; returns its
        dotted path (shortest match), or None."""
        try:
            keys = list(hp.keys()) if hasattr(hp, "keys") else []
        except Exception:
            keys = []
        # Prefer an exact match at this level.
        for k in keys:
            if str(k).lower() == leaf:
                return f"{_prefix}{k}"
        # Recurse into nested dicts.
        best = None
        for k in keys:
            child = cls._hp_read_path(hp, k)
            if isinstance(child, dict) or hasattr(child, "keys"):
                found = cls._search_hp_leaf(child, leaf, _prefix=f"{_prefix}{k}.")
                if found and (best is None or found.count(".") < best.count(".")):
                    best = found
        return best

    @staticmethod
    def _hp_scalarize(v):
        """Snapshot a live HP value to a plain scalar. Top-level scalars come
        back from the Proxy as a *live* _ValueProxy that would change under us
        after set_hyperparam, so freeze it to a real int/float/str for the
        'was X' message and for type inference."""
        if v is None:
            return None
        # Use type() not isinstance(): a live _ValueProxy masquerades its
        # __class__ as the wrapped builtin, so isinstance(v, int) is True and
        # would let the live proxy through unfrozen. type() sees the real class.
        if type(v) in (bool, int, float, str):
            return v
        if isinstance(v, dict) or hasattr(v, "keys"):
            return v
        try:
            f = float(v)
        except (TypeError, ValueError):
            try:
                return str(v)
            except Exception:
                return v
        return int(f) if f.is_integer() else f

    @staticmethod
    def _coerce_hp_value(new_value, current):
        """Keep the value's type consistent with the existing one: integer HPs
        (batch_size, ratios) stay integers; floats stay floats."""
        try:
            if isinstance(current, bool):
                return bool(new_value)
            if isinstance(current, int) and not isinstance(current, bool):
                return int(round(float(new_value)))
            if isinstance(current, float):
                return float(new_value)
            # Unknown/None current: infer from the new value.
            f = float(new_value)
            return int(f) if float(f).is_integer() else f
        except (TypeError, ValueError):
            return new_value

    def _agent_set_hyperparam(self, param, op="set", value=None) -> str:
        """Agent action: change a hyperparameter in the live (wrapped) HP config.

        ``param`` is a semantic name (``batch_size``/``learning_rate``/
        ``dump_ratio``/``eval_ratio``) or a dotted path; ``op`` is ``set`` (value
        is the new absolute value) or ``scale`` (value is a multiplier, e.g. 1.1
        for "+10%"). The change mutates the shared HP Proxy in place, so training
        reads the new value on its next iteration."""
        from weightslab.backend.ledgers import get_hyperparams, set_hyperparam, resolve_hp_name

        if not param:
            return "Action: no hyperparameter specified."
        if value is None:
            return f"Action: no value given for hyperparameter '{param}'."

        hp_name = resolve_hp_name()
        hp = get_hyperparams(hp_name)
        if hp is None:
            return "Action: no hyperparameters are registered; cannot tune."

        path = self._resolve_hp_path(hp, param)
        if path is None:
            return (f"Action: could not find hyperparameter '{param}' in the current config "
                    "(try an exact dotted path like 'data.train_loader.batch_size').")

        current = self._hp_scalarize(self._hp_read_path(hp, path))
        op = str(op or "set").lower()
        try:
            if op in ("set", "=", "assign"):
                new_value = value
            elif op in ("scale", "multiply", "mul", "*", "increase", "decrease"):
                if current is None:
                    return f"Action: cannot scale '{path}' — it has no current value."
                new_value = float(current) * float(value)
            elif op in ("delta", "add", "increment", "+"):
                if current is None:
                    return f"Action: cannot adjust '{path}' — it has no current value."
                new_value = float(current) + float(value)
            else:
                return f"Action: unknown hyperparameter op '{op}' (use 'set' or 'scale')."

            new_value = self._coerce_hp_value(new_value, current)
            set_hyperparam(name=hp_name, key_path=path, value=new_value)
            # Verify against the wrapped HP (set_hyperparam mutates it in place).
            confirmed = self._hp_scalarize(self._hp_read_path(get_hyperparams(hp_name), path))
            return f"Action: set {path} = {confirmed} (was {current})."
        except Exception as e:
            return f"Action: failed to set hyperparameter '{param}': {e}"

    @classmethod
    def _materialize_hp(cls, v, _depth=0):
        """Recursively convert an HP Proxy / _ValueProxy tree into plain Python
        (dict/list/scalars) so it can be JSON-formatted for display. Read-only."""
        if _depth > 25:
            return str(v)
        if isinstance(v, dict) or hasattr(v, "keys"):
            out = {}
            try:
                for k in v.keys():
                    out[str(k)] = cls._materialize_hp(cls._hp_read_path(v, k), _depth + 1)
            except Exception:
                return str(v)
            return out
        scal = cls._hp_scalarize(v)
        if scal is None or type(scal) in (bool, int, float, str):
            return scal
        if isinstance(scal, (list, tuple)):
            return [cls._materialize_hp(x, _depth + 1) for x in scal]
        return str(scal)

    def _agent_show_config(self, param=None) -> str:
        """Agent action (READ-ONLY): show the whole experiment configuration, or
        the value at a single key/dotted-path (e.g. 'root_log_dir'). Never
        mutates anything."""
        import json as _json
        from weightslab.backend.ledgers import get_hyperparams, resolve_hp_name

        hp_name = resolve_hp_name()
        hp = get_hyperparams(hp_name)
        if hp is None:
            return "Config: no configuration is registered."

        # Specific key requested.
        if param:
            path = self._resolve_hp_path(hp, param)
            if path is None:
                return (f"Config: could not find '{param}' in the configuration "
                        "(ask to 'show the whole configuration' to see available keys).")
            value = self._materialize_hp(self._hp_read_path(hp, path))
            if isinstance(value, (dict, list)):
                try:
                    value = _json.dumps(value, indent=2, default=str)
                except Exception:
                    value = str(value)
            return f"Config: {path} = {value}"

        # Whole config dump.
        materialized = self._materialize_hp(hp)
        if not materialized:
            return "Config: the configuration is empty."
        try:
            text = _json.dumps(materialized, indent=2, default=str, sort_keys=True)
        except Exception:
            text = str(materialized)
        max_len = 6000
        if len(text) > max_len:
            text = text[:max_len] + "\n... (truncated; ask for a specific key, e.g. 'show the root log dir')"
        return f"Configuration ({hp_name}):\n{text}"

    def _apply_agent_operation(self, df, func: str, params: dict) -> str:
        """
        Apply an agent-described operation to df in-place.
        Now supports Actions and Clarifications.
        """
        # --- 1. CLARIFICATION & SCOPE ---
        if func == "out_of_scope":
            return params.get("reason", "I cannot help with that.")

        if func == "clarify":
            # Pass the LLM's question back to the UI
            return params.get("reason", "I need more information to complete this request.")

        # --- 2. ACTIONS (New Capability) ---
        if func.startswith("action."):
            action_name = func.replace("action.", "")

            # Example: "Save Dataset" Action
            if action_name == "save_dataset":
                filename = params.get("filename", "dataset_export")
                # TODO: Implement your actual save logic here
                # e.g., self._stats_store.save_snapshot(df, filename)
                logger.info(f"Action triggered: Saving dataset as {filename}")
                return f"Action: Dataset saved as '{filename}'"

            # Example: "Plot" Action
            elif action_name == "plot_distribution":
                col = params.get("column")
                # TODO: Implement plot logic
                return f"Action: Plotted distribution for {col}"

            # Example: "Compute Natural Sort" Action
            elif action_name == "compute_natural_sort":
                msg = self._compute_natural_sort_stats()
                return f"Action: {msg}"

            # Save a model checkpoint (weights; + architecture when requested).
            elif action_name in ("save_checkpoint", "save_model", "checkpoint", "dump_model"):
                include_arch = bool(params.get("architecture") or params.get("include_architecture"))
                return self._agent_save_checkpoint(include_architecture=include_arch)

            # Save the current data state (tags + discard flags) as a snapshot.
            elif action_name in ("save_data", "save_data_state", "dump_data"):
                return self._agent_save_data_state()

            # Load a full experiment state (model + weights + data [+ config]) by hash.
            elif action_name in ("load_experiment", "load_state", "load_checkpoint"):
                return self._agent_load_experiment(params.get("hash") or params.get("exp_hash"))

            # Load only model weights (optionally at a specific training step).
            elif action_name in ("load_weights", "load_model_weights"):
                return self._agent_load_weights(
                    step=params.get("step"),
                    exp_hash=params.get("hash") or params.get("exp_hash"),
                )

            # Set / scale a hyperparameter (batch size, learning rate, ratios, ...).
            elif action_name in ("set_hyperparam", "set_hyperparameter", "set_hp", "tune_hyperparam"):
                return self._agent_set_hyperparam(
                    param=params.get("param") or params.get("name") or params.get("key_path"),
                    op=params.get("op", "set"),
                    value=params.get("value"),
                )

            # READ-ONLY: show the config / a specific config value (no mutation).
            elif action_name in ("show_config", "get_config", "show_configuration",
                                 "get_hyperparam", "show_hyperparam", "config_info"):
                return self._agent_show_config(
                    param=params.get("param") or params.get("name") or params.get("key_path")
                )

            return f"Action triggered: {action_name} (Not implemented)"

        # --- 3. DATAFRAME MANIPULATION ---
        # A) Agent-driven df.apply_mask (for complex filters)
        if func == "df.apply_mask":
            code = rewrite_boolean_keywords_to_bitwise(params.get("code", ""))
            try:
                eval_globals, origin_series = self._build_agent_eval_globals(df)
                mask = self._eval_agent_code(code, eval_globals, origin_series)
                if isinstance(mask, (pd.Series, np.ndarray, list, pd.Index)):
                    if isinstance(mask, (list, pd.Index)) and not pd.api.types.is_bool_dtype(pd.Series(mask)):
                         kept = df.loc[mask]
                    else:
                         kept = df[mask]

                    df.drop(index=df.index.difference(kept.index), inplace=True)
                    return f"Applied mask: {code}"
                else:
                    return f"Expression returned unsupported type {type(mask).__name__}: {code}"
            except Exception as e:
                logger.error(f"Failed to apply mask {code}: {e}")
                return f"Failed to apply mask: {e}"

        # B) Agent-driven df.query
        if func == "df.query":
            expr = params.get("expr", "")
            try:
                kept = df.query(expr)
                df.drop(index=df.index.difference(kept.index), inplace=True)
                return f"Applied query: {expr}"
            except Exception as e:
                # Fallback to eval if query fails
                try:
                    mask = df.eval(expr, local_dict={"df": df, "np": np, "pd": pd})
                    kept = df[mask]
                    df.drop(index=df.index.difference(kept.index), inplace=True)
                    return f"Applied query (eval): {expr}"
                except Exception as eval_e:
                    logger.error(f"Query failed: {e} | Eval failed: {eval_e}")
                    return f"Failed to filter: {expr}"

        # C) Column Modification (Transform)
        if func == "df.modify":
            col = params.get("col")
            code = rewrite_boolean_keywords_to_bitwise(params.get("code"))

            # Hard safety gate: the agent (and Quick Filters) may create NEW
            # columns freely, but must never overwrite values already recorded
            # in an existing column. Only the `discarded` deny-list flag and
            # `tag:*` boolean columns are writable once they already exist.
            if col in df.columns and col != SampleStatsEx.DISCARDED.value and not str(col).startswith("tag:"):
                msg = (
                    f"Safety Violation: '{col}' already holds recorded data and cannot be "
                    "overwritten. Create a new column instead, or update tag:*/discarded "
                    "control columns."
                )
                logger.warning(msg)
                return msg

            try:
                # 0. Safety Check: If target column exists, check compatibility
                if col in df.columns:
                     # Heuristic: If existing column is not numeric, but code implies math (contains +,-,*,/), warn/block
                     # This prevents accidental string concatenation (e.g. 1.0 + "tag" -> "tag1.0")
                     if not pd.api.types.is_numeric_dtype(df[col]):
                         # Reliance on try-except to catch invalid math is safer than heuristic string checking
                         # because heuristics fail on column names like 'signals//loss'
                         pass

                # 1. Evaluate the expression with safe context.
                # Keep df as-is (no copy), but expose `origin`/`sample_id`
                # whether they are columns or index levels, plus a
                # reset_index view (df_reset) -- see _build_agent_eval_globals.
                eval_globals, origin_series = self._build_agent_eval_globals(df)
                new_values = self._eval_agent_code(code, eval_globals, origin_series)

                # 2. Check for scalar vs series compatibility
                # (Pandas handles most of this, but we ensure robustness)
                if isinstance(new_values, (pd.Series, np.ndarray, list)):
                    if len(new_values) != len(df):
                        # Attempt alignment if it's a Series
                        # ... (existing code)
                        if isinstance(new_values, pd.Series):
                            # Auto-align to df index
                            df[col] = new_values
                        else:
                            return f"Error: Length mismatch. Result has {len(new_values)}, df has {len(df)}"
                    else:
                         df[col] = new_values
                else:
                    # Scalar assignment
                    df[col] = new_values

                # 3. CRITICAL: Persist to Ledger (H5/Disk)
                # We must split the updates by origin and upsert them to the manager
                if self._df_manager is not None:
                    # Create a minimal update dataframe with just the modified column
                    update_payload = df[[col]] # .copy() # Remove copy because memory waste and slowdown

                    # Ensure origin is available for grouping
                    if isinstance(df.index, pd.MultiIndex) and "origin" in df.index.names:
                        # Index is (origin, sample_id) - ideal for grouping
                        for origin, group in update_payload.groupby(level="origin"):
                            # Reset index to just sample_id for upsert
                            # group.index is (origin, sample_id), droplevel(0) gives sample_id
                            clean_group = group.droplevel("origin")
                            clean_group[SampleStatsEx.ORIGIN.value] = origin
                            self._df_manager.upsert_df(clean_group, origin=origin, force_flush=True)

                    elif SampleStatsEx.ORIGIN.value in df.columns:
                        # Origin is a column
                        update_payload[SampleStatsEx.ORIGIN.value] = df[SampleStatsEx.ORIGIN.value]
                        for origin, group in update_payload.groupby(SampleStatsEx.ORIGIN.value):
                            # Ensure index is sample_id
                            if group.index.name != SampleStatsEx.SAMPLE_ID.value:
                                # Try to find sample_id
                                if SampleStatsEx.SAMPLE_ID.value in group.columns:
                                    group = group.set_index(SampleStatsEx.SAMPLE_ID.value)

                            self._df_manager.upsert_df(group, origin=origin, force_flush=True)

                # Explicitly flush to disk to avoid race conditions with _slowUpdateInternals
                if self._df_manager:
                    try:
                        self._df_manager.flush()
                    except Exception as e:
                        logger.warning(f"Flush after modify failed: {e}")

                return f"Modified column '{col}' using: {code}"
            except Exception as e:
                logger.error(f"Modify failed: {e}")
                return f"Failed to modify column {col}: {e}"

        # C) Standard Pandas Ops (drop, sort, head, tail, sample, view_sort)
        if func in {"df.drop", "df.sort_values", "df.sort_index", "df.head", "df.tail", "df.sample", "df.sort_view_slice"}:
            func_name = func.replace("df.", "")
            try:
                if func_name == "sort_view_slice":
                     start = int(params.get("start", 0))
                     count = int(params.get("count", 0))
                     if count <= 0: count = len(df)
                     end = min(start + count, len(df))

                     if start < len(df):
                         logger.debug(f"[sort_view_slice] Sorting slice {start}:{end}")
                         # Extract and sort slice
                         sub_df = df.iloc[start:end] # .copy() # Remove copy because memory waste and slowdown

                         # Apply sort to slice
                         # Filter params for sort_values
                         sort_params = {k: v for k, v in params.items() if k not in ["start", "count"]}

                         by_cols = sort_params.get("by", [])
                         # Handle special 'index' sort case which sort_values doesn't handle if 'index' is not a column
                         if by_cols and len(by_cols) == 1 and by_cols[0] == "index":
                             asc = sort_params.get("ascending", [True])
                             ascending_val = asc[0] if isinstance(asc, list) and asc else True
                             sub_df.sort_index(inplace=True, ascending=ascending_val)
                         else:
                             try:
                                 self._sort_values_numeric_aware(sub_df, sort_params)
                             except (TypeError, ValueError, KeyError) as e:
                                 # Fallback for ambiguity or missing column (if it's in the index)
                                 # Most common case: 'origin' or 'sample_id' are in the index but not columns
                                 if "ambiguous" in str(e).lower() or isinstance(e, KeyError) or "not in index" in str(e).lower():
                                      by = sort_params.get("by")
                                      by_list = [by] if isinstance(by, str) else by

                                      # Check if ANY of the sort columns are in the index levels
                                      index_names = getattr(sub_df.index, 'names', [])
                                      needs_index_sort = any(col in index_names for col in by_list)

                                      if needs_index_sort:
                                          # Robust fallback: temporarily reset index so we can sort by everything at once
                                          # then restore the index. This handles mixed index/column multi-sorts.
                                          try:
                                              # Save index names to restore later
                                              orig_index_names = sub_df.index.names
                                              temp_df = safe_reset_index(sub_df)

                                              # Adjust 'by' if needed (e.g. if 'index' was used, it's now 'sample_id' etc)
                                              # but here columns usually match.
                                              self._sort_values_numeric_aware(temp_df, sort_params)

                                              # Restore index and update sub_df
                                              sub_df = temp_df.set_index(list(orig_index_names))
                                          except Exception as inner_e:
                                              logger.error(f"Mixed sort fallback failed: {inner_e}")
                                              # If that failed, try forcing string keys as last ditch
                                              sub_df.sort_values(inplace=True, **sort_params, key=lambda x: x.astype(str))

                                 # Fallback for mixed types
                                 elif "key" not in sort_params and isinstance(e, TypeError):
                                     sort_params["key"] = lambda x: x.astype(str)
                                     sub_df.sort_values(inplace=True, **sort_params)
                                 else:
                                     raise e

                         # Reassign values (ignoring index alignment to swap rows in place)
                         # We use .values to ensure we just paste the sorted data into these slots
                         df.iloc[start:end] = sub_df.values

                         # CRITICAL: We must also update the index to match the moved data,
                         # otherwise Sample ID X will point to data from Sample ID Y (corruption).
                         try:
                             if isinstance(df.index, pd.MultiIndex):
                                 new_index_values = df.index.to_numpy() # .copy() # Remove copy because memory waste and slowdown
                                 new_index_values[start:end] = sub_df.index.to_numpy()
                                 df.index = pd.MultiIndex.from_tuples(new_index_values, names=df.index.names)
                             else:
                                 idx_name = df.index.name
                                 new_index = df.index.to_numpy() # .copy() # Remove copy because memory waste and slowdown
                                 new_index[start:end] = sub_df.index.to_numpy()
                                 df.index = pd.Index(new_index, name=idx_name)
                         except Exception as e:
                             logger.error(f"Failed to update index in sort_view_slice: {e}")
                             raise e

                     return f"Applied operation: sort_view_slice({start}:{end})"

                if func_name == "drop" and "index" in params:
                    index_to_drop = eval(params["index"], {"df": df, "np": np})
                    df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: drop"

                if func_name == "sort_values":
                    # sample_id may live in the index (single 'sample_id' or the
                    # ('origin', 'sample_id') multi-index). We reset it to a column to
                    # sort it numerically, then ALWAYS restore the SAME index we started
                    # with. Critically, we must NEVER leave the frame on a bare
                    # RangeIndex: self._all_datasets_df is keyed by sample identity, and
                    # a positional index desyncs it from updated_df in
                    # _slowUpdateInternals (intersection() becomes empty → the view
                    # silently stops refreshing).
                    orig_index_names = [n for n in df.index.names if n is not None]

                    def _restore_index():
                        cols = [n for n in orig_index_names if n in df.columns]
                        if cols and not isinstance(df.index, pd.MultiIndex):
                            df.set_index(cols, inplace=True)
                        elif cols and list(df.index.names) != orig_index_names:
                            df.set_index(cols, inplace=True)

                    try:
                        df.reset_index(inplace=True)
                        for junk in ('level_0', 'index'):
                            if junk in df.columns:
                                df.pop(junk)
                        if 'sample_id' in df.columns:
                            df['sample_id'] = df['sample_id'].astype(int)
                        df.sort_values(inplace=True, **params)
                        if 'sample_id' in df.columns:
                            df['sample_id'] = df['sample_id'].astype(str)
                        _restore_index()
                    except (TypeError, ValueError, KeyError) as e:
                        # Restore the index BEFORE retrying so the index-level fallbacks
                        # below operate on the real (sample_id) index, never a RangeIndex.
                        _restore_index()
                        by = params.get("by")
                        if "ambiguous" in str(e).lower() or isinstance(e, KeyError):
                            if isinstance(by, str) and by in getattr(df.index, 'names', []):
                                ascending = params.get("ascending", True)
                                if by == SampleStatsEx.SAMPLE_ID.value and params.get("key") is None:
                                    df.sort_index(level=by, inplace=True, ascending=ascending,
                                                  key=self._sample_id_sortable_series)
                                else:
                                    df.sort_index(level=by, inplace=True, ascending=ascending)
                            elif by in df.columns:
                                df.sort_values(inplace=True, **params, key=lambda x: x)
                            else:
                                raise e
                        elif "key" not in params and isinstance(e, TypeError):
                            logger.warning(f"Sort failed due to type mismatch ({e}). Retrying with string conversion...")
                            params["key"] = lambda x: x.astype(str)
                            try:
                                df.sort_values(inplace=True, **params)
                            except Exception:
                                if isinstance(by, str) and by in getattr(df.index, 'names', []):
                                    df.sort_index(level=by, inplace=True,
                                                  ascending=params.get("ascending", True))
                        else:
                            raise e
                    return "Applied operation: sort_values"

                if func_name == "sort_index":
                    df.sort_index(inplace=True, **params)
                    return "Applied operation: sort_index"

                if func_name == "head":
                    n_raw = params.get("n", 5)
                    # Handle "%" strings
                    if isinstance(n_raw, str) and "%" in n_raw:
                        try:
                            n = int(len(df) * float(n_raw.replace("%", "")) / 100.0)
                        except:
                            n = 5
                    else:
                        n = int(n_raw)

                    if n < len(df):
                        df.drop(index=df.index.difference(df.index[:n]), inplace=True)
                    return f"Applied operation: head({n})"

                if func_name == "tail":
                    n_raw = params.get("n", 5)
                    # Handle "%" strings
                    if isinstance(n_raw, str) and "%" in n_raw:
                        try:
                            n = int(len(df) * float(n_raw.replace("%", "")) / 100.0)
                        except:
                            n = 5
                    else:
                        n = int(n_raw)

                    if n < len(df):
                        df.drop(index=df.index.difference(df.index[-n:]), inplace=True)
                    return f"Applied operation: tail({n})"

                # ... existing sample logic ...

            except Exception as e:
                logger.error(f"Op {func_name} failed: {e}")
                return f"Failed to apply {func_name}: {e}"

        # C2) Cleanup of agent-created scratch columns (is_temporary=True on a
        # transform step). Only ever emitted by the agent itself for a column
        # it just created in this same request, never by the LLM directly, so
        # this cannot be used to delete pre-existing user data.
        if func == "df.drop_column":
            col = params.get("col")
            if not col or col not in df.columns:
                return f"No temporary column '{col}' to remove."
            try:
                df.drop(columns=[col], inplace=True)
                if self._df_manager is not None:
                    try:
                        self._df_manager.drop_column(col)
                    except Exception as e:
                        logger.warning(f"Failed to drop temporary column '{col}' from ledger: {e}")
                return f"Removed temporary column '{col}'"
            except Exception as e:
                logger.error(f"Failed to drop temporary column {col}: {e}")
                return f"Failed to remove temporary column '{col}': {e}"

        # D) Analysis (Read-Only)
        if func == "df.analyze":
            code = params.get("code")
            if not code: return "No code provided"
            if "import " in code or "__" in code: return "Safety Violation"
            code = rewrite_boolean_keywords_to_bitwise(code)
            try:
                # Same eval context as df.modify: exposes `origin`/`sample_id`
                # when they live in the index, plus the df['origin'] backward-
                # compat fallback -- without this, "average loss" on a
                # dataset where origin is an index level raised a bare
                # `Analysis Error: 'origin'` KeyError.
                eval_globals, origin_series = self._build_agent_eval_globals(df)
                result = self._eval_agent_code(code, eval_globals, origin_series)
                return f"Analysis Result: {result}"
            except Exception as e:
                return f"Analysis Error: {e}"

        # E) Model introspection / architecture management (from the agent's model_info/model_action)
        if func in {"model.info", "model.error"}:
            return params.get("text") or params.get("reason") or "No model information available."

        if func in {"model.freeze", "model.reset"}:
            return self._apply_model_action(func.replace("model.", ""), params)

        return "No operation applied"

    # Maps agent model_action names to the existing ManipulateWeights op types
    # (the exact same architecture ops the grid's freeze/reset controls use).
    _MODEL_ACTION_OP_TYPES = {
        "freeze": pb2.WeightOperationType.FREEZE,
        "reset": pb2.WeightOperationType.REINITIALIZE,
    }

    def _apply_model_action(self, action: str, params: dict) -> str:
        """
        Apply a freeze/reset architecture op to the layer(s) the agent
        resolved. Delegates to ModelService.ManipulateWeights so this reuses
        the exact same code path (locking included) as the grid's freeze/reset
        controls.
        """
        op_type = self._MODEL_ACTION_OP_TYPES.get(action)
        if op_type is None:
            return f"Unsupported model action: {action}"

        layer_ids = params.get("layer_ids") or []
        if not layer_ids:
            return "No layers matched the given criteria; nothing was changed."

        model_service = getattr(self, "model_service", None)
        if model_service is None:
            return "Model service is not available; cannot modify architecture."

        neuron_ids = params.get("neuron_ids") or []
        applied = []
        for layer_id in layer_ids:
            weight_op = pb2.WeightOperation(op_type=op_type, layer_id=int(layer_id))
            if neuron_ids:
                weight_op.neuron_ids.extend(
                    pb2.NeuronId(layer_id=int(layer_id), neuron_id=int(n)) for n in neuron_ids
                )
            request = pb2.WeightsOperationRequest(weight_operation=weight_op)
            response = model_service.ManipulateWeights(request, None)
            if not response.success:
                return f"Failed to apply '{action}' to layer {layer_id}: {response.message}"
            applied.append(layer_id)

        # Freeze/reset mutate per-neuron learning rates on the same model
        # object, which the agent's cheap model-schema cache can't detect by
        # identity — force it to rebuild so `frozen` flags aren't stale.
        agent = getattr(self, "_agent", None)
        if agent is not None:
            try:
                agent.invalidate_model_schema()
            except Exception as exc:
                logger.debug("invalidate_model_schema failed: %s", exc)

        return f"Applied '{action}' to layer(s): {applied}"

    # ------------------------------------------------------------------
    # Lock watchdog helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Lock watchdog helpers (build on MonitoredRLock from watchdog/)
    # ------------------------------------------------------------------
    @staticmethod
    def _lock_caller() -> str:
        """Return 'file:function:line' of the nearest non-data_service frame.

        Used to identify WHICH gRPC handler (or other caller) is waiting on /
        holding a lock — visible in DEBUG logs as '[LockWatchdog]' lines.
        """
        import traceback
        frames = traceback.extract_stack()[:-1]
        for f in reversed(frames):
            if "data_service" not in f.filename:
                name = f.filename.rsplit("/", 1)[-1].rsplit(chr(92), 1)[-1]
                return f"{name}:{f.name}:{f.lineno}"
        if frames:
            f = frames[-2]
            return f"{f.name}:{f.lineno}"
        return "unknown"

    from contextlib import contextmanager

    @contextmanager
    def _watched_lock(self, lock_name: str = "_lock"):
        """Acquire self._lock (a MonitoredRLock) with caller-identity logging.

        On acquisition: logs thread name, caller, and wait time.
        On release: logs thread name and hold duration; warns when > 1 s.
        The MonitoredRLock itself exposes holder_tid() / held_duration() to
        the existing gRPC watchdog for stuck-lock detection and recovery.
        """
        thread = threading.current_thread().name
        caller = self._lock_caller()
        t0 = time.time()
        self._lock.acquire()
        waited_ms = (time.time() - t0) * 1000
        logger.debug(
            "[LockWatchdog] %-36s ACQUIRED by %-30s caller=%s (waited %.1f ms)",
            lock_name, thread, caller, waited_ms,
        )
        t_held = time.time()
        try:
            yield
        finally:
            held_ms = (time.time() - t_held) * 1000
            self._lock.release()
            if held_ms > 1000:
                logger.warning(
                    "[LockWatchdog] %-36s RELEASED by %-30s held %.1f ms ← SLOW",
                    lock_name, thread, held_ms,
                )
            else:
                logger.debug(
                    "[LockWatchdog] %-36s RELEASED by %-30s held %.1f ms",
                    lock_name, thread, held_ms,
                )

    # ------------------------------------------------------------------
    # Main update method
    # ------------------------------------------------------------------
    def _bg_view_refresh(self) -> None:
        """Background view rebuild for reader-triggered (non-force) refreshes. Runs the
        real rebuild+swap via force=True OFF the request path, then releases the guard so
        a later stale read can trigger another. Never raises into a request."""
        try:
            self._slowUpdateInternals(force=True)
        except Exception:
            logger.exception("[ViewRefresh] background view refresh failed")
        finally:
            self._refresh_in_flight.release()

    def _slowUpdateInternals(self, force: bool = False, reset_view: bool = False) -> None:
        """Update the internal dataframe view with the latest data from the manager.

        Concurrency model (thundering-herd prevention):
          • Only ONE gRPC worker runs the expensive pull+reindex at a time
            (_update_lock, non-blocking acquire).
          • Workers that lose the race WAIT for the in-progress update to finish
            (_update_done Event) and then return immediately, reusing the fresh
            result — they do NOT queue up to redo the same work.
          • _update_done stays SET when idle so the very first caller always
            proceeds without waiting.

        Lock watchdog:
          Every acquisition / release of _update_lock is logged with the holding
          thread name, caller function, wait time, and hold duration so stuck
          workers are immediately visible in the logs.

        Args:
            force: If True, bypass the 10-second throttle and force an update.
            reset_view: If True, reset user/agent filters and show the full dataset.
        """
        current_time = time.time()
        logger.debug(f"[_slowUpdateInternals] Called with force={force}, reset_view={reset_view}. Last update at {self._last_internals_update_time}, current time {current_time}")

        # Fast throttle — avoids even trying to acquire the lock when data is fresh.
        if not force and self._last_internals_update_time is not None and \
                current_time - self._last_internals_update_time <= 10:
            return

        # --- Non-force (reader-triggered) refresh: run it in the BACKGROUND ---
        # The view is stale, but a reader (grid/histogram/periodic fetch) must NOT block
        # on the multi-second collapse+combine rebuild. Kick a single background refresh
        # (the WL-ViewRefresh thread calls force=True, which does the real rebuild+atomic
        # swap below) and return immediately — the caller reads the current (last-completed)
        # snapshot. If a refresh is already running, just return; the next fetch sees the swap.
        if not force:
            if self._refresh_in_flight.acquire(blocking=False):
                try:
                    threading.Thread(
                        target=self._bg_view_refresh, name="WL-ViewRefresh", daemon=True
                    ).start()
                except Exception:
                    self._refresh_in_flight.release() # never leak the guard
                    logger.exception("[ViewRefresh] failed to start background refresh")
            return

        # --- Try to become the single updater (force path: rebuild inline) ---
        t_wait_start = time.time()
        acquired = self._update_lock.acquire(blocking=False)

        if not acquired:
            # Another worker is already updating. Wait for it to finish (bounded),
            # then return — the caller will read the already-refreshed view.
            thread = threading.current_thread().name
            logger.debug(
                "[LockWatchdog] _update_lock CONTENDED — %s waiting for in-progress update",
                thread,
            )
            self._update_done.wait(timeout=15)
            logger.debug("[LockWatchdog] _update_lock WAIT DONE — %s continuing with fresh view", thread)
            return

        # We won the race.
        waited_ms = (time.time() - t_wait_start) * 1000
        thread = threading.current_thread().name
        caller = self._lock_caller()
        logger.debug(
            "[LockWatchdog] %-36s ACQUIRED by %-30s caller=%s (waited %.1f ms)",
            "_update_lock[_slowUpdateInternals]", thread, caller, waited_ms,
        )
        # Signal to latecomers that an update is now in progress.
        self._update_done.clear()
        t_held_start = time.time()

        try:
            # Re-check throttle now that we hold the lock (avoids double work when two
            # workers raced and the first already updated while the second was acquiring).
            if not force and self._last_internals_update_time is not None and \
                    time.time() - self._last_internals_update_time <= 10:
                return

            updated_df = self._pull_into_all_data_view_df()

            # Guard against init race conditions
            if updated_df is None:
                return

            # Capture a consistent snapshot of the current state
            with self._lock:
                self._is_filtered = not reset_view and self._is_filtered
                is_filtered = self._is_filtered
                current_all_df = self._all_datasets_df

            # Ensure default columns exist
            if self._compute_natural_sort and "signals.defaults.natural" not in updated_df.columns:
                updated_df["signals.defaults.natural"] = np.nan

            if SampleStatsEx.DISCARDED.value not in updated_df.columns:
                updated_df[SampleStatsEx.DISCARDED.value] = False

            if is_filtered and current_all_df is not None:
                # The user has applied a custom view (Filter, Sort, or Aggregation).
                try:
                    # Fast path: the view (current_all_df) and the fresh pull (updated_df)
                    # are normally keyed identically (string sample_id), so a plain
                    # intersection is O(n) and cheap.
                    target_order = current_all_df.index
                    common_indices = target_order.intersection(updated_df.index)

                    # Defensive lazy fallback (only when the fast path finds nothing):
                    # if some operation left the view on a mismatched index dtype
                    # (e.g. an int RangeIndex vs string sample_ids), intersection()
                    # is falsely empty. Retry once with string-normalized keys before
                    # giving up. This extra O(n) pass runs only in that rare case.
                    if len(common_indices) == 0 and len(target_order) and len(updated_df.index):
                        cur_str = target_order.map(str)
                        upd_str = updated_df.index.map(str)
                        if len(cur_str.intersection(upd_str)) > 0:
                            updated_df = updated_df.copy()
                            updated_df.index = upd_str
                            target_order = cur_str
                            common_indices = cur_str.intersection(upd_str)

                    if len(common_indices) > 0:
                        updated_df = updated_df.reindex(target_order)
                    else:
                        # Aggregation/Transformation — skip auto-update.
                        return
                except Exception as e:
                    logger.debug(f"[_slowUpdateInternals] Error matching indices for filtered view: {e}")
                    return

            elif current_all_df is not None and not current_all_df.empty:
                # Standard/Unfiltered View: preserve sticky sort, append new samples.
                if not current_all_df.index.is_monotonic_increasing:
                    try:
                        key_cols = [SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value]

                        old_df_keys = safe_reset_index(current_all_df)
                        new_df_keys = safe_reset_index(updated_df)

                        if not all(col in old_df_keys.columns for col in key_cols) or \
                                not all(col in new_df_keys.columns for col in key_cols):
                            raise KeyError(f"Missing key columns for reindex: required={key_cols}")

                        old_keys = list(old_df_keys[key_cols].itertuples(index=False, name=None))
                        new_keys = list(new_df_keys[key_cols].itertuples(index=False, name=None))

                        new_key_set = set(new_keys)
                        kept_keys = [key for key in old_keys if key in new_key_set]

                        old_key_set = set(old_keys)
                        newly_added_keys = [key for key in new_keys if key not in old_key_set]

                        full_order = kept_keys + newly_added_keys
                        unique_order = pd.MultiIndex.from_tuples(full_order, names=key_cols).unique()

                        keyed_updated = new_df_keys.set_index(key_cols, drop=True)
                        updated_df = keyed_updated.reindex(unique_order)
                    except Exception as e:
                        logger.warning(f"[_slowUpdateInternals] Reindex failed: {e}. Falling back to previous order.")
                        return

            # Deduplicate before saving back to prevent index corruption
            if updated_df.index.has_duplicates:
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]

            # Atomic swap to make the new view available to readers
            self._all_datasets_df = updated_df
            self._last_internals_update_time = current_time

        finally:
            held_ms = (time.time() - t_held_start) * 1000
            self._update_lock.release()
            if held_ms > 1000:
                logger.warning(
                    "[LockWatchdog] %-36s RELEASED by %-30s held %.1f ms ← SLOW",
                    "_update_lock[_slowUpdateInternals]", threading.current_thread().name, held_ms,
                )
            else:
                logger.debug(
                    "[LockWatchdog] %-36s RELEASED by %-30s held %.1f ms",
                    "_update_lock[_slowUpdateInternals]", threading.current_thread().name, held_ms,
                )
            # Unblock all workers that were waiting on this update.
            self._update_done.set()

    def _is_metadata_only_request(self, request) -> bool:
        """True when caller requests metadata columns only, without image payloads."""
        try:
            stats_to_retrieve = list(getattr(request, 'stats_to_retrieve', []) or [])
            include_raw = bool(getattr(request, 'include_raw_data', False))
            include_transformed = bool(getattr(request, 'include_transformed_data', False))
            if include_raw or include_transformed:
                return False
            if stats_to_retrieve:
                return True
            # An empty stats request with no image payload and no resize is a
            # metadata/histogram sweep. Route it through the fast filtered path (which
            # drops heavy blob columns like pred/target) instead of the slow per-sample
            # path that serializes every column. See _build_metadata_only_response.
            resize_w = int(getattr(request, 'resize_width', 0) or 0)
            resize_h = int(getattr(request, 'resize_height', 0) or 0)
            return resize_w <= 0 and resize_h <= 0
        except Exception:
            return False

    def _build_metadata_only_response(self, df_slice: pd.DataFrame, requested_cols=None):
        """Build a DataSamplesResponse of metadata DataRecords from dataframe columns only.

        No dataset/image traversal: the entire df_slice is processed at once using
        vectorized pandas operations rather than dispatching per-sample_id work to
        the thread pool. ``requested_cols`` restricts the columns; when None/empty
        all columns are returned except heavy per-sample blobs. Used by GetMetaData.
        """
        if df_slice is None or df_slice.empty:
            return pb2.DataSamplesResponse(
                success=False,
                message="No metadata rows available",
                data_records=[],
            )

        requested_cols = list(requested_cols or [])
        # NOTE: ORIGIN is intentionally NOT excluded. The histogram (and any caller
        # that needs per-sample split coloring) requests 'origin' explicitly and
        # relies on this fast vectorized path to return it — without this, the client
        # had to fall back to the full per-sample path (image traversal + thread pool),
        # which took ~30s for large datasets. sample_id is already sent as the record
        # id, and task_type carries no per-sample signal, so both stay excluded.
        excluded_cols = {
            SampleStatsEx.SAMPLE_ID.value,
            # SampleStatsEx.TARGET.value,
            # SampleStatsEx.PREDICTION.value,
            SampleStatsEx.TASK_TYPE.value,
        }

        # When no explicit columns are requested (histogram/metadata sweep), default to
        # all columns EXCEPT heavy per-sample blob columns. For dense tasks (e.g. 3D
        # detection) pred/target are large JSON arrays (~310 KB/record) that bloat the
        # response to 100s of MB and silently break the histogram fetch.
        if not requested_cols:
            _HEAVY_BLOB_COLS = {"prediction_raw"}
            requested_cols = [c for c in df_slice.columns if c not in _HEAVY_BLOB_COLS]

        metadata_cols = [
            col for col in requested_cols
            if col in df_slice.columns and col not in excluded_cols
        ]

        if not metadata_cols:
            return pb2.DataSamplesResponse(
                success=False,
                message="Requested metadata columns are not available in dataframe view",
                data_records=[],
            )

        sample_id_col = SampleStatsEx.SAMPLE_ID.value
        tag_prefix = f"{SampleStatsEx.TAG.value}:"

        # -- Vectorized pre-processing: build string matrices via pandas ------
        # Separate tag columns from regular metadata columns for different
        # handling. All heavy conversion is done once on the full column
        # vectors, not per-row.
        tag_cols = [c for c in metadata_cols if c.startswith(tag_prefix)]
        meta_cols = [c for c in metadata_cols if not c.startswith(tag_prefix)]

        sample_ids = df_slice[sample_id_col].tolist()
        n_rows = len(sample_ids)

        # Pre-allocate per-row stat bins – avoids repeated list creation
        row_stats: list[list] = [[] for _ in range(n_rows)]

        # -- Column-wise DataStat construction --------------------------------
        # Build all DataStat objects for one column at a time using list
        # comprehensions (CPython fast-path) and inline pb2.DataStat() to
        # eliminate the create_data_stat wrapper overhead. Then scatter
        # them into the per-row bins. At 1M rows × 10 cols this avoids
        # a 10M-iteration nested Python loop.
        _DataStat = pb2.DataStat # local ref – avoids repeated attr lookup

        for col in meta_cols:
            series = df_slice[col]
            if series.dtype.kind == 'f':
                str_vals = series.round(7).astype(str).str[:512].tolist()
            elif series.dtype.kind == 'b':
                # Booleans (e.g. 'discarded') → "1"/"0" so the UI's boolean/discarded
                # handling — which expects the legacy per-sample "1"/"0" form — keeps
                # working now that metadata is served exclusively by GetMetaData.
                str_vals = series.astype(int).astype(str).tolist()
            else:
                str_vals = series.astype(str).str[:512].tolist()
            # NaN → None
            nan_mask = series.isna()
            if nan_mask.any():
                for pos in nan_mask.values.nonzero()[0]:
                    str_vals[pos] = None
            # Scatter non-None stats into per-row bins
            for i, v in enumerate(str_vals):
                if v is not None:
                    row_stats[i].append(
                        _DataStat(name=col, type="string", shape=[1], value_string=v)
                    )

        categorical_names = self._get_categorical_tag_names()
        for col in tag_cols:
            tag_name = col[len(tag_prefix):]
            series = df_slice[col]
            if tag_name in categorical_names:
                # Categorical tag: send the actual category value (string), skip unset.
                str_vals = series.astype(str).tolist()
                nan_mask = series.isna()
                for i, v in enumerate(str_vals):
                    if not nan_mask.iloc[i] and v not in ("", "nan", "None"):
                        row_stats[i].append(
                            _DataStat(name=col, type="string", shape=[1], value_string=v)
                        )
            else:
                # Boolean tag: presence indicator "1" when True.
                bools = series.astype(bool).tolist()
                for i, b in enumerate(bools):
                    if b:
                        row_stats[i].append(
                            _DataStat(name=col, type="string", shape=[1], value_string="1")
                        )

        # -- Build DataRecord list in one comprehension -----------------------
        _DataRecord = pb2.DataRecord
        data_records = [
            _DataRecord(sample_id=str(sid), data_stats=stats)
            for sid, stats in zip(sample_ids, row_stats)
        ]

        return pb2.DataSamplesResponse(
            success=True,
            message=f"Retrieved {len(data_records)} metadata records (columns: {', '.join(metadata_cols)})",
            data_records=data_records,
        )

    def _get_all_metadata_column_names(self) -> list:
        """Return every metadata column name available across the WHOLE dataset.

        Excludes heavy per-sample blob columns (pred/target) and internal
        bookkeeping columns, matching the column set _build_metadata_only_response
        emits. Order follows the dataframe columns (de-duplicated) so the UI column
        picker stays stable across refreshes.
        """
        try:
            df = self._all_datasets_df
            if df is None or df.empty:
                return []
            _HEAVY_BLOB_COLS = {"pred", "prediction", "prediction_raw", "target"}
            _INTERNAL_COLS = {
                SampleStatsEx.SAMPLE_ID.value,
                SampleStatsEx.TASK_TYPE.value,
                "annotation_id",
                "_instance_signals",
            }
            seen, names = set(), []
            for col in df.columns:
                name = str(col)
                if col in _HEAVY_BLOB_COLS or col in _INTERNAL_COLS:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
            # Include index-level names too (e.g. 'origin' in the (origin, sample_id)
            # multi-index), excluding internal levels like sample_id/annotation_id.
            if isinstance(df.index, pd.MultiIndex):
                index_names = [n for n in (df.index.names or []) if n]
            elif df.index.name:
                index_names = [df.index.name]
            else:
                index_names = []
            for n in index_names:
                name = str(n)
                if n in _HEAVY_BLOB_COLS or n in _INTERNAL_COLS or name in seen:
                    continue
                seen.add(name)
                names.append(name)
            return names
        except Exception as e:
            logger.warning("Error enumerating metadata column names: %s", e)
            return []

    def GetMetaData(self, request, context):
        """Metadata-only retrieval, separated from GetDataSamples.

        Returns:
            - all_metadata_names: every metadata column for the WHOLE dataset
            - grid_records: per-sample metadata for the requested grid slice
            - modal_record: metadata for the open modal sample (by sample_id), if any
        """
        try:
            # Read the current view directly (kept fresh by the same mechanisms
            # GetDataSamples relies on); no forced refresh on the 15s metadata poll.
            all_names = self._get_all_metadata_column_names()
            df = self._all_datasets_df

            if df is None or df.empty:
                return pb2.GetMetaDataResponse(
                    success=False,
                    message="Internal dataframe is empty or not initialized.",
                    all_metadata_names=all_names,
                    grid_records=[],
                )

            # ---- Grid slice metadata (current view order) ----
            grid_records = []
            start = max(0, int(getattr(request, "start_index", 0)))
            count = int(getattr(request, "records_cnt", 0))
            if count > 0:
                try:
                    df_slice = safe_reset_index(df.iloc[start:start + count])
                except IndexError:
                    df_slice = None
                if df_slice is not None and not df_slice.empty:
                    df_slice, _ = self._merge_multi_instance_signals(df_slice)
                    grid_resp = self._build_metadata_only_response(df_slice)
                    if grid_resp.success:
                        grid_records = list(grid_resp.data_records)

            # ---- Modal sample metadata (optional, by sample_id) ----
            modal_record = None
            modal_id = str(getattr(request, "modal_sample_id", "") or "").strip()
            if modal_id:
                try:
                    sid_col = SampleStatsEx.SAMPLE_ID.value
                    matches = None
                    if sid_col in df.columns:
                        matches = df[df[sid_col].astype(str) == modal_id]
                    elif isinstance(df.index, pd.MultiIndex) and sid_col in (df.index.names or []):
                        # sample_id is a multi-index level (origin, sample_id).
                        level_vals = df.index.get_level_values(sid_col).astype(str)
                        matches = df[level_vals == modal_id]
                    else:
                        matches = df[df.index.astype(str) == modal_id]
                    if matches is not None and not matches.empty:
                        modal_df = safe_reset_index(matches.iloc[[0]])
                        modal_df, _ = self._merge_multi_instance_signals(modal_df)
                        modal_resp = self._build_metadata_only_response(modal_df)
                        if modal_resp.success and modal_resp.data_records:
                            modal_record = modal_resp.data_records[0]
                except Exception as e:
                    logger.warning("GetMetaData modal lookup failed for %s: %s", modal_id, e)

            resp = pb2.GetMetaDataResponse(
                success=True,
                message=f"Retrieved {len(grid_records)} metadata records, {len(all_names)} columns",
                all_metadata_names=all_names,
                grid_records=grid_records,
            )
            if modal_record is not None:
                resp.modal_record.CopyFrom(modal_record)
            return resp
        except Exception as e:
            logger.error("Error in GetMetaData: %s", str(e), exc_info=True)
            return pb2.GetMetaDataResponse(
                success=False,
                message=f"Failed to retrieve metadata: {str(e)}",
                all_metadata_names=[],
                grid_records=[],
            )

    def _merge_multi_instance_signals(self, df_slice):
        """Merge per-instance signals into dictionaries for multi-index dataframes.

        When dataframe has (sample_id, annotation_id) multi-index, group instances
        by sample_id and merge per-instance signal columns into dictionaries:
        {annotation_id_0: signal_value_0, annotation_id_1: signal_value_1, ...}

        This allows UI to display all instance signals for a sample while maintaining
        per-sample analysis view.

        Args:
            df_slice: Dataframe slice with potential multi-index

        Returns:
            Tuple of (merged_df, signal_dict_mapping):
            - merged_df: Single row per sample with merged data
            - signal_dict_mapping: Dict mapping sample_id to signal dictionaries
        """
        # Check if multi-index (sample_id, annotation_id)
        if not isinstance(df_slice.index, pd.MultiIndex) or df_slice.index.nlevels < 2:
            return df_slice, {}

        signal_dict_mapping = {}
        merged_rows = []

        # Identify signal columns (those starting with signal:)
        signal_cols = [col for col in df_slice.columns if str(col).startswith('signal:')]

        if not signal_cols:
            # No per-instance signals, just group by sample_id
            return df_slice.drop_duplicates(subset=['sample_id'], keep='first'), {}

        # Group by sample_id
        sample_ids = df_slice.index.get_level_values(0).unique()

        for sample_id in sample_ids:
            sample_rows = df_slice.xs(sample_id, level=0)

            # Handle both Series (single annotation) and DataFrame (multiple annotations)
            if isinstance(sample_rows, pd.Series):
                sample_rows = sample_rows.to_frame().T

            # Take first row as template
            merged_row = sample_rows.iloc[0].copy()

            # Build signal dictionary: {annotation_id: {signal_name: value, ...}}
            signal_dict = {}
            for _, instance_row in sample_rows.iterrows():
                annotation_id = instance_row.get('annotation_id', 0)
                instance_signals = {}

                for signal_col in signal_cols:
                    value = instance_row.get(signal_col)
                    if value is not None:
                        instance_signals[signal_col] = value

                if instance_signals:
                    signal_dict[str(annotation_id)] = instance_signals

            # Store merged signal dictionary if we have signals
            if signal_dict:
                signal_dict_mapping[sample_id] = signal_dict
                # Add merged signals dict to row for UI display
                merged_row['_instance_signals'] = signal_dict

            merged_rows.append(merged_row)

        # Return merged dataframe (one row per sample) and signal mapping
        merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
        return merged_df, signal_dict_mapping

    def _process_get_data_samples(self, request, context):
        """
        Actual implementation of GetDataSamples.

        Two optimisations:
        1. **Preview-cache fast path** – If the preview cache has a 64×64 or less
           thumbnail for the requested sample *and* the request is for a tiny
           resolution (both dims ≤ ``_PREVIEW_CACHE_THRESHOLD``), serve from
           the cache instantly without touching the file system.
        2. **Parallel batch processing** – All samples are submitted to the
           thread pool at once so all 8 workers stay busy. The chunk-size
           env-var ``WL_BATCH_CHUNK_SIZE`` is kept for backward compat but
           the default is now the full request size (all at once).
        """
        _PREVIEW_CACHE_THRESHOLD = 80 # max px to consider a "preview" request
        # Default: process ALL rows at once in the thread pool (workers = 8).
        # Override with WL_BATCH_CHUNK_SIZE to throttle concurrency.
        _BATCH_CHUNK_SIZE = int(os.environ.get("WL_BATCH_CHUNK_SIZE", "0")) # 0 = all at once

        try:
            start_time = time.time()
            logger.debug(
                "GetDataSamples processing: start_index=%s, records_cnt=%s, peer=%s, thread=%s, timestart=%s",
                request.start_index,
                request.records_cnt,
                context.peer(),
                threading.current_thread().name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            # Validate request parameters
            if request.start_index < 0 or request.records_cnt <= 0:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Invalid start_index or records_cnt",
                    data_records=[]
                )

            # Trigger update if needed (it has its own internal locking)
            self._slowUpdateInternals()

            # Atomic snapshot of the current authoritative dataframe
            current_df = self._all_datasets_df

            if current_df is None or current_df.empty:
                logger.warning("Internal dataframe is empty or not initialized.")
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Internal dataframe is empty or not initialized.",
                    data_records=[]
                )

            # Slice the snapshot
            try:
                df_slice = safe_reset_index(current_df.iloc[request.start_index:request.start_index + request.records_cnt])
            except IndexError:
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"Index {request.start_index} out of bounds for dataframe size {len(current_df)}",
                    data_records=[]
                )

            if df_slice.empty:
                logger.warning("No samples found at index %s:%s", request.start_index, request.start_index + request.records_cnt)
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}:{request.start_index + request.records_cnt}",
                    data_records=[]
                )

            # Handle multi-instance signal merging if needed
            df_slice, _ = self._merge_multi_instance_signals(df_slice)

            logger.debug(
                "Retrieving samples from %s to %s", request.start_index, request.start_index + request.records_cnt)

            # NOTE: metadata-only requests are no longer served here. GetDataSamples
            # returns image / label / prediction data only; metadata columns are
            # served by the dedicated GetMetaData RPC.

            # ---- Preview-cache tolerant path -------------------------------
            # For preview-tier requests, serve what is available from cache
            # and compute only cache misses at fixed 64px preview size.
            # This avoids whole-page fallback to full-size processing when
            # the cache is still warming up.
            is_preview_request = (
                bool(getattr(request, "include_raw_data", False))
                and abs(getattr(request, "resize_width", 0)) <= _PREVIEW_CACHE_THRESHOLD
                and abs(getattr(request, "resize_height", 0)) <= _PREVIEW_CACHE_THRESHOLD
            )
            if is_preview_request:
                ordered_preview_records: list[pb2.DataRecord | None] = [None] * len(df_slice)
                missing_rows: list[tuple[int, pd.Series]] = []

                for row_pos, (_, row) in enumerate(df_slice.iterrows()):
                    sid_raw = row.get(SampleStatsEx.SAMPLE_ID.value, -1)
                    try:
                        sid = int(sid_raw)
                    except (TypeError, ValueError):
                        sid = -1

                    rec = self._get_preview_record(sid)
                    if rec is None:
                        missing_rows.append((row_pos, row))
                    else:
                        ordered_preview_records[row_pos] = self._refresh_preview_masks_from_row(rec, row)

                if missing_rows:
                    wait_ms_raw = os.environ.get("WL_PREVIEW_CACHE_WARMUP_WAIT_MS", "100")
                    try:
                        wait_ms = max(0, min(1000, int(wait_ms_raw)))
                    except (TypeError, ValueError):
                        wait_ms = 100

                    if wait_ms > 0 and not self._preview_cache_ready.is_set():
                        self._preview_cache_ready.wait(wait_ms / 1000.0)

                        still_missing_rows: list[tuple[int, pd.Series]] = []
                        for row_pos, row in missing_rows:
                            sid_raw = row.get(SampleStatsEx.SAMPLE_ID.value, -1)
                            try:
                                sid = int(sid_raw)
                            except (TypeError, ValueError):
                                sid = -1

                            rec = self._get_preview_record(sid)
                            if rec is None:
                                still_missing_rows.append((row_pos, row))
                            else:
                                ordered_preview_records[row_pos] = self._refresh_preview_masks_from_row(rec, row)

                        missing_rows = still_missing_rows

                    preview_request = pb2.DataSamplesRequest(
                        start_index=request.start_index,
                        records_cnt=request.records_cnt,
                        include_raw_data=True,
                        include_transformed_data=False,
                        stats_to_retrieve=list(getattr(request, "stats_to_retrieve", []) or []),
                        resize_width=64,
                        resize_height=64,
                    )

                    missing_count = len(missing_rows)
                    missing_chunk = _BATCH_CHUNK_SIZE if _BATCH_CHUNK_SIZE > 0 else missing_count
                    for chunk_start in range(0, missing_count, missing_chunk):
                        chunk_end = min(chunk_start + missing_chunk, missing_count)
                        missing_slice = missing_rows[chunk_start:chunk_end]
                        chunk_tasks = [(row, preview_request, df_slice.columns) for _, row in missing_slice]
                        chunk_results = list(self._data_executor.map(self._process_sample_row, chunk_tasks, timeout=120))

                        for (row_pos, _), rec in zip(missing_slice, chunk_results):
                            if rec is None:
                                continue
                            ordered_preview_records[row_pos] = rec
                            try:
                                self._preview_cache[int(rec.sample_id)] = rec
                            except (TypeError, ValueError):
                                pass

                if all(rec is not None for rec in ordered_preview_records):
                    elapsed = time.time() - start_time
                    ready_count = len(ordered_preview_records)
                    warmed_count = len(missing_rows)
                    logger.debug(
                        "[PreviewCache] Served %d preview records in %.1fms (%d cache miss(es) warmed on-demand)",
                        ready_count,
                        elapsed * 1000,
                        warmed_count,
                    )
                    return pb2.DataSamplesResponse(
                        success=True,
                        message=f"Served {ready_count} preview records ({warmed_count} warmed on-demand)",
                        data_records=[rec for rec in ordered_preview_records if rec is not None],
                    )

                logger.debug(
                    "[PreviewCache] Partial preview assembly failed (%d/%d ready), falling back to standard pipeline",
                    sum(1 for rec in ordered_preview_records if rec is not None),
                    len(ordered_preview_records),
                )

            # ---- Parallel batch processing ---------------------------------
            # Submit ALL rows to the thread pool at once so all 8 workers
            # stay busy. This avoids the old sequential-chunk bottleneck
            # where each sub-batch had to finish before the next started.
            data_records: list = []
            rows_list = list(df_slice.iterrows())
            n_rows = len(rows_list)
            tasks = [(row, request, df_slice.columns) for _, row in rows_list]
            effective_chunk = _BATCH_CHUNK_SIZE if _BATCH_CHUNK_SIZE > 0 else n_rows

            for chunk_start in range(0, n_rows, effective_chunk):
                chunk_end = min(chunk_start + effective_chunk, n_rows)
                chunk_tasks = tasks[chunk_start:chunk_end]

                logger.debug("Processing chunk [%d:%d] of %d at %s",
                             chunk_start, chunk_end, n_rows,
                             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                chunk_results = list(self._data_executor.map(self._process_sample_row, chunk_tasks, timeout=120))
                data_records.extend(r for r in chunk_results if r is not None)

            logger.debug("Completed processing %d records at %s in %.2f seconds",
                         len(data_records),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time)

            if not data_records:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Failed to retrieve samples (all records None)",
                    data_records=[]
                )

            return pb2.DataSamplesResponse(
                success=True,
                message=f"Retrieved {len(data_records)} data records",
                data_records=data_records
            )

        except Exception as e:
            logger.error("Failed to retrieve samples: %s", str(e), exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}\n{traceback.format_exc()}",
                data_records=[]
            )

    def _calculate_tag_column_updates(self, sample_id: int, origin: str, new_tag_name: str, edit_type) -> dict:
        """
        Calculate individual tag column updates based on the edit type.

        Returns a dictionary of column updates like:
        {"tags_tag1": 1, "tags_tag2": 1, ...}

        For EDIT_ACCUMULATE: adds the new tag
        For EDIT_REMOVE: removes the specified tag
        For EDIT_OVERRIDE: replaces all tags with the new value
        """
        tag_updates = {}
        uses_multiindex = self._all_datasets_df is not None and isinstance(self._all_datasets_df.index, pd.MultiIndex)
        # No tag name -> no per-sample update. Whole-column deletes (EDIT_REMOVE with
        # float_value == -1) carry the column in stat_name and an empty string_value;
        # without this guard we'd build a bogus "tag:" column (prefix + empty name) and
        # upsert it right before dropping the real column, leaving an empty "tag:" stub.
        stripped_tag_name = (new_tag_name or "").strip()
        if not stripped_tag_name:
            return tag_updates
        new_tag_name = f'{SampleStatsEx.TAG.value}:{stripped_tag_name}'

        # Get current tags from the in-memory dataframe or df_manager
        existing_tag_value = True # Default to True for new tags
        try:
            if self._all_datasets_df is not None:
                # Read current tag columns from in-memory dataframe
                if uses_multiindex:
                    row = self._all_datasets_df.loc[(origin, sample_id)]
                else:
                    mask = (self._all_datasets_df.index == sample_id) & \
                           (self._all_datasets_df[SampleStatsEx.ORIGIN.value] == origin)
                    if mask.any():
                        row = self._all_datasets_df.loc[mask].iloc[0]
                    else:
                        row = None

                if row is not None:
                    for col in row.index:
                        if col == new_tag_name and row[col]: # If existing, revert the value
                            existing_tag_value = bool(1 - row[col])

        except (KeyError, AttributeError) as e:
            logger.debug(f"Could not read current tags: {e}")

        # Calculate target tags based on edit type
        if edit_type == SampleEditType.EDIT_REMOVE:
            existing_tag_value = False # For removal, we set the tag to False
            target_tags_set = self._parse_tags(new_tag_name)
        else:
            # Override: replace all tags with the new value
            target_tags_set = self._parse_tags(new_tag_name)

        # Create column updates for all target tags
        for tag in target_tags_set:
            tag_updates[tag] = bool(existing_tag_value)

        return tag_updates

    def _parse_tags(self, tag_value: str) -> set:
        """
        Parse a tag string into individual tag names.
        Handles comma, semicolon, or mixed separators.

        Example:
            "tag1,tag2;tag3" → {'tag1', 'tag2', 'tag3'}
        """
        if not tag_value or not isinstance(tag_value, str):
            return set()

        tags = set()
        for tag in tag_value.split(';'):
            for t in tag.split(','):
                clean_tag = t.strip()
                if clean_tag:
                    tags.add(clean_tag)

        return tags


    # ===================
    # RPC Implementations
    # ===================

    def ApplyDataQuery(self, request, context):
        """
        Apply a query on the in-memory dataframe.

        Modes:
          - request.query == "" -> just return counts, do not modify df
          - request.query != "" -> always handled by the agent (natural language path)

        Counts returned:
          - number_of_all_samples: all rows currently in the dataframe
          - number_of_samples_in_the_loop: rows not discarded
          - number_of_discarded_samples: rows with discarded == True
        """

        # Sync context components before processing the query to ensure we have the latest data and state from the ledger
        self._ctx.ensure_components()

        # 1) No query: just report counts (Needs lock for consistency)
        if request.query == "":
            with self._watched_lock("_lock[ApplyDataQuery/counts]"):
                 df = self._all_datasets_df
                 return self._build_success_response(
                    df=df,
                    message=f"Current dataframe has {len(df)} samples",
                )

        try:
            # 2) Check if we should bypass the agent (Quick Filters path)
            if not request.is_natural_language:
                logger.info(
                    "[ApplyDataQuery] BYPASSING AGENT - Direct query execution: %r",
                    request.query,
                )

                # Parse the query directly without agent
                # Expected format: "col > val and col2 == val2 sortby col desc"
                operations = self._parse_direct_query(request.query)

                # Apply operations with lock
                with self._watched_lock("_lock[ApplyDataQuery/ops]"):
                    # Skip the forced full-view rebuild for SORT-ONLY operations. Sorting just
                    # re-orders the existing snapshot, so a fresh collapse+combine (hundreds of
                    # ms on large views, and — being lock-held — contends with the training
                    # thread for multi-second stalls) is unnecessary. Filters/edits still refresh
                    # so they operate on the latest data. The view is frozen on direct queries
                    # anyway (_is_filtered=True), so it wasn't auto-refreshing mid-sort regardless.
                    _SORT_FUNCS = {"df.sort_values", "df.sort_index", "df.sort_view_slice"}
                    is_sort_only = bool(operations) and all(
                        op.get("function") in _SORT_FUNCS for op in operations)
                    if not is_sort_only:
                        self._slowUpdateInternals(force=True) # Refresh internals before applying non-sort operations

                    # Work on a copy to allow concurrent readers to see a consistent state
                    df = self._all_datasets_df # Remove copy because memory waste and slowdown
                    messages = []

                    for op in operations:
                        func = op.get("function")
                        params = op.get("params", {}) or {}
                        msg = self._apply_agent_operation(df, func, params)
                        messages.append(msg)

                    final_message = " | ".join(messages) if messages else "No operation performed"

                    # Atomic swap
                    self._all_datasets_df = df

                    # Direct queries are manipulations -> Freeze the view
                    if operations:
                         self._is_filtered = True

                return self._build_success_response(
                    df=df,
                    message=final_message,
                    intent_type=pb2.INTENT_FILTER
                )

            # Pandas instruction from the user, bypassing LLM agent - execute directly on the dataframe
            elif request.is_natural_language:
                if request.query.lower().replace(" ", "").replace("'''", "\"\"\"").replace("\'\'\'", "\"\"\"").startswith("@\"\"\""):
                    operations = request.query
                    logger.info(f"[ApplyDataQuery] BYPASSING AGENT - Direct DataFrame operation: {request.query[:100]}...")

                    with self._lock:
                        self._all_datasets_df, message = execute_df_operation(self._all_datasets_df, request.query) # in-place operation, or replace previous dataframe
                        logger.info(f"[ApplyDataQuery] Executed direct DataFrame operation. Message: {message}")

                        if operations:
                            self._is_filtered = True

                    return self._build_success_response(
                        df=self._all_datasets_df,
                        message=message,
                        intent_type=pb2.INTENT_FILTER
                    )

                elif request.query.lower().replace("'''", "\"\"\"").replace('\"\"\"', "").replace('\'\'\'', "").replace(" ", "").startswith("@reset") or request.query.lower().replace('\"\"\"', "").replace('\'\'\'', "").replace(" ", "").startswith("@clear"):
                    logger.info(f"[ApplyDataQuery] BYPASSING AGENT - Direct reset/clear operation: {request.query[:100]}...")
                    # Force view reset
                    with self._lock:
                        self._is_filtered = False # Unfreeze view first
                        self._slowUpdateInternals(force=True) # Force update to ensure we have the latest data
                        logger.info(f"[ApplyDataQuery] Force view reset and unfrozen.")

                        return pb2.DataQueryResponse(
                            success=True,
                            message="View has been reset successfully.",
                        )
                elif request.query.lower().replace("'''", "\"\"\"").replace('\"\"\"', "").replace('\'\'\'', "").replace(" ", "").startswith("@overview"):
                    logger.info(f"[ApplyDataQuery] BYPASSING AGENT - Direct overview operation: {request.query[:100]}...")
                    # Force view reset
                    with self._lock:
                        message = generate_overview(self._all_datasets_df)
                        logger.info(f"[ApplyDataQuery] Generated overview.")
                        return pb2.DataQueryResponse(
                            success=True,
                            message=f"Overview of the dataframe:\n{message}",
                        )

                else:
                    # 3) Natural language path - go through agent
                    logger.debug(
                        "[ApplyDataQuery] Using AGENT for natural language query: %r",
                        request.query,
                    )

                    if self._agent is None:
                        return pb2.DataQueryResponse(
                            success=False,
                            message="Natural language queries require agent (not available)",
                        )

                    # Agent translates query text -> operations spec (List[dict])
                    # Executed outside the lock to keep grid responsive during LLM waiting time
                    abort_event = threading.Event()
                    if context:
                        context.add_callback(lambda: abort_event.set())

                    def status_cb(msg: str):
                        logger.debug(f"[ApplyDataQuery] Status: {msg}")

                    operations = self._agent.query(request.query, abort_event=abort_event, status_callback=status_cb)
                    if isinstance(operations, dict): operations = [operations] # Backwards compat
                    if not operations: operations = []

                    with self._lock:
                        self._slowUpdateInternals(force=True)

                        if self._all_datasets_df is None:
                            pulled = self._pull_into_all_data_view_df()
                            self._all_datasets_df = pulled if pulled is not None else pd.DataFrame()

                        df = self._all_datasets_df # .copy() # Remove copy because memory waste and slowdown
                        messages = []
                        intent_type = pb2.INTENT_FILTER
                        analysis_result = ""
                        schema_mutated = False
                        _SCHEMA_MUTATING_FUNCS = {"df.modify", "df.drop_column"}

                        for op in operations:
                            func = op.get("function")
                            params = op.get("params", {}) or {}

                            if params.get("__agent_reset__"):
                                logger.debug("[ApplyDataQuery] Agent requested reset")
                                pulled = self._pull_into_all_data_view_df()
                                df = pulled if pulled is not None else pd.DataFrame()
                                self._is_filtered = False
                                messages.append("Reset view")
                                continue

                            if func in _SCHEMA_MUTATING_FUNCS:
                                schema_mutated = True

                            msg = self._apply_agent_operation(df, func, params)
                            messages.append(msg)

                            if "Clarification needed" in msg or "I need more information" in msg:
                                intent_type = pb2.INTENT_ANALYSIS
                                analysis_result = msg
                            elif msg.startswith("Action:"):
                                intent_type = pb2.INTENT_ANALYSIS
                                analysis_result = msg
                            elif msg.startswith("Analysis Result:"):
                                intent_type = pb2.INTENT_ANALYSIS
                                analysis_result = msg.replace("Analysis Result:", "").strip()
                            elif msg.startswith("Analysis Error:") or msg.startswith("Safety Violation:"):
                                intent_type = pb2.INTENT_ANALYSIS
                                analysis_result = msg

                        # A request that both creates/drops a column AND asks an analysis
                        # question in the same turn must still be reported as a FILTER
                        # intent: the frontend only refreshes the grid/column list (via
                        # GetMetaData) on INTENT_FILTER, so downgrading to INTENT_ANALYSIS
                        # here would silently hide a real schema change from the UI.
                        if schema_mutated:
                            intent_type = pb2.INTENT_FILTER

                        final_message = " | ".join(messages) if messages else "No operation performed"

                        if intent_type == pb2.INTENT_FILTER:
                            if df.index.has_duplicates:
                                df = df[~df.index.duplicated(keep='last')]

                            self._all_datasets_df = df
                            self._is_filtered = True

                        return self._build_success_response(
                            df=df,
                            message=final_message,
                            intent_type=intent_type,
                            analysis_result=analysis_result
                        )

        except Exception as e:
            logger.error(f"ApplyDataQuery: Failed to apply query: {e}", exc_info=True)
            return pb2.DataQueryResponse(
                success=False,
                message=f"Failed to apply query: {str(e)}",
            )

    def GetDataSamples(self, request, context):
        """
        Retrieve samples from the dataframe with their data statistics.
        Implements request deduplication to prevent duplicate concurrent requests.
        """

        try:
            # Process the request directly without deduplication logicj
            return self._process_get_data_samples(request, context)

        except Exception as e:
            logger.error("Error in GetDataSamples: %s", str(e), exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}",
                data_records=[]
            )

    def GetHistogram(self, request, context):
        """Server-side histogram binning of one column (typed RPC).

        Numeric columns: bins the current all-data view by ROW ORDER into
        <= max_bins equal-population bins; each bin carries {min,max,avg,count}
        plus a per-(origin,discarded) sub-bar breakdown.

        Categorical/string columns: counts occurrences per unique value and
        returns CategoricalHistogramBar entries sorted by count descending.
        The response carries is_categorical=True and categorical_bars instead
        of bins (bins will be empty).
        """
        try:
            column = request.column or ""
            max_bins = int(request.max_bins) if request.max_bins > 0 else 512
            df = getattr(self, "_all_datasets_df", None)
            if df is None or df.empty:
                return pb2.HistogramResponse(
                    success=False, message="empty dataframe view", total_rows=0, bins=[])
            df = safe_reset_index(df)
            n = len(df)
            if column not in df.columns:
                return pb2.HistogramResponse(
                    success=False, message=f"column '{column}' not in view",
                    total_rows=n, bins=[])

            origin = (df["origin"].astype(str).to_numpy() if "origin" in df.columns
                      else np.full(n, ""))
            disc = (df["discarded"].astype(bool).to_numpy() if "discarded" in df.columns
                    else np.zeros(n, bool))

            # Detect whether column is categorical (string/object) or numeric.
            # A column is numeric if ANY value coerces to a finite number — even
            # when the pandas dtype is ``object`` (e.g. a loss column holding
            # floats mixed with None/NaN for samples that have no value yet).
            # None/NaN are MISSING data, not a category, so they must never turn
            # a numeric column into a categorical one (which would surface them
            # as a spurious "unset" bar). We therefore treat as categorical only
            # a genuine pandas ``category`` dtype, or a column whose values do
            # not coerce to any numeric value at all (pure strings).
            col_series = df[column]
            numeric_vals = pd.to_numeric(col_series, errors="coerce")
            is_category_dtype = (
                str(col_series.dtype) == "category" or hasattr(col_series, "cat")
            )
            # A genuine numeric dtype (float/int/bool) is ALWAYS numeric — even
            # when every value is NaN (e.g. a loss column no sample has filled
            # yet). Such a column yields an empty numeric histogram, never a
            # spurious "unset" categorical bar.
            is_numeric_dtype = (
                pd.api.types.is_numeric_dtype(col_series) and not is_category_dtype
            )
            is_categorical = (not is_numeric_dtype) and (
                is_category_dtype or not bool(numeric_vals.notna().any())
            )

            if is_categorical:
                # --- Categorical path ---
                labels = col_series.astype(str).where(col_series.notna(), "")
                gf = pd.DataFrame({"l": labels, "o": origin, "d": disc})
                total_count = gf.groupby("l")["l"].count().rename("count")
                sub_map: dict = {}
                for (lbl, d, o), c in gf.groupby(["l", "d", "o"]).size().items():
                    sub_map.setdefault(str(lbl), []).append(
                        pb2.HistogramSubBar(origin=str(o), discarded=bool(d), count=int(c)))
                cat_bars = [
                    pb2.CategoricalHistogramBar(
                        label=str(lbl),
                        count=int(cnt),
                        sub_bars=sub_map.get(str(lbl), []),
                    )
                    for lbl, cnt in total_count.sort_values(ascending=False).items()
                ]
                logger.info("[HistCat] column=%s rows=%d categories=%d",
                            column, n, len(cat_bars))
                return pb2.HistogramResponse(
                    success=True,
                    message=f"categorical histogram {column}: {len(cat_bars)} categories from {n} rows",
                    total_rows=n,
                    bins=[],
                    is_categorical=True,
                    categorical_bars=cat_bars,
                )

            # --- Numeric path (unchanged) ---
            bars = max(1, min(n, max_bins))
            vals = numeric_vals.to_numpy()
            edges = (np.arange(bars + 1) * n) // bars
            bin_of_row = np.searchsorted(edges, np.arange(n), side="right") - 1
            fin = np.isfinite(vals)
            gf = pd.DataFrame({"b": bin_of_row[fin], "v": vals[fin],
                               "o": origin[fin], "d": disc[fin]})
            stats = gf.groupby("b")["v"].agg(["min", "max", "mean", "count"])
            sub_by_bin = {}
            for (b, d, o), c in gf.groupby(["b", "d", "o"]).size().items():
                sub_by_bin.setdefault(int(b), []).append(
                    pb2.HistogramSubBar(origin=str(o), discarded=bool(d), count=int(c)))
            have = stats.index.to_numpy()
            mn, mx, av, cn = (stats["min"].to_numpy(), stats["max"].to_numpy(),
                              stats["mean"].to_numpy(), stats["count"].to_numpy())
            pos = {int(b): i for i, b in enumerate(have)}
            _nan = float("nan")
            bins = []
            for b in range(bars):
                i = pos.get(b)
                if i is None:
                    bins.append(pb2.HistogramBin(
                        min=_nan, max=_nan, avg=_nan, count=0, sub_bars=[]))
                else:
                    bins.append(pb2.HistogramBin(
                        min=float(mn[i]), max=float(mx[i]), avg=float(av[i]),
                        count=int(cn[i]), sub_bars=sub_by_bin.get(b, [])))
            logger.info("[HistBin] column=%s rows=%d bins=%d", column, n, len(bins))
            return pb2.HistogramResponse(
                success=True,
                message=f"histogram {column}: {len(bins)} bins from {n} rows",
                total_rows=n, bins=bins, is_categorical=False, categorical_bars=[])
        except Exception as e:
            logger.error("Error in GetHistogram: %s", str(e), exc_info=True)
            return pb2.HistogramResponse(
                success=False, message=f"histogram failed: {str(e)}",
                total_rows=0, bins=[])

    # Streamed chunk size for GetPointCloud (raw float32 bytes per message),
    # configurable via the WL_POINT_CLOUD_CHUNK_BYTES env var (default 1 MiB).
    _POINT_CLOUD_CHUNK_BYTES = _point_cloud_chunk_bytes()

    def GetPointCloud(self, request, context):
        """Stream one sample's raw point cloud as binary float32 chunks.

        Used by the studio's interactive 3D viewer (task_type "detection_pointcloud").
        The cloud is pad-filtered and optionally downsampled (request
        ``max_points``, capped by env WL_MAX_POINT_CLOUD_POINTS, default 1.5M)
        before serialization; the first chunk carries num_points/num_features
        and the dataset's pc_range so the client can frame the camera.
        """
        from weightslab.data.data_utils import load_raw_point_cloud
        from weightslab.data.point_cloud_utils import (
            get_pc_range, get_point_feature_names, pack_point_cloud,
        )

        try:
            sample_id = int(str(request.sample_id))
            origins = [request.origin] if request.origin else []
            if not origins:
                # No origin provided: scan known loaders for the sample.
                origins = list(get_dataloaders().keys())

            dataset, ds_idx, member_rank = None, None, 0
            for origin in origins:
                candidate = self._get_dataset(origin)
                if candidate is None:
                    continue
                try:
                    if hasattr(candidate, "get_physical_location"):
                        ds_idx, member_rank = candidate.get_physical_location(sample_id)
                    else:
                        ds_idx = candidate.get_index_from_sample_id(sample_id)
                        member_rank = 0
                    dataset = candidate
                    break
                except (KeyError, ValueError, AttributeError, IndexError):
                    continue

            if dataset is None or ds_idx is None:
                yield pb2.PointCloudChunk(
                    success=False,
                    message=f"Sample {request.sample_id} not found (origin={request.origin or 'any'})",
                )
                return

            points = load_raw_point_cloud(dataset, ds_idx, rank=member_rank)
            if points is None:
                yield pb2.PointCloudChunk(
                    success=False,
                    message=f"Sample {request.sample_id} is not a point cloud",
                )
                return

            server_cap = int(os.environ.get("WL_MAX_POINT_CLOUD_POINTS", "1500000"))
            max_points = int(request.max_points) if request.max_points > 0 else server_cap
            max_points = min(max_points, server_cap) if server_cap > 0 else max_points

            data, num_points, num_features = pack_point_cloud(
                points, max_points=max_points, seed=sample_id)
            ds = getattr(dataset, "wrapped_dataset", dataset)
            pc_range = get_pc_range(ds, points) or ()
            feature_names = get_point_feature_names(ds, num_features)

            chunk_size = self._POINT_CLOUD_CHUNK_BYTES
            total_chunks = max(1, (len(data) + chunk_size - 1) // chunk_size)
            for i in range(total_chunks):
                chunk = pb2.PointCloudChunk(
                    success=True,
                    data=data[i * chunk_size:(i + 1) * chunk_size],
                    chunk_index=i,
                    total_chunks=total_chunks,
                )
                if i == 0:
                    chunk.num_points = num_points
                    chunk.num_features = num_features
                    chunk.pc_range.extend(float(v) for v in pc_range)
                    chunk.feature_names.extend(feature_names)
                yield chunk
        except Exception as e:
            logger.error("Error in GetPointCloud: %s", str(e), exc_info=True)
            yield pb2.PointCloudChunk(
                success=False,
                message=f"Failed to retrieve point cloud: {str(e)}",
            )

    def _set_h5_persistence_enabled(self, enabled: bool) -> bool:
        """Toggle dataframe manager H5 persistence flag when available."""
        if self._df_manager is None:
            return False
        if not hasattr(self._df_manager, "_enable_h5_persistence"):
            return False
        try:
            setattr(self._df_manager, "_enable_h5_persistence", bool(enabled))
            return True
        except Exception:
            return False

    def _manual_save_data_state(self, force_enable_h5: bool = False):
        """Persist current dataframe state to H5 and checkpoint JSON snapshot."""
        self._ctx.ensure_components()
        components = self._ctx.components

        if self._df_manager is None:
            return pb2.DataEditsResponse(
                success=False,
                message="Dataframe manager is not available; cannot save data state.",
            )

        h5_enabled = bool(getattr(self._df_manager, "_enable_h5_persistence", False))
        if force_enable_h5:
            toggled = self._set_h5_persistence_enabled(True)
            h5_enabled = bool(getattr(self._df_manager, "_enable_h5_persistence", False))
            if not toggled or not h5_enabled:
                return pb2.DataEditsResponse(
                    success=False,
                    message="Failed to force-enable H5 persistence before manual save.",
                )

        if not h5_enabled:
            return pb2.DataEditsResponse(
                success=False,
                message="H5 writing is disabled. Enable it first (right-click Manual Save -> Force H5 writing).",
            )

        try:
            if hasattr(self._df_manager, "flush"):
                self._df_manager.flush()
            else:
                self._df_manager.flush_if_needed_nonblocking(force=True)
        except Exception as flush_error:
            logger.error(f"[EditDataSample] Manual save flush failed: {flush_error}", exc_info=True)
            return pb2.DataEditsResponse(
                success=False,
                message=f"Failed to flush dataframe to H5: {flush_error}",
            )

        snapshot_saved = False
        checkpoint_manager = components.get("checkpoint_manager")
        if checkpoint_manager is not None and hasattr(checkpoint_manager, "save_data_snapshot"):
            try:
                if not getattr(checkpoint_manager, "current_exp_hash", None):
                    if hasattr(checkpoint_manager, "update_experiment_hash"):
                        checkpoint_manager.update_experiment_hash()
                snapshot_path = checkpoint_manager.save_data_snapshot(force_new_state=True)
                logger.info(f"[EditDataSample] Data snapshot saved to: {snapshot_path}")
                snapshot_saved = snapshot_path is not None
            except Exception as snapshot_error:
                logger.warning(f"[EditDataSample] Manual save snapshot failed: {snapshot_error}")

        self._slowUpdateInternals(force=True)

        if snapshot_saved:
            return pb2.DataEditsResponse(
                success=True,
                message="Data state saved to H5 and checkpoint JSON snapshot.",
            )

        return pb2.DataEditsResponse(
            success=True,
            message="Data state saved to H5 (JSON snapshot not available).",
        )

    def EditDataSample(self, request, context):
        """
        Edit sample metadata (tags and discarded).

        Tags are stored as individual boolean columns (tags_<tagname>) instead of
        a single comma-separated string. This allows for efficient dataframe indexing, grouping,
        and sorting by tags.
        """

        if self._all_datasets_df is None:
            self._initialize_data_service()

        self._ctx.ensure_components()
        components = self._ctx.components

        # Pause training if it's currently running
        trainer = components.get("trainer")
        hp = components.get("hyperparams")
        if trainer:
            logger.info("Pausing training before restore...")
            trainer.pause()
            if "is_training" in hp:
                hp['is_training'] = False
            else:
                hp["is_training"] = False

        if request.stat_name == "__save_data_state__":
            return self._manual_save_data_state(force_enable_h5=False)

        if request.stat_name == "__force_h5_write_and_save__":
            return self._manual_save_data_state(force_enable_h5=True)

        if request.stat_name == "__copy_metadata__":
            source_column = str(request.string_value or "").strip()
            if not source_column:
                return pb2.DataEditsResponse(
                    success=False,
                    message="Missing source metadata name to copy.",
                )

            with self._watched_lock("_lock[EditDataSample/__copy_metadata__]"):
                try:
                    self._slowUpdateInternals()
                    if self._all_datasets_df is None or self._all_datasets_df.empty:
                        return pb2.DataEditsResponse(
                            success=False,
                            message="No dataframe available to copy metadata from.",
                        )

                    if source_column not in self._all_datasets_df.columns and source_column not in self._all_datasets_df.index.names:
                        return pb2.DataEditsResponse(
                            success=False,
                            message=f"Metadata column not found: {source_column}",
                        )

                    checkpoint_manager = components.get("checkpoint_manager")
                    experiment_hash = None
                    if checkpoint_manager is not None and hasattr(checkpoint_manager, "get_current_experiment_hash"):
                        experiment_hash = checkpoint_manager.get_current_experiment_hash()
                    experiment_hash = str(experiment_hash or "current_experiment_hash")

                    self._all_datasets_df = safe_reset_index(self._all_datasets_df)
                    self._all_datasets_df, backend_column_name = duplicate_metadata_column_in_dataframe(
                        self._all_datasets_df,
                        source_column=source_column,
                        experiment_hash=experiment_hash,
                    )

                    if SampleStatsEx.ORIGIN.value not in self._all_datasets_df.columns:
                        return pb2.DataEditsResponse(
                            success=False,
                            message=f"Cannot copy metadata without '{SampleStatsEx.ORIGIN.value}' column in dataframe.",
                        )

                    for origin, origin_df in self._all_datasets_df.groupby(SampleStatsEx.ORIGIN.value, sort=False):
                        update_df = pd.DataFrame(
                            {
                                "sample_id": origin_df.index.astype(str),
                                SampleStatsEx.ORIGIN.value: origin,
                                backend_column_name: origin_df[backend_column_name].values,
                            }
                        ).set_index("sample_id")
                        self._df_manager.upsert_df(update_df, origin=str(origin), force_flush=True)

                    # Force global dataframe to save state to h5 if possible
                    try:
                        self._df_manager.flush_if_needed_nonblocking(force=True)
                    except Exception as flush_error:
                        logger.warning(f"[EditDataSample] Flush after metadata clone failed: {flush_error}")

                    self._slowUpdateInternals(force=True)

                    return pb2.DataEditsResponse(
                        success=True,
                        message=f"Saved metadata '{source_column}' as '{backend_column_name}'.",
                    )
                except Exception as e:
                    logger.error(f"[EditDataSample] Failed to copy metadata column: {e}", exc_info=True)
                    return pb2.DataEditsResponse(
                        success=False,
                        message=f"Failed to copy metadata column: {str(e)}",
                    )

        if request.stat_name == "__delete_metadata__":
            target_column = str(request.string_value or "").strip()
            if not target_column:
                return pb2.DataEditsResponse(
                    success=False,
                    message="Missing metadata name to delete.",
                )

            with self._watched_lock("_lock[EditDataSample/delete-col]"):
                try:
                    self._slowUpdateInternals()
                    if self._all_datasets_df is None or self._all_datasets_df.empty:
                        return pb2.DataEditsResponse(
                            success=False,
                            message="No dataframe available to delete metadata from.",
                        )

                    if target_column not in self._all_datasets_df.columns:
                        return pb2.DataEditsResponse(
                            success=False,
                            message=f"Metadata column not found: {target_column}",
                        )

                    if is_protected_metadata_name(target_column):
                        return pb2.DataEditsResponse(
                            success=False,
                            message=f"Cannot delete protected metadata field: {target_column}",
                        )

                    if not is_copy_metadata_column_name(target_column):
                        return pb2.DataEditsResponse(
                            success=False,
                            message=f"Only copied metadata can be deleted: {target_column}",
                        )

                    if self._df_manager is not None:
                        self._df_manager.drop_column(target_column)
                        try:
                            self._df_manager.flush_if_needed_nonblocking(force=True)
                        except Exception as flush_error:
                            logger.warning(f"[EditDataSample] Flush after metadata delete failed: {flush_error}")

                    if target_column in self._all_datasets_df.columns:
                        self._all_datasets_df = self._all_datasets_df.drop(columns=[target_column])

                    # Kick a background view-refresh (non-blocking) — the in-memory view
                    # is already consistent after the drop above, so blocking inline rebuild
                    # is unnecessary and causes the gRPC response to stall for 5-10 s.
                    self._slowUpdateInternals()

                    return pb2.DataEditsResponse(
                        success=True,
                        message=f"Deleted metadata '{target_column}'",
                    )
                except Exception as e:
                    logger.error(f"[EditDataSample] Failed to delete metadata column: {e}", exc_info=True)
                    return pb2.DataEditsResponse(
                        success=False,
                        message=f"Failed to delete metadata column: {str(e)}",
                    )

        if request.stat_name == "__discard_by_tag__":
            tag_name = str(request.string_value or "").strip()
            if not tag_name:
                return pb2.DataEditsResponse(
                    success=False,
                    message="Missing tag name for discard-by-tag operation.",
                )

            with self._watched_lock("_lock[EditDataSample/__discard_by_tag__]"):
                try:
                    self._slowUpdateInternals()
                    if self._all_datasets_df is None or self._all_datasets_df.empty:
                        return pb2.DataEditsResponse(
                            success=False,
                            message="No dataframe available.",
                        )

                    tag_col = f"{SampleStatsEx.TAG.value}:{tag_name}"
                    if tag_col not in self._all_datasets_df.columns:
                        self._slowUpdateInternals()
                    df = safe_reset_index(self._all_datasets_df)
                    if tag_col not in df.columns:
                        return pb2.DataEditsResponse(
                            success=True,
                            message=f"No samples found with tag '{tag_name}'.",
                        )

                    tagged = df[df[tag_col] == 1]
                    if tagged.empty:
                        return pb2.DataEditsResponse(
                            success=True,
                            message=f"No samples found with tag '{tag_name}'.",
                        )

                    for origin, origin_df in tagged.groupby(SampleStatsEx.ORIGIN.value, sort=False):
                        sample_ids = origin_df.index.astype(str).tolist()
                        rows = [
                            {"sample_id": sid, SampleStatsEx.ORIGIN.value: origin, SampleStatsEx.DISCARDED.value: True}
                            for sid in sample_ids
                        ]
                        df_update = pd.DataFrame(rows).set_index("sample_id")
                        self._df_manager.upsert_df(df_update, origin=origin, force_flush=True)

                    count = len(tagged)
                    self._slowUpdateInternals(force=True)
                    return pb2.DataEditsResponse(
                        success=True,
                        message=f"Discarded {count} samples with tag '{tag_name}'.",
                    )
                except Exception as e:
                    logger.error(f"[EditDataSample] discard_by_tag failed: {e}", exc_info=True)
                    return pb2.DataEditsResponse(
                        success=False,
                        message=f"Failed to discard by tag: {str(e)}",
                    )

        if not request.stat_name or not request.stat_name.startswith(SampleStatsEx.TAG.value) and request.stat_name not in [SampleStatsEx.DISCARDED.value]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags', 'discarded', '__copy_metadata__', '__delete_metadata__', '__save_data_state__', '__force_h5_write_and_save__', and '__discard_by_tag__' edits are supported.",
            )

        # =====================================================================
        # Process tag edits using the new column-based tag system
        # =====================================================================
        with self._watched_lock("_lock[EditDataSample/tag-discard]"):
            try:
                if request.samples_ids and self._df_manager is not None:
                    updates_by_origin = {}
                    is_categorical = bool(getattr(request, "is_categorical", False))
                    is_tag_request = (not is_categorical) and (
                        request.stat_name == SampleStatsEx.TAG.value
                        or request.stat_name.startswith(SampleStatsEx.TAG.value)
                    )

                    # Categorical tag set/clear: normalize column + (re)declare categories once.
                    cat_tag_col = None
                    cat_value = None
                    if is_categorical:
                        pref = f"{SampleStatsEx.TAG.value}:"
                        raw = (request.stat_name or "").strip()
                        cat_tag_name = raw[len(pref):] if raw.startswith(pref) else raw
                        cat_tag_col = f"{pref}{cat_tag_name}"
                        declared = [c for c in request.categories if c]
                        if declared:
                            self._df_manager.register_categorical_tag(cat_tag_name, declared)
                        if request.type != SampleEditType.EDIT_REMOVE:
                            cat_value = request.string_value.strip() if request.string_value else None
                            if cat_value:
                                self._df_manager.register_categorical_tag(cat_tag_name, [cat_value])

                    for sid, origin in zip(request.samples_ids, request.sample_origins):
                        sid_value = str(sid)
                        # =====================
                        # CATEGORICAL TAG EDITS
                        # =====================
                        if is_categorical:
                            if origin not in updates_by_origin:
                                updates_by_origin[origin] = {}
                            if sid_value not in updates_by_origin[origin]:
                                updates_by_origin[origin][sid_value] = {
                                    "sample_id": sid_value,
                                    SampleStatsEx.ORIGIN.value: origin,
                                }
                            # EDIT_REMOVE clears the value (None); otherwise set the chosen category.
                            updates_by_origin[origin][sid_value][cat_tag_col] = cat_value

                        # =========
                        # TAG EDITS
                        # =========
                        elif is_tag_request:
                            # Calculate tag column updates based on edit type
                            tag_updates = self._calculate_tag_column_updates(
                                sid,
                                origin,
                                request.string_value,
                                request.type
                            )

                            # Add all tag column updates for this sample
                            if origin not in updates_by_origin:
                                updates_by_origin[origin] = {}
                            if sid_value not in updates_by_origin[origin]:
                                updates_by_origin[origin][sid_value] = {
                                    "sample_id": sid_value,
                                    SampleStatsEx.ORIGIN.value: origin,
                                }

                            # Merge tag column updates into the sample's updates
                            updates_by_origin[origin][sid_value].update(tag_updates)

                        # =================
                        # DENY LISTED EDITS
                        # =================
                        else:
                            # Deny_listed
                            if origin not in updates_by_origin:
                                updates_by_origin[origin] = {}
                            updates_by_origin[origin][sid_value] = {
                                "sample_id": sid_value,
                                SampleStatsEx.ORIGIN.value: origin,
                                request.stat_name: bool(request.bool_value),
                            }

                    # Upsert all tag column updates into the global dataframe
                    for origin, samples in updates_by_origin.items():
                        rows = list(samples.values())
                        df_update = pd.DataFrame(rows).set_index("sample_id")
                        self._df_manager.upsert_df(df_update, origin=origin, force_flush=True)

                    # Auto-cleanup: if EDIT_REMOVE was used on a tag, delete the entire column immediately
                    if (is_tag_request or is_categorical) and request.type == SampleEditType.EDIT_REMOVE and request.float_value == -1:
                        column_name = cat_tag_col if is_categorical else request.stat_name.strip()
                        if is_categorical and cat_tag_col:
                            # Drop the categorical tag from the registry as well.
                            try:
                                self._df_manager._categorical_tags.pop(cat_tag_col[len(f"{SampleStatsEx.TAG.value}:"):], None)
                                self._df_manager._persist_tag_registry()
                            except Exception:
                                pass
                        if column_name:
                            logger.info(f"[EditDataSample] Deleting tag column: {column_name}")
                            # Delete the column from storage
                            self._df_manager.drop_column(column_name)

                            # Remove from in-memory dataframe if it exists
                            if self._all_datasets_df is not None:
                                if column_name in self._all_datasets_df.columns:
                                    self._all_datasets_df = self._all_datasets_df.drop(columns=[column_name])

                    # Reload dataframe to reflect all changes without destroying current sort/view
                    self._slowUpdateInternals(force=True)

                # Prevent _slowUpdateInternals from automatically overwriting our edits with stale data
                self._last_internals_update_time = time.time()

                # Log audit event for data edits
                if is_tag_request:
                    action_type = "tag_add" if request.type == SampleEditType.EDIT_ACCUMULATE else "tag_remove"
                    self._log_audit(
                        action_type,
                        "success",
                        {
                            "tag_name": request.string_value,
                            "samples_affected": len(request.samples_ids),
                            "sample_ids": list(request.samples_ids),
                            "origins": list(request.sample_origins),
                        },
                    )
                else:
                    # This is a discard/restore operation
                    action_type = "sample_discard" if request.bool_value else "sample_restore"
                    self._log_audit(
                        action_type,
                        "success",
                        {
                            "samples_affected": len(request.samples_ids),
                            "sample_ids": list(request.samples_ids),
                            "origins": list(request.sample_origins),
                        },
                    )

                return pb2.DataEditsResponse(
                    success=True,
                    message=f"Edited {len(request.samples_ids)} samples",
                )

            except Exception as e:
                logger.error(f"[EditDataSample] Failed to edit samples: {e}", exc_info=True)
                return pb2.DataEditsResponse(
                    success=False,
                    message=f"Failed to edit samples: {str(e)}",
                )

    def GetDataSplits(self, request, context):
        """
        Return the list of available dataset splits (train, test, val, etc.)
        Extracted from the global dataframe's origin column.
        """

        try:
            if context is not None and not context.is_active():
                return pb2.DataSplitsResponse(success=False, split_names=[])

            # IMPORTANT: keep lock ordering consistent (_update_lock -> _lock).
            # Calling _slowUpdateInternals() while holding _lock can deadlock
            # with concurrent readers/writers under high UI refresh pressure.
            self._slowUpdateInternals()

            if context is not None and not context.is_active():
                return pb2.DataSplitsResponse(success=False, split_names=[])

            split_names = []
            with self._lock:
                if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                    if SampleStatsEx.ORIGIN.value in self._all_datasets_df.columns:
                        raw_splits = self._all_datasets_df[SampleStatsEx.ORIGIN.value].unique().tolist()
                        split_names = sorted([str(s) for s in raw_splits if s is not None and pd.notna(s)])
                    elif isinstance(self._all_datasets_df.index, pd.MultiIndex):
                        if SampleStatsEx.ORIGIN.value in self._all_datasets_df.index.names:
                            raw_splits = self._all_datasets_df.index.get_level_values(SampleStatsEx.ORIGIN.value).unique().tolist()
                            split_names = sorted([str(s) for s in raw_splits if s is not None and pd.notna(s)])
            logger.info(f"GetDataSplits returning: {split_names}")

            return pb2.DataSplitsResponse(
                success=True,
                split_names=split_names
            )

        except Exception as e:
            logger.error(f"GetDataSplits failed: {e}", exc_info=True)
            return pb2.DataSplitsResponse(
                success=False,
                split_names=[]
            )
