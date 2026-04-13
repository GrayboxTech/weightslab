import io
import time
import logging
import os
import struct
import traceback
import threading
import json
import re

import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2

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
    logger.warning("OmegaConf not found, DictConfig will be treated as dict. Install OmegaConf for full functionality.")

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.proto.experiment_service_pb2 import SampleEditType
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.backend.ledgers import get_dataloaders, get_dataframe, list_signals, Proxy
from weightslab.data.data_utils import load_label, load_raw_image, to_numpy_safe
from weightslab.trainer.trainer_tools import execute_df_operation, generate_overview, encode_image_to_raw_bytes
from weightslab.data.data_utils import load_raw_image_array

# Image encoding / mask compression / proto helpers (extracted)
from weightslab.trainer.services.data_image_utils import (
    rle_encode_mask,
    create_data_stat,
    generate_thumbnail,
    encode_image_webp,
    resize_mask_nearest,
)


# Get global logger
logger = logging.getLogger(__name__)


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


def build_metadata_copy_column_names(existing_columns, experiment_hash: str, source_name: str):
    """Build backend/ui copied metadata names with incrementing suffix _1, _2, ..."""
    exp_hash = str(experiment_hash or "current_experiment_hash").strip() or "current_experiment_hash"
    normalized_source = normalize_metadata_copy_source_name(source_name, exp_hash)
    existing_iterable = [] if existing_columns is None else existing_columns
    existing = {str(col) for col in existing_iterable}

    index = 1
    while True:
        copy_name = f"{normalized_source}_{index}@{exp_hash}"
        if copy_name not in existing:
            return copy_name
        index += 1


def duplicate_metadata_column_in_dataframe(df: pd.DataFrame, source_column: str, experiment_hash: str):
    """Return a dataframe copy with one duplicated metadata column using experiment-hash naming."""
    if source_column not in df.columns:
        raise KeyError(f"Source metadata column not found: {source_column}")

    copy_name = build_metadata_copy_column_names(df.columns, experiment_hash, source_column)
    df[copy_name] = df[source_column]
    return df, copy_name


def is_copy_metadata_column_name(column_name: str) -> bool:
    name = str(column_name or "").strip()
    return bool(re.match(r".+_\d+@.+$", name))


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


class DataService:

    """
    Data service helpers + RPCs (for weights_studio UI).

    Images are sent over gRPC as bytes (JPEG) for simplicity and correctness.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.RLock()
        self._update_lock = threading.Lock()
        self._df_manager = get_dataframe()

        # init references to the context components
        self._ctx.ensure_components()

        # Resolve root log directory and H5 path for data storage
        self._root_log_dir = self._resolve_root_log_dir()
        self._h5_path = self._resolve_h5_path()
        self._stats_store = H5DataFrameStore(self._h5_path) if self._h5_path else None

        # Check hyperparameters for compute_natural_sort flag (default: False)
        # Users can enable it by setting compute_natural_sort=True in their hyperparameters.
        hp = self._ctx.components.get("hyperparams") if self._ctx and self._ctx.components else None
        hp_dict = hp.get() if Proxy.is_proxy(hp) else (hp if isinstance(hp, dict) else {})  # is it already a proxy ?
        self._compute_natural_sort = bool((hp_dict or {}).get("compute_natural_sort", False))

        # In-memory dataframe view of all datasets combined (streamed to UI)
        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = 0.0

        # Shared thread pool for data processing (avoid thread explosion)
        # Size: min(CPU cores * 2, 16) to balance concurrency without excessive threading
        self._data_executor = futures.ThreadPoolExecutor(
            thread_name_prefix="WL-DataProcessing",
            max_workers=8
        )

        self._is_filtered = False  # Track if the current view is filtered/modified by user
        # logger.info("[DataService] Skipping expensive startup computations (aspect ratio, natural sort, signals).")
        # These should be triggered on-demand or run in background to avoid blocking training start.
        self._deduce_and_set_aspect_ratios()

        if self._compute_natural_sort:
           self._compute_natural_sort_stats()

        # self._compute_custom_signals()
        self._is_filtered = False  # Track if the current view is filtered/modified by user

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
            self._preview_cache_ready.set()  # No preload → mark immediately ready

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
                    • raw_data  (bytes) — 64×64 or less or less WebP thumbnail
                    • target    (rle_mask) — RLE-encoded GT mask resized to 64×64 or less or less
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

            PREVIEW_SIZE = 64  # fixed low-res dimension
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

                    # --- GT mask ---
                    label = load_label(dataset, sample_id)
                    if label is not None:
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
                    if pred is not None:
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

    def _deduce_and_set_aspect_ratios(self):
        """Automatically deduce and set aspect_ratio for all registered datasets.

        It loads the first raw image from each dataset to determine the
        canonical aspect ratio, then monkey-patches the 'aspect_ratio'
        attribute onto the dataset object if it's not already set.
        """
        try:
            from weightslab.data.data_utils import load_raw_image
            from weightslab.backend.ledgers import get_dataloaders

            loaders_dict = get_dataloaders()
            for name, loader in loaders_dict.items():
                if not loader or not hasattr(loader, "dataset"):
                    continue

                # Unwrap to find the base dataset
                dataset = loader.dataset
                ds = getattr(dataset, "wrapped_dataset", dataset)

                # Skip if already set manually
                if hasattr(ds, "aspect_ratio") and ds.aspect_ratio is not None:
                    logger.debug(f"[DataService] Dataset '{name}' already has aspect_ratio={ds.aspect_ratio}")
                    continue

                # Load first image to deduce ratio
                try:
                    if len(dataset) > 0:
                        pil_img = load_raw_image(dataset, 0)
                        if pil_img:
                            w, h = pil_img.size
                            ratio = w / h if h > 0 else 1.0
                            ds.aspect_ratio = ratio
                            logger.info(f"[DataService] Deduced aspect_ratio={ratio:.2f} for dataset '{name}' from first sample.")
                except Exception as e:
                    logger.debug(f"[DataService] Failed to deduce aspect_ratio for dataset '{name}': {e}")

        except ImportError:
            pass

        except Exception as e:
            logger.warning(f"[DataService] Unexpected error during aspect ratio deduction: {e}")

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

    def _is_agent_available(self) -> bool:
        """
        Check if the agent (Ollama) is available for natural language queries.

        Returns:
            bool: True if agent is available, False otherwise
        """
        if self._agent is None:
            return False
        try:
            return self._agent.is_ollama_available()
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

                # Ensure sample_id is a column if it was the index
                if df.index.name == SampleStatsEx.SAMPLE_ID.value:
                    df = df.reset_index()

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
                        origins = [val] if val.strip() else []  # Filter empty strings
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

        # Ensure natural_sort_score exists (init with NaN if missing)
        if self._compute_natural_sort and "natural_sort_score" not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df["natural_sort_score"] = np.nan
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
        Compute natural sort statistics (brightness, hue, saturation, entropy) for all samples
        and update the dataframe.

        Includes a 'natural_sort_score' for sorting, configured by weights.
        """
        # --- CONFIGURATION: Define Natural Sort Cues & Weights ---
        # Weights should sum to 1.0 ideally, but relative magnitude matters most.
        # Strategies:
        # 1. "Day vs Night" focus: Brightness=0.8, Entropy=0.2
        # 2. "Complexity" focus: Brightness=0.2, Entropy=0.8
        # 3. "Balanced": Brightness=0.5, Entropy=0.5
        # 4. "Grouped" (Pseudo-primary key): Brightness=5.0, Entropy=1.0 (Forces clustering by light)

        SORT_WEIGHTS = {
            "brightness": 0.7,  # Primary cue: Lighting conditions
            "entropy": 0.3,     # Secondary cue: Texture/Scene complexity
            "hue": 0.0          # Optional: Color tint
        }

        logger.info(f"[DataService] Starting natural sort stats computation with weights: {SORT_WEIGHTS}")

        try:
            import cv2
        except ImportError:
            logger.warning("[DataService] OpenCV not found. Skipping natural sort computation.")
            return "OpenCV not installed"

        if self._all_datasets_df is None or self._all_datasets_df.empty:
             return "No data to process"

        # Helper: Calculate Shannon Entropy (Complexity)
        def calc_entropy(img_gray):
            try:
                # Calculate histogram (256 bins for 8-bit)
                hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
                # Normalize histogram to get probabilities
                p = hist.ravel() / hist.sum()
                # Filter out zero probabilities to avoid log(0)
                p = p[p > 0]
                # Shannon Entropy in bits
                return -np.sum(p * np.log2(p))
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
                if "natural_sort_score" in row and not pd.isna(row["natural_sort_score"]):
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

                # Brightness (mean pixel intensity)
                # If RGB, convert to Gray, else just mean
                if img_np.ndim == 3:
                    # OpenCV expects BGR usually, but PIL gives RGB.
                    # cvtColor RGB2GRAY is correct.
                    try:
                        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    except Exception:
                         gray = img_np
                else:
                    gray = img_np

                brightness = np.mean(gray)
                entropy = calc_entropy(gray)

                # HSV Stats
                if img_np.ndim == 3:
                    try:
                        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                        hue = np.mean(hsv[:, :, 0])
                        saturation = np.mean(hsv[:, :, 1])
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

                # Hue: 0-179 in OpenCV
                norm_hue = min(max(hue / 179.0, 0.0), 1.0)

                score = (
                    SORT_WEIGHTS.get("brightness", 0) * norm_brightness +
                    SORT_WEIGHTS.get("entropy", 0) * norm_entropy +
                    SORT_WEIGHTS.get("hue", 0) * norm_hue
                )

                return {
                    "sample_id": sample_id,
                    "origin": origin,
                    "brightness": brightness,
                    "entropy": entropy,
                    "hue": hue,
                    "saturation": saturation,
                    "natural_sort_score": score
                }
            except Exception as e:
                # Log only the first few errors globally (using a simple counter if we could, but here we can't share state easily)
                # Fallback: Just log warning. To avoid spam, we can check if it's one of the first few tasks?
                # No, standard logger is fine, just use debug for spammy ones or INFO for specific tracking.
                logger.warning(f"Failed to compute stats for sample {idx} (id={row.get(SampleStatsEx.SAMPLE_ID.value, 'unknown')}): {e}")
                return None

        # Prepare tasks
        tasks = []
        for idx, row in self._all_datasets_df.reset_index().iterrows():
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
            # logger.debug(f"Processing sample_id={sample_id} from origin={origin} with request: {request}")

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
                task_type = "classification"  # Default fallback
                if label is not None:
                    if isinstance(label, dict):
                        if 'boxes' in label or 'bboxes' in label:
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
                                    # Segmentation: check spatial dims
                                    elif len(shape) >= 2 and shape[-2] >= 16 and shape[-1] >= 16:
                                        task_type = 'segmentation'
                            except Exception:
                                pass

            # ====== Step 5a: Process stats ======
            stats_to_retrieve = list(request.stats_to_retrieve)

            # These columns are handled explicitly later in the pipeline
            exclude_cols = {
                SampleStatsEx.SAMPLE_ID.value,
                SampleStatsEx.ORIGIN.value,
                SampleStatsEx.TARGET.value if not skip_label_for_request else None,
                SampleStatsEx.PREDICTION.value,
                SampleStatsEx.TASK_TYPE.value,
            }

            if not stats_to_retrieve:
                stats_to_retrieve = [col for col in df_columns if col not in exclude_cols]

            # Optimized bulk processing of stats
            for stat_name in stats_to_retrieve:
                # Never re-process core fields generically (prevents duplicates/bad db state overwriting calculated state)
                if stat_name in exclude_cols:
                    continue

                value = row.get(stat_name)

                # Skip prediction raw array
                if (isinstance(value, np.ndarray) and value.ndim > 1) or (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0):
                    continue
                elif isinstance(value, float):
                    value = round(value, 7)
                elif isinstance(value, bool):
                    value = int(value)

                # Check if it s a tag column here and handle it as a string stat with the tag name as value
                if stat_name.startswith(f"{SampleStatsEx.TAG.value}"):
                    if value == 1:
                        tag_name = stat_name[len(f"{SampleStatsEx.TAG.value}:"):]  # Remove "tags_" prefix to get tag name
                        if value:  # Only include if the tag is True for this sample
                            data_stats.append(
                                create_data_stat(f"{SampleStatsEx.TAG.value}:{tag_name}", "string", shape=[1], value_string="1", thumbnail=b"")
                            )
                    else:
                        continue  # Skip false tags
                else:
                    data_stats.append(
                        create_data_stat(stat_name, "string", shape=[1], value_string=str(value)[:512], thumbnail=b"")
                    )

            # ====== Step 6: Add origin and task_type stats ======
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

            target_mask_stat_index = None
            pred_mask_stat_index = None
            target_mask_u8 = None
            pred_mask_u8 = None

            # ====== Step 7: Process labels ======
            t_label_convert = 0.0
            if not skip_label_for_request and task_type != "classification":
                t0_gt_conv = time.time()
                label_raw = row.get(SampleStatsEx.TARGET.value) if label is None else label
                label_arr = to_numpy_safe(label_raw)
                if label_arr is None:
                    try:
                        label_arr = np.asarray(label_raw)
                    except Exception:
                        label_arr = np.array([])

                # Ensure mask is uint8 (class IDs are 0-255) and RLE-encode for efficient transfer.
                # RLE-encoded masks are ~100-500x smaller than float arrays for typical segmentation masks.
                label_u8 = label_arr.astype(np.uint8) if label_arr.size > 0 else label_arr
                t_label_convert = time.time() - t0_gt_conv

                t0_rle = time.time()
                rle_bytes = rle_encode_mask(label_u8.ravel()) if label_u8.size > 0 else b""
                t_mask_encode += time.time() - t0_rle

                data_stats.append(
                    create_data_stat(
                        name='target',
                        stat_type='rle_mask',
                        shape=list(label_arr.shape),
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
                    or getattr(self._ctx.components.get("model"), "num_classes", None)
                )
                if num_classes is None:
                    # Fallback: infer from this label
                    if label_arr.size > 0:
                        max_id = int(label_arr.max())
                        num_classes = max(1, max_id + 1)
                    else:
                        num_classes = 1

                data_stats.append(
                    create_data_stat(
                        name="num_classes",
                        stat_type="scalar",
                        shape=[1],
                        value=[float(num_classes)],
                        thumbnail=b""
                    )
                )

                # Append class_names if available
                model_comp = self._ctx.components.get("model")
                class_names = (
                    row.get('class_names')
                    or (getattr(dataset, "class_names", None) if dataset else None)
                    or getattr(model_comp, "class_names", None)
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
                    data_stats.append(
                        create_data_stat(
                            name='target',
                            stat_type='string',
                            shape=[1],
                            value_string=str(label), # Simplified visualization
                            thumbnail=b""
                        )
                    )
                else:
                    # Check if label is NaN (handle both scalars and arrays)
                    if self._is_nan_value(label):
                        pass  # Skip NaN labels

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
                try:
                    t0_pmask = time.time()
                    pred_arr = np.asarray(pred, dtype=np.uint8)
                    rle_bytes = rle_encode_mask(pred_arr.ravel()) if pred_arr.size > 0 else b""
                    t_mask_encode += time.time() - t0_pmask
                    data_stats.append(
                        create_data_stat(
                            name='pred_mask',
                            stat_type='rle_mask',
                            shape=list(pred_arr.shape),
                            value=[],
                            thumbnail=rle_bytes
                        )
                    )
                    pred_mask_stat_index = len(data_stats) - 1
                    pred_mask_u8 = pred_arr
                except Exception:
                    pass
            else:
                # Classification: get prediction from row or dataset
                if pred is None:
                    pass  # No prediction to process

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
                    aspect_ratio = getattr(ds, "aspect_ratio", None)
                    if aspect_ratio is not None:
                        # Normalize target dimensions to honor explicit ratio before scaling
                        target_width = int(target_height * aspect_ratio)
                    else:
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
                        target_height = int(os.environ.get("WL_DEFAULT_THUMBNAIL_SIZE", 720))  # Default full resolution image is 720p on the longest side, but can be overridden by env var
                        target_width = int(target_height * aspect_ratio)

                    if is_full_resolution:
                        max_modal_height = int(
                            os.environ.get(
                                "WL_MODAL_MAX_RESOLUTION",
                                os.environ.get("WL_DEFAULT_THUMBNAIL_SIZE", 720),
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

            record = pb2.DataRecord(sample_id=str(sample_id), data_stats=data_stats)

            return record

        except Exception as e:
            total_time = time.time() - start_total
            logger.error(f"[Sample {row.get(SampleStatsEx.SAMPLE_ID.value, -1)}] X Error after {total_time:.3f}s: {e}", exc_info=True)
            return None

    def _get_unique_tags(self) -> List[str]:
        """Collect all unique tags currently present in the tracked datasets.

        Tags are stored as individual boolean columns with prefix "tags_".
        This method extracts all tag names from the column names.
        """
        tags = set()
        try:
            # Extract tags from individual tag columns (tags_<tagname>)
            if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                for col in self._all_datasets_df.columns:
                    if col.startswith(f"{SampleStatsEx.TAG.value}:"):
                        # Extract tag name by removing "tags_" prefix
                        tag_name = col[len(f"{SampleStatsEx.TAG.value}:"):]  # len("tags_") == 5
                        tags.add(tag_name)
        except Exception as e:
            logger.warning(f"Error collecting unique tags: {e}")
        return sorted(list(tags))

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
            len(df[df.get(SampleStatsEx.DISCARDED.value, False) == True])  # noqa: E712
            if df is not None and SampleStatsEx.DISCARDED.value in df.columns
            else 0
        )
        in_loop_count = total_count - discarded_count
        unique_tags = self._get_unique_tags()

        return pb2.DataQueryResponse(
            success=True,
            message=message,
            number_of_all_samples=total_count,
            number_of_samples_in_the_loop=in_loop_count,
            number_of_discarded_samples=discarded_count,
            unique_tags=unique_tags,
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

            return f"Action triggered: {action_name} (Not implemented)"

        # --- 3. DATAFRAME MANIPULATION ---

        # A) Agent-driven df.apply_mask (for complex filters)
        if func == "df.apply_mask":
            code = params.get("code", "")
            try:
                mask = eval(code, {"df": df, "np": np, "pd": pd})
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
            code = params.get("code")
            try:
                # 0. Safety Check: If target column exists, check compatibility
                if col in df.columns:
                     # Heuristic: If existing column is not numeric, but code implies math (contains +,-,*,/), warn/block
                     # This prevents accidental string concatenation (e.g. 1.0 + "tag" -> "tag1.0")
                     if not pd.api.types.is_numeric_dtype(df[col]):
                         # Reliance on try-except to catch invalid math is safer than heuristic string checking
                         # because heuristics fail on column names like 'signals//loss'
                         pass

                # 1. Evaluate the expression with safe context
                new_values = eval(code, {"df": df, "np": np, "pd": pd})

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
                    update_payload = df[[col]].copy()

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
                         sub_df = df.iloc[start:end].copy()

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
                                 sub_df.sort_values(inplace=True, **sort_params)
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
                                              temp_df = sub_df.reset_index()

                                              # Adjust 'by' if needed (e.g. if 'index' was used, it's now 'sample_id' etc)
                                              # but here columns usually match.
                                              temp_df.sort_values(inplace=True, **sort_params)

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
                                 new_index_values = df.index.to_numpy().copy()
                                 new_index_values[start:end] = sub_df.index.to_numpy()
                                 df.index = pd.MultiIndex.from_tuples(new_index_values, names=df.index.names)
                             else:
                                 idx_name = df.index.name
                                 new_index = df.index.to_numpy().copy()
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
                    # Params are already cleaned by the Agent's SortHandler
                    try:
                        df.sort_values(inplace=True, **params)
                    except (TypeError, ValueError, KeyError) as e:
                        # Fallback for ambiguity or missing column (if it's in the index)
                        if "ambiguous" in str(e).lower() or isinstance(e, KeyError):
                             # If ambiguous or missing from columns, and it's a simple sort by one field, try sorting index level
                             by = params.get("by")
                             if isinstance(by, str) and by in getattr(df.index, 'names', []):
                                 ascending = params.get("ascending", True)
                                 df.sort_index(level=by, inplace=True, ascending=ascending)
                             elif isinstance(e, KeyError):
                                 # It's actually missing, raise the original error
                                 raise e
                             else:
                                 # Multi-column sort or other ambiguity, fallback to converting columns
                                 df.sort_values(inplace=True, **params, key=lambda x: x)

                        # Fallback for mixed types (e.g. lists vs strings/floats): sort by string representation
                        elif "key" not in params and isinstance(e, TypeError):
                            logger.warning(f"Sort failed due to type mismatch ({e}). Retrying with string conversion...")
                            params["key"] = lambda x: x.astype(str)
                            df.sort_values(inplace=True, **params)
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

        # D) Analysis (Read-Only)
        if func == "df.analyze":
            code = params.get("code")
            if not code: return "No code provided"
            if "import " in code or "__" in code: return "Safety Violation"
            try:
                result = eval(code, {"df": df, "pd": pd, "np": np})
                return f"Analysis Result: {result}"
            except Exception as e:
                return f"Analysis Error: {e}"

        return "No operation applied"

    def _slowUpdateInternals(self, force: bool = False):
        """
        This method is responsible for updating the internal dataframe view with the latest data from the manager.
        It uses a dedicated update lock to ensure only one thread performs the expensive update at a time,
        while allowing other threads to read the existing dataframe without blocking.
        """
        current_time = time.time()
        # Fast throttling check
        if not force and self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 10:
            return

        with self._update_lock:
            # Re-check throttling inside lock to avoid redundant updates
            if not force and self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 10:
                return

            updated_df = self._pull_into_all_data_view_df()

            # Guard against init race conditions
            if updated_df is None:
                return

            # Capture a consistent snapshot of the current state
            with self._lock:
                is_filtered = self._is_filtered
                current_all_df = self._all_datasets_df

            # Ensure default columns exist
            if self._compute_natural_sort and "natural_sort_score" not in updated_df.columns:
                 updated_df["natural_sort_score"] = np.nan
            if SampleStatsEx.DISCARDED.value not in updated_df.columns:
                 updated_df[SampleStatsEx.DISCARDED.value] = False

            if is_filtered and current_all_df is not None:
                # The user has applied a custom view (Filter, Sort, or Aggregation).
                try:
                    # Use intersection to detect rows we are currently watching in the filtered view
                    common_indices = current_all_df.index.intersection(updated_df.index)

                    if len(common_indices) > 0:
                         # Force the new data to conform to the USER'S current view (rows/order).
                         target_order = current_all_df.index
                         updated_df = updated_df.reindex(target_order)
                    else:
                         # Aggregation/Transformation - skip auto-update.
                         return
                except Exception as e:
                    logger.debug(f"[_slowUpdateInternals] Error matching indices for filtered view: {e}")
                    return

            elif current_all_df is not None and not current_all_df.empty:
                # Standard/Unfiltered View.
                # Preserves sticky sort and appends new samples.
                if not current_all_df.index.is_monotonic_increasing:
                     try:
                         key_cols = [SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value]

                         old_df_keys = current_all_df.reset_index()
                         new_df_keys = updated_df.reset_index()

                         if not all(col in old_df_keys.columns for col in key_cols) or not all(col in new_df_keys.columns for col in key_cols):
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

    def _is_metadata_only_request(self, request) -> bool:
        """True when caller requests metadata columns only, without image payloads."""
        try:
            stats_to_retrieve = list(getattr(request, 'stats_to_retrieve', []) or [])
            include_raw = bool(getattr(request, 'include_raw_data', False))
            include_transformed = bool(getattr(request, 'include_transformed_data', False))
            return (not include_raw) and (not include_transformed) and bool(stats_to_retrieve)
        except Exception:
            return False

    def _build_metadata_only_response(self, df_slice: pd.DataFrame, request):
        """Build DataSamplesResponse from dataframe columns only (no dataset/image traversal).

        This is a single-job vectorized path: the entire df_slice is processed
        at once using pandas operations rather than dispatching per-sample_id
        work to the thread pool.
        """
        if df_slice is None or df_slice.empty:
            return pb2.DataSamplesResponse(
                success=False,
                message="No metadata rows available",
                data_records=[],
            )

        requested_cols = list(getattr(request, 'stats_to_retrieve', []) or [])
        excluded_cols = {
            SampleStatsEx.SAMPLE_ID.value,
            SampleStatsEx.ORIGIN.value,
            # SampleStatsEx.TARGET.value,
            # SampleStatsEx.PREDICTION.value,
            SampleStatsEx.TASK_TYPE.value,
        }

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
        # handling.  All heavy conversion is done once on the full column
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
        # eliminate the create_data_stat wrapper overhead.  Then scatter
        # them into the per-row bins.  At 1M rows × 10 cols this avoids
        # a 10M-iteration nested Python loop.
        _DataStat = pb2.DataStat  # local ref – avoids repeated attr lookup

        for col in meta_cols:
            series = df_slice[col]
            if series.dtype.kind == 'f':
                str_vals = series.round(7).astype(str).str[:512].tolist()
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

        for col in tag_cols:
            bools = df_slice[col].astype(bool).tolist()
            # Pre-build a single shared DataStat for this tag column –
            # protobuf messages are mutable so we need one per row, but
            # the tag "1" stat is tiny so construction is cheap.
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

    def _process_get_data_samples(self, request, context):
        """
        Actual implementation of GetDataSamples.

        Two optimisations:
        1. **Preview-cache fast path** – If the preview cache has a 64×64 or less
           thumbnail for the requested sample *and* the request is for a tiny
           resolution (both dims ≤ ``_PREVIEW_CACHE_THRESHOLD``), serve from
           the cache instantly without touching the file system.
        2. **Parallel batch processing** – All samples are submitted to the
           thread pool at once so all 8 workers stay busy.  The chunk-size
           env-var ``WL_BATCH_CHUNK_SIZE`` is kept for backward compat but
           the default is now the full request size (all at once).
        """
        _PREVIEW_CACHE_THRESHOLD = 80      # max px to consider a "preview" request
        # Default: process ALL rows at once in the thread pool (workers = 8).
        # Override with WL_BATCH_CHUNK_SIZE to throttle concurrency.
        _BATCH_CHUNK_SIZE = int(os.environ.get("WL_BATCH_CHUNK_SIZE", "0"))  # 0 = all at once

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
                df_slice = current_df.iloc[request.start_index:request.start_index + request.records_cnt].reset_index()
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

            logger.debug(
                "Retrieving samples from %s to %s", request.start_index, request.start_index + request.records_cnt)

            if self._is_metadata_only_request(request):
                return self._build_metadata_only_response(df_slice, request)

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
            # stay busy.  This avoids the old sequential-chunk bottleneck
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
        new_tag_name = f'{SampleStatsEx.TAG.value}:{new_tag_name.strip()}'

        # Get current tags from the in-memory dataframe or df_manager
        existing_tag_value = True  # Default to True for new tags
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
                        if col == new_tag_name and row[col]:  # If existing, revert the value
                            existing_tag_value = bool(1 - row[col])

        except (KeyError, AttributeError) as e:
            logger.debug(f"Could not read current tags: {e}")

        # Calculate target tags based on edit type
        if edit_type == SampleEditType.EDIT_REMOVE:
            existing_tag_value = False  # For removal, we set the tag to False
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

    # RPC Implementations
    # ===================
    def ApplyDataQuery(self, request, context):
        """
        Apply a query on the in-memory dataframe.

        Modes:
          - request.query == ""  -> just return counts, do not modify df
          - request.query != ""  -> always handled by the agent (natural language path)

        Counts returned:
          - number_of_all_samples: all rows currently in the dataframe
          - number_of_samples_in_the_loop: rows not discarded
          - number_of_discarded_samples: rows with discarded == True
        """

        self._ctx.ensure_components()
        components = self._ctx.components

        # 1) No query: just report counts (Needs lock for consistency)
        if request.query == "":
            with self._lock:
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
                with self._lock:
                    # If this is JUST a view-sort for a specific page, DO NOT force an internal refresh
                    # as that wipes out the existing slice/sort state before we try to modify the next slice.
                    is_only_view_sort = len(operations) == 1 and operations[0].get("function") == "df.sort_view_slice"
                    if not is_only_view_sort:
                        self._slowUpdateInternals(force=True)  # Refresh internals before applying Agent operations

                    # Work on a copy to allow concurrent readers to see a consistent state
                    df = self._all_datasets_df.copy()
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
                        # Work on a copy to allow concurrent readers to see a consistent state
                        working_df = self._all_datasets_df.copy()
                        df, message = execute_df_operation(working_df, request.query)  # in-place operation, or replace previous dataframe
                        logger.info(f"[ApplyDataQuery] Executed direct DataFrame operation. Message: {message}")
                        if operations:
                            self._is_filtered = True

                        # Atomic swap
                        self._all_datasets_df = df
                    return self._build_success_response(
                        df=df,
                        message=message,
                        intent_type=pb2.INTENT_FILTER
                    )
                elif request.query.lower().replace("'''", "\"\"\"").replace('\"\"\"', "").replace('\'\'\'', "").replace(" ", "").startswith("@reset") or request.query.lower().replace('\"\"\"', "").replace('\'\'\'', "").replace(" ", "").startswith("@clear"):
                    logger.info(f"[ApplyDataQuery] BYPASSING AGENT - Direct reset/clear operation: {request.query[:100]}...")
                    # Force view reset
                    with self._lock:
                        self._is_filtered = False  # Unfreeze view first
                        self._slowUpdateInternals(force=True)  # Force update to ensure we have the latest data
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
                            self._all_datasets_df = self._pull_into_all_data_view_df() or pd.DataFrame()

                        df = self._all_datasets_df.copy()
                        messages = []
                        intent_type = pb2.INTENT_FILTER
                        analysis_result = ""

                        for op in operations:
                            func = op.get("function")
                            params = op.get("params", {}) or {}

                            if params.get("__agent_reset__"):
                                logger.debug("[ApplyDataQuery] Agent requested reset")
                                df = self._pull_into_all_data_view_df() or pd.DataFrame()
                                self._is_filtered = False
                                messages.append("Reset view")
                                continue

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
                snapshot_path = checkpoint_manager.save_data_snapshot()
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

            with self._lock:
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

                    self._all_datasets_df = self._all_datasets_df.reset_index()
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

            with self._lock:
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

                    self._slowUpdateInternals(force=True)

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

        if not request.stat_name or not request.stat_name.startswith(SampleStatsEx.TAG.value) and request.stat_name not in [SampleStatsEx.DISCARDED.value]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags', 'discarded', '__copy_metadata__', '__delete_metadata__', '__save_data_state__', and '__force_h5_write_and_save__' edits are supported.",
            )

        # =====================================================================
        # Process tag edits using the new column-based tag system
        # =====================================================================
        with self._lock:
            try:
                if request.samples_ids and self._df_manager is not None:
                    updates_by_origin = {}
                    is_tag_request = request.stat_name == SampleStatsEx.TAG.value or request.stat_name.startswith(SampleStatsEx.TAG.value)
                    for sid, origin in zip(request.samples_ids, request.sample_origins):
                        sid_value = str(sid)
                        # =========
                        # TAG EDITS
                        # =========
                        if is_tag_request:
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
                    if is_tag_request and request.type == SampleEditType.EDIT_REMOVE and request.float_value == -1:
                        column_name = request.stat_name.strip()
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

    def CheckAgentHealth(self, request, context):
        """
        gRPC method to check if the agent is available for natural language queries.
        Returns:
            AgentHealthResponse { available: bool, message: str }
        """

        try:
            available = self._is_agent_available()
            msg = "Agent is available" if available else "Agent is not available"
            return pb2.AgentHealthResponse(available=available, message=msg)
        except Exception as e:
            return pb2.AgentHealthResponse(available=False, message=f"Error: {e}")
