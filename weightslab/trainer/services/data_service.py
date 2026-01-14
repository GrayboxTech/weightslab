import io
from typing import List
import time
import json
import torch
import logging
import os
import traceback
import threading

import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2

from PIL import Image
from typing import List
from pathlib import Path
from concurrent import futures

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.service_utils import load_raw_image, load_label, get_mask
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.backend.ledgers import get_dataloaders, get_dataframe


# Get global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _get_stat_from_row(row, stat_name):
    """Extract stat from dataframe row and convert to DataStat message."""
    value = row.get(stat_name)
    if not isinstance(value, (list, np.ndarray, torch.Tensor)) and (value is None or pd.isna(value)):
        return None

    # Helper for creating DataStat messages
    def make_stat(type_, shape, **kwargs):
        name = kwargs.pop("name", stat_name)
        return pb2.DataStat(name=name, type=type_, shape=shape, **kwargs)

    # 0) Specific case: string from signals
    if stat_name == "prediction_signals_values" and (isinstance(value, dict) or isinstance(value, str)):
        if isinstance(value, dict):
            value_string = str(value)
        else:
            value_string = value

        # Replace to json like standard format, i.e., '""'
        value = json.loads(value_string.replace('\'', '@').replace('\"', '\'').replace('@', '\"'))
        data_list = []
        for k, v in value.items():
            data_list.append(
                make_stat("scalar", [], value=[float(v)], name=str(k))
            )
        return data_list

    # 1) Fast-path: None
    if value is None:
        return None

    # Detect array-like first
    is_array_like = isinstance(value, (np.ndarray, list, tuple)) or isinstance(value, torch.Tensor)

    # 2) NaN handling ONLY for non-array-like values
    if not is_array_like:
        try:
            if pd.isna(value):
                return None
        except TypeError:
            # Some objects don't like pd.isna, just ignore
            pass

    # 3) torch.Tensor -> numpy
    if isinstance(value, torch.Tensor):
        try:
            value = value.detach().cpu().numpy()
        except Exception:
            return None

    # 4) numpy arrays -> array stat
    if isinstance(value, np.ndarray):
        # 0-dim array -> scalar
        if value.ndim == 0:
            v = float(value.item())
            return make_stat("scalar", [], value=[v])

        a = value
        # NOTE: if you ever need to cap size, you could do it here:
        # if a.size > MAX_ALLOWED:
        #     a = a.reshape(-1)[:MAX_ALLOWED]
        return make_stat(
            "array",
            list(a.shape),
            value=a.ravel().astype(float).tolist(),
        )

    # 5) list/tuple -> treat as 1D array
    if isinstance(value, (list, tuple)):
        a = np.asarray(value)
        if a.ndim == 0:
            v = float(a.item())
            return make_stat("scalar", [], value=[v])
        return make_stat(
            "array",
            list(a.shape),
            value=a.ravel().astype(float).tolist(),
        )

    # 6) scalars
    if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
        return make_stat("scalar", [], value=[float(value)])

    # 7) strings
    if isinstance(value, str):
        return make_stat("string", [1], value_string=value)

    # 8) Fallback: stringify
    return make_stat("string", [1], value_string=str(value)[:512])

def _infer_task_type_from_label(label, default="classification"):
    """
    Heuristic: guess task type based on label shape / dtype.

    - 0D or size==1 → classification-like
    - 2D integer mask → segmentation-like
    - 3D 1-channel integer tensor → segmentation-like
    - otherwise → fall back to default
    """
    try:
        arr = label.cpu().numpy() if hasattr(label, "cpu") else np.asarray(label)
    except Exception:
        return default

    # Scalar / single element → treat as classification
    if arr.ndim == 0 or arr.size == 1:
        return "classification"

    # 2D integer-ish → very likely segmentation mask or detection
    if arr.shape[0] < 28 or arr.shape[1] < 28:
        return 'segmentation'  # 'detection' interpreted as segmentation
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        return "segmentation"

    # 3D with a single channel can also be a mask (1, H, W) or (H, W, 1)
    if arr.ndim == 3:
        if arr.shape[0] == 1 or arr.shape[-1] == 1:
            if np.issubdtype(arr.dtype, np.integer):
                return "segmentation"

    # Anything else: keep the caller's default guess
    return default


def _infer_task_type_from_label(label, default="classification"):
    """
    Heuristic: guess task type based on label shape / dtype.

    - 0D or size==1 → classification-like
    - 2D integer mask → segmentation-like
    - 3D 1-channel integer tensor → segmentation-like
    - otherwise → fall back to default
    """
    try:
        arr = label.cpu().numpy() if hasattr(label, "cpu") else np.asarray(label)
    except Exception:
        return default

    # Scalar / single element → treat as classification
    if arr.ndim == 0 or arr.size == 1:
        return "classification"

    # 2D integer-ish → very likely segmentation mask
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        return "segmentation"

    # 3D with a single channel can also be a mask (1, H, W) or (H, W, 1)
    if arr.ndim == 3:
        if arr.shape[0] == 1 or arr.shape[-1] == 1:
            if np.issubdtype(arr.dtype, np.integer):
                return "segmentation"

    # Anything else: keep the caller's default guess
    return default


def generate_thumbnail(pil_image, max_size=(128, 128), quality=85):
    """Generate a JPEG thumbnail from a PIL image.

    Args:
        pil_image: PIL Image object
        max_size: Max dimensions (width, height) for thumbnail
        quality: JPEG quality (1-95)

    Returns:
        bytes: JPEG thumbnail as bytes
    """
    try:
        # Create a copy to avoid modifying original
        thumb = pil_image.copy()

        # Use LANCZOS for high-quality downsampling
        thumb.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed (JPEG doesn't support RGBA)
        if thumb.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', thumb.size, (255, 255, 255))
            if thumb.mode == 'P':
                thumb = thumb.convert('RGBA')
            if thumb.mode in ('RGBA', 'LA'):
                background.paste(thumb, mask=thumb.split()[-1])  # Use alpha channel as mask
                thumb = background
        elif thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')

        # Save as JPEG to buffer
        buffer = io.BytesIO()
        thumb.save(buffer, format='JPEG', quality=quality, optimize=True)
        return buffer.getvalue()
    except Exception as e:
        logger.warning(f"Failed to generate thumbnail: {e}")
        return b""


class DataService:

    """
    Data service helpers + RPCs (for weights_studio UI).

    Images are sent over gRPC as bytes (JPEG) for simplicity and correctness.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.Lock()
        self._df_manager = get_dataframe()

        # init references to the context components
        self._ctx.ensure_components()

        self._root_log_dir = self._resolve_root_log_dir()
        self._h5_path = self._resolve_h5_path()
        self._stats_store = H5DataFrameStore(self._h5_path) if self._h5_path else None

        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = None

        # Shared thread pool for data processing (avoid thread explosion)
        # Size: min(CPU cores * 2, 16) to balance concurrency without excessive threading
        cpu_count = os.cpu_count() or 4
        max_data_workers = min(cpu_count * 2, 16)
        self._data_executor = futures.ThreadPoolExecutor(
            max_workers=max_data_workers,
            thread_name_prefix="WL-DataProcessing"
        )

        logger.info("DataService initialized.", extra={
            "data_workers": max_data_workers,
            "cpu_count": cpu_count
        })

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
                hp_dict = hp.get() if not isinstance(hp, dict) else hp
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

    def get_root_log_dir(self) -> str:
        """Get the root log directory as a string.

        Returns:
            Absolute path to root_log_dir
        """
        return str(self._root_log_dir.absolute())

    def is_agent_available(self) -> bool:
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
                hp_dict = hp.get() if not isinstance(hp, dict) else hp
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
                df = self._df_manager.get_combined_df() if self._df_manager is not None else pd.DataFrame()
                if df.empty:
                    return df

                # Ensure MultiIndex for stable slicing and agent operations
                if "origin" in df.columns and not isinstance(df.index, pd.MultiIndex):
                    try:
                        df = df.reset_index().set_index(["origin", "sample_id"])
                    except Exception as e:
                        logger.warning(f"Failed to set index on streamed dataframe: {e}")

                return df
            except Exception:
                logger.debug("[DataService] Falling back to cached snapshot of dataframe")
                return self._all_datasets_df if self._all_datasets_df is not None else pd.DataFrame()

    def _get_origin_filter(self, request):
        """Extract requested origins if present on request (backward compatible)."""
        origins = None
        for attr in ("sample_origins", "origins", "origin"):
            try:
                val = getattr(request, attr, None)
                if val:
                    # Normalize to list
                    if isinstance(val, str):
                        origins = [val]
                    else:
                        origins = list(val)
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
        """Ensure tags column is present on the streamed dataframe."""
        if self._all_datasets_df is None or self._all_datasets_df.empty:
            return

        if SampleStatsEx.TAGS.value not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df[SampleStatsEx.TAGS.value] = ""
            except Exception:
                pass

    def _get_dataset(self, origin: str):
        loader = self._get_loader_by_origin(origin)
        if loader is not None:
            dataset = getattr(loader, "tracked_dataset", None)
            return dataset

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args

        try:
            origin = row.get(SampleStatsEx.ORIGIN.value, 'unknown')
            sample_id = int(row.get(SampleStatsEx.SAMPLE_ID.value, 0))

            num_classes = None
            data_stats = []
            raw_data_bytes, transformed_data_bytes = b"", b""
            raw_shape, transformed_shape = [], []

            if request.include_raw_data:
                try:
                    # Get Dataset
                    dataset = self._get_dataset(origin)

                    # Attempt to load raw image from dataset if available
                    if dataset:
                        index = dataset.get_index_from_sample_id(sample_id)
                        raw_img = load_raw_image(dataset, index)
                        raw_img_array = np.array(raw_img)
                        original_size = raw_img.size

                        # Handle resize request
                        # Negative values indicate percentage mode (e.g., -50 means 50% of original)
                        # Positive values indicate absolute pixel dimensions
                        # Zero means no resize
                        if request.resize_width < 0 and request.resize_height < 0:
                            percent = abs(request.resize_width) / 100.0
                            target_width = int(original_size[0] * percent)
                            target_height = int(original_size[1] * percent)

                            # Only resize if we're actually reducing size
                            if target_width < original_size[0] or target_height < original_size[1]:
                                raw_img = Image.fromarray(raw_img_array).resize((target_width, target_height))

                        elif request.resize_width > 0 and request.resize_height > 0:
                            if request.resize_width < original_size[0] or request.resize_height < original_size[1]:
                                raw_img = Image.fromarray(raw_img_array).resize((request.resize_width, request.resize_height))

                        else:
                            raw_img = Image.fromarray(raw_img_array)

                        raw_shape = [raw_img.height, raw_img.width, len(raw_img.getbands())]
                        raw_buf = io.BytesIO()
                        raw_img.save(raw_buf, format='JPEG')
                        raw_data_bytes = raw_buf.getvalue()

                except Exception as e:
                    logger.debug(f"Could not load raw image for sample {sample_id}: {e}")
                    raw_data_bytes = transformed_data_bytes
                    raw_shape = transformed_shape

            # Determine task type early so we can conditionally send stats
            # And get label from the dataset
            label = row.get(SampleStatsEx.TARGET.value)
            dataset = self._get_dataset(origin)
            dataset_index = dataset.get_index_from_sample_id(sample_id) if dataset else None
            if label is None:
                # Try loading label from dataset if available
                if dataset:
                    label = load_label(dataset, sample_id)  # Load cls, seg, or det labels
            task_type = _infer_task_type_from_label(label, default='Segmentation')

            stats_to_retrieve = list(request.stats_to_retrieve)
            if not stats_to_retrieve:
                stats_to_retrieve = [col for col in df_columns if col not in [SampleStatsEx.SAMPLE_ID.value, SampleStatsEx.ORIGIN.value]]
                if task_type != "classification":
                    # Also pre-emptively include per-class loss columns if we can find num_classes
                    num_classes = (
                        row.get('num_classes')
                        or (getattr(dataset, "num_classes", None) if dataset else None)
                        or getattr(self._ctx.components.get("model"), "num_classes", None)
                    )

            for stat_name in stats_to_retrieve:
                # Get value
                stat = _get_stat_from_row(row, stat_name)
                if stat is None:
                    continue

                # Sanity check: wrap single stat into list
                if not isinstance(stat, list):
                    stat = [stat]

                # Save every stats
                if stat is not None:
                    data_stats.extend(stat)

            # ====================================
            # Expose origin and task_type as stats
            data_stats.append(
                pb2.DataStat(
                    name="origin", type='string', shape=[1], value_string=origin
                )
            )
            data_stats.append(
                pb2.DataStat(
                    name="task_type", type='string', shape=[1], value_string=str(task_type)
                )
            )

            # ====================================================================
            # ========================== Labels ==================================
            # Encode label safely depending on task_type  (GT mask / label)
            if task_type != "classification":
                label_arr = row.get(SampleStatsEx.TARGET.value) or label
                if label_arr is not None and not isinstance(label_arr, np.ndarray):
                    try:
                        label_arr = np.array(label_arr.detach().cpu() if hasattr(label_arr, 'detach') else label_arr)
                    except Exception:
                        label_arr = np.array(label_arr)

                # If pred_arr appears to be bounding boxes, convert to mask
                label_arr = get_mask(label_arr, dataset, dataset_index=dataset_index)

                # Treat label as segmentation mask → array stat
                data_stats.append(
                    pb2.DataStat(
                        name='label',
                        type='array',
                        shape=list(label_arr.shape),
                        value=label_arr.astype(float).ravel().tolist(),
                    )
                )

                try:
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
                        pb2.DataStat(
                            name="num_classes",
                            type="scalar",
                            shape=[1],
                            value=[float(num_classes)],
                        )
                    )

                except Exception as e:
                    logger.warning(f"Could not infer num_classes for sample {sample_id}: {e}")

            else:
                # Classification / other scalar-like labels
                label_arr = np.array(label.cpu() if hasattr(label, 'cpu') else label)
                if label_arr.size == 1:
                    label_val = float(label_arr.reshape(-1)[0])
                    data_stats.append(
                        pb2.DataStat(
                            name='label',
                            type='scalar',
                            shape=[1],
                            value=[label_val],
                        )
                    )
                else:
                    # Fallback for non-scalar labels in non-segmentation tasks
                    data_stats.append(
                        pb2.DataStat(
                            name='label',
                            type='array',
                            shape=list(label_arr.shape),
                            value=label_arr.astype(float).ravel().tolist(),
                        )
                    )

            # ====================================================================
            # ======================== Segmentations =============================
            # Predicted mask for segmentation (if available) from row or dataset
            try:
                pred = row.get(SampleStatsEx.PREDICTION.value)
                if task_type != "classification":
                    if pred is not None:
                        try:
                            pred_arr = np.asarray(
                                pred.cpu() if hasattr(pred, "cpu") else pred
                            )

                            # If pred_arr appears to be bounding boxes, convert to mask
                            pred_arr = get_mask(pred_arr, dataset, dataset_index=dataset_index)

                            # Add predicted mask stat
                            data_stats.append(
                                pb2.DataStat(
                                    name='pred_mask',
                                    type='array',
                                    shape=list(pred_arr.shape),
                                    value=pred_arr.astype(float).ravel().tolist(),
                                )
                            )
                        except Exception:
                            pass
                else:
                    # Classification: get prediction from row or dataset
                    if pred is not None:
                        if isinstance(pred, (int, float)):
                            pred_val = pred
                            data_stats.append(
                                pb2.DataStat(
                                    name='pred',
                                    type='scalar',
                                    shape=[1],
                                    value=[pred_val],
                                )
                            )

                        elif isinstance(pred, np.ndarray) and pred.size == 1:
                            pred_val = int(pred.reshape(-1)[0])
                            data_stats.append(
                                pb2.DataStat(
                                    name='pred',
                                    type='scalar',
                                    shape=[1],
                                    value=[pred_val],
                                )
                            )
                        else:
                            # Fallback for non-scalar labels in non-segmentation tasks
                            pred_ = np.array(pred)
                            data_stats.append(
                                pb2.DataStat(
                                    name='pred',
                                    type='array',
                                    shape=list(pred_.shape),
                                    value=pred_.astype(float).ravel().tolist(),
                                )
                            )

            except Exception as e:
                logger.warning(
                    f"Could not get prediction for sample {sample_id}: {e}"
                )

            # ====================================================================
            # ========================== Processing ==============================
            if raw_data_bytes:
                # Generate thumbnail for grid display
                try:
                    raw_img_for_thumb = load_raw_image(dataset, dataset.get_index_from_sample_id(sample_id))
                    thumbnail_bytes = generate_thumbnail(raw_img_for_thumb, max_size=(256, 256), quality=85)
                except Exception as e:
                    logger.debug(f"Could not generate thumbnail for sample {sample_id}: {e}")
                    thumbnail_bytes = b""

                data_stats.append(
                    pb2.DataStat(
                        name='raw_data',
                        type='bytes',
                        shape=raw_shape,
                        value=raw_data_bytes,
                        thumbnail=thumbnail_bytes  # Add thumbnail
                    )
                )
            if transformed_data_bytes:
                data_stats.append(
                    pb2.DataStat(
                        name='transformed_data', type='bytes',
                        shape=transformed_shape, value=transformed_data_bytes
                    )
                )

            return pb2.DataRecord(sample_id=sample_id, data_stats=data_stats)

        except Exception as e:
            logger.error(f"Error processing row for sample_id {row.get(SampleStatsEx.SAMPLE_ID.value, -1)}: {e}", exc_info=True)
            return None

    def _get_unique_tags(self) -> List[str]:
        """Collect all unique tags currently present in the tracked datasets."""
        tags = set()
        try:
            # Extract tags from the dataframe if it exists and has a tags column
            if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                if SampleStatsEx.TAGS.value in self._all_datasets_df.columns:
                    tag_values = self._all_datasets_df[SampleStatsEx.TAGS.value].dropna()
                    for tag_val in tag_values:
                        if tag_val:
                            # Tags are stored as comma-separated strings
                            for t in str(tag_val).split(';'):
                                clean_t = t.strip()
                                if clean_t:
                                    tags.add(clean_t)
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
        - number_of_discarded_samples: rows with deny_listed == True (if column exists)
        - number_of_samples_in_the_loop: rows not deny_listed
        """
        total_count = len(df)
        discarded_count = (
            len(df[df.get("deny_listed", False) == True])  # noqa: E712
            if df is not None and "deny_listed" in df.columns
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
            # Robust parsing for columns with spaces
            sort_part = sort_part.strip()
            ascending = True
            col_raw = sort_part
            
            # Check for direction suffix
            lower_s = sort_part.lower()
            if lower_s.endswith(" asc"):
                ascending = True
                col_raw = sort_part[:-4].strip()
            elif lower_s.endswith(" desc"):
                ascending = False
                col_raw = sort_part[:-5].strip()
            
            # Clean up quotes from column name
            col = col_raw.replace('`', '').strip()
            
            # Split if multiple columns? (Not supported by simple "sortby" textual interface easily yet, assumes single col)
            # But the front-end sends single col.
            
            if col:
                logger.debug(f"[_parse_direct_query] Sort: col={repr(col)}, ascending={ascending}")
                
                if col.lower() == 'index':
                    operations.append({
                        "function": "df.sort_index",
                        "params": {"ascending": ascending}
                    })
                else:
                    operations.append({
                        "function": "df.sort_values",
                        "params": {"by": col, "ascending": ascending}
                    })
        
        logger.debug(f"[_parse_direct_query] Parsed into {len(operations)} operations: {operations}")
        return operations

    def _apply_agent_operation(self, df, func: str, params: dict) -> str:
        """
        Apply an agent-described operation to df in-place.

        Returns a short human-readable message describing what was applied.
        """
        # A) Agent-driven df.query → keep/filter rows via in-place drop
        if func == "df.query":
            expr = params.get("expr", "")

            try:
                # 1. Try pandas query() first: Handles 'col == val' and backticks natively.
                # This is the cleanest syntax for the agent.
                kept = df.query(expr)
                df.drop(index=df.index.difference(kept.index), inplace=True)
                return f"Applied query: {expr}"
            except Exception as query_error:
                try:
                    # 2. Try df.eval(): Handles column names AND explicit 'df' prefixes.
                    # This covers cases like "mean_loss > df['mean_loss'].mean()"
                    mask = df.eval(expr, local_dict={"df": df, "np": np, "pd": pd})
                    if isinstance(mask, (pd.Series, np.ndarray)):
                        kept = df[mask]
                        df.drop(index=df.index.difference(kept.index), inplace=True)
                        return f"Applied query (df.eval): {expr}"
                    else:
                        raise ValueError("eval did not return a boolean mask")
                except Exception as eval_error:
                    try:
                        # 3. Final fallback to raw eval() for complex logic that pandas might block.
                        mask = eval(expr, {"df": df, "np": np, "pd": pd})
                        if isinstance(mask, (pd.Series, np.ndarray, list)):
                            kept = df[mask]
                            df.drop(index=df.index.difference(kept.index), inplace=True)
                            return f"Applied query (raw eval): {expr}"
                        else:
                            raise ValueError("raw eval did not return a mask")
                    except Exception as raw_eval_error:
                        logger.error(f"Query failed. query() error: {query_error}, eval() error: {eval_error}, raw_eval() error: {raw_eval_error}")
                        return f"Failed to apply query: {query_error}"

        # B) Other supported Pandas operations (drop, sort, head, tail, sample)
        if func in {"df.drop", "df.sort_values", "df.sort_index", "df.head", "df.tail", "df.sample"}:
            func_name = func.replace("df.", "")

            try:
                # -------- DROP --------
                if func_name == "drop" and "index" in params:
                    index_expr = params["index"]
                    logger.debug(
                        "[ApplyDataQuery] Applying df.drop with index expression: %r",
                        index_expr
                    )
                    index_to_drop = eval(index_expr, {"df": df, "np": np})
                    df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: drop"

                # ---- SORT_VALUES ----
                if func_name == "sort_values":
                    safe_params = params.copy()
                    by = safe_params.get("by", [])
                    if isinstance(by, str):
                        by = [by]

                    logger.debug(
                        "[ApplyDataQuery] Preparing in-place sort_values on columns %s with params %s",
                        by, safe_params
                    )

                    # Case-insensitive and format-tolerant column matching
                    corrected_by = []
                    
                    def normalize_name(n):
                        return str(n).lower().replace(' ', '').replace('_', '')

                    # Build map of normalized names to actual column/index names
                    df_cols_map = {}
                    # Priority to columns, then index
                    candidates = list(df.columns) + list(df.index.names)
                    for c in candidates:
                        if c:
                            df_cols_map[normalize_name(c)] = c

                    for col in by:
                        if col in df.columns or col in df.index.names:
                            corrected_by.append(col)
                        else:
                            norm_col = normalize_name(col)
                            if norm_col in df_cols_map:
                                corrected_col = df_cols_map[norm_col]
                                logger.debug(f"[ApplyDataQuery] Fuzzy matched sort column '{col}' to '{corrected_col}'")
                                corrected_by.append(corrected_col)
                            else:
                                corrected_by.append(col)
                    by = corrected_by

                    from pandas.api.types import (
                        is_categorical_dtype,
                        is_numeric_dtype,
                        is_object_dtype,
                    )

                    # Sanitize sort columns so sort_values is less fragile
                    for col in by:
                        # Skip index columns for sanitization
                        if col in df.index.names and col not in df.columns:
                             continue
                             
                        if col not in df.columns:
                            continue

                        s = df[col]

                        # 1) Categorical → cast to str to avoid "categories must be unique"
                        if is_categorical_dtype(s.dtype):
                            logger.debug(
                                "[ApplyDataQuery] Column %r is categorical; casting to str before sorting",
                                col,
                            )
                            df[col] = s.astype(str)
                            continue

                        # 2) Object/string → try to interpret as numeric for better sorting
                        if is_object_dtype(s.dtype) and not is_numeric_dtype(s.dtype):
                            logger.debug(
                                "[ApplyDataQuery] Column %r is object; attempting numeric conversion for sort",
                                col,
                            )
                            try:
                                converted = pd.to_numeric(s)
                                if is_numeric_dtype(converted.dtype):
                                    logger.debug(
                                        "[ApplyDataQuery] Column %r converted to numeric dtype %s",
                                        col, converted.dtype,
                                    )
                                    df[col] = converted
                            except (ValueError, TypeError):
                                pass

                    # Ensure all sort columns exist and are sortable (scalars only)
                    valid_cols = []
                    for c in by:
                        is_index = c in df.index.names
                        if c not in df.columns and not is_index:
                            continue
                            
                        # Skip array check for index (assumed scalar/hashable)
                        if is_index and c not in df.columns:
                             valid_cols.append(c)
                             continue

                        # Check for non-scalar types (like numpy arrays in 'prediction_loss')
                        # We use a heuristic on the first non-null value
                        non_null_s = df[c].dropna()
                        if not non_null_s.empty:
                            first_val = non_null_s.iloc[0]
                            # If it's a collection but not a string/bytes, pandas can't sort it directly
                            if hasattr(first_val, "__len__") and not isinstance(first_val, (str, bytes)):
                                c_lower = c.lower()
                                # Fallback: if user asked for 'prediction_loss', help them by using 'mean_loss'
                                if c_lower == "prediction_loss" and "mean_loss" in df.columns:
                                    logger.info("[ApplyDataQuery] Column %r contains arrays; redirecting to 'mean_loss' for sorting", c)
                                    valid_cols.append("mean_loss")
                                    continue
                                # Special handling for tags/categories: sort by string representation (grouping)
                                elif c_lower in ["tags", "task_type"]:
                                    logger.debug("[ApplyDataQuery] Column %r is list-like; casting to string for sorting", c)
                                    df[c] = df[c].astype(str)
                                    valid_cols.append(c)
                                    continue
                                # Robust handling for prediction/target: try to extract scalar (classification) or group as string
                                elif c_lower in ["prediction", "target", "label", "pred"]:
                                    logger.debug("[ApplyDataQuery] Column %r is list-like; attempting scalar extraction for sort", c)
                                    try:
                                        def try_scalar(x):
                                            if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
                                                if len(x) > 0: return x[0]
                                                return None
                                            return x
                                        df[c] = df[c].apply(try_scalar)
                                        df[c] = pd.to_numeric(df[c], errors='ignore')
                                    except Exception as e:
                                        logger.debug(f"[ApplyDataQuery] Scalar extraction failed for {c}: {e}")
                                        df[c] = df[c].astype(str)
                                    valid_cols.append(c)
                                    continue
                                else:
                                    logger.warning("[ApplyDataQuery] Skipping column %r for sorting: contains non-scalar values (e.g. arrays)", c)
                                    continue

                        valid_cols.append(c)

                    if not valid_cols:
                        logger.warning("[ApplyDataQuery] No valid sort columns found in %s", by)
                        return "Failed to sort: columns not found"

                    safe_params["by"] = valid_cols

                    # Fill NaN values in sort columns to avoid sort failures or inconsistent behaviors
                    for c in valid_cols:
                         if c in df.index.names and c not in df.columns:
                             continue
                         if df[c].isna().any():
                             # Use a type-appropriate fill value
                             if is_numeric_dtype(df[c].dtype):
                                 df[c] = df[c].fillna(-1e9) # Sort NaNs to start/end depending on order
                             else:
                                 df[c] = df[c].fillna("")

                    safe_params["inplace"] = True

                    logger.debug(
                        "[ApplyDataQuery] Applying df.sort_values(inplace=True) with params=%s on df shape=%s",
                        safe_params, df.shape
                    )

                    try:
                        df.sort_values(**safe_params)
                    except ValueError as e:
                        # Fallback for categorical issues
                        if "Categorical categories must be unique" in str(e):
                            logger.warning(
                                "[ApplyDataQuery] sort_values failed due to non-unique categorical "
                                "categories; casting sort columns to str and retrying."
                            )
                            for col in by:
                                if col in df.columns:
                                    df[col] = df[col].astype(str)
                            df.sort_values(**safe_params)
                        else:
                            raise

                    return "Applied operation: sort_values"

                # -------- SORT INDEX --------
                if func_name == "sort_index":
                    safe_params = {}
                    if "ascending" in params:
                        safe_params["ascending"] = params["ascending"]
                    
                    logger.debug(
                        "[ApplyDataQuery] Applying df.sort_index(inplace=True) with params=%s",
                        safe_params
                    )
                    df.sort_index(inplace=True, **safe_params)
                    return "Applied operation: sort_index"

                # -------- HEAD --------
                if func_name == "head":
                    n = int(params.get("n", 5))
                    logger.debug(
                        "[ApplyDataQuery] Applying head (in-place) with n=%d on df shape=%s",
                        n, df.shape
                    )
                    if n < len(df):
                        index_to_keep = df.index[:n]
                        index_to_drop = df.index.difference(index_to_keep)
                        df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: head"

                # -------- TAIL --------
                if func_name == "tail":
                    n = int(params.get("n", 5))
                    logger.debug(
                        "[ApplyDataQuery] Applying tail (in-place) with n=%d on df shape=%s",
                        n, df.shape
                    )
                    if n < len(df):
                        index_to_keep = df.index[-n:]
                        index_to_drop = df.index.difference(index_to_keep)
                        df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: tail"

                # ------ SAMPLE -------
                if func_name == "sample":
                    logger.debug(
                        "[ApplyDataQuery] Applying sample (in-place) with params=%s on df shape=%s",
                        params, df.shape
                    )
                    # Support either n or frac; default to 50% if unspecified
                    n = params.get("n")
                    frac = params.get("frac")
                    if n is not None:
                        sampled = df.sample(n=int(n))
                    elif frac is not None:
                        sampled = df.sample(frac=float(frac))
                    else:
                        sampled = df.sample(frac=0.5)

                    index_to_drop = df.index.difference(sampled.index)
                    df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: sample"

            except Exception as e:
                logger.error(
                    f"Failed to apply agent operation {func_name} with params {params}: {e}",
                    exc_info=True
                )
                return f"Failed to apply {func_name}: {e}"

        # C) ANALYSIS (Read-only queries)
        if func == "df.analyze":
            code = params.get("code")
            if not code: return "No code provided for analysis"

            # Simple safety check: ensure code starts with df or looks like expression
            # We trust the Developer environment, but basic guardrails help
            if "import " in code or "__" in code:
                return "Safety Violation: Code contains restricted keywords"

            try:
                # We need a context where 'df' is available
                # Note: eval() expects an expression, not statements.
                # If the agent generated statements, we might need exec() but Intent schema asks for expression.
                result = eval(code, {"df": df, "pd": pd, "np": np})

                # Format the result gracefully
                # Format the result gracefully
                if isinstance(result, (int, np.integer)):
                     return f"Analysis Result: {result}"
                elif isinstance(result, (float, np.floating)):
                     return f"Analysis Result: {result:.4f}"
                elif isinstance(result, (list, dict, set, tuple)):
                    return f"Analysis Result: {result}"
                else:
                    return f"Analysis Result: {str(result)}"

            except Exception as e:
                logger.error(f"Analysis Failed: code={code}, error={e}")
                return f"Analysis Error: {e}"

        # C) Unrecognized function: no-op, but log it
        logger.warning(
            "[ApplyDataQuery] Agent returned unrecognized function: %s. No operation applied.",
            func
        )
        return "No operation applied"

    def _hydrate_tags_for_slice(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """Ensure tags column exists on the slice (values already streamed from H5)."""
        if df_slice is None or df_slice.empty:
            return df_slice

        if SampleStatsEx.TAGS.value not in df_slice.columns:
            df_slice[SampleStatsEx.TAGS.value] = ""
        return df_slice

    def _slowUpdateInternals(self):
        current_time = time.time()
        if self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 5:
            return

        updated_df = self._pull_into_all_data_view_df()

        if hasattr(self, "_all_datasets_df") and self._all_datasets_df is not None and not self._all_datasets_df.empty:
            # If the current DF is sorted differently than the default 'sample_id' order,
            # we should try to maintain that sort order with the new data.
            # Simple heuristic: if the index has changed (reordered), re-apply it.

            # 1. Update the new DF with the new data
            # 2. Reindex the new DF to match the old DF's index order (intersection)
            # This keeps the user's sort valid for existing items.
            pass

            # Check if we have a custom sort (index is not strictly increasing monotonic)
            # AND if the index types are compatible (both numeric)
            if not self._all_datasets_df.index.is_monotonic_increasing:
                 # We have a custom sort.
                 # 1. Check strict equality first (fastest)
                 if self._all_datasets_df.index.equals(updated_df.index):
                     pass  # Index match, just use updated_df as is

                 else:
                     # We have a custom sort or mismatch.
                     old_index = self._all_datasets_df.index
                     new_index = updated_df.index

                     # Optimize intersection using set for O(1) lookups
                     new_index_set = set(new_index)
                     kept_indices = [x for x in old_index if x in new_index_set]

                     # Identify rows that are NEW
                     old_index_set = set(old_index)
                     newly_added_indices = [x for x in new_index if x not in old_index_set]

                     # Construct full order
                     full_order = kept_indices + newly_added_indices

                     # Reindex using this full order.
                     updated_df = updated_df.reindex(full_order)

        self._all_datasets_df = updated_df
        self._last_internals_update_time = current_time

    def ApplyDataQuery(self, request, context):
        """
        Apply a query on the in-memory dataframe.

        Modes:
          - request.query == ""  → just return counts, do not modify df
          - request.query != ""  → always handled by the agent (natural language path)

        Counts returned:
          - number_of_all_samples: all rows currently in the dataframe
          - number_of_samples_in_the_loop: rows not deny_listed
          - number_of_discarded_samples: rows with deny_listed == True
        """
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
                    "[ApplyDataQuery] ⚡ BYPASSING AGENT - Direct query execution: %r",
                    request.query,
                )
                
                # Parse the query directly without agent
                # Expected format: "col > val and col2 == val2 sortby col desc"
                operations = self._parse_direct_query(request.query)
                
                # Apply operations with lock
                with self._lock:
                    df = self._all_datasets_df
                    messages = []
                    
                    for op in operations:
                        func = op.get("function")
                        params = op.get("params", {}) or {}
                        msg = self._apply_agent_operation(df, func, params)
                        messages.append(msg)
                    
                    final_message = " | ".join(messages) if messages else "No operation performed"
                    self._all_datasets_df = df
                    
                    return self._build_success_response(
                        df=df,
                        message=final_message,
                        intent_type=pb2.INTENT_FILTER
                    )
            
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

            # Agent translates query text → operations spec (List[dict])
            # Executed outside the lock to keep grid responsive during LLM waiting time
            operations = self._agent.query(request.query)
            if isinstance(operations, dict): operations = [operations] # Backwards compat
            if not operations: operations = []

            # 3) Apply Operations (CPU/MEMORY BOUND - REQUIRES LOCK)
            with self._lock:
                # Start with the current authoritative DF
                df = self._all_datasets_df
                messages = []
                intent_type = pb2.INTENT_FILTER
                analysis_result = ""

                for i, op in enumerate(operations):
                    func = op.get("function")
                    params = op.get("params", {}) or {}

                    # 2a) Agent-driven RESET has highest priority
                    if params.get("__agent_reset__"):
                        logger.debug("[ApplyDataQuery] Agent requested reset")
                        # Rebuild from loaders; this is the only place we replace the df object
                        self._all_datasets_df = self._pull_into_all_data_view_df()
                        df = self._all_datasets_df  # Reset df to full dataset
                        messages.append("Reset view")
                        continue

                    # 2b) All other agent operations mutate df in-place
                    # df is now carried forward across iterations
                    msg = self._apply_agent_operation(df, func, params)
                    messages.append(msg)

                    # Determine Intent Type based on message prefix (Last analysis wins or combined?)
                    if msg.startswith("Analysis Result:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg.replace("Analysis Result:", "").strip()
                    elif msg.startswith("Analysis Error:") or msg.startswith("Safety Violation:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg

                final_message = " | ".join(messages) if messages else "No operation performed"

                # 4) Persist the filtered df back for next query (CRITICAL for sequential filters!)
                # Only update if it was a manipulation query, not analysis
                if intent_type == pb2.INTENT_FILTER:
                    self._all_datasets_df = df

                # 5) Return updated counts after mutation
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
        Only allowed when training is paused.
        """
        try:
            logger.info(
                "GetSamples called with start_index=%s, records_cnt=%s",
                request.start_index, request.records_cnt
            )

            # Validate request parameters
            if request.start_index < 0 or request.records_cnt <= 0:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Invalid start_index or records_cnt",
                    data_records=[]
                )

            # Get the requested slice of the dataframe (optionally filtered by origin)
            origin_filter = self._get_origin_filter(request)

            with self._lock:
                self._slowUpdateInternals()
                df_view = self._filter_df_by_origin(self._all_datasets_df, origin_filter)
                end_index = request.start_index + request.records_cnt
                df_slice = df_view.iloc[request.start_index:end_index].reset_index()

            # Load tags only for the displayed slice (stream-friendly)
            df_slice = self._hydrate_tags_for_slice(df_slice)

            if df_slice.empty:
                logger.warning(f"No samples found at index {request.start_index}:{end_index}")
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}:{end_index}",
                    data_records=[]
                )

            logger.info(
                "Retrieving samples from %s to %s", request.start_index, end_index)

            # Build the data records list using shared executor
            data_records = []
            tasks = [(row, request, df_slice.columns) for _, row in df_slice.iterrows()]

            results = self._data_executor.map(self._process_sample_row, tasks, timeout=30)
            data_records = [res for res in results if res is not None]

            logger.info("Retrieved %s data records", len(data_records))
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

    def EditDataSample(self, request, context):
        """
        Edit sample metadata (tags and deny_listed).
        """
        if self._all_datasets_df is None:
            self._initialize_data_service()

        self._ctx.ensure_components()

        if request.stat_name not in [SampleStatsEx.TAGS.value, SampleStatsEx.DENY_LISTED.value]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported for now.",
            )

        if request.stat_name == SampleStatsEx.TAGS.value:
            request.string_value = request.string_value or ""

        # No dataset lookups needed; all edits apply directly to the global dataframe.

        # ---------------------------------------------------------------------
        # 1) Mirror edits into the in-memory DataFrame
        # ---------------------------------------------------------------------
        with self._lock:
            if self._all_datasets_df is not None:
                uses_multiindex = isinstance(self._all_datasets_df.index, pd.MultiIndex)
                for sid, origin in zip(request.samples_ids, request.sample_origins):
                    if request.stat_name == SampleStatsEx.TAGS.value:
                        new_val = request.string_value
                        if request.type == pb2.SampleEditType.EDIT_OVERRIDE:
                            # EDIT_OVERRIDE: directly use the new value
                            target_val = new_val or ""
                        elif (request.type == pb2.SampleEditType.EDIT_ACCUMULATE or request.type == pb2.SampleEditType.EDIT_REMOVE):
                            try:
                                if uses_multiindex:
                                    current_val = self._all_datasets_df.loc[(origin, sid), SampleStatsEx.TAGS.value]
                                else:
                                    current_val = self._all_datasets_df.loc[sid, SampleStatsEx.TAGS.value]
                            except KeyError:
                                current_val = ""

                            if pd.isna(current_val): current_val = ""
                            current_tags = [t.strip() for t in str(current_val).split(';') if t.strip()]

                            if request.type == pb2.SampleEditType.EDIT_ACCUMULATE:
                                if new_val not in current_tags:
                                    current_tags.append(new_val)
                            else: # EDIT_REMOVE
                                if new_val in current_tags:
                                    current_tags.remove(new_val)

                            target_val = ";".join(current_tags)
                        else:
                            target_val = new_val or ""
                    else:
                        target_val = request.bool_value

                    try:
                        if uses_multiindex:
                            self._all_datasets_df.loc[(origin, sid), request.stat_name] = target_val
                        else:
                            # Fallback: origin / sample_id as columns
                            mask = (
                                (self._all_datasets_df[SampleStatsEx.SAMPLE_ID.value] == sid)
                                & (self._all_datasets_df[SampleStatsEx.ORIGIN.value] == origin)
                            )
                            self._all_datasets_df.loc[mask, request.stat_name] = target_val


                    except Exception as e:
                        logger.debug(
                            f"[EditDataSample] Failed to update dataframe for sample {sid}: {e}"
                        )

            # ------------------------------------------------------------------
            # 3) Persist edits into the global dataframe manager (in-memory ledger)
            # MOVED INSIDE LOCK to prevent race conditions with _slowUpdateInternals
            # ------------------------------------------------------------------
            try:
                if request.samples_ids and self._df_manager is not None:
                    updates_by_origin = {}
                    for sid, origin in zip(request.samples_ids, request.sample_origins):
                        # Tags
                        if request.stat_name == SampleStatsEx.TAGS.value:
                            value = request.string_value  #if request.stat_name == SampleStatsEx.TAGS.value else request.bool_value

                            # Logic to calculate new tags based on edit type
                            # use self._all_datasets_df (which was just updated above) as source of truth
                            # instead of pulling from _df_manager again, to ensure consistency inside the lock.
                            
                            uses_multiindex = self._all_datasets_df is not None and isinstance(self._all_datasets_df.index, pd.MultiIndex)
                            current_val = ""
                            try:
                                if uses_multiindex:
                                    current_val = self._all_datasets_df.loc[(origin, sid), SampleStatsEx.TAGS.value]
                                else:
                                    # Fallback
                                    mask = (self._all_datasets_df[SampleStatsEx.SAMPLE_ID.value] == sid) & (self._all_datasets_df[SampleStatsEx.ORIGIN.value] == origin)
                                    if mask.any():
                                        current_val = self._all_datasets_df.loc[mask, SampleStatsEx.TAGS.value].iloc[0]
                            except Exception:
                                pass
                                
                            target_val = current_val # Already updated above in step 1

                            updates_by_origin.setdefault(origin, []).append({
                                "sample_id": int(sid),
                                SampleStatsEx.ORIGIN.value: origin,
                                request.stat_name: target_val,
                            })
                        else:
                            # Deny_listed
                            updates_by_origin.setdefault(origin, []).append({
                                "sample_id": int(sid),
                                SampleStatsEx.ORIGIN.value: origin,
                                request.stat_name: request.bool_value,
                            })

                    for origin, rows in updates_by_origin.items():
                        df_update = pd.DataFrame(rows).set_index("sample_id")
                        # upsert_df updates the ledger's dataframe immediately
                        self._df_manager.upsert_df(df_update, origin=origin, force_flush=True)

            except Exception as e:
                logger.debug(f"[EditDataSample] Failed to upsert edits into global dataframe: {e}")


        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )

    def GetDataSplits(self, request, context):
        """
        Return the list of available dataset splits (train, test, val, etc.)
        Extracted from the global dataframe's origin column.
        """
        try:
            split_names = []
            with self._lock:
                self._slowUpdateInternals()
                if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                    if SampleStatsEx.ORIGIN.value in self._all_datasets_df.columns:
                        split_names = sorted(self._all_datasets_df[SampleStatsEx.ORIGIN.value].unique().tolist())
                    elif isinstance(self._all_datasets_df.index, pd.MultiIndex):
                        if SampleStatsEx.ORIGIN.value in self._all_datasets_df.index.names:
                            split_names = sorted(self._all_datasets_df.index.get_level_values(SampleStatsEx.ORIGIN.value).unique().tolist())
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
            available = self.is_agent_available()
            msg = "Agent is available" if available else "Agent is not available"
            return pb2.AgentHealthResponse(available=available, message=msg)
        except Exception as e:
            return pb2.AgentHealthResponse(available=False, message=f"Error: {e}")
