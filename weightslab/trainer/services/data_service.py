import io
from typing import List
import time
import logging
import os
import traceback
import threading
from hashlib import md5

import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.proto.experiment_service_pb2 import SampleEditType

from PIL import Image
from typing import List
from pathlib import Path
from datetime import datetime
from concurrent import futures

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.backend.ledgers import get_dataloaders, get_dataframe
from weightslab.data.data_utils import load_raw_image, load_label


# Get global logger
logger = logging.getLogger(__name__)


def create_data_stat(name, stat_type, shape=None, value=None, value_string="", thumbnail=b""):
    """Helper to create DataStat with all fields properly initialized.

    Args:
        name: Stat name
        stat_type: Type string (scalar, array, string, etc)
        shape: List of shape dimensions
        value: List of float values
        value_string: String value
        thumbnail: Bytes object for thumbnail (default empty bytes)

    Returns:
        pb2.DataStat: Properly initialized DataStat
    """
    return pb2.DataStat(
        name=name,
        type=stat_type,
        shape=shape or [],
        value=value or [],
        value_string=value_string,
        thumbnail=thumbnail
    )


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

        # Resolve root log directory and H5 path for data storage
        self._root_log_dir = self._resolve_root_log_dir()
        self._h5_path = self._resolve_h5_path()
        self._stats_store = H5DataFrameStore(self._h5_path) if self._h5_path else None

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

        self._is_filtered = False  # Track if the current view is filtered/modified by user

        logger.info("DataService initialized.")

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
                # Load dataframe from the shared dataframe manager with arrays autoloaded from h5 storage
                df = self._df_manager.get_combined_df() if self._df_manager is not None else pd.DataFrame()
                if df.empty:
                    return df

                # Ensure sample_id is in index for consistency
                df = df.reset_index().set_index(["sample_id"])

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
        """
        Legacy method - no longer needed with the new tag column system.
        Tags are now stored as individual boolean columns (tags_<tagname>)
        instead of a single "tags" string column.
        """
        # Tags are now handled via individual columns created on demand
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

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        start_total = time.time()
        try:
            origin = row.get(SampleStatsEx.ORIGIN.value, 'unknown')
            sample_id = int(row.get(SampleStatsEx.SAMPLE_ID.value, 0))

            # ===== Step 0: Initialize Variables======
            raw_shape, data_stats = [], []
            raw_data_bytes = b""

            # ====== Step 1: Load dataset ======
            dataset = self._get_dataset(origin)

            # ====== Step 2: Determine task type ======
            label = row.get(SampleStatsEx.TARGET.value)
            if (label is None or (isinstance(label, list) and label == [])) and dataset:
                label = load_label(dataset, sample_id)

            # Enccode task type based on label shape heuristics
            # Maybe we should not care and send data.
            label_ndim = label.ndim if hasattr(label, 'ndim') else len(getattr(label, 'shape', []))
            if label_ndim >= 3:  # if label ndim superior to 3, it should be segmentation, otherwise classification or other scalar-like task (interpreted as cls)
                task_type = 'segmentation'
            else:
                task_type = "classification"

            # ====== Step 5a: Process stats ======
            stats_to_retrieve = list(request.stats_to_retrieve)
            if not stats_to_retrieve:
                # Use set for O(1) lookup instead of O(n) list lookup
                exclude_cols = {
                    SampleStatsEx.SAMPLE_ID.value,
                    SampleStatsEx.ORIGIN.value,
                    SampleStatsEx.TARGET.value,
                    SampleStatsEx.PREDICTION.value,
                    # SampleStatsEx.PREDICTION_RAW.value,  # Show prediction raw in metatadata
                    SampleStatsEx.TASK_TYPE.value,
                }
                stats_to_retrieve = [col for col in df_columns if col not in exclude_cols]

            # Optimized bulk processing of stats
            for stat_name in stats_to_retrieve:
                value = row.get(stat_name)
                # Skip prediction raw array
                if (isinstance(value, np.ndarray) and value.ndim > 1) or (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 0):
                    continue
                if isinstance(value, float):
                    value = round(value, 7)
                if isinstance(value, bool):
                    value = int(value)
                
                # Check if it s a tag column here and handle it as a string stat with the tag name as value
                if stat_name.startswith(f"{SampleStatsEx.TAG.value}"):
                    if value == 1:
                        tag_name = stat_name[len(f"{SampleStatsEx.TAG.value}_"):]  # Remove "tags_" prefix to get tag name
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

            # ====== Step 7: Process labels ======
            if task_type != "classification":
                if label is None:
                    label_arr = np.asarray(row.get(SampleStatsEx.TARGET.value))
                else:
                    label_arr = label

                # Treat label as segmentation mask -> array stat
                data_stats.append(
                    create_data_stat(
                        name='label',
                        stat_type='array',
                        shape=list(label_arr.shape),
                        value=label_arr.astype(float).ravel().tolist(),
                        thumbnail=b""
                    )
                )

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

            elif label is not None:
                # Classification / other scalar-like labels
                # Check if label is NaN (handle both scalars and arrays)
                if self._is_nan_value(label):
                    pass  # Skip NaN labels

                # Handle scalar labels
                try:
                    data_stats.append(
                        create_data_stat(
                            name='label',
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
                                name='label',
                                stat_type='array',
                                shape=list(label_arr.shape),
                                value=label_arr.astype(float).ravel().tolist(),
                                thumbnail=b""
                            )
                        )  
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert label to array: {label}, error: {e}")

            # ====== Step 8: Process predictions ======
            pred = row.get(SampleStatsEx.PREDICTION.value)
            if task_type != "classification":
                if pred is not None:
                    pred_arr = np.asarray(pred)

                    # Add predicted mask stat
                    data_stats.append(
                        create_data_stat(
                            name='pred_mask',
                            stat_type='array',
                            shape=list(pred_arr.shape),
                            value=pred_arr.astype(float).ravel().tolist(),
                            thumbnail=b""
                        )
                    )
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

            # ====== Step 9: Generate raw data bytes and thumbnail (handles 4D volumetric) ======
            if request.include_raw_data:
                from weightslab.data.data_utils import load_raw_image_array
                
                np_img, is_volumetric, original_shape, middle_pil = load_raw_image_array(
                    dataset, dataset.get_index_from_sample_id(sample_id)
                )
                
                if middle_pil is not None:
                    original_size = middle_pil.size
                    target_width = original_size[0]
                    target_height = original_size[1]
                    aspect_ratio = original_size[0] / original_size[1]
                    if request.resize_width < 0 and request.resize_height < 0:
                        percent = abs(request.resize_width) / 100.0
                        target_width = int(original_size[0] * percent)
                        target_height = int(original_size[1] * percent)
                    elif request.resize_width > 0 and request.resize_height > 0:
                        w_limit, h_limit = request.resize_width, request.resize_height
                        if w_limit / h_limit > aspect_ratio:
                            target_height = h_limit
                            target_width = int(target_height * aspect_ratio)
                        else:
                            target_width = w_limit
                            target_height = int(target_width / aspect_ratio)
                    else:
                        if request.resize_width == 0 and request.resize_height == 0:
                            target_height = 360
                            target_width = int(target_height * aspect_ratio)
                    
                    # Resize middle slice for thumbnail if requested (maintain aspect ratio)
                    if target_width != original_size[0] or target_height != original_size[1]:
                        middle_pil = middle_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)

                    # Determine if this is a full-resolution request (modal) or thumbnail (grid)
                    is_full_resolution = (request.resize_width < 0 and abs(request.resize_width) >= 100) or \
                                        (request.resize_height < 0 and abs(request.resize_height) >= 100)

                    if is_volumetric and np_img is not None and is_full_resolution:
                        # Full resolution modal view: Send full 4D array as raw bytes (C-contiguous, float32)
                        np_img_f32 = np.asarray(np_img, dtype=np.float32)
                        if not np_img_f32.flags['C_CONTIGUOUS']:
                            np_img_f32 = np.ascontiguousarray(np_img_f32)
                        raw_data_bytes = np_img_f32.tobytes()
                        # Shape: [Z, H, W, C] using ORIGINAL 4D dimensions, not thumbnail dimensions
                        # original_shape is (Z, H, W, C) or (Z, H, W)
                        if len(original_shape) == 4:
                            if original_shape[1] > original_shape[-1]:
                                raw_shape = list(original_shape)  # [Z, H, W, C]
                            elif original_shape[1] < original_shape[-1]:
                                raw_shape = [original_shape[0], original_shape[2], original_shape[3], original_shape[1]]  # [Z, W, C, H]
                        else:
                            # If original_shape is (Z, H, W), C is 1 for monochrome volumetric data
                            raw_shape = [original_shape[0], original_shape[1], original_shape[2], 1]
                        logger.info(f"[Volumetric] Sending full res: np_img.shape={np_img.shape}, original_shape={original_shape}, raw_shape={raw_shape}, bytes={len(raw_data_bytes)}")
                    else:
                        # Thumbnail for grid OR non-volumetric: Send JPEG of middle slice only
                        raw_buf = io.BytesIO()
                        middle_pil.save(raw_buf, format='JPEG')
                        raw_data_bytes = raw_buf.getvalue()
                        # Shape: [H, W, C] for 3D RGB or [H, W] for 2D grayscale
                        num_channels = len(middle_pil.getbands())
                        if num_channels == 1:
                            raw_shape = [target_height, target_width, 1]
                        else:
                            raw_shape = [target_height, target_width, num_channels]

                    data_stats.append(
                        create_data_stat(
                            name='raw_data',
                            stat_type='bytes',
                            value=raw_data_bytes,
                            shape=raw_shape,
                        )
                    )

            # ====== Step 10: Create DataRecord ======
            record = pb2.DataRecord(sample_id=sample_id, data_stats=data_stats)

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
        - number_of_discarded_samples: rows with deny_listed == True (if column exists)
        - number_of_samples_in_the_loop: rows not deny_listed
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
            sort_cols = []
            sort_ascs = []

            # Split by comma to support multiple columns: "tags asc, target desc"
            # We need to respect quotes potentially, but for now assume simple CSV structure
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
                logger.debug(f"[_parse_direct_query] Sort: cols={sort_cols}, asc={sort_ascs}")

                # Special optimization for single 'index' sort
                if len(sort_cols) == 1 and sort_cols[0].lower() == 'index':
                    operations.append({
                        "function": "df.sort_index",
                        "params": {"ascending": sort_ascs[0]}
                    })
                else:
                    # Multi-column or single column sort
                    # Note: 'index' mixed with columns in sort_values requires actual column named 'index'
                    # or index level support (which sort_values handles if named).
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

        # C) Standard Pandas Ops (drop, sort, head, tail, sample)
        if func in {"df.drop", "df.sort_values", "df.sort_index", "df.head", "df.tail", "df.sample"}:
            func_name = func.replace("df.", "")
            try:
                if func_name == "drop" and "index" in params:
                    index_to_drop = eval(params["index"], {"df": df, "np": np})
                    df.drop(index=index_to_drop, inplace=True)
                    return "Applied operation: drop"

                if func_name == "sort_values":
                    # Params are already cleaned by the Agent's SortHandler
                    try:
                        df.sort_values(inplace=True, **params)
                    except TypeError as e:
                        # Fallback for mixed types (e.g. lists vs strings/floats): sort by string representation
                        logger.warning(f"Sort failed due to type mismatch ({e}). Retrying with string conversion...")
                        if "key" not in params:
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

    def _slowUpdateInternals(self):
        current_time = time.time()
        if self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 10:
            return

        updated_df = self._pull_into_all_data_view_df()

        # Guard against init race conditions
        if updated_df is None:
            return

        if self._is_filtered and self._all_datasets_df is not None:
            # The user has applied a custom view (Filter, Sort, or Aggregation).
            # We want to support Live Updates of values where possible (e.g. "Keep highest loss"),
            # but prevent overwriting the structure with the raw dataset.

            # Check if the current view's index is compatible with the raw source (Sample IDs)
            # If there is overlap, it's likely a Filter/Subset operation -> Update values, keep rows.
            # If no overlap, it's likely an Aggregation (Index changed) -> Freeze view (can't update from raw).

            try:
                # Use intersection to detect compatibility
                # We need to handle MultiIndex vs Index comparisons carefully, generally simplistic check is enough
                common_indices = self._all_datasets_df.index.intersection(updated_df.index)

                if len(common_indices) > 0:
                     # Case A: Filter/Subset. Indices match.
                     # We force the new data to conform to the USER'S current view (rows/order).
                     # providing live updates for the specific samples they are watching.
                     updated_df = updated_df.reindex(self._all_datasets_df.index)
                else:
                     # Case B: Aggregation/Transformation. Indices don't match.
                     # We cannot update an aggregated view (e.g. "Mean Loss by Class") from raw samples
                     # without re-running the aggregation query.
                     # Best behavior: Freeze the view (keep current df) so the user doesn't lose their chart.
                     return
            except Exception as e:
                logger.debug(f"[_slowUpdateInternals] Error matching indices for filtered view: {e}")
                return

        elif hasattr(self, "_all_datasets_df") and self._all_datasets_df is not None and not self._all_datasets_df.empty:
            # Case C: Standard/Unfiltered View.
            # Preserves Sticky Sort if user manually sorted the full list.

            # Simple heuristic: if the index has changed (reordered), re-apply it.

            # 1. Update the new DF with the new data
            # 2. Reindex the new DF to match the old DF's index order (intersection)
            # This keeps the user's sort valid for existing items.

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

    def _process_get_data_samples(self, request, context):
        """
        Actual implementation of GetDataSamples.
        Process the request and retrieve data samples from the dataframe.
        """
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

            with self._lock:
                self._slowUpdateInternals()
                df_slice = self._all_datasets_df.iloc[request.start_index:request.start_index + request.records_cnt].reset_index()  # Reset index for proper row access with sample_id column

            if df_slice.empty:
                logger.warning(f"No samples found at index {request.start_index}:{request.start_index + request.records_cnt}")
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}:{request.start_index + request.records_cnt}",
                    data_records=[]
                )

            logger.debug(
                "Retrieving samples from %s to %s", request.start_index, request.start_index + request.records_cnt)

            # Build the data records list using shared executor
            data_records = []
            tasks = [(row, request, df_slice.columns) for _, row in df_slice.iterrows()]

            logger.debug("Processing %s samples at %s", len(tasks), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            data_records = self._data_executor.map(self._process_sample_row, tasks, timeout=120)
            data_records = list(data_records)
            logger.debug("Completed processing at %s in %.2f seconds\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time)

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
        current_tags_set = set()
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
            tag_updates[tag] = existing_tag_value
        
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
          - number_of_samples_in_the_loop: rows not deny_listed
          - number_of_discarded_samples: rows with deny_listed == True
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
                    df = self._all_datasets_df
                    messages = []

                    for op in operations:
                        func = op.get("function")
                        params = op.get("params", {}) or {}
                        msg = self._apply_agent_operation(df, func, params)
                        messages.append(msg)

                    final_message = " | ".join(messages) if messages else "No operation performed"
                    self._all_datasets_df = df

                    # Direct queries are manipulations -> Freeze the view
                    if operations:
                         self._is_filtered = True

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

            # 3) Apply Operations (CPU/MEMORY BOUND - REQUIRES LOCK)
            with self._lock:
                # Start with the current authoritative DF
                df = self._all_datasets_df
                messages = []
                # Default to FILTER, switch to ANALYSIS if we detect analysis/action output
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
                        self._is_filtered = False   # Unfreeze updates
                        messages.append("Reset view")
                        continue

                    # 2b) All other agent operations mutate df in-place
                    # df is now carried forward across iterations
                    msg = self._apply_agent_operation(df, func, params)
                    messages.append(msg)

                    # --- UPDATED INTENT CLASSIFICATION ---
                    # Check for Clarification
                    if "Clarification needed" in msg or "I need more information" in msg:
                        intent_type = pb2.INTENT_ANALYSIS  # Usually presented as a message/analysis
                        analysis_result = msg

                    # Check for Action Triggers (e.g. "Action: Dataset saved...")
                    elif msg.startswith("Action:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg

                    # Check for Analysis Results
                    elif msg.startswith("Analysis Result:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg.replace("Analysis Result:", "").strip()

                    # Check for Errors
                    elif msg.startswith("Analysis Error:") or msg.startswith("Safety Violation:"):
                        intent_type = pb2.INTENT_ANALYSIS
                        analysis_result = msg

                final_message = " | ".join(messages) if messages else "No operation performed"

                # 4) Persist the filtered df back for next query (CRITICAL for sequential filters!)
                # Only update if it was a manipulation query, not analysis
                if intent_type == pb2.INTENT_FILTER:
                    self._all_datasets_df = df
                    # If we modified the DF, we should freeze it (unless it was a Reset which handled above)
                    # We check if *any* operation was effectively applied.
                    # For simplicity, if intent is FILTER, we assume manipulation happened.
                    self._is_filtered = True

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
        Implements request deduplication to prevent duplicate concurrent requests.
        """

        try:
            # Process the request directly without deduplication logic
            return self._process_get_data_samples(request, context)

        except Exception as e:
            logger.error("Error in GetDataSamples: %s", str(e), exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}",
                data_records=[]
            )

    def EditDataSample(self, request, context):
        """
        Edit sample metadata (tags and deny_listed).
        
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

        if not request.stat_name or not request.stat_name.startswith(SampleStatsEx.TAG.value) and request.stat_name not in [SampleStatsEx.DISCARDED.value]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported for now.",
            )

        # No dataset lookups needed; all edits apply directly to the global dataframe.

        # =====================================================================
        # Process tag edits using the new column-based tag system
        # =====================================================================
        with self._lock:
            try:
                if request.samples_ids and self._df_manager is not None:
                    updates_by_origin = {}
                    is_tag_request = request.stat_name == SampleStatsEx.TAG.value or request.stat_name.startswith(SampleStatsEx.TAG.value)
                    for sid, origin in zip(request.samples_ids, request.sample_origins):
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
                            if sid not in updates_by_origin[origin]:
                                updates_by_origin[origin][sid] = {
                                    "sample_id": int(sid),
                                    SampleStatsEx.ORIGIN.value: origin,
                                }
                            
                            # Merge tag column updates into the sample's updates
                            updates_by_origin[origin][sid].update(tag_updates)
                        
                        # =================
                        # DENY LISTED EDITS
                        # =================
                        else:
                            # Deny_listed
                            if origin not in updates_by_origin:
                                updates_by_origin[origin] = {}
                            updates_by_origin[origin][sid] = {
                                "sample_id": int(sid),
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
                    
                    # Reload dataframe to reflect all changes
                    self._all_datasets_df = self._pull_into_all_data_view_df()
            
                # Prevent _slowUpdateInternals from overwriting our edits with stale data
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
