import io
from typing import List
import time
import torch
import logging
import os
import traceback
import threading

import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2

from pathlib import Path
from concurrent import futures
from torchvision import transforms
from weightslab.data.h5_dataframe_store import H5DataFrameStore
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.service_utils import load_raw_image
from weightslab.trainer.services.agent.agent import DataManipulationAgent
from weightslab.backend.ledgers import get_dataloaders


# Get global logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _get_stat_from_row(row, stat_name):
    """Extract stat from dataframe row and convert to DataStat message."""
    value = row.get(stat_name)

    if not isinstance(value, (np.ndarray, torch.Tensor)) and (value is None or pd.isna(value)):
        return None

    # Helper for creating DataStat messages
    def make_stat(type_, shape, **kwargs):
        return pb2.DataStat(name=stat_name, type=type_, shape=shape, **kwargs)

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


class DataService:

    """
    Data service helpers + RPCs (for weights_studio UI).

    Images are sent over gRPC as bytes (JPEG) for simplicity and correctness.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.Lock()

        # init references to the context components
        self._ctx.ensure_components()
        self.df_lock = threading.RLock()

        # Dynamically get all available data loaders from the ledger
        loader_names = get_dataloaders()
        self._loaders = {name: self._ctx.components.get(name) for name in loader_names}

        if not self._loaders:
            logger.warning("DataService initialized without any data loaders.")
        else:
            logger.info(f"DataService found loaders: {list(self._loaders.keys())}")

        self._root_log_dir = self._resolve_root_log_dir()
        self._h5_path = self._resolve_h5_path()
        self._stats_store = H5DataFrameStore(self._h5_path) if self._h5_path else None
        self._stats_last_mtime = None

        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = None
        logger.info("DataService initialized.")

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

    def _interaction_allowed(self):
        if self._is_training_active():
            return False, "Training is running; pause to browse or edit data."
        return True, ""

    def _pull_into_all_data_view_df(self):
            """Stream stats from the shared H5 store with timeout to avoid blocking training.

            Uses a timeout on the H5 read lock - if H5 is busy (training is writing),
            we return the last known good snapshot rather than waiting.
            This ensures DataService communication never blocks the training loop.
            """
            if self._stats_store is None:
                return pd.DataFrame()

            # Dynamically discover all loader origins from tracked datasets
            origins = []
            for _, loader in self._loaders.items():
                if loader is not None:
                    tracked_ds = getattr(loader, "tracked_dataset", None)
                    if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                        origin = tracked_ds._dataset_split
                        if origin not in origins:
                            origins.append(origin)

            if not origins:
                logger.warning("[DataService] No dataset origins found from loaders")
                return pd.DataFrame()

            # Attempt non-blocking read from H5 store
            # If H5 is locked (training writing), timeout and return cached snapshot
            try:
                df = self._stats_store.load_all(origins, non_blocking=False)
                if df.empty:
                    return df

                # Ensure MultiIndex for stable slicing and agent operations
                try:
                    if not isinstance(df.index, pd.MultiIndex):
                        df.set_index(["origin", "sample_id"], inplace=True)
                except Exception as e:
                    logger.warning(f"Failed to set index on streamed dataframe: {e}")

                try:
                    self._stats_last_mtime = self._stats_store.path().stat().st_mtime
                except Exception:
                    self._stats_last_mtime = None

                return df
            except TimeoutError:
                # H5 is locked by training - return last snapshot to avoid blocking
                logger.debug("[DataService] H5 read timeout - returning cached snapshot")
                return self._all_datasets_df if self._all_datasets_df is not None else pd.DataFrame()

    def _load_existing_tags(self):
        """Ensure tags column is present on the streamed dataframe."""
        if self._all_datasets_df is None or self._all_datasets_df.empty:
            return

        if "tags" not in self._all_datasets_df.columns:
            try:
                self._all_datasets_df["tags"] = ""
            except Exception:
                pass

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        try:
            origin = row.get('origin', 'unknown')
            # TODO (GP): should be index returned here not sample_id directly, wrong name
            sample_id = int(row.get('sample_id', 0))

            # Find the dataset for this origin dynamically
            dataset = None
            for _, loader in self._loaders.items():
                if loader is None:
                    continue
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == origin:
                        dataset = tracked_ds
                        break

            if dataset is None:
                logger.warning("Unknown or missing dataset for origin '%s' sample %s", origin, sample_id)
                return None

            data_stats = []
            raw_data_bytes, transformed_data_bytes = b"", b""
            raw_shape, transformed_shape = [], []

            if hasattr(dataset, "_getitem_raw"):
                data = dataset._getitem_raw(id=sample_id)
            else:
                data = dataset[sample_id]

            # Unpack data tuple depending on its length
            if len(data) == 1:
                tensor, _, = data
                label = None
            elif len(data) == 2:
                tensor, label = data
            elif len(data) == 3:
                tensor, _ , label= data
            elif len(data) == 4:
                tensor, _, label, _ = data

            if request.include_transformed_data:
                img = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
                transformed_shape = list(img.shape)
                pil_img = transforms.ToPILImage()(img.detach().cpu())
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                transformed_data_bytes = buf.getvalue()

            if request.include_raw_data:
                try:
                    index = dataset.get_index_from_sample_id(sample_id)
                    raw_img = load_raw_image(dataset, index)
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
                            raw_img = raw_img.resize((target_width, target_height))
                    elif request.resize_width > 0 and request.resize_height > 0:
                        if request.resize_width < original_size[0] or request.resize_height < original_size[1]:
                            raw_img = raw_img.resize((request.resize_width, request.resize_height))

                    raw_shape = [raw_img.height, raw_img.width, len(raw_img.getbands())]
                    raw_buf = io.BytesIO()
                    raw_img.save(raw_buf, format='PNG')
                    raw_data_bytes = raw_buf.getvalue()
                except Exception as e:
                    logger.warning(f"Could not load raw image for sample {sample_id}: {e}")
                    raw_data_bytes = transformed_data_bytes
                    raw_shape = transformed_shape

            # Determine task type early so we can conditionally send stats
            base_task_type = getattr(
                dataset,
                "task_type",
                getattr(self._ctx.components.get("model"), "task_type", "classification"),
            )
            task_type = _infer_task_type_from_label(label, default=base_task_type)

            stats_to_retrieve = list(request.stats_to_retrieve)
            if not stats_to_retrieve:
                stats_to_retrieve = [col for col in df_columns if col not in ['sample_id', 'origin']]
                if task_type == "segmentation":
                    # For segmentation: ALWAYS try to retrieve extended loss stats
                    for s in ["mean_loss", "median_loss", "max_loss", "min_loss", "std_loss",
                             "num_classes_present", "dominant_class", "dominant_class_ratio", "background_ratio"]:
                        if s not in stats_to_retrieve:
                            stats_to_retrieve.append(s)

                    # Also pre-emptively include per-class loss columns if we can find num_classes
                    num_classes = getattr(dataset, "num_classes",
                                         getattr(self._ctx.components.get("model"), "num_classes", None))
                    if num_classes is not None:
                        for i in range(int(num_classes)):
                            col_name = f"loss_class_{i}"
                            if col_name not in stats_to_retrieve:
                                stats_to_retrieve.append(col_name)

            # Determine task type early so we can conditionally send stats
            base_task_type = getattr(
                dataset,
                "task_type",
                getattr(self._ctx.components.get("model"), "task_type", "classification"),
            )
            task_type = _infer_task_type_from_label(label, default=base_task_type)

            for stat_name in stats_to_retrieve:
                stat = _get_stat_from_row(row, stat_name)
                if stat is not None:
                    data_stats.append(stat)
                elif stat_name == "tags":
                    # Always send tags, even if empty, so frontend shows it in metadata
                    data_stats.append(pb2.DataStat(
                        name="tags", type="string", shape=[1], value_string=""
                    ))
                elif task_type == "segmentation":
                    # For segmentation: send extended loss stats and per-class losses even if null
                    if stat_name in ["mean_loss", "median_loss", "max_loss", "min_loss", "std_loss",
                                     "num_classes_present", "dominant_class", "dominant_class_ratio", "background_ratio"]:
                        data_stats.append(pb2.DataStat(
                            name=stat_name, type="scalar", shape=[1], value=[0.0]
                        ))
                    elif stat_name.startswith("loss_class_"):
                        # Per-class losses (loss_class_0, loss_class_1, etc.)
                        data_stats.append(pb2.DataStat(
                            name=stat_name, type="scalar", shape=[1], value=[0.0]
                        ))

            # Expose origin and task_type as stats
            data_stats.append(pb2.DataStat(
                name='origin', type='string', shape=[1], value_string=origin))

            data_stats.append(pb2.DataStat(
                name='task_type', type='string', shape=[1], value_string=str(task_type)))

            # Encode label safely depending on task_type  (GT mask / label)
            label_arr = np.array(label.cpu() if hasattr(label, 'cpu') else label)

            if task_type == "segmentation":
                # Treat label as segmentation mask → array stat
                data_stats.append(pb2.DataStat(
                    name='label',
                    type='array',
                    shape=list(label_arr.shape),
                    value=label_arr.astype(float).ravel().tolist(),
                ))

                try:
                    # Prefer dataset attribute if available
                    num_classes = getattr(dataset, "num_classes", None)
                    if num_classes is None:
                        # Fallback: infer from this label
                        if label_arr.size > 0:
                            max_id = int(label_arr.max())
                            num_classes = max(1, max_id + 1)
                        else:
                            num_classes = 1

                    data_stats.append(pb2.DataStat(
                        name="num_classes",
                        type="scalar",
                        shape=[1],
                        value=[float(num_classes)],
                    ))
                except Exception as e:
                    logger.warning(f"Could not infer num_classes for sample {sample_id}: {e}")
            else:
                # Classification / other scalar-like labels
                label_arr = np.asarray(label_arr)
                if label_arr.size == 1:
                    label_val = float(label_arr.reshape(-1)[0])
                    data_stats.append(pb2.DataStat(
                        name='label',
                        type='scalar',
                        shape=[1],
                        value=[label_val],
                    ))
                else:
                    # Fallback for non-scalar labels in non-segmentation tasks
                    data_stats.append(pb2.DataStat(
                        name='label',
                        type='array',
                        shape=list(label_arr.shape),
                        value=label_arr.astype(float).ravel().tolist(),
                    ))

            # Predicted mask for segmentation (if available)
            if task_type == "segmentation" and hasattr(dataset, "get_prediction_mask"):
                try:
                    pred_mask = dataset.get_prediction_mask(sample_id)
                    if pred_mask is not None:
                        pred_arr = np.asarray(
                            pred_mask.cpu() if hasattr(pred_mask, "cpu") else pred_mask
                        )

                        data_stats.append(pb2.DataStat(
                            name='pred_mask',
                            type='array',
                            shape=list(pred_arr.shape),
                            value=pred_arr.astype(float).ravel().tolist(),
                        ))
                except Exception as e:
                    logger.warning(
                        f"Could not get prediction mask for sample {sample_id}: {e}"
                    )

            if raw_data_bytes:
                data_stats.append(pb2.DataStat(
                    name='raw_data', type='bytes', shape=raw_shape,
                    value=raw_data_bytes))
            if transformed_data_bytes:
                data_stats.append(pb2.DataStat(
                    name='transformed_data', type='bytes',
                    shape=transformed_shape, value=transformed_data_bytes))

            return pb2.DataRecord(sample_id=sample_id, data_stats=data_stats)
        except Exception as e:
            logger.error(f"Error processing row for sample_id {row.get('sample_id', -1)}: {e}", exc_info=True)
            return None

    def _get_unique_tags(self) -> List[str]:
        """Collect all unique tags currently present in the tracked datasets."""
        tags = set()
        try:
            # Extract tags from the dataframe if it exists and has a tags column
            if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                if "tags" in self._all_datasets_df.columns:
                    tag_values = self._all_datasets_df["tags"].dropna()
                    for tag_val in tag_values:
                        if tag_val:
                            # Tags are stored as comma-separated strings
                            for t in str(tag_val).split(','):
                                clean_t = t.strip()
                                if clean_t:
                                    tags.add(clean_t)
        except Exception as e:
            logger.warning(f"Error collecting unique tags: {e}")

        return sorted(list(tags))

    def _build_success_response(self, df, message: str) -> pb2.DataQueryResponse:
        """
        Centralized helper so every code path reports counts consistently.

        - number_of_all_samples: all rows in df
        - number_of_discarded_samples: rows with deny_listed == True (if column exists)
        - number_of_samples_in_the_loop: rows not deny_listed
        """
        total_count = len(df)
        discarded_count = (
            len(df[df.get("deny_listed", False) == True])  # noqa: E712
            if "deny_listed" in df.columns
            else 0
        )
        in_loop_count = total_count - discarded_count
        unique_tags = self._get_unique_tags()

        return pb2.DataQueryResponse(
            success=True,
            message=message,
            number_of_all_samples=total_count,
            number_of_samples_in_the_loop=in_loop_count,
            number_of_discarded_samples=discarded_count
        )

    def _apply_agent_operation(self, df, func: str, params: dict) -> str:
        """
        Apply an agent-described operation to df in-place.

        Returns a short human-readable message describing what was applied.
        """
        # A) Agent-driven df.query → keep/filter rows via in-place drop
        if func == "df.query":
            expr = params.get("expr", "")
            print(f"[DEBUG] ENTERED df.query branch with expr={expr}")
            before = len(df)
            kept = df.query(expr)
            print(f"[DEBUG] df.query kept {len(kept)} rows out of {before}")
            df.drop(index=df.index.difference(kept.index), inplace=True)
            print(f"[DEBUG] AFTER DROP df_len={len(df)}")
            return f"Applied query: {expr}"

        # B) Other supported Pandas operations (drop, sort, head, tail, sample)
        if func in {"df.drop", "df.sort_values", "df.head", "df.tail", "df.sample"}:
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

                    from pandas.api.types import (
                        is_categorical_dtype,
                        is_numeric_dtype,
                        is_object_dtype,
                    )

                    # Sanitize sort columns so sort_values is less fragile
                    for col in by:
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
                            converted = pd.to_numeric(s, errors="ignore")
                            if is_numeric_dtype(converted.dtype):
                                logger.debug(
                                    "[ApplyDataQuery] Column %r converted to numeric dtype %s",
                                    col, converted.dtype,
                                )
                                df[col] = converted


                    # Ensure all sort columns exist
                    valid_cols = [c for c in by if c in df.columns]
                    if not valid_cols:
                        logger.warning("[ApplyDataQuery] No valid sort columns found in %s", by)
                        return "Failed to sort: columns not found"

                    safe_params["by"] = valid_cols

                    # Fill NaN values in sort columns to avoid sort failures or inconsistent behaviors
                    for c in valid_cols:
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

        if "tags" not in df_slice.columns:
            df_slice["tags"] = ""
        return df_slice

    def _slowUpdateInternals(self):
        current_time = time.time()
        if self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 5:
            return

        if self._stats_store and not self._stats_store.has_changed_since(self._stats_last_mtime):
            self._last_internals_update_time = current_time
            return

        updated_df = self._pull_into_all_data_view_df()

        if self._all_datasets_df is not None and not self._all_datasets_df.empty:
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
                 old_index = self._all_datasets_df.index
                 new_index = updated_df.index

                 # 1. Identify rows that are in BOTH (intersection) -> keep old order
                 # use .intersection to preserve order of left argument (old_index)
                 kept_indices = [x for x in old_index if x in new_index]

                 # 2. Identify rows that are NEW (difference) -> append to end
                 # Use set difference for speed, then sort or just append
                 old_index_set = set(old_index)
                 newly_added_indices = [x for x in new_index if x not in old_index_set]

                 # 3. Construct the full requested order
                 full_order = kept_indices + newly_added_indices

                 # 4. Reindex using this full order.
                 # The 'updated_df' contains ALL the data (old items updated + new items).
                 # This safely reorders it.
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
        with self._lock:
            df = self._all_datasets_df  # authoritative DF, mutated in-place

            # 1) No query: just report counts
            if request.query == "":
                return self._build_success_response(
                    df=df,
                    message=f"Current dataframe has {len(df)} samples",
                )

            try:
                # 2) All non-empty queries go through the agent
                if not request.is_natural_language:
                    logger.debug(
                        "[ApplyDataQuery] Non-NL flag received but structured path was removed; "
                        "treating query as natural language: %r",
                        request.query,
                    )

                if self._agent is None:
                    return pb2.DataQueryResponse(
                        success=False,
                        message="Natural language queries require agent (not available)",
                    )

                # Agent translates query text → operation spec
                operation = self._agent.query(request.query) or {}
                func = operation.get("function")  # e.g., 'df.query', 'df.sort_values', 'df.drop', ...
                params = operation.get("params", {}) or {}

                # 2a) Agent-driven RESET has highest priority
                if params.get("__agent_reset__"):
                    logger.debug("[ApplyDataQuery] Agent requested reset")

                    # Rebuild from loaders; this is the only place we replace the df object
                    self._all_datasets_df = self._pull_into_all_data_view_df()
                    df = self._all_datasets_df

                    return self._build_success_response(
                        df=df,
                        message="Reset view to base dataset",
                    )

                # 2b) All other agent operations mutate df in-place
                message = self._apply_agent_operation(df, func, params)

                # 3) Return updated counts after mutation
                return self._build_success_response(df=df, message=message)

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

            # Get the requested slice of the dataframe
            # Protect the update and slice with the lock
            with self._lock:
                self._slowUpdateInternals()
                end_index = request.start_index + request.records_cnt
                df_slice = self._all_datasets_df.iloc[request.start_index:end_index].reset_index()

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

            # Build the data records list in parallel with optimized worker count
            data_records = []
            tasks = [(row, request, df_slice.columns) for _, row in df_slice.iterrows()]

            # Use more workers for I/O-bound image processing (CPU count * 2)
            import os
            max_workers = max(min(len(tasks), os.cpu_count() * 2 if os.cpu_count() else 8), 1)

            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = executor.map(self._process_sample_row, tasks, timeout=30)
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
        Edit sample metadata (tags, deny_listed, etc.).
        """
        if self._all_datasets_df is None:
            self._initialize_data_service()

        self._ctx.ensure_components()
        components = self._ctx.components

        if request.stat_name not in ["tags", "deny_listed"]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported",
            )

        if request.stat_name == "tags":
            request.string_value = request.string_value or ""

        # ---------------------------------------------------------------------
        # 1) Apply edits to the underlying editable dataset wrapper (H5 persistence)
        # ---------------------------------------------------------------------
        for sid, origin in zip(request.samples_ids, request.sample_origins):
            dataset = None

            # Find the loader for this origin (match by name suffix or exact match)
            for loader_name, loader in self._loaders.items():
                # Match origin like "train" to "train_loader", "val" to "val_loader", etc.
                if loader is None:
                    continue
                # Get the tracked dataset's split name
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == origin:
                        dataset = tracked_ds
                        break
                # Fallback: match by loader name (e.g., "train_loader" contains "train")
                elif origin in loader_name:
                    dataset = getattr(loader, "tracked_dataset", loader)
                    break

            if dataset is None: continue

            try:
                if hasattr(dataset, "set"):
                    if request.stat_name == "tags":
                        new_val = request.string_value
                        # Get current tags from the dataframe instead of sample_statistics
                        current_tags_str = ""
                        if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                            try:
                                if isinstance(self._all_datasets_df.index, pd.MultiIndex):
                                    current_tags_str = self._all_datasets_df.loc[(origin, sid), "tags"] if "tags" in self._all_datasets_df.columns else ""
                                else:
                                    # Look for the row matching origin and sample_id
                                    mask = (
                                        (self._all_datasets_df["sample_id"] == sid)
                                        & (self._all_datasets_df["origin"] == origin)
                                    )
                                    matching_rows = self._all_datasets_df[mask]
                                    if not matching_rows.empty and "tags" in self._all_datasets_df.columns:
                                        current_tags_str = matching_rows.iloc[0]["tags"]
                            except (KeyError, IndexError):
                                current_tags_str = ""

                        if pd.isna(current_tags_str):
                            current_tags_str = ""
                        current_tags = [t.strip() for t in str(current_tags_str).split(',') if t.strip()]

                        if request.type == pb2.SampleEditType.EDIT_ACCUMULATE and new_val:
                            if new_val not in current_tags:
                                current_tags.append(new_val)
                            new_val = ", ".join(current_tags)
                        elif request.type == pb2.SampleEditType.EDIT_REMOVE and new_val:
                            if new_val in current_tags:
                                current_tags.remove(new_val)
                            new_val = ", ".join(current_tags)
                        elif request.type == pb2.SampleEditType.EDIT_OVERRIDE:
                            # EDIT_OVERRIDE: completely replace tags with new value
                            new_val = new_val or ""

                        dataset.set(sid, "tags", new_val)
                    elif request.stat_name == "deny_listed":
                        dataset.set(sid, "deny_listed", request.bool_value)
            except Exception as e:
                logger.warning(f"Could not edit sample {sid}: {e}")

        # ---------------------------------------------------------------------
        # 2) Mirror edits into the in-memory DataFrame
        # ---------------------------------------------------------------------
        with self._lock:
            if self._all_datasets_df is not None:
                uses_multiindex = isinstance(self._all_datasets_df.index, pd.MultiIndex)
                for sid, origin in zip(request.samples_ids, request.sample_origins):
                    if request.stat_name == "tags":
                        new_val = request.string_value
                        if request.type == pb2.SampleEditType.EDIT_OVERRIDE:
                            # EDIT_OVERRIDE: directly use the new value
                            target_val = new_val or ""
                        elif (request.type == pb2.SampleEditType.EDIT_ACCUMULATE or request.type == pb2.SampleEditType.EDIT_REMOVE):
                            try:
                                if uses_multiindex:
                                    current_val = self._all_datasets_df.loc[(origin, sid), "tags"]
                                else:
                                    current_val = self._all_datasets_df.loc[sid, "tags"]
                            except KeyError:
                                current_val = ""

                            if pd.isna(current_val): current_val = ""
                            current_tags = [t.strip() for t in str(current_val).split(',') if t.strip()]

                            if request.type == pb2.SampleEditType.EDIT_ACCUMULATE:
                                if new_val not in current_tags:
                                    current_tags.append(new_val)
                            else: # EDIT_REMOVE
                                if new_val in current_tags:
                                    current_tags.remove(new_val)

                            target_val = ", ".join(current_tags)
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
                                (self._all_datasets_df["sample_id"] == sid)
                                & (self._all_datasets_df["origin"] == origin)
                            )
                            self._all_datasets_df.loc[mask, request.stat_name] = value

                    except Exception as e:
                        logger.debug(
                            f"[EditDataSample] Failed to update dataframe for sample {sid}: {e}"
                        )

        # ------------------------------------------------------------------
        # 4) Persist tag edits to the tracked dataset (auto-sync to H5)
        # ------------------------------------------------------------------
        if request.stat_name == "tags" and request.samples_ids:
            try:
                for sid in request.samples_ids:
                    tags_str = request.string_value or ""
                    for loader_name, loader in self._loaders.items():
                        if loader is None:
                            continue
                        tracked_ds = getattr(loader, "tracked_dataset", None)
                        if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                            origin = tracked_ds._dataset_split
                            if origin in request.sample_origins:
                                if hasattr(tracked_ds, 'set'):
                                    tracked_ds.set(int(sid), "tags", tags_str)
            except Exception as e:
                logger.warning(f"Could not persist tags to tracked dataset: {e}")

        # ------------------------------------------------------------------
        # 5) Persist edits back to the shared H5 store for all origins
        # ------------------------------------------------------------------
        if self._stats_store and request.samples_ids:
            updates_by_origin = {}
            for sid, origin in zip(request.samples_ids, request.sample_origins):
                value = request.string_value if request.stat_name == "tags" else request.bool_value
                updates_by_origin.setdefault(origin, []).append({"sample_id": sid, request.stat_name: value})

            for origin, rows in updates_by_origin.items():
                try:
                    df_update = pd.DataFrame(rows).set_index("sample_id")
                    self._stats_store.upsert(origin, df_update)
                except Exception as e:
                    logger.debug(f"[EditDataSample] Failed to persist edits for origin={origin}: {e}")

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )

    def GetDataSplits(self, request, context):
        """
        Return the list of available dataset splits (train, test, val, etc.)
        """
        try:
            from weightslab.backend.ledgers import get_dataloaders

            split_names = []
            loader_names = get_dataloaders()

            for loader_name in loader_names:
                loader = self._ctx.components.get(loader_name)
                if loader is None:
                    continue

                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    split_name = tracked_ds._dataset_split
                    if split_name and split_name not in split_names:
                        split_names.append(split_name)

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
