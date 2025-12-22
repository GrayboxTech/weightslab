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
from weightslab.components.global_monitoring import pause_controller
from weightslab.trainer.services.service_utils import load_raw_image
from weightslab.trainer.services.agent.agent import DataManipulationAgent


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

def _get_stats(loader, origin: str):
    """Extract dataset statistics from a loader."""
    if loader is None or not hasattr(loader, "tracked_dataset"):
        return pd.DataFrame()
    df = pd.DataFrame(loader.tracked_dataset.as_records())
    df["origin"] = origin
    return df

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
        self._trn_loader = self._ctx.components.get("train_loader")
        self._tst_loader = self._ctx.components.get("test_loader")

        if self._trn_loader is None or self._tst_loader is None:
            logger.error(
                "DataService initialized without train_loader or test_loader.")

        self._root_log_dir = self._resolve_root_log_dir()

        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._load_existing_tags()
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = None
        logger.info("DataService initialized.")

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
        """Pull stats from both loaders into a single indexed dataframe."""
        # Sanity check
        if self._trn_loader is None:
            self._trn_loader = self._ctx.components.get("train_loader")
        if self._tst_loader is None:
            self._tst_loader = self._ctx.components.get("test_loader")

        # Pull stats from both datasets (skip if empty)
        records = []

        tstats = _get_stats(self._trn_loader, "train")
        if not tstats.empty:
            records.extend(tstats.to_dict("records"))

        estats = _get_stats(self._tst_loader, "eval")
        if not estats.empty:
            records.extend(estats.to_dict("records"))

        if not records:
            return pd.DataFrame()

        # Build dataframe once to avoid concat overhead
        df = pd.DataFrame.from_records(records)

        try:
            df.set_index(["origin", "sample_id"], inplace=True)
        except KeyError as e:
            logger.warning(f"Failed to set index on dataframe: {e}")

        return df

    def _load_existing_tags(self):
        """Load all existing tags from tracked dataset on startup and merge into dataframe."""
        if self._all_datasets_df is None or self._all_datasets_df.empty:
            return

        try:
            # Initialize tags column if not present
            if "tags" not in self._all_datasets_df.columns:
                self._all_datasets_df["tags"] = ""

            # Get tracked datasets to load tags
            trn_tracked = self._trn_loader.tracked_dataset if self._trn_loader else None
            tst_tracked = self._tst_loader.tracked_dataset if self._tst_loader else None

            if not trn_tracked and not tst_tracked:
                return

            # Build unified tag lookup dict (faster than per-row lookups)
            tag_dict = {}
            if trn_tracked:
                tag_dict.update(trn_tracked.sample_statistics.get("tags", {}))
            if tst_tracked:
                tag_dict.update(tst_tracked.sample_statistics.get("tags", {}))

            if not tag_dict:
                return

            # Vectorized update: use map for efficiency
            if isinstance(self._all_datasets_df.index, pd.MultiIndex):
                # For MultiIndex, map from sample_id level
                sample_ids = self._all_datasets_df.index.get_level_values("sample_id")
                self._all_datasets_df["tags"] = sample_ids.map(lambda sid: tag_dict.get(int(sid), "")).fillna("")
            else:
                # For regular columns
                self._all_datasets_df["tags"] = self._all_datasets_df["sample_id"].map(
                    lambda sid: tag_dict.get(int(sid), "")
                ).fillna("")

            loaded_count = sum(1 for v in tag_dict.values() if v)
            logger.info(f"Loaded existing tags for {loaded_count} UID(s) from tracked dataset(s)")
        except Exception as e:
            logger.warning(f"Failed to load existing tags: {e}")

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        try:
            origin = row.get('origin', 'unknown')
            # TODO (GP): should be index returned here not sample_id directly, wrong name
            sample_id = int(row.get('sample_id', 0))

            if origin == 'train':
                dataset = self._trn_loader.tracked_dataset
            elif origin == 'test' or origin == 'eval':
                dataset = self._tst_loader.tracked_dataset
            else:
                logger.warning("Unknown origin '%s' for sample %s", origin, sample_id)
                return None

            data_stats = []
            raw_data_bytes, transformed_data_bytes = b"", b""
            raw_shape, transformed_shape = [], []

            if hasattr(dataset, "_getitem_raw"):
                data = dataset._getitem_raw(id=sample_id)
            else:
                data = dataset[sample_id]
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

            stats_to_retrieve = request.stats_to_retrieve
            if not stats_to_retrieve:
                stats_to_retrieve = [col for col in df_columns if col not in ['sample_id', 'origin']]

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
            # Check both loaders for tracked datasets
            for loader in [self._trn_loader, self._tst_loader]:
                if loader and hasattr(loader, "tracked_dataset"):
                    tracked_tags = loader.tracked_dataset.sample_statistics.get("tags", {})
                    for tag_val in tracked_tags.values():
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
            number_of_discarded_samples=discarded_count,
            unique_tags=unique_tags
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

                    safe_params["by"] = by
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
        """Load tags for the provided slice only and update in-memory views."""
        if df_slice is None or df_slice.empty or "sample_id" not in df_slice.columns:
            return df_slice

        try:
            # Get tracked datasets to load tags
            trn_tracked = self._trn_loader.tracked_dataset if self._trn_loader else None
            tst_tracked = self._tst_loader.tracked_dataset if self._tst_loader else None
        except Exception as e:
            logger.warning(f"Could not access tracked datasets: {e}")
            return df_slice

        if "tags" not in df_slice.columns:
            df_slice["tags"] = ""

        # Build tag lookup dict from tracked datasets (avoid repeated dict lookups in loop)
        tag_dict = {}
        if trn_tracked:
            tag_dict.update(trn_tracked.sample_statistics.get("tags", {}))
        if tst_tracked:
            tag_dict.update(tst_tracked.sample_statistics.get("tags", {}))

        # Vectorized update: use map instead of iterating with .loc
        df_slice["tags"] = df_slice["sample_id"].map(lambda sid: tag_dict.get(int(sid), "")).fillna("")

        # Update main dataframe in one vectorized operation
        if self._all_datasets_df is not None:
            try:
                if isinstance(self._all_datasets_df.index, pd.MultiIndex) and {
                    "origin", "sample_id"
                }.issubset(set(self._all_datasets_df.index.names or [])):
                    # For MultiIndex, update matching rows
                    sample_id_level = self._all_datasets_df.index.get_level_values("sample_id")
                    mask = sample_id_level.isin(df_slice["sample_id"].values)
                    for sid in df_slice["sample_id"].unique():
                        tag_val = df_slice[df_slice["sample_id"] == sid]["tags"].iloc[0]
                        idx_mask = sample_id_level == sid
                        self._all_datasets_df.loc[idx_mask, "tags"] = tag_val
                elif "sample_id" in self._all_datasets_df.columns:
                    # For regular columns, update using isin (faster than loop)
                    mask = self._all_datasets_df["sample_id"].isin(df_slice["sample_id"].values)
                    self._all_datasets_df.loc[mask, "tags"] = self._all_datasets_df.loc[mask, "sample_id"].map(
                        lambda sid: tag_dict.get(int(sid), "")
                    )
            except Exception as e:
                logger.debug(f"[_hydrate_tags_for_slice] Could not update main dataframe: {e}")

        return df_slice

    def _slowUpdateInternals(self):
        current_time = time.time()
        if self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= 10:
            return

        # Just this line, will mess any filtering/ordering that is being applied
        updated_df = self._pull_into_all_data_view_df()

        # Order the rows in updated_df the order in self._all_datasets_df but
        # also make sure that we only keep the rows that are in self._all_datasets_df
        updated_df = updated_df.reindex(self._all_datasets_df.index, copy=False)

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
                logger.warning("No samples found at index %s", request.start_index)
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}",
                    data_records=[]
                )

            logger.info(
                "Retrieving samples from %s to %s", request.start_index, end_index)

            # Build the data records list in parallel with optimized worker count
            data_records = []
            tasks = [(row, request, df_slice.columns) for _, row in df_slice.iterrows()]

            # Use more workers for I/O-bound image processing (CPU count * 2)
            import os
            max_workers = min(len(tasks), os.cpu_count() * 2 if os.cpu_count() else 8)

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
        train_loader = components.get("train_loader")
        test_loader = components.get("test_loader")

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
            if origin == "train":
                dataset = getattr(train_loader, "tracked_dataset", train_loader) if train_loader else None
            elif origin in ("test", "eval"):
                dataset = getattr(test_loader, "tracked_dataset", test_loader) if test_loader else None

            if dataset is None: continue

            try:
                if hasattr(dataset, "set"):
                    if request.stat_name == "tags":
                        new_val = request.string_value
                        current_tags_str = dataset.sample_statistics.get("tags", {}).get(sid, "")
                        current_tags = [t.strip() for t in str(current_tags_str).split(',') if t.strip()]
                        
                        if request.type == pb2.SampleEditType.EDIT_ACCUMULATE and new_val:
                            if new_val not in current_tags:
                                current_tags.append(new_val)
                            new_val = ", ".join(current_tags)
                        elif request.type == pb2.SampleEditType.EDIT_REMOVE and new_val:
                            if new_val in current_tags:
                                current_tags.remove(new_val)
                            new_val = ", ".join(current_tags)
                        
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
                        if (request.type == pb2.SampleEditType.EDIT_ACCUMULATE or request.type == pb2.SampleEditType.EDIT_REMOVE) and new_val:
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
                                    
                            new_val = ", ".join(current_tags)
                        target_val = new_val
                    else:
                        target_val = request.bool_value

                    try:
                        if uses_multiindex:
                            self._all_datasets_df.loc[(origin, sid), request.stat_name] = target_val
                        else:
                            self._all_datasets_df.loc[sid, request.stat_name] = target_val
                    except KeyError:
                        pass

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )
