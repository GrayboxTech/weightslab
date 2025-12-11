from io import BytesIO
import logging
import os
import traceback
import time
from pathlib import Path
from concurrent import futures
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.components.global_monitoring import pause_controller
from .service_utils import load_raw_image
from .agent import DataManipulationAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _get_stat_from_row(row, stat_name):
    """Extract stat from dataframe row and convert to DataStat message."""
    value = row.get(stat_name)
    
    if value is None or pd.isna(value):
        return None
    
    # Helper for creating DataStat messages
    def make_stat(type_, shape, **kwargs):
        return pb2.DataStat(name=stat_name, type=type_, shape=shape, **kwargs)
    
    if isinstance(value, (int, float)):
        return make_stat("scalar", [1], value=[float(value)])
    
    if isinstance(value, str):
        return make_stat("string", [1], value_string=value)
    
    if isinstance(value, (list, np.ndarray)):
        a = np.asarray(value)
        return make_stat(
            "array", list(a.shape), value=a.flatten().astype(float).tolist())
    
    return None


def _get_stats(loader, origin: str):
    """Extract dataset statistics from a loader."""
    if loader is None or not hasattr(loader, "tracked_dataset"):
        return pd.DataFrame()
    df = pd.DataFrame(loader.tracked_dataset.as_records())
    df["origin"] = origin
    return df


class DataService:
    """
    Data service helpers + RPCs (for weights_studio UI).
    
    Images are sent over gRPC as bytes (JPEG) for simplicity and correctness.
    """

    def __init__(self, ctx):
        self._ctx = ctx

        # init references to the context components
        self._ctx.ensure_components()

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
        dfs_to_concat = []
        tstats = _get_stats(self._trn_loader, "train")
        if not tstats.empty:
            dfs_to_concat.append(tstats)
        
        estats = _get_stats(self._tst_loader, "eval")
        if not estats.empty:
            dfs_to_concat.append(estats)
        
        if not dfs_to_concat:
            return pd.DataFrame()
        
        # Efficient concat with minimal copying
        concat_df = pd.concat(dfs_to_concat, ignore_index=False, sort=False)
        
        try:
            concat_df.set_index(["origin", "sample_id"], inplace=True)
        except KeyError as e:
            logger.warning(f"Failed to set index on concat_df: {e}")
        
        return concat_df
    
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
    
    def _dataframe_update(self) -> bool:
        """
        Incrementally refresh the in-memory dataframe from the datasets.
        Only updates columns that actually changed (deny_listed, losses, etc).
        Preserves filtering/ordering and tags edits.
        """
        if self._all_datasets_df is None or self._all_datasets_df.empty:
            self._all_datasets_df = self._pull_into_all_data_view_df()
            return

        try:
            # Sanity check loaders
            if self._trn_loader is None:
                self._trn_loader = self._ctx.components.get("train_loader")
            if self._tst_loader is None:
                self._tst_loader = self._ctx.components.get("test_loader")

            # Get updated stats from both datasets (minimal - just the changing columns)
            tstats = _get_stats(self._trn_loader, "train")
            estats = _get_stats(self._tst_loader, "eval")
            
            if tstats.empty and estats.empty:
                return

            # Columns that change during training (losses, predictions, etc.)
            # but preserve user-edited columns (tags, deny_listed)
            preserve_cols = {"tags", "deny_listed"}
            
            for origin, stats_df in [("train", tstats), ("eval", estats)]:
                if stats_df.empty:
                    continue
                
                for _, row in stats_df.iterrows():
                    sample_id = int(row.get("sample_id", -1))
                    if sample_id < 0:
                        continue
                    
                    try:
                        if isinstance(self._all_datasets_df.index, pd.MultiIndex):
                            # MultiIndex: (origin, sample_id)
                            idx = (origin, sample_id)
                            if idx in self._all_datasets_df.index:
                                # Update only non-preserved columns
                                for col in row.index:
                                    if col not in ["origin", "sample_id"] and col not in preserve_cols:
                                        self._all_datasets_df.loc[idx, col] = row[col]
                        else:
                            # Regular index: update by mask
                            mask = (
                                (self._all_datasets_df["origin"] == origin)
                                & (self._all_datasets_df["sample_id"] == sample_id)
                            )
                            if mask.any():
                                for col in row.index:
                                    if col not in ["origin", "sample_id"] and col not in preserve_cols:
                                        self._all_datasets_df.loc[mask, col] = row[col]
                    except Exception as e:
                        logger.debug(f"Could not update row {origin}/{sample_id}: {e}")
        except Exception as e:
            logger.warning(f"Incremental dataframe update failed, falling back to full refresh: {e}")
            # Fallback: full refresh
            updated_df = self._pull_into_all_data_view_df()
            if self._all_datasets_df is not None and not self._all_datasets_df.empty:
                updated_df = updated_df.reindex(self._all_datasets_df.index, copy=False)
            self._all_datasets_df = updated_df

    def _slowUpdateInternals(self, time_delta: float = 10.0) -> None:
        current_time = time.time()
        if self._last_internals_update_time is not None and current_time - self._last_internals_update_time <= time_delta:
            return
        self._dataframe_update()
        self._last_internals_update_time = current_time

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert torch tensor to PIL Image."""
        img = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
        return transforms.ToPILImage()(img.detach().cpu())

    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        try:
            origin = row.get('origin', 'unknown')
            sample_id = int(row.get('sample_id', 0))

            # Cache dataset lookup
            if origin == 'train':
                dataset = self._trn_loader.tracked_dataset if self._trn_loader else None
            elif origin in ('test', 'eval'):
                dataset = self._tst_loader.tracked_dataset if self._tst_loader else None
            else:
                logger.warning("Unknown origin '%s' for sample %s", origin, sample_id)
                return None

            if dataset is None:
                return None

            data_stats = []

            # Only load images if requested
            if not request.include_transformed_data and not request.include_raw_data:
                # Skip image loading entirely - just get metadata stats
                if hasattr(dataset, "_getitem_raw"):
                    _, _, label = dataset._getitem_raw(id=sample_id)
                else:
                    _, _, label = dataset[sample_id]
            else:
                if hasattr(dataset, "_getitem_raw"):
                    tensor, _, label = dataset._getitem_raw(id=sample_id)
                else:
                    tensor, _, label = dataset[sample_id]

                if request.include_transformed_data and tensor is not None:
                    try:
                        pil_img = self._tensor_to_pil(tensor)
                        transformed_shape = list(np.array(pil_img).shape)
                        buf = BytesIO()
                        # Reduced quality and disabled optimize for speed (quality=75, optimize=False)
                        pil_img.save(buf, format='JPEG', quality=75, optimize=False)
                        buf.seek(0)
                        img_bytes = list(buf.getvalue())
                        data_stats.append(pb2.DataStat(
                            name='transformed_data', type='bytes', shape=transformed_shape,
                            value=img_bytes))
                    except Exception as e:
                        logger.warning(f"Could not process transformed image for sample {sample_id}: {e}")

                if request.include_raw_data:
                    try:
                        raw_img = load_raw_image(dataset, id=sample_id)
                        raw_shape = [raw_img.height, raw_img.width, len(raw_img.getbands())]
                        buf = BytesIO()
                        # Reduced quality and disabled optimize for speed (quality=75, optimize=False)
                        raw_img.save(buf, format='JPEG', quality=75, optimize=False)
                        buf.seek(0)
                        img_bytes = list(buf.getvalue())
                        data_stats.append(pb2.DataStat(
                            name='raw_data', type='bytes', shape=raw_shape,
                            value=img_bytes))
                    except Exception as e:
                        logger.warning(f"Could not load raw image for sample {sample_id}: {e}")

            # Get metadata stats requested
            stats_to_retrieve = request.stats_to_retrieve
            if not stats_to_retrieve:
                stats_to_retrieve = [col for col in df_columns if col not in ['sample_id', 'origin']]

            for stat_name in stats_to_retrieve:
                stat = _get_stat_from_row(row, stat_name)
                if stat is not None:
                    data_stats.append(stat)

            # Add origin stat
            data_stats.append(pb2.DataStat(
                name='origin', type='string', shape=[1], value_string=origin))
            
            # Add label stat (avoid double conversion)
            try:
                if hasattr(label, 'cpu'):
                    label_val = int(label.cpu().item())
                elif hasattr(label, 'item'):
                    label_val = int(label.item())
                else:
                    label_val = int(label)
            except Exception:
                label_val = 0
                
            data_stats.append(pb2.DataStat(
                name='label', type='scalar', shape=[1], value=[float(label_val)]))

            return pb2.DataRecord(sample_id=sample_id, data_stats=data_stats)
        except Exception as e:
            logger.error(f"Error processing row for sample_id {row.get('sample_id', -1)}: {e}", exc_info=True)
            return None

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


    def ApplyDataQuery(self, request, context):
        """
        Apply a query (structured or natural language) on the in-memory dataframe.

        - If request.query == "": just return counts.
        - If is_natural_language: let the agent translate to an operation.
        - Otherwise: treat query as a df.query expression.
        """

        # Make sure we have a dataframe
        if self._all_datasets_df is None:
            try:
                self._all_datasets_df = self._pull_into_all_data_view_df()
            except Exception as e:
                logger.error(f"ApplyDataQuery: could not build dataframe: {e}", exc_info=True)
                return pb2.DataQueryResponse(
                    success=False,
                    message="Data service not available",
                )

        df = self._all_datasets_df

        # ---------------------------------------------------------------------
        # 1) No query: just report counts
        # ---------------------------------------------------------------------
        if request.query == "":
            total_count = len(df)
            discarded_count = (
                len(df[df.get("deny_listed", False) == True])  # noqa: E712
                if "deny_listed" in df.columns
                else 0
            )
            in_loop_count = total_count - discarded_count

            return pb2.DataQueryResponse(
                success=True,
                message=f"Current dataframe has {total_count} samples",
                number_of_all_samples=total_count,
                number_of_samples_in_the_loop=in_loop_count,
                number_of_discarded_samples=discarded_count,
            )

        try:
            # -----------------------------------------------------------------
            # 2) Natural-language query (via agent)
            # -----------------------------------------------------------------
            if request.is_natural_language:
                if self._agent is None:
                    return pb2.DataQueryResponse(
                        success=False,
                        message="Natural language queries require agent (not available)",
                    )

                # Let the agent translate NL → operation spec
                self._agent.df = df
                operation = self._agent.query(request.query) or {}
                func = operation.get("function")
                params = operation.get("params", {}) or {}

                # Agent-driven reset
                if params.get("__agent_reset__"):
                    logger.debug("[ApplyDataQuery] Agent requested reset")
                    # Rebuild from datasets
                    self._all_datasets_df = self._pull_into_all_data_view_df()
                    df = self._all_datasets_df

                    total_count = len(df)
                    discarded_count = (
                        len(df[df.get("deny_listed", False) == True])  # noqa: E712
                        if "deny_listed" in df.columns
                        else 0
                    )
                    in_loop_count = total_count - discarded_count

                    return pb2.DataQueryResponse(
                        success=True,
                        message="Reset view to base dataset",
                        number_of_all_samples=total_count,
                        number_of_samples_in_the_loop=in_loop_count,
                        number_of_discarded_samples=discarded_count,
                    )

                # df.query(expr) operation
                if func == "df.query":
                    expr = params.get("expr", "")
                    logger.debug(
                        "[ApplyDataQuery] Applying df.query with expr=%r on df shape=%s",
                        expr, df.shape
                    )
                    self._all_datasets_df = df.query(expr)
                    message = f"Applied query: {expr}"
                else:
                    # generic operation via agent
                    logger.debug(
                        "[ApplyDataQuery] Applying operation %s on df shape=%s",
                        func, df.shape
                    )
                    self._all_datasets_df = self._agent.apply_operation(df, operation)
                    message = f"Applied operation: {func}"

            # -----------------------------------------------------------------
            # 3) Structured query supplied directly by UI (df.query expression)
            # -----------------------------------------------------------------
            else:
                expr = request.query
                logger.debug(
                    "[ApplyDataQuery] Applying structured df.query with expr=%r on df shape=%s",
                    expr, df.shape
                )
                self._all_datasets_df = df.query(expr)
                message = f"Query [{request.query}] applied"

            # -----------------------------------------------------------------
            # 4) Return updated counts
            # -----------------------------------------------------------------
            df = self._all_datasets_df
            total_count = len(df)
            discarded_count = (
                len(df[df.get("deny_listed", False) == True])  # noqa: E712
                if "deny_listed" in df.columns
                else 0
            )
            in_loop_count = total_count - discarded_count

            return pb2.DataQueryResponse(
                success=True,
                message=message,
                number_of_all_samples=total_count,
                number_of_samples_in_the_loop=in_loop_count,
                number_of_discarded_samples=discarded_count,
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

        self._slowUpdateInternals(time_delta=15)  # update at most every 60s - get last dataframe state
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
        """Edit sample metadata (tags, deny_listed, etc.)."""

        # Make sure dataframe + editable wrappers are initialized
        if self._all_datasets_df is None:
            self._initialize_data_service()

        self._ctx.ensure_components()
        components = self._ctx.components

        # Only support editing these stats for now
        if request.stat_name not in ["tags", "deny_listed"]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported",
            )

        # We currently do not implement accumulate semantics
        if request.type == pb2.SampleEditType.EDIT_ACCUMULATE:
            return pb2.DataEditsResponse(
                success=False,
                message="Accumulate tagging not supported",
            )

        train_loader = components.get("train_loader")
        test_loader = components.get("test_loader")

        # ---------------------------------------------------------------------
        # 1) Apply edits to the underlying editable dataset wrapper
        # ---------------------------------------------------------------------
        for sid, origin in zip(request.samples_ids, request.sample_origins):
            dataset = None
            if origin == "train":
                dataset = getattr(train_loader, "tracked_dataset", train_loader) if train_loader else None
            elif origin in ("test", "eval"):  # accept both, see below
                dataset = getattr(test_loader, "tracked_dataset", test_loader) if test_loader else None

            if dataset is None:
                continue

            try:
                if hasattr(dataset, "set"):
                    if request.stat_name == "tags":
                        dataset.set(sid, "tags", request.string_value)
                    elif request.stat_name == "deny_listed":
                        dataset.set(sid, "deny_listed", request.bool_value)
                else:
                    logger.warning(
                        f"[EditDataSample] Dataset for origin={origin} does not support 'set'; "
                        "only DataFrame will be updated."
                    )
            except Exception as e:
                logger.warning(f"Could not edit sample {sid}: {e}")

        # ---------------------------------------------------------------------
        # 2) Mirror edits into the in-memory DataFrame (used by the UI / agent)
        # ---------------------------------------------------------------------
        if self._all_datasets_df is not None:
            # If origin/sample_id are in the index, use index levels
            uses_multiindex = isinstance(self._all_datasets_df.index, pd.MultiIndex)

            for sid, origin in zip(request.samples_ids, request.sample_origins):
                value = (
                    request.string_value
                    if request.stat_name == "tags"
                    else request.bool_value
                )

                try:
                    if uses_multiindex and set(self._all_datasets_df.index.names) >= {"origin", "sample_id"}:
                        # MultiIndex: select rows by index
                        idx = (origin, sid)
                        # (use .loc on the index tuple directly)
                        self._all_datasets_df.loc[idx, request.stat_name] = value
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

            # Debug AFTER the updates
            try:
                ids = list(request.samples_ids)
                origins = list(request.sample_origins)

                debug_rows = self._all_datasets_df[
                    (self._all_datasets_df["sample_id"].isin(ids))
                    & (self._all_datasets_df["origin"].isin(origins))
                ]
                logger.debug(
                    "[DEBUG EditDataSample] Updated rows:\n%s",
                    debug_rows[["sample_id", "origin", "tags", "deny_listed"]].head(),
                )

                if request.stat_name == "tags":
                    tagged = self._all_datasets_df[
                        self._all_datasets_df["tags"] == request.string_value
                    ]
                    logger.debug(
                        "[DEBUG EditDataSample] rows with tags == %r right after edit: %d",
                        request.string_value,
                        len(tagged),
                    )
            except Exception as e:
                logger.debug(f"[DEBUG EditDataSample] Could not inspect updated rows: {e}")

        # ---------------------------------------------------------------------
        # 3) Keep the optional agent in sync
        # ---------------------------------------------------------------------
        if self._agent is not None and self._all_datasets_df is not None:
            self._agent.df = self._all_datasets_df

        # ------------------------------------------------------------------
        # 4) Persist tag edits to the tracked dataset (auto-sync to H5)
        # ------------------------------------------------------------------
        if request.stat_name == "tags" and request.samples_ids:
            try:
                for sid in request.samples_ids:
                    tags_str = request.string_value or ""
                    # Save to both train and test tracked datasets
                    if self._trn_loader and hasattr(self._trn_loader.tracked_dataset, 'set'):
                        self._trn_loader.tracked_dataset.set(int(sid), "tags", tags_str)
                    if self._tst_loader and hasattr(self._tst_loader.tracked_dataset, 'set'):
                        self._tst_loader.tracked_dataset.set(int(sid), "tags", tags_str)
            except Exception as e:
                logger.warning(f"Could not persist tags to tracked dataset: {e}")

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )