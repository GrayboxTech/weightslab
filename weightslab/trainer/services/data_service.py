import io
import logging
import os
import traceback
import time
from pathlib import Path
from concurrent import futures

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.components.global_monitoring import pause_controller
from .service_utils import load_raw_image
from .agent import DataManipulationAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


"""
class DataService:
    def __init__():
        self.tdata: SampleTrackingDatasetWrapper = ...
        self.edata: SampleTrackingDatasetWrapper = ...

        self.alldf: pd.DataFrame | None = None

    def _pull_signals_from_dataset_into_df(self):
        if not self.alldf:
            self.alldf = pd.concat([self.tdata.df(), self.edata.df()])
        if self._if_partial_update_required():
            # traverse all_df
            # update samples with the fresh signals
            pass
        return self.alldf

    def GetSamples(self, rqst):
        data_records = []
        for sample in self._pull_signals_from_dataset_into_df()[
                rqst.start_index: rqst.start_index + rqst.sample_number]:
            data_records.append(DataRecord.from_sample(sample))
        return pb.SamplesResponse(data_records)

    def EditSamples(self, rqst):
        for sample_id, sample_origin in zip(
                rqst.sample_ids, rqst.sample_origins):
            dataset = self.tdata if sample_origin == "train" else self.edata
            dataset.set(rqst.sample_id, rqst.stat_name, rqst.new_stat_value)

    def ApplyQuery(self, rqst):
        if self._all_df_should_be_refreshed():
            self._pull_signals_from_dataset_into_df()

        self.alldf = self.agent.apply_query(self.alldf, rqst.query)
"""


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
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = time.time()
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
        # We could maintain an efficient synchronized structure that pulls only
        # the rows with updates.

        # Sanity check
        if self._trn_loader is None:
            self._trn_loader = self._ctx.components.get("train_loader")
        if self._tst_loader is None:
            self._tst_loader = self._ctx.components.get("test_loader")

        # Pull stats from both datasets
        tstats = _get_stats(self._trn_loader, "train")
        estats = _get_stats(self._tst_loader, "test")
        concat_df = pd.concat([tstats, estats])
        try:
            concat_df.set_index(["origin", "sample_id"], inplace=True)
        except KeyError as e:
            logger.warning(f"Failed to set index on concat_df: {e}")
        return concat_df
    
    def SlowUpdateInternals(self):
        current_time = time.time()
        if current_time - self._last_internals_update_time <= 10:
            return

        # Just this line, will mess any filtering/ordering that is being applied
        updated_df = self._pull_into_all_data_view_df()

        # Order the rows in updated_df the order in self._all_datasets_df but 
        # also make sure that we only keep the rows that are in self._all_datasets_df
        if self._all_datasets_df is not None and not self._all_datasets_df.empty:
            updated_df = updated_df.reindex(self._all_datasets_df.index, copy=False)

        self._all_datasets_df = updated_df
        self._last_internals_update_time = current_time


    def _process_sample_row(self, args):
        """Process a single dataframe row to create a DataRecord."""
        row, request, df_columns = args
        try:
            origin = row.get('origin', 'unknown')
            sample_id = int(row.get('sample_id', 0))

            if origin == 'train':
                dataset = self._trn_loader.tracked_dataset
            elif origin == 'test':
                dataset = self._tst_loader.tracked_dataset
            else:
                logger.warning("Unknown origin '%s' for sample %s", origin, sample_id)
                return None

            data_stats = []
            raw_data_bytes, transformed_data_bytes = b"", b""
            raw_shape, transformed_shape = [], []

            if hasattr(dataset, "_getitem_raw"):
                tensor, _, label = dataset._getitem_raw(id=sample_id)
            else:
                tensor, _, label = dataset[sample_id]

            if request.include_transformed_data:
                img = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
                transformed_shape = list(img.shape)
                pil_img = transforms.ToPILImage()(img.detach().cpu())
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                transformed_data_bytes = buf.getvalue()

            if request.include_raw_data:
                try:
                    raw_img = load_raw_image(dataset, id=sample_id)
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

            for stat_name in stats_to_retrieve:
                stat = _get_stat_from_row(row, stat_name)
                if stat is not None:
                    data_stats.append(stat)

            data_stats.append(pb2.DataStat(
                name='origin', type='string', shape=[1], value_string=origin))
            label_val = int(np.array(label.cpu() if hasattr(label, 'cpu') else label).item())
            data_stats.append(pb2.DataStat(
                name='label', type='scalar', shape=[1], value=[float(label_val)]))

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

    def _hydrate_tags_for_slice(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """Load tags for the provided slice only and update in-memory views."""
        if df_slice is None or df_slice.empty or "sample_id" not in df_slice.columns:
            return df_slice

        sample_ids = [int(sid) for sid in df_slice["sample_id"].tolist()]
        try:
            tags_map = self._tags_store.load_tags(sample_ids)
        except Exception as e:
            logger.warning(f"Could not load tags from store: {e}")
            return df_slice

        if not tags_map:
            return df_slice

        if "tags" not in df_slice.columns:
            df_slice["tags"] = ""

        # Update the returned slice and the long-lived dataframe (only for these rows)
        for sid, tag in tags_map.items():
            df_slice.loc[df_slice["sample_id"] == sid, "tags"] = tag
            try:
                if isinstance(self._all_datasets_df.index, pd.MultiIndex) and {
                    "origin", "sample_id"
                }.issubset(set(self._all_datasets_df.index.names or [])):
                    idx_mask = self._all_datasets_df.index.get_level_values("sample_id") == sid
                    self._all_datasets_df.loc[idx_mask, "tags"] = tag
                elif "sample_id" in self._all_datasets_df.columns:
                    mask = self._all_datasets_df["sample_id"] == sid
                    self._all_datasets_df.loc[mask, "tags"] = tag
            except Exception:
                continue

        return df_slice


    def ApplyDataQuery(self, request, context):
        """
        Apply a query (structured or natural language) on the in-memory dataframe.

        - If request.query == "": just return counts.
        - If is_natural_language: let the agent translate to an operation.
        - Otherwise: treat query as a df.query expression.
        """

        allowed, msg = self._interaction_allowed()
        if not allowed:
            return pb2.DataQueryResponse(
                success=False,
                message=msg,
            )

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
        Uses the actual proto messages from data_service.proto
        """
        allowed, msg = self._interaction_allowed()
        if not allowed:
            return pb2.DataSamplesResponse(
                success=False,
                message=msg,
                data_records=[],
            )

        self.SlowUpdateInternals()
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

            # Build the data records list in parallel
            data_records = []
            tasks = [(row, request, df_slice.columns) for _, row in df_slice.iterrows()]

            with futures.ThreadPoolExecutor() as executor:
                results = executor.map(self._process_sample_row, tasks)
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

        allowed, msg = self._interaction_allowed()
        if not allowed:
            return pb2.DataEditsResponse(
                success=False,
                message=msg,
            )

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
        # 4) Persist tag edits to the HDF5 store
        # ------------------------------------------------------------------
        if request.stat_name == "tags" and request.samples_ids:
            try:
                tag_map = {int(sid): request.string_value for sid in request.samples_ids}
                self._tags_store.save_tags(tag_map)
            except Exception as e:
                logger.warning(f"Could not persist tags to store: {e}")

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )