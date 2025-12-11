import io
import logging
import traceback
import time
from concurrent import futures
import threading

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

import weightslab.proto.experiment_service_pb2 as pb2
from .service_utils import load_raw_image
from .agent.agent import DataManipulationAgent

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
    df = pd.DataFrame(loader.tracked_dataset.as_records())
    df["origin"] = origin
    return df


class DataService:
    """
    Data service helpers + RPCs (for weights_studio UI).
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self._lock = threading.Lock()

        # init references to the context components
        self._ctx.ensure_components()

        self._trn_loader = self._ctx.components.get("train_loader")
        self._tst_loader = self._ctx.components.get("test_loader")

        if self._trn_loader is None or self._tst_loader is None:
            logger.error(
                "DataService initialized without train_loader or test_loader.")

        self._all_datasets_df = self._pull_into_all_data_view_df()
        self._agent = DataManipulationAgent(self)

        self._last_internals_update_time = time.time()
        logger.info("DataService initialized.")

    def _pull_into_all_data_view_df(self):
        """Pulls data from train and test loaders into a single DataFrame for all data view."""

        # TODO: implement logic for skipping the full regen if not necessary... if not self.dirty()

        tstats = _get_stats(self._trn_loader, "train")
        estats = _get_stats(self._tst_loader, "test")
        concat_df = pd.concat([tstats, estats])
        concat_df.set_index(["origin", "sample_id"], inplace=True)
        return concat_df

    def SlowUpdateInternals(self):
        current_time = time.time()
        if current_time - self._last_internals_update_time <= 10:
            return

        # Just this line, will mess any filtering/ordering that is being applied
        updated_df = self._pull_into_all_data_view_df()

        # Order the rows in updated_df the order in self._all_datasets_df but 
        # also make sure that we only keep the rows that are in self._all_datasets_df
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
                tensor, _, label = dataset._getitem_raw(sample_id)
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
                    
                    raw_img = load_raw_image(dataset, sample_id)
                    original_size = raw_img.size
                    
                    # Handle resize request
                    # Negative values indicate percentage mode (e.g., -50 means 50% of original)
                    # Positive values indicate absolute pixel dimensions
                    # Zero means no resize
                    if request.resize_width < 0 and request.resize_height < 0:
                        # Percentage mode
                        percent = abs(request.resize_width) / 100.0
                        target_width = int(original_size[0] * percent)
                        target_height = int(original_size[1] * percent)
                        
                        # Only resize if we're actually reducing size
                        if target_width < original_size[0] or target_height < original_size[1]:
                            raw_img = raw_img.resize((target_width, target_height))
                    elif request.resize_width > 0 and request.resize_height > 0:
                        # Absolute pixel mode - only resize if smaller than original to avoid upscaling
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

        return pb2.DataQueryResponse(
            success=True,
            message=message,
            number_of_all_samples=total_count,
            number_of_samples_in_the_loop=in_loop_count,
            number_of_discarded_samples=discarded_count,
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

    def GetDataSamples(self, request, context):
        """
        Retrieve samples from the dataframe with their data statistics.
        Uses the actual proto messages from data_service.proto
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
                self.SlowUpdateInternals()
                end_index = request.start_index + request.records_cnt
                df_slice = self._all_datasets_df.iloc[request.start_index:end_index].reset_index()

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
        """
        Edit sample metadata (tags, deny_listed, etc.).
        """

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
        with self._lock:
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

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )