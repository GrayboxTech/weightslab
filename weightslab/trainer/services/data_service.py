import logging
import traceback
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import weightslab.proto.experiment_service_pb2 as pb2

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class EditableDatasetAdapter(Dataset):
    """
    Wraps an arbitrary dataset to make it compatible with DataService.EditDataSample
    and GetDataSamples.

    It:
      - delegates __len__ and __getitem__ to the underlying dataset
      - adds .set(sample_id, stat_name, value)
      - exposes .sample_statistics as a dict of per-sample metadata
      - forwards unknown attributes to the base dataset
    """

    def __init__(self, base_dataset):
        self.wrapped_dataset = base_dataset  # important: for _dataset_to_df unwrapping
        self.sample_statistics = {
            "tags": {},         # sample_id -> string
            "deny_listed": {},  # sample_id -> bool
        }

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, idx):
        return self.wrapped_dataset[idx]

    def set(self, sample_id: int, stat_name: str, value):
        """
        Called by DataService.EditDataSample:
            dataset.set(sid, "tags", request.string_value)
            dataset.set(sid, "deny_listed", request.bool_value)
        """
        sid = int(sample_id)
        if stat_name not in self.sample_statistics:
            self.sample_statistics[stat_name] = {}

        if stat_name == "tags":
            self.sample_statistics["tags"][sid] = str(value)
        elif stat_name == "deny_listed":
            self.sample_statistics["deny_listed"][sid] = bool(value)
        else:
            self.sample_statistics[stat_name][sid] = value

    def __getattr__(self, name):
        """
        Delegate all other attributes to the wrapped dataset.
        This preserves things like .classes, .class_to_idx, .data, .targets, etc.
        """
        return getattr(self.wrapped_dataset, name)

    def as_records(self):
        """
        If the wrapped dataset supports as_records, call it and then
        inject our local edits (tags, deny_listed) into the result.
        """
        # 1. Get base records
        if hasattr(self.wrapped_dataset, "as_records"):
            records = self.wrapped_dataset.as_records()
        else:
            # If wrapped dataset doesn't have as_records, we raise AttributeError
            # so that the caller knows it's not supported (or __getattr__ would have handled it).
            raise AttributeError(f"'{type(self.wrapped_dataset).__name__}' object has no attribute 'as_records'")

        # 2. Inject stats
        if not records:
            return records

        # We'll iterate and update.
        for row in records:
            sid = row.get("sample_id")
            if sid is None:
                continue
            sid = int(sid)

            # Update tags
            if "tags" in self.sample_statistics and sid in self.sample_statistics["tags"]:
                row["tags"] = self.sample_statistics["tags"][sid]

            # Update deny_listed
            if "deny_listed" in self.sample_statistics and sid in self.sample_statistics["deny_listed"]:
                row["deny_listed"] = self.sample_statistics["deny_listed"][sid]

        return records


class DataService:
    """
    Data service helpers + RPCs (for weights_studio UI).
    """

    def __init__(self, ctx):
        self._ctx = ctx
        # Data service components (initialized lazily on first use)
        # Single dataframe representing the current "view".
        self._all_datasets_df: pd.DataFrame | None = None
        self._agent = None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _make_loader_editable(self, loader):
        """
        Ensure that the loader's underlying dataset is editable:
          - has .set(...)
          - has .sample_statistics

        We wrap loader.dataset (or loader itself) with EditableDatasetAdapter
        if needed, and mutate in place so the rest of the system sees it too.
        """
        if loader is None:
            return None

        # Get the underlying dataset if it's a DataLoader-like
        base_ds = getattr(loader, "dataset", loader)

        # Already supports editing? leave it alone
        if hasattr(base_ds, "set") and hasattr(base_ds, "sample_statistics"):
            return loader

        # Wrap it
        adapted = EditableDatasetAdapter(base_ds)

        # If loader has a .dataset attribute, update it; else we just return the adapted dataset
        if hasattr(loader, "dataset"):
            loader.dataset = adapted
            return loader
        else:
            # loader itself was a dataset instance
            return adapted

    def _get_stat_from_row(self, row, stat_name):
        """Extract stat from dataframe row and convert to DataStat message."""
        try:
            value = row[stat_name]
        except (KeyError, IndexError):
            return None

        if value is None:
            return None

        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return pb2.DataStat(
                name=stat_name,
                type="scalar",
                shape=[1],
                value=[float(value)],
            )
        elif isinstance(value, str):
            return pb2.DataStat(
                name=stat_name,
                type="string",
                shape=[1],
                value_string=value,
            )
        elif isinstance(value, (list, np.ndarray)):
            arr = np.array(value)
            return pb2.DataStat(
                name=stat_name,
                type="array",
                shape=list(arr.shape),
                value=arr.flatten().astype(float).tolist(),
            )
        return None

    # -------------------------------------------------------------------------
    # NEW: overlay runtime sample_statistics (tags / deny_listed / etc.)
    # -------------------------------------------------------------------------
    def _with_runtime_sample_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of df where columns like 'tags' / 'deny_listed' are refreshed
        from the underlying EditableDatasetAdapter.sample_statistics.

        This makes the wrapper the single source of truth, and ensures that
        queries like "tags == 'abc'" see the latest edits, even if the in-memory
        dataframe got out of sync.

        We merge on (origin, sample_id).
        """
        if df is None or df.empty:
            return df

        try:
            self._ctx.ensure_components()
            components = self._ctx.components

            records: list[dict] = []

            def _collect_for_loader(loader, origin: str):
                if not loader:
                    return
                ds = getattr(loader, "dataset", loader)
                if not hasattr(ds, "sample_statistics"):
                    return

                ss = ds.sample_statistics  # dict: stat_name -> {sample_id -> value}
                if not ss:
                    return

                for stat_name, mapping in ss.items():
                    if not mapping:
                        continue
                    for sid, value in mapping.items():
                        records.append(
                            {
                                "origin": origin,
                                "sample_id": int(sid),
                                stat_name: value,
                            }
                        )

            _collect_for_loader(components.get("train_loader"), "train")
            _collect_for_loader(components.get("test_loader"), "eval")

            if not records:
                # nothing to overlay
                return df

            stats_df = pd.DataFrame(records)
            # drop duplicates in case we collected multiple times; keep last
            stats_df = stats_df.drop_duplicates(subset=["origin", "sample_id"], keep="last")

            merged = df.merge(
                stats_df,
                on=["origin", "sample_id"],
                how="left",
                suffixes=("", "_rt"),
            )

            # For every added column (tags, deny_listed, etc.), prefer the runtime value
            for col in stats_df.columns:
                if col in ("origin", "sample_id"):
                    continue
                rt_col = f"{col}_rt"
                if rt_col in merged.columns:
                    mask = merged[rt_col].notna()
                    merged.loc[mask, col] = merged.loc[mask, rt_col]
                    merged = merged.drop(columns=[rt_col])

            return merged

        except Exception as e:
            logger.warning("Failed to overlay runtime sample_statistics: %s", e)
            return df

    # -------------------------------------------------------------------------
    # Initialization / refresh
    # -------------------------------------------------------------------------
    def _initialize_data_service(self):
        """Initialize data service components using ledger-resolved dataloaders."""
        try:
            self._ctx.ensure_components()

            components = self._ctx.components

            train_loader = components.get("train_loader")
            test_loader = components.get("test_loader")

            if train_loader is None or test_loader is None:
                logger.warning("Cannot initialize data service: dataloaders not in ledger")
                return

            # Make sure both loaders have editable datasets (set + sample_statistics)
            train_loader = self._make_loader_editable(train_loader)
            test_loader = self._make_loader_editable(test_loader)

            # Write back into ctx.components so later calls (EditDataSample, etc.) see the wrapped versions
            components["train_loader"] = train_loader
            components["test_loader"] = test_loader

            def _dataset_to_df(dataset_or_loader, origin: str) -> pd.DataFrame:
                """Convert a dataset/loader into a DataFrame usable by the UI."""
                raw_ds = dataset_or_loader
                while True:
                    if hasattr(raw_ds, "wrapped_dataset"):
                        new_ds = raw_ds.wrapped_dataset
                        if new_ds is not None:
                            raw_ds = new_ds
                        else:
                            break
                    elif hasattr(raw_ds, "dataset"):
                        new_ds = raw_ds.dataset
                        if new_ds is not None:
                            raw_ds = new_ds
                        else:
                            break
                    else:
                        break

                    if raw_ds is None:
                        break

                if raw_ds is None:
                    logger.warning(f"raw_ds is None for {origin}, returning empty DF")
                    return pd.DataFrame()

                records = []

                # Fast path for torchvision-style datasets with data/targets (MNIST, etc.)
                if hasattr(raw_ds, "data") and hasattr(raw_ds, "targets"):
                    try:
                        images = raw_ds.data.numpy()
                        labels = raw_ds.targets.numpy()
                        for i in range(len(raw_ds)):
                            records.append(
                                {
                                    "sample_id": i,
                                    "label": int(labels[i]),
                                    "image": images[i],
                                    "origin": origin,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Fast path failed for {origin}: {e}")

                # Fast path for ImageFolder datasets - only load metadata, not images
                # This prevents OOM and timeouts with large datasets
                if not records and hasattr(raw_ds, "samples") and hasattr(raw_ds, "targets"):
                    try:
                        logger.debug(f"Using ImageFolder fast path for {origin} with {len(raw_ds)} samples")
                        for i in range(len(raw_ds)):
                            records.append(
                                {
                                    "sample_id": i,
                                    "label": int(raw_ds.targets[i]),
                                    # Don't load image here - will be loaded on-demand by GetDataSamples
                                    "origin": origin,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"ImageFolder fast path failed for {origin}: {e}")

                # Fallback: iterate samples (WARNING: slow for large datasets!)
                if not records:
                    logger.warning(
                        f"Using slow fallback path for {origin} - this may take a while "
                        f"and consume significant memory for {len(raw_ds)} samples"
                    )
                    for i in range(len(raw_ds)):
                        try:
                            item = raw_ds[i]
                            if isinstance(item, (tuple, list)):
                                img, lbl = item[0], item[-1]
                            else:
                                img, lbl = item, None

                            if hasattr(img, "numpy"):
                                img_arr = img.numpy()
                            else:
                                img_arr = np.array(img)

                            records.append(
                                {
                                    "sample_id": i,
                                    "label": int(lbl) if lbl is not None else None,
                                    "image": img_arr,
                                    "origin": origin,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to convert sample {i}: {e}")
                            continue

                df = pd.DataFrame(records)

                # Merge dynamic stats from wrapper if available
                stats_source = dataset_or_loader
                try:
                    if hasattr(stats_source, "as_records"):
                        stats_records = stats_source.as_records()
                        if stats_records:
                            stats_df = pd.DataFrame(stats_records)
                            if "sample_id" in stats_df.columns:
                                stats_df["sample_id"] = stats_df["sample_id"].astype(int)
                            df = pd.merge(
                                df,
                                stats_df,
                                on="sample_id",
                                how="left",
                                suffixes=("", "_stats"),
                            )

                    # Merge EditableDatasetAdapter.sample_statistics if present
                    # We look on the underlying dataset, not the loader.
                    ds = getattr(stats_source, "dataset", stats_source)
                    if hasattr(ds, "sample_statistics"):
                        ss = ds.sample_statistics  # e.g. {"tags": {sid: str}, "deny_listed": {sid: bool}}
                        stats_records = []
                        # iterate over all sample_ids we know from df
                        for sid in df["sample_id"].unique():
                            sid = int(sid)
                            row = {"sample_id": sid}
                            changed = False
                            for key, mapping in ss.items():
                                if sid in mapping:
                                    row[key] = mapping[sid]
                                    changed = True
                            if changed:
                                stats_records.append(row)

                        if stats_records:
                            ss_df = pd.DataFrame(stats_records)
                            ss_df["sample_id"] = ss_df["sample_id"].astype(int)
                            df = pd.merge(
                                df,
                                ss_df,
                                on="sample_id",
                                how="left",
                                suffixes=("", "_stat"),
                            )
                except Exception as e:
                    logger.warning(f"Failed to merge stats for {origin}: {e}")

                return df

            train_df = _dataset_to_df(train_loader, "train")
            eval_df = _dataset_to_df(test_loader, "eval")

            self._all_datasets_df = pd.concat([train_df, eval_df], ignore_index=True)

            if "tags" not in self._all_datasets_df.columns:
                self._all_datasets_df["tags"] = ""
            if "deny_listed" not in self._all_datasets_df.columns:
                self._all_datasets_df["deny_listed"] = False

            logger.debug(f"Created combined DataFrame with {len(self._all_datasets_df)} samples")
            logger.debug(f"DataFrame columns: {list(self._all_datasets_df.columns)}")

            # Optional: external agent (weights_studio integration)
            try:
                import sys, os

                # path to trainer_services.py
                current_dir = os.path.dirname(os.path.abspath(__file__))

                repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))

                # weights_studio location: /Users/.../v0/weights_studio
                weights_studio_path = os.path.join(repo_root, "weights_studio")

                if os.path.isdir(weights_studio_path) and weights_studio_path not in sys.path:
                    sys.path.append(weights_studio_path)

                from agent.agent import DataManipulationAgent
                import agent.agent as agent

                self._agent = DataManipulationAgent(self._all_datasets_df)
                logger.debug("Data service initialized successfully with agent")

            except ImportError as e:
                logger.warning(f"DataManipulationAgent not available: {e}")
                self._agent = None

        except Exception as e:
            logger.error(f"Data service initialization failed: {e}")
            self._agent = None

    def _refresh_data_stats(self):
        """Refresh dynamic stats in the dataframe from underlying datasets."""
        if self._all_datasets_df is None:
            return

        try:
            self._ctx.ensure_components()
            components = self._ctx.components

            dfs = []

            def _get_stats(loader, origin: str):
                if not loader:
                    return None
                recs = None
                if hasattr(loader, "as_records"):
                    recs = loader.as_records()
                else:
                    ds = getattr(loader, "dataset", loader)
                    if hasattr(ds, "as_records"):
                        recs = ds.as_records()
                if recs:
                    df = pd.DataFrame(recs)
                    df["origin"] = origin
                    if "sample_id" in df.columns:
                        df["sample_id"] = df["sample_id"].astype(int)
                    return df
                return None

            train_stats = _get_stats(components.get("train_loader"), "train")
            if train_stats is not None:
                dfs.append(train_stats)

            eval_stats = _get_stats(components.get("test_loader"), "eval")
            if eval_stats is not None:
                dfs.append(eval_stats)

            if not dfs:
                return

            all_stats = pd.concat(dfs, ignore_index=True)
            if all_stats.empty:
                return

            target_df = self._all_datasets_df.set_index(["origin", "sample_id"])
            source_df = all_stats.set_index(["origin", "sample_id"])

            for col in source_df.columns:
                target_df[col] = source_df[col]

            self._all_datasets_df = target_df.reset_index()

            if self._agent:
                self._agent.df = self._all_datasets_df

        except Exception as e:
            logger.warning(f"Failed to refresh data stats: {e}")

    # -------------------------------------------------------------------------
    # Data service RPCs
    # -------------------------------------------------------------------------
    def ApplyDataQuery(self, request, context):
        """Apply query to filter/sort/manipulate dataset."""
        # Ensure DF + agent are initialized once
        if self._agent is None or self._all_datasets_df is None:
            self._initialize_data_service()
        else:
            self._refresh_data_stats()

        if self._all_datasets_df is None:
            return pb2.DataQueryResponse(
                success=False,
                message="Data service not available",
            )

        # No query: just report counts on the current df
        if request.query == "":
            total_count = len(self._all_datasets_df)
            discarded_count = (
                len(
                    self._all_datasets_df[
                        self._all_datasets_df.get("deny_listed", False) == True  # noqa: E712
                    ]
                )
                if "deny_listed" in self._all_datasets_df.columns
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
            # Start from the current view, but overlay latest runtime stats (tags, deny_listed, etc.)
            source_df = self._with_runtime_sample_statistics(self._all_datasets_df)

            # Debug: tag distribution before any query is applied
            try:
                if "tags" in source_df.columns:
                    logger.debug(
                        "[ApplyDataQuery] source_df tags value_counts before query: %s",
                        source_df["tags"].value_counts().to_dict()
                    )
            except Exception as e:
                logger.warning("[ApplyDataQuery] failed logging tags value_counts: %s", e)

            if request.is_natural_language:
                if self._agent is None:
                    return pb2.DataQueryResponse(
                        success=False,
                        message="Natural language queries require Ollama agent (not available)",
                    )

                # Let the agent translate NL → operation spec
                self._agent.df = source_df
                operation = self._agent.query(request.query)

                func = operation.get("function")
                params = operation.get("params", {})

                if func == "df.query":
                    expr = params.get("expr", "")
                    logger.debug(
                        "[ApplyDataQuery] Applying df.query with expr=%r on df shape=%s",
                        expr, source_df.shape
                    )
                    self._all_datasets_df = source_df.query(expr)
                    message = f"Applied query: {expr}"
                else:
                    # For other operations delegate to agent.apply_operation
                    logger.debug(
                        "[ApplyDataQuery] Applying operation %s on df shape=%s",
                        func, source_df.shape
                    )
                    self._all_datasets_df = self._agent.apply_operation(source_df, operation)
                    message = f"Applied operation: {func}"
            else:
                # Structured query supplied directly by UI
                expr = request.query
                logger.debug(
                    "[ApplyDataQuery] Applying structured df.query with expr=%r on df shape=%s",
                    expr, source_df.shape
                )
                self._all_datasets_df = source_df.query(expr)
                message = f"Query [{request.query}] applied"

            total_count = len(self._all_datasets_df)
            discarded_count = (
                len(
                    self._all_datasets_df[
                        self._all_datasets_df.get("deny_listed", False) == True  # noqa: E712
                    ]
                )
                if "deny_listed" in self._all_datasets_df.columns
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
            logger.error(f"Failed to apply query: {e}", exc_info=True)
            return pb2.DataQueryResponse(
                success=False,
                message=f"Failed to apply query: {str(e)}",
            )

    def GetDataSamples(self, request, context):
        """Retrieve samples with their data statistics."""
        logger.debug(
            f"GetDataSamples called: start_index={request.start_index}, "
            f"records_cnt={request.records_cnt}"
        )

        if self._all_datasets_df is None:
            logger.debug("Initializing data service (first call)")
            self._initialize_data_service()

        if self._all_datasets_df is None:
            logger.error("Data service initialization failed - no dataframe available")
            return pb2.DataSamplesResponse(
                success=False,
                message="Data service not available",
                data_records=[],
            )

        # Always work with a view that has up-to-date runtime stats
        view_df = self._with_runtime_sample_statistics(self._all_datasets_df)

        logger.debug(f"Current view DataFrame has {len(view_df)} total samples")

        try:
            if request.start_index < 0 or request.records_cnt <= 0:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Invalid start_index or records_cnt",
                    data_records=[],
                )

            end_index = request.start_index + request.records_cnt
            df_slice = view_df.iloc[request.start_index:end_index]

            if df_slice.empty:
                return pb2.DataSamplesResponse(
                    success=False,
                    message=f"No samples found at index {request.start_index}",
                    data_records=[],
                )

            self._ctx.ensure_components()
            components = self._ctx.components

            train_loader = components.get("train_loader")
            test_loader = components.get("test_loader")

            data_records = []
            for _, row in df_slice.iterrows():
                origin = row.get("origin", "unknown")
                sample_id = int(row.get("sample_id", 0))

                if origin == "train":
                    dataset = getattr(train_loader, "dataset", train_loader) if train_loader else None
                elif origin == "eval":
                    dataset = getattr(test_loader, "dataset", test_loader) if test_loader else None
                else:
                    continue

                if dataset is None:
                    continue

                data_stats = []
                stats_to_retrieve = list(request.stats_to_retrieve)
                if not stats_to_retrieve:
                    stats_to_retrieve = [c for c in df_slice.columns if c != "sample_id"]
                
                # Always include 'image' if include_raw_data is true, even if not in DataFrame
                # (for lazy-loaded ImageFolder datasets)
                if request.include_raw_data and "image" not in stats_to_retrieve:
                    stats_to_retrieve.append("image")

                for stat_name in stats_to_retrieve:
                    stat = self._get_stat_from_row(row, stat_name)

                    # Special handling for 'image' - load on-demand if not in DataFrame
                    if stat_name == "image" and stat is None and dataset is not None:
                        try:
                            # Unwrap to get the raw dataset (similar to _dataset_to_df)
                            raw_ds = dataset
                            while True:
                                if hasattr(raw_ds, "wrapped_dataset"):
                                    new_ds = raw_ds.wrapped_dataset
                                    if new_ds is not None:
                                        raw_ds = new_ds
                                    else:
                                        break
                                elif hasattr(raw_ds, "dataset"):
                                    new_ds = raw_ds.dataset
                                    if new_ds is not None:
                                        raw_ds = new_ds
                                    else:
                                        break
                                else:
                                    break
                                if raw_ds is None:
                                    break
                            
                            # Load image on-demand from the raw dataset
                            if raw_ds is not None:
                                item = raw_ds[sample_id]
                                if isinstance(item, (tuple, list)):
                                    img = item[0]
                                else:
                                    img = item
                                
                                if hasattr(img, "numpy"):
                                    img_arr = img.numpy()
                                else:
                                    img_arr = np.array(img)
                                
                                stat = pb2.DataStat(
                                    name="image",
                                    type="array",
                                    shape=list(img_arr.shape),
                                    value=img_arr.flatten().astype(float).tolist(),
                                )
                        except Exception as e:
                            logger.warning(f"Could not load image on-demand for sample {sample_id} from {origin}: {e}")

                    if (
                        stat_name in ["tags", "deny_listed"]
                        and dataset is not None
                        and hasattr(dataset, "sample_statistics")
                    ):
                        try:
                            if stat_name in dataset.sample_statistics:
                                wrapper_value = dataset.sample_statistics[stat_name].get(sample_id)
                                if wrapper_value is not None:
                                    if stat_name == "tags" and wrapper_value != "":
                                        stat = pb2.DataStat(
                                            name=stat_name,
                                            type="string",
                                            shape=[1],
                                            value_string=wrapper_value,
                                        )
                                    elif stat_name == "deny_listed":
                                        stat = pb2.DataStat(
                                            name=stat_name,
                                            type="scalar",
                                            shape=[1],
                                            value=[float(wrapper_value)],
                                        )
                        except Exception as e:
                            logger.debug(f"Could not get {stat_name} from dataset wrapper: {e}")

                    if stat:
                        data_stats.append(stat)

                data_records.append(
                    pb2.DataRecord(
                        sample_id=sample_id,
                        data_stats=data_stats,
                    )
                )

            logger.debug(
                f"Successfully created {len(data_records)} data records from "
                f"{len(df_slice)} dataframe rows"
            )
            return pb2.DataSamplesResponse(
                success=True,
                message=f"Retrieved {len(data_records)} data records",
                data_records=data_records,
            )
        except Exception as e:
            logger.error(f"Failed to retrieve samples: {e}", exc_info=True)
            return pb2.DataSamplesResponse(
                success=False,
                message=f"Failed to retrieve samples: {str(e)}",
                data_records=[],
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
                dataset = getattr(train_loader, "dataset", train_loader) if train_loader else None
            elif origin == "eval":
                dataset = getattr(test_loader, "dataset", test_loader) if test_loader else None

            if dataset is None:
                continue

            try:
                if request.stat_name == "tags":
                    dataset.set(sid, "tags", request.string_value)
                elif request.stat_name == "deny_listed":
                    dataset.set(sid, "deny_listed", request.bool_value)
            except Exception as e:
                logger.warning(f"Could not edit sample {sid}: {e}")

        # ---------------------------------------------------------------------
        # 2) Mirror edits into the in-memory DataFrame (used by the UI / agent)
        # ---------------------------------------------------------------------
        if self._all_datasets_df is not None:
            # Update the DataFrame
            for sid, origin in zip(request.samples_ids, request.sample_origins):
                mask = (
                    (self._all_datasets_df["sample_id"] == sid)
                    & (self._all_datasets_df["origin"] == origin)
                )
                value = request.string_value if request.stat_name == "tags" else request.bool_value
                try:
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

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )
