import logging
import traceback
import numpy as np
import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2

logger = logging.getLogger(__name__)


class DataService:
    """
    Data service helpers + RPCs (for weights_studio UI).
    """

    def __init__(self, ctx):
        self._ctx = ctx
        # Data service components (initialized lazily on first use)
        self._all_datasets_df: pd.DataFrame | None = None
        self._agent = None

    # -------------------------------------------------------------------------
    # Data service helpers + RPCs (for weights_studio UI)
    # -------------------------------------------------------------------------
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

            def _dataset_to_df(dataset_or_loader, origin: str) -> pd.DataFrame:
                """Convert a dataset/loader into a DataFrame usable by the UI."""
                raw_ds = dataset_or_loader
                # logger.info(f"DEBUG: Unwrapping {type(raw_ds)} for {origin}")
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

                # Fast path for torchvision-style datasets with data/targets
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

                # Fallback: iterate samples
                if not records:
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

            logger.info(f"Created combined DataFrame with {len(self._all_datasets_df)} samples")
            logger.info(f"DataFrame columns: {list(self._all_datasets_df.columns)}")

            # Optional: external agent (weights_studio integration)
            try:
                import sys, os

                # path to trainer_services.py
                current_dir = os.path.dirname(os.path.abspath(__file__))

                repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))

                # weights_studio location: /Users/.../v0/weights_studio
                weights_studio_path = os.path.join(repo_root, "weights_studio")

                print(weights_studio_path)

                if os.path.isdir(weights_studio_path) and weights_studio_path not in sys.path:
                    sys.path.append(weights_studio_path)

                from agent import DataManipulationAgent
                import agent

                logger.info(f"DEBUG: agent module loaded from: {agent.__file__}")
                self._agent = DataManipulationAgent(self._all_datasets_df)
                logger.info("Data service initialized successfully with agent")

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

    def ApplyDataQuery(self, request, context):
        """Apply query to filter/sort/manipulate dataset."""
        if self._agent is None:
            self._initialize_data_service()
        else:
            self._refresh_data_stats()

        if request.query == "":
            if self._all_datasets_df is None:
                self._initialize_data_service()

            if self._all_datasets_df is None:
                return pb2.DataQueryResponse(
                    success=False,
                    message="Data service not available",
                )

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

        if not request.accumulate:
            self._initialize_data_service()

        if self._all_datasets_df is None:
            return pb2.DataQueryResponse(
                success=False,
                message="Data service not initialized",
            )

        try:
            if request.is_natural_language:
                if self._agent is None:
                    return pb2.DataQueryResponse(
                        success=False,
                        message="Natural language queries require Ollama agent (not available)",
                    )
                operation = self._agent.query(request.query)
                self._all_datasets_df = self._agent.apply_operation(self._all_datasets_df, operation)
                message = f"Applied operation: {operation['function']}"
            else:
                self._all_datasets_df = self._all_datasets_df.query(request.query)
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
        logger.info(f"GetDataSamples called: start_index={request.start_index}, records_cnt={request.records_cnt}")

        if self._all_datasets_df is None:
            logger.info("Initializing data service (first call)")
            self._initialize_data_service()

        if self._all_datasets_df is None:
            logger.error("Data service initialization failed - no dataframe available")
            return pb2.DataSamplesResponse(
                success=False,
                message="Data service not available",
                data_records=[],
            )

        logger.info(f"DataFrame has {len(self._all_datasets_df)} total samples")

        try:
            if request.start_index < 0 or request.records_cnt <= 0:
                return pb2.DataSamplesResponse(
                    success=False,
                    message="Invalid start_index or records_cnt",
                    data_records=[],
                )

            end_index = request.start_index + request.records_cnt
            df_slice = self._all_datasets_df.iloc[request.start_index:end_index]

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

                for stat_name in stats_to_retrieve:
                    stat = self._get_stat_from_row(row, stat_name)

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

            logger.info(f"Successfully created {len(data_records)} data records from {len(df_slice)} dataframe rows")
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
        self._ctx.ensure_components()
        components = self._ctx.components

        if request.stat_name not in ["tags", "deny_listed"]:
            return pb2.DataEditsResponse(
                success=False,
                message="Only 'tags' and 'deny_listed' stat editing is supported",
            )

        if request.type == pb2.SampleEditType.EDIT_ACCUMULATE:
            return pb2.DataEditsResponse(
                success=False,
                message="Accumulate tagging not supported",
            )

        train_loader = components.get("train_loader")
        test_loader = components.get("test_loader")

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

        if self._all_datasets_df is not None:
            for sid, origin in zip(request.samples_ids, request.sample_origins):
                mask = (self._all_datasets_df["sample_id"] == sid) & (
                    self._all_datasets_df["origin"] == origin
                )
                value = request.string_value if request.stat_name == "tags" else request.bool_value
                self._all_datasets_df.loc[mask, request.stat_name] = value

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )
