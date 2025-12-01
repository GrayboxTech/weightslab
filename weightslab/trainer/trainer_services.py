import io
import types
import time
import grpc
import torch
import logging
import traceback
import numpy as np
import pandas as pd  # <- needed for data service

import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from threading import Thread
from concurrent import futures

from weightslab.components.global_monitoring import weightslab_rlock, pause_controller
from weightslab.trainer.trainer_tools import *
from weightslab.trainer.trainer_tools import _get_input_tensor_for_sample
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType

# Global logger
logger = logging.getLogger(__name__)


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    """
    gRPC servicer for experiment-related services.

    This class provides implementations for streaming training status,
    handling experiment commands, retrieving samples, manipulating weights,
    getting weights and activations, and data-service operations used by
    the weights_studio UI.

    Components (model, dataloaders, optimizer, hyperparams, logger) are
    resolved from the WeightsLab GLOBAL_LEDGER as needed.
    """

    def __init__(self, exp_name: str = None):
        # Accept an explicit experiment name or resolve from the ledger.
        self._exp_name = exp_name
        # Components resolved from GLOBAL_LEDGER (model, dataloaders, optimizer,
        # hyperparams, logger). Populated lazily by _ensure_components().
        self._components = {}

        # Data service components (initialized lazily on first use)
        self._all_datasets_df: pd.DataFrame | None = None
        self._agent = None

    # -------------------------------------------------------------------------
    # Core ledger-backed components resolution
    # -------------------------------------------------------------------------
    def _ensure_components(self):
        """Ensure ledger-backed components are resolved and available on
        `self` (model, train/test dataloaders, optimizer, hyperparams,
        logger). Raises RuntimeError when mandatory components are missing.
        """
        if getattr(self, "_components", None) and self._components:
            return

        from weightslab.backend.ledgers import (
            get_hyperparams,
            list_hyperparams,
            get_model,
            list_models,
            get_dataloader,
            list_dataloaders,
            get_optimizer,
            list_optimizers,
            get_logger,
            list_loggers,
        )

        # resolve model
        model = None
        try:
            names = list_models()
            if self._exp_name and self._exp_name in names:
                model = get_model(self._exp_name)
            elif "experiment" in names:
                model = get_model("experiment")
            elif len(names) == 1:
                model = get_model()
        except Exception:
            model = None

        # resolve dataloaders (prefer explicit names 'train' / 'eval' / 'test' / 'train_loader' / 'test_loader')
        train_loader = None
        test_loader = None
        try:
            dnames = list_dataloaders()
            if "train" in dnames:
                train_loader = get_dataloader("train")
            elif "train_loader" in dnames:
                train_loader = get_dataloader("train_loader")
            elif len(dnames) == 1:
                train_loader = get_dataloader()

            if "eval" in dnames:
                test_loader = get_dataloader("eval")
            elif "test_loader" in dnames:
                test_loader = get_dataloader("test_loader")
            elif "test" in dnames:
                test_loader = get_dataloader("test")
            elif "test_loader" in dnames:
                test_loader = get_dataloader("test_loader")
            elif len(dnames) == 1 and train_loader is not None:
                test_loader = train_loader
        except Exception:
            train_loader = None
            test_loader = None

        # resolve optimizer
        optimizer = None
        try:
            onames = list_optimizers()
            if len(onames) == 1:
                optimizer = get_optimizer()
            elif "_optimizer" in onames:
                optimizer = get_optimizer("_optimizer")
        except Exception:
            optimizer = None

        # resolve hyperparams (by exp_name or single set)
        hyperparams = None
        try:
            hp_names = list_hyperparams()
            if self._exp_name and self._exp_name in hp_names:
                hyperparams = get_hyperparams(self._exp_name)
            elif len(hp_names) == 1:
                hyperparams = get_hyperparams()
        except Exception:
            hyperparams = None

        # resolve logger
        signal_logger = None
        try:
            lnames = list_loggers()
            if len(lnames) == 1:
                signal_logger = get_logger()
            elif "main" in lnames:
                signal_logger = get_logger("main")
        except Exception:
            signal_logger = None

        self._components = {
            "model": model,
            "train_loader": train_loader,
            "test_loader": test_loader,
            "optimizer": optimizer,
            "hyperparams": hyperparams,
            "signal_logger": signal_logger,
        }

        # Build hyper-parameter descriptors used by the protocol. Use
        # ledger-backed hyperparams when available, with safe fallbacks.
        def _hp_getter(key, default=None):
            def _g():
                try:
                    hp = self._components.get("hyperparams")
                    if "." in key:
                        parts = key.split(".") if key else []
                        cur = hp
                        for p in parts:
                            cur = cur[p]
                        return cur
                    if isinstance(hp, dict):
                        return hp.get(key, default)
                    elif hasattr(hp, "get"):
                        return hp.get(key, default)
                except Exception:
                    pass
                return default

            return _g

        self.hyper_parameters = {
            ("Experiment Name", "experiment_name", "text", lambda: _hp_getter("experiment_name", "Anonymous")()),
            ("Left Training Steps", "training_left", "number", _hp_getter("training_steps_to_do", 999)),
            ("Eval Frequency", "eval_frequency", "number", _hp_getter("eval_full_to_train_steps_ratio", 100)),
            ("Checkpoint Frequency", "checkpooint_frequency", "number", _hp_getter("experiment_dump_to_train_steps_ratio", 100)),
            ("Learning Rate", "learning_rate", "number", _hp_getter("optimizer.lr", 1e-4)),
            ("Batch Size", "batch_size", "number", _hp_getter("data.train_loader.batch_size", 8)),
        }

    # -------------------------------------------------------------------------
    # Training status stream
    # -------------------------------------------------------------------------
    def StreamStatus(self, request_iterator, context):
        logger.debug(f"ExperimentServiceServicer.StreamStatus({request_iterator})")

        self._ensure_components()

        while True:
            signal_logger = self._components.get("signal_logger") if getattr(self, "_components", None) else None
            if signal_logger is None or not hasattr(signal_logger, "queue"):
                raise RuntimeError("No logger with a queue registered in GLOBAL_LEDGER")

            signal_log = signal_logger.queue.get()

            if "metric_name" in signal_log and "acc" in signal_log["metric_name"]:
                logger.debug(f"[signal_log] {signal_log['metric_name']} = {signal_log['metric_value']:.2f}")

            if signal_log is None:
                break

            metrics_status, annotat_status = None, None
            if "metric_name" in signal_log:
                metrics_status = pb2.MetricsStatus(
                    name=signal_log["metric_name"],
                    value=signal_log["metric_value"],
                )
            elif "annotation" in signal_log:
                annotat_status = pb2.AnnotatStatus(name=signal_log["annotation"])
                for key, value in signal_log["metadata"].items():
                    annotat_status.metadata[key] = value

            training_status = pb2.TrainingStatusEx(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                experiment_name=signal_log["experiment_name"],
                model_age=signal_log["model_age"],
            )

            if metrics_status:
                training_status.metrics_status.CopyFrom(metrics_status)
            if annotat_status:
                training_status.annotat_status.CopyFrom(annotat_status)

            # mark task done on ledger logger queue
            signal_logger = self._components.get("signal_logger")
            if signal_logger is not None and hasattr(signal_logger, "queue"):
                try:
                    signal_logger.queue.task_done()
                except Exception:
                    pass

            yield training_status

    # -------------------------------------------------------------------------
    # Sample retrieval (images / segmentation / recon)
    # -------------------------------------------------------------------------
    def GetSamples(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetSamples({request})")

        import concurrent.futures

        self._ensure_components()

        ds = self._components.get("train_loader") if request.origin == "train" else self._components.get("test_loader")
        dataset = getattr(ds, "tracked_dataset", ds)
        response = pb2.BatchSampleResponse()

        do_resize = request.HasField("resize_width") and request.HasField("resize_height")
        resize_dims = (request.resize_width, request.resize_height) if do_resize else None
        task_type = getattr(
            dataset,
            "task_type",
            getattr(self._components.get("model"), "task_type", "classification"),
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            fut_map = {
                executor.submit(
                    process_sample,
                    sid,
                    dataset,
                    do_resize,
                    resize_dims,
                    types.SimpleNamespace(
                        **{
                            "tasks": getattr(self._components.get("model"), "tasks", None),
                            "task_type": getattr(
                                self._components.get("model"),
                                "task_type",
                                getattr(dataset, "task_type", "classification"),
                            ),
                            "num_classes": getattr(self._components.get("model"), "num_classes", None),
                        }
                    ),
                ): sid
                for sid in request.sample_ids
            }
            results = {}
            for future in concurrent.futures.as_completed(fut_map):
                sid, transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes = future.result()
                results[sid] = (transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes)

        # build response preserving input order
        for sid in request.sample_ids:
            transformed_bytes, raw_bytes, cls_label, mask_bytes, pred_bytes = results.get(
                sid, (None, None, -1, b"", b"")
            )
            if transformed_bytes is None or raw_bytes is None:
                continue

            if task_type == "segmentation":
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=cls_label,
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=mask_bytes,
                    prediction=pred_bytes or b"",
                )
            elif pred_bytes and len(pred_bytes) > 0:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=-1,
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=b"",
                    prediction=pred_bytes,
                )
            else:
                sample_response = pb2.SampleRequestResponse(
                    sample_id=sid,
                    label=cls_label,
                    data=transformed_bytes,
                    raw_data=raw_bytes,
                    mask=b"",
                    prediction=b"",
                )
            response.samples.append(sample_response)

        return response

    # -------------------------------------------------------------------------
    # Weights inspection
    # -------------------------------------------------------------------------
    def GetWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetWeights({request})")

        self._ensure_components()

        answer = pb2.WeightsResponse(success=True, error_message="")

        neuron_id = request.neuron_id
        layer = None

        try:
            model = self._components.get("model")
            if model is None:
                answer.success = False
                answer.error_messages = "No model registered"
                return answer
            layer = model.get_layer_by_id(neuron_id.layer_id)
        except Exception as e:
            answer.success = False
            answer.error_messages = str(e)
            return answer

        answer.neuron_id.CopyFrom(request.neuron_id)
        answer.layer_name = layer.__class__.__name__
        answer.incoming = layer.in_neurons
        answer.outgoing = layer.out_neurons
        if "Conv2d" in layer.__class__.__name__:
            answer.layer_type = "Conv2d"
            answer.kernel_size = layer.kernel_size[0]
        elif "Linear" in layer.__class__.__name__:
            answer.layer_type = "Linear"

        if neuron_id.neuron_id >= layer.out_neurons:
            answer.success = False
            answer.error_messages = f"Neuron {neuron_id.neuron_id} outside bounds."
            return answer

        if neuron_id.neuron_id < 0:
            weights = layer.weight.data.cpu().detach().numpy().flatten()
        else:
            weights = layer.weight[neuron_id.neuron_id].data.cpu().detach().numpy().flatten()
        answer.weights.extend(weights)

        return answer

    # -------------------------------------------------------------------------
    # Activations
    # -------------------------------------------------------------------------
    def GetActivations(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetActivations({request})")

        self._ensure_components()

        empty_resp = pb2.ActivationResponse(layer_type="", neurons_count=0)

        try:
            model = self._components.get("model")
            if model is None:
                return empty_resp
            last_layer = model.layers[-1]
            last_layer_id = int(last_layer.get_module_id())
            if int(request.layer_id) == last_layer_id:
                return empty_resp

            ds = getattr(self._components.get("train_loader"), "tracked_dataset", self._components.get("train_loader"))
            if request.origin == "eval":
                ds = getattr(self._components.get("test_loader"), "tracked_dataset", self._components.get("test_loader"))

            if request.sample_id < 0 or request.sample_id >= len(ds):
                raise ValueError(f"No sample id {request.sample_id} for {request.origin}")

            x = _get_input_tensor_for_sample(ds, request.sample_id, getattr(model, "device", "cpu"))

            with torch.no_grad():
                intermediaries = {}
                handles = []

                try:
                    def make_hook(module):
                        def hook(mod, inp, out):
                            try:
                                mid = None
                                if hasattr(mod, "get_module_id"):
                                    try:
                                        mid = mod.get_module_id()
                                    except Exception:
                                        mid = None
                                if mid is None:
                                    return
                                key = mid
                                try:
                                    intermediaries[key] = out.detach().cpu()
                                except Exception:
                                    try:
                                        intermediaries[key] = out[0].detach().cpu()
                                    except Exception:
                                        intermediaries[key] = None
                            except Exception:
                                pass

                        return hook

                    for layer in model.layers:
                        try:
                            h = layer.register_forward_hook(make_hook(layer))
                            handles.append(h)
                        except Exception:
                            pass

                    try:
                        _ = model(x)
                    except Exception:
                        pass
                finally:
                    for h in handles:
                        try:
                            h.remove()
                        except Exception:
                            pass

            if intermediaries[request.layer_id] is None:
                raise ValueError(f"No intermediary layer {request.layer_id}")

            layer = model.get_layer_by_id(request.layer_id)
            layer_type = layer.__class__.__name__
            amap = intermediaries[request.layer_id].squeeze(0).detach().cpu().numpy()
            resp = pb2.ActivationResponse(layer_type=layer_type)

            C, H, W = 1, 1, 1
            if amap.ndim == 3:
                C, H, W = amap.shape
            elif amap.ndim == 1:
                C = amap.shape[0]

            resp.neurons_count = C
            for c in range(C):
                vals = amap[c].astype(np.float32).reshape(-1).tolist()
                if not isinstance(vals, list):
                    vals = [vals]
                resp.activations.append(
                    pb2.ActivationMap(neuron_id=c, values=vals, H=H, W=W)
                )
            return resp
        except (ValueError, Exception) as e:
            logger.error(f"Error in GetActivations: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

        return empty_resp

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
            self._ensure_components()

            train_loader = self._components.get("train_loader")
            test_loader = self._components.get("test_loader")

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

                # repo root: /Users/.../v0
                repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

                # weights_studio location: /Users/.../v0/weights_studio
                weights_studio_path = os.path.join(repo_root, "weights_studio")

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

            train_stats = _get_stats(self._components.get("train_loader"), "train")
            if train_stats is not None:
                dfs.append(train_stats)

            eval_stats = _get_stats(self._components.get("test_loader"), "eval")
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

            self._ensure_components()
            train_loader = self._components.get("train_loader")
            test_loader = self._components.get("test_loader")

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
        self._ensure_components()

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

        train_loader = self._components.get("train_loader")
        test_loader = self._components.get("test_loader")

        # --- update underlying tracking wrapper ---
        for sid, origin in zip(request.samples_ids, request.sample_origins):
            ds = None
            if origin == "train":
                ds = train_loader
            elif origin == "eval":
                ds = test_loader

            if ds is None:
                continue

            # Prefer the tracking wrapper if present
            dataset = getattr(ds, "tracked_dataset", ds)

            try:
                if request.stat_name == "tags":
                    dataset.set(int(sid), "tags", request.string_value)
                elif request.stat_name == "deny_listed":
                    dataset.set(int(sid), "deny_listed", request.bool_value)
            except Exception as e:
                logger.warning(f"Could not edit sample {sid}: {e}")

        # --- update cached dataframe, if already built ---
        if self._all_datasets_df is not None:
            for sid, origin in zip(request.samples_ids, request.sample_origins):
                mask = (
                    (self._all_datasets_df["sample_id"] == int(sid)) &
                    (self._all_datasets_df["origin"] == origin)
                )
                value = (
                    request.string_value
                    if request.stat_name == "tags"
                    else request.bool_value
                )
                self._all_datasets_df.loc[mask, request.stat_name] = value

        return pb2.DataEditsResponse(
            success=True,
            message=f"Edited {len(request.samples_ids)} samples",
        )

    # -------------------------------------------------------------------------
    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def ExperimentCommand(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ExperimentCommand({request})")

        self._ensure_components()

        # Write requests
        if request.HasField("hyper_parameter_change"):
            with weightslab_rlock:
                hyper_parameters = request.hyper_parameter_change.hyper_parameters
                from weightslab.backend.ledgers import set_hyperparam, list_hyperparams

                hp_name = None
                if self._exp_name:
                    hp_name = self._exp_name
                else:
                    hps = list_hyperparams()
                    if len(hps) == 1:
                        hp_name = hps[0]

                if hp_name is None:
                    return pb2.CommandResponse(success=False, message="Cannot resolve hyperparams name")

                try:
                    if hyper_parameters.HasField("is_training"):
                        if hyper_parameters.is_training:
                            pause_controller.resume()
                        else:
                            pause_controller.pause()
                        set_hyperparam(hp_name, "is_training", hyper_parameters.is_training)

                    if hyper_parameters.HasField("training_steps_to_do"):
                        set_hyperparam(
                            hp_name,
                            "training_steps_to_do",
                            hyper_parameters.training_steps_to_do,
                        )

                    if hyper_parameters.HasField("learning_rate"):
                        set_hyperparam(hp_name, "optimizer.lr", hyper_parameters.learning_rate)

                    if hyper_parameters.HasField("batch_size"):
                        set_hyperparam(
                            hp_name,
                            "data.train_loader.batch_size",
                            hyper_parameters.batch_size,
                        )

                    if hyper_parameters.HasField("full_eval_frequency"):
                        set_hyperparam(
                            hp_name,
                            "eval_full_to_train_steps_ratio",
                            hyper_parameters.full_eval_frequency,
                        )

                    if hyper_parameters.HasField("checkpont_frequency"):
                        set_hyperparam(
                            hp_name,
                            "experiment_dump_to_train_steps_ratio",
                            hyper_parameters.checkpont_frequency,
                        )

                except Exception as e:
                    return pb2.CommandResponse(
                        success=False,
                        message=f"Failed to set hyperparameters: {e}",
                    )

                return pb2.CommandResponse(success=True, message="Hyper parameter changed")

        if request.HasField("deny_samples_operation"):
            with weightslab_rlock:
                denied_cnt = len(request.deny_samples_operation.sample_ids)
                ds = self._components.get("train_loader")
                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message="No train dataloader registered",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.denylist_samples(
                    set(request.deny_samples_operation.sample_ids),
                    accumulate=request.deny_samples_operation.accumulate,
                )
                return pb2.CommandResponse(
                    success=True,
                    message=f"Denied {denied_cnt} train samples",
                )

        if request.HasField("deny_eval_samples_operation"):
            with weightslab_rlock:
                denied_cnt = len(request.deny_eval_samples_operation.sample_ids)
                ds = self._components.get("test_loader")
                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message="No eval dataloader registered",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.denylist_samples(
                    set(request.deny_eval_samples_operation.sample_ids),
                    accumulate=request.deny_eval_samples_operation.accumulate,
                )
            return pb2.CommandResponse(
                success=True,
                message=f"Denied {denied_cnt} eval samples",
            )

        if request.HasField("remove_from_denylist_operation"):
            with weightslab_rlock:
                allowed = set(request.remove_from_denylist_operation.sample_ids)
                ds = self._components.get("train_loader")
                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message="No train dataloader registered",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.allowlist_samples(allowed)
                return pb2.CommandResponse(
                    success=True,
                    message=f"Un-denied {len(allowed)} train samples",
                )

        if request.HasField("remove_eval_from_denylist_operation"):
            with weightslab_rlock:
                allowed = set(request.remove_eval_from_denylist_operation.sample_ids)
                ds = self._components.get("test_loader")
                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message="No eval dataloader registered",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.allowlist_samples(allowed)
                return pb2.CommandResponse(
                    success=True,
                    message=f"Un-denied {len(allowed)} eval samples",
                )

        if request.HasField("load_checkpoint_operation"):
            with weightslab_rlock:
                checkpoint_id = request.load_checkpoint_operation.checkpoint_id
                model = self._components.get("model")
                if model is None:
                    return pb2.CommandResponse(
                        success=False,
                        message="No model registered to load checkpoint",
                    )
                if hasattr(model, "load"):
                    try:
                        model.load(checkpoint_id)
                    except Exception as e:
                        return pb2.CommandResponse(
                            success=False,
                            message=str(e),
                        )

        # Read requests
        response = pb2.CommandResponse(success=True, message="")
        if request.get_hyper_parameters:
            response.hyper_parameters_descs.extend(
                get_hyper_parameters_pb(self.hyper_parameters)
            )

        if request.get_interactive_layers:
            model = self._components.get("model")
            if model is not None:
                if request.HasField("get_single_layer_info_id"):
                    response.layer_representations.extend(
                        [
                            get_layer_representation(
                                model.get_layer_by_id(request.get_single_layer_info_id)
                            )
                        ]
                    )
                else:
                    response.layer_representations.extend(
                        get_layer_representations(model)
                    )

        if request.get_data_records:
            if request.get_data_records == "train":
                ds = self._components.get("train_loader")
                if ds is not None:
                    dataset = getattr(ds, "tracked_dataset", ds)
                    response.sample_statistics.CopyFrom(
                        get_data_set_representation(
                            dataset,
                            types.SimpleNamespace(
                                **{
                                    "tasks": getattr(self._components.get("model"), "tasks", None),
                                    "task_type": getattr(
                                        self._components.get("model"),
                                        "task_type",
                                        getattr(dataset, "task_type", "classification"),
                                    ),
                                    "num_classes": getattr(
                                        self._components.get("model"), "num_classes", None
                                    ),
                                }
                            ),
                        )
                    )
                    response.sample_statistics.origin = "train"
            elif request.get_data_records == "eval":
                ds = self._components.get("test_loader")
                if ds is not None:
                    dataset = getattr(ds, "tracked_dataset", ds)
                    response.sample_statistics.CopyFrom(
                        get_data_set_representation(
                            dataset,
                            types.SimpleNamespace(
                                **{
                                    "tasks": getattr(self._components.get("model"), "tasks", None),
                                    "task_type": getattr(
                                        self._components.get("model"),
                                        "task_type",
                                        getattr(dataset, "task_type", "classification"),
                                    ),
                                    "num_classes": getattr(
                                        self._components.get("model"), "num_classes", None
                                    ),
                                }
                            ),
                        )
                    )
                    response.sample_statistics.origin = "eval"

        return response

    # -------------------------------------------------------------------------
    # Weight manipulation (architecture operations)
    # -------------------------------------------------------------------------
    def ManipulateWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ManipulateWeights({request})")

        self._ensure_components()

        answer = pb2.WeightsOperationResponse(success=False, message="Unknown error")
        weight_operations = request.weight_operation

        if weight_operations.op_type == pb2.WeightOperationType.ADD_NEURONS:
            op_type = ArchitectureNeuronsOpType.ADD
        elif weight_operations.op_type == pb2.WeightOperationType.REMOVE_NEURONS:
            op_type = ArchitectureNeuronsOpType.PRUNE
        elif weight_operations.op_type == pb2.WeightOperationType.FREEZE:
            op_type = ArchitectureNeuronsOpType.FREEZE
        elif weight_operations.op_type == pb2.WeightOperationType.REINITIALIZE:
            op_type = ArchitectureNeuronsOpType.RESET
        else:
            op_type = None

        model = self._components.get("model")

        if model is None or op_type is None:
            return pb2.WeightsOperationResponse(
                success=False,
                message="Model not found or invalid op_type",
            )

        if len(weight_operations.neuron_ids) == 0:
            layer_id = weight_operations.layer_id
            neuron_id = []

            with weightslab_rlock:
                model.apply_architecture_op(
                    op_type=op_type,
                    layer_id=layer_id,
                    neuron_indices=neuron_id,
                )

        else:
            for neuron_details in weight_operations.neuron_ids:
                layer_id = neuron_details.layer_id
                neuron_id = neuron_details.neuron_id

                with weightslab_rlock:
                    model.apply_architecture_op(
                        op_type=op_type,
                        layer_id=layer_id,
                        neuron_indices=neuron_id,
                    )
        

        answer = pb2.WeightsOperationResponse(
            success=True,
            message=f"{weight_operations.op_type} - {weight_operations.neuron_ids}",
        )

        return answer


# -----------------------------------------------------------------------------
# Serving gRPC communication
# -----------------------------------------------------------------------------
def grpc_serve(n_workers_grpc: int = 6, port_grpc: int = 50051, **_):
    """Configure trainer services such as gRPC server.

    Args:
        n_workers_grpc (int): Number of threads for the gRPC server.
        port_grpc (int): Port number for the gRPC server.
    """
    import weightslab.trainer.trainer_services as trainer
    from weightslab.trainer.trainer_tools import force_kill_all_python_processes

    def serving_thread_callback():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=n_workers_grpc))
        servicer = trainer.ExperimentServiceServicer()
        pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"[::]:{port_grpc}")
        try:
            server.start()
            logger.info("gRPC Server started on port %d. Press Ctrl+C to stop.", port_grpc)
            server.wait_for_termination()
        except KeyboardInterrupt:
            force_kill_all_python_processes()

    training_thread = Thread(target=serving_thread_callback)
    training_thread.start()


if __name__ == "__main__":
    grpc_serve()
