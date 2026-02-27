import time
import types
import logging

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.components.global_monitoring import weightslab_rlock
from weightslab.trainer.trainer_tools import get_hyper_parameters_pb, get_layer_representation, get_layer_representations, get_data_set_representation
from weightslab.backend.ledgers import set_hyperparam, list_hyperparams, resolve_hp_name, get_hyperparams
from weightslab.backend import ledgers
from weightslab.trainer.services.model_service import ModelService
from weightslab.trainer.services.data_service import DataService


# Logger
logger = logging.getLogger(__name__)


class ExperimentService:
    """
    Domain-level experiment service that orchestrates model/data services
    and handles general experiment-related commands.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self.model_service = ModelService(ctx)
        self.data_service = DataService(ctx)

    # -------------------------------------------------------------------------
    # Logger queue sync for WeightsStudio
    # -------------------------------------------------------------------------
    def GetLatestLoggerData(self, request, context):
        """
        Returns logger data for WeightsStudio polling.
        - If request_full_history is True: returns full history (limited by max_points per signal)
        - If request_full_history is False: returns only new data from the queue since last request
        - If break_by_slices is True: returns per-sample data filtered by tags
        """
        self._ctx.ensure_components()
        components = self._ctx.components
        signal_logger = components.get("signal_logger")
        if signal_logger ==  None:
            return pb2.GetLatestLoggerDataResponse(points=[])

        points = []

        # Handle break_by_slices mode
        if request.break_by_slices:
            tags = list(request.tags) if request.tags else []
            graph_name = request.graph_name if hasattr(request, 'graph_name') and request.graph_name else None

            # Get sample IDs that match the given tags
            sample_ids = set()
            if tags:
                # Get dataframe manager to query samples by tags
                df_manager = components.get("df_manager")
                if df_manager:
                    df = df_manager.get_df_view()
                    # Filter by tags: samples should have ALL specified tags
                    mask = None
                    for tag in tags:
                        tag_col = f"tag:{tag}"
                        if tag_col in df.columns:
                            if mask is None:
                                mask = df[tag_col] == True
                            else:
                                mask = mask & (df[tag_col] == True)

                    if mask is not None:
                        filtered_df = df[mask]
                        sample_ids = set(filtered_df.index.tolist())

            # Get per-sample history from signal_logger
            history_per_sample = signal_logger.get_signal_history_per_sample()

            # Collect all points for matching samples, filtered by graph_name if specified
            if graph_name not in history_per_sample:
                return pb2.GetLatestLoggerDataResponse(points=[])  # No data for this graph_name

            # Collect points for the specified graph_name and sample_ids
            sample_data_by_hash = history_per_sample[graph_name]
            if sample_ids:
                # Filter by sample_ids if tags were specified
                for sid in sample_ids:
                    for _, signals in sample_data_by_hash.items():
                        for data in signals:
                            if data["sample_id"] == str(sid):
                                points.append(
                                    pb2.LoggerDataPoint(
                                        metric_name=graph_name,
                                        model_age=data.get("model_age", 0),
                                        metric_value=data.get("metric_value", 0.0),
                                        experiment_hash=data.get("experiment_hash", "N.A."),
                                        timestamp=int(data.get("timestamp", time.time())),
                                        sample_id=str(sid)
                                    )
                                )

            return pb2.GetLatestLoggerDataResponse(points=points)

        # Original behavior for non-break_by_slices mode
        if request.request_full_history:
            # Return full history
            max_points = request.max_points or 10000
            history = signal_logger.get_signal_history()

            # Group by metric_name and limit each
            signal_groups = {}
            for s in history:
                metric_name = s.get("metric_name", "")
                if metric_name not in signal_groups:
                    signal_groups[metric_name] = []
                signal_groups[metric_name].append(s)

            # Take last max_points_per_signal for each signal and downsample if needed
            for metric_name, signal_history in signal_groups.items():

                # Downsample if we have more than 1000 points
                if len(signal_history) > max_points:
                    # Calculate step to downsample (e.g., if 5000 points, step=5 to get ~1000)
                    step = max(1, len(signal_history) // max_points)
                    signal_history = signal_history[::step]

                for s in signal_history:
                    points.append(
                        pb2.LoggerDataPoint(
                            metric_name=metric_name,
                            model_age=s.get("model_age", 0),
                            metric_value=s.get("metric_value", 0.0),
                            experiment_hash=s.get("experiment_hash", "N.A."),
                            timestamp=int(s.get("timestamp", time.time())),
                            sample_id=""  # No sample_id in aggregated mode
                        )
                    )
        else:
            # Return only queue (new data since last poll)
            queue_data = signal_logger.get_and_clear_queue()
            for s in queue_data:
                points.append(
                    pb2.LoggerDataPoint(
                        metric_name=s.get("metric_name", ""),
                        model_age=s.get("model_age", 0),
                        metric_value=s.get("metric_value", 0.0),
                        experiment_hash=s.get("experiment_hash", "N.A."),
                        timestamp=int(s.get("timestamp", time.time())),
                        sample_id=""  # No sample_id in queue mode
                    )
                )

        return pb2.GetLatestLoggerDataResponse(points=points)

    def RestoreCheckpoint(self, request, context):
        """
        Restore a checkpoint from a given experiment hash.
        - Pauses training if not already paused
        - Calls checkpoint manager to load the state
        - Returns success flag and message
        """
        try:
            raw_experiment_hash = request.experiment_hash
            experiment_hash = raw_experiment_hash
            target_step = None
            load_weights_only = False

            if "@@weights_step=" in raw_experiment_hash:
                base_hash, payload = raw_experiment_hash.split("@@weights_step=", 1)
                experiment_hash = base_hash
                try:
                    target_step = int(payload.strip())
                    load_weights_only = True
                except Exception:
                    target_step = None
                    load_weights_only = False

            logger.info(
                f"Restoring checkpoint from hash: {experiment_hash}"
                + (f" (weights-only, target_step={target_step})" if load_weights_only and target_step is not None else "")
            )

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

            # Get checkpoint manager and load state
            checkpoint_manager = components.get("checkpoint_manager")
            if checkpoint_manager == None:
                checkpoint_manager = ledgers.get_checkpoint_manager()
                if checkpoint_manager == None:
                    return pb2.RestoreCheckpointResponse(
                        success=False,
                        message="Checkpoint manager not initialized"
                    )

            # Load checkpoint by hash
            if load_weights_only and target_step is not None:
                success = checkpoint_manager.load_state(
                    experiment_hash,
                    load_model=True,
                    load_weights=True,
                    load_config=True,
                    load_data=True,
                    target_step=target_step,
                )
            else:
                success = checkpoint_manager.load_state(experiment_hash)

            # Reply
            if success:
                logger.info(f"Successfully restored checkpoint: {experiment_hash}")
                return pb2.RestoreCheckpointResponse(
                    success=True,
                    message=(
                        f"Weights restored from checkpoint {experiment_hash}"
                        if load_weights_only and target_step is not None
                        else f"Checkpoint {experiment_hash} restored successfully"
                    )
                )
            else:
                logger.warning(f"Failed to restore checkpoint: {experiment_hash}")
                return pb2.RestoreCheckpointResponse(
                    success=False,
                    message=f"Failed to restore checkpoint {experiment_hash}"
                )
        except Exception as e:
            logger.error(f"Error during checkpoint restore: {str(e)}")
            return pb2.RestoreCheckpointResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def ExperimentCommand(self, request, context):
        self._ctx.ensure_components()
        components = self._ctx.components

        # Write requests
        if request.HasField("hyper_parameter_change"):
            hyper_parameters = request.hyper_parameter_change.hyper_parameters
            hp_name = None
            if self._ctx.exp_name:
                hp_name = self._ctx.exp_name
            else:
                hp_name = resolve_hp_name()

            if hp_name is None:
                hps = list_hyperparams()
                detailed_msg = f"Cannot find an active hyperparameter set (LEDGER_HPS={hps}, CTX_EXP={self._ctx.exp_name})"
                logger.error(detailed_msg)
                return pb2.CommandResponse(success=False, message=detailed_msg)

            try:
                if hyper_parameters.HasField("training_steps_to_do"):
                    set_hyperparam(
                        name=hp_name,
                        key_path="training_steps_to_do",
                        value=hyper_parameters.training_steps_to_do
                    )

                if hyper_parameters.HasField("learning_rate"):
                    set_hyperparam(
                        name=hp_name,
                        key_path="optimizer.lr",
                        value=hyper_parameters.learning_rate
                    )
                if hyper_parameters.HasField("batch_size"):
                    set_hyperparam(
                        name=hp_name,
                        key_path="data.train_loader.batch_size",
                        value=hyper_parameters.batch_size
                    )

                if hyper_parameters.HasField("full_eval_frequency"):
                    set_hyperparam(
                        name=hp_name,
                        key_path="eval_full_to_train_steps_ratio",
                        value=hyper_parameters.full_eval_frequency
                    )

                if hyper_parameters.HasField("checkpont_frequency"):
                    set_hyperparam(
                        name=hp_name,
                        key_path="experiment_dump_to_train_steps_ratio",
                        value=hyper_parameters.checkpont_frequency
                    )

                # Process auditor_mode FIRST so mode is set before we resume
                try:
                    if hyper_parameters.HasField("auditor_mode"):
                        incoming_audit = bool(hyper_parameters.auditor_mode)

                        # Read the CURRENT stored value to detect a real change
                        hp_now = get_hyperparams(hp_name)
                        current_audit = bool(hp_now.get("auditor_mode", False)) if hp_now else False

                        if incoming_audit != current_audit:
                            # Mode actually changed — pause and announce switch
                            trainer = components.get("trainer")
                            if trainer:
                                print(f"\n[WeightsLab] Pausing to switch mode...", flush=True)
                                trainer.pause()
                                set_hyperparam(name=hp_name, key_path="is_training", value=False)
                            mode_label = "AUDIT" if incoming_audit else "TRAIN"
                            print(f"\n[WeightsLab] UI Command: Switch to {mode_label} Mode", flush=True)

                        # Always update the stored value (harmless no-op if unchanged)
                        set_hyperparam(name=hp_name, key_path="auditor_mode", value=incoming_audit)
                except ValueError:
                    pass

                # Process is_training AFTER mode is set — so Resume fires with correct mode
                if hyper_parameters.HasField("is_training"):
                    trainer = components.get("trainer")
                    if trainer is not None:
                        if hyper_parameters.is_training:
                            print("\n[WeightsLab] UI Command: RESUME", flush=True)
                            trainer.resume()
                        else:
                            print("\n[WeightsLab] UI Command: PAUSE", flush=True)
                            trainer.pause()
                    set_hyperparam(
                        name=hp_name,
                        key_path="is_training",
                        value=hyper_parameters.is_training
                    )

            except Exception as e:
                return pb2.CommandResponse(
                    success=False,
                    message=f"Failed to set hyperparameters: {e}",
                )

            return pb2.CommandResponse(success=True, message="Hyper parameter changed")

            if request.HasField("load_checkpoint_operation"):
                with weightslab_rlock:
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

                checkpoint_id = request.load_checkpoint_operation.checkpoint_id
                model = components.get("model")
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
                get_hyper_parameters_pb(self._ctx.hyper_parameters)
            )

        if request.get_interactive_layers:
            model = components.get("model")
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
            from weightslab.backend.ledgers import get_dataloaders

            # Find the loader for the requested origin dynamically
            loader_names = get_dataloaders()
            ds = None
            for loader_name in loader_names:
                loader = components.get(loader_name)
                if loader is None:
                    continue
                tracked_ds = getattr(loader, "tracked_dataset", None)
                if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                    if tracked_ds._dataset_split == request.get_data_records:
                        ds = loader
                        break

            if ds is not None:
                dataset = getattr(ds, "tracked_dataset", ds)
                response.sample_statistics.CopyFrom(
                    get_data_set_representation(
                        dataset,
                        types.SimpleNamespace(
                            **{
                                "tasks": getattr(components.get("model"), "tasks", None),
                                "task_type": getattr(
                                    components.get("model"),
                                    "task_type",
                                    getattr(dataset, "task_type", "classification"),
                                ),
                                "num_classes": getattr(
                                    components.get("model"), "num_classes", None
                                ),
                            }
                        ),
                    )
                )
                response.sample_statistics.origin = request.get_data_records

        return response
