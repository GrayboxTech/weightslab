import time
import logging
import types

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.components.global_monitoring import weightslab_rlock
from weightslab.trainer.trainer_tools import get_hyper_parameters_pb, get_layer_representation, get_layer_representations, get_data_set_representation
from weightslab.trainer.services.model_service import ModelService
from weightslab.trainer.services.data_service import DataService

logger = logging.getLogger(__name__)


class ExperimentService:
    """
    Domain-level experiment service that orchestrates model/data services
    and handles training-related commands.
    """

    def __init__(self, ctx):
        self._ctx = ctx
        self.model_service = ModelService(ctx)
        self.data_service = DataService(ctx)

    def get_root_log_dir(self) -> str:
        """Get the root log directory.

        Returns:
            Absolute path to root_log_dir
        """
        return self.data_service.get_root_log_dir()

    # -------------------------------------------------------------------------
    # Training status stream
    # -------------------------------------------------------------------------
    def stream_status(self, request_iterator):
        self._ctx.ensure_components()
        components = self._ctx.components

        while True:
            signal_logger = components.get("signal_logger") if getattr(self._ctx, "_components", None) else None

            if signal_logger != None:
                signal_log = signal_logger.queue.get()

                if "metric_name" in signal_log and "acc" in signal_log["metric_name"]:
                    logger.debug(f"[signal_log] {signal_log['metric_name']} = {signal_log['metric_value']:.2f}")

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
                try:
                    signal_logger.queue.task_done()
                except Exception:
                    pass

                yield training_status

    # -------------------------------------------------------------------------
    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def ExperimentCommand(self, request, context):
        self._ctx.ensure_components()
        components = self._ctx.components

        # Write requests
        if request.HasField("hyper_parameter_change"):
            with weightslab_rlock:
                hyper_parameters = request.hyper_parameter_change.hyper_parameters
                from weightslab.backend.ledgers import set_hyperparam, list_hyperparams, resolve_hp_name
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

                    if hyper_parameters.HasField("is_training"):
                        set_hyperparam(hp_name, "is_training", hyper_parameters.is_training)

                except Exception as e:
                    return pb2.CommandResponse(
                        success=False,
                        message=f"Failed to set hyperparameters: {e}",
                    )

                return pb2.CommandResponse(success=True, message="Hyper parameter changed")

        if request.HasField("deny_samples_operation"):
            with weightslab_rlock:
                from weightslab.backend.ledgers import get_dataloaders

                denied_cnt = len(request.deny_samples_operation.sample_ids)
                origin = request.deny_samples_operation.origin if hasattr(request.deny_samples_operation, 'origin') else 'train'

                # Find the loader for the requested origin
                loader_names = get_dataloaders()
                ds = None
                for loader_name in loader_names:
                    loader = components.get(loader_name)
                    if loader is None:
                        continue
                    tracked_ds = getattr(loader, "tracked_dataset", None)
                    if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                        if tracked_ds._dataset_split == origin:
                            ds = loader
                            break

                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message=f"No dataloader registered for origin '{origin}'",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.denylist_samples(
                    set(request.deny_samples_operation.sample_ids),
                    accumulate=request.deny_samples_operation.accumulate,
                )
                return pb2.CommandResponse(
                    success=True,
                    message=f"Denied {denied_cnt} samples from '{origin}'",
                )

        if request.HasField("deny_eval_samples_operation"):
            with weightslab_rlock:
                from weightslab.backend.ledgers import get_dataloaders

                denied_cnt = len(request.deny_eval_samples_operation.sample_ids)
                origin = request.deny_eval_samples_operation.origin if hasattr(request.deny_eval_samples_operation, 'origin') else 'eval'

                # Find the loader for the requested origin
                loader_names = get_dataloaders()
                ds = None
                for loader_name in loader_names:
                    loader = components.get(loader_name)
                    if loader is None:
                        continue
                    tracked_ds = getattr(loader, "tracked_dataset", None)
                    if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                        if tracked_ds._dataset_split == origin:
                            ds = loader
                            break

                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message=f"No dataloader registered for origin '{origin}'",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.denylist_samples(
                    set(request.deny_eval_samples_operation.sample_ids),
                    accumulate=request.deny_eval_samples_operation.accumulate,
                )
            return pb2.CommandResponse(
                success=True,
                message=f"Denied {denied_cnt} samples from '{origin}'",
            )

        if request.HasField("remove_from_denylist_operation"):
            with weightslab_rlock:
                from weightslab.backend.ledgers import get_dataloaders

                allowed = set(request.remove_from_denylist_operation.sample_ids)
                origin = request.remove_from_denylist_operation.origin if hasattr(request.remove_from_denylist_operation, 'origin') else 'train'

                # Find the loader for the requested origin
                loader_names = get_dataloaders()
                ds = None
                for loader_name in loader_names:
                    loader = components.get(loader_name)
                    if loader is None:
                        continue
                    tracked_ds = getattr(loader, "tracked_dataset", None)
                    if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                        if tracked_ds._dataset_split == origin:
                            ds = loader
                            break

                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message=f"No dataloader registered for origin '{origin}'",
                    )
                dataset = getattr(ds, "tracked_dataset", ds)
                dataset.allowlist_samples(allowed)
                return pb2.CommandResponse(
                    success=True,
                    message=f"Un-denied {len(allowed)} samples from '{origin}'",
                )

        if request.HasField("remove_eval_from_denylist_operation"):
            with weightslab_rlock:
                from weightslab.backend.ledgers import get_dataloaders

                allowed = set(request.remove_eval_from_denylist_operation.sample_ids)
                origin = request.remove_eval_from_denylist_operation.origin if hasattr(request.remove_eval_from_denylist_operation, 'origin') else 'eval'

                # Find the loader for the requested origin
                loader_names = get_dataloaders()
                ds = None
                for loader_name in loader_names:
                    loader = components.get(loader_name)
                    if loader is None:
                        continue
                    tracked_ds = getattr(loader, "tracked_dataset", None)
                    if tracked_ds and hasattr(tracked_ds, "_dataset_split"):
                        if tracked_ds._dataset_split == origin:
                            ds = loader
                            break

                if ds is None:
                    return pb2.CommandResponse(
                        success=False,
                        message=f"No dataloader registered for origin '{origin}'",
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
