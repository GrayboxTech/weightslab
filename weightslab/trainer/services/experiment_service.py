import os
import time
import types
import logging
import threading
import grpc

import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from weightslab.components.global_monitoring import weightslab_rlock, try_acquire_rlock, _GRPC_LOCK_TIMEOUT_S
from weightslab.trainer.trainer_tools import get_hyper_parameters_pb, get_layer_representation, get_layer_representations, get_data_set_representation
from weightslab.backend.ledgers import set_hyperparam, list_hyperparams, resolve_hp_name, get_hyperparams
from weightslab.backend import ledgers
from weightslab.backend.audit_logger import AuditLogger
from weightslab.trainer.services.model_service import ModelService
from weightslab.trainer.services.data_service import DataService
from weightslab.trainer.services.agent_service import AgentService
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.components.evaluation_controller import eval_controller


# Logger
logger = logging.getLogger(__name__)


# Default cap on the number of per-sample logger points returned per sample in the
# break-by-slices view. A run with 10k tagged samples × 10k steps would otherwise
# stream 100M points. Override with the WL_MAX_POINTS_PER_SAMPLE env variable
# (see docs/configuration.rst). Set to 0 (or a negative value) to disable the cap.
_DEFAULT_MAX_POINTS_PER_SAMPLE = 200


def _max_points_per_sample() -> int:
    """Read WL_MAX_POINTS_PER_SAMPLE; 0/negative/invalid disables the cap (returns 0)."""
    raw = os.getenv("WL_MAX_POINTS_PER_SAMPLE")
    if raw is None or raw == "":
        return _DEFAULT_MAX_POINTS_PER_SAMPLE
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "WL_MAX_POINTS_PER_SAMPLE=%r is not an integer — using default %d",
            raw, _DEFAULT_MAX_POINTS_PER_SAMPLE,
        )
        return _DEFAULT_MAX_POINTS_PER_SAMPLE
    return max(0, val)


def _downsample_uniform(series, max_points: int):
    """Reduce a per-sample, step-ordered ``series`` to at most ``max_points`` items.

    Uses evenly-spaced index selection that ALWAYS keeps the first and last point
    (so the curve's endpoints — start and most-recent value — are preserved) and
    samples the interior uniformly. This is a cheap, shape-preserving downsample;
    it does not invent interpolated values, it picks a representative subset of the
    real logged points. ``max_points <= 0`` means "no cap" (returns ``series``).
    """
    n = len(series)
    if max_points <= 0 or n <= max_points:
        return series
    if max_points == 1:
        return [series[-1]]
    # Evenly spaced indices across [0, n-1], inclusive of both endpoints.
    seen = set()
    picked = []
    for i in range(max_points):
        idx = round(i * (n - 1) / (max_points - 1))
        if idx not in seen:
            seen.add(idx)
            picked.append(series[idx])
    return picked


class ExperimentService(pb2_grpc.ExperimentServiceServicer):
    """
    Domain-level experiment service that orchestrates model/data services
    and handles general experiment-related commands.
    """

    # Limit concurrent GetLatestLoggerData calls to 3 to avoid piling up under slow I/O
    _logger_data_semaphore = threading.Semaphore(3)

    def __init__(self, ctx, root_log_dir: str = None):
        self._ctx = ctx
        self.model_service = ModelService(ctx)
        self.data_service = DataService(ctx)
        self.agent_service = AgentService(self.data_service)
        # Per-instance in-flight counter for GetLatestLoggerData
        self._logger_data_in_flight = 0
        self._logger_data_counter_lock = threading.Lock()

        # Initialize audit logger
        self.audit_logger = None
        if root_log_dir:
            try:
                self.audit_logger = AuditLogger(root_log_dir, ctx.exp_name or "default")
            except Exception as e:
                logger.warning(f"Failed to initialize audit logger: {e}")

    def _log_audit(
        self,
        action_type: str,
        status: str,
        details: dict = None,
        error: str = None,
    ) -> None:
        """Helper to log audit events if logger is available."""
        if self.audit_logger:
            try:
                self.audit_logger.log_event(
                    action_type=action_type,
                    status=status,
                    details=details,
                    error=error,
                )
            except Exception as e:
                logger.debug(f"Failed to log audit event: {e}")

    def _kick_eval_worker(self) -> None:
        """Best-effort trigger of backend-thread evaluation execution."""
        try:
            from weightslab import src as wl_src
            wl_src.trigger_pending_evaluation_async()
        except Exception as exc:
            logger.debug("[ExperimentService] Failed to kick eval worker: %s", exc)

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
        _t0 = time.monotonic()

        with self._logger_data_counter_lock:
            self._logger_data_in_flight += 1
            _in_flight = self._logger_data_in_flight

        logger.debug("GetLatestLoggerData: start (in_flight=%d, full_history=%s)",
                     _in_flight, request.request_full_history)

        # Concurrency cap: if already 3 calls in-flight, reject immediately
        acquired = self._logger_data_semaphore.acquire(blocking=False)
        if not acquired:
            elapsed_ms = (time.monotonic() - _t0) * 1000
            logger.warning("GetLatestLoggerData: concurrency cap hit (in_flight=%d), dropping after %.1fms",
                           _in_flight, elapsed_ms)
            with self._logger_data_counter_lock:
                self._logger_data_in_flight -= 1
            return pb2.GetLatestLoggerDataResponse(points=[])

        try:
            return self._get_latest_logger_data_impl(request, context)
        except Exception:
            elapsed_ms = (time.monotonic() - _t0) * 1000
            logger.exception("GetLatestLoggerData: handler exception after %.1fms (in_flight=%d)",
                             elapsed_ms, _in_flight)
            raise
        finally:
            self._logger_data_semaphore.release()
            with self._logger_data_counter_lock:
                self._logger_data_in_flight -= 1
            elapsed_ms = (time.monotonic() - _t0) * 1000
            _active = context.is_active() if context else True
            _level = logger.warning if elapsed_ms > 2000 else logger.debug
            _level("GetLatestLoggerData: done elapsed=%.1fms in_flight_peak=%d client_active=%s",
                   elapsed_ms, _in_flight, _active)

    def _get_latest_logger_data_impl(self, request, context):
        self._ctx.ensure_components()
        components = self._ctx.components
        signal_logger = components.get("signal_logger")
        if signal_logger ==  None:
            return pb2.GetLatestLoggerDataResponse(points=[])

        # Drop the request early if the client already disconnected
        if context and not context.is_active():
            logger.debug("GetLatestLoggerData: client cancelled before processing")
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
                        tag_col = f"{SampleStatsEx.TAG.value}:{tag}"
                        if tag_col in df.columns:
                            if mask is None:
                                mask = df[tag_col] == True
                            else:
                                mask = mask & (df[tag_col] == True)

                    if mask is not None:
                        filtered_df = df[mask]
                        # Normalize to strings because logger sample_id is serialized as text.
                        sample_ids = {str(sid) for sid in filtered_df.index.tolist()}

            if not graph_name or not sample_ids:
                return pb2.GetLatestLoggerDataResponse(points=[])

            # Query only this signal's compact arrays instead of inflating the whole
            # per-sample history into dicts (which OOMs on large signals). Eval/audit
            # per-sample data lives under eval-marker hashes, so derive audit_mode.
            now = int(time.time())
            eval_hashes = set(signal_logger.get_evaluation_marker_hashes())

            # AGGREGATE the per-sample curves into a single MEAN curve per
            # experiment_hash, computed here in the backend. Instead of streaming one
            # curve per sample (10k samples × 10k steps → 100M points), we send the
            # mean over the matching samples at each step → one curve per run. We
            # accumulate sum/count per (exp_hash, step) in a single pass (O(points),
            # no per-sample buffering), then emit mean = sum / count.
            from collections import defaultdict
            sums = defaultdict(float)
            counts = defaultdict(int)
            for sid, step, val, exp_hash in signal_logger.query_per_sample(
                    graph_name, sample_ids=sample_ids):
                key = (exp_hash if exp_hash else "N.A.", int(step))
                sums[key] += float(val)
                counts[key] += 1

            # Regroup the (exp_hash, step) means into one step-ordered series per hash.
            per_hash = defaultdict(list)
            for (exp_hash, step), total in sums.items():
                per_hash[exp_hash].append((step, total / counts[(exp_hash, step)]))

            # Still cap each returned curve's length (a run can have 10k+ steps);
            # WL_MAX_POINTS_PER_SAMPLE bounds points per returned curve (endpoints kept).
            max_points = _max_points_per_sample()
            for exp_hash, series in per_hash.items():
                series.sort(key=lambda sv: sv[0])  # order by step (model age)
                series = _downsample_uniform(series, max_points)
                audit = exp_hash in eval_hashes
                for step, mean_val in series:
                    points.append(
                        pb2.LoggerDataPoint(
                            metric_name=graph_name,
                            model_age=step,
                            metric_value=mean_val,
                            experiment_hash=exp_hash,
                            timestamp=now,
                            sample_id="",  # aggregated mean curve — not a single sample
                            audit_mode=audit,
                        )
                    )

            return pb2.GetLatestLoggerDataResponse(points=points)

        # Original behavior for non-break_by_slices mode
        if request.request_full_history:
            # Return full history
            max_points = request.max_points or 10000
            # Cancellation guard before the potentially heavy history fetch
            if context and not context.is_active():
                logger.debug("GetLatestLoggerData: client cancelled before get_signal_history")
                return pb2.GetLatestLoggerDataResponse(points=[])
            _th = time.monotonic()
            history = signal_logger.get_signal_history()
            _th_ms = (time.monotonic() - _th) * 1000
            if _th_ms > 500:
                logger.warning("get_signal_history() took %.1fms (slow — possible lock contention)", _th_ms)
            else:
                logger.debug("get_signal_history() took %.1fms", _th_ms)

            # Normalize history to grouped list format:
            # - Legacy: List[signal_entry]
            # - Current: Dict[metric_name][experiment_hash][step] -> List[signal_entry]
            signal_groups = {}
            if isinstance(history, dict):
                for metric_name, experiments in history.items():
                    if metric_name not in signal_groups:
                        signal_groups[metric_name] = []
                    if not isinstance(experiments, dict):
                        continue
                    for _, steps in experiments.items():
                        if not isinstance(steps, dict):
                            continue
                        for _, entries in steps.items():
                            if isinstance(entries, list):
                                signal_groups[metric_name].extend(
                                    [entry for entry in entries if isinstance(entry, dict)]
                                )
                            elif isinstance(entries, dict):
                                signal_groups[metric_name].append(entries)
            elif isinstance(history, list):
                for s in history:
                    if not isinstance(s, dict):
                        continue
                    metric_name = s.get("metric_name", "")
                    if metric_name not in signal_groups:
                        signal_groups[metric_name] = []
                    signal_groups[metric_name].append(s)

            # Take last max_points_per_signal for each signal and downsample if needed
            for metric_name, signal_history in signal_groups.items():
                # Keep deterministic order by model_age before sampling
                signal_history = sorted(signal_history, key=lambda item: item.get("model_age", 0))

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
                            sample_id="",  # No sample_id in aggregated mode
                            is_evaluation_marker=bool(s.get("is_evaluation_marker", False)),
                            split_name=str(s.get("split_name", "")),
                            evaluation_tags=[str(tag) for tag in s.get("evaluation_tags", []) or []],
                            point_note=str(s.get("point_note", "")),
                            audit_mode=bool(s.get("audit_mode", False)),
                        )
                    )
        else:
            # Return only queue (new data since last poll)
            if context and not context.is_active():
                logger.debug("GetLatestLoggerData: client cancelled before get_and_clear_queue")
                return pb2.GetLatestLoggerDataResponse(points=[])
            _tq = time.monotonic()
            queue_data = signal_logger.get_and_clear_queue()
            _tq_ms = (time.monotonic() - _tq) * 1000
            if _tq_ms > 200:
                logger.warning("get_and_clear_queue() took %.1fms (slow — possible lock contention)", _tq_ms)
            for s in queue_data:
                points.append(
                    pb2.LoggerDataPoint(
                        metric_name=s.get("metric_name", ""),
                        model_age=s.get("model_age", 0),
                        metric_value=s.get("metric_value", 0.0),
                        experiment_hash=s.get("experiment_hash", "N.A."),
                        timestamp=int(s.get("timestamp", time.time())),
                        sample_id="",  # No sample_id in queue mode
                        is_evaluation_marker=bool(s.get("is_evaluation_marker", False)),
                        split_name=str(s.get("split_name", "")),
                        evaluation_tags=[str(tag) for tag in s.get("evaluation_tags", []) or []],
                        point_note=str(s.get("point_note", "")),
                        audit_mode=bool(s.get("audit_mode", False)),
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
                    load_logger=False,  # Don't load logger for weights-only restore to avoid overwriting signals,
                    target_step=target_step,
                )
            else:
                success = checkpoint_manager.load_state(experiment_hash, load_logger=False)  # Don't load logger for full restore to avoid overwriting signals already in memory

            # Reply
            if success:
                logger.info(f"Successfully restored checkpoint: {experiment_hash}")
                self._log_audit(
                    "checkpoint_restore",
                    "success",
                    {
                        "checkpoint_id": experiment_hash,
                        "checkpoint_step": target_step,
                        "weights_only": load_weights_only,
                    },
                )
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
                self._log_audit(
                    "checkpoint_restore",
                    "failed",
                    {"checkpoint_id": experiment_hash},
                    error=f"Failed to restore checkpoint {experiment_hash}",
                )
                return pb2.RestoreCheckpointResponse(
                    success=False,
                    message=f"Failed to restore checkpoint {experiment_hash}"
                )
        except Exception as e:
            logger.error(f"Error during checkpoint restore: {str(e)}")
            self._log_audit(
                "checkpoint_restore",
                "failed",
                {"checkpoint_id": experiment_hash},
                error=str(e),
            )
            return pb2.RestoreCheckpointResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Evaluation mode RPC handlers
    # -------------------------------------------------------------------------
    def TriggerEvaluation(self, request, context):
        """Trigger an evaluation pass on the requested split.

        This stores the evaluation request in the global eval_controller.
        The actual pass runs in the training thread via ``run_pending_evaluation()``.
        """
        split_name = request.split_name or ""
        tags = list(request.tags) if request.tags else []
        use_full_set = bool(request.use_full_set)

        if not split_name:
            self._log_audit(
                "evaluation_start",
                "failed",
                error="split_name is required",
            )
            return pb2.TriggerEvaluationResponse(
                success=False,
                message="split_name is required",
            )

        accepted = eval_controller.request_evaluation(
            split_name=split_name,
            tags=tags,
            use_full_set=use_full_set,
        )

        if accepted:
            self._kick_eval_worker()
            logger.info(
                "[ExperimentService] TriggerEvaluation accepted: split=%s tags=%s full=%s",
                split_name, tags, use_full_set,
            )
            self._log_audit(
                "evaluation_start",
                "success",
                {
                    "eval_mode": split_name,
                    "tags": tags,
                    "use_full_set": use_full_set,
                },
            )
            return pb2.TriggerEvaluationResponse(
                success=True,
                message=f"Evaluation on '{split_name}' requested",
            )
        else:
            self._log_audit(
                "evaluation_start",
                "failed",
                error="Evaluation already in progress",
            )
            return pb2.TriggerEvaluationResponse(
                success=False,
                message="Evaluation already in progress",
            )

    def GetEvaluationStatus(self, request, context):
        """Return the current state of the evaluation controller."""
        status = eval_controller.get_status()
        progress = status.get("progress", {})
        return pb2.GetEvaluationStatusResponse(
            status=status.get("status", "idle"),
            current=int(progress.get("current", 0)),
            total=int(progress.get("total", 0)),
            message=str(progress.get("message", "")),
            error=str(status.get("error", "")),
            split_name=str(status.get("split_name", "")),
        )

    def CancelEvaluation(self, request, context):
        """Cancel a pending or running evaluation."""
        reason = str(request.reason or "Canceled by user")
        accepted = eval_controller.request_cancel(reason=reason)

        if accepted:
            logger.info("[ExperimentService] CancelEvaluation accepted: %s", reason)
            return pb2.CancelEvaluationResponse(
                success=True,
                message=f"Evaluation canceled: {reason}",
            )
        else:
            logger.info("[ExperimentService] CancelEvaluation rejected: no active evaluation")
            return pb2.CancelEvaluationResponse(
                success=False,
                message="No active evaluation to cancel",
            )

    def _get_live_hyper_parameter_descs(self, components):
        hyper_parameter_descs = list(get_hyper_parameters_pb(self._ctx.hyper_parameters))

        trainer = components.get("trainer") if components else None
        if trainer is None or not hasattr(trainer, "is_paused"):
            return hyper_parameter_descs

        # Trick for is training - tmp solution TODO(GP) Fix it
        is_training = not trainer.is_paused()

        hp_name = self._ctx.exp_name or resolve_hp_name()
        if hp_name:
            try:
                hp = get_hyperparams(hp_name)
                current_is_training = bool(hp.get("is_training", False)) if hasattr(hp, "get") else None
                if current_is_training is not None and current_is_training != is_training:
                    set_hyperparam(name=hp_name, key_path="is_training", value=is_training)
            except Exception:
                logger.debug("Failed to resync ledger is_training for %s", hp_name, exc_info=True)

        for desc in hyper_parameter_descs:
            if desc.name == "is_training" or desc.label in {"is_training", "Is Training"}:
                desc.type = "number"
                desc.numerical_value = 1.0 if is_training else 0.0
                desc.ClearField("string_value")
                return hyper_parameter_descs

        hyper_parameter_descs.append(
            pb2.HyperParameterDesc(
                label="Is Training",
                name="is_training",
                type="number",
                numerical_value=1.0 if is_training else 0.0,
            )
        )
        return hyper_parameter_descs

    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def ExperimentCommand(self, request, context):
        self._ctx.ensure_components()
        components = self._ctx.components

        # Write requests
        if request.HasField("plot_note_operation"):
            note_op = request.plot_note_operation
            metric_name = str(note_op.metric_name or "")
            experiment_hash = str(note_op.experiment_hash or "")
            note_text = str(note_op.note or "")

            # Get Current Model Age
            try:
                model_age = int(note_op.model_age)
            except Exception:
                model_age = 0

            if not metric_name or not experiment_hash:
                return pb2.CommandResponse(success=False, message="metric_name and experiment_hash are required for plot notes")

            signal_logger = components.get("signal_logger") if isinstance(components, dict) else None
            if signal_logger is None:
                signal_logger = ledgers.get_logger()
            if signal_logger is None or not hasattr(signal_logger, "set_point_note"):
                return pb2.CommandResponse(success=False, message="Signal logger unavailable for plot notes")

            updated = bool(signal_logger.set_point_note(metric_name, experiment_hash, model_age, note_text))
            if not updated:
                return pb2.CommandResponse(
                    success=False,
                    message=f"Signal point not found for note update: {metric_name}@{experiment_hash}:{model_age}",
                )

            checkpoint_manager = components.get("checkpoint_manager") if isinstance(components, dict) else None
            if checkpoint_manager is None:
                try:
                    checkpoint_manager = ledgers.get_checkpoint_manager()
                except Exception:
                    checkpoint_manager = None
            if checkpoint_manager is not None and hasattr(checkpoint_manager, "save_logger_snapshot"):
                try:
                    checkpoint_manager.save_logger_snapshot()
                except Exception:
                    logger.debug("Could not persist logger snapshot after note update", exc_info=True)

            # Log to audit trail
            self._log_audit_event(
                "note_write",
                "success",
                {
                    "metric_name": metric_name,
                    "model_age": model_age,
                    "note_text": note_text,
                    "note_action": "saved" if note_text.strip() else "cleared",
                },
            )

            return pb2.CommandResponse(
                success=True,
                message=(
                    f"Saved note for {metric_name} @ step {model_age}"
                    if note_text.strip()
                    else f"Cleared note for {metric_name} @ step {model_age}"
                ),
            )

        if request.HasField("hyper_parameter_change"):
            hyper_parameters = request.hyper_parameter_change.hyper_parameters

            # Get Hyper parameter context
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

            # Pause the experiment for HP chanage
            trainer = components.get("trainer")
            if trainer:
                trainer.pause()
                set_hyperparam(name=hp_name, key_path="is_training", value=False)
            logger.info("[WeightsLab] UI Command: HP changed, experiment paused!")

            # Update HP
            try:
                hp_now = None
                hp_changes = {}
                if hyper_parameters.HasField("training_steps_to_do"):
                    hp_changes["training_steps_to_do"] = hyper_parameters.training_steps_to_do
                    set_hyperparam(
                        name=hp_name,
                        key_path="training_steps_to_do",
                        value=hyper_parameters.training_steps_to_do
                    )
                if hyper_parameters.HasField("learning_rate"):
                    hp_changes["learning_rate"] = hyper_parameters.learning_rate
                    set_hyperparam(
                        name=hp_name,
                        key_path="optimizer.lr",
                        value=hyper_parameters.learning_rate
                    )
                if hyper_parameters.HasField("batch_size"):
                    hp_changes["batch_size"] = hyper_parameters.batch_size
                    set_hyperparam(
                        name=hp_name,
                        key_path="data.train_loader.batch_size",
                        value=hyper_parameters.batch_size
                    )
                if hasattr(hyper_parameters, 'val_batch_size') and hyper_parameters.HasField("val_batch_size"):
                    hp_changes["val_batch_size"] = hyper_parameters.val_batch_size
                    set_hyperparam(
                        name=hp_name,
                        key_path="data.val_loader.batch_size",
                        value=hyper_parameters.val_batch_size
                    )
                elif hasattr(hyper_parameters, 'eval_batch_size') and hyper_parameters.HasField("eval_batch_size"):
                    hp_changes["val_batch_size"] = hyper_parameters.eval_batch_size
                    set_hyperparam(
                        name=hp_name,
                        key_path="data.val_loader.batch_size",
                        value=hyper_parameters.eval_batch_size
                    )
                if hyper_parameters.HasField("test_batch_size"):
                    hp_changes["test_batch_size"] = hyper_parameters.test_batch_size
                    set_hyperparam(
                        name=hp_name,
                        key_path="data.test_loader.batch_size",
                        value=hyper_parameters.test_batch_size
                    )
                if hyper_parameters.HasField("full_eval_frequency"):
                    hp_changes["full_eval_frequency"] = hyper_parameters.full_eval_frequency
                    set_hyperparam(
                        name=hp_name,
                        key_path="eval_full_to_train_steps_ratio",
                        value=hyper_parameters.full_eval_frequency
                    )
                if hyper_parameters.HasField("checkpont_frequency"):
                    hp_changes["checkpont_frequency"] = hyper_parameters.checkpont_frequency
                    set_hyperparam(
                        name=hp_name,
                        key_path="experiment_dump_to_train_steps_ratio",
                        value=hyper_parameters.checkpont_frequency
                    )

                # Log HP changes if any
                if hp_changes:
                    self._log_audit(
                        "hp_change",
                        "success",
                        {"changed_params": hp_changes},
                    )

                # Process evaluation_mode toggle (UI button switch)
                try:
                    if hyper_parameters.HasField("evaluation_mode"):
                        incoming_eval = bool(hyper_parameters.evaluation_mode)
                        set_hyperparam(
                            name=hp_name,
                            key_path="evaluation_mode",
                            value=incoming_eval,
                        )
                        self._log_audit(
                            "mode_switch",
                            "success",
                            {
                                "from_mode": "train",
                                "to_mode": "evaluation" if incoming_eval else "train",
                                "evaluation_mode": incoming_eval,
                            },
                        )
                        if incoming_eval:
                            # Switching to eval mode implies pausing training
                            trainer = components.get("trainer")
                            if trainer:
                                trainer.pause()
                                set_hyperparam(name=hp_name, key_path="is_training", value=False)
                            logger.info("[WeightsLab] UI Command: Switch to EVALUATION Mode")
                        else:
                            logger.info("[WeightsLab] UI Command: Exit EVALUATION Mode")
                except ValueError:
                    pass

                # If evaluation_config JSON was sent, trigger evaluation immediately
                try:
                    if hyper_parameters.HasField("evaluation_config") and hyper_parameters.evaluation_config:
                        import json as _json
                        cfg = _json.loads(hyper_parameters.evaluation_config)
                        if bool(cfg.get("cancel", False)):
                            accepted = eval_controller.request_cancel(str(cfg.get("reason", "Canceled by user")))
                            if not accepted:
                                logger.info("[ExperimentService] Cancel requested but no active evaluation")
                        else:
                            raw_max_steps = cfg.get("max_steps", cfg.get("steps", None))
                            max_steps = None
                            try:
                                if raw_max_steps is not None:
                                    parsed_steps = int(raw_max_steps)
                                    if parsed_steps > 0:
                                        max_steps = parsed_steps
                            except Exception:
                                max_steps = None

                            eval_controller.request_evaluation(
                                split_name=cfg.get("split", ""),
                                tags=cfg.get("tags", []),
                                use_full_set=cfg.get("use_full_set", True),
                                max_steps=max_steps,
                            )
                            self._kick_eval_worker()
                except (ValueError, Exception) as _ec:
                    logger.debug("evaluation_config parse error: %s", _ec)

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
                                logger.info(f"\n[WeightsLab] Pausing to switch mode...")
                                trainer.pause()
                                set_hyperparam(name=hp_name, key_path="is_training", value=False)
                            mode_label = "AUDIT" if incoming_audit else "TRAIN"
                            logger.info(f"\n[WeightsLab] UI Command: Switch to {mode_label} Mode")
                            self._log_audit(
                                "mode_switch",
                                "success",
                                {
                                    "from_mode": "audit" if current_audit else "train",
                                    "to_mode": "audit" if incoming_audit else "train",
                                    "auditor_mode": incoming_audit,
                                },
                            )

                        # Always update the stored value (harmless no-op if unchanged)
                        set_hyperparam(
                            name=hp_name,
                            key_path="auditor_mode",
                            value=incoming_audit
                        )

                except ValueError:
                    pass

                # Process is_training AFTER mode is set — so Resume fires with correct mode
                if hyper_parameters.HasField("is_training"):
                    trainer = components.get("trainer")
                    if trainer is not None:
                        desired_is_training = bool(hyper_parameters.is_training)

                        if desired_is_training and (eval_controller.is_pending() or eval_controller.is_running()):
                            logger.info("\n[WeightsLab] Resume blocked: evaluation is pending/running")
                            desired_is_training = False

                        # Set number of steps desired to run before next pause if provided, based on current model age + requested nb_steps
                        if hyper_parameters.HasField("nb_steps"):
                            m = components.get("model")  # Get model
                            m_age = m.get_age()
                            logger.info(f"\n[WeightsLab] UI Command: Define number of steps at {hyper_parameters.nb_steps}")
                            if hyper_parameters.nb_steps > 0:
                                set_hyperparam(
                                    name=hp_name,
                                    key_path="pause_at_step",
                                    value=m_age+hyper_parameters.nb_steps
                                )

                        # If is_training flag is set, pause or resume accordingly
                        if desired_is_training:
                            logger.info("\n[WeightsLab] UI Command: RESUME")
                            trainer.resume()
                            self._log_audit(
                                "resume",
                                "success",
                                {"trainer_state": "running"},
                            )
                        else:
                            logger.info("\n[WeightsLab] UI Command: PAUSE")
                            trainer.pause()
                            self._log_audit(
                                "pause",
                                "success",
                                {"trainer_state": "paused"},
                            )

                        # Set training state and pause/resume accordingly
                        set_hyperparam(
                            name=hp_name,
                            key_path="is_training",
                            value=desired_is_training
                        )

            except Exception as e:
                return pb2.CommandResponse(
                    success=False,
                    message=f"Failed to set hyperparameters: {e}",
                )

            if request.HasField("load_checkpoint_operation"):
                if not try_acquire_rlock():
                    logger.error("[ExperimentCommand] weightslab_rlock timed out after %.0fs", _GRPC_LOCK_TIMEOUT_S)
                    context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"Training lock not acquired within {_GRPC_LOCK_TIMEOUT_S:.0f}s")
                    return pb2.CommandResponse(success=False, message="Lock timeout")
                try:
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
                finally:
                    weightslab_rlock.release()

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
                # Successful checkpoint load return
                return pb2.CommandResponse(success=True, message=f"Loaded checkpoint {checkpoint_id}")

            # Default return for other commands (e.g. hyperparameter changes)
            return pb2.CommandResponse(success=True, message="Command executed successfully")

        # Read requests
        response = pb2.CommandResponse(success=True, message="")
        if request.get_hyper_parameters:
            response.hyper_parameters_descs.extend(
                self._get_live_hyper_parameter_descs(components)
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
