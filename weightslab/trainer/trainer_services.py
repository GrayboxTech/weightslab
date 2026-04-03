import os
import time
import grpc
import logging
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from threading import Thread
from concurrent import futures

from weightslab.trainer.trainer_tools import *

from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService

# Watchdog module — also registers WATCHDOG log level and logger.watchdog()
from weightslab.watchdog import (
    WeighlabsWatchdog,
    RpcWatchdogState,
    RpcTimingAndWatchdogInterceptor,
    GrpcServerManager,
)
from weightslab.components.global_monitoring import weightslab_rlock

# Global logger
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compat note: RpcWatchdogState, RpcTimingAndWatchdogInterceptor and
# GrpcServerManager are now defined in weightslab.watchdog.grpc_watchdog and
# re-exported above.  External code that imported them from trainer_services
# continues to work unchanged.
# ---------------------------------------------------------------------------


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    """
    gRPC servicer for experiment-related services.

    This class now delegates to a domain-level ExperimentService instance
    which in turn uses smaller sub-services (model/data/etc.).
    """

    def __init__(self, exp_name: str = None, exp_service: ExperimentService | None = None):
        if exp_service is None:
            ctx = ExperimentContext(exp_name=exp_name)
            exp_service = ExperimentService(ctx=ctx)
            self._ctx = ctx
        self._exp_service = exp_service

    # -------------------------------------------------------------------------
    # Sample retrieval (images / segmentation / recon)
    # -------------------------------------------------------------------------
    def GetSamples(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetSamples({request})")
        return self._exp_service.model_service.GetSamples(request, context)

    # -------------------------------------------------------------------------
    # Weights inspection
    # -------------------------------------------------------------------------
    def GetWeights(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetWeights({request})")
        return self._exp_service.model_service.GetWeights(request, context)

    # -------------------------------------------------------------------------
    # Activations
    # -------------------------------------------------------------------------
    def GetActivations(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetActivations({request})")
        return self._exp_service.model_service.GetActivations(request, context)

    # -------------------------------------------------------------------------
    # Data service helpers + RPCs (for weights_studio UI)
    # -------------------------------------------------------------------------
    def ApplyDataQuery(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.ApplyDataQuery({request})")
        return self._exp_service.data_service.ApplyDataQuery(request, context)

    def GetDataSamples(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetDataSamples({request})")
        return self._exp_service.data_service.GetDataSamples(request, context)

    def EditDataSample(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.EditDataSample({request})")
        return self._exp_service.data_service.EditDataSample(request, context)

    def GetDataSplits(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetDataSplits({request})")
        return self._exp_service.data_service.GetDataSplits(request, context)

    def CheckAgentHealth(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.CheckAgentHealth({request})")
        return self._exp_service.data_service.CheckAgentHealth(request, context)

    # -------------------------------------------------------------------------
    # Logger data sync for WeightsStudio
    # -------------------------------------------------------------------------
    def GetLatestLoggerData(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetLatestLoggerData({request})")
        return self._exp_service.GetLatestLoggerData(request, context)

    # -------------------------------------------------------------------------
    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def ExperimentCommand(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.ExperimentCommand({request})")
        return self._exp_service.ExperimentCommand(request, context)

    # -------------------------------------------------------------------------
    # Weight manipulation (architecture operations)
    # -------------------------------------------------------------------------
    def ManipulateWeights(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.ManipulateWeights({request})")
        return self._exp_service.model_service.ManipulateWeights(request, context)

    # -------------------------------------------------------------------------
    # Checkpoint restore
    # -------------------------------------------------------------------------
    def RestoreCheckpoint(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.RestoreCheckpoint({request})")
        return self._exp_service.RestoreCheckpoint(request, context)


# -----------------------------------------------------------------------------
# Serving gRPC communication
# -----------------------------------------------------------------------------
def grpc_serve(
    n_workers_grpc: int = None,
    grpc_host: str = "0.0.0.0",
    grpc_port: int = 50051,
    force_parameters: bool = True,
    max_concurrent_rpcs: int = None,
    **_,
):
    """Configure trainer services such as gRPC server.
    Args:
        n_workers_grpc (int): Number of threads for the gRPC server.
        grpc_port (int): Port number for the gRPC server.
    """
    import weightslab.trainer.trainer_services as trainer
    from weightslab.trainer.trainer_tools import force_kill_all_python_processes

    grpc_host = os.getenv("GRPC_BACKEND_HOST", grpc_host) if not force_parameters else grpc_host
    grpc_port = int(os.getenv("GRPC_BACKEND_PORT", grpc_port)) if not force_parameters else grpc_port
    watchdog_threshold_s = float(os.getenv("GRPC_WATCHDOG_STUCK_SECONDS", "60"))
    watchdog_interval_s = float(os.getenv("GRPC_WATCHDOG_INTERVAL_SECONDS", "5"))
    watchdog_exit_on_stuck = str(os.getenv("GRPC_WATCHDOG_EXIT_ON_STUCK", "0")).strip().lower() in {"1", "true", "yes", "on"}
    watchdog_restart_threshold = int(os.getenv("GRPC_WATCHDOG_RESTART_THRESHOLD", "3"))  # Restart after 3 unhealthy checks
    watchdog_details_limit = int(os.getenv("GRPC_WATCHDOG_INFLIGHT_DETAILS_LIMIT", "10"))
    max_concurrent_rpcs_env = os.getenv("GRPC_MAX_CONCURRENT_RPCS")
    if max_concurrent_rpcs_env is not None:
        max_concurrent_rpcs = int(max_concurrent_rpcs_env)
    elif max_concurrent_rpcs is None and n_workers_grpc is not None:
        max_concurrent_rpcs = int(n_workers_grpc)

    # Build unified watchdog (manages locks + gRPC threads)
    watchdog = WeighlabsWatchdog(
        stuck_threshold_s=watchdog_threshold_s,
        poll_interval_s=watchdog_interval_s,
        restart_threshold=watchdog_restart_threshold,
        exit_on_stuck=watchdog_exit_on_stuck,
        details_limit=watchdog_details_limit,
    )
    watchdog.register_lock("weightslab_rlock", weightslab_rlock)
    watchdog_state = watchdog.rpc_state       # shared with RpcTimingAndWatchdogInterceptor
    server_manager = watchdog.server_manager  # shared with serving_thread_callback

    def serving_thread_callback():
        logger.info("[gRPC] Thread callback started")
        try:
            while True:  # Loop to allow restarts
                _effective_workers = n_workers_grpc or min(32, (os.cpu_count() or 1) + 4)
                logger.info(
                    "[gRPC] Creating ThreadPoolExecutor with %d worker threads (n_workers_grpc=%s, max_concurrent_rpcs=%s)",
                    _effective_workers, n_workers_grpc, max_concurrent_rpcs,
                )
                _max_msg = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", 256 * 1024 * 1024))  # 256 MB
                server = grpc.server(
                    futures.ThreadPoolExecutor(
                        thread_name_prefix="WL-gRPC-Worker",
                        max_workers=_effective_workers,
                    ),
                    interceptors=[RpcTimingAndWatchdogInterceptor(watchdog_state)],
                    options=[
                        ("grpc.max_send_message_length", _max_msg),
                        ("grpc.max_receive_message_length", _max_msg),
                    ],
                    maximum_concurrent_rpcs=max_concurrent_rpcs,
                )
                logger.info("[gRPC] Server object created")
                server_manager.set_server(server)
                servicer = trainer.ExperimentServiceServicer()
                pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
                logger.info("[gRPC] Servicer added")

                bind_addr = f"{grpc_host}:{grpc_port}"
                logger.info("[gRPC] Attempting to bind to %s", bind_addr)
                bound_port = server.add_insecure_port(bind_addr)
                if bound_port == 0:
                    logger.error("[gRPC] Failed to bind to %s. Port might be in use.", bind_addr)
                    return

                logger.info("[gRPC] Port %d bound successfully.", bound_port)
                server.start()
                logger.info("[gRPC] Server started and listening on %s", bind_addr)

                # Wait for restart signal from watchdog
                while not server_manager.should_restart():
                    time.sleep(0.5)

                logger.watchdog("[gRPC] Restart requested. Gracefully shutting down (5s grace)...")  # type: ignore[attr-defined]
                stop_event = server.stop(grace=5)
                stopped = stop_event.wait(timeout=6.0)
                if not stopped:
                    logger.watchdog("[gRPC] Graceful stop timed out; forcing immediate stop.")  # type: ignore[attr-defined]
                    server.stop(grace=0).wait(timeout=1.0)

                cleared = watchdog_state.clear_for_restart()
                if cleared:
                    logger.watchdog("[gRPC] Cleared %d stale in-flight RPC records after restart.", cleared)  # type: ignore[attr-defined]
                server_manager.clear_restart_request()
                logger.info("[gRPC] Server stopped. Restarting in 2s...")
                time.sleep(2)

        except Exception as e:
            logger.exception("[gRPC] Critical error in gRPC thread: %s", e)
        except KeyboardInterrupt:
            force_kill_all_python_processes()

    serving_thread = Thread(
        target=serving_thread_callback,
        daemon=True,
        name="WL-gRPC_Server",
    )
    serving_thread.start()
    watchdog.start()

    logger.info(
        "[gRPC] Server and watchdog started (host=%s port=%d workers=%s threshold=%.1fs interval=%.1fs restart_after=%d exit_on_stuck=%s)",
        grpc_host, grpc_port, n_workers_grpc,
        watchdog_threshold_s, watchdog_interval_s,
        watchdog_restart_threshold, watchdog_exit_on_stuck,
    )


if __name__ == "__main__":
    grpc_serve()
