import os
import time
import grpc
import logging
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from threading import Thread, Lock, Event
from concurrent import futures

from weightslab.trainer.trainer_tools import *

from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService

from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService

# Global logger
logger = logging.getLogger(__name__)


class RpcWatchdogState:
    """Tracks in-flight RPCs so prolonged stalls are detectable."""

    def __init__(self, stuck_threshold_s: float = 60.0):
        self._lock = Lock()
        self._next_id = 0
        self._in_flight = {}
        self._stuck_threshold_s = stuck_threshold_s
        self._unhealthy_count = 0  # Track consecutive unhealthy detections

    def begin(self, method_name: str) -> int:
        now = time.monotonic()
        with self._lock:
            self._next_id += 1
            rpc_id = self._next_id
            self._in_flight[rpc_id] = (method_name, now)
            return rpc_id

    def end(self, rpc_id: int) -> None:
        with self._lock:
            self._in_flight.pop(rpc_id, None)

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            count = len(self._in_flight)
            oldest_age_s = 0.0
            oldest_method = None
            if self._in_flight:
                oldest_id, (method, started_at) = min(self._in_flight.items(), key=lambda kv: kv[1][1])
                _ = oldest_id
                oldest_age_s = max(0.0, now - started_at)
                oldest_method = method
            unhealthy = oldest_age_s >= self._stuck_threshold_s
            return {
                "in_flight": count,
                "oldest_age_s": oldest_age_s,
                "oldest_method": oldest_method,
                "unhealthy": unhealthy,
            }

    def record_unhealthy(self):
        """Increment unhealthy counter, reset on healthy."""
        with self._lock:
            self._unhealthy_count += 1
            return self._unhealthy_count

    def record_healthy(self):
        """Reset unhealthy counter on healthy state."""
        with self._lock:
            self._unhealthy_count = 0
            return self._unhealthy_count


class RpcTimingAndWatchdogInterceptor(grpc.ServerInterceptor):
    """Logs per-RPC timings and keeps watchdog state for long-running calls."""

    def __init__(self, watchdog_state: RpcWatchdogState):
        self._watchdog_state = watchdog_state

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler

        method_name = handler_call_details.method

        def unary_unary_wrapper(request, context):
            rpc_id = self._watchdog_state.begin(method_name)
            t0 = time.monotonic()
            try:
                return handler.unary_unary(request, context)
            finally:
                elapsed_ms = (time.monotonic() - t0) * 1000
                self._watchdog_state.end(rpc_id)
                level = logger.warning if elapsed_ms > 2000 else logger.debug
                level("[gRPC] %s completed in %.1fms", method_name, elapsed_ms)

        return grpc.unary_unary_rpc_method_handler(
            unary_unary_wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


class GrpcServerManager:
    """Manages the gRPC server lifecycle, allowing restart from watchdog."""

    def __init__(self):
        self._lock = Lock()
        self._server = None
        self._restart_requested = Event()

    def set_server(self, server):
        """Store reference to server."""
        with self._lock:
            self._server = server

    def stop(self, grace: float = 5.0):
        """Gracefully stop the server."""
        with self._lock:
            if self._server:
                logger.info(f"[gRPC] Requesting graceful shutdown with {grace}s grace period")
                self._server.stop(grace=grace)
                self._server = None

    def request_restart(self):
        """Signal that server should restart."""
        self._restart_requested.set()

    def should_restart(self) -> bool:
        """Check if restart was requested."""
        return self._restart_requested.is_set()

    def clear_restart_request(self):
        """Clear the restart flag."""
        self._restart_requested.clear()


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
        logger.debug(f"ExperimentServiceServicer.GetSamples({request})")
        return self._exp_service.model_service.GetSamples(request, context)

    # -------------------------------------------------------------------------
    # Weights inspection
    # -------------------------------------------------------------------------
    def GetWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetWeights({request})")
        return self._exp_service.model_service.GetWeights(request, context)

    # -------------------------------------------------------------------------
    # Activations
    # -------------------------------------------------------------------------
    def GetActivations(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetActivations({request})")
        return self._exp_service.model_service.GetActivations(request, context)

    # -------------------------------------------------------------------------
    # Data service helpers + RPCs (for weights_studio UI)
    # -------------------------------------------------------------------------
    def ApplyDataQuery(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ApplyDataQuery({request})")
        return self._exp_service.data_service.ApplyDataQuery(request, context)

    def GetDataSamples(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetDataSamples({request})")
        return self._exp_service.data_service.GetDataSamples(request, context)

    def EditDataSample(self, request, context):
        logger.debug(f"ExperimentServiceServicer.EditDataSample({request})")
        return self._exp_service.data_service.EditDataSample(request, context)

    def GetDataSplits(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetDataSplits({request})")
        return self._exp_service.data_service.GetDataSplits(request, context)

    def CheckAgentHealth(self, request, context):
        logger.debug(f"ExperimentServiceServicer.CheckAgentHealth({request})")
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
        logger.debug(f"ExperimentServiceServicer.ExperimentCommand({request})")
        return self._exp_service.ExperimentCommand(request, context)

    # -------------------------------------------------------------------------
    # Weight manipulation (architecture operations)
    # -------------------------------------------------------------------------
    def ManipulateWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ManipulateWeights({request})")
        return self._exp_service.model_service.ManipulateWeights(request, context)

    # -------------------------------------------------------------------------
    # Checkpoint restore
    # -------------------------------------------------------------------------
    def RestoreCheckpoint(self, request, context):
        logger.debug(f"ExperimentServiceServicer.RestoreCheckpoint({request})")
        return self._exp_service.RestoreCheckpoint(request, context)


# -----------------------------------------------------------------------------
# Serving gRPC communication
# -----------------------------------------------------------------------------
def grpc_serve(n_workers_grpc: int = None, grpc_host: str = "0.0.0.0", grpc_port: int = 50051, **_):
    """Configure trainer services such as gRPC server.
    Args:
        n_workers_grpc (int): Number of threads for the gRPC server.
        grpc_port (int): Port number for the gRPC server.
    """
    import weightslab.trainer.trainer_services as trainer
    from weightslab.trainer.trainer_tools import force_kill_all_python_processes

    grpc_host = os.getenv("GRPC_BACKEND_HOST", grpc_host)
    grpc_port = int(os.getenv("GRPC_BACKEND_PORT", grpc_port))
    watchdog_threshold_s = float(os.getenv("GRPC_WATCHDOG_STUCK_SECONDS", "60"))
    watchdog_interval_s = float(os.getenv("GRPC_WATCHDOG_INTERVAL_SECONDS", "5"))
    watchdog_exit_on_stuck = str(os.getenv("GRPC_WATCHDOG_EXIT_ON_STUCK", "0")).strip().lower() in {"1", "true", "yes", "on"}
    watchdog_restart_threshold = int(os.getenv("GRPC_WATCHDOG_RESTART_THRESHOLD", "3"))  # Restart after 3 unhealthy checks

    watchdog_state = RpcWatchdogState(stuck_threshold_s=watchdog_threshold_s)
    server_manager = GrpcServerManager()

    def watchdog_thread_callback():
        while True:
            snap = watchdog_state.snapshot()
            if snap["unhealthy"]:
                unhealthy_count = watchdog_state.record_unhealthy()
                logger.error(
                    "[gRPC-Watchdog] unhealthy: in_flight=%d oldest_age=%.1fs oldest_method=%s threshold=%.1fs (unhealthy_count=%d)",
                    snap["in_flight"], snap["oldest_age_s"], snap["oldest_method"], watchdog_threshold_s, unhealthy_count,
                )
                if watchdog_exit_on_stuck:
                    logger.critical("[gRPC-Watchdog] exiting process due to stuck RPC (GRPC_WATCHDOG_EXIT_ON_STUCK enabled)")
                    os._exit(1)
                elif unhealthy_count >= watchdog_restart_threshold:
                    logger.warning(
                        "[gRPC-Watchdog] Too many consecutive unhealthy checks (%d >= %d). Requesting server restart.",
                        unhealthy_count, watchdog_restart_threshold
                    )
                    server_manager.request_restart()
            else:
                watchdog_state.record_healthy()
                logger.debug(
                    "[gRPC-Watchdog] healthy: in_flight=%d oldest_age=%.1fs",
                    snap["in_flight"], snap["oldest_age_s"],
                )
            time.sleep(max(1.0, watchdog_interval_s))

    def serving_thread_callback():
        logger.info("[gRPC] Thread callback started")
        try:
            while True:  # Loop to allow restarts
                logger.info("[gRPC] Creating ThreadPoolExecutor")
                _effective_workers = n_workers_grpc or min(32, (os.cpu_count() or 1) + 4)
                logger.info("[gRPC] Creating ThreadPoolExecutor with %d worker threads (n_workers_grpc=%s)",
                            _effective_workers, n_workers_grpc)
                # Allow large payloads for batches of HD images + segmentation masks.
                # Default 4 MB is too small for 720p image grids with mask arrays.
                _max_msg = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", 256 * 1024 * 1024))  # 256 MB
                server = grpc.server(
                    futures.ThreadPoolExecutor(
                        thread_name_prefix="WL-gRPC-Worker",
                        max_workers=_effective_workers
                    ),
                    interceptors=[RpcTimingAndWatchdogInterceptor(watchdog_state)],
                    options=[
                        ("grpc.max_send_message_length", _max_msg),
                        ("grpc.max_receive_message_length", _max_msg),
                    ],
                )
                logger.info("[gRPC] Server object created")
                server_manager.set_server(server)
                servicer = trainer.ExperimentServiceServicer()
                pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
                logger.info("[gRPC] Servicer added")

                # Bind to host:port
                bind_addr = f'{grpc_host}:{grpc_port}'
                logger.info(f"[gRPC] Attempting to bind to {bind_addr}")
                bound_port = server.add_insecure_port(bind_addr)

                if bound_port == 0:
                    logger.error(f"[gRPC] Failed to bind to {bind_addr}. Port might be in use.")
                    return

                logger.info(f"[gRPC] Port {bound_port} bound successfully.")
                server.start()
                logger.info(f"[gRPC] Server started and listening on {bind_addr}")

                # Wait for termination or restart signal
                while not server_manager.should_restart():
                    time.sleep(0.5)

                # Restart requested
                logger.warning("[gRPC] Restart requested by watchdog. Gracefully shutting down server (5s grace)...")
                server.stop(grace=5)
                server_manager.clear_restart_request()
                logger.info("[gRPC] Server stopped. Restarting...")
                time.sleep(2)  # Brief delay before restart

        except Exception as e:
            logger.exception(f"[gRPC] Critical error in gRPC thread: {e}")
        except KeyboardInterrupt:
            force_kill_all_python_processes()

    training_thread = Thread(
        target=serving_thread_callback,
        daemon=True,
        name="WL-gRPC_Server",
    )
    training_thread.start()

    watchdog_thread = Thread(
        target=watchdog_thread_callback,
        daemon=True,
        name="WL-gRPC_Watchdog",
    )
    watchdog_thread.start()

    logger.info("grpc_thread_started", extra={
        "thread_name": training_thread.name,
        "thread_id": training_thread.ident,
        "grpc_host": grpc_host,
        "grpc_port": grpc_port,
        "n_workers_grpc": n_workers_grpc,
        "watchdog_threshold_s": watchdog_threshold_s,
        "watchdog_interval_s": watchdog_interval_s,
        "watchdog_exit_on_stuck": watchdog_exit_on_stuck,
        "watchdog_restart_threshold": watchdog_restart_threshold,
    })


if __name__ == "__main__":
    grpc_serve()
