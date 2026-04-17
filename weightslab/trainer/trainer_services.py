import os
import time
import grpc
import logging
from typing import Iterable
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from threading import Thread
from concurrent import futures

from weightslab.trainer.trainer_tools import *

from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService

# Watchdog module ? also registers WATCHDOG log level and logger.watchdog()
from weightslab.watchdog import (
    WeighlabsWatchdog,
    RpcWatchdogState,
    RpcTimingAndWatchdogInterceptor,
    GrpcServerManager,
)
from weightslab.components.global_monitoring import weightslab_rlock

# Global logger
logger = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_auth_tokens() -> set[str]:
    """Load accepted bearer/API-key tokens from env vars."""
    raw_values: list[str] = []
    single = os.getenv("GRPC_AUTH_TOKEN")
    many = os.getenv("GRPC_AUTH_TOKENS")
    if single:
        raw_values.append(single)
    if many:
        raw_values.extend(many.split(","))
    return {v.strip() for v in raw_values if v and v.strip()}


def _validate_file_exists(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise RuntimeError(
            f"{label} not found at '{path}'. "
            "Generate local certificates with 'weights_studio/docker/generate-dev-certs.ps1' "
            "(PowerShell) or 'weights_studio/docker/generate-dev-certs.sh' (bash), "
            "or set GRPC_TLS_* env vars to valid paths."
        )


def _run_security_preflight(
    *,
    grpc_tls_enabled: bool,
    grpc_tls_key_file: str,
    grpc_tls_cert_file: str,
    grpc_tls_ca_file: str,
    grpc_tls_require_client_auth: bool,
    auth_tokens: set[str],
) -> None:
    """Validate security-critical configuration before spawning gRPC threads."""
    if not grpc_tls_enabled:
        logger.warning(
            "[gRPC] GRPC_TLS_ENABLED=0. Traffic will be unencrypted. "
            "Use only for isolated local debugging."
        )
    else:
        _validate_file_exists(grpc_tls_key_file, "gRPC TLS private key (GRPC_TLS_KEY_FILE)")
        _validate_file_exists(grpc_tls_cert_file, "gRPC TLS certificate (GRPC_TLS_CERT_FILE)")
        if grpc_tls_require_client_auth:
            _validate_file_exists(grpc_tls_ca_file, "gRPC TLS CA file (GRPC_TLS_CA_FILE)")
        logger.info(
            "[gRPC] TLS preflight OK (mTLS=%s, cert=%s, key=%s, ca=%s)",
            grpc_tls_require_client_auth,
            grpc_tls_cert_file,
            grpc_tls_key_file,
            grpc_tls_ca_file if grpc_tls_require_client_auth else "<unused>",
        )

    if auth_tokens:
        logger.info("[gRPC] Auth token preflight OK (%d token(s) loaded)", len(auth_tokens))
    else:
        logger.warning(
            "[gRPC] No GRPC_AUTH_TOKEN/GRPC_AUTH_TOKENS configured. "
            "Only transport-level trust (TLS/mTLS) will protect RPC access."
        )


def _iter_possible_tokens(invocation_metadata: Iterable[tuple[str, str]] | None):
    if not invocation_metadata:
        return

    for key, value in invocation_metadata:
        if not value:
            continue
        norm_key = str(key).strip().lower()
        norm_value = str(value).strip()

        if norm_key == "authorization":
            if norm_value.lower().startswith("bearer "):
                token = norm_value[7:].strip()
                if token:
                    yield token
            else:
                yield norm_value
        elif norm_key in {"x-api-key", "x-grpc-auth-token"}:
            yield norm_value


class AuthTokenInterceptor(grpc.ServerInterceptor):
    """Validates a token from gRPC metadata for every incoming RPC."""

    def __init__(self, accepted_tokens: set[str]):
        self._accepted_tokens = accepted_tokens

    def _is_authorized(self, handler_call_details: grpc.HandlerCallDetails) -> bool:
        for token in _iter_possible_tokens(handler_call_details.invocation_metadata):
            if token in self._accepted_tokens:
                return True
        return False

    @staticmethod
    def _abort_unary_unary(_, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid gRPC auth token")

    @staticmethod
    def _abort_unary_stream(_, context):
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid gRPC auth token")
        yield from ()

    @staticmethod
    def _abort_stream_unary(request_iterator, context):
        del request_iterator
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid gRPC auth token")

    @staticmethod
    def _abort_stream_stream(request_iterator, context):
        del request_iterator
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid gRPC auth token")
        yield from ()

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None:
            return None

        if self._is_authorized(handler_call_details):
            return handler

        logger.warning("[gRPC] Unauthorized request denied for method=%s", handler_call_details.method)

        if handler.unary_unary:
            return grpc.unary_unary_rpc_method_handler(
                self._abort_unary_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.unary_stream:
            return grpc.unary_stream_rpc_method_handler(
                self._abort_unary_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_unary:
            return grpc.stream_unary_rpc_method_handler(
                self._abort_stream_unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        if handler.stream_stream:
            return grpc.stream_stream_rpc_method_handler(
                self._abort_stream_stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )
        return handler


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
        # Prefer explicit AgentService when present (new wiring).
        # Use vars(...) to avoid MagicMock auto-creating attributes that can
        # hide the real fallback path in unit/integration tests.
        agent_service = vars(self._exp_service).get("agent_service")
        if agent_service is not None and hasattr(agent_service, "CheckAgentHealth"):
            return agent_service.CheckAgentHealth(request, context)

        data_service = vars(self._exp_service).get("data_service")
        if data_service is not None:
            # Backward-compatible delegation expected by older tests/mocks.
            if hasattr(data_service, "CheckAgentHealth"):
                return data_service.CheckAgentHealth(request, context)

            # Real DataService fallback when no dedicated RPC method exists.
            if hasattr(data_service, "_is_agent_available"):
                import weightslab.proto.experiment_service_pb2 as pb2

                try:
                    available = bool(data_service._is_agent_available())
                except Exception:
                    available = False

                message = (
                    "Agent available. Ready to help you. Type /model to test another model or /reset to clear your API key and start over."
                    if available
                    else "Agent not configured. Type /init to set up."
                )
                return pb2.AgentHealthResponse(available=available, message=message)

        raise RuntimeError("ExperimentServiceServicer has no agent health provider configured")

    def InitializeAgent(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.InitializeAgent({request})")
        return self._exp_service.agent_service.InitializeAgent(request, context)

    def ChangeAgentModel(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.ChangeAgentModel({request})")
        return self._exp_service.agent_service.ChangeAgentModel(request, context)

    def GetAgentModels(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetAgentModels({request})")
        return self._exp_service.agent_service.GetAgentModels(request, context)

    def ResetAgent(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.ResetAgent({request})")
        return self._exp_service.agent_service.ResetAgent(request, context)

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

    # -------------------------------------------------------------------------
    # Evaluation mode
    # -------------------------------------------------------------------------
    def TriggerEvaluation(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.TriggerEvaluation({request})")
        return self._exp_service.TriggerEvaluation(request, context)

    def GetEvaluationStatus(self, request, context):
        logger.debug(f"\nExperimentServiceServicer.GetEvaluationStatus({request})")
        return self._exp_service.GetEvaluationStatus(request, context)


# -----------------------------------------------------------------------------
# Serving gRPC communication
# -----------------------------------------------------------------------------
def grpc_serve(
    n_workers_grpc: int = None,
    grpc_host: str = None,
    grpc_port: int = None,
    force_parameters: bool = False,
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

    grpc_host = os.getenv("GRPC_BACKEND_HOST", "0.0.0.0") if not force_parameters or grpc_host is None else grpc_host
    grpc_port = int(os.getenv("GRPC_BACKEND_PORT", 50051)) if not force_parameters or grpc_port is None else grpc_port
    watchdog_threshold_s = float(os.getenv("GRPC_WATCHDOG_STUCK_SECONDS", "60"))
    watchdog_interval_s = float(os.getenv("GRPC_WATCHDOG_INTERVAL_SECONDS", "5"))
    watchdog_exit_on_stuck = str(os.getenv("GRPC_WATCHDOG_EXIT_ON_STUCK", "0")).strip().lower() in {"1", "true", "yes", "on"}
    watchdog_restart_threshold = int(os.getenv("GRPC_WATCHDOG_RESTART_THRESHOLD", "3"))  # Restart after 3 unhealthy checks
    watchdog_details_limit = int(os.getenv("GRPC_WATCHDOG_INFLIGHT_DETAILS_LIMIT", "10"))
    watchdog_disabled = str(os.getenv("WEIGHTSLAB_DISABLE_WATCHDOGS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    grpc_tls_enabled = _is_truthy(os.getenv("GRPC_TLS_ENABLED", "1"))
    grpc_tls_key_file = os.getenv("GRPC_TLS_KEY_FILE", "certs/backend-server.key")
    grpc_tls_cert_file = os.getenv("GRPC_TLS_CERT_FILE", "certs/backend-server.crt")
    grpc_tls_ca_file = os.getenv("GRPC_TLS_CA_FILE", "certs/ca.crt")
    grpc_tls_require_client_auth = _is_truthy(os.getenv("GRPC_TLS_REQUIRE_CLIENT_AUTH", "1"))
    auth_tokens = _load_auth_tokens()
    max_concurrent_rpcs_env = os.getenv("GRPC_MAX_CONCURRENT_RPCS")
    if max_concurrent_rpcs_env is not None:
        max_concurrent_rpcs = int(max_concurrent_rpcs_env)
    elif max_concurrent_rpcs is None and n_workers_grpc is not None:
        max_concurrent_rpcs = int(n_workers_grpc)

    _run_security_preflight(
        grpc_tls_enabled=grpc_tls_enabled,
        grpc_tls_key_file=grpc_tls_key_file,
        grpc_tls_cert_file=grpc_tls_cert_file,
        grpc_tls_ca_file=grpc_tls_ca_file,
        grpc_tls_require_client_auth=grpc_tls_require_client_auth,
        auth_tokens=auth_tokens,
    )

    # Build watchdog components. In debug sessions, watchdogs can be disabled
    # to avoid lock/RPC timeout interruptions while paused on breakpoints.
    if watchdog_disabled:
        watchdog = None
        watchdog_state = RpcWatchdogState(stuck_threshold_s=watchdog_threshold_s)
        server_manager = GrpcServerManager()
        logger.info("[gRPC] Watchdogs disabled via WEIGHTSLAB_DISABLE_WATCHDOGS.")
    else:
        watchdog = WeighlabsWatchdog(
            stuck_threshold_s=watchdog_threshold_s,
            poll_interval_s=watchdog_interval_s,
            restart_threshold=watchdog_restart_threshold,
            exit_on_stuck=watchdog_exit_on_stuck,
            details_limit=watchdog_details_limit,
        )
        watchdog.register_lock("weightslab_rlock", weightslab_rlock)

        # Eval thread monitor — no timeout, just liveness.  Lazy imports avoid
        # circular dependencies since weightslab.src imports trainer code.
        def _get_eval_controller():
            from weightslab.components.evaluation_controller import eval_controller as _ec
            return _ec

        def _get_eval_thread():
            import weightslab.src as _src
            return _src._EVAL_WORKER_THREAD

        watchdog.register_eval_monitor(
            get_controller=_get_eval_controller,
            get_thread=_get_eval_thread,
        )
        watchdog_state = watchdog.rpc_state       # shared with RpcTimingAndWatchdogInterceptor
        server_manager = watchdog.server_manager  # shared with serving_thread_callback
    logger.debug(
        f"grpc_serve called with parameters: n_workers_grpc={n_workers_grpc}, grpc_host={grpc_host}, grpc_port={grpc_port}, "
        f"watchdog_threshold_s={watchdog_threshold_s}, watchdog_interval_s={watchdog_interval_s}, watchdog_exit_on_stuck={watchdog_exit_on_stuck}, watchdog_restart_threshold={watchdog_restart_threshold}, "
        f"watchdog_details_limit={watchdog_details_limit}, max_concurrent_rpcs={max_concurrent_rpcs}"
    )

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
                    interceptors=[
                        RpcTimingAndWatchdogInterceptor(watchdog_state),
                        *( [AuthTokenInterceptor(auth_tokens)] if auth_tokens else [] ),
                    ],
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

                if grpc_tls_enabled:
                    with open(grpc_tls_key_file, "rb") as f:
                        private_key = f.read()
                    with open(grpc_tls_cert_file, "rb") as f:
                        certificate_chain = f.read()

                    root_certificates = None
                    if grpc_tls_require_client_auth:
                        with open(grpc_tls_ca_file, "rb") as f:
                            root_certificates = f.read()

                    credentials = grpc.ssl_server_credentials(
                        private_key_certificate_chain_pairs=[(private_key, certificate_chain)],
                        root_certificates=root_certificates,
                        require_client_auth=grpc_tls_require_client_auth,
                    )
                    bound_port = server.add_secure_port(bind_addr, credentials)
                    logger.info(
                        "[gRPC] TLS enabled (mTLS=%s, cert=%s, key=%s, ca=%s)",
                        grpc_tls_require_client_auth,
                        grpc_tls_cert_file,
                        grpc_tls_key_file,
                        grpc_tls_ca_file if grpc_tls_require_client_auth else "<unused>",
                    )
                else:
                    bound_port = server.add_insecure_port(bind_addr)
                    logger.warning("[gRPC] TLS disabled; using insecure transport on %s", bind_addr)

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
    if watchdog is not None:
        watchdog.start()

        logger.info(
            "[gRPC] Server and watchdog started (host=%s port=%d workers=%s threshold=%.1fs interval=%.1fs restart_after=%d exit_on_stuck=%s)",
            grpc_host, grpc_port, n_workers_grpc,
            watchdog_threshold_s, watchdog_interval_s,
            watchdog_restart_threshold, watchdog_exit_on_stuck,
        )
    else:
        logger.info(
            "[gRPC] Server started with watchdogs disabled (host=%s port=%d workers=%s)",
            grpc_host, grpc_port, n_workers_grpc,
        )

if __name__ == "__main__":
    grpc_serve()

