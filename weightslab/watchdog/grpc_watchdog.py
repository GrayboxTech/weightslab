"""gRPC-specific watchdog primitives.

Moved from weightslab/trainer/trainer_services.py so the unified
WeighlabsWatchdog can own them without creating a circular import.

trainer_services.py re-exports these for backward compatibility.
"""

import time
import logging
import grpc

from threading import Lock, Event

from weightslab.watchdog.log_level import WATCHDOG  # noqa: F401 — registers level


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RpcWatchdogState
# ---------------------------------------------------------------------------

class RpcWatchdogState:
    """Tracks in-flight RPCs so prolonged stalls are detectable."""

    def __init__(self, stuck_threshold_s: float = 60.0) -> None:
        self._lock = Lock()
        self._next_id = 0
        self._in_flight: dict = {}
        self._stuck_threshold_s = stuck_threshold_s
        self._unhealthy_count = 0

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

    def snapshot(self, details_limit: int = 0) -> dict:
        now = time.monotonic()
        with self._lock:
            count = len(self._in_flight)
            oldest_age_s = 0.0
            oldest_method = None
            in_flight_details = []
            if self._in_flight:
                _, (method, started_at) = min(
                    self._in_flight.items(), key=lambda kv: kv[1][1]
                )
                oldest_age_s = max(0.0, now - started_at)
                oldest_method = method
                if details_limit > 0:
                    ordered = sorted(self._in_flight.items(), key=lambda kv: kv[1][1])
                    for rpc_id, (rpc_method, rpc_started_at) in ordered[:details_limit]:
                        in_flight_details.append({
                            "rpc_id": rpc_id,
                            "method": rpc_method,
                            "age_s": max(0.0, now - rpc_started_at),
                        })
            unhealthy = oldest_age_s >= self._stuck_threshold_s
            return {
                "in_flight": count,
                "oldest_age_s": oldest_age_s,
                "oldest_method": oldest_method,
                "unhealthy": unhealthy,
                "in_flight_details": in_flight_details,
            }

    def record_unhealthy(self) -> int:
        with self._lock:
            self._unhealthy_count += 1
            return self._unhealthy_count

    def record_healthy(self) -> int:
        with self._lock:
            self._unhealthy_count = 0
            return 0

    def clear_for_restart(self) -> int:
        """Drop stale in-flight records after a forced server recycle."""
        with self._lock:
            cleared = len(self._in_flight)
            self._in_flight.clear()
            self._unhealthy_count = 0
            return cleared


# ---------------------------------------------------------------------------
# RpcTimingAndWatchdogInterceptor
# ---------------------------------------------------------------------------

class RpcTimingAndWatchdogInterceptor(grpc.ServerInterceptor):
    """Logs per-RPC timings and keeps RpcWatchdogState for long-running calls."""

    def __init__(self, watchdog_state: RpcWatchdogState) -> None:
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
                if elapsed_ms > 2000:
                    logger.warning("[gRPC] %s completed in %.1fms", method_name, elapsed_ms)
                else:
                    logger.debug("[gRPC] %s completed in %.1fms", method_name, elapsed_ms)

        return grpc.unary_unary_rpc_method_handler(
            unary_unary_wrapper,
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )


# ---------------------------------------------------------------------------
# GrpcServerManager
# ---------------------------------------------------------------------------

class GrpcServerManager:
    """Manages the gRPC server lifecycle, allowing restart from the watchdog."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._server = None
        self._restart_requested = Event()

    def set_server(self, server) -> None:
        with self._lock:
            self._server = server

    def stop(self, grace: float = 5.0) -> None:
        with self._lock:
            if self._server:
                logger.watchdog("[gRPC] Requesting graceful shutdown with %.1fs grace", grace)  # type: ignore[attr-defined]
                self._server.stop(grace=grace)
                self._server = None

    def request_restart(self) -> None:
        self._restart_requested.set()

    def should_restart(self) -> bool:
        return self._restart_requested.is_set()

    def clear_restart_request(self) -> None:
        self._restart_requested.clear()
