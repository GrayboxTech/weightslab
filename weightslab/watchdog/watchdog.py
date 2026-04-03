"""WeighlabsWatchdog — unified watchdog for locks and gRPC threads.

Combines:
  1. Lock monitoring  — polls MonitoredRLock instances, raises _WatchdogInterrupt
                        in the holder thread when the lock is held too long.
  2. gRPC monitoring  — detects stuck in-flight RPCs via RpcWatchdogState and
                        requests a server restart when the threshold is exceeded.

Typical usage (inside grpc_serve):

    watchdog = WeighlabsWatchdog(
        stuck_threshold_s=60.0,
        poll_interval_s=5.0,
        restart_threshold=3,
        exit_on_stuck=False,
    )
    watchdog.register_lock("weightslab_rlock", weightslab_rlock)
    watchdog.start()
    # watchdog.rpc_state  → pass to RpcTimingAndWatchdogInterceptor
    # watchdog.server_manager → used by serving_thread_callback
"""

import os
import logging
import threading
from typing import Dict, Optional

from weightslab.watchdog.log_level import WATCHDOG  # noqa: F401 — registers level
from weightslab.watchdog.lock_monitor import MonitoredRLock, raise_in_thread
from weightslab.watchdog.grpc_watchdog import RpcWatchdogState, GrpcServerManager


logger = logging.getLogger(__name__)


class WeighlabsWatchdog:
    """Unified watchdog that monitors both training locks and gRPC threads."""

    def __init__(
        self,
        stuck_threshold_s: float = 60.0,
        poll_interval_s: float = 5.0,
        restart_threshold: int = 3,
        exit_on_stuck: bool = False,
        details_limit: int = 10,
    ) -> None:
        self._stuck_threshold_s = stuck_threshold_s
        self._poll_interval_s = max(1.0, poll_interval_s)
        self._restart_threshold = restart_threshold
        self._exit_on_stuck = exit_on_stuck
        self._details_limit = details_limit

        # gRPC watchdog state (shared with RpcTimingAndWatchdogInterceptor)
        self.rpc_state = RpcWatchdogState(stuck_threshold_s=stuck_threshold_s)
        self.server_manager = GrpcServerManager()

        # Lock monitoring
        self._monitored_locks: Dict[str, MonitoredRLock] = {}

        # Internal state
        self._unhealthy_count: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lock registration
    # ------------------------------------------------------------------

    def register_lock(self, name: str, lock: MonitoredRLock) -> None:
        """Register a MonitoredRLock to be polled by the watchdog."""
        self._monitored_locks[name] = lock

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the watchdog background thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="WL-Watchdog",
            daemon=True,
        )
        self._thread.start()
        logger.watchdog(  # type: ignore[attr-defined]
            "[Watchdog] Started (threshold=%.1fs poll=%.1fs restart_after=%d exit_on_stuck=%s locks=%s)",
            self._stuck_threshold_s,
            self._poll_interval_s,
            self._restart_threshold,
            self._exit_on_stuck,
            list(self._monitored_locks.keys()),
        )

    def stop(self) -> None:
        """Stop the watchdog background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_s + 1.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            try:
                self._check_locks()
            except Exception:
                logger.exception("[Watchdog] Unexpected error in lock check")
            try:
                self._check_grpc()
            except Exception:
                logger.exception("[Watchdog] Unexpected error in gRPC check")

    # ------------------------------------------------------------------
    # Lock monitoring
    # ------------------------------------------------------------------

    def _check_locks(self) -> None:
        for name, lock in list(self._monitored_locks.items()):
            duration = lock.held_duration()
            if duration is None:
                continue
            if duration >= self._stuck_threshold_s:
                tid = lock.holder_tid()
                logger.watchdog(  # type: ignore[attr-defined]
                    "[Watchdog] Lock '%s' held for %.1fs by tid=%s — sending interrupt",
                    name, duration, tid,
                )
                if tid is not None:
                    killed = raise_in_thread(tid)
                    if killed:
                        logger.watchdog(  # type: ignore[attr-defined]
                            "[Watchdog] Interrupt delivered to tid=%s (lock '%s' will be released by finally/with)",
                            tid, name,
                        )
                    else:
                        logger.watchdog(  # type: ignore[attr-defined]
                            "[Watchdog] Could not deliver interrupt to tid=%s — thread may have already exited",
                            tid,
                        )

    # ------------------------------------------------------------------
    # gRPC monitoring
    # ------------------------------------------------------------------

    def _format_in_flight(self, details: list) -> str:
        if not details:
            return "[]"
        return "[" + ", ".join(
            f"id={d['rpc_id']} method={d['method']} age={d['age_s']:.1f}s"
            for d in details
        ) + "]"

    def _check_grpc(self) -> None:
        snap = self.rpc_state.snapshot(details_limit=self._details_limit)
        in_flight_str = self._format_in_flight(snap["in_flight_details"])

        if snap["unhealthy"]:
            self._unhealthy_count += 1
            self.rpc_state.record_unhealthy()

            logger.watchdog(  # type: ignore[attr-defined]
                "[Watchdog] gRPC unhealthy #%d: in_flight=%d oldest=%.1fs method=%s threshold=%.1fs | %s",
                self._unhealthy_count,
                snap["in_flight"],
                snap["oldest_age_s"],
                snap["oldest_method"],
                self._stuck_threshold_s,
                in_flight_str,
            )

            if self._exit_on_stuck:
                logger.watchdog(  # type: ignore[attr-defined]
                    "[Watchdog] GRPC_WATCHDOG_EXIT_ON_STUCK=1 — calling os._exit(1)"
                )
                os._exit(1)

            if self._unhealthy_count >= self._restart_threshold:
                logger.watchdog(  # type: ignore[attr-defined]
                    "[Watchdog] Restart threshold reached (%d/%d) — requesting server restart",
                    self._unhealthy_count, self._restart_threshold,
                )
                self.server_manager.request_restart()

        else:
            if self._unhealthy_count > 0:
                logger.watchdog(  # type: ignore[attr-defined]
                    "[Watchdog] gRPC recovered after %d unhealthy checks", self._unhealthy_count
                )
            self._unhealthy_count = 0
            self.rpc_state.record_healthy()
            logger.debug(
                "[Watchdog] gRPC healthy: in_flight=%d oldest=%.1fs | %s",
                snap["in_flight"], snap["oldest_age_s"], in_flight_str,
            )
