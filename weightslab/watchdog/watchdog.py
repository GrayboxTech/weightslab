"""WeighlabsWatchdog — unified watchdog for locks and gRPC threads.

Combines:
  1. Lock monitoring  — polls MonitoredRLock instances, raises _WatchdogInterrupt
                        in the holder thread when the lock is held too long.
  2. gRPC monitoring  — detects stuck in-flight RPCs via RpcWatchdogState and
                        requests a server restart when the threshold is exceeded.
  3. Eval thread monitoring — checks that the evaluation worker thread is still
                        alive whenever eval_controller reports is_running() or
                        is_pending().  If the thread is dead the controller is
                        transitioned to error state automatically.  No timeout is
                        applied — evaluation may run for an arbitrarily long time.

Typical usage (inside grpc_serve):

    watchdog = WeighlabsWatchdog(
        stuck_threshold_s=60.0,
        poll_interval_s=5.0,
        restart_threshold=3,
        exit_on_stuck=False,
    )
    watchdog.register_lock("weightslab_rlock", weightslab_rlock)
    watchdog.register_eval_monitor(
        get_controller=lambda: eval_controller,
        get_thread=lambda: _EVAL_WORKER_THREAD,
    )
    watchdog.start()
    # watchdog.rpc_state  → pass to RpcTimingAndWatchdogInterceptor
    # watchdog.server_manager → used by serving_thread_callback
"""

import os
import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

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
        self._poll_interval_s = poll_interval_s
        self._restart_threshold = restart_threshold
        self._exit_on_stuck = exit_on_stuck
        self._details_limit = details_limit

        # gRPC watchdog state (shared with RpcTimingAndWatchdogInterceptor)
        self.rpc_state = RpcWatchdogState(stuck_threshold_s=stuck_threshold_s)
        self.server_manager = GrpcServerManager()

        # Lock monitoring
        self._monitored_locks: Dict[str, MonitoredRLock] = {}

        # Eval thread monitoring (no timeout — evaluation can run indefinitely)
        self._eval_monitors: List[Tuple[Callable, Callable]] = []

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

    def register_eval_monitor(
        self,
        get_controller: Callable,
        get_thread: Callable,
    ) -> None:
        """Register an evaluation controller/thread pair for liveness monitoring.

        The watchdog will call ``mark_error()`` on the controller when it reports
        ``is_running()`` or ``is_pending()`` but the worker thread is no longer
        alive.  **No timeout is applied** — evaluation is allowed to run for as
        long as needed.

        Args:
            get_controller: Zero-arg callable that returns the EvaluationController.
            get_thread:      Zero-arg callable that returns the current worker
                             ``threading.Thread`` (or ``None`` if not started yet).
        """
        self._eval_monitors.append((get_controller, get_thread))

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
            try:
                self._check_eval_threads()
            except Exception:
                logger.exception("[Watchdog] Unexpected error in eval thread check")

    # ------------------------------------------------------------------
    # Lock monitoring
    # ------------------------------------------------------------------

    def _check_locks(self) -> None:  # noqa: C901
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
    # Eval thread monitoring
    # ------------------------------------------------------------------

    def _check_eval_threads(self) -> None:
        """Detect a dead eval worker whose controller is still in running/pending state."""
        for get_controller, get_thread in self._eval_monitors:
            try:
                controller = get_controller()
                thread: Optional[threading.Thread] = get_thread()
            except Exception:
                logger.exception("[Watchdog] Failed to obtain eval monitor objects")
                continue

            if not (controller.is_running() or controller.is_pending()):
                continue  # nothing active — nothing to check

            if thread is not None and thread.is_alive():
                continue  # worker is alive — all good

            # Controller believes eval is active but the thread is dead or missing.
            status = controller.get_status() if hasattr(controller, "get_status") else "unknown"
            logger.watchdog(  # type: ignore[attr-defined]
                "[Watchdog] Eval controller is '%s' but worker thread is dead — marking error",
                status,
            )
            try:
                controller.mark_error("Evaluation worker terminated unexpectedly")
            except Exception:
                logger.exception("[Watchdog] Failed to mark eval controller error")

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
