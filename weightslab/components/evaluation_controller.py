"""EvaluationController: thread-safe orchestration of evaluation-mode passes.

The evaluation mode lets the user request a full inference pass over a
specific data split (optionally filtered by tags) while training is paused.
Results are stored as evaluation *markers* in the signal history (using a
modified experiment hash with a ``_N`` suffix) so the UI can render them
as individual points rather than part of a training curve.

Flow
----
1. UI right-clicks the Evaluation button → selects split + optional tags
   → gRPC ``TriggerEvaluation`` RPC fires, calls
   ``eval_controller.request_evaluation()``.
2. The training loop calls ``wl.run_pending_evaluation(...)`` at the start
   of every iteration (before ``train()``).
   * If the controller is in ``requested`` state, evaluation runs.
   * If training was paused, the controller wakes up the pause event,
     the loop processes the evaluation, then re-pauses automatically.
3. ``run_pending_evaluation()`` iterates the target loader (shuffle=False,
   tag-filtered if needed), accumulates all signals through the logger's
   evaluation-mode buffer, then calls ``signal_logger.stop_evaluation_mode()``
   which stores final-average markers with hash ``base_hash_N``.
4. UI polls ``GetEvaluationStatus`` for live progress, and
   ``GetLatestLoggerData`` to receive the new markers.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Status literals
_STATUS_IDLE = "idle"
_STATUS_REQUESTED = "requested"
_STATUS_RUNNING = "running"
_STATUS_DONE = "done"
_STATUS_ERROR = "error"


class EvaluationController:
    """Thread-safe state machine for evaluation-mode passes.

    All public methods are safe to call from any thread (gRPC handlers,
    training thread, background threads).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request: Optional[Dict[str, Any]] = None
        self._status: str = _STATUS_IDLE
        self._progress: Dict[str, Any] = {"current": 0, "total": 0, "message": ""}
        self._result: Optional[Dict[str, float]] = None
        self._error: str = ""
        # Notification queue – drained by gRPC GetLatestLoggerData via special
        # LoggerDataPoints with metric_name == "__eval_notification__"
        self._notification_queue: List[Dict[str, Any]] = []
        # Event used to wake the training thread from wait_if_paused()
        self._wake_event = threading.Event()

    # ------------------------------------------------------------------
    # Called from gRPC handler (TriggerEvaluation)
    # ------------------------------------------------------------------
    def request_evaluation(
        self,
        split_name: str,
        tags: List[str],
        use_full_set: bool,
    ) -> bool:
        """Register an evaluation request.

        Returns True if accepted, False if an evaluation is already running.
        Also wakes the training thread if it is currently blocked in
        ``PauseController.wait_if_paused()``.
        """
        with self._lock:
            if self._status == _STATUS_RUNNING:
                logger.warning(
                    "[EvalController] Evaluation already running – ignoring new request"
                )
                return False

            # Lazily import to avoid circular imports
            from weightslab.components.global_monitoring import pause_controller
            was_paused = pause_controller.is_paused()

            self._request = {
                "split_name": split_name,
                "tags": list(tags),
                "use_full_set": use_full_set,
                "was_paused": was_paused,
            }
            self._status = _STATUS_REQUESTED
            self._progress = {"current": 0, "total": 0, "message": "Evaluation pending…"}
            self._result = None
            self._error = ""

            tag_info = f" [tags: {tags}]" if tags and not use_full_set else ""
            self._notification_queue.append({
                "type": "eval_started",
                "message": f"Evaluation on '{split_name}' started{tag_info}",
                "split_name": split_name,
                "timestamp": time.time(),
            })

            logger.info(
                "[EvalController] Evaluation requested: split=%s tags=%s full_set=%s was_paused=%s",
                split_name, tags, use_full_set, was_paused,
            )

            # Wake the training thread so it can call run_pending_evaluation()
            self._wake_event.set()
            if was_paused:
                pause_controller._event.set()   # Temporarily unblock wait_if_paused

        return True

    # ------------------------------------------------------------------
    # Called from the training thread (run_pending_evaluation)
    # ------------------------------------------------------------------
    def consume_request(self) -> Optional[Dict[str, Any]]:
        """Return the pending evaluation config and move to RUNNING state.

        Returns None if no evaluation is pending.
        """
        with self._lock:
            if self._status == _STATUS_REQUESTED and self._request is not None:
                req = dict(self._request)
                self._status = _STATUS_RUNNING
                self._wake_event.clear()
                return req
        return None

    def is_pending(self) -> bool:
        """True if an evaluation has been requested but not yet started."""
        with self._lock:
            return self._status == _STATUS_REQUESTED

    def is_running(self) -> bool:
        with self._lock:
            return self._status == _STATUS_RUNNING

    # ------------------------------------------------------------------
    # Progress reporting (called from run_pending_evaluation loop)
    # ------------------------------------------------------------------
    def report_progress(self, current: int, total: int, message: str = "") -> None:
        with self._lock:
            self._progress = {"current": current, "total": total, "message": message}

    # ------------------------------------------------------------------
    # Completion / error (called from run_pending_evaluation)
    # ------------------------------------------------------------------
    def mark_done(self, result: Optional[Dict[str, float]] = None) -> None:
        with self._lock:
            self._status = _STATUS_DONE
            self._result = result or {}
            self._notification_queue.append({
                "type": "eval_done",
                "message": "Evaluation completed",
                "result": self._result,
                "timestamp": time.time(),
            })
            logger.info("[EvalController] Evaluation done: %s", result)

    def mark_error(self, error: str) -> None:
        with self._lock:
            self._status = _STATUS_ERROR
            self._error = error
            self._notification_queue.append({
                "type": "eval_error",
                "message": f"Evaluation failed: {error}",
                "timestamp": time.time(),
            })
            logger.error("[EvalController] Evaluation error: %s", error)

    def reset(self) -> None:
        """Return to idle state (e.g. after the UI acknowledges completion)."""
        with self._lock:
            self._status = _STATUS_IDLE
            self._request = None
            self._progress = {"current": 0, "total": 0, "message": ""}
            self._result = None
            self._error = ""
            self._wake_event.clear()

    # ------------------------------------------------------------------
    # Status polling (called from GetEvaluationStatus RPC)
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "progress": dict(self._progress),
                "result": dict(self._result) if self._result is not None else None,
                "error": self._error,
                "split_name": (self._request or {}).get("split_name", ""),
            }

    def pop_notifications(self) -> List[Dict[str, Any]]:
        """Drain the notification queue (for embedding in LoggerDataPoints)."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
            return notifs


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
eval_controller = EvaluationController()
