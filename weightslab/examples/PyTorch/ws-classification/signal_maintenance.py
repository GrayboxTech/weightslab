"""Prototype: a dedicated thread that owns per-sample signal COMPUTE off the
training hot path.

The training thread calls ``submit(batch_ids, **tensors)`` — a cheap detach +
bounded-queue put — and returns immediately. This thread drains the queue, runs
the registered ``compute`` fn (all metrics + any derived/composite signals,
computed from the in-batch values so there is NO dataframe read), and hands the
results to the EXISTING async write path (``wl.save_signals`` -> buffer ->
flush thread). Training never touches the dataframe or its lock.

Backpressure: the queue is bounded; ``submit`` blocks when full so signals can't
fall arbitrarily behind training (correctness over throughput). ``maxsize=0`` =
unbounded (throughput over bounded memory).

This is the generalization of the DDP "buffer per-sample tensors + drain off hot
path" optimization, applied to the generic signal path. Prototype lives next to
the example; promote into the SDK (e.g. weightslab/signals/maintenance.py) once
validated.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)


def _as_list(v):
    """Coerce a per-sample result (tensor / ndarray / sequence) to a plain list."""
    if hasattr(v, "detach"):
        return v.detach().cpu().tolist()
    if hasattr(v, "tolist"):
        return v.tolist()
    return list(v)


class SignalMaintenanceThread:
    def __init__(self, compute, origin: str = "train", maxsize: int = 256):
        """
        Args:
            compute: callable(tensors: dict) -> dict[signal_name, per-sample seq].
                Receives the detached tensors submitted by the train loop and
                returns one value per sample for each signal (metrics + derived).
            origin: dataframe split the signals belong to.
            maxsize: bounded queue length (0 = unbounded).
        """
        self._compute = compute
        self._origin = origin
        self._q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="WL-SignalMaintenance", daemon=True)
        self.submitted = 0
        self.processed = 0
        self.max_backlog = 0
        self.compute_ms_sum = 0.0      # cumulative per-batch compute time
        self.compute_ms_max = 0.0
        self.compute_ms_last = 0.0

    def start(self) -> "SignalMaintenanceThread":
        self._thread.start()
        return self

    def submit(self, batch_ids, **tensors) -> None:
        """Hot-path call: detach (kept on-device) + enqueue. O(1), no df touch."""
        det = {k: (v.detach() if hasattr(v, "detach") else v)
               for k, v in tensors.items()}
        self._q.put((batch_ids, det))           # blocks if a bounded queue is full
        self.submitted += 1
        self.max_backlog = max(self.max_backlog, self._q.qsize())

    def _run(self) -> None:
        from weightslab import save_signals       # existing async enqueue -> buffer
        while not self._stop.is_set():
            try:
                batch_ids, tensors = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                _t = time.perf_counter()
                computed = self._compute(tensors)
                self.compute_ms_last = (time.perf_counter() - _t) * 1000
                self.compute_ms_sum += self.compute_ms_last
                self.compute_ms_max = max(self.compute_ms_max, self.compute_ms_last)
                signals = {k: _as_list(v) for k, v in computed.items()}
                save_signals(batch_ids=batch_ids, signals=signals, log=True)
                self.processed += 1
            except Exception:
                logger.exception("SignalMaintenance compute failed")
            finally:
                self._q.task_done()

    def stop(self, drain: bool = True, timeout: float = 5.0) -> None:
        if drain:
            self._q.join()                        # wait for the backlog to clear
        self._stop.set()
        self._thread.join(timeout=timeout)

    @property
    def backlog(self) -> int:
        return self._q.qsize()
