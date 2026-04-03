"""Lock monitoring for weightslab watchdog.

Provides:
  - MonitoredRLock  : drop-in RLock replacement that tracks the holder thread
                      and how long it has been held, so the watchdog can detect
                      and recover from stuck locks.
  - _WatchdogInterrupt : BaseException raised asynchronously in stuck threads.
  - raise_in_thread    : deliver _WatchdogInterrupt to any thread by id.

When the watchdog raises _WatchdogInterrupt in a thread that holds a
MonitoredRLock via ``with`` or a ``try/finally: release()``, Python's
context-manager / finally protocol guarantees the lock is released before
the exception propagates further.
"""

import ctypes
import threading
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Async exception type
# ---------------------------------------------------------------------------

class _WatchdogInterrupt(BaseException):
    """Raised asynchronously in stuck threads by the lock watchdog.

    Subclasses BaseException (not Exception) so ``except Exception:`` guards
    in user training loops do NOT accidentally swallow it.
    """


def raise_in_thread(tid: int, exc_type: type = _WatchdogInterrupt) -> bool:
    """Raise *exc_type* asynchronously in the thread identified by *tid*.

    Uses ``ctypes.pythonapi.PyThreadState_SetAsyncExc`` which delivers the
    exception at the next Python bytecode boundary.  Any active ``finally:``
    or ``with`` block in the target thread will execute before the exception
    propagates, so held locks are released cleanly.

    Returns True if the thread was found and the exception queued,
    False if the thread does not exist (already exited).
    """
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid),
        ctypes.py_object(exc_type),
    )
    if res == 0:
        return False  # thread not found
    if res > 1:
        # More than one state was modified — undo to be safe (shouldn't happen)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    return True


# ---------------------------------------------------------------------------
# MonitoredRLock
# ---------------------------------------------------------------------------

class MonitoredRLock:
    """RLock replacement that records the holder thread and acquisition time.

    API is fully compatible with ``threading.RLock``:
      - acquire(blocking=True, timeout=-1) -> bool
      - release()
      - __enter__ / __exit__ for ``with`` statements

    The watchdog polls ``held_duration()`` and ``holder_tid()`` to decide
    whether to kill the holder via ``raise_in_thread``.

    Re-entrancy is fully supported: the same thread can acquire multiple
    times.  ``_acquired_at`` records the time of the *first* acquisition and
    is cleared only when the lock becomes fully free (count reaches 0).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._meta = threading.Lock()   # guards the three fields below
        self._holder_tid: Optional[int] = None
        self._acquired_at: Optional[float] = None
        self._count: int = 0

    # ------------------------------------------------------------------
    # Core acquire / release
    # ------------------------------------------------------------------

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        acquired = self._lock.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            with self._meta:
                if self._count == 0:
                    self._holder_tid = threading.current_thread().ident
                    self._acquired_at = time.monotonic()
                self._count += 1
        return acquired

    def release(self) -> None:
        # Update tracking first, then release the underlying lock.
        # The try/finally ensures _lock.release() is called even if
        # _WatchdogInterrupt fires between the two statements.
        try:
            with self._meta:
                self._count -= 1
                if self._count <= 0:
                    self._count = 0
                    self._holder_tid = None
                    self._acquired_at = None
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MonitoredRLock":
        self.acquire()
        return self

    def __exit__(self, *_) -> bool:
        self.release()
        return False

    # ------------------------------------------------------------------
    # Watchdog inspection
    # ------------------------------------------------------------------

    def held_duration(self) -> Optional[float]:
        """Seconds the lock has been held by the current owner, or None if free."""
        with self._meta:
            if self._acquired_at is None:
                return None
            return time.monotonic() - self._acquired_at

    def holder_tid(self) -> Optional[int]:
        """Thread ident of the current owner, or None if free."""
        with self._meta:
            return self._holder_tid

    def is_held(self) -> bool:
        """True if any thread currently owns the lock."""
        with self._meta:
            return self._count > 0

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._meta:
            if self._holder_tid is None:
                return "<MonitoredRLock (free)>"
            return (
                f"<MonitoredRLock held by tid={self._holder_tid} "
                f"for {time.monotonic() - (self._acquired_at or 0):.1f}s "
                f"count={self._count}>"
            )
