"""Unit tests for weightslab.watchdog.lock_monitor.

Covers:
  - MonitoredRLock basic acquire / release tracking
  - held_duration() and holder_tid() accuracy
  - Re-entrant acquisition count tracking
  - raise_in_thread delivers _WatchdogInterrupt and the lock is released
  - Lock is free again after the interrupt-triggered finally block runs
"""

import threading
import time
import unittest

from weightslab.watchdog.lock_monitor import (
    MonitoredRLock,
    _WatchdogInterrupt,
    raise_in_thread,
)


class TestMonitoredRLockBasic(unittest.TestCase):

    def test_free_by_default(self):
        lock = MonitoredRLock()
        self.assertFalse(lock.is_held())
        self.assertIsNone(lock.held_duration())
        self.assertIsNone(lock.holder_tid())

    def test_acquire_sets_tracking(self):
        lock = MonitoredRLock()
        lock.acquire()
        try:
            self.assertTrue(lock.is_held())
            self.assertIsNotNone(lock.held_duration())
            self.assertGreaterEqual(lock.held_duration(), 0.0)
            self.assertEqual(lock.holder_tid(), threading.current_thread().ident)
        finally:
            lock.release()

    def test_release_clears_tracking(self):
        lock = MonitoredRLock()
        lock.acquire()
        lock.release()
        self.assertFalse(lock.is_held())
        self.assertIsNone(lock.held_duration())
        self.assertIsNone(lock.holder_tid())

    def test_context_manager(self):
        lock = MonitoredRLock()
        with lock:
            self.assertTrue(lock.is_held())
        self.assertFalse(lock.is_held())

    def test_held_duration_grows(self):
        lock = MonitoredRLock()
        lock.acquire()
        try:
            d1 = lock.held_duration()
            time.sleep(0.05)
            d2 = lock.held_duration()
            self.assertGreater(d2, d1)
        finally:
            lock.release()

    def test_timeout_acquire_returns_false(self):
        lock = MonitoredRLock()
        # Acquire from another thread, then try with short timeout
        barrier = threading.Event()
        holder_ready = threading.Event()

        def holder():
            lock.acquire()
            holder_ready.set()
            barrier.wait(timeout=2.0)
            lock.release()

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        holder_ready.wait(timeout=1.0)

        acquired = lock.acquire(timeout=0.05)
        self.assertFalse(acquired)
        barrier.set()
        t.join(timeout=1.0)


class TestMonitoredRLockReentrant(unittest.TestCase):

    def test_same_thread_can_reacquire(self):
        lock = MonitoredRLock()
        lock.acquire()
        lock.acquire()  # reentrant — must not deadlock
        try:
            self.assertTrue(lock.is_held())
        finally:
            lock.release()
            lock.release()
        self.assertFalse(lock.is_held())

    def test_acquired_at_set_on_first_only(self):
        lock = MonitoredRLock()
        lock.acquire()
        t1 = lock._acquired_at
        time.sleep(0.02)
        lock.acquire()
        t2 = lock._acquired_at
        self.assertEqual(t1, t2, "acquired_at should not change on re-entrant acquire")
        lock.release()
        lock.release()

    def test_free_only_after_all_releases(self):
        lock = MonitoredRLock()
        lock.acquire()
        lock.acquire()
        lock.release()
        self.assertTrue(lock.is_held(), "still held after one release of two")
        lock.release()
        self.assertFalse(lock.is_held())


class TestRaiseInThread(unittest.TestCase):

    def test_returns_false_for_nonexistent_thread(self):
        # Use a tid that is certainly not running
        result = raise_in_thread(999_999_999)
        self.assertFalse(result)

    def test_lock_released_when_interrupt_raised_in_holder(self):
        """Core guarantee: a stuck thread holding a MonitoredRLock releases it
        when _WatchdogInterrupt is delivered via raise_in_thread."""
        lock = MonitoredRLock()
        released = threading.Event()
        thread_started = threading.Event()
        tid_box = [None]

        def stuck_holder():
            lock.acquire()
            tid_box[0] = threading.current_thread().ident
            thread_started.set()
            try:
                # Short-sleep loop so async exceptions are delivered promptly
                # on Windows (PyThreadState_SetAsyncExc only fires at bytecode
                # boundaries, not inside a long blocking C-level sleep).
                for _ in range(1000):
                    time.sleep(0.02)
            except _WatchdogInterrupt:
                pass
            finally:
                lock.release()
                released.set()

        t = threading.Thread(target=stuck_holder, daemon=True)
        t.start()

        # Wait until the thread holds the lock
        thread_started.wait(timeout=2.0)
        self.assertTrue(lock.is_held())

        # Kill the stuck thread
        ok = raise_in_thread(tid_box[0])
        self.assertTrue(ok)

        # Lock must be released by the finally block
        released.wait(timeout=2.0)
        self.assertFalse(lock.is_held(), "Lock must be free after interrupt-triggered release")
        t.join(timeout=2.0)

    def test_lock_released_via_with_block(self):
        """Same guarantee when the holder uses 'with lock:' instead of try/finally."""
        lock = MonitoredRLock()
        released = threading.Event()
        thread_started = threading.Event()
        tid_box = [None]

        def stuck_holder():
            with lock:
                tid_box[0] = threading.current_thread().ident
                thread_started.set()
                try:
                    for _ in range(1000):
                        time.sleep(0.02)
                except _WatchdogInterrupt:
                    pass
            released.set()

        t = threading.Thread(target=stuck_holder, daemon=True)
        t.start()
        thread_started.wait(timeout=2.0)
        self.assertTrue(lock.is_held())

        raise_in_thread(tid_box[0])
        released.wait(timeout=2.0)
        self.assertFalse(lock.is_held())
        t.join(timeout=2.0)

    def test_other_thread_can_acquire_after_release(self):
        """After the stuck holder is interrupted, another thread must be able
        to acquire the lock without blocking."""
        lock = MonitoredRLock()
        thread_started = threading.Event()
        tid_box = [None]
        stuck_done = threading.Event()

        def stuck_holder():
            lock.acquire()
            tid_box[0] = threading.current_thread().ident
            thread_started.set()
            try:
                for _ in range(1000):
                    time.sleep(0.02)
            except _WatchdogInterrupt:
                pass
            finally:
                lock.release()
                stuck_done.set()

        t = threading.Thread(target=stuck_holder, daemon=True)
        t.start()
        thread_started.wait(timeout=2.0)

        raise_in_thread(tid_box[0])
        stuck_done.wait(timeout=2.0)

        # Now the lock should be acquirable immediately
        acquired = lock.acquire(timeout=0.5)
        self.assertTrue(acquired, "Lock must be acquirable after stuck thread is killed")
        lock.release()
        t.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
