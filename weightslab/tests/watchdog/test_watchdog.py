"""Unit tests for weightslab.watchdog.watchdog.WeighlabsWatchdog.

Covers:
  - Watchdog detects a stuck lock and delivers interrupt (lock released)
  - Watchdog detects stuck gRPC RPC and requests server restart
  - Watchdog does not flag a healthy state
  - Watchdog stop() terminates the background thread cleanly
  - Custom WATCHDOG log level is registered and usable
"""

import logging
import threading
import time
import unittest

from weightslab.watchdog.lock_monitor import MonitoredRLock, _WatchdogInterrupt
from weightslab.watchdog.grpc_watchdog import RpcWatchdogState, GrpcServerManager
from weightslab.watchdog.watchdog import WeighlabsWatchdog
from weightslab.watchdog.log_level import WATCHDOG


class TestWatchdogLogLevel(unittest.TestCase):

    def test_watchdog_level_value(self):
        self.assertEqual(WATCHDOG, 35)
        self.assertGreater(WATCHDOG, logging.WARNING)
        self.assertLess(WATCHDOG, logging.ERROR)

    def test_watchdog_level_name(self):
        self.assertEqual(logging.getLevelName(WATCHDOG), "WATCHDOG")

    def test_logger_has_watchdog_method(self):
        log = logging.getLogger("test.watchdog_method")
        self.assertTrue(callable(getattr(log, "watchdog", None)))

    def test_logger_watchdog_emits_at_correct_level(self):
        log = logging.getLogger("test.emit")
        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Capture()
        handler.setLevel(logging.DEBUG)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            log.watchdog("hello %s", "world")  # type: ignore[attr-defined]
        finally:
            log.removeHandler(handler)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].levelno, WATCHDOG)
        self.assertIn("hello", records[0].getMessage())


class TestWeighlabsWatchdogLockMonitoring(unittest.TestCase):

    def test_stuck_lock_triggers_interrupt_and_releases(self):
        """Watchdog must detect a lock held too long and release it via interrupt."""
        lock = MonitoredRLock()
        released = threading.Event()
        thread_started = threading.Event()

        def stuck_holder():
            lock.acquire()
            thread_started.set()
            try:
                # Short-sleep loop so async exceptions fire promptly on Windows
                # (PyThreadState_SetAsyncExc only fires at bytecode boundaries).
                for _ in range(1500):
                    time.sleep(0.02)
            except _WatchdogInterrupt:
                pass
            finally:
                lock.release()
                released.set()

        t = threading.Thread(target=stuck_holder, daemon=True)
        t.start()
        thread_started.wait(timeout=2.0)
        self.assertTrue(lock.is_held())

        # Watchdog with very short threshold to trigger quickly
        watchdog = WeighlabsWatchdog(
            stuck_threshold_s=0.05,
            poll_interval_s=0.05,
        )
        watchdog.register_lock("test_lock", lock)
        watchdog.start()

        # Lock must be released by the watchdog within a reasonable time
        released.wait(timeout=3.0)
        watchdog.stop()

        self.assertFalse(lock.is_held(), "Lock must be free after watchdog interrupt")
        t.join(timeout=2.0)

    def test_healthy_lock_not_interrupted(self):
        """A lock acquired and released quickly must not be interrupted."""
        lock = MonitoredRLock()
        interrupted = threading.Event()

        def quick_holder():
            lock.acquire()
            time.sleep(0.02)    # well below threshold
            lock.release()

        watchdog = WeighlabsWatchdog(
            stuck_threshold_s=5.0,  # high threshold — should not fire
            poll_interval_s=0.05,
        )
        watchdog.register_lock("safe_lock", lock)
        watchdog.start()

        t = threading.Thread(target=quick_holder, daemon=True)
        t.start()
        t.join(timeout=1.0)

        watchdog.stop()
        self.assertFalse(lock.is_held())
        self.assertFalse(interrupted.is_set())


class TestWeighlabsWatchdogGrpc(unittest.TestCase):

    def test_stuck_rpc_triggers_restart_request(self):
        watchdog = WeighlabsWatchdog(
            stuck_threshold_s=0.02,
            poll_interval_s=0.02,
            restart_threshold=1,
        )
        watchdog.start()

        rpc_id = watchdog.rpc_state.begin("/test/StuckMethod")
        # Let the watchdog fire
        time.sleep(0.2)

        watchdog.stop()
        watchdog.rpc_state.end(rpc_id)

        self.assertTrue(
            watchdog.server_manager.should_restart(),
            "Watchdog must request restart after stuck RPC",
        )

    def test_healthy_rpc_does_not_trigger_restart(self):
        watchdog = WeighlabsWatchdog(
            stuck_threshold_s=5.0,  # high threshold
            poll_interval_s=0.05,
            restart_threshold=1,
        )
        watchdog.start()

        rpc_id = watchdog.rpc_state.begin("/test/FastMethod")
        time.sleep(0.02)            # much less than threshold
        watchdog.rpc_state.end(rpc_id)
        time.sleep(0.1)             # let watchdog tick

        watchdog.stop()
        self.assertFalse(watchdog.server_manager.should_restart())

    def test_unhealthy_count_resets_on_recovery(self):
        watchdog = WeighlabsWatchdog(
            stuck_threshold_s=0.02,
            poll_interval_s=0.02,
            restart_threshold=10,  # high — won't restart
        )
        watchdog.start()

        rpc_id = watchdog.rpc_state.begin("/test/SlowThenFast")
        time.sleep(0.12)            # trigger unhealthy
        watchdog.rpc_state.end(rpc_id)
        time.sleep(0.15)            # let watchdog see healthy state

        watchdog.stop()
        self.assertEqual(watchdog._unhealthy_count, 0, "unhealthy_count must reset to 0 on recovery")

    def test_watchdog_stop_joins_thread(self):
        watchdog = WeighlabsWatchdog(poll_interval_s=0.05)
        watchdog.start()
        self.assertIsNotNone(watchdog._thread)
        self.assertTrue(watchdog._thread.is_alive())

        watchdog.stop()
        # Thread should terminate shortly after stop()
        self.assertFalse(watchdog._thread.is_alive(), "Watchdog thread must exit after stop()")


class TestGrpcServerManager(unittest.TestCase):

    def test_restart_flag_cycle(self):
        mgr = GrpcServerManager()
        self.assertFalse(mgr.should_restart())
        mgr.request_restart()
        self.assertTrue(mgr.should_restart())
        mgr.clear_restart_request()
        self.assertFalse(mgr.should_restart())


class TestRpcWatchdogState(unittest.TestCase):

    def test_healthy_snapshot(self):
        state = RpcWatchdogState(stuck_threshold_s=60.0)
        snap = state.snapshot()
        self.assertEqual(snap["in_flight"], 0)
        self.assertFalse(snap["unhealthy"])

    def test_unhealthy_when_rpc_exceeds_threshold(self):
        state = RpcWatchdogState(stuck_threshold_s=0.02)
        rpc_id = state.begin("/test/Slow")
        time.sleep(0.05)
        snap = state.snapshot()
        state.end(rpc_id)
        self.assertTrue(snap["unhealthy"])
        self.assertGreater(snap["oldest_age_s"], 0.02)

    def test_in_flight_cleared_for_restart(self):
        state = RpcWatchdogState(stuck_threshold_s=60.0)
        state.begin("/a")
        state.begin("/b")
        cleared = state.clear_for_restart()
        self.assertEqual(cleared, 2)
        snap = state.snapshot()
        self.assertEqual(snap["in_flight"], 0)


if __name__ == "__main__":
    unittest.main()
