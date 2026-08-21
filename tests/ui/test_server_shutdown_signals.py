"""Tests for weightslab/ui/server.py's termination-handler coverage.

Problem: `weightslab start` only ever caught Ctrl+C (SIGINT, which Python
turns into a catchable KeyboardInterrupt by default). Closing the terminal
window or a bare `kill <pid>` delivers a DIFFERENT signal/event
(SIGTERM/SIGHUP on POSIX, CTRL_CLOSE_EVENT on Windows) that Python does NOT
convert into a Python-level exception on its own -- so neither of those
ever ran this server's cleanup (_run_shutdown_cleanup: stopping tracked
detached processes, its own OpenCode/Jupyter children, /loop jobs),
confirmed live against real platform behavior. _install_termination_handlers
closes that gap.
"""

import os
import signal
import threading
import time
import unittest
from unittest.mock import patch

from weightslab.ui import server as ui_server


class TestRunShutdownCleanup(unittest.TestCase):
    def test_calls_all_four_shutdown_methods(self):
        with patch.object(ui_server._tracked_processes, "shutdown") as tp, \
                patch.object(ui_server._opencode_session, "shutdown") as oc, \
                patch.object(ui_server._loop_registry, "shutdown") as lr, \
                patch.object(ui_server._jupyter_session, "shutdown") as js:
            ui_server._run_shutdown_cleanup()
        tp.assert_called_once()
        oc.assert_called_once()
        lr.assert_called_once()
        js.assert_called_once()


class TestRaiseKeyboardInterrupt(unittest.TestCase):
    def test_raises_keyboardinterrupt(self):
        # Exactly how Python's own signal machinery would invoke this: a
        # (signum, frame) callback. Called directly, not via a real signal,
        # so this passes identically on every platform.
        with self.assertRaises(KeyboardInterrupt):
            ui_server._raise_keyboard_interrupt(signal.SIGTERM, None)


class TestOnWindowsCtrlEvent(unittest.TestCase):
    """Pure logic, no ctypes/real Windows API involved -- safe to run on
    any platform, unlike _install_windows_console_handler itself below."""

    def test_terminating_events_run_cleanup_and_report_handled(self):
        for ctrl_type in (2, 5, 6):  # CLOSE, LOGOFF, SHUTDOWN
            with patch.object(ui_server, "_run_shutdown_cleanup") as cleanup_mock:
                result = ui_server._on_windows_ctrl_event(ctrl_type)
            cleanup_mock.assert_called_once()
            self.assertTrue(result)

    def test_ctrl_c_and_ctrl_break_are_left_alone(self):
        # Python's own signal module already turns these into SIGINT/
        # SIGBREAK -- this handler must not double-handle them.
        for ctrl_type in (0, 1):
            with patch.object(ui_server, "_run_shutdown_cleanup") as cleanup_mock:
                result = ui_server._on_windows_ctrl_event(ctrl_type)
            cleanup_mock.assert_not_called()
            self.assertFalse(result)

    def test_unknown_event_is_left_alone(self):
        with patch.object(ui_server, "_run_shutdown_cleanup") as cleanup_mock:
            result = ui_server._on_windows_ctrl_event(99)
        cleanup_mock.assert_not_called()
        self.assertFalse(result)


@unittest.skipUnless(os.name == "nt", "SetConsoleCtrlHandler only exists on Windows")
class TestInstallWindowsConsoleHandler(unittest.TestCase):
    def test_registers_a_real_handler_without_raising(self):
        ui_server._install_windows_console_handler()
        self.assertIsNotNone(ui_server._console_ctrl_handler_ref)


class TestInstallTerminationHandlers(unittest.TestCase):
    def setUp(self):
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)
        self._orig_sighup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None

    def tearDown(self):
        signal.signal(signal.SIGTERM, self._orig_sigterm)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._orig_sighup)

    def test_registers_sigterm_to_raise_keyboardinterrupt(self):
        ui_server._install_termination_handlers()
        self.assertIs(signal.getsignal(signal.SIGTERM), ui_server._raise_keyboard_interrupt)

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "SIGHUP does not exist on this platform")
    def test_registers_sighup_to_raise_keyboardinterrupt(self):
        ui_server._install_termination_handlers()
        self.assertIs(signal.getsignal(signal.SIGHUP), ui_server._raise_keyboard_interrupt)

    def test_installs_the_windows_console_handler_only_on_windows(self):
        with patch.object(ui_server, "_install_windows_console_handler") as install_mock, \
                patch.object(ui_server.os, "name", "nt"):
            ui_server._install_termination_handlers()
        install_mock.assert_called_once()

    def test_skips_the_windows_console_handler_on_posix(self):
        with patch.object(ui_server, "_install_windows_console_handler") as install_mock, \
                patch.object(ui_server.os, "name", "posix"):
            ui_server._install_termination_handlers()
        install_mock.assert_not_called()


@unittest.skipIf(
    os.name == "nt",
    "os.kill(pid, SIGTERM) maps to TerminateProcess on Windows (a hard kill "
    "that bypasses any registered handler), so a self-SIGTERM isn't a safe "
    "way to test this there -- the Windows-specific path is covered by "
    "TestOnWindowsCtrlEvent/TestInstallWindowsConsoleHandler instead.",
)
class TestServeUiRealSigtermEndToEnd(unittest.TestCase):
    """The real thing: an actual SIGTERM delivered to this process while
    serve_ui(block=True) is blocking on the MAIN thread (signal handlers
    only ever run on the main thread, so this only proves anything when
    serve_forever() itself is there too -- exactly how `weightslab start`
    really calls it)."""

    def test_sigterm_interrupts_serve_forever_and_runs_cleanup(self):
        import tempfile

        tmp = tempfile.mkdtemp()
        cleanup_called = threading.Event()

        def _send_sigterm_shortly():
            time.sleep(0.4)
            os.kill(os.getpid(), signal.SIGTERM)

        with patch.object(ui_server, "_run_shutdown_cleanup", side_effect=cleanup_called.set):
            sender = threading.Thread(target=_send_sigterm_shortly, daemon=True)
            sender.start()
            ui_server.serve_ui(
                ui_host="127.0.0.1", ui_port=0,
                backend_host="localhost", backend_port=50051,
                open_browser=False, block=True,
                experiment_dir=tmp,
            )
            sender.join(timeout=5)

        self.assertTrue(cleanup_called.is_set())


if __name__ == "__main__":
    unittest.main()
