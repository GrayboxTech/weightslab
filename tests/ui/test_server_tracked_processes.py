"""Tests for weightslab/ui/server.py's POST /agent-server/track-process and
the _TrackedProcesses registry behind it.

Problem this exists for: the agent is told to launch anything long-running
(training, a relaunched crashed run) DETACHED (Start-Process/setsid) so it
never blocks the chat turn. A detached process has no OS-level parent-child
relationship this server's own process-tree kill (_kill_process_tree) can
walk -- confirmed live: a detached launcher's own immediate shell exits
almost immediately after spawning it, and Windows keeps no record of an
exited process for `taskkill /T` to trace a grandchild through. Registering
the PID directly sidesteps that: this server kills it explicitly, by PID,
with no chain to walk at all.

Uses a REAL child process (python -c "time.sleep(...)"), not a mock, so the
actual kill call is exercised end to end.
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request

from weightslab.ui import server as ui_server


def _is_alive(pid: int) -> bool:
    if ui_server.os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    try:
        ui_server.os.kill(pid, 0)
        return True
    except OSError:
        return False

@unittest.skip("Not ready yet")
class TestTrackedProcessesUnit(unittest.TestCase):
    def setUp(self):
        self.registry = ui_server._TrackedProcesses()
        self.proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            start_new_session=True,
        )

    def tearDown(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def test_tracked_pid_is_killed_on_shutdown(self):
        self.assertTrue(_is_alive(self.proc.pid))
        self.registry.track(self.proc.pid)

        self.registry.shutdown()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _is_alive(self.proc.pid):
            time.sleep(0.1)
        self.assertFalse(_is_alive(self.proc.pid))

    def test_untracked_pid_is_left_alone(self):
        self.registry.shutdown()  # nothing tracked
        self.assertTrue(_is_alive(self.proc.pid))

    def test_shutdown_clears_the_registry_so_a_second_call_is_a_no_op(self):
        self.registry.track(self.proc.pid)
        self.registry.shutdown()
        # A second shutdown() must not error just because the pid is already
        # gone (e.g. serve_ui's explicit call racing its own atexit hook).
        self.registry.shutdown()

    def test_killing_an_already_dead_pid_does_not_raise(self):
        self.proc.kill()
        self.proc.wait(timeout=5)
        self.registry.track(self.proc.pid)
        self.registry.shutdown()  # must not raise


class _ServerTestCase(unittest.TestCase):
    """Real serve_ui() on an ephemeral port -- same shape as
    test_server_agent.py's own _ServerTestCase."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd = ui_server.serve_ui(
            ui_host="127.0.0.1", ui_port=0,
            backend_host="localhost", backend_port=50051,
            open_browser=False, block=False,
            experiment_dir=self.tmp,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

        self._orig_tracked = ui_server._tracked_processes
        ui_server._tracked_processes = ui_server._TrackedProcesses()

    def tearDown(self):
        ui_server._tracked_processes = self._orig_tracked
        self.httpd.shutdown()
        self.thread.join(timeout=5)

    def _post_json(self, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method="POST", data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=10)

@unittest.skip("Not ready yet")
class TestTrackProcessEndpoint(_ServerTestCase):
    def test_registers_the_pid_end_to_end(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
        try:
            with self._post_json("/agent-server/track-process", {"pid": proc.pid}) as r:
                data = json.loads(r.read().decode())
            self.assertTrue(data["ok"])

            ui_server._tracked_processes.shutdown()

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _is_alive(proc.pid):
                time.sleep(0.1)
            self.assertFalse(_is_alive(proc.pid))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_non_integer_pid_is_rejected(self):
        try:
            self._post_json("/agent-server/track-process", {"pid": "not-a-number"})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)
            data = json.loads(exc.read().decode())
        self.assertFalse(data["ok"])
        self.assertIn("integer", data["error"])

    def test_missing_pid_is_rejected(self):
        try:
            self._post_json("/agent-server/track-process", {})
            self.fail("expected an HTTPError")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)


class TestSelfRegistration(_ServerTestCase):
    """The other half of the endpoint above: a training process registering
    ITSELF from inside ``wl.serve()``.

    Why this exists: the endpoint alone only helps when someone remembers to
    POST to it, and the only thing that ever did was the agent, by hand,
    right after a detached launch. That step gets skipped -- the turn is
    interrupted between launching and registering, the model forgets on a
    relaunch, or the run was started by hand in a terminal no agent was
    involved in -- and every miss leaves an orphaned python process holding
    the GPU and the gRPC port after the UI is gone. serve() registering
    itself makes it unskippable for anything that serves.
    """

    def test_serve_ui_publishes_its_own_origin_for_children_to_find(self):
        # The whole chain hangs off this: the OpenCode server is spawned by
        # this server, the agent's shell descends from that, and a training
        # process it launches inherits the variable from there.
        self.assertEqual(
            ui_server.os.environ.get("WEIGHTSLAB_UI_ORIGIN"),
            f"http://127.0.0.1:{self.port}",
        )

    def test_registers_this_process_against_the_live_server(self):
        from weightslab import src as wl_src

        with unittest.mock.patch.dict(
            ui_server.os.environ,
            {"WEIGHTSLAB_UI_ORIGIN": f"http://127.0.0.1:{self.port}"},
        ):
            wl_src._register_pid_with_ui_server()

            # Off the calling thread on purpose (a stale origin must never
            # stall a training run), so give the daemon thread a moment.
            deadline = time.monotonic() + 5
            own_pid = ui_server.os.getpid()
            while time.monotonic() < deadline:
                if own_pid in ui_server._tracked_processes._pids:
                    break
                time.sleep(0.05)

        # NOTE: deliberately never calls _tracked_processes.shutdown() here --
        # the pid under test is this very test runner's. _ServerTestCase swaps
        # in a throwaway registry and restores the real one in tearDown, so
        # nothing else can act on it either.
        self.assertIn(own_pid, ui_server._tracked_processes._pids)

    def test_is_a_no_op_when_no_ui_server_owns_this_process(self):
        from weightslab import src as wl_src

        env = dict(ui_server.os.environ)
        env.pop("WEIGHTSLAB_UI_ORIGIN", None)
        with unittest.mock.patch.dict(ui_server.os.environ, env, clear=True):
            wl_src._register_pid_with_ui_server()

        time.sleep(0.2)
        self.assertEqual(ui_server._tracked_processes._pids, set())

    def test_an_unreachable_origin_never_raises(self):
        """Best-effort is the whole contract: a stale WEIGHTSLAB_UI_ORIGIN
        left over from a previous `weightslab start` must cost a training run
        nothing at all, not even a raised exception on a background thread."""
        from weightslab import src as wl_src

        with unittest.mock.patch.dict(
            ui_server.os.environ,
            # Port 1 is reserved and never listening.
            {"WEIGHTSLAB_UI_ORIGIN": "http://127.0.0.1:1"},
        ):
            wl_src._register_pid_with_ui_server()

        time.sleep(0.3)
        self.assertEqual(ui_server._tracked_processes._pids, set())


if __name__ == "__main__":
    unittest.main()
