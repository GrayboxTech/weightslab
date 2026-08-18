"""Tests for weightslab/opencode_process.py -- the cross-process discovery/
spawn handshake that lets the backend SDK agent (agent.py's
DataManipulationAgent) and the UI server's _OpencodeSession (server.py,
backing the browser landing-page chat and /loop jobs) converge on ONE
OpenCode server for a given workspace directory, regardless of which one
needs it first.

CI has no real `opencode` binary, so the real-spawn tests point
resolve_opencode_argv at a tiny stand-in HTTP server (started via
`python -c`, same pattern tests/ui/test_server_agent.py already uses) that
answers /global/health the way the real OpenCode server does -- everything
else here is exercised via that real subprocess + real lock-file I/O in a
temp directory, not mocked away.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from weightslab import opencode_process


_FAKE_OPENCODE_SRC = r"""
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer

def _port():
    for i, a in enumerate(sys.argv):
        if a == "--port":
            return int(sys.argv[i + 1])
    return 4096

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        if self.path == "/global/health":
            body = json.dumps({"version": "0.0.0-fake"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

HTTPServer(("127.0.0.1", _port()), H).serve_forever()
"""

_FAKE_ARGV = [sys.executable, "-c", _FAKE_OPENCODE_SRC]


class TestLockFileRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_absent_lock_reads_as_none(self):
        self.assertIsNone(opencode_process.read_lock(self.tmp))

    def test_write_then_read_roundtrips(self):
        opencode_process.write_lock(self.tmp, "http://127.0.0.1:9999", pid=1234)
        lock = opencode_process.read_lock(self.tmp)
        self.assertEqual(lock["url"], "http://127.0.0.1:9999")
        self.assertEqual(lock["pid"], 1234)

    def test_malformed_lock_file_reads_as_none_not_an_exception(self):
        with open(opencode_process.lock_path(self.tmp), "w") as f:
            f.write("{not json")
        self.assertIsNone(opencode_process.read_lock(self.tmp))

    def test_lock_file_with_no_url_reads_as_none(self):
        with open(opencode_process.lock_path(self.tmp), "w") as f:
            f.write('{"pid": 1}')
        self.assertIsNone(opencode_process.read_lock(self.tmp))


class TestResolveOrSpawnUnit(unittest.TestCase):
    """The precedence chain (env > lockfile > spawn), mocked so it runs in
    milliseconds -- the real-subprocess path is covered separately below."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ.pop("OPENCODE_URL", None)

    def test_healthy_env_var_wins_outright(self):
        os.environ["OPENCODE_URL"] = "http://127.0.0.1:1111"
        try:
            with patch.object(opencode_process, "opencode_healthy", return_value=True):
                result = opencode_process.resolve_or_spawn_opencode(self.tmp)
        finally:
            del os.environ["OPENCODE_URL"]
        self.assertEqual(result, {"ok": True, "url": "http://127.0.0.1:1111", "source": "env"})

    def test_unhealthy_env_var_is_ignored_in_favor_of_the_lockfile(self):
        os.environ["OPENCODE_URL"] = "http://127.0.0.1:1111"
        opencode_process.write_lock(self.tmp, "http://127.0.0.1:2222")
        try:
            with patch.object(opencode_process, "opencode_healthy",
                               side_effect=lambda url, timeout=1.5: url == "http://127.0.0.1:2222"):
                result = opencode_process.resolve_or_spawn_opencode(self.tmp)
        finally:
            del os.environ["OPENCODE_URL"]
        self.assertEqual(result, {"ok": True, "url": "http://127.0.0.1:2222", "source": "lockfile"})

    def test_healthy_lockfile_is_adopted_without_spawning(self):
        opencode_process.write_lock(self.tmp, "http://127.0.0.1:3333")
        with patch.object(opencode_process, "opencode_healthy", return_value=True), \
                patch("subprocess.Popen") as popen:
            result = opencode_process.resolve_or_spawn_opencode(self.tmp)
        popen.assert_not_called()
        self.assertEqual(result, {"ok": True, "url": "http://127.0.0.1:3333", "source": "lockfile"})

    def test_stale_lockfile_is_ignored_and_a_fresh_server_is_spawned(self):
        # The process the file names is gone -- health check on ITS url fails,
        # but the newly-spawned one's succeeds.
        opencode_process.write_lock(self.tmp, "http://127.0.0.1:4444")
        with patch.object(opencode_process, "opencode_healthy",
                           side_effect=lambda url, timeout=1.5: url != "http://127.0.0.1:4444"), \
                patch.object(opencode_process, "resolve_opencode_argv", return_value=_FAKE_ARGV), \
                patch.object(opencode_process, "free_port", return_value=15000):
            result = opencode_process.resolve_or_spawn_opencode(self.tmp)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["source"], "spawned")
        # The stale entry was overwritten with the freshly-spawned server.
        self.assertEqual(opencode_process.read_lock(self.tmp)["url"], result["url"])

    def test_no_opencode_or_npx_available_reports_a_clear_error(self):
        with patch.object(opencode_process, "resolve_opencode_argv", return_value=None):
            result = opencode_process.resolve_or_spawn_opencode(self.tmp)
        self.assertFalse(result["ok"])
        self.assertIn("opencode", result["error"])


class TestResolveOrSpawnRealSubprocess(unittest.TestCase):
    """Exercises the actual subprocess.Popen + health-poll + lock-file-write
    path against a real (fake-OpenCode) child process, not a mock."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ.pop("OPENCODE_URL", None)
        self._timeout_patch = patch.object(opencode_process, "OPENCODE_START_TIMEOUT", 5.0)
        self._timeout_patch.start()

    def tearDown(self):
        self._timeout_patch.stop()

    def test_first_caller_spawns_and_writes_the_lock_file(self):
        with patch.object(opencode_process, "resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = opencode_process.resolve_or_spawn_opencode(self.tmp, origin="http://localhost:5173")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["source"], "spawned")
        lock = opencode_process.read_lock(self.tmp)
        self.assertEqual(lock["url"], result["url"])
        self.assertTrue(opencode_process.opencode_healthy(result["url"]))

    def test_second_caller_for_the_same_workspace_adopts_instead_of_spawning(self):
        """The actual cross-process handshake this feature exists for: call
        it once (simulating whichever side -- backend agent or UI server --
        happens to start first), then again for the SAME workspace_dir
        (simulating the other side starting later) and confirm the second
        call adopts the first call's server rather than spawning a second
        one."""
        with patch.object(opencode_process, "resolve_opencode_argv", return_value=_FAKE_ARGV):
            first = opencode_process.resolve_or_spawn_opencode(self.tmp)
            self.assertEqual(first["source"], "spawned")

            with patch("subprocess.Popen") as popen:
                second = opencode_process.resolve_or_spawn_opencode(self.tmp)
            popen.assert_not_called()

        self.assertEqual(second["source"], "lockfile")
        self.assertEqual(second["url"], first["url"])

    def test_never_becoming_healthy_times_out_and_reports_an_error(self):
        hanging_argv = [sys.executable, "-c", "import time; time.sleep(60)"]
        with patch.object(opencode_process, "resolve_opencode_argv", return_value=hanging_argv):
            result = opencode_process.resolve_or_spawn_opencode(self.tmp)
        self.assertFalse(result["ok"])
        self.assertIn("did not come up", result["error"])
        self.assertIsNone(opencode_process.read_lock(self.tmp))


if __name__ == "__main__":
    unittest.main()
