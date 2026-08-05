"""Tests for weightslab/ui/server.py's OpenCode-agent supervisor:

- POST /agent-server/start -- spawns (or reuses) a local OpenCode server rooted
  at the experiment directory, so the browser never has to run `opencode serve`
  by hand. Mirrors /local-notebook's "the browser can't spawn a process, so it
  asks us to" shape.
- GET /agent-server/status -- none / running / killed, for the composer's status
  line to poll.

CI has no real `opencode` binary, so every test replaces
`ui_server._resolve_opencode_argv` with a tiny stand-in HTTP server (started via
`python -c`) that serves /global/health the same way the real OpenCode server
does. That is the only thing `_OpencodeSession` depends on to decide the child
started successfully, so it exercises the real spawn/health-poll/status code
path without depending on Node or the OpenCode package being installed.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from weightslab.ui import server as ui_server

# A minimal stand-in for `opencode serve`: binds the --port it was given and
# answers /global/health like the real server does. Reads --port out of
# sys.argv positionally rather than assuming argv[0] is anything in particular,
# since "python -c <src> serve --hostname H --port N" hands the child an argv
# whose exact shape depends on the platform's python launcher.
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

# A stand-in that never becomes healthy -- models a child that starts but never
# binds (a bad flag, a crash loop) so ensure()'s timeout path is exercised.
_HANGING_ARGV = [sys.executable, "-c", "import time; time.sleep(60)"]


class TestOpencodeSessionUnit(unittest.TestCase):
    """Exercises _OpencodeSession directly -- no HTTP layer, no real OpenCode."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.session = ui_server._OpencodeSession()
        # Keep polling fast so a genuine failure test doesn't sit for 45s.
        self._timeout_patch = patch.object(ui_server, "_OPENCODE_START_TIMEOUT", 3.0)
        self._timeout_patch.start()

    def tearDown(self):
        self.session.shutdown()
        self._timeout_patch.stop()

    def test_starts_and_reports_running(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["reused"])
        self.assertEqual(result["workspace"], self.tmp)
        self.assertTrue(result["url"].startswith("http://127.0.0.1:"))

        status = self.session.status()
        self.assertEqual(status["state"], "running")
        self.assertEqual(status["workspace"], self.tmp)

    def test_second_call_reuses_the_same_process(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            first = self.session.ensure(self.tmp, "http://localhost:5173")
            second = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(first["url"], second["url"])

    def test_missing_binary_and_missing_npx_reports_a_clear_error(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=None):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertFalse(result["ok"])
        self.assertIn("opencode", result["error"].lower())

        status = self.session.status()
        self.assertEqual(status["state"], "none")

    def test_child_that_never_becomes_healthy_times_out_and_is_killed(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_HANGING_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertFalse(result["ok"])
        self.assertIn("did not come up", result["error"])

        # The timed-out child must actually be killed, not leaked.
        status = self.session.status()
        self.assertEqual(status["state"], "killed")

    def test_shutdown_stops_a_running_process(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"])
        process = self.session._process
        self.session.shutdown()
        self.assertIsNotNone(process.poll())  # exited


class TestCorsOriginVariants(unittest.TestCase):
    """The localhost <-> 127.0.0.1 expansion is the #1 way this feature goes
    silently wrong -- a mismatch here makes every request look like the agent
    server is simply not there."""

    def test_expands_localhost_to_127_0_0_1(self):
        variants = ui_server._cors_origin_variants("http://localhost:5173")
        self.assertIn("http://localhost:5173", variants)
        self.assertIn("http://127.0.0.1:5173", variants)

    def test_expands_127_0_0_1_to_localhost(self):
        variants = ui_server._cors_origin_variants("http://127.0.0.1:8080")
        self.assertIn("http://127.0.0.1:8080", variants)
        self.assertIn("http://localhost:8080", variants)

    def test_leaves_a_non_loopback_origin_alone(self):
        variants = ui_server._cors_origin_variants("https://weightslab.example.com")
        self.assertEqual(variants, ["https://weightslab.example.com"])

    def test_none_origin_yields_no_variants(self):
        self.assertEqual(ui_server._cors_origin_variants(None), [])


class _ServerTestCase(unittest.TestCase):
    """Spins up a real serve_ui() on 127.0.0.1:<ephemeral> per test, rooted at a
    fresh temp dir passed as experiment_dir -- same shape as
    test_server_experiment_reports.py."""

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

        # The module-level singleton is shared across the whole test process;
        # swap in a fresh one per test so a leftover child from another test
        # can't make this one see {reused: true} unexpectedly.
        self._orig_session = ui_server._opencode_session
        ui_server._opencode_session = ui_server._OpencodeSession()
        self._timeout_patch = patch.object(ui_server, "_OPENCODE_START_TIMEOUT", 3.0)
        self._timeout_patch.start()

    def tearDown(self):
        self._timeout_patch.stop()
        ui_server._opencode_session.shutdown()
        ui_server._opencode_session = self._orig_session
        self.httpd.shutdown()
        self.thread.join(timeout=5)

    def _get(self, path):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5)

    def _post(self, path, origin=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method="POST", data=b"",
        )
        if origin:
            req.add_header("Origin", origin)
        return urllib.request.urlopen(req, timeout=10)


class TestAgentServerEndpoint(_ServerTestCase):

    def test_status_is_none_before_anything_starts(self):
        with self._get("/agent-server/status") as r:
            data = json.loads(r.read().decode())
        self.assertEqual(data["state"], "none")

    def test_start_spawns_rooted_at_the_experiment_dir(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            with self._post("/agent-server/start", origin="http://localhost:5173") as r:
                data = json.loads(r.read().decode())
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["workspace"], self.tmp)

        with self._get("/agent-server/status") as r:
            status = json.loads(r.read().decode())
        self.assertEqual(status["state"], "running")

    def test_missing_opencode_and_npx_returns_a_clear_error_not_a_500_crash(self):
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=None):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._post("/agent-server/start", origin="http://localhost:5173")
        self.assertEqual(ctx.exception.code, 500)
        data = json.loads(ctx.exception.read().decode())
        self.assertFalse(data["ok"])
        self.assertIn("opencode", data["error"].lower())

    def test_falls_back_to_reconstructing_origin_from_host_header(self):
        # No Origin header at all (e.g. a same-origin fetch some browsers omit
        # it for) -- must not crash, and must still start the server.
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            with self._post("/agent-server/start") as r:
                data = json.loads(r.read().decode())
        self.assertTrue(data["ok"], data)


if __name__ == "__main__":
    unittest.main()
