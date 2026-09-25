"""Tests for the UI server's same-origin shared-model endpoints:

- GET  /agent-server/model  -- the model OpenCode's config names, or null.
- POST /agent-server/model  -- set it (global scope, confirmed by reading back).

Why they exist: the browser CAN call OpenCode directly, but only while
OpenCode's ``--cors`` allowlist contains the page's exact origin. A LAN
address, a tunnel hostname, or an ``opencode serve`` started by hand with no
``--cors`` all make that cross-origin PATCH fail its preflight, so a model
picked in the studio silently never reached OpenCode -- and therefore never
reached weightslab's backend, which reads that same field to choose the model
for its own queries. Proxying through this server is same-origin: no
preflight, no allowlist.

A fake OpenCode stands in for the real server, reproducing the two behaviours
confirmed live: PATCH /global/config sticks, PATCH /config answers 200 and
echoes the value back without changing what GET /config reports.
"""

import json
import threading
import time
import unittest
import unittest.mock
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from weightslab.ui import server as ui_server


class _FakeOpencode(BaseHTTPRequestHandler):
    """model lives in the class so every request sees the same value."""

    model = None
    workspace_patch_sticks = False
    reachable = True

    def log_message(self, *args):  # silence
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not type(self).reachable:
            self._json(500, {"error": "down"})
            return
        if self.path == "/config":
            self._json(200, {"model": type(self).model} if type(self).model else {})
            return
        self._json(404, {})

    def do_PATCH(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = json.loads(self.rfile.read(length).decode() or "{}") if length else {}
        model = body.get("model")
        if self.path == "/global/config":
            type(self).model = model
            self._json(200, {"model": model})
            return
        if self.path == "/config":
            # Answers 200 and echoes the value, but only *stores* it when the
            # server actually honours workspace scope -- the live one does not.
            if type(self).workspace_patch_sticks:
                type(self).model = model
            self._json(200, {"model": model})
            return
        self._json(404, {})


class TestSharedModelEndpoints(unittest.TestCase):
    def setUp(self):
        _FakeOpencode.model = None
        _FakeOpencode.workspace_patch_sticks = False
        _FakeOpencode.reachable = True

        self.oc = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpencode)
        self.oc_thread = threading.Thread(target=self.oc.serve_forever, daemon=True)
        self.oc_thread.start()
        oc_url = f"http://127.0.0.1:{self.oc.server_address[1]}"

        # The UI server resolves OpenCode from the running session, then from
        # OPENCODE_URL -- point it at the fake.
        self._prev_url = ui_server.os.environ.get("OPENCODE_URL")
        ui_server.os.environ["OPENCODE_URL"] = oc_url

        self.httpd = ui_server.serve_ui(
            ui_host="127.0.0.1", ui_port=0,
            backend_host="localhost", backend_port=50051,
            open_browser=False, block=False,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        self.oc.shutdown()
        self.oc_thread.join(timeout=5)
        if self._prev_url is None:
            ui_server.os.environ.pop("OPENCODE_URL", None)
        else:
            ui_server.os.environ["OPENCODE_URL"] = self._prev_url

    # -- helpers ---------------------------------------------------------
    def _get(self):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/agent-server/model", timeout=5) as r:
            return json.loads(r.read().decode())

    def _post(self, model):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/agent-server/model", method="POST",
            data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode() or "{}")

    # -- tests -----------------------------------------------------------
    def test_get_reports_nothing_when_no_model_is_configured(self):
        self.assertEqual(self._get(), {"ok": True, "model": None})

    def test_get_reports_the_configured_model(self):
        _FakeOpencode.model = "opencode/big-pickle"
        self.assertEqual(self._get(), {"ok": True, "model": "opencode/big-pickle"})

    def test_get_ignores_a_malformed_model_field(self):
        _FakeOpencode.model = "not-a-provider-model-pair"
        self.assertEqual(self._get(), {"ok": True, "model": None})

    def test_post_writes_the_global_scope_and_confirms_it(self):
        status, payload = self._post("openrouter/openai/gpt-5-mini")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], "openrouter/openai/gpt-5-mini")
        self.assertEqual(payload["via"], "/global/config")
        # ...and it really is what OpenCode now reports.
        self.assertEqual(_FakeOpencode.model, "openrouter/openai/gpt-5-mini")
        self.assertEqual(self._get()["model"], "openrouter/openai/gpt-5-mini")

    def test_post_rejects_a_model_without_a_provider(self):
        status, payload = self._post("big-pickle")
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])

    def test_post_reports_failure_when_the_write_does_not_stick(self):
        # Both routes answer 200 but nothing is stored -- exactly the live
        # workspace-scope behaviour, generalised.
        class _EchoOnly(_FakeOpencode):
            pass

        def do_PATCH(self):  # noqa: N802 -- HTTP handler naming
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length:
                self.rfile.read(length)
            self._json(200, {"model": "whatever"})

        with unittest.mock.patch.object(_FakeOpencode, "do_PATCH", do_PATCH):
            status, payload = self._post("opencode/big-pickle")
        self.assertEqual(status, 200)
        self.assertFalse(payload["ok"])
        self.assertIn("did not accept", payload["error"])

    def test_get_says_so_when_opencode_is_unreachable(self):
        _FakeOpencode.reachable = False
        payload = self._get()
        self.assertFalse(payload["ok"])
        self.assertIsNone(payload["model"])
        self.assertIn("not reachable", payload["error"])


if __name__ == "__main__":
    unittest.main()
