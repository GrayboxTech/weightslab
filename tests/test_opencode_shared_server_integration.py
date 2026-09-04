"""End-to-end proof that the backend SDK agent (OpenCodeChat, reached via the
gRPC query bar) and the UI server's _OpencodeSession (backing the browser
landing-page chat and /loop jobs) converge on ONE OpenCode server for a
shared workspace directory -- regardless of which one needs a server first.

Each side's own half of this handshake already has focused unit coverage
(tests/test_opencode_process.py for the shared resolve_or_spawn_opencode
precedence chain; tests/ui/test_server_agent.py and
tests/trainer/services/test_opencode_chat.py for each side's own call into
it). This file instead drives BOTH real classes together against one real
(fake-OpenCode) subprocess, proving the actual scenario end to end rather
than trusting that the separately-tested pieces compose correctly.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_opencode_process import stop_workspace_server
from weightslab import opencode_process
from weightslab.trainer.services.agent.opencode_chat import OpenCodeChat
from weightslab.ui import server as ui_server


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

# The real opencode_healthy, forced to treat OpenCode's literal default
# address as dead regardless of the actual machine's state -- a real,
# unrelated `opencode serve` left running on the default port (easy to
# accumulate: nothing in this codebase kills one automatically once
# started, confirmed live more than once during development) would
# otherwise make _ensure_reachable's "already healthy, leave it alone"
# branch fire for real here, which is correct behaviour but defeats the
# point of THIS test -- it wants to force the "was dead, needed
# resolving" path deterministically. Every other address still gets a
# real health check.
_real_opencode_healthy = opencode_process.opencode_healthy


def _healthy_except_bare_default(url: str, timeout: float = 1.5) -> bool:
    if url.rstrip("/") == "http://127.0.0.1:4096":
        return False
    return _real_opencode_healthy(url, timeout=timeout)



def setUpModule():
    """Force fresh spawns onto an OS-assigned port for this file only.

    Both spawn paths now ASK for opencode_process.DEFAULT_OPENCODE_PORT (4096)
    before falling back to a random one, and 4096 is the very address this file
    patches ``opencode_healthy`` to report dead (see
    _healthy_except_bare_default). Left alone, a spawn here lands on 4096, the
    patch declares the server it just started dead, and the health poll times
    out -- so the test would only pass on a machine where something else
    already happened to hold 4096, which is precisely the accidental
    dependence on ambient machine state the patch exists to remove.

    Pinning the override to 0 (= "OS-assigned", the behaviour that predates
    the default) keeps "the bare default address" and "wherever a real spawn
    lands" as two distinct things, which is what these tests are about.
    """
    global _saved_port_env
    _saved_port_env = os.environ.get(opencode_process.PORT_ENV_VAR)
    os.environ[opencode_process.PORT_ENV_VAR] = "0"


def tearDownModule():
    if _saved_port_env is None:
        os.environ.pop(opencode_process.PORT_ENV_VAR, None)
    else:
        os.environ[opencode_process.PORT_ENV_VAR] = _saved_port_env


class TestBackendAgentStartsFirst(unittest.TestCase):
    """Order (a) from the user's own description: the backend SDK agent
    needs a server before `weightslab start` ever calls /agent-server/start
    for the same experiment directory."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(stop_workspace_server, self.tmp)

    def test_ui_server_adopts_the_backend_agents_server_instead_of_spawning(self):
        with patch("weightslab.opencode_process.opencode_healthy", side_effect=_healthy_except_bare_default):
            chat = OpenCodeChat("http://127.0.0.1:4096", workspace_dir=self.tmp, url_is_explicit=False)
            with patch("weightslab.opencode_process.resolve_opencode_argv", return_value=_FAKE_ARGV):
                chat._ensure_reachable()  # the backend agent spawns first
            backend_url = chat.base_url
            self.assertNotEqual(backend_url, "http://127.0.0.1:4096", "the dead default should have been replaced")

            session = ui_server._OpencodeSession()
            try:
                with patch.object(ui_server, "_resolve_opencode_argv") as argv_mock, \
                        patch.object(ui_server, "_opencode_healthy", side_effect=_healthy_except_bare_default):
                    result = session.ensure(self.tmp, "http://localhost:5173")
            finally:
                session.shutdown()

        argv_mock.assert_not_called()  # no second server spawned
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["url"], backend_url)
        self.assertEqual(result.get("adopted"), "lockfile")


class TestUiServerStartsFirst(unittest.TestCase):
    """Order (b): `weightslab start` (the UI server) needs a server first,
    and the backend SDK agent's first query comes along afterward."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(stop_workspace_server, self.tmp)

    def test_backend_agent_adopts_the_ui_servers_server_instead_of_spawning(self):
        with patch("weightslab.opencode_process.opencode_healthy", side_effect=_healthy_except_bare_default), \
                patch.object(ui_server, "_opencode_healthy", side_effect=_healthy_except_bare_default):
            session = ui_server._OpencodeSession()
            try:
                with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
                    started = session.ensure(self.tmp, "http://localhost:5173")
                self.assertTrue(started["ok"], started)

                chat = OpenCodeChat("http://127.0.0.1:4096", workspace_dir=self.tmp, url_is_explicit=False)
                with patch("weightslab.opencode_process.resolve_opencode_argv") as argv_mock:
                    chat._ensure_reachable()
            finally:
                session.shutdown()

        argv_mock.assert_not_called()  # no second server spawned
        self.assertEqual(chat.base_url, started["url"])


if __name__ == "__main__":
    unittest.main()
