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

    def test_ensure_drops_agents_md_into_a_fresh_workspace(self):
        # This test process runs from an actual repo checkout, so AGENTS.md
        # resolves for real via _repo_doc_path -- no mocking needed to prove
        # ensure() actually reaches the workspace with it.
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"], result)
        copied = os.path.join(self.tmp, "AGENTS.md")
        self.assertTrue(os.path.isfile(copied))
        with open(copied, encoding="utf-8") as fh:
            self.assertTrue(fh.read().strip())

    def test_ensure_never_overwrites_a_workspace_own_agents_md(self):
        # A workspace's own AGENTS.md might be the USER's project
        # instructions -- ensure() must never clobber it with ours.
        own_path = os.path.join(self.tmp, "AGENTS.md")
        with open(own_path, "w", encoding="utf-8") as fh:
            fh.write("this workspace's own instructions")
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"], result)
        with open(own_path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "this workspace's own instructions")

    def test_ensure_copy_is_best_effort_when_no_source_seed_files_exist(self):
        # No source to copy from (e.g. a stripped-down install) must not
        # fail ensure() outright -- the agent server should still start.
        with patch.object(ui_server.opencode_process, "_packaged_file", return_value=None):
            with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
                result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"], result)
        for filename in ui_server.opencode_process.WORKSPACE_SEED_FILES:
            self.assertFalse(os.path.isfile(os.path.join(self.tmp, filename)))

    def test_successful_spawn_writes_a_lock_file_for_this_workspace(self):
        # The other half of the cross-process handoff: the backend SDK agent
        # (agent.py's OpenCodeChat) discovers THIS server via the same file
        # -- see test_opencode_process.py's cross-process test for the full
        # round trip.
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"], result)
        lock = ui_server.opencode_process.read_lock(self.tmp)
        self.assertEqual(lock["url"], result["url"])

    def test_adopts_a_healthy_lockfile_instead_of_spawning_a_second_server(self):
        # Simulates order (a) from the tabbed-agent-window plan: the backend
        # SDK agent already published a server for this workspace before
        # `weightslab start` (this session) ever called ensure().
        other = ui_server._OpencodeSession()
        try:
            with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
                published = other.ensure(self.tmp, None)
            self.assertTrue(published["ok"], published)

            with patch.object(ui_server, "_resolve_opencode_argv") as argv_mock:
                result = self.session.ensure(self.tmp, "http://localhost:5173")
        finally:
            other.shutdown()

        argv_mock.assert_not_called()
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["url"], published["url"])
        self.assertEqual(result.get("adopted"), "lockfile")
        # Adopted, not spawned -- nothing of this session's own to kill.
        self.assertIsNone(self.session._process)

    def test_stale_lockfile_is_ignored_and_a_fresh_server_is_spawned(self):
        # The process a stale lock file names is long gone -- must fall
        # through to a normal spawn rather than failing or hanging.
        ui_server.opencode_process.write_lock(self.tmp, "http://127.0.0.1:1", pid=999999)
        with patch.object(ui_server, "_resolve_opencode_argv", return_value=_FAKE_ARGV):
            result = self.session.ensure(self.tmp, "http://localhost:5173")
        self.assertTrue(result["ok"], result)
        self.assertNotEqual(result["url"], "http://127.0.0.1:1")
        # The stale entry was overwritten with the newly-spawned server.
        self.assertEqual(ui_server.opencode_process.read_lock(self.tmp)["url"], result["url"])


class TestEnsureWorkspaceAgentFiles(unittest.TestCase):
    """ensure_workspace_agent_files directly -- no OpenCode process involved.

    Both seeded files matter for the same reason: OpenCode reads a project
    AGENTS.md and a project opencode.json out of the directory it is started
    in, which is this workspace. A workspace missing either one silently
    downgrades what the agent knows.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_copies_both_seed_files_from_the_installed_package_location(self):
        ui_server._ensure_workspace_agent_files(self.tmp)
        for filename in ui_server.opencode_process.WORKSPACE_SEED_FILES:
            target = os.path.join(self.tmp, filename)
            self.assertTrue(os.path.isfile(target), f"{filename} was not seeded")
            with open(target, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), ui_server._read_repo_doc(filename))

    def test_the_seeded_config_points_opencode_at_agents_md(self):
        """The whole reason opencode.json is seeded at all -- without the
        `instructions` entry it would be an empty config doing nothing."""
        ui_server._ensure_workspace_agent_files(self.tmp)
        with open(os.path.join(self.tmp, "opencode.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        self.assertIn("AGENTS.md", config["instructions"])
        self.assertEqual(config["$schema"], "https://opencode.ai/config.json")

    def test_is_a_no_op_when_the_workspace_already_has_one(self):
        for filename in ui_server.opencode_process.WORKSPACE_SEED_FILES:
            with open(os.path.join(self.tmp, filename), "w", encoding="utf-8") as fh:
                fh.write("mine")
        ui_server._ensure_workspace_agent_files(self.tmp)
        for filename in ui_server.opencode_process.WORKSPACE_SEED_FILES:
            with open(os.path.join(self.tmp, filename), encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "mine", f"{filename} was overwritten")

    def test_seeds_the_other_file_when_one_is_already_present(self):
        with open(os.path.join(self.tmp, "AGENTS.md"), "w", encoding="utf-8") as fh:
            fh.write("mine")
        ui_server._ensure_workspace_agent_files(self.tmp)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "opencode.json")))

    def test_does_nothing_when_no_source_is_found(self):
        with patch.object(ui_server.opencode_process, "_packaged_file", return_value=None):
            ui_server._ensure_workspace_agent_files(self.tmp)
        for filename in ui_server.opencode_process.WORKSPACE_SEED_FILES:
            self.assertFalse(os.path.isfile(os.path.join(self.tmp, filename)))


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


class TestAgentDocsEndpoint(_ServerTestCase):
    """GET /agent-server/docs[?example=<usecase>] -- AGENTS.md, plus
    optionally one matching PyTorch usecase example, for the landing chat's
    preset prompts to attach (see agentChat.ts's PRESET_PROMPTS). README.md
    was dropped from this endpoint on purpose -- AGENTS.md alone carries the
    weightslab integration pattern the presets need."""

    def test_returns_agents_md_when_present_in_a_repo_checkout(self):
        # This test process runs from an actual repo checkout, so it should
        # resolve via _read_repo_doc.
        with self._get("/agent-server/docs") as r:
            data = json.loads(r.read().decode())
        names = {f["name"] for f in data["files"]}
        self.assertEqual(names, {"AGENTS.md"})
        for f in data["files"]:
            self.assertTrue(f["content"].strip())

    def test_omits_the_doc_when_it_cannot_be_found_instead_of_erroring(self):
        with patch.object(ui_server, "_read_repo_doc", return_value=None):
            with self._get("/agent-server/docs") as r:
                data = json.loads(r.read().decode())
        self.assertEqual(data["files"], [])

    def test_example_query_param_additionally_attaches_that_usecases_main_py(self):
        with self._get("/agent-server/docs?example=wl-classification") as r:
            data = json.loads(r.read().decode())
        names = {f["name"] for f in data["files"]}
        self.assertEqual(names, {"AGENTS.md", "examples/PyTorch/wl-classification/main.py"})

    def test_example_query_param_is_repeatable_for_multiple_usecases(self):
        with self._get("/agent-server/docs?example=wl-detection&example=wl-segmentation") as r:
            data = json.loads(r.read().decode())
        names = {f["name"] for f in data["files"]}
        self.assertEqual(names, {
            "AGENTS.md",
            "examples/PyTorch/wl-detection/main.py",
            "examples/PyTorch/wl-segmentation/main.py",
        })

    def test_example_query_param_is_ignored_when_not_a_known_usecase(self):
        # Client-supplied -- must be checked against the allowlist, never
        # trusted as a path component (e.g. "../../../etc/passwd").
        with self._get("/agent-server/docs?example=../../../etc/passwd") as r:
            data = json.loads(r.read().decode())
        names = {f["name"] for f in data["files"]}
        self.assertEqual(names, {"AGENTS.md"})

    def test_no_example_query_param_means_no_example_file(self):
        with self._get("/agent-server/docs") as r:
            data = json.loads(r.read().decode())
        names = {f["name"] for f in data["files"]}
        self.assertNotIn("examples/PyTorch/wl-classification/main.py", names)


class TestLoopRegistryUnit(unittest.TestCase):
    """_LoopRegistry directly -- no HTTP server, no real OpenCode process.
    Mocks the module-level _opencode_json_request/_opencode_send_and_collect/
    _opencode_get_messages functions the registry itself calls, so these
    exercise its own eager-session-creation/locking/preamble-once logic in
    isolation. A fresh registry per test, not the module-level singleton."""

    def setUp(self):
        self.registry = ui_server._LoopRegistry()

    def tearDown(self):
        for job in self.registry._jobs.values():
            if job.timer is not None:
                job.timer.cancel()

    def test_start_creates_the_session_eagerly_not_on_first_tick(self):
        with patch.object(ui_server, "_opencode_session") as mock_session, \
                patch.object(ui_server, "_opencode_json_request", return_value={"id": "sess-1"}) as mock_req, \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            mock_session.ensure.return_value = {"ok": True, "url": "http://fake"}
            result = self.registry.start("monitor training", 60.0, "/tmp/ws", "http://localhost:5173")

        self.assertTrue(result["ok"], result)
        job = self.registry._jobs[result["id"]]
        # Eager: already set by start() itself, not left for _fire's first
        # tick (which runs in a background thread started right after).
        self.assertEqual(job.session_id, "sess-1")
        mock_req.assert_any_call("http://fake", "/session", method="POST", body=unittest.mock.ANY)

    def test_rejected_after_session_creation_deletes_the_orphaned_session(self):
        def _ensure_and_fill_concurrently(*_args, **_kwargs):
            # Simulates 3 OTHER starts winning the race while this one's
            # ensure() call was in flight -- by the time start() re-checks
            # under the lock, the cap has already been hit by them.
            for i in range(3):
                self.registry._jobs[str(i)] = ui_server._LoopJob(str(i), "p", 60.0, "/tmp")
            return {"ok": True, "url": "http://fake"}

        with patch.object(ui_server, "_opencode_session") as mock_session, \
                patch.object(ui_server, "_opencode_json_request", return_value={"id": "sess-orphan"}) as mock_req:
            mock_session.ensure.side_effect = _ensure_and_fill_concurrently
            result = self.registry.start("monitor training", 60.0, "/tmp/ws", "http://localhost:5173")

        self.assertFalse(result["ok"])
        self.assertIn("already running", result["error"])
        mock_req.assert_any_call("http://fake", "/session/sess-orphan", method="DELETE")

    def test_fire_sends_the_preamble_once_then_plain_prompt_on_later_ticks(self):
        job = ui_server._LoopJob("1", "check the loss", 60.0, "/tmp")
        job.session_id, job.base_url = "sess-1", "http://fake"
        self.registry._jobs["1"] = job

        sent_texts = []

        def _fake_send(_base_url, _session_id, text, _model=None, timeout=600.0):  # noqa: ARG001
            sent_texts.append(text)
            return "tick result", None

        with patch.object(ui_server, "_opencode_send_and_collect", side_effect=_fake_send):
            self.registry._fire("1", "http://fake")
            self.assertTrue(job.preamble_sent)
            self.assertIn("recurring monitoring agent", sent_texts[0])
            self.assertIn("check the loss", sent_texts[0])
            job.timer.cancel()

            self.registry._fire("1", "http://fake")
            self.assertEqual(sent_texts[1], "check the loss")
            job.timer.cancel()

    def test_get_messages_success(self):
        job = ui_server._LoopJob("1", "p", 60.0, "/tmp")
        job.session_id, job.base_url = "sess-1", "http://fake"
        self.registry._jobs["1"] = job
        canned = [{"info": {"role": "user"}, "parts": [{"type": "text", "text": "hi"}]}]
        with patch.object(ui_server, "_opencode_get_messages", return_value=canned):
            result = self.registry.get_messages("1")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["messages"], canned)

    def test_get_messages_unknown_job(self):
        result = self.registry.get_messages("nope")
        self.assertFalse(result["ok"])


class TestLoopMessagesEndpoint(_ServerTestCase):
    """GET /agent-server/loop/<id>/messages -- a loop tab's read-only
    transcript, proxied through the module-level _loop_registry singleton to
    a job's own OpenCode session (mocked here; no real OpenCode process).
    Loopback-gating itself mirrors the existing loop routes (_stop_loop et
    al.), which have no dedicated test for the negative case either -- a
    real test client is always loopback."""

    def _seed_job(self):
        job = ui_server._LoopJob("1", "check the loss", 60.0, self.tmp)
        job.session_id, job.base_url = "sess-1", "http://fake"
        ui_server._loop_registry._jobs["1"] = job
        return job

    def tearDown(self):
        ui_server._loop_registry._jobs.clear()
        super().tearDown()

    def test_get_messages_returns_the_sessions_history(self):
        self._seed_job()
        canned = [{"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "hi"}]}]
        with patch.object(ui_server, "_opencode_get_messages", return_value=canned):
            with self._get("/agent-server/loop/1/messages") as r:
                data = json.loads(r.read().decode())
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["messages"], canned)

    def test_get_messages_404s_for_an_unknown_loop(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/agent-server/loop/999/messages")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
