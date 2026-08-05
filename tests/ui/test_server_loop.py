"""Tests for weightslab/ui/server.py's /loop feature -- recurring
OpenCode-backed monitoring jobs (`_LoopRegistry` + the
/agent-server/loop/{start,list,stop} endpoints).

_LoopRegistry.start() delegates session/message plumbing to the module-level
_opencode_json_request/_opencode_send_and_collect helpers (already covered at
the wire-protocol level by tests/trainer/services/test_opencode_chat.py's
fake-SSE-server tests for the sibling Python client). Here those two helpers
are mocked so the tests exercise _LoopRegistry's OWN logic -- validation,
session reuse across ticks, error bookkeeping, stop/list, and the HTTP
wiring -- the same split test_server_agent.py uses (direct _OpencodeSession
unit tests, then a separate endpoint-level test class).
"""

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

from weightslab.ui import server as ui_server


class TestLoopRegistryUnit(unittest.TestCase):
    """Exercises _LoopRegistry directly, with the opencode session assumed
    already up (_opencode_session.ensure mocked) and the session-create/
    send-and-collect wire calls mocked -- those are covered elsewhere (see
    module docstring)."""

    def setUp(self):
        self.registry = ui_server._LoopRegistry()
        self._ensure_patch = patch.object(
            ui_server._opencode_session, "ensure",
            return_value={"ok": True, "url": "http://127.0.0.1:1", "workspace": "/tmp", "reused": False},
        )
        self._ensure_patch.start()

    def tearDown(self):
        self.registry.shutdown()
        self._ensure_patch.stop()

    def _wait_for_first_tick(self, job_id, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            jobs = {j["id"]: j for j in self.registry.list()}
            job = jobs.get(job_id)
            if job and (job["lastResult"] is not None or job["lastError"] is not None):
                return job
            time.sleep(0.01)
        self.fail("loop job never completed its first tick")

    def test_rejects_an_empty_prompt(self):
        result = self.registry.start("   ", 120, "/tmp", None)
        self.assertFalse(result["ok"])
        self.assertIn("prompt", result["error"])
        self.assertEqual(self.registry.list(), [])

    def test_rejects_an_interval_below_the_minimum(self):
        result = self.registry.start("watch training", 10, "/tmp", None)
        self.assertFalse(result["ok"])
        self.assertIn("Minimum", result["error"])

    def test_surfaces_an_ensure_failure_without_starting_a_job(self):
        self._ensure_patch.stop()
        try:
            with patch.object(ui_server._opencode_session, "ensure",
                               return_value={"ok": False, "error": "no opencode binary"}):
                result = self.registry.start("watch training", 120, "/tmp", None)
        finally:
            self._ensure_patch.start()

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "no opencode binary")
        self.assertEqual(self.registry.list(), [])

    def test_first_tick_creates_a_session_seeded_with_the_system_preamble(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}) as create_mock, \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="all good") as send_mock:
            result = self.registry.start("watch the loss", 120, "/tmp", "http://localhost:5173")
            self.assertTrue(result["ok"], result)
            job = self._wait_for_first_tick(result["id"])

        self.assertEqual(job["lastResult"], "all good")
        self.assertIsNone(job["lastError"])
        create_mock.assert_called_once()
        self.assertEqual(create_mock.call_args.args[1], "/session")
        sent_text = send_mock.call_args.args[2]
        self.assertIn("watch the loss", sent_text)
        self.assertIn("weightslab pause", sent_text)  # system preamble documents the CLI verbs

    def test_second_tick_reuses_the_session_and_sends_the_bare_prompt(self):
        with patch.object(ui_server, "_LOOP_MIN_INTERVAL_SECONDS", 0.01), \
                patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}) as create_mock, \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok") as send_mock:
            result = self.registry.start("watch the loss", 0.02, "/tmp", None)
            self._wait_for_first_tick(result["id"])

            deadline = time.monotonic() + 2
            while send_mock.call_count < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.registry.stop(result["id"])

        self.assertGreaterEqual(send_mock.call_count, 2)
        create_mock.assert_called_once()  # session created once, reused on tick 2
        second_call_text = send_mock.call_args_list[1].args[2]
        self.assertEqual(second_call_text, "watch the loss")  # no preamble wrapper on repeat ticks

    def test_records_last_error_and_does_not_set_last_result_on_a_failed_tick(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", side_effect=RuntimeError("boom")):
            result = self.registry.start("watch training", 120, "/tmp", None)
            job = self._wait_for_first_tick(result["id"])

        self.assertEqual(job["lastError"], "boom")
        self.assertIsNone(job["lastResult"])

    def test_stop_removes_the_job_and_prevents_a_further_tick(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok") as send_mock:
            result = self.registry.start("watch training", 120, "/tmp", None)
            self._wait_for_first_tick(result["id"])

            stop_result = self.registry.stop(result["id"])
            self.assertTrue(stop_result["ok"])
            self.assertEqual(self.registry.list(), [])

            calls_at_stop = send_mock.call_count
            time.sleep(0.1)
            self.assertEqual(send_mock.call_count, calls_at_stop)  # no tick after stop

    def test_stopping_an_unknown_job_id_reports_not_found(self):
        result = self.registry.stop("does-not-exist")
        self.assertFalse(result["ok"])

    def test_list_reflects_multiple_concurrent_jobs(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            first = self.registry.start("watch a", 120, "/tmp", None)
            second = self.registry.start("watch b", 120, "/tmp", None)
            self._wait_for_first_tick(first["id"])
            self._wait_for_first_tick(second["id"])

        ids = {j["id"] for j in self.registry.list()}
        self.assertEqual(ids, {first["id"], second["id"]})

    def test_shutdown_stops_every_job(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            result = self.registry.start("watch training", 120, "/tmp", None)
            self._wait_for_first_tick(result["id"])

        self.registry.shutdown()
        self.assertEqual(self.registry.list(), [])

    def test_rejects_a_fourth_concurrent_job(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            for i in range(ui_server._LOOP_MAX_CONCURRENT):
                result = self.registry.start(f"watch {i}", 120, "/tmp", None)
                self.assertTrue(result["ok"], result)

            fourth = self.registry.start("one too many", 120, "/tmp", None)

        self.assertFalse(fourth["ok"])
        self.assertIn("already running", fourth["error"])
        self.assertEqual(len(self.registry.list()), ui_server._LOOP_MAX_CONCURRENT)

    def test_update_changes_the_prompt_without_touching_the_interval(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            result = self.registry.start("watch training", 120, "/tmp", None)
            self._wait_for_first_tick(result["id"])

            update_result = self.registry.update(result["id"], prompt="watch training more closely")

        self.assertTrue(update_result["ok"], update_result)
        job = {j["id"]: j for j in self.registry.list()}[result["id"]]
        self.assertEqual(job["prompt"], "watch training more closely")
        self.assertEqual(job["intervalSeconds"], 120)

    def test_update_changes_the_interval_and_reschedules_immediately(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok") as send_mock:
            result = self.registry.start("watch training", 120, "/tmp", None)
            self._wait_for_first_tick(result["id"])

            with patch.object(ui_server, "_LOOP_MIN_INTERVAL_SECONDS", 0.01):
                update_result = self.registry.update(result["id"], interval_seconds=0.02)
            self.assertTrue(update_result["ok"], update_result)
            self.assertEqual(update_result["intervalSeconds"], 0.02)

            # If the reschedule took effect immediately, a second tick lands
            # almost at once rather than after the original 120s interval.
            deadline = time.monotonic() + 2
            while send_mock.call_count < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.registry.stop(result["id"])

        self.assertGreaterEqual(send_mock.call_count, 2)

    def test_update_on_an_unknown_job_reports_not_found(self):
        result = self.registry.update("does-not-exist", prompt="anything")
        self.assertFalse(result["ok"])
        self.assertIn("No loop job", result["error"])

    def test_update_rejects_an_empty_prompt(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            result = self.registry.start("watch training", 120, "/tmp", None)
            self._wait_for_first_tick(result["id"])

            update_result = self.registry.update(result["id"], prompt="   ")

        self.assertFalse(update_result["ok"])
        self.assertIn("prompt", update_result["error"])

    def test_update_rejects_an_interval_below_the_minimum(self):
        with patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="ok"):
            result = self.registry.start("watch training", 120, "/tmp", None)
            self._wait_for_first_tick(result["id"])

            update_result = self.registry.update(result["id"], interval_seconds=10)

        self.assertFalse(update_result["ok"])
        self.assertIn("Minimum", update_result["error"])


class _LoopServerTestCase(unittest.TestCase):
    """Spins up a real serve_ui() on 127.0.0.1:<ephemeral>, same shape as
    test_server_agent.py's _ServerTestCase -- reused rather than imported
    since that one is a module-private helper of its own file."""

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

        self._orig_registry = ui_server._loop_registry
        ui_server._loop_registry = ui_server._LoopRegistry()

    def tearDown(self):
        ui_server._loop_registry.shutdown()
        ui_server._loop_registry = self._orig_registry
        self.httpd.shutdown()
        self.thread.join(timeout=5)

    def _get(self, path):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5)

    def _post_json(self, path, body):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", method="POST", data=data,
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=10)

    def _post(self, path):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method="POST", data=b"")
        return urllib.request.urlopen(req, timeout=10)


class TestLoopEndpoints(_LoopServerTestCase):

    def test_list_is_empty_before_anything_starts(self):
        with self._get("/agent-server/loop/list") as r:
            data = json.loads(r.read().decode())
        self.assertEqual(data["loops"], [])

    def test_start_delegates_to_the_registry_and_the_new_job_shows_up_in_list(self):
        with patch.object(ui_server._loop_registry, "start",
                           return_value={"ok": True, "id": "1", "intervalSeconds": 1800.0}) as start_mock:
            with self._post_json("/agent-server/loop/start", {"prompt": "watch the loss", "intervalMinutes": 30}) as r:
                data = json.loads(r.read().decode())
        self.assertTrue(data["ok"], data)
        self.assertEqual(data["id"], "1")
        start_mock.assert_called_once()
        args = start_mock.call_args.args
        self.assertEqual(args[0], "watch the loss")
        self.assertEqual(args[1], 30 * 60.0)
        self.assertEqual(args[2], self.tmp)  # rooted at the experiment dir

    def test_start_with_an_empty_prompt_returns_a_400(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post_json("/agent-server/loop/start", {"prompt": "", "intervalMinutes": 30})
        self.assertEqual(ctx.exception.code, 400)
        data = json.loads(ctx.exception.read().decode())
        self.assertFalse(data["ok"])

    def test_stop_delegates_to_the_registry_with_the_id_parsed_out_of_the_path(self):
        with patch.object(ui_server._loop_registry, "stop", return_value={"ok": True}) as stop_mock:
            with self._post("/agent-server/loop/42/stop") as r:
                data = json.loads(r.read().decode())
        self.assertTrue(data["ok"])
        stop_mock.assert_called_once_with("42")

    def test_stopping_an_unknown_job_returns_a_404(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post("/agent-server/loop/does-not-exist/stop")
        self.assertEqual(ctx.exception.code, 404)

    def test_update_delegates_to_the_registry_with_the_id_parsed_out_of_the_path(self):
        with patch.object(ui_server._loop_registry, "update",
                           return_value={"ok": True, "id": "42", "prompt": "new prompt", "intervalSeconds": 300.0}) as update_mock:
            with self._post_json("/agent-server/loop/42/update", {"prompt": "new prompt", "intervalMinutes": 5}) as r:
                data = json.loads(r.read().decode())
        self.assertTrue(data["ok"], data)
        update_mock.assert_called_once_with("42", prompt="new prompt", interval_seconds=5 * 60.0)

    def test_updating_an_unknown_job_returns_a_400(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post_json("/agent-server/loop/does-not-exist/update", {"prompt": "x"})
        self.assertEqual(ctx.exception.code, 400)
        data = json.loads(ctx.exception.read().decode())
        self.assertFalse(data["ok"])

    def test_list_reflects_a_real_start_end_to_end_through_the_registry(self):
        with patch.object(ui_server._opencode_session, "ensure",
                           return_value={"ok": True, "url": "http://127.0.0.1:1", "workspace": self.tmp, "reused": False}), \
                patch.object(ui_server, "_opencode_json_request", return_value={"id": "ses_abc"}), \
                patch.object(ui_server, "_opencode_send_and_collect", return_value="all quiet"):
            with self._post_json("/agent-server/loop/start", {"prompt": "watch training", "intervalMinutes": 30}) as r:
                started = json.loads(r.read().decode())
            self.assertTrue(started["ok"], started)

            deadline = time.monotonic() + 2
            job = None
            while time.monotonic() < deadline:
                with self._get("/agent-server/loop/list") as r:
                    loops = json.loads(r.read().decode())["loops"]
                job = next((j for j in loops if j["id"] == started["id"]), None)
                if job and job["lastResult"]:
                    break
                time.sleep(0.02)

        self.assertIsNotNone(job)
        self.assertEqual(job["lastResult"], "all quiet")
        self.assertEqual(job["prompt"], "watch training")


if __name__ == "__main__":
    unittest.main()
