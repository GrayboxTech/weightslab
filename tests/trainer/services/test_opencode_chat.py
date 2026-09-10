"""Tests for OpenCodeChat (weightslab/trainer/services/agent/opencode_chat.py).

Exercises the module against a REAL minimal HTTP server implementing the
subset of OpenCode's protocol this class needs (POST /session, POST
/session/{id}/message, GET /event as text/event-stream) rather than mocking
urllib -- the class's correctness hinges on stream-first ordering and SSE
event parsing, which a mocked urlopen would not honestly exercise. Mirrors
the fake-server technique already used in tests/ui/test_server_agent.py for
the same reason.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from weightslab.trainer.services.agent.opencode_chat import OpenCodeChat, OpenCodeError


class _FakeOpenCodeHandler(BaseHTTPRequestHandler):
    """Implements just enough of OpenCode's HTTP+SSE surface to drive
    OpenCodeChat through a full create -> send -> stream -> idle cycle.
    Configured per-server-instance via class attributes the test sets before
    starting it (see _FakeOpenCodeServer below)."""

    def log_message(self, *a):
        pass

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            body = {}

        if self.path == "/session":
            self.server.recorded_session_titles.append(body.get("title"))
            self._send_json({"id": self.server.session_id})
            return

        if self.path == f"/session/{self.server.session_id}/message":
            self.server.recorded_messages.append(body)
            # A real server holds this open for the whole turn; this fake
            # returns immediately -- OpenCodeChat must not rely on this
            # response for content, only on the SSE stream (see its own
            # docstring). Sleep briefly so the message genuinely arrives
            # after the stream has had a chance to open, exercising the
            # stream-first ordering rather than accidentally passing by luck.
            time.sleep(0.05)
            self._send_json({})
            return

        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path != "/event":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(event: dict) -> None:
            payload = f"data: {json.dumps(event)}\n\n".encode("utf-8")
            self.wfile.write(payload)
            self.wfile.flush()

        session_id = self.server.session_id
        # Wait for the message POST to actually land before emitting anything
        # -- proves OpenCodeChat is genuinely reading a live stream, not just
        # replaying a canned response.
        deadline = time.monotonic() + 5
        while not self.server.recorded_messages and time.monotonic() < deadline:
            time.sleep(0.01)

        info = {"id": "msg_1", "role": "assistant", "sessionID": session_id}
        if self.server.reply_tokens is not None:
            info["tokens"] = self.server.reply_tokens
        emit({"type": "message.updated", "properties": {"info": info}})
        for delta in self.server.reply_deltas:
            emit({"type": "message.part.updated", "properties": {"part": {
                "id": f"prt_{delta[:4]}", "type": "text", "text": delta, "messageID": "msg_1", "sessionID": session_id,
            }}})
        if self.server.emit_error:
            emit({"type": "session.error", "properties": {"sessionID": session_id}})
        else:
            emit({"type": "session.idle", "properties": {"sessionID": session_id}})
        # Keep the connection open a moment so OpenCodeChat's break-on-idle has
        # definitely already fired before we tear down.
        time.sleep(0.05)


class _FakeOpenCodeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, reply_deltas, emit_error=False, reply_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = "ses_test1"
        self.reply_deltas = reply_deltas
        self.emit_error = emit_error
        self.reply_tokens = reply_tokens
        self.recorded_messages = []
        self.recorded_session_titles = []


class _ServerTestCase(unittest.TestCase):
    def _start_server(self, reply_deltas, emit_error=False, reply_tokens=None):
        self.httpd = _FakeOpenCodeServer(
            ("127.0.0.1", 0), _FakeOpenCodeHandler,
            reply_deltas=reply_deltas, emit_error=emit_error, reply_tokens=reply_tokens,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        if hasattr(self, "httpd"):
            self.httpd.shutdown()
            self.thread.join(timeout=5)


class TestOpenCodeChatCall(_ServerTestCase):
    def test_collects_streamed_text_and_returns_an_ai_message(self):
        self._start_server(reply_deltas=["Here is ", "the answer."])
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model="openrouter/openai/gpt-5", timeout=10)

        result = chat.as_runnable().invoke("do the thing")

        self.assertEqual(result.content, "Here is the answer.")

    def test_sends_the_model_ref_split_on_first_slash_only(self):
        self._start_server(reply_deltas=["ok"])
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model="openrouter/anthropic/claude-opus-4.6", timeout=10)

        chat._call("hello")

        self.assertEqual(len(self.httpd.recorded_messages), 1)
        self.assertEqual(
            self.httpd.recorded_messages[0]["model"],
            {"providerID": "openrouter", "modelID": "anthropic/claude-opus-4.6"},
        )

    def test_disables_every_mutating_tool_on_the_outgoing_message(self):
        """This wrapper backs the SDK agent's text/JSON call sites, which parse
        the reply themselves -- unlike the Weights Studio landing chat, it must
        never let OpenCode write files as a side effect."""
        self._start_server(reply_deltas=["ok"])
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model=None, timeout=10)

        chat._call("hello")

        tools = self.httpd.recorded_messages[0]["tools"]
        self.assertEqual(tools, {"write": False, "edit": False, "patch": False, "bash": False})

    def test_omits_model_field_when_none_configured(self):
        self._start_server(reply_deltas=["ok"])
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model=None, timeout=10)

        chat._call("hello")

        self.assertNotIn("model", self.httpd.recorded_messages[0])

    def test_creates_a_fresh_session_per_call(self):
        self._start_server(reply_deltas=["a"])
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model=None, timeout=10)

        chat._call("first")
        chat._call("second")

        # A fresh session per call (self.history on DataManipulationAgent
        # already carries cross-call context) -- both calls hit /session, not
        # a reused id.
        self.assertEqual(len(self.httpd.recorded_session_titles), 2)

    def test_degrades_to_empty_string_on_session_error_rather_than_raising(self):
        self._start_server(reply_deltas=["partial"], emit_error=True)
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model=None, timeout=10)

        # session.error still ends the read loop cleanly; whatever text arrived
        # before the error is still returned rather than raising.
        result = chat._call("hello")
        self.assertEqual(result.content, "partial")

    def test_raises_opencode_error_when_the_server_is_unreachable(self):
        chat = OpenCodeChat("http://127.0.0.1:1", model=None, timeout=2)  # nothing listens on port 1
        with self.assertRaises(OpenCodeError):
            chat._call("hello")

    def test_populates_last_usage_from_the_assistant_messages_tokens(self):
        self._start_server(
            reply_deltas=["ok"],
            reply_tokens={"input": 120, "output": 30, "reasoning": 5, "cache": {"read": 80, "write": 10}},
        )
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model=None, timeout=10)

        self.assertIsNone(chat.last_usage)
        chat._call("hello")

        self.assertEqual(
            chat.last_usage,
            {"input": 120, "output": 30, "reasoning": 5, "cache_read": 80, "cache_write": 10},
        )

    def test_last_usage_is_none_when_the_reply_carries_no_tokens_field(self):
        self._start_server(reply_deltas=["ok"])  # reply_tokens defaults to None
        chat = OpenCodeChat(f"http://127.0.0.1:{self.port}", model=None, timeout=10)

        chat._call("hello")

        self.assertIsNone(chat.last_usage)


class TestModelRefParsing(unittest.TestCase):
    def test_splits_on_first_slash_only(self):
        chat = OpenCodeChat("http://x", model="openrouter/anthropic/claude-opus-4.6")
        self.assertEqual(chat._model_ref(), {"providerID": "openrouter", "modelID": "anthropic/claude-opus-4.6"})

    def test_none_when_no_model_configured(self):
        chat = OpenCodeChat("http://x", model=None)
        self.assertIsNone(chat._model_ref())

    def test_none_when_model_has_no_slash(self):
        chat = OpenCodeChat("http://x", model="justamodel")
        self.assertIsNone(chat._model_ref())


class TestHandleEvent(unittest.TestCase):
    """Unit-level checks on the event state machine, independent of the
    network -- complements the end-to-end server tests above."""

    def test_ignores_parts_from_a_message_not_yet_known_to_be_assistant(self):
        text_parts = {}
        assistant_ids = set()
        payload = json.dumps({
            "type": "message.part.updated",
            "properties": {"part": {"id": "p1", "type": "text", "text": "x", "messageID": "msg_unknown", "sessionID": "s1"}},
        })
        outcome = OpenCodeChat._handle_event(payload, "s1", assistant_ids, text_parts)
        self.assertIsNone(outcome)
        self.assertEqual(text_parts, {})

    def test_ignores_events_for_a_different_session(self):
        text_parts = {}
        assistant_ids = {"msg_1"}
        payload = json.dumps({
            "type": "message.part.updated",
            "properties": {"part": {"id": "p1", "type": "text", "text": "x", "messageID": "msg_1", "sessionID": "OTHER"}},
        })
        outcome = OpenCodeChat._handle_event(payload, "s1", assistant_ids, text_parts)
        self.assertIsNone(outcome)
        self.assertEqual(text_parts, {})

    def test_malformed_payload_is_ignored_not_raised(self):
        outcome = OpenCodeChat._handle_event("not json", "s1", set(), {})
        self.assertIsNone(outcome)

    def test_session_idle_signals_completion(self):
        outcome = OpenCodeChat._handle_event(
            json.dumps({"type": "session.idle", "properties": {"sessionID": "s1"}}), "s1", set(), {},
        )
        self.assertEqual(outcome, "idle")

    def test_populates_the_usage_dict_when_the_assistant_message_carries_tokens(self):
        usage = {}
        payload = json.dumps({
            "type": "message.updated",
            "properties": {"info": {
                "id": "msg_1", "role": "assistant", "sessionID": "s1",
                "tokens": {"input": 10, "output": 2, "reasoning": 0, "cache": {"read": 5, "write": 1}},
            }},
        })
        outcome = OpenCodeChat._handle_event(payload, "s1", set(), {}, usage)
        self.assertIsNone(outcome)
        self.assertEqual(usage, {"input": 10, "output": 2, "reasoning": 0, "cache_read": 5, "cache_write": 1})

    def test_usage_param_is_optional_and_ignored_when_omitted(self):
        # Existing call sites that predate the usage tracking must keep working.
        payload = json.dumps({
            "type": "message.updated",
            "properties": {"info": {"id": "msg_1", "role": "assistant", "sessionID": "s1", "tokens": {"input": 1}}},
        })
        outcome = OpenCodeChat._handle_event(payload, "s1", set(), {})
        self.assertIsNone(outcome)


class TestEnsureReachable(unittest.TestCase):
    """The other half of the cross-process handoff opencode_process.py
    implements: this side self-heals its own base_url via that module's
    resolve_or_spawn_opencode instead of staying pointed at a dead address
    forever. The module itself (env/lockfile/spawn precedence, real
    subprocess spawn+poll) is covered by tests/test_opencode_process.py --
    these mock it out to pin down exactly when OpenCodeChat calls it."""

    def test_explicit_url_is_never_auto_replaced_even_if_dead(self):
        chat = OpenCodeChat("http://127.0.0.1:1", workspace_dir="/tmp/x", url_is_explicit=True)
        with mock.patch("weightslab.opencode_process.opencode_healthy") as healthy, \
                mock.patch("weightslab.opencode_process.resolve_or_spawn_opencode") as resolve:
            chat._ensure_reachable()
        healthy.assert_not_called()
        resolve.assert_not_called()
        self.assertEqual(chat.base_url, "http://127.0.0.1:1")

    def test_already_healthy_non_explicit_url_is_left_alone(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", workspace_dir="/tmp/x", url_is_explicit=False)
        with mock.patch("weightslab.opencode_process.opencode_healthy", return_value=True) as healthy, \
                mock.patch("weightslab.opencode_process.resolve_or_spawn_opencode") as resolve:
            chat._ensure_reachable()
        healthy.assert_called_once_with("http://127.0.0.1:4096")
        resolve.assert_not_called()
        self.assertEqual(chat.base_url, "http://127.0.0.1:4096")

    def test_dead_non_explicit_url_resolves_or_spawns_a_replacement(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", workspace_dir="/tmp/x", url_is_explicit=False)
        with mock.patch("weightslab.opencode_process.opencode_healthy", return_value=False), \
                mock.patch("weightslab.opencode_process.resolve_or_spawn_opencode",
                           return_value={"ok": True, "url": "http://127.0.0.1:9999"}) as resolve:
            chat._ensure_reachable()
        resolve.assert_called_once_with("/tmp/x")
        self.assertEqual(chat.base_url, "http://127.0.0.1:9999")

    def test_failed_resolve_leaves_the_dead_url_in_place(self):
        # A visible connection error on the next real call is more honest
        # than silently pretending nothing changed.
        chat = OpenCodeChat("http://127.0.0.1:4096", workspace_dir="/tmp/x", url_is_explicit=False)
        with mock.patch("weightslab.opencode_process.opencode_healthy", return_value=False), \
                mock.patch("weightslab.opencode_process.resolve_or_spawn_opencode",
                           return_value={"ok": False, "error": "no opencode"}):
            chat._ensure_reachable()
        self.assertEqual(chat.base_url, "http://127.0.0.1:4096")

    def test_missing_workspace_dir_falls_back_to_the_current_directory(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", url_is_explicit=False)  # workspace_dir defaults to None
        with mock.patch("weightslab.opencode_process.opencode_healthy", return_value=False), \
                mock.patch("weightslab.opencode_process.resolve_or_spawn_opencode",
                           return_value={"ok": True, "url": "http://127.0.0.1:9999"}) as resolve:
            chat._ensure_reachable()
        resolve.assert_called_once_with(".")

    def test_call_invokes_ensure_reachable_before_creating_a_session(self):
        # _call is the single real entry point all three of
        # DataManipulationAgent's call sites go through (see module
        # docstring) -- this pins down that self-healing actually happens
        # on the path real turns take, not just when called directly.
        chat = OpenCodeChat("http://127.0.0.1:4096", url_is_explicit=False)
        calls = []
        chat._ensure_reachable = lambda: calls.append("ensure_reachable")
        chat._ensure_model_resolved = lambda: calls.append("ensure_model_resolved")
        chat._create_session = lambda: (calls.append("create_session") or "ses_x")
        chat._collect_reply = lambda session_id, text: (calls.append("collect_reply") or "ok")
        chat._call("hello")
        self.assertEqual(
            calls,
            ["ensure_reachable", "ensure_model_resolved", "create_session", "collect_reply"],
        )


class TestEnsureModelResolved(unittest.TestCase):
    """Confirmed live: leaving `model` unset does NOT mean "OpenCode picks a
    sensible default" -- it means OpenCode picks whatever's configured,
    arbitrarily (an image-generation preview model, in the case that
    surfaced this). _ensure_model_resolved is the fix: resolve a REAL
    default (the user's last actual pick, or a provider's own configured
    default) instead of leaving it to chance. Mirrors TestEnsureReachable's
    own mocked-request style -- the wire format itself (GET /config,
    GET /config/providers) is exercised for real in
    TestGetAvailableModelsOpenCode (test_agent_opencode_provider.py)."""

    def _fake_response(self, payload):
        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return json.dumps(payload).encode()
        return _Resp()

    def test_explicit_model_is_never_auto_replaced(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="openrouter/anthropic/claude-opus-4.6", model_is_explicit=True)
        with mock.patch.object(chat, "_request") as request_mock:
            chat._ensure_model_resolved()
        request_mock.assert_not_called()
        self.assertEqual(chat.model, "openrouter/anthropic/claude-opus-4.6")

    def test_already_set_non_explicit_model_follows_a_later_ui_pick(self):
        # The studio's model picker writes the pick into OpenCode's own config
        # (PUT /config). A backend that kept the model it resolved on its first
        # turn went on answering with the old one for the rest of the run, so a
        # NON-explicit model re-checks /config every turn and follows it.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="openrouter/openai/gpt-5", model_is_explicit=False)
        with mock.patch.object(
            chat, "_request",
            return_value=self._fake_response({"model": "opencode/big-pickle"}),
        ) as request_mock:
            chat._ensure_model_resolved()
        request_mock.assert_called_once_with("/config")
        self.assertEqual(chat.model, "opencode/big-pickle")

    def test_explicit_model_is_never_overridden_by_the_config(self):
        # OPENCODE_MODEL / agent_config.yaml's opencode_model PIN the model:
        # no request, and the UI picker cannot move it.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="openrouter/openai/gpt-5", model_is_explicit=True)
        with mock.patch.object(chat, "_request") as request_mock:
            chat._ensure_model_resolved()
        request_mock.assert_not_called()
        self.assertEqual(chat.model, "openrouter/openai/gpt-5")

    def test_a_failed_config_read_keeps_the_model_already_resolved(self):
        # One local request failing must not drop a working model for the
        # fallback mid-run.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="anthropic/claude-haiku-4.5", model_is_explicit=False)
        with mock.patch.object(chat, "_request", side_effect=OSError("refused")):
            chat._ensure_model_resolved()
        self.assertEqual(chat.model, "anthropic/claude-haiku-4.5")

    def test_unset_model_resolves_from_config_own_model_field(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model=None, model_is_explicit=False)
        with mock.patch.object(chat, "_request", return_value=self._fake_response({"model": "anthropic/claude-haiku-4.5"})) as request_mock:
            chat._ensure_model_resolved()
        request_mock.assert_called_once_with("/config")
        self.assertEqual(chat.model, "anthropic/claude-haiku-4.5")

    def test_falls_back_to_the_hardcoded_default_when_config_has_no_model(self):
        # Confirmed live: a provider-reported default (/config/providers, no
        # longer consulted at all -- see _ensure_model_resolved's own
        # docstring) could itself be an arbitrary, non-text-reasoning model
        # whenever ANY provider had credentials configured, silently
        # pre-empting this default every time. A single /config miss now
        # goes straight to the known-good free model, with no second request.
        chat = OpenCodeChat("http://127.0.0.1:4096", model=None, model_is_explicit=False)
        with mock.patch.object(chat, "_request", return_value=self._fake_response({})) as request_mock:
            chat._ensure_model_resolved()
        request_mock.assert_called_once_with("/config")
        self.assertEqual(chat.model, "opencode/big-pickle")

    def test_no_resolvable_model_falls_back_to_the_hardcoded_default(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model=None, model_is_explicit=False)
        with mock.patch.object(chat, "_request", side_effect=OSError("refused")):
            chat._ensure_model_resolved()  # must not raise
        self.assertEqual(chat.model, "opencode/big-pickle")

    def test_config_field_that_is_not_provider_slash_model_falls_through(self):
        # A malformed/unexpected `model` field (missing the "/") is treated
        # the same as absent, not used as-is -- falls to the hardcoded
        # default, not a second request.
        chat = OpenCodeChat("http://127.0.0.1:4096", model=None, model_is_explicit=False)
        with mock.patch.object(chat, "_request", return_value=self._fake_response({"model": "not-a-provider-model-pair"})) as request_mock:
            chat._ensure_model_resolved()
        request_mock.assert_called_once_with("/config")
        self.assertEqual(chat.model, "opencode/big-pickle")



class PublishModelTests(unittest.TestCase):
    """publish_model / resolve_model -- how the two sides converge on one model.

    Verified against a live server before these were written: PATCH /config
    (workspace scope) answers 200 and echoes the value back but does NOT change
    what GET /config reports, while PATCH /global/config does. GET /config is
    what both the studio picker and this backend read, so the write has to go
    to the global scope, and be CONFIRMED rather than trusted.
    """

    @staticmethod
    def _resp(payload):
        return mock.MagicMock(
            __enter__=mock.MagicMock(
                return_value=mock.MagicMock(read=lambda: json.dumps(payload).encode())),
            __exit__=mock.MagicMock(return_value=False),
        )

    def test_publish_writes_the_global_scope_first(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="opencode/big-pickle",
                            model_is_explicit=False)
        calls = []

        def fake_request(path, method="GET", body=None, headers=None):
            calls.append((path, method, body))
            if method == "GET":
                return self._resp({"model": "opencode/big-pickle"})
            return self._resp({})

        with mock.patch.object(chat, "_request", side_effect=fake_request):
            self.assertTrue(chat.publish_model())
        self.assertEqual(calls[0], ("/global/config", "PATCH", {"model": "opencode/big-pickle"}))
        # Confirmed by reading the effective config back, not by the echo.
        self.assertIn(("/config", "GET", None), calls)

    def test_publish_falls_back_to_the_workspace_route(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="opencode/big-pickle",
                            model_is_explicit=False)
        seen = []

        def fake_request(path, method="GET", body=None, headers=None):
            seen.append((path, method))
            if path == "/global/config":
                raise OSError("no such route")
            if method == "GET":
                return self._resp({"model": "opencode/big-pickle"})
            return self._resp({})

        with mock.patch.object(chat, "_request", side_effect=fake_request):
            self.assertTrue(chat.publish_model())
        self.assertIn(("/config", "PATCH"), seen)

    def test_publish_reports_failure_when_the_write_does_not_stick(self):
        # Exactly the live failure mode: 200 back, effective config unchanged.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="opencode/big-pickle",
                            model_is_explicit=False)

        def fake_request(path, method="GET", body=None, headers=None):
            if method == "GET":
                return self._resp({})          # no model -- nothing was stored
            return self._resp({"model": "opencode/big-pickle"})   # echo only

        with mock.patch.object(chat, "_request", side_effect=fake_request):
            self.assertFalse(chat.publish_model())

    def test_resolve_publishes_the_fallback_so_the_studio_adopts_it(self):
        # weightslab started FIRST: nothing chosen anywhere, so the built-in
        # default is resolved AND published for the UI to read.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="", model_is_explicit=False)
        with mock.patch.object(chat, "_configured_model", side_effect=[None, "opencode/big-pickle"]),              mock.patch.object(chat, "_request", return_value=self._resp({})),              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual(model, "opencode/big-pickle")
        self.assertEqual(source, "default-published")

    def test_resolve_adopts_the_studio_pick_without_publishing(self):
        # studio started FIRST: its pick is already in OpenCode's config, so
        # follow it and write nothing.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="", model_is_explicit=False)
        with mock.patch.object(chat, "_configured_model", return_value="openrouter/openai/gpt-5-mini"),              mock.patch.object(chat, "publish_model") as publish,              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual((model, source), ("openrouter/openai/gpt-5-mini", "opencode-config"))
        publish.assert_not_called()

    def test_the_studio_pick_wins_over_a_yaml_seed(self):
        # The reported bug: pick a model in the UI, then start a run --
        # agent_config.yaml's opencode_model pinned the backend back to its own
        # value, ignoring the pick. A yaml model is a SEED, not a pin.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="", model_is_explicit=False,
                            seed_model="opencode/muse-spark-1.3-contributor-free")
        with mock.patch.object(chat, "_configured_model", return_value="openrouter/openai/gpt-5-mini"),              mock.patch.object(chat, "publish_model") as publish,              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual((model, source), ("openrouter/openai/gpt-5-mini", "opencode-config"))
        publish.assert_not_called()

    def test_the_yaml_seed_is_used_and_published_when_nothing_is_chosen(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="", model_is_explicit=False,
                            seed_model="opencode/muse-spark-1.3-contributor-free")
        with mock.patch.object(chat, "_configured_model", return_value=None),              mock.patch.object(chat, "publish_model", return_value=True) as publish,              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual(
            (model, source),
            ("opencode/muse-spark-1.3-contributor-free", "config-seed-published"))
        publish.assert_called_once()

    def test_the_yaml_seed_beats_the_builtin_default(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="", model_is_explicit=False,
                            seed_model="openrouter/openai/gpt-5-mini")
        with mock.patch.object(chat, "_configured_model", return_value=None),              mock.patch.object(chat, "publish_model", return_value=False),              mock.patch.object(chat, "_ensure_reachable"):
            model, _ = chat.resolve_model(publish_default=True)
        self.assertEqual(model, "openrouter/openai/gpt-5-mini")

    def test_the_env_pin_still_beats_the_studio_pick(self):
        # OPENCODE_MODEL is per-process and deliberate: automation must be able
        # to force a model regardless of what anyone picked in the UI.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="openrouter/pinned/model",
                            model_is_explicit=True, seed_model="opencode/seed")
        with mock.patch.object(chat, "_configured_model", return_value="openrouter/openai/gpt-5-mini"),              mock.patch.object(chat, "publish_model", return_value=True),              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual((model, source), ("openrouter/pinned/model", "pinned-published"))

    def test_resolve_publishes_a_pinned_model_so_the_studio_shows_it(self):
        # A model pinned in agent_config.yaml / OPENCODE_MODEL is a deliberate
        # choice too. It used to stay invisible to the studio, which then
        # displayed an unrelated default while every backend query ran on the
        # pinned model.
        chat = OpenCodeChat("http://127.0.0.1:4096", model="openrouter/pinned/model",
                            model_is_explicit=True)
        with mock.patch.object(chat, "publish_model", return_value=True) as publish,              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual((model, source), ("openrouter/pinned/model", "pinned-published"))
        publish.assert_called_once()

    def test_a_pinned_model_stays_pinned_when_it_cannot_be_published(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="openrouter/pinned/model",
                            model_is_explicit=True)
        with mock.patch.object(chat, "publish_model", return_value=False),              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual((model, source), ("openrouter/pinned/model", "pinned"))

    def test_a_model_read_from_the_config_is_never_republished(self):
        chat = OpenCodeChat("http://127.0.0.1:4096", model="", model_is_explicit=False)
        with mock.patch.object(chat, "_configured_model", return_value="opencode/big-pickle"),              mock.patch.object(chat, "publish_model") as publish,              mock.patch.object(chat, "_ensure_reachable"):
            model, source = chat.resolve_model(publish_default=True)
        self.assertEqual((model, source), ("opencode/big-pickle", "opencode-config"))
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
