"""The OpenCode wire layer in ui/server.py, against a real fake server.

Every other loop/agent test in this directory patches
``_opencode_json_request``/``_opencode_send_and_collect`` out -- correct for
testing ``_LoopRegistry``'s scheduling, but it means the protocol code itself
(SSE framing, stream-before-send ordering, which events count, error vs idle
termination, text assembly) has never actually run under test. This file runs
it, over real HTTP, against ``fake_opencode.FakeOpencode``.

No network, no credentials, no model calls: the fake binds 127.0.0.1:0 and
every reply is scripted, so these are deterministic and fast.
"""

import threading
import time
import unittest
from unittest import mock

from weightslab.ui import server as ui_server

from tests.ui.fake_opencode import (
    FakeOpencode,
    assistant_message,
    session_error,
    session_idle,
    text_part,
    tool_part,
    user_message,
)

SES = "ses_1"


class _FakeServerTestCase(unittest.TestCase):
    def setUp(self):
        self.server = FakeOpencode().start()
        self.addCleanup(self.server.stop)

    def collect(self, timeout=10.0, session=SES, text="check in", model=None):
        return ui_server._opencode_send_and_collect(
            self.server.base_url, session, text, model=model, timeout=timeout,
        )


class TestPlainJsonRoutes(_FakeServerTestCase):
    def test_get_round_trip_parses_json(self):
        self.server.responses["GET /config"] = {"model": "openrouter/anthropic/claude-opus-4.6"}
        result = ui_server._opencode_json_request(self.server.base_url, "/config")
        self.assertEqual(result["model"], "openrouter/anthropic/claude-opus-4.6")

    def test_post_sends_a_json_body_and_content_type(self):
        ui_server._opencode_json_request(
            self.server.base_url, "/session", method="POST", body={"title": "loop"},
        )
        method, path, body = self.server.requests[-1]
        self.assertEqual((method, path), ("POST", "/session"))
        self.assertEqual(body, {"title": "loop"})

    def test_a_non_2xx_status_raises_rather_than_returning_a_falsy_value(self):
        # urllib raises HTTPError for 4xx/5xx; callers rely on that (the loop's
        # own start() catches it to report "could not reach OpenCode").
        self.server.post_status = 500
        with self.assertRaises(Exception):
            ui_server._opencode_json_request(
                self.server.base_url, f"/session/{SES}/message", method="POST", body={"parts": []},
            )

    def test_get_messages_passes_the_list_straight_through(self):
        self.server.messages = [{"info": {"id": "m1", "role": "assistant"}, "parts": []}]
        result = ui_server._opencode_get_messages(self.server.base_url, SES)
        self.assertEqual(result, self.server.messages)


class TestSendAndCollectHappyPath(_FakeServerTestCase):
    def test_assembles_the_reply_text_and_reports_no_error(self):
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "The run looks healthy."),
            session_idle(SES),
        ]
        text, error = self.collect()
        self.assertEqual(text, "The run looks healthy.")
        self.assertIsNone(error)

    def test_the_stream_is_opened_BEFORE_the_prompt_is_sent(self):
        # The fake only releases its scripted events when the POST arrives, so
        # collecting any text at all proves the subscription was already live.
        # Sending first and subscribing after would lose the whole turn.
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "caught it"),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "caught it")
        self.assertEqual([r[1] for r in self.server.requests if r[0] == "GET"], ["/event"])

    def test_a_later_delta_for_the_same_part_replaces_it_instead_of_appending(self):
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "Loss is "),
            text_part("p1", "m1", SES, "Loss is 0.31 and falling."),
            session_idle(SES),
        ]
        text, error = self.collect()
        self.assertEqual(text, "Loss is 0.31 and falling.")
        self.assertIsNone(error)

    def test_several_parts_are_joined_in_arrival_order(self):
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "First. "),
            text_part("p2", "m1", SES, "Second. "),
            text_part("p3", "m1", SES, "Third."),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "First. Second. Third.")

    def test_keep_alive_comments_are_skipped_not_parsed(self):
        self.server.send_keepalive = True
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "fine"),
            session_idle(SES),
        ]
        text, error = self.collect()
        self.assertEqual(text, "fine")
        self.assertIsNone(error)

    def test_the_selected_model_is_forwarded_on_the_prompt(self):
        self.server.script = [session_idle(SES)]
        model = {"providerID": "openrouter", "modelID": "anthropic/claude-opus-4.6"}
        self.collect(model=model)
        self.assertEqual(self.server.prompt_bodies[0]["model"], model)
        self.assertEqual(self.server.prompt_bodies[0]["parts"][0]["text"], "check in")

    def test_no_model_key_is_sent_when_none_was_resolved(self):
        self.server.script = [session_idle(SES)]
        self.collect(model=None)
        self.assertNotIn("model", self.server.prompt_bodies[0])

    def test_a_clean_turn_that_said_nothing_returns_empty_text_and_no_error(self):
        # Distinct from a failure: the loop reports "no reply" for this, which
        # is only correct because `error` is None here.
        self.server.script = [assistant_message("m1", SES), session_idle(SES)]
        text, error = self.collect()
        self.assertEqual(text, "")
        self.assertIsNone(error)


class TestSendAndCollectFiltering(_FakeServerTestCase):
    """Which events count toward the reply, and which must be ignored."""

    def test_ignores_text_from_a_different_session(self):
        self.server.script = [
            assistant_message("m_other", "ses_stranger"),
            text_part("p1", "m_other", "ses_stranger", "not for this loop"),
            assistant_message("m1", SES),
            text_part("p2", "m1", SES, "mine"),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "mine")

    def test_ignores_the_user_side_of_the_conversation(self):
        self.server.script = [
            user_message("m_user", SES),
            text_part("p1", "m_user", SES, "the prompt echoed back"),
            assistant_message("m1", SES),
            text_part("p2", "m1", SES, "the answer"),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "the answer")

    def test_ignores_non_text_parts_such_as_tool_calls(self):
        self.server.script = [
            assistant_message("m1", SES),
            tool_part("p1", "m1", SES, tool="bash"),
            text_part("p2", "m1", SES, "ran it"),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "ran it")

    def test_a_part_arriving_BEFORE_its_message_announcement_is_dropped(self):
        # Documents a real asymmetry with the browser client, which queues such
        # parts and replays them once the message id is known (agentChat.ts's
        # pendingParts). This reader has no queue: the id gate is checked once,
        # so an out-of-order part is lost. Harmless in practice -- OpenCode
        # announces the message first -- but worth pinning so the difference is
        # a decision rather than a surprise.
        self.server.script = [
            text_part("p1", "m1", SES, "early"),
            assistant_message("m1", SES),
            text_part("p2", "m1", SES, "late"),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "late")

    def test_a_malformed_event_payload_does_not_abort_the_turn(self):
        self.server.script = [
            {"not": "a valid event"},          # no `type`
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "survived"),
            session_idle(SES),
        ]
        text, error = self.collect()
        self.assertEqual(text, "survived")
        self.assertIsNone(error)

    def test_an_idle_for_a_DIFFERENT_session_does_not_end_this_turn(self):
        self.server.script = [
            assistant_message("m1", SES),
            session_idle("ses_stranger"),
            text_part("p1", "m1", SES, "still streaming after the other idle"),
            session_idle(SES),
        ]
        text, _ = self.collect()
        self.assertEqual(text, "still streaming after the other idle")


class TestSendAndCollectFailures(_FakeServerTestCase):
    def test_a_session_error_with_no_text_is_reported_as_an_error_not_silence(self):
        # The exact bug the (text, error) tuple exists for: a provider
        # rejecting the request produced an empty string that looked
        # identical to "the agent had nothing to say", so the job recorded no
        # error and the tab showed silence under the prompt every interval.
        self.server.script = [
            session_error(SES, "ProviderAuthError", "OpenRouter rejected the API key"),
        ]
        text, error = self.collect()
        self.assertEqual(text, "")
        self.assertEqual(error, "OpenRouter rejected the API key")

    def test_keeps_whatever_text_streamed_in_before_the_error(self):
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "Started looking, then "),
            session_error(SES, "APIError", "upstream 502"),
        ]
        text, error = self.collect()
        self.assertEqual(text, "Started looking, then ")
        self.assertEqual(error, "upstream 502")

    def test_falls_back_to_the_error_name_when_no_message_is_carried(self):
        self.server.script = [session_error(SES, "ContextOverflowError")]
        _text, error = self.collect()
        self.assertEqual(error, "ContextOverflowError")

    def test_an_error_for_a_different_session_is_ignored(self):
        self.server.script = [
            session_error("ses_stranger", "ProviderAuthError", "not mine"),
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "unaffected"),
            session_idle(SES),
        ]
        text, error = self.collect()
        self.assertEqual(text, "unaffected")
        self.assertIsNone(error)

    def test_a_rejected_prompt_with_no_text_collected_raises(self):
        # The loop's _fire catches this and records it as job.last_error, which
        # is what makes a failing check-in visible in its tab.
        self.server.post_status = 500
        self.server.script = []
        with self.assertRaises(Exception):
            self.collect(timeout=5.0)

    def test_the_stream_closing_early_degrades_to_the_text_collected_so_far(self):
        # A server restart mid-turn: EOF on the stream rather than an idle.
        self.server.script = [
            assistant_message("m1", SES),
            text_part("p1", "m1", SES, "partial answer"),
        ]
        stop = threading.Timer(0.3, self.server.end_stream)
        stop.start()
        self.addCleanup(stop.cancel)
        text, error = self.collect(timeout=5.0)
        self.assertEqual(text, "partial answer")
        self.assertIsNone(error)


class TestLoopEndToEndOverTheWire(unittest.TestCase):
    """_LoopRegistry driving a real (fake) server with NOTHING patched out --
    session creation, the check-in, and the transcript read all go over HTTP."""

    def setUp(self):
        self.server = FakeOpencode().start()
        self.addCleanup(self.server.stop)
        # The registry normally spawns/reuses an `opencode serve` child; point
        # it straight at the fake instead. This is the one thing still stubbed,
        # and deliberately: process spawning is covered by test_opencode_process.py.
        patcher = mock.patch.object(
            ui_server._opencode_session, "ensure",
            return_value={"ok": True, "url": self.server.base_url},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = ui_server._LoopRegistry()
        self.addCleanup(self.registry.shutdown)

    def test_a_started_loop_creates_a_session_and_records_its_first_check_in(self):
        self.server.responses["POST /session"] = {"id": "ses_loop"}
        self.server.script = [
            assistant_message("m1", "ses_loop"),
            text_part("p1", "m1", "ses_loop", "Training is progressing; loss 0.42."),
            session_idle("ses_loop"),
        ]

        result = self.registry.start("watch the loss", 60.0, "/tmp/ws", origin="http://localhost:8080")
        self.assertTrue(result.get("ok"), result)
        job_id = str(result["id"])

        job = self.registry._jobs[job_id]
        deadline = time.monotonic() + 10.0
        while job.last_result is None and job.last_error is None and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertIsNone(job.last_error)
        self.assertIn("loss 0.42", job.last_result or "")
        # The prompt that went out carries the monitoring preamble AND the task.
        sent = self.server.prompt_bodies[0]["parts"][0]["text"]
        self.assertIn("watch the loss", sent)
        self.assertIn("weightslab cli", sent)
        self.assertIn("NEVER stop/kill a process you did not yourself start", sent)

    def test_a_failing_check_in_is_recorded_as_the_jobs_error(self):
        self.server.responses["POST /session"] = {"id": "ses_loop"}
        self.server.script = [session_error("ses_loop", "ProviderAuthError", "no credentials")]

        result = self.registry.start("watch it", 60.0, "/tmp/ws", None)
        job = self.registry._jobs[str(result["id"])]
        deadline = time.monotonic() + 10.0
        while job.last_error is None and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertEqual(job.last_error, "no credentials")

    def test_the_tabs_transcript_is_read_from_the_servers_own_message_list(self):
        self.server.responses["POST /session"] = {"id": "ses_loop"}
        self.server.script = [session_idle("ses_loop")]
        self.server.messages = [
            {"info": {"id": "m1", "role": "assistant"}, "parts": [{"type": "text", "text": "all good"}]},
        ]
        result = self.registry.start("watch it", 60.0, "/tmp/ws", None)
        job_id = str(result["id"])
        deadline = time.monotonic() + 10.0
        job = self.registry._jobs[job_id]
        while job.session_id is None and time.monotonic() < deadline:
            time.sleep(0.05)

        messages = self.registry.get_messages(job_id)
        self.assertTrue(messages.get("ok"), messages)
        self.assertEqual(messages["messages"], self.server.messages)


if __name__ == "__main__":
    unittest.main()
