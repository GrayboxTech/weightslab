"""LangChain-`Runnable`-compatible wrapper around a local OpenCode server.

OpenCode (https://opencode.ai) is a session-based, tool-using coding-agent
server -- not a plain chat-completions endpoint like OpenRouter/Ollama. This
module lets `DataManipulationAgent` (agent.py) use one as a third provider
without changing any of its three existing call sites (`query()`,
`generate_code()`, `generate_report_narrative()`), all of which do
`(prompt | chain).invoke(...)` and read `.content` off the result.

Mirrors the wire protocol already implemented in
weights_studio/src/landing/agent/opencodeClient.ts: create a session, send a
message, and collect the assistant's text from the global SSE event stream
ending on `session.idle` -- a bare POST .../message response is not a
reliable completion/content signal on its own (weights_studio's TS client
treats it the same way, with its own idle-driven finish as the source of
truth). The stream must be opened BEFORE the message is sent, or events
emitted in between are lost (same "stream-first" rule the TS client follows).

A fresh session is created per call rather than reused across calls:
`self.history` on `DataManipulationAgent` already carries cross-call context
via `INTENT_PROMPT`'s own `history=` placeholder, so OpenCode's own
multi-turn session memory isn't needed here, and a fresh session per call
avoids an ever-growing OpenCode-side history for a long-running experiment.

Every mutating tool (write/edit/patch/bash) is explicitly disabled on the
outgoing message: this call wants a text/JSON reply for the SDK agent to
parse and act on itself, not file writes as a side effect. That is the
opposite default from the Weights Studio landing-page chat, which
deliberately runs OpenCode's full toolset.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Mutating tools, disabled on every message this wrapper sends -- see module
# docstring. Named explicitly (not an allowlist) so a newly-added read-only
# tool on the OpenCode side keeps working without a change here.
_MUTATING_TOOLS = ("write", "edit", "patch", "bash")


class OpenCodeError(RuntimeError):
    """Raised when a call to the OpenCode server fails outright (not just an
    empty/partial reply -- those degrade to whatever text was collected)."""


class OpenCodeChat:
    """One instance per configured `(base_url, model)` pair; safe to reuse
    across calls (`_call` is the only state-touching method, and it is
    self-contained per invocation)."""

    def __init__(self, base_url: str, model: Optional[str] = None, timeout: float = 60.0):
        self.base_url = (base_url or "http://127.0.0.1:4096").rstrip("/")
        self.model = model
        self.timeout = timeout

    # -- wire helpers --------------------------------------------------- #

    def _model_ref(self) -> Optional[dict]:
        """OpenCode identifies a model as {providerID, modelID}; our config
        carries it as one "providerID/modelID" string (matching the exact
        convention used throughout the frontend -- see opencodeClient.ts's
        formatModelValue/parseModelValue). Split on the FIRST slash only:
        model IDs themselves often contain slashes (e.g. OpenRouter's
        "anthropic/claude-opus-4.6"), so a naive split would truncate it."""
        if not self.model or "/" not in self.model:
            return None
        provider_id, model_id = self.model.split("/", 1)
        return {"providerID": provider_id, "modelID": model_id}

    def _request(self, path: str, method: str = "GET", body: Optional[dict] = None, headers: Optional[dict] = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        all_headers = {"Content-Type": "application/json"} if body is not None else {}
        all_headers.update(headers or {})
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=all_headers, method=method)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _create_session(self) -> str:
        try:
            with self._request("/session", method="POST", body={"title": "weightslab-sdk-agent"}) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ValueError) as exc:
            raise OpenCodeError(f"Could not create an OpenCode session at {self.base_url}: {exc}") from exc
        session_id = data.get("id")
        if not session_id:
            raise OpenCodeError(f"OpenCode did not return a session id: {data!r}")
        return session_id

    def _send_message(self, session_id: str, text: str) -> None:
        body = {
            "parts": [{"type": "text", "text": text}],
            "tools": {name: False for name in _MUTATING_TOOLS},
        }
        model_ref = self._model_ref()
        if model_ref:
            body["model"] = model_ref
        try:
            self._request(f"/session/{session_id}/message", method="POST", body=body).read()
        except urllib.error.URLError as exc:
            raise OpenCodeError(f"OpenCode rejected the prompt: {exc}") from exc

    @staticmethod
    def _handle_event(payload: str, session_id: str, assistant_message_ids: set, text_parts: dict) -> Optional[str]:
        """Returns None while the turn is still in progress, or a string
        ("idle"/"error") once this session's turn has finished. Field names
        are read defensively -- OpenCode's event shapes are not part of its
        published docs, so a minor server change should degrade to "no text
        collected" rather than raising inside the stream loop."""
        try:
            event = json.loads(payload)
        except (ValueError, TypeError):
            return None
        event_type = event.get("type")
        props = event.get("properties") or {}

        if event_type == "message.updated":
            msg = props.get("info") or props.get("message") or props
            msg_session = str(msg.get("sessionID") or "")
            if msg_session and msg_session != session_id:
                return None
            if str(msg.get("role") or "") == "assistant" and msg.get("id"):
                assistant_message_ids.add(str(msg["id"]))
            return None

        if event_type == "message.part.updated":
            part = props.get("part") or props
            part_session = str(part.get("sessionID") or "")
            if part_session and part_session != session_id:
                return None
            message_id = str(part.get("messageID") or "")
            if message_id not in assistant_message_ids:
                return None
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                part_id = str(part.get("id") or message_id)
                text_parts[part_id] = part["text"]
            return None

        if event_type == "session.idle" and str(props.get("sessionID") or "") == session_id:
            return "idle"

        if event_type == "session.error" and str(props.get("sessionID") or "") == session_id:
            return "error"

        return None

    def _collect_reply(self, session_id: str, text: str) -> str:
        """Open the SSE stream, THEN send the message on a background thread,
        THEN read the stream until this session goes idle/errors. Ordering
        matters: opening the stream first means it starts buffering before the
        message is sent, so nothing emitted in the gap between "create
        session" and "start reading" is lost (see module docstring)."""
        text_parts: dict = {}
        assistant_message_ids: set = set()

        try:
            stream = self._request("/event", headers={"Accept": "text/event-stream"})
        except urllib.error.URLError as exc:
            raise OpenCodeError(f"Could not open the OpenCode event stream at {self.base_url}: {exc}") from exc

        send_errors: list = []

        def _send() -> None:
            try:
                self._send_message(session_id, text)
            except Exception as exc:  # noqa: BLE001 - surfaced via send_errors
                send_errors.append(exc)

        sender = threading.Thread(target=_send, daemon=True)

        try:
            sender.start()
            data_lines: list = []
            deadline = time.monotonic() + self.timeout
            for raw_line in stream:
                if time.monotonic() > deadline:
                    _LOGGER.warning("OpenCode event stream timed out waiting for session %s", session_id)
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines = []
                        outcome = self._handle_event(payload, session_id, assistant_message_ids, text_parts)
                        if outcome is not None:
                            break
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # A dropped stream degrades to whatever text was collected so far,
            # matching the browser client's "stream dropped -> don't leave the
            # turn hanging" behavior, rather than losing a partial reply.
            _LOGGER.warning("OpenCode event stream dropped for session %s: %s", session_id, exc)
        finally:
            stream.close()

        sender.join(timeout=1.0)
        if send_errors:
            raise send_errors[0]

        return "".join(text_parts[key] for key in text_parts)

    # -- public surface --------------------------------------------------- #

    def _call(self, prompt_value):
        from langchain_core.messages import AIMessage

        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        session_id = self._create_session()
        reply = self._collect_reply(session_id, text)
        return AIMessage(content=reply)

    def as_runnable(self):
        """A `Runnable` usable exactly like `ChatOpenAI`/`ChatOllama` in
        `(prompt | chain).invoke(...)` — the one integration point
        `_query_langchain`/`generate_code`/`generate_report_narrative` need."""
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(self._call)
