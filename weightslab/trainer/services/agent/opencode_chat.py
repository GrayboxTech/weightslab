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

`_ensure_reachable` (called at the top of every `_call`) is the other half
of that convergence: if `base_url` wasn't explicitly chosen and isn't
currently answering, it resolves or spawns one via
`weightslab/opencode_process.py`'s cross-process lock file, so this agent
and the browser landing-page chat end up on the SAME OpenCode server
regardless of which one happens to start first -- two separate sessions on
it (this class still creates a fresh one per call, as above), not one
shared session, since the two sides send incompatible message shapes
(structured-JSON/no-tools here, free-form/full-tools there) that would
otherwise bleed into each other's context.
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

# Last-resort model for _ensure_model_resolved, below: a free-tier model so a
# totally fresh OpenCode install (no provider credentials configured at all,
# so /config has no model and /config/providers has no defaults either)
# still gets a usable text-reasoning model, instead of leaving `self.model`
# unset -- which is exactly the "OpenCode picks WHATEVER model happens to be
# configured, arbitrarily" failure this method exists to avoid in the first
# place.
_DEFAULT_MODEL = "opencode/big-pickle"


class OpenCodeError(RuntimeError):
    """Raised when a call to the OpenCode server fails outright (not just an
    empty/partial reply -- those degrade to whatever text was collected)."""


class OpenCodeChat:
    """One instance per configured `(base_url, model)` pair; safe to reuse
    across calls (`_call` is the only state-touching method, and it is
    self-contained per invocation)."""

    def __init__(self, base_url: str, model: Optional[str] = None, timeout: float = 60.0,
                 workspace_dir: Optional[str] = None, url_is_explicit: bool = True,
                 model_is_explicit: bool = True, seed_model: Optional[str] = None):
        self.base_url = (base_url or "http://127.0.0.1:4096").rstrip("/")
        self.model = model
        self.timeout = timeout
        # Token usage from the LAST completed `_call`, or None before any call
        # has finished -- {"input", "output", "reasoning", "cache_read",
        # "cache_write"}. Populated by `_collect_reply`; read by
        # `DataManipulationAgent.get_context_usage()` for the /context command.
        # There is no persistent OpenCode session to total across calls (see
        # module docstring), so this is deliberately per-call, not cumulative.
        self.last_usage: Optional[dict] = None
        # See _ensure_reachable: workspace_dir is where opencode_process.py's
        # cross-process lock file for this experiment lives, and
        # url_is_explicit says whether base_url came from something the user
        # (or agent_config.yaml) actually chose -- if so, a dead address
        # stays dead rather than being silently swapped for an auto-spawned
        # one on a different port.
        self.workspace_dir = workspace_dir
        self.url_is_explicit = url_is_explicit
        # See _ensure_model_resolved: model_is_explicit says whether `model`
        # came from something the user (or agent_config.yaml) actually
        # chose -- if not, an empty model here isn't "let OpenCode pick",
        # it's "OpenCode already picks arbitrarily when none is given"
        # (confirmed live: an image-generation preview model, useless for
        # this class's structured-JSON-reply use case).
        self.model_is_explicit = model_is_explicit
        # A model from agent_config.yaml SEEDS the shared choice rather than
        # pinning it: it is what to use when nobody has chosen anything yet
        # (and it is then published, so the studio shows it), but a model
        # picked in the UI afterwards wins. It used to pin, so a run started
        # after picking a model in the studio quietly went back to the yaml
        # value. OPENCODE_MODEL stays a hard pin -- automation needs one.
        self.seed_model = (seed_model or "").strip() or None

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
    def _handle_event(
        payload: str,
        session_id: str,
        assistant_message_ids: set,
        text_parts: dict,
        usage: Optional[dict] = None,
    ) -> Optional[str]:
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
                # AssistantMessage.tokens (GET /doc): {input, output,
                # reasoning, cache: {read, write}} -- may be absent on early
                # deltas of the same message id, so this overwrites `usage`
                # in place rather than accumulating, same "latest wins"
                # convention weights_studio's TS client uses for its own
                # per-message token map.
                tokens = msg.get("tokens")
                if usage is not None and isinstance(tokens, dict):
                    cache = tokens.get("cache") or {}
                    usage["input"] = tokens.get("input") or 0
                    usage["output"] = tokens.get("output") or 0
                    usage["reasoning"] = tokens.get("reasoning") or 0
                    usage["cache_read"] = cache.get("read") or 0
                    usage["cache_write"] = cache.get("write") or 0
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
        usage: dict = {}

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
                        outcome = self._handle_event(payload, session_id, assistant_message_ids, text_parts, usage)
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

        # Set even on a dropped/errored stream (whatever was captured before
        # that point) -- a partial usage read is still more useful than none,
        # matching how a partial text reply is still returned above.
        self.last_usage = usage or None
        return "".join(text_parts[key] for key in text_parts)

    # -- public surface --------------------------------------------------- #

    def _ensure_reachable(self) -> None:
        """Self-heal `base_url` before the first network call of a turn, so
        this side and the browser landing-page chat converge on ONE OpenCode
        server (see weightslab/opencode_process.py) regardless of which one
        actually starts first -- without this, an agent constructed before
        anything else has ever needed a server would stay pointed at a dead
        default address for its entire life.

        Deliberately lazy (called here, not from __init__): constructing
        this class must stay fast/side-effect-free even when nothing is
        listening yet, since it is built unconditionally on every
        DataService startup whether or not the user ever opens the agent
        chat (see docs/agent.rst on why connectivity is never eagerly
        checked). A dead explicit URL (url_is_explicit=True) is left alone
        -- that address was deliberately chosen, not a placeholder to
        auto-replace.
        """
        if self.url_is_explicit:
            return
        from weightslab.opencode_process import opencode_healthy, resolve_or_spawn_opencode
        if opencode_healthy(self.base_url):
            return
        result = resolve_or_spawn_opencode(self.workspace_dir or ".")
        if result.get("ok") and result.get("url"):
            self.base_url = result["url"].rstrip("/")

    def _ensure_model_resolved(self) -> None:
        """Self-heal `self.model` before the first network call of a turn,
        same reasoning as _ensure_reachable but for the model instead of the
        address: leaving it unset does NOT mean "OpenCode picks a sensible
        default" -- it means OpenCode picks WHATEVER model happens to be
        configured, arbitrarily (confirmed live: an image-generation
        preview model was picked this way, and produced replies useless for
        this class's structured-JSON intent-parsing, since it isn't a
        text-reasoning model at all).

        Returns a short source label ("pinned" | "opencode-config" |
        "config-seed" | "default" | "kept") for the banner/logs.

        Resolution order:
          0. `OPENCODE_MODEL` (model_is_explicit) -- a hard pin, left alone.
          1. `GET /config`'s own `model` field -- the one the model picker
             writes back to opencode.json on every pick (opencodeClient.ts's
             setDefaultModel), so it's "whatever the user last actually
             chose", and survives across the browser, the CLI, and other
             machines. The ONLY thing that counts as an actual choice here.
          2. `_DEFAULT_MODEL` -- a free-tier, always-available text/tool
             model, used whenever step 1 comes back empty.

        `/config/providers`'s own `default` mapping used to be tried in
        between (what OpenCode itself would otherwise fall back to for
        whichever provider happens to be configured) -- dropped after this
        was confirmed live to still resolve to an arbitrary, sometimes
        non-text-reasoning model (an image-generation preview, the same
        failure mode this method exists to avoid) whenever ANY provider had
        credentials configured, even with no real model chosen -- silently
        pre-empting _DEFAULT_MODEL every time. "At UI init, with nothing
        explicitly chosen, land on the known-good free model" now means
        exactly that, with no provider-reported default able to override it.

        Re-checked before every turn, NOT cached for the life of the
        process: the studio's model picker (.wl-ag-model) writes the pick
        into OpenCode's own config via `PUT /config`, and a backend that
        latched onto the model it saw at startup went on answering with the
        old one for the rest of the run. `GET /config` is a local request on
        the same machine, so following it per turn costs nothing next to the
        completion it precedes. A failed read keeps the current model instead
        of falling back.

        An explicit model (model_is_explicit=True -- OPENCODE_MODEL or
        agent_config.yaml's `opencode_model`) is left alone: deliberately
        chosen, not a placeholder to override, and NOT overridable from the
        UI picker either.
        """
        if self.model_is_explicit:
            return "pinned"
        model_id = self._configured_model()
        if model_id:
            if model_id != self.model:
                _LOGGER.info("[OpenCodeChat] following OpenCode's configured "
                             "model: %s (was %s)", model_id, self.model or "unset")
            self.model = model_id
            return "opencode-config"
        # Nothing chosen anywhere yet -- fall back to the configured seed
        # before the built-in default, so a project's agent_config.yaml still
        # decides which model a fresh setup starts on.
        if self.seed_model:
            self.model = self.seed_model
            return "config-seed"
        # /config could not be read (or names no model): keep whatever was
        # resolved on an earlier turn rather than dropping a working model for
        # the fallback because one local request happened to fail.
        if not self.model:
            self.model = _DEFAULT_MODEL
            return "default"
        return "kept"

    def publish_model(self, model: Optional[str] = None) -> bool:
        """Write `model` (default: the resolved one) into OpenCode's own config.

        Same call the studio's model picker makes, so whichever side starts
        first leaves ONE answer behind for the other to read, and both ends of
        a session agree on the model without talking to each other.

        GLOBAL scope, with the workspace route as fallback: verified live,
        PATCH /config echoes the value back but does NOT change what GET
        /config then reports, while PATCH /global/config
        (`global.config.update`) does, immediately -- and GET /config is what
        both sides read.

        Best-effort: a read-only config or an older server must never stop the
        agent from working with the model it already resolved in-process.
        """
        model = model or self.model
        if not model:
            return False
        for path in ("/global/config", "/config"):
            try:
                with self._request(path, method="PATCH", body={"model": model}) as resp:
                    resp.read()
            except Exception as exc:  # noqa: BLE001 -- advisory write, never fatal
                _LOGGER.debug("[OpenCodeChat] could not publish model %s via %s: %s",
                              model, path, exc)
                continue
            # Confirm against the effective config rather than trusting the
            # echo: the workspace route answers 200 for a write it drops.
            if self._configured_model() == model:
                _LOGGER.info("[OpenCodeChat] published model %s to OpenCode (%s) at %s",
                             model, path, self.base_url)
                return True
        _LOGGER.debug("[OpenCodeChat] model %s could not be published to %s",
                      model, self.base_url)
        return False

    def _configured_model(self) -> Optional[str]:
        """The model OpenCode itself reports (GET /config) -- the shared choice
        the studio picker writes and this backend follows."""
        try:
            with self._request("/config") as resp:
                config = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
        model_id = (config or {}).get("model")
        return model_id if isinstance(model_id, str) and "/" in model_id else None

    def resolve_model(self, publish_default: bool = False):
        """Resolve the model NOW instead of lazily on the first turn, and say
        where it came from: ("pinned" | "pinned-published" | "opencode-config"
        | "default" | "default-published" | "kept").

        Called at agent start-up so the banner states the model actually in
        use -- it used to print "(server default)" whenever nothing was pinned,
        which read as "the studio's choice was ignored" even when the first
        turn would have picked it up correctly.

        With publish_default=True, the model is also written back to
        OpenCode's config whenever this side is the one deciding it -- a pinned
        model (OPENCODE_MODEL), a seed from agent_config.yaml, or the built-in
        fallback. A backend started BEFORE the studio then hands the UI the
        model it is itself using, instead of the studio showing an unrelated
        default while every backend query ran on the pinned one.

        A model that CAME from OpenCode's config is never re-published: there
        is nothing to write, and doing so would fight the picker.
        """
        self._ensure_reachable()
        source = self._ensure_model_resolved()
        if publish_default and source in ("default", "config-seed", "pinned")                 and self.publish_model():
            source = f"{source}-published"
        return self.model, source

    def _call(self, prompt_value):
        from langchain_core.messages import AIMessage

        self._ensure_reachable()
        self._ensure_model_resolved()
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
