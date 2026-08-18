"""A fake OpenCode server, over real HTTP, for the backend agent tests.

WHY. ``weightslab/ui/server.py`` talks to OpenCode with three stdlib helpers:
``_opencode_json_request`` (plain JSON round trip), ``_opencode_get_messages``
(a passthrough), and ``_opencode_send_and_collect`` -- which is where all the
real complexity lives. That one opens the SSE event stream FIRST, sends the
prompt from a second thread, then reads the stream until this session goes
idle, assembling reply text per part id and distinguishing "the turn ended
having said nothing" from "the turn failed".

Every existing test patches those helpers out (see test_server_loop.py's own
header), which is right for exercising ``_LoopRegistry``'s scheduling logic but
means the wire layer itself -- stream-before-send ordering, SSE framing,
keep-alive comments, which events count and which are ignored, error vs idle
termination -- has never been executed against anything. This serves the real
protocol on a real socket so it is.

Deliberately stdlib-only, matching the module it tests (``ui/server.py`` is
stdlib-only by design so ``weightslab start`` needs no extra dependency).

USAGE::

    with FakeOpencode() as server:
        server.script = [
            {"type": "message.updated",
             "properties": {"info": {"id": "m1", "role": "assistant",
                                     "sessionID": "ses_1"}}},
            {"type": "message.part.updated",
             "properties": {"part": {"id": "p1", "messageID": "m1",
                                     "sessionID": "ses_1",
                                     "type": "text", "text": "hello"}}},
            {"type": "session.idle", "properties": {"sessionID": "ses_1"}},
        ]
        text, error = ui_server._opencode_send_and_collect(
            server.base_url, "ses_1", "check in")

The scripted events are emitted when the prompt POST arrives, which is what a
real server does: the turn only starts once the message is sent.
"""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpencode:
    """A scriptable OpenCode stand-in on 127.0.0.1, speaking real HTTP/SSE."""

    def __init__(self):
        #: Events emitted (in order) once the prompt POST arrives.
        self.script: list = []
        #: HTTP status the prompt POST answers with.
        self.post_status: int = 200
        #: Seconds the prompt POST blocks before answering, to exercise the
        #: "the send itself is what times out" path.
        self.post_delay: float = 0.0
        #: Emit a `:` keep-alive comment ahead of the scripted events -- real
        #: servers do, and the reader must skip them rather than treat one as
        #: a truncated event.
        self.send_keepalive: bool = True
        #: Replies for the plain JSON routes, by "METHOD /path".
        self.responses: dict = {}
        #: What GET /session/<id>/message returns.
        self.messages: list = []

        #: Every request received, as (method, path, parsed-body-or-None).
        self.requests: list = []
        #: Bodies POSTed to /session/<id>/message specifically.
        self.prompt_bodies: list = []

        self._events: "queue.Queue" = queue.Queue()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeOpencode":
        self._thread.start()
        return self

    def stop(self) -> None:
        # Unblock any streaming handler still parked on the queue, so
        # shutdown() isn't waiting on a request that never finishes.
        self._events.put(None)
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "FakeOpencode":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    # -- scripting ---------------------------------------------------------
    def emit(self, event: dict) -> None:
        """Push one event onto the live stream, outside the scripted batch."""
        self._events.put(event)

    def end_stream(self) -> None:
        """Close the event stream from the server side (EOF), as a restarting
        or crashing server would."""
        self._events.put(None)

    def _release_script(self) -> None:
        for event in self.script:
            self._events.put(event)


def _make_handler(state: FakeOpencode):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: the response body is delimited by connection close, so a
        # streamed SSE body needs no chunked framing and the client's
        # line-by-line read still sees each event as it is written.
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args) -> None:
            pass  # keep the test output clean

        # -- helpers --
        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return raw

        def _send_json(self, payload, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream_events(self) -> None:
            """GET /event -- hold the connection open and write SSE frames as
            they are queued, until a None sentinel or the client hangs up."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                if state.send_keepalive:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                while True:
                    event = state._events.get()
                    if event is None:
                        return
                    frame = f"data: {json.dumps(event)}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # The reader breaks out on session.idle/session.error and
                # closes its end -- an expected, ordinary hang-up here.
                return

        # -- verbs --
        def do_GET(self) -> None:
            state.requests.append(("GET", self.path, None))
            if self.path == "/event":
                self._stream_events()
                return
            if self.path.startswith("/session/") and self.path.endswith("/message"):
                self._send_json(state.messages)
                return
            key = f"GET {self.path}"
            if key in state.responses:
                self._send_json(state.responses[key])
                return
            self._send_json({})

        def do_POST(self) -> None:
            body = self._read_body()
            state.requests.append(("POST", self.path, body))

            if self.path.startswith("/session/") and self.path.endswith("/message"):
                state.prompt_bodies.append(body)
                # The turn starts on send -- release the scripted events now,
                # which is also what makes the stream-first ordering in
                # _opencode_send_and_collect observable: had it sent first and
                # subscribed after, it would miss everything below.
                state._release_script()
                if state.post_delay:
                    import time
                    time.sleep(state.post_delay)
                if state.post_status != 200:
                    self._send_json({"error": "prompt rejected"}, status=state.post_status)
                    return
                self._send_json({"ok": True})
                return

            key = f"POST {self.path}"
            if key in state.responses:
                self._send_json(state.responses[key])
                return
            self._send_json({})

        def do_DELETE(self) -> None:
            state.requests.append(("DELETE", self.path, None))
            self._send_json({})

    return Handler


# --------------------------------------------------------------------------
# Event builders -- the shapes ui/server.py's reader actually looks for, in one
# place so a protocol change lands here rather than across every test.
# --------------------------------------------------------------------------

def assistant_message(message_id: str, session_id: str) -> dict:
    return {
        "type": "message.updated",
        "properties": {"info": {"id": message_id, "role": "assistant", "sessionID": session_id}},
    }


def user_message(message_id: str, session_id: str) -> dict:
    return {
        "type": "message.updated",
        "properties": {"info": {"id": message_id, "role": "user", "sessionID": session_id}},
    }


def text_part(part_id: str, message_id: str, session_id: str, text: str) -> dict:
    return {
        "type": "message.part.updated",
        "properties": {"part": {
            "id": part_id, "messageID": message_id, "sessionID": session_id,
            "type": "text", "text": text,
        }},
    }


def tool_part(part_id: str, message_id: str, session_id: str, tool: str = "bash") -> dict:
    return {
        "type": "message.part.updated",
        "properties": {"part": {
            "id": part_id, "messageID": message_id, "sessionID": session_id,
            "type": "tool", "tool": tool, "state": {"status": "completed"},
        }},
    }


def session_idle(session_id: str) -> dict:
    return {"type": "session.idle", "properties": {"sessionID": session_id}}


def session_error(session_id: str, name: str, message: str = None) -> dict:
    error = {"name": name}
    if message is not None:
        error["data"] = {"message": message}
    return {"type": "session.error", "properties": {"sessionID": session_id, "error": error}}
