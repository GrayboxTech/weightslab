#!/usr/bin/env python3
"""End-to-end agent smoke test for a *pip-installed* weightslab.

Run against a clean environment where weightslab was installed from a wheel and
Node.js is deliberately absent. It proves the Option-2 promise: after
``pip install weightslab`` the OpenCode agent works with no manual install.

Modes (argv[1]):
  provision  Provision the managed OpenCode binary and run `--version`.
             This is "initializing opencode" with no Node/npx on the box.
  start      `weightslab start` (the UI): boot it headless, then drive
             POST /agent-server/start + GET /agent-server/status. Asserts the
             agent server comes up WITHOUT any credential configured -- i.e. an
             unconfigured user still gets a running agent they can then configure
             (opencode auth login / the landing login modal), rather than a hard
             failure. ("allow user to configure agent if not already done")
  example    `weightslab start example`: boot the bundled training example and
             assert it starts cleanly and never hits the "no opencode/npx" path.
  all        provision, then start, then example.

Exit code is non-zero on the first failure, with a clear reason.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

OPENCODE_MISSING_MARKER = "Could not provision OpenCode"
NOT_CONFIGURED_MARKER = "not initialized"
INSTALL_MARKERS = ("installing now", "OpenCode installed", "OpenCode ready")
STARTUP_BUDGET = 90.0          # UI / agent readiness
EXAMPLE_MIN_UPTIME = 60.0      # example must survive this long past import
EXAMPLE_BUDGET = 300.0


def log(msg: str) -> None:
    print(f"[agent-smoke] {msg}", flush=True)


def fail(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    log(f"FAIL: {msg}")
    sys.exit(1)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_get(url: str, timeout: float = 3.0):
    req = urllib.request.Request(url, headers={"Origin": "http://127.0.0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def http_post(url: str, timeout: float = 60.0):
    req = urllib.request.Request(
        url, data=b"{}", method="POST",
        headers={"Content-Type": "application/json", "Origin": "http://127.0.0.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def poll_until(fn, budget: float, what: str):
    deadline = time.monotonic() + budget
    last = None
    while time.monotonic() < deadline:
        try:
            if fn():
                return True
        except Exception as exc:  # not up yet
            last = exc
        time.sleep(1.0)
    log(f"timed out waiting for {what} ({last})")
    return False


class Proc:
    """A weightslab subprocess with combined-output capture and tree kill."""

    def __init__(self, args, env=None):
        self.args = args
        self.lines = []
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            env=env or os.environ.copy(),
            start_new_session=True,  # own process group so we can kill the tree
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        for line in self._proc.stdout:
            self.lines.append(line.rstrip("\n"))
            print(f"    | {line.rstrip()}", flush=True)

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def alive(self) -> bool:
        return self._proc.poll() is None

    def returncode(self):
        return self._proc.poll()

    def stop(self):
        if self._proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except Exception:
            self._proc.terminate()
        try:
            self._proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except Exception:
                self._proc.kill()


def mode_provision() -> None:
    log("provisioning managed OpenCode binary (no Node.js expected on PATH)...")
    from weightslab import opencode_binary

    if _which("npx") or _which("node"):
        log("note: Node is present; the managed path is still exercised explicitly")

    path = opencode_binary.ensure_managed_binary()
    if not path:
        fail("ensure_managed_binary() returned None -- provisioning failed")
    log(f"managed binary: {path}")

    out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        fail(f"`opencode --version` failed (rc={out.returncode}): {out.stderr.strip()}")
    log(f"opencode --version -> {out.stdout.strip() or out.stderr.strip()}")

    # And confirm the resolver actually selects it.
    from weightslab import opencode_process
    argv = opencode_process.resolve_opencode_argv()
    if not argv or Path(argv[0]) != Path(path):
        fail(f"resolver did not select the managed binary: {argv}")
    log("resolver selects the managed binary. provision OK")


def _which(name: str):
    from shutil import which
    return which(name)


def mode_start() -> None:
    port = free_port()
    workspace = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"wl-smoke-start-{port}"
    workspace.mkdir(parents=True, exist_ok=True)
    log(f"launching `weightslab start` on port {port} (workspace {workspace})...")

    proc = Proc([
        "weightslab", "start", str(workspace),
        "--no-browser", "--host", "127.0.0.1", "--port", str(port),
    ])
    try:
        base = f"http://127.0.0.1:{port}"
        # /agent-server/status always answers 200 JSON once the HTTP server is
        # up, independent of whether the bundled SPA assets are present -- a more
        # robust readiness probe than "/" (which 404s on an assets-less build).
        if not poll_until(lambda: http_get(base + "/agent-server/status")[0] == 200,
                          STARTUP_BUDGET, "UI server"):
            fail(f"UI did not serve on {base}\n---\n{proc.output}")
        log("UI is serving")

        # No credential is configured in CI. The agent server must still come up
        # -- the user configures the model/login afterwards. That is the whole
        # "configure agent if not already done" guarantee.
        status, body = http_post(base + "/agent-server/start", timeout=STARTUP_BUDGET)
        payload = json.loads(body or b"{}")
        if OPENCODE_MISSING_MARKER in proc.output:
            fail("agent start hit the no-opencode path despite a pip install")
        if not payload.get("ok"):
            fail(f"/agent-server/start not ok: {payload}")
        if not payload.get("url"):
            fail(f"/agent-server/start returned no url: {payload}")
        log(f"agent server up (unconfigured) at {payload['url']}")

        # Status endpoint should now report the running agent.
        s_status, s_body = http_get(base + "/agent-server/status")
        log(f"/agent-server/status -> {s_status} {s_body[:200]!r}")
        log("start mode OK")
    finally:
        proc.stop()


def mode_example() -> None:
    """`weightslab start example` is pure training: the agent is lazy and
    optional. With NO agent configured it must (c) boot cleanly, (c) never hit
    the no-opencode path, and just log an info hint -- no init, no error. We
    deliberately do NOT provision opencode here (that would be an init the user
    never asked for)."""
    workspace = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "wl-smoke-example"
    workspace.mkdir(parents=True, exist_ok=True)
    log("launching `weightslab start example` with NO agent configured...")

    # Force the unconfigured state so the info-hint path is what we test.
    env = {**os.environ, "WEIGHTSLAB_SUPPRESS_BANNER": "1"}
    env.pop("OPENCODE_URL", None)
    proc = Proc(["weightslab", "start", "example"], env=env)
    try:
        start = time.monotonic()
        while time.monotonic() - start < EXAMPLE_BUDGET:
            if OPENCODE_MISSING_MARKER in proc.output:
                fail("example surfaced an opencode error despite the agent being optional")
            rc = proc.returncode()
            if rc is not None:
                if rc == 0:
                    log("example exited 0 during boot window")
                    break
                fail(f"example exited early with rc={rc}\n---\n{proc.output}")
            if time.monotonic() - start >= EXAMPLE_MIN_UPTIME:
                log(f"example stayed up {int(EXAMPLE_MIN_UPTIME)}s with no opencode error")
                break
            time.sleep(2.0)
        # Soft checks (may land slightly after boot): the "no init, just info"
        # sign-in hint, and the background install being logged.
        if NOT_CONFIGURED_MARKER in proc.output:
            log("info hint present: user told how to `weightslab agent init`")
        else:
            log("note: agent-config info hint not observed in captured output")
        if any(m in proc.output for m in INSTALL_MARKERS):
            log("opencode install was logged during the example run")
        else:
            log("note: opencode install log not observed (may finish after window)")
        log("example mode OK")
    finally:
        proc.stop()


def mode_cli_init() -> None:
    """(b) The user can initialize the agent from the CLI. Exercise the
    headless path: `weightslab agent init --provision-only` must provision a
    working opencode with no Node and no interactive prompt."""
    log("running `weightslab agent init --provision-only`...")
    out = subprocess.run(
        ["weightslab", "agent", "init", "--provision-only"],
        capture_output=True, text=True, timeout=300,
    )
    combined = (out.stdout or "") + (out.stderr or "")
    print(combined, flush=True)
    if out.returncode != 0:
        fail(f"`weightslab agent init --provision-only` exited {out.returncode}")
    if "OpenCode ready" not in combined:
        fail("agent init did not report a provisioned OpenCode binary")

    from weightslab import opencode_binary
    path = opencode_binary.find_managed_binary()
    if not path:
        fail("agent init reported success but no managed binary is present")
    ver = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=60)
    if ver.returncode != 0:
        fail(f"provisioned opencode failed `--version` (rc={ver.returncode})")
    log(f"cli init OK — opencode {ver.stdout.strip() or ver.stderr.strip()} at {path}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "provision":
        mode_provision()
    elif mode == "start":
        mode_provision()
        mode_start()
    elif mode == "example":
        # No provisioning: the example must be clean and agent-free on its own.
        mode_example()
    elif mode == "cli-init":
        mode_cli_init()
    elif mode == "all":
        mode_provision()
        mode_start()
        mode_cli_init()
        mode_example()
    else:
        fail(f"unknown mode {mode!r}")
    log(f"mode {mode!r}: PASS")


if __name__ == "__main__":
    main()
