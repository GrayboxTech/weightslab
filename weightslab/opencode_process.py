"""Cross-process discovery/spawn for a single OpenCode server per workspace.

Two independent things in this codebase each want a local ``opencode serve``
process rooted at an experiment directory: ``weightslab/ui/server.py``'s
``_OpencodeSession`` (backing the browser landing-page chat and ``/loop``
jobs) and ``trainer/services/agent/agent.py``'s ``DataManipulationAgent``
(the backend SDK agent, reached via the gRPC query bar). Historically the
only way to make them share ONE server instead of each spawning its own was
a human manually exporting ``OPENCODE_URL`` into both processes' environments
before starting either -- fragile in practice (a new terminal tab doesn't
inherit anything exported in another one, especially on Windows), and it did
nothing for the "whichever starts first" case: env vars can't be pushed
backwards in time into a process that already started.

This module adds a second, automatic handoff for that case: a small JSON
lock file dropped in the experiment's own workspace directory recording
which URL a live server is already answering on. Whichever side needs a
server first spawns one and writes the file; whichever side comes second
finds the file, health-checks the URL it names, and adopts it instead of
spawning a duplicate. The workspace directory is *already* the one thing
both sides independently agree on (``weightslab start <dir>`` and a training
script configured with ``root_log_dir=os.environ["WEIGHTSLAB_ROOT_LOG_DIR"]``
point at the exact same directory by construction), so it needs no new
coordination from the user -- unlike the env var, which needed the same
value typed into two separate shells.

``OPENCODE_URL`` remains the explicit override it always was and is checked
first here too: an operator who has deliberately pointed both sides at a
specific address (CI, a shared dev server, ...) still wins outright.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger(__name__)

# Generous: a cold `npx` run downloads the package before the server binds.
OPENCODE_START_TIMEOUT = 45.0

# Dropped directly in the workspace directory, next to (and alongside) the
# AGENTS.md the landing-page agent already seeds there -- same "lives with
# the experiment" reasoning, and it means deleting/moving the experiment
# directory cleans this up too, with nothing to remember to do separately.
LOCK_FILENAME = ".wl_opencode.json"

# Seeded into a workspace before any OpenCode server is pointed at it. Both
# ship inside the package (pyproject.toml package-data) and are copied
# verbatim:
#
#   AGENTS.md      the weightslab integration reference. OpenCode loads a
#                  project-root AGENTS.md into the session context on its own
#                  (opencode.ai/docs/rules), and every spawn below uses
#                  cwd=workspace_dir, so this is what puts it in front of both
#                  chat surfaces -- including the /loop agent, which never
#                  fetches it over HTTP the way the landing chat can.
#   opencode.json  project config, found in that same directory
#                  (opencode.ai/docs/config: OpenCode "first looks for a
#                  config file in the current directory"). It names AGENTS.md
#                  in `instructions`, so the reference is declared explicitly
#                  rather than left to an auto-load traversal that depends on
#                  cwd and on locating a git root.
#
# Lives here rather than in ui/server.py because BOTH spawn paths need it and
# only one of them goes through the UI server: the backend SDK agent reaches
# resolve_or_spawn_opencode directly (OpenCodeChat._ensure_reachable), and a
# workspace it warms up first must not be left without them.
WORKSPACE_SEED_FILES = ("AGENTS.md", "opencode.json")


def _packaged_file(filename: str) -> Optional[Path]:
    """Locate a seed file shipped with the package.

    The package directory is what a `pip install` actually ships; the repo
    root is a checkout-only fallback for anything not (yet) moved inside.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent / filename, here.parents[1] / filename):
        if candidate.is_file():
            return candidate
    return None


def _default_shell() -> Optional[str]:
    """The shell OpenCode's exec tool should use in a seeded workspace, picked
    from what this machine actually has -- not left to OpenCode's own
    auto-discovery.

    On Windows that auto-discovery has picked inconsistently between
    cmd.exe, PowerShell and Git Bash depending on what happens to be on
    PATH (OpenCode issue #16479: its bash-tool prompt doesn't tell the model
    which shell it landed in), which is exactly what produced a live session
    retrying `sleep`/`cat`/`ps aux`, then `type`, across several failed
    turns before giving up on guessing. Every spawn in this module uses
    cwd=workspace_dir on THIS host (see resolve_or_spawn_opencode), so
    ``os.name`` here is authoritative for what the exec tool will run on --
    pinning it via opencode.json's ``shell`` key turns that guess into a
    stated fact the preamble can hand the model outright.

    POSIX already gets a sensible default (zsh/bash) from OpenCode itself
    with no observed mismatch, so only Windows needs pinning here.
    """
    if os.name != "nt":
        return None
    return "pwsh" if shutil.which("pwsh") else "powershell"


def _seed_opencode_config(source: Path, target: Path) -> None:
    """Copy opencode.json, pinning ``shell`` to _default_shell()'s answer.

    Parsed and re-serialized rather than byte-copied like AGENTS.md: the
    shipped template is deliberately platform-agnostic (checked into one
    repo, built on any OS), and the ``shell`` value can only be decided on
    the machine actually running this seed.
    """
    with open(source, "r", encoding="utf-8") as f:
        config = json.load(f)
    shell = _default_shell()
    if shell:
        config["shell"] = shell
    with open(target, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def ensure_workspace_agent_files(workspace_dir: str) -> None:
    """Copy WORKSPACE_SEED_FILES into ``workspace_dir`` if absent.

    Never overwrites: an existing AGENTS.md or opencode.json may be the
    user's OWN project instructions/config, which are not ours to replace.
    Best-effort per file, so one missing from the install or one unwritable
    target can't stop the other from being seeded.
    """
    for filename in WORKSPACE_SEED_FILES:
        target = Path(workspace_dir) / filename
        if target.exists():
            continue
        source = _packaged_file(filename)
        if source is None:
            continue
        try:
            if filename == "opencode.json":
                _seed_opencode_config(source, target)
            else:
                shutil.copyfile(source, target)
        except OSError:
            _LOGGER.debug("Could not seed %s into %s", filename, workspace_dir)


def shell_platform_note() -> dict:
    """Describe the exec-tool shell pinned by ensure_workspace_agent_files(),
    for server.py to hand to the browser (``/agent-server/start``'s response)
    so the landing-chat preamble can state the fact outright instead of
    hedging "if you're on Windows..." -- see _default_shell()'s docstring
    for why that hedge produced retry-guessing in practice. ``{"platform":
    "posix"}`` when nothing is pinned (this machine already gets a sensible
    default from OpenCode itself).
    """
    shell = _default_shell()
    if shell:
        return {"platform": "windows", "shell": shell}
    return {"platform": "posix"}

# Applied whenever this module spawns a fresh server with no browser Origin
# to base --cors on (the backend SDK agent has no such thing -- it never
# talks to OpenCode from a browser). These are the same origins `npm run
# agent`/`weightslab start` already treat as the standard dev/prod pair
# (localhost and 127.0.0.1, both the Vite dev port and weightslab's own
# default), so a browser landing-page chat that adopts this server later via
# the lock file is very likely covered without needing its own spawn at all.
# OpenCode's CORS list is fixed at process start, so this is a best-effort
# hedge, not a guarantee, for a workspace served on some other port.
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
)


def lock_path(workspace_dir: str) -> Path:
    return Path(workspace_dir) / LOCK_FILENAME


def read_lock(workspace_dir: str) -> Optional[dict]:
    try:
        with open(lock_path(workspace_dir), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("url"):
        return None
    return data


def write_lock(workspace_dir: str, url: str, pid: Optional[int] = None) -> None:
    try:
        os.makedirs(workspace_dir, exist_ok=True)
        with open(lock_path(workspace_dir), "w", encoding="utf-8") as f:
            json.dump({"url": url, "pid": pid}, f)
    except OSError:
        # Best-effort: a later caller simply won't find this server via the
        # lock file and will spawn its own instead. Not worth failing the
        # caller's own successful spawn over a write it doesn't need for
        # itself.
        _LOGGER.warning("Could not write OpenCode lock file under %s", workspace_dir)


def free_port() -> int:
    """Reserve an unused loopback port by binding and releasing it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resolve_opencode_argv() -> Optional[list]:
    """Locate a way to run OpenCode, preferring an already-installed binary.

    Falls back to ``npx --yes``, which fetches the package into the npx
    cache on first use -- deliberately not a global ``npm i -g``, which can
    need elevated permissions and mutates the user's toolchain silently.
    """
    exe = shutil.which("opencode")
    if exe:
        return [exe]
    npx = shutil.which("npx")
    if npx:
        return [npx, "--yes", "opencode-ai@latest"]
    return None


def opencode_healthy(base_url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/global/health", timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _cors_origin_variants(origin: Optional[str]) -> list:
    """`localhost` and `127.0.0.1` are different origins to a CORS check --
    allow both spellings of whichever one we were given."""
    origins: list = []

    def add(value: str) -> None:
        if value and value not in origins:
            origins.append(value)

    if origin:
        add(origin)
        if "localhost" in origin:
            add(origin.replace("localhost", "127.0.0.1"))
        elif "127.0.0.1" in origin:
            add(origin.replace("127.0.0.1", "localhost"))
    return origins


def _kill_quietly(process: "subprocess.Popen") -> None:
    """Best-effort teardown for a spawn that turned out to be redundant (lost
    a race to another process's spawn, or never became healthy) -- NOT the
    general "stop a server someone is using" path, which this module has no
    opinion on: once adopted (by us or anyone else), nobody here kills it on
    exit. See the module docstring's reasoning on why a server discovered
    through this module is always treated as a detached, shared resource.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            process.terminate()
        process.wait(timeout=5)
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def resolve_or_spawn_opencode(workspace_dir: str, origin: Optional[str] = None,
                               timeout: float = OPENCODE_START_TIMEOUT) -> dict:
    """Find a live OpenCode server for ``workspace_dir``, spawning one only
    if nothing already answers for it. Returns ``{"ok": True, "url": ...,
    "source": "env"|"lockfile"|"spawned"}`` or ``{"ok": False, "error": ...}``.

    Precedence: an explicit ``OPENCODE_URL`` env var (if it's actually
    healthy) always wins -- an operator who set that deliberately is opting
    out of auto-discovery, not asking for it. Then the lock file for this
    workspace, if its recorded URL is still healthy (a dead/stale entry --
    the process it named exited -- is treated the same as no file at all).
    Only then does this spawn a new one.
    """
    # Before resolving anything, not just before spawning: an ADOPTED server
    # (env/lockfile) still creates its sessions in this workspace, and reads
    # both files per session rather than once at startup.
    ensure_workspace_agent_files(workspace_dir)

    external = os.environ.get("OPENCODE_URL", "").strip()
    if external and opencode_healthy(external):
        return {"ok": True, "url": external, "source": "env"}

    lock = read_lock(workspace_dir)
    if lock and opencode_healthy(lock["url"]):
        return {"ok": True, "url": lock["url"], "source": "lockfile"}

    argv = resolve_opencode_argv()
    if argv is None:
        return {
            "ok": False,
            "error": "Could not find `opencode` or `npx`. Install Node.js 20+ "
                     "(which provides npx), or `npm i -g opencode-ai`.",
        }

    port = free_port()
    cors = list(_cors_origin_variants(origin))
    for value in DEFAULT_CORS_ORIGINS:
        if value not in cors:
            cors.append(value)
    cmd = argv + ["serve", "--hostname", "127.0.0.1", "--port", str(port)]
    for value in cors:
        cmd += ["--cors", value]

    try:
        process = subprocess.Popen(
            cmd, cwd=workspace_dir,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            # Detached: this server is meant to outlive whichever side
            # happened to spawn it (the other side may still be using it
            # long after this process exits), so nobody's shutdown path
            # should take it down as a side effect. See _kill_quietly's
            # docstring for the one exception (a redundant spawn that lost
            # a race, below).
            start_new_session=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": str(exc)}

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if opencode_healthy(base_url):
            # Another process may have won the same race and already
            # published its own server for this workspace while we were
            # spawning ours -- defer to it (first writer wins) rather than
            # leaving two live servers for one workspace.
            raced = read_lock(workspace_dir)
            if raced and raced["url"] != base_url and opencode_healthy(raced["url"]):
                _kill_quietly(process)
                return {"ok": True, "url": raced["url"], "source": "lockfile"}
            write_lock(workspace_dir, base_url, process.pid)
            return {"ok": True, "url": base_url, "source": "spawned", "pid": process.pid}
        time.sleep(0.4)

    _kill_quietly(process)
    return {"ok": False, "error": f"The agent server did not come up within {int(timeout)}s."}
