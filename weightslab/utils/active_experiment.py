"""Cross-process handoff of the active experiment directory.

``weightslab start`` establishes the experiment directory (checkpoints, logs,
``notebooks/``, ``reports/``) and exports ``WEIGHTSLAB_ROOT_LOG_DIR`` -- but it
can only export it into ITS OWN process. A training run launched from a second
terminal, or by ``weightslab start example``, is a different process tree and
never saw that variable: it fell through to a throwaway ``tempfile.mkdtemp()``,
so the run wrote into ``%TEMP%\\tmpXXXXXXXX`` while the UI listed an empty
``reports/`` (the "right-click Generate Report shows nothing, yet I generated
reports" symptom) from the directory it had established itself.

Hence this marker file: one small JSON document, per user, that both sides
write to and read from. Each side is a LIST, because running two experiments
side by side -- a classification UI on one port, a segmentation UI on another --
is a supported thing to do. With a single slot per side, the second
``weightslab start`` erased the first, and both UIs would then have listed the
reports of whichever backend started last.

    {
      "ui":      [{"root_log_dir": "...", "pid": 123, "ui_port": 8080,
                   "backend_port": 50051, "updated_at": "..."}, ...],
      "backend": [{"root_log_dir": "...", "pid": 456, "grpc_port": 50051,
                   "updated_at": "..."}, ...]
    }

* ``ui`` is written by ``weightslab start`` -- the directory it established.
  A later training process with nothing configured adopts it, which is what
  makes the two halves land in the same experiment.
* ``backend`` is written by ``wl.serve()`` -- the directory training ACTUALLY
  resolved, whatever the route (an explicit ``root_log_dir:`` in a config file,
  the environment, or the ``ui`` value above). The UI prefers it when listing
  reports and notebooks, so those lists stay right even when training was
  pointed somewhere the UI never chose.

Entries are keyed by pid: a process replaces its own, dead ones are pruned on
every write, and the list is bounded. Readers never guess -- a port pins the
entry when the caller knows one (a UI asking for ITS backend), a single live
entry is unambiguous, and two or more return nothing with a line in the log,
leaving the caller its own directory. Showing one experiment's reports inside
another experiment's UI, or writing a run into the wrong experiment, is worse
than declining to answer.

Neither side is required: every reader validates that the recorded directory
still exists and falls back to its previous behaviour otherwise, and every
write is best-effort -- a read-only home directory must never stop a run.

Set ``WEIGHTSLAB_STATE_DIR`` to relocate the file (tests use it to stay out of
the developer's real state).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_FILE_NAME = "active_experiment.json"
_SECTIONS = ("ui", "backend")
# A long-lived machine should not accumulate entries nobody can attribute.
_MAX_ENTRIES = 16


def state_dir() -> Path:
    """Directory holding the marker: ``$WEIGHTSLAB_STATE_DIR`` or ``~/.weightslab``."""
    override = (os.environ.get("WEIGHTSLAB_STATE_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".weightslab"


def state_path() -> Path:
    """Absolute path of the marker file (may not exist yet)."""
    return state_dir() / _FILE_NAME


def read_state() -> dict:
    """The whole marker, or ``{}`` when it is absent or unreadable."""
    path = state_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 -- a corrupt marker is not fatal
        logger.debug("[active-experiment] could not read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def entries(section: str, state: Optional[dict] = None) -> list:
    """One section as a list, accepting the older single-object shape."""
    value = (read_state() if state is None else state).get(section)
    if isinstance(value, list):
        return [e for e in value if isinstance(e, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _pid_is_running(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:  # noqa: BLE001 -- psutil missing/unusable: assume gone
        return False


class _MarkerLock:
    """Brief exclusive hold on the marker, via an O_EXCL lock file.

    Read-modify-write on a shared file loses updates when two processes do it
    at once, and two processes doing it at once is precisely the case this
    marker exists for: starting a classification UI and a segmentation UI
    together dropped one of their port stamps, which is the field that tells
    them apart afterwards.

    Best-effort by design: if the lock cannot be taken (a stale file nobody
    cleaned up, a read-only home), the write proceeds anyway -- an advisory
    record must never block a run. A stale lock older than a few seconds is
    broken on purpose, since nothing here holds it for more than a file write.
    """

    STALE_AFTER = 5.0

    def __init__(self, path: Path, attempts: int = 60, delay: float = 0.02):
        self._path = Path(str(path) + ".lock")
        self._attempts = attempts
        self._delay = delay
        self._held = False

    def __enter__(self):
        import time
        for _ in range(self._attempts):
            try:
                fd = os.open(str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self._held = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self._path)
                    if age > self.STALE_AFTER:
                        os.unlink(self._path)
                        continue
                except OSError:
                    pass
                time.sleep(self._delay)
            except OSError:
                break  # cannot lock here at all; proceed unlocked
        return self

    def __exit__(self, *exc):
        if self._held:
            try:
                os.unlink(self._path)
            except OSError:
                pass
        return False


def _write_section(section: str, root_log_dir, **meta) -> Optional[Path]:
    """Record this process's entry in one section, keeping the others.

    Replaces the entry for this pid (a process re-recording, e.g. once its port
    is known), drops entries whose process is gone, and leaves every other live
    entry in place -- that is what lets two experiments run side by side. Held
    under _MarkerLock so two processes starting together cannot lose each
    other's entry.
    """
    if section not in _SECTIONS:
        raise ValueError(f"unknown section {section!r}")
    if not root_log_dir:
        return None

    pid = os.getpid()
    entry = {
        "root_log_dir": str(Path(root_log_dir).expanduser().resolve()),
        "pid": pid,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry.update({k: v for k, v in meta.items() if v is not None})

    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _MarkerLock(path):
            # Re-read INSIDE the lock: another process may have added its own
            # entry since this function started.
            state = read_state()
            kept = [e for e in entries(section, state)
                    if e.get("pid") != pid and _pid_is_running(e.get("pid"))]
            state[section] = (kept + [entry])[-_MAX_ENTRIES:]
            # Written via a temp file in the same directory, then replaced, so a
            # concurrent reader never sees a half-written document.
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".active-", suffix=".json")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(state, fh, indent=2)
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
    except Exception as exc:  # noqa: BLE001 -- advisory record, never fatal
        logger.debug("[active-experiment] could not record %s dir in %s: %s",
                     section, path, exc)
        return None
    logger.debug("[active-experiment] recorded %s root_log_dir=%s", section, entry["root_log_dir"])
    return path


def record_ui_experiment(root_log_dir, ui_port: Optional[int] = None,
                         backend_port: Optional[int] = None) -> Optional[Path]:
    """Record the directory ``weightslab start`` just established.

    ``backend_port`` is what makes two concurrent UIs distinguishable: it is
    the gRPC port this UI proxies to, so it can later ask for the experiment
    directory of ITS backend rather than of whichever one started last.
    """
    return _write_section("ui", root_log_dir, ui_port=ui_port, backend_port=backend_port)


def record_backend_experiment(root_log_dir, grpc_port: Optional[int] = None) -> Optional[Path]:
    """Record the directory training actually resolved (``wl.serve()``)."""
    return _write_section("backend", root_log_dir, grpc_port=grpc_port)


def _entry_dir(section: str, entry: dict) -> Optional[str]:
    value = (entry or {}).get("root_log_dir")
    if not isinstance(value, str) or not value:
        return None
    if not os.path.isdir(value):
        # A run whose directory was deleted (or a marker copied between
        # machines) must not redirect anything.
        logger.debug("[active-experiment] %s dir %s no longer exists; ignoring", section, value)
        return None
    return value


def _live_entries(section: str) -> list:
    return [e for e in entries(section)
            if _pid_is_running(e.get("pid")) and _entry_dir(section, e)]


def _sole_live_dir(section: str, port_key: str = "", port: Optional[int] = None) -> Optional[str]:
    """The directory of the ONE live entry that matches, or None.

    With a port, an entry naming it wins outright -- that is how a UI finds ITS
    backend rather than whichever backend started last. Otherwise a single live
    entry is unambiguous and is used; two or more are not guessed between.
    """
    live = _live_entries(section)
    if port is not None and port_key:
        matching = [e for e in live if e.get(port_key) == port]
        if matching:
            # The same port twice can only be stale bookkeeping: newest wins.
            return _entry_dir(section, matching[-1])
    if len(live) == 1:
        return _entry_dir(section, live[0])
    if len(live) > 1:
        logger.info(
            "[active-experiment] %d live %s experiments recorded (%s); not "
            "guessing between them -- name the directory explicitly "
            "(WEIGHTSLAB_ROOT_LOG_DIR, or root_log_dir in the config).",
            len(live), section,
            ", ".join(str(e.get("root_log_dir")) for e in live))
    return None


def ui_experiment_dir() -> Optional[str]:
    """Directory of the most recent recorded ``weightslab start``, live or not.

    Raw record: it outlives the process that wrote it. Callers that REDIRECT a
    run on this should use :func:`live_ui_experiment_dir` instead.
    """
    for entry in reversed(entries("ui")):
        found = _entry_dir("ui", entry)
        if found:
            return found
    return None


def live_ui_experiment_dir() -> Optional[str]:
    """Directory of a ``weightslab start`` that is *still running*.

    The handoff exists for "the UI is up over there, put this run in its
    experiment". A record left behind by a UI that has since exited must not
    silently redirect an unrelated run months later -- which is exactly what
    happened to this repo's own gRPC tests: they resolved into a previous
    session's experiment directory and loaded ITS config. And with two UIs up
    (two experiments side by side) there is no right answer to guess.
    """
    return _sole_live_dir("ui")


def backend_experiment_dir() -> Optional[str]:
    """Directory of the most recent recorded ``wl.serve()``, live or not."""
    for entry in reversed(entries("backend")):
        found = _entry_dir("backend", entry)
        if found:
            return found
    return None


def live_backend_experiment_dir(grpc_port: Optional[int] = None) -> Optional[str]:
    """Directory of a backend that is *still running*.

    Pass ``grpc_port`` -- the port the caller actually talks to -- and the
    backend serving it is picked out by name. Without it, one live backend is
    unambiguous and two are not guessed between: a UI showing the OTHER
    experiment's reports is worse than a UI showing its own directory.

    Dead entries are ignored: the record outlives the process that wrote it.
    """
    return _sole_live_dir("backend", "grpc_port", grpc_port)


def clear() -> None:
    """Remove the marker (best-effort). Used by tests and ``weightslab`` teardown."""
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("[active-experiment] could not clear marker: %s", exc)
