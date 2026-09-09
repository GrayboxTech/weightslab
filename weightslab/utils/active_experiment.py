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
write to and read from.

    {
      "ui":      {"root_log_dir": "...", "pid": 123, "updated_at": "..."},
      "backend": {"root_log_dir": "...", "pid": 456, "updated_at": "..."}
    }

* ``ui`` is written by ``weightslab start`` -- the directory it established.
  A later training process with nothing configured adopts it, which is what
  makes the two halves land in the same experiment.
* ``backend`` is written by ``wl.serve()`` -- the directory training ACTUALLY
  resolved, whatever the route (an explicit ``root_log_dir:`` in a config file,
  the environment, or the ``ui`` value above). The UI prefers it when listing
  reports and notebooks, so those lists stay right even when training was
  pointed somewhere the UI never chose.

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


def _write_section(section: str, root_log_dir, **meta) -> Optional[Path]:
    """Merge one section into the marker, leaving the other side's entry alone."""
    if section not in _SECTIONS:
        raise ValueError(f"unknown section {section!r}")
    if not root_log_dir:
        return None

    entry = {
        "root_log_dir": str(Path(root_log_dir).expanduser().resolve()),
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entry.update({k: v for k, v in meta.items() if v is not None})

    state = read_state()
    state[section] = entry
    path = state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
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


def record_ui_experiment(root_log_dir, ui_port: Optional[int] = None) -> Optional[Path]:
    """Record the directory ``weightslab start`` just established."""
    return _write_section("ui", root_log_dir, ui_port=ui_port)


def record_backend_experiment(root_log_dir) -> Optional[Path]:
    """Record the directory training actually resolved (``wl.serve()``)."""
    return _write_section("backend", root_log_dir)


def _section_dir(section: str) -> Optional[str]:
    entry = read_state().get(section)
    if not isinstance(entry, dict):
        return None
    value = entry.get("root_log_dir")
    if not isinstance(value, str) or not value:
        return None
    if not os.path.isdir(value):
        # A run whose directory was deleted (or a marker copied between
        # machines) must not redirect anything.
        logger.debug("[active-experiment] %s dir %s no longer exists; ignoring", section, value)
        return None
    return value


def ui_experiment_dir() -> Optional[str]:
    """Directory established by the most recent ``weightslab start``, if it still exists."""
    return _section_dir("ui")


def backend_experiment_dir() -> Optional[str]:
    """Directory the most recent ``wl.serve()`` resolved, if it still exists."""
    return _section_dir("backend")


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


def live_backend_experiment_dir() -> Optional[str]:
    """Directory of a backend that is *still running*.

    Callers that redirect on this (the UI's report/notebook listings) must not
    be sent to a directory recorded by some earlier, finished run: the record
    outlives the process that wrote it. A dead entry is treated as absent, so
    the caller keeps its own directory.
    """
    entry = read_state().get("backend")
    if not isinstance(entry, dict) or not _pid_is_running(entry.get("pid")):
        return None
    return _section_dir("backend")


def clear() -> None:
    """Remove the marker (best-effort). Used by tests and ``weightslab`` teardown."""
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("[active-experiment] could not clear marker: %s", exc)
