"""weightslab package

Expose commonly used helpers at package level so users can do::

    import weightslab as wl
    wl.watch_or_edit(...)

This file re-exports selected symbols from `weightslab.src`.
"""
import os
import logging
import threading

from .src import watch_or_edit, start_training, serve, keep_serving, save_signals, save_instance_signals, save_group_signals, tag_samples, register_categorical_tag, set_categorical_tag, discard_samples, get_samples_by_tag, get_discarded_samples, signal, eval_fn, compute_signals, SignalContext, clear_all, run_pending_evaluation, trigger_pending_evaluation_async, query_signal_history, query_sample_history, query_instance_history, write_history, write_dataframe, get_current_experiment_hash, pointcloud_thumbnail, pointcloud_boxes
from .backend.ledgers import GLOBAL_LEDGER as ledger
from .art import _BANNER
from .utils.logs import setup_logging, set_log_directory, is_main_process
from .utils.tools import seed_everything
from .components.global_monitoring import guard_training_context, guard_testing_context

# If you already have other top-level exports, keep them.
# This snippet ensures __version__ is available even when setuptools_scm hasn't written the file yet.


# Change the name of the current (main) thread
threading.current_thread().name = "WL-MainThread"

# Auto-initialize logging on import. Python's module cache (sys.modules) makes
# this block run once per process — but spawned/forked workers (e.g. PyTorch
# DataLoader workers, which on Windows' 'spawn' start method re-import this
# package) are separate processes and would each re-emit the banner and create
# their own temp log file during training. The noisy parts below are therefore
# gated to the main process; workers keep a quiet console-only logger.
log_level = os.getenv('WEIGHTSLAB_LOG_LEVEL', 'INFO')
log_to_file = os.getenv('WEIGHTSLAB_LOG_TO_FILE', 'true').lower() == 'true'

_IS_MAIN_PROCESS = is_main_process()

setup_logging(log_level, log_to_file=(log_to_file and _IS_MAIN_PROCESS))

logger = logging.getLogger(__name__)
if _IS_MAIN_PROCESS:
    logger.info(f"WeightsLab package initialized - Log level: {log_level}, Log to file: {log_to_file}")
    if os.getenv('WEIGHTSLAB_SUPPRESS_BANNER', '0') != '1':
        logger.info(_BANNER)

grpc_tls_enabled = os.environ.get('GRPC_TLS_ENABLED', 'true').lower() == 'true'
if _IS_MAIN_PROCESS and grpc_tls_enabled and os.environ.get('WEIGHTSLAB_SKIP_SECURE_INIT', 'false').lower() != 'true':
	try:
		from weightslab.security import CertAuthManager

		enable_auth = os.environ.get('WL_ENABLE_GRPC_AUTH_TOKEN', 'true').lower() == 'true'
		manager = CertAuthManager.from_env_or_default(enable_auth=enable_auth)

		success, msg = manager.check_and_apply()
		if success:
			logger.debug(f"Secure environment applied: {msg}")
		else:
			logger.debug(f"Running in unsecured mode — no certs found. To set up: weightslab ui se")
	except Exception as e:
		logger.debug(f"Secure environment check skipped: {e}")

# Get Package Metadata. Resolve the version from the most authoritative source
# available, so a live/editable checkout reports the CURRENT git tag rather than
# a stale generated file. weightslab/_version.py is written by setuptools_scm at
# BUILD time only — in a working tree it goes stale as new tags land, which is
# why the banner used to show an old version. Preference order:
#   1. live git checkout  -> derive from the current (or nearest older) git tag
#   2. built/installed pkg -> the generated _version.py, else dist metadata
#   3. last resort         -> a UTC timestamp (keeps import from ever failing)
def _resolve_version() -> str:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(pkg_dir)

    # 1. Live checkout: the generated file may lag behind, so ask git first.
    # Gated to the main process — spawned DataLoader workers re-import this
    # package and must not each shell out to git; they fall through to the
    # generated file below (they never display/report the version anyway).
    if _IS_MAIN_PROCESS and os.path.isdir(os.path.join(repo_root, ".git")):
        # Drop setuptools_scm's local segment ("+g<sha>.d<date>") to honor
        # local_scheme="no-local-version" from pyproject at runtime too.
        def _clean(v: str) -> str:
            return v.split("+", 1)[0]
        # setuptools_scm (a build dep; may be absent at runtime) yields a full
        # PEP 440 version incl. commits-since-tag, e.g. 1.3.1.dev3.
        try:
            from setuptools_scm import get_version # type: ignore
            return _clean(get_version(root=repo_root))
        except Exception:
            pass
        # Plain git as a lighter fallback: the nearest reachable tag (older tags
        # included), normalized from e.g. "v1.3.1-dev0" -> "1.3.1-dev0".
        try:
            import subprocess
            out = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=repo_root, capture_output=True, text=True,
            )
            tag = out.stdout.strip()
            if tag:
                return tag[1:] if tag.startswith("v") else tag
        except Exception:
            pass

    # 2. Built/installed package: the generated file (or dist metadata) is authoritative.
    try:
        from ._version import __version__ as _v # type: ignore
        if _v:
            return _v
    except Exception:
        pass
    try:
        from importlib.metadata import version as _dist_version
        return _dist_version("weightslab")
    except Exception:
        pass

    # 3. Never let version resolution break the import.
    from datetime import datetime
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


__version__ = _resolve_version()

if _IS_MAIN_PROCESS:
    try:
        from weightslab.utils.telemetry import ping_import
        ping_import(__version__)
    except Exception as e:
        logger.debug("Telemetry ping failed: %s", e)

__author__ = 'Alexandru-Andrei ROTARY'
__maintainer__ = 'Guillaume PELLUET'
__credits__ = 'GrayBx'
__license__ = 'BSD 2-clause'
__all__ = [
    "watch_or_edit",
    "serve",
    "keep_serving",
    "save_signals",
    "save_instance_signals",
    "save_group_signals",
    "signal",
    "compute_signals",
    "set_log_directory",
	"tag_samples",
  	"discard_samples",
  	"get_samples_by_tag",
    "get_discarded_samples",
    "SignalContext",
    "clear_all",
  	"seed_everything",
    "run_pending_evaluation",
	"eval_fn",
	"trigger_pending_evaluation_async",
  	"guard_training_context",
    "guard_testing_context",
    "ledger",
	"start_training",
	"register_categorical_tag",
	"set_categorical_tag",
    "get_current_experiment_hash",
    "query_signal_history",
    "query_sample_history",
    "query_instance_history",

    "write_history",
    "write_dataframe",

    "pointcloud_thumbnail",
    "pointcloud_boxes",

    "_BANNER",
    "__version__",
    "__license__",
    "__author__",
    "__maintainer__",
    "__credits__"
]
