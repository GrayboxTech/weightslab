import builtins
import logging
import sys
import tempfile
import atexit
import os
import shutil
import multiprocessing
from datetime import datetime


# Define the log format to include timestamp, level, module name, and function name
FORMAT = '%(asctime)s.%(msecs)03d %(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(funcName)s: %(message)s'
DATE_FORMAT = '%d/%m/%Y-%H:%M:%S'

# Sub-directory, under the experiment ``root_log_dir``, holding session logs.
# Both the initial ``setup_logging`` and the later relocation onto the resolved
# experiment directory go through ``experiment_log_dir``, so a session log never
# moves between two different layouts mid-run (which is what made log files look
# stale: the file was created in ``<root>/weightslab_logs/`` and then moved to
# ``<root>/``, so whoever kept watching the first path saw it stop growing).
LOG_SUBDIR = 'weightslab_logs'

# Logger the tqdm progress mirror writes to (weightslab/utils/tqdm_logging.py).
# It is filtered out of the terminal handler: the live bar is already on screen
# there, and a second, frozen copy of it scrolling past would fight the bar for
# the same lines. The file has no bar, so that is exactly where it belongs.
PROGRESS_LOGGER_NAME = 'weightslab.progress'

# Global variables to track the log file path and handlers
_LOG_FILE_PATH = None
_TMP_DIR_PATH = None
_FILE_HANDLER = None
_CONSOLE_HANDLER = None
# Levels resolved by the last ``setup_logging`` call. The console level is what
# the user asked for (``WEIGHTSLAB_LOG_LEVEL``); the file level defaults to
# NOTSET so the on-disk log keeps *everything*, whatever the console shows.
_CONSOLE_LEVEL = logging.INFO
_FILE_LEVEL = logging.NOTSET
# The exit hook is per-process, not per-setup_logging call: registering it again
# would print the location once per call at shutdown.
_EXIT_HOOK_REGISTERED = False


def _env_flag(name: str, default: str = 'false') -> bool:
    return os.getenv(name, default).lower() in {'1', 'true', 'yes', 'on'}


class _DropProgressMirror(logging.Filter):
    """Keep the tqdm mirror out of the terminal *handler*.

    Not because it doesn't belong on the terminal — it does, and
    ``tqdm_logging`` puts it there — but because a plain handler write would
    land in the middle of the live bar's own line and corrupt it. The echo goes
    through ``tqdm.write`` instead, which lifts the bar, prints, and redraws.
    Turn the echo off with ``WEIGHTSLAB_TQDM_LOG_TO_TERMINAL=0``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != PROGRESS_LOGGER_NAME


def _resolve_level(value, default=logging.INFO) -> int:
    """Map ``'DEBUG'`` / ``'WATCHDOG'`` / ``10`` / ``None`` onto a numeric level.

    Accepts anything a user may reasonably put in a config or an env var. An
    unknown name falls back to ``default`` rather than raising, so a typo in
    ``WEIGHTSLAB_LOG_LEVEL`` degrades the logging instead of killing the import.
    """
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return default
    if text.isdigit():
        return int(text)
    name = text.upper()
    # Standard levels resolve as module attributes; custom ones registered with
    # ``logging.addLevelName`` (weightslab's own WATCHDOG=35) need the name map,
    # which only exists on 3.11+ — hence the getattr dance for 3.10.
    resolved = getattr(logging, name, None)
    if not isinstance(resolved, int):
        name_map = getattr(logging, 'getLevelNamesMapping', None)
        if name_map is not None:
            resolved = name_map().get(name)
        else:  # 3.10: the reverse lookup is all there is. Unknown -> "Level X" (a str).
            resolved = logging.getLevelName(name)
    return resolved if isinstance(resolved, int) else default


def _level_name(level: int) -> str:
    return 'ALL' if level == logging.NOTSET else logging.getLevelName(level)


def experiment_log_dir(root_log_dir) -> str:
    """Session-log directory for an experiment ``root_log_dir``."""
    return os.path.join(str(root_log_dir), LOG_SUBDIR)


def get_log_file_path():
    """Absolute path of the current session log file (``None`` if file logging is off)."""
    return _LOG_FILE_PATH


def is_main_process() -> bool:
    """True only in the top-level process.

    Returns False inside spawned/forked workers (PyTorch ``DataLoader`` workers,
    ``multiprocessing`` children, DDP ranks). On Windows the default ``spawn``
    start method re-imports this package in every worker, so without this guard
    each worker re-runs the banner / creates its own temp log file during
    training. Detection is reliable at import time: ``multiprocessing`` sets the
    child's process *name* during the spawn "prepare" phase, before the main
    module (and therefore this package) is re-imported — even though
    ``parent_process()`` is not populated until slightly later in the child.
    """
    try:
        proc = multiprocessing.current_process()
        if proc is not None and getattr(proc, 'name', 'MainProcess') != 'MainProcess':
            return False
    except Exception:
        pass
    try:
        # Populated in forked children and once a spawned child is bootstrapped.
        if multiprocessing.parent_process() is not None:
            return False
    except Exception:
        pass
    return True


def _configure_dependency_log_levels() -> None:
    """Silence noisy dependency loggers that flood DEBUG output.

    Pinned on the *logger*, not on a handler, so they stay quiet in the file
    too: the root logger is otherwise wide open (see ``setup_logging``) and
    third-party DEBUG chatter would swamp the session log. Each group has its
    own opt-in env var.
    """
    suppressed_logger_groups = {
        'WEIGHTSLAB_ENABLE_ONNX_DEBUG_LOGS': (
            'onnx',
            'onnxscript',
            'onnxscript._internal',
            'onnxscript._internal.values',
            'torch.onnx',
            'torch.onnx._internal',
        ),
        'WEIGHTSLAB_ENABLE_MATPLOTLIB_DEBUG_LOGS': (
            'matplotlib',
            'matplotlib.pyplot',
            'matplotlib.font_manager',
        ),
        'WEIGHTSLAB_ENABLE_GRPC_DEBUG_LOGS': (
            'grpc',
            'grpc._channel',
            'grpc._server',
        ),
        'WEIGHTSLAB_ENABLE_PIL_DEBUG_LOGS': (
            'PIL',
            'PIL.Image',
            'PIL.PngImagePlugin',
        ),
    }

    for env_var, logger_names in suppressed_logger_groups.items():
        if _env_flag(env_var):
            continue
        for logger_name in logger_names:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def flush_logs() -> None:
    """Push anything still buffered in the file handler out to disk.

    ``logging`` flushes after every record, so this only matters when the
    process is about to be torn down in a way that skips ``logging.shutdown``
    (``os._exit``, a watchdog kill, a hard crash): the ``fsync`` makes the tail
    of the session survive it.
    """
    handler = _FILE_HANDLER
    if handler is None:
        return
    try:
        handler.flush()
        stream = getattr(handler, 'stream', None)
        if stream is not None and not getattr(stream, 'closed', False):
            os.fsync(stream.fileno())
    except Exception:
        pass


def _print_log_location():
    """Flush the log and print its location when Python exits.

    Deliberately ``builtins.print``: this module shadows ``print`` with a
    logging shim (below), and routing the exit notice through logging meant it
    was emitted while the interpreter was already tearing the streams down —
    which surfaced as a "``I/O operation on closed file``" logging error at the
    end of a run instead of the path the user was looking for.
    """
    flush_logs()
    if _LOG_FILE_PATH and os.path.exists(_LOG_FILE_PATH):
        try:
            builtins.print(
                f"\n{'='*60}\nWeightsLab session log saved to:\n{_LOG_FILE_PATH}\n{'='*60}",
                flush=True)
        except Exception:
            pass


class _SessionFileHandler(logging.FileHandler):
    """The session log's handler: logging must never be what fails a program.

    ``FileHandler.emit`` reopens a closed stream *outside* its own try/except,
    so once the log's directory was gone -- a temporary ``root_log_dir`` the
    log had been moved into, then deleted -- every later log call anywhere in
    the process raised ``FileNotFoundError``, failing code that has nothing to
    do with logging (tornado creating an event loop, in the release tests).
    The directory is recreated when it can be; any other failure goes through
    ``handleError``, as a failed write already does.
    """

    def _open(self):
        try:
            return super()._open()
        except FileNotFoundError:
            os.makedirs(os.path.dirname(self.baseFilename) or ".", exist_ok=True)
            return super()._open()

    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


def _make_file_handler(path: str) -> logging.FileHandler:
    """Open the session log file.

    Deliberately append mode, even for a brand-new file. A ``mode='w'`` handler
    that someone else closes is dead for good: ``FileHandler.emit`` refuses to
    reopen it (so a reopen can never truncate the file — CPython bpo-42378), and
    since nothing detaches it from the root logger it goes on accepting records
    and dropping them silently. That is not hypothetical here — see
    ``ensure_logging_intact``. Append mode makes ``emit`` reopen instead, so the
    log survives. The filename carries a per-process timestamp, so appending is
    equivalent to truncating for a fresh file, and strictly better if two
    processes ever land on the same name.
    """
    handler = _SessionFileHandler(path, mode='a', encoding='utf-8')
    handler.setLevel(_FILE_LEVEL)
    handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATE_FORMAT))
    return handler


def ensure_logging_intact() -> bool:
    """Repair the root logger after something else reconfigured logging.

    ``logging.config.dictConfig`` closes **every** handler in the process on its
    way in (``_clearExistingHandlers`` → ``logging.shutdown``), and it does not
    detach them from the root logger. weightslab's handlers therefore stay
    attached and keep being called, but the file handler's stream is gone, so
    the session log stops mid-run while the terminal carries on — with no error
    anywhere. traitlets runs ``dictConfig`` while building any ``Application``,
    which means every time the studio's embedded notebook kernel starts
    (``IPKernelApp.initialize``); uvicorn, celery and friends do the same thing.

    Re-attaches the handlers, reopens the log file and restores the root level.
    Returns True if anything actually needed repairing.
    """
    global _FILE_HANDLER

    if _CONSOLE_HANDLER is None and _FILE_HANDLER is None:
        # setup_logging never ran in this process: there is nothing of ours to
        # protect, and the root logger belongs to whoever did configure it.
        return False

    root = logging.getLogger()
    repaired = False

    if _CONSOLE_HANDLER is not None and _CONSOLE_HANDLER not in root.handlers:
        root.addHandler(_CONSOLE_HANDLER)
        repaired = True

    if _FILE_HANDLER is not None and _LOG_FILE_PATH:
        if getattr(_FILE_HANDLER, 'stream', None) is None:
            # Closed underneath us. Build a fresh handler rather than revive the
            # old one: `logging.shutdown` also dropped it from logging's own
            # handler list, so a replacement is what re-registers it for the
            # flush at interpreter exit.
            root.removeHandler(_FILE_HANDLER)
            _FILE_HANDLER = _make_file_handler(_LOG_FILE_PATH)
            root.addHandler(_FILE_HANDLER)
            repaired = True
        elif _FILE_HANDLER not in root.handlers:
            root.addHandler(_FILE_HANDLER)
            repaired = True

    expected_level = (min(_CONSOLE_LEVEL, _FILE_LEVEL) if _FILE_HANDLER is not None
                      else _CONSOLE_LEVEL)
    if root.level != expected_level:
        root.setLevel(expected_level)
        repaired = True

    if repaired:
        logging.getLogger(__name__).debug(
            "Restored weightslab logging after an external logging reconfiguration "
            "(log file: %s)", _LOG_FILE_PATH)
    return repaired


def setup_logging(level, log_to_file=True, file_level=None):
    """
    Configures the logging system.

    The terminal and the log file are filtered **independently**:

    * the terminal shows ``level`` and above (``WEIGHTSLAB_LOG_LEVEL``);
    * the file records *everything* — the root logger is opened all the way up
      so records below ``level`` are not dropped before they reach a handler,
      and the file handler itself sits at ``NOTSET``.

    Cap the file too with ``file_level`` / ``WEIGHTSLAB_LOG_FILE_LEVEL`` when
    the full-fidelity log is more than you want on disk.

    Args:
        level (str|int): Minimum level printed to the terminal (e.g. 'DEBUG', 'INFO').
        log_to_file (bool): If True, logs are written to a file (default: True).
        file_level (str|int|None): Minimum level written to the file. Defaults to
            ``WEIGHTSLAB_LOG_FILE_LEVEL``, else NOTSET (no restriction).
    """
    global _TMP_DIR_PATH, _LOG_FILE_PATH, _FILE_HANDLER, _CONSOLE_HANDLER
    global _CONSOLE_LEVEL, _FILE_LEVEL, _EXIT_HOOK_REGISTERED

    # Reset logger handlers to ensure previous configurations don't interfere
    logging.getLogger().handlers = []

    # Best-effort: reconfigure stdio to UTF-8 (Windows-safe)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    _CONSOLE_LEVEL = _resolve_level(level, default=logging.INFO)
    if file_level is None:
        file_level = os.getenv('WEIGHTSLAB_LOG_FILE_LEVEL')
    # NOTSET (0) on both the root logger and the handler means "no restriction".
    _FILE_LEVEL = _resolve_level(file_level, default=logging.NOTSET)

    # Create formatters
    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(_CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_DropProgressMirror())

    # Get root logger. Its level gates records BEFORE any handler sees them, so
    # it has to sit at the most permissive of the two sinks — otherwise the file
    # handler's own level is moot and e.g. WEIGHTSLAB_LOG_LEVEL=INFO silently
    # drops every DEBUG record from the file as well as from the terminal.
    root_logger = logging.getLogger()
    root_logger.setLevel(min(_CONSOLE_LEVEL, _FILE_LEVEL) if log_to_file else _CONSOLE_LEVEL)
    _CONSOLE_HANDLER = console_handler
    root_logger.addHandler(console_handler)
    _configure_dependency_log_levels()

    # File handler. The old one (if any) was just detached from the root logger,
    # so drop the references too rather than leave set_log_directory / flush_logs
    # holding a handler nothing writes through any more.
    if _FILE_HANDLER is not None:
        try:
            _FILE_HANDLER.close()
        except Exception:
            pass
    _FILE_HANDLER = None
    _LOG_FILE_PATH = None
    if log_to_file:
        # `weightslab start [DIR]` (or the user) exports WEIGHTSLAB_ROOT_LOG_DIR;
        # if neither is set we start in a temp dir and relocate onto the
        # experiment directory as soon as it resolves (see set_log_directory).
        root_dir = os.environ.get('WEIGHTSLAB_ROOT_LOG_DIR') or tempfile.mkdtemp()
        log_dir = experiment_log_dir(root_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        _TMP_DIR_PATH = log_dir
        _LOG_FILE_PATH = os.path.join(log_dir, f'weightslab_{timestamp}.log')

        _FILE_HANDLER = _make_file_handler(_LOG_FILE_PATH)
        root_logger.addHandler(_FILE_HANDLER)

        # Register exit handler to flush the log and print its location
        if not _EXIT_HOOK_REGISTERED:
            atexit.register(_print_log_location)
            _EXIT_HOOK_REGISTERED = True

        # Log the initialization
        logging.info(
            f"WeightsLab logging initialized - Log file: {_LOG_FILE_PATH} "
            f"(terminal: {_level_name(_CONSOLE_LEVEL)}, file: {_level_name(_FILE_LEVEL)})")


def set_log_directory(new_log_dir):
    """
    Updates the log file location to a new directory.
    Moves the existing log file from its current location to the new one.

    Called automatically once the experiment ``root_log_dir`` resolves, so the
    session log lands with the checkpoints instead of staying behind in a temp
    directory. Can also be called manually to relocate logs.

    Args:
        new_log_dir (str): The new directory where logs should be saved.

    Example:
        >>> import weightslab as wl
        >>> # Logging starts in a temporary directory automatically
        >>> # Later, when you define your experiment directory:
        >>> wl.set_log_directory("./my_experiment/logs")
        >>> # Log file is moved from temp to ./my_experiment/logs/

    Note:
        - The log file keeps its original timestamped filename
        - All subsequent logs are written to the new location
        - The old log file is moved (not copied)
        - Calling it again for the directory already in use is a no-op
    """
    global _TMP_DIR_PATH, _LOG_FILE_PATH, _FILE_HANDLER

    if not _LOG_FILE_PATH or not _FILE_HANDLER:
        logging.warning("No log file to relocate. Call setup_logging() first.")
        return

    new_log_dir = os.path.abspath(str(new_log_dir))
    old_log_path = _LOG_FILE_PATH

    # Already there. Relocating again would churn the handler and re-log the
    # "log directory updated" lines every time hyperparameters are re-registered.
    current_dir = os.path.dirname(os.path.abspath(old_log_path))
    if os.path.normcase(current_dir) == os.path.normcase(new_log_dir):
        return

    # Create new log directory
    os.makedirs(new_log_dir, exist_ok=True)

    # Keep the original timestamped filename
    new_log_path = os.path.join(new_log_dir, os.path.basename(old_log_path))

    # Get root logger
    root_logger = logging.getLogger()

    # Flush and close the current file handler before moving the file
    flush_logs()
    _FILE_HANDLER.close()
    root_logger.removeHandler(_FILE_HANDLER)

    moved, move_error = False, None
    try:
        if os.path.exists(old_log_path):
            shutil.move(old_log_path, new_log_path)
            moved = True
    except Exception as error:
        move_error = error

    # Update global path
    _LOG_FILE_PATH = new_log_path
    _TMP_DIR_PATH = new_log_dir

    # Re-open at the new location, preserving the configured file level.
    # Appending rather than truncating means a failed move costs the history
    # already written but never the rest of the session.
    _FILE_HANDLER = _make_file_handler(_LOG_FILE_PATH)
    root_logger.addHandler(_FILE_HANDLER)

    if moved:
        logging.info(f"Log file moved from {old_log_path} to {new_log_path}")
    elif move_error is not None:
        logging.warning(
            f"Could not move log file: {move_error}. "
            f"Continuing in a new log file at {new_log_path}")
    logging.info(f"Log directory updated to: {new_log_dir}")


def print(first_element, *other_elements, sep=' ', **kwargs):
    """
    Overrides the built-in print function to use logging features.

    The output level (DEBUG, INFO, WARNING, etc.) can be controlled
    using the 'level' keyword argument. Defaults to 'INFO'.

    Args:
        first_element: The mandatory first element to log.
        *other_elements: All subsequent positional arguments.
        sep (str): The separator to use between elements (default: ' ').
        **kwargs: Optional keyword arguments, including 'level' to set the
        severity.
    """
    # 0. Setup logging
    level_str = kwargs.pop('level', 'INFO').upper()

    # 1. Combine all positional elements into a single log message string.
    all_elements = (first_element,) + other_elements
    log_message = sep.join(map(str, all_elements))

    # 2. Map the string level to the corresponding logging method.
    if level_str == "DEBUG":
        logging.debug(log_message)
    elif level_str == "INFO":
        logging.info(log_message)
    elif level_str == "WARNING":
        logging.warning(log_message)
    elif level_str == "ERROR":
        logging.error(log_message)
    elif level_str == "CRITICAL":
        logging.critical(log_message)
    else:
        # Default fallback if an unknown level is provided
        logging.info(log_message)


if __name__ == "__main__":
    # Test 1: Setup logging — terminal at INFO, file wide open
    setup_logging('INFO')
    print('This is a default INFO message')

    # Test 2: DEBUG message — file only, never printed to the terminal
    print('This message is DEBUG-only', 'and uses sep', sep='|', level='debug')

    # Test 3: Log message at WARNING level
    print('Warning: Something unusual happened.', level='WARNING')

    # Test 4: Relocate log directory
    new_log_dir = os.path.join(tempfile.gettempdir(), 'weightslab_test_logs')
    print(f'Relocating logs to: {new_log_dir}')
    set_log_directory(new_log_dir)

    # Test 5: Log after relocation
    print('This is a message after log relocation.', 'All good.')
    print(f'New log file location: {_LOG_FILE_PATH}')
