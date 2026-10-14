"""Mirror live tqdm progress bars into the session log.

A tqdm bar paints itself onto a terminal stream with carriage returns and never
goes anywhere near ``logging``. So the one thing you actually want when reading
a training log after the fact — how far the run got, and what the loss was
doing — is the one thing the log file never had. Worse, a bar built *after* the
studio's embedded notebook kernel swapped ``sys.stdout``/``sys.stderr`` for
ipykernel's ``OutStream`` renders into the notebook's iopub channel, so it
vanishes from the terminal too and the run looks stalled.

This samples every live bar on a timer and writes one compact line per bar per
interval — no carriage returns, no block-drawing characters — so the log gets a
readable trace of progress instead of the hundreds of repaints a second a real
bar produces. Unchanged bars are skipped, so a paused run does not fill the log
with identical lines.

The lines go to ``logs.PROGRESS_LOGGER_NAME``, which the terminal handler
filters out: the live bar is already on screen there. Set
``WEIGHTSLAB_TQDM_LOG_TO_TERMINAL=1`` to see them in both places.

Configuration:
    ``WEIGHTSLAB_TQDM_LOG_INTERVAL``     seconds between samples (default 30; 0 disables)
    ``WEIGHTSLAB_TQDM_LOG_TO_TERMINAL``  also echo to the terminal (default on)
"""

import logging
import os
import threading

from weightslab.utils.logs import PROGRESS_LOGGER_NAME

logger = logging.getLogger(__name__)
progress_logger = logging.getLogger(PROGRESS_LOGGER_NAME)

DEFAULT_INTERVAL_SECONDS = 30.0

_state_lock = threading.Lock()
_thread = None
_stop_event = None


def _terminal_echo_enabled() -> bool:
    """Whether to also print each sampled line to the terminal (default: yes).

    The live bar is usually already on screen, so this is partly redundant —
    but only partly, and only sometimes: the bar is a single repainting line
    that shows no history, and it disappears entirely whenever stdout is not a
    terminal (piped, ``nohup``, CI) or has been taken over, as the studio's
    embedded notebook kernel does. The echo is what makes progress visible in
    all of those. It goes through ``tqdm.write``, which lifts the bar, prints,
    and redraws it, so the bar is never corrupted.
    """
    return os.getenv('WEIGHTSLAB_TQDM_LOG_TO_TERMINAL', '1').strip().lower() not in {
        '0', 'false', 'no', 'off'}


def _interval_from_env() -> float:
    raw = os.getenv('WEIGHTSLAB_TQDM_LOG_INTERVAL')
    if raw is None or not str(raw).strip():
        return DEFAULT_INTERVAL_SECONDS
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Ignoring non-numeric WEIGHTSLAB_TQDM_LOG_INTERVAL=%r", raw)
        return DEFAULT_INTERVAL_SECONDS


def _live_bars():
    """Every tqdm bar currently alive, newest last. Empty if tqdm isn't in use."""
    try:
        from tqdm import tqdm as tqdm_cls
    except Exception:
        return []
    try:
        # _instances is a WeakSet; copy it so a bar closing mid-iteration
        # can't raise "Set changed size during iteration" on the timer thread.
        return [bar for bar in list(getattr(tqdm_cls, '_instances', ())) if bar is not None]
    except Exception:
        return []


def _render(bar) -> str | None:
    """One log-friendly line for *bar*.

    Built from ``format_dict`` rather than ``str(bar)``: the rendered bar is
    mostly block-drawing characters sized for a terminal, which carry nothing in
    a log file and wrap badly. This keeps the parts that mean something —
    position, elapsed, rate and whatever the training loop put in the postfix.
    """
    try:
        from tqdm import tqdm as tqdm_cls
        fields = bar.format_dict
    except Exception:
        return None

    try:
        description = str(fields.get('prefix') or '').strip().rstrip(':') or 'progress'
        count = fields.get('n')
        total = fields.get('total')
        parts = []
        if total:
            parts.append(f"{count}/{total} ({count / total:.0%})")
        else:
            parts.append(f"{count} steps")
        parts.append(tqdm_cls.format_interval(fields.get('elapsed') or 0))
        rate = fields.get('rate')
        if rate:
            parts.append(f"{rate:.2f} it/s")
        line = f"{description}: " + ", ".join(parts)
        postfix = fields.get('postfix')
        if postfix:
            line += f" | {postfix}"
        return line
    except Exception:
        return None


def _echo_to_terminal(line: str) -> None:
    """Print *line* above the live bars, via tqdm so they redraw cleanly."""
    if not _terminal_echo_enabled():
        return
    try:
        from tqdm import tqdm as tqdm_cls
        tqdm_cls.write(line)
    except Exception as error:
        logger.debug("Could not echo progress to the terminal: %s", error)


def _progress_signature(bar):
    """What "the bar moved" means, for skipping repeats.

    Deliberately NOT the rendered line: that carries elapsed time, which ticks
    on every sample even when the run is paused, so a rendered-line comparison
    never matches and a stopped run writes an almost-identical entry every
    interval forever. Only the fields that represent actual progress count.
    """
    try:
        fields = bar.format_dict
        return (fields.get('prefix'), fields.get('n'),
                fields.get('total'), fields.get('postfix'))
    except Exception:
        return None


def _sample_once(previous: dict) -> None:
    """Log every bar that actually moved since the last sample."""
    alive = _live_bars()
    for bar in alive:
        signature = _progress_signature(bar)
        if signature is None:
            continue
        key = id(bar)
        if previous.get(key) == signature:
            continue  # paused or finished; don't repeat ourselves
        line = _render(bar)
        if not line:
            continue
        previous[key] = signature
        progress_logger.info("%s", line)
        _echo_to_terminal(line)

    live_keys = {id(bar) for bar in alive}
    for key in [k for k in previous if k not in live_keys]:
        del previous[key]


def _loop(interval: float, stop_event: threading.Event) -> None:
    previous: dict = {}
    while not stop_event.wait(interval):
        try:
            _sample_once(previous)
        except Exception as error:  # never let the mirror take the run down
            logger.debug("tqdm progress mirror sample failed: %s", error)


def start_tqdm_log_mirror(interval: float | None = None) -> bool:
    """Start mirroring live tqdm bars into the session log.

    Idempotent: a second call while the mirror is running is a no-op. Returns
    True if the mirror is running when this returns.

    Args:
        interval: Seconds between samples. Defaults to
            ``WEIGHTSLAB_TQDM_LOG_INTERVAL``, else 30. Zero or negative disables.
    """
    global _thread, _stop_event

    if interval is None:
        interval = _interval_from_env()
    if interval <= 0:
        logger.debug("tqdm progress mirror disabled (interval=%s)", interval)
        return False

    with _state_lock:
        if _thread is not None and _thread.is_alive():
            return True
        _stop_event = threading.Event()
        _thread = threading.Thread(
            target=_loop, args=(interval, _stop_event),
            name="wl-tqdm-log-mirror", daemon=True)
        _thread.start()
    logger.debug("tqdm progress mirror started (every %.1fs -> %s)",
                 interval, PROGRESS_LOGGER_NAME)
    return True


def stop_tqdm_log_mirror() -> None:
    """Stop the mirror. Safe to call when it was never started."""
    global _thread, _stop_event

    with _state_lock:
        if _stop_event is not None:
            _stop_event.set()
        thread, _thread, _stop_event = _thread, None, None
    if thread is not None:
        thread.join(timeout=2.0)
