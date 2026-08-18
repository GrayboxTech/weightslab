"""Begin/end operation tracing for dataframe, array-store, duckdb and
experiment-service operations.

Off by default (near-zero overhead: one bool check) — set ``WL_OPTRACE=1`` to
turn it on. Every traced call prints ONE line at start and ONE line at end to
stdout (unbuffered, same stream as main.py's ``[timing]`` prints), tagged
``[optrace]`` so a run's LOG file can be parsed the same way:

    grep -a "\\[optrace\\]" LOG | ...

Line format (space-separated key=value tokens, so ``awk`` can pick fields by
name without caring about column position)::

    [optrace] BEGIN domain=dataframe op=dfm.upsert_df call=482 tid=140234 ts=1723300000.123456 site=dataframe_manager.py:680 n_in=24 bytes_in=196608 args=origin=train_loader,force_flush=False
    [optrace] END   domain=dataframe op=dfm.upsert_df call=482 tid=140234 ts=1723300000.234567 dur_ms=111.111 ok=True site=dataframe_manager.py:680 n_in=24 bytes_in=196608 args=origin=train_loader,force_flush=False mem_delta_kb=512 n_out=24 bytes_out=-

``call=`` pairs a BEGIN with its END even when the same op runs concurrently
on multiple threads (same op+tid can otherwise appear twice before either
finishes). For call-count/timing/bytes/memory/object-count reports, the END
line alone carries every field -- see ``code/optrace_report.py``.

When ``@traced``/``trace_op`` wraps a whole function (the normal case), the
extra fields beyond ``dur_ms``/``ok`` are filled in automatically:

    site        file:line of the function's ``def`` (not the call site --
                stable across callers, and enough to jump to the code).
    n_in/n_out  best-effort element counts for arguments / return value
                (numpy array .size, len() of dict/list/etc).
    bytes_in/out  best-effort byte counts (numpy .nbytes, len() of bytes),
                summed recursively through dict/list/tuple containers.
    args        sanitized ``name=repr`` for each bound argument (arrays
                collapse to ``ndarray(shape=...,dtype=...)`` rather than
                dumping their contents) -- the "which sample_id did this"
                detail needed to trace back a specific weird call.
    mem_delta_kb  RSS delta (psutil) across the call. Peak-agnostic and can
                be noisy under concurrent threads sharing one process, but
                cheap and good enough to spot a call that's allocating much
                more than its neighbours.

A bare ``with trace_op(domain, op, **extra):`` (not decorating a function,
e.g. ``TracingDuckDBConn``) has no function to introspect, so it only gets
whatever ``extra`` the caller passed plus ``mem_delta_kb`` -- no
site/n_in/n_out/bytes_in/bytes_out/args.
"""
import functools
import inspect
import itertools
import os
import sys
import threading
import time

try:
    import numpy as _np
except Exception:
    _np = None

try:
    import pandas as _pd
except Exception:
    _pd = None

try:
    import psutil as _psutil
    _PROC = _psutil.Process()
except Exception:
    _PROC = None

_TRUTHY = {"1", "true", "yes", "on"}
_ENABLED = os.environ.get("WL_OPTRACE", "0").strip().lower() in _TRUTHY

_counter = itertools.count()
_counter_lock = threading.Lock()

# print(msg, flush=True) is two separate write()s under the hood (message,
# then the trailing newline) with no atomicity guarantee between them, so two
# threads tracing concurrently (training thread, flush thread, grpc workers)
# can interleave mid-line -- observed in practice as garbled/merged [optrace]
# lines. Serialize the full write+flush per line instead.
_print_lock = threading.Lock()


def _emit(line: str) -> None:
    with _print_lock:
        print(line, flush=True)


def trace_enabled() -> bool:
    return _ENABLED


def _next_call_id() -> int:
    with _counter_lock:
        return next(_counter)


def _fmt_extra(extra: dict) -> str:
    if not extra:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in extra.items())


def sanitize(value, maxlen: int = 48) -> str:
    """Collapse whitespace and truncate so a value is safe as a bare token
    in the space-separated log line (e.g. a SQL statement)."""
    s = " ".join(str(value).split())
    if len(s) > maxlen:
        s = s[:maxlen] + "..."
    return s.replace(" ", "_")


def _rss_kb():
    if _PROC is None:
        return None
    try:
        return _PROC.memory_info().rss / 1024.0
    except Exception:
        return None


def _obj_metrics(obj):
    """Best-effort (count, bytes) size hints for an object; either may be None."""
    if obj is None:
        return None, None
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.size, obj.nbytes
    # deep=False ONLY. deep=True walks every element of every object/string
    # column: measured at ~1000ms on a 3.96M-row frame vs ~1ms shallow (854x),
    # and this runs on every traced call -- it turns tracing itself into the
    # O(dataset) hot-path work this module exists to hunt down. Shallow
    # undercounts object columns (it counts the 8-byte pointers, not the
    # referenced strings), so bytes_in/out for string-heavy frames is a lower
    # bound; that is the right trade for a diagnostic that must not distort
    # what it measures.
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        try:
            return len(obj), int(obj.memory_usage(deep=False).sum())
        except Exception:
            return len(obj), None
    if _pd is not None and isinstance(obj, _pd.Series):
        try:
            return len(obj), int(obj.memory_usage(deep=False))
        except Exception:
            return len(obj), None
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return len(obj), len(obj)
    if isinstance(obj, dict):
        nbytes = 0
        for v in obj.values():
            _, vb = _obj_metrics(v)
            if vb:
                nbytes += vb
        return len(obj), (nbytes or None)
    if isinstance(obj, (list, tuple, set)):
        nbytes = 0
        for v in obj:
            _, vb = _obj_metrics(v)
            if vb:
                nbytes += vb
        return len(obj), (nbytes or None)
    if isinstance(obj, (str, int, float, bool)):
        return None, None
    if hasattr(obj, "__len__"):
        try:
            return len(obj), None
        except Exception:
            return None, None
    return None, None


def _fmt_arg_value(value, maxlen: int = 40) -> str:
    if _np is not None and isinstance(value, _np.ndarray):
        return f"ndarray(shape={value.shape},dtype={value.dtype})"
    if _pd is not None and isinstance(value, _pd.DataFrame):
        return f"DataFrame(rows={len(value)},cols={value.shape[1]})"
    if _pd is not None and isinstance(value, _pd.Series):
        return f"Series(len={len(value)},dtype={value.dtype})"
    s = repr(value)
    return s if len(s) <= maxlen else s[: maxlen - 3] + "..."


def _in_metrics(sig, args, kwargs) -> dict:
    """n_in/bytes_in/args extras for a decorated function's bound arguments."""
    if sig is None:
        return {}
    try:
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except Exception:
        return {}
    arg_items = [(n, v) for n, v in bound.arguments.items() if n != "self"]
    n_in = b_in = 0
    has_n, has_b = False, False
    for _, v in arg_items:
        n, b = _obj_metrics(v)
        if n is not None:
            n_in += n
            has_n = True
        if b is not None:
            b_in += b
            has_b = True
    out = {}
    if has_n:
        out["n_in"] = n_in
    if has_b:
        out["bytes_in"] = b_in
    if arg_items:
        args_str = ",".join(f"{n}={_fmt_arg_value(v)}" for n, v in arg_items)
        out["args"] = sanitize(args_str, maxlen=160)
    return out


def _out_metrics(result) -> dict:
    n_out, b_out = _obj_metrics(result)
    out = {}
    if n_out is not None:
        out["n_out"] = n_out
    if b_out is not None:
        out["bytes_out"] = b_out
    return out


class trace_op:
    """Context manager: logs BEGIN on enter, END (with duration) on exit.

    Also usable as a decorator: ``@trace_op("dfm.upsert_df")``.
    """

    __slots__ = ("domain", "op", "extra", "_call_id", "_t0", "_mem0")

    def __init__(self, domain: str, op: str, **extra):
        self.domain = domain
        self.op = op
        self.extra = extra

    def set(self, **kv) -> None:
        """Attach fields (e.g. n_out/bytes_out) to the END line, from inside
        the ``with`` block, once they're known (e.g. after computing a
        result)."""
        self.extra.update(kv)

    def __call__(self, fn):
        site = f"{os.path.basename(fn.__code__.co_filename)}:{fn.__code__.co_firstlineno}"
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _ENABLED:
                return fn(*args, **kwargs)
            call_extra = {"site": site}
            call_extra.update(_in_metrics(sig, args, kwargs))
            call_extra.update(self.extra)
            op_ctx = trace_op(self.domain, self.op, **call_extra)
            with op_ctx:
                result = fn(*args, **kwargs)
                try:
                    op_ctx.set(**_out_metrics(result))
                except Exception:
                    pass
                return result
        return wrapper

    def __enter__(self):
        if not _ENABLED:
            return self
        self._call_id = _next_call_id()
        tid = threading.get_ident()
        self._mem0 = _rss_kb()
        self._t0 = time.perf_counter()
        _emit(f"[optrace] BEGIN domain={self.domain} op={self.op} "
              f"call={self._call_id} tid={tid} ts={time.time():.6f}"
              f"{_fmt_extra(self.extra)}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if not _ENABLED:
            return False
        dur_ms = (time.perf_counter() - self._t0) * 1000.0
        tid = threading.get_ident()
        mem1 = _rss_kb()
        if mem1 is not None and self._mem0 is not None:
            self.extra["mem_delta_kb"] = f"{mem1 - self._mem0:.0f}"
        _emit(f"[optrace] END   domain={self.domain} op={self.op} "
              f"call={self._call_id} tid={tid} ts={time.time():.6f} "
              f"dur_ms={dur_ms:.3f} ok={exc_type is None}"
              f"{_fmt_extra(self.extra)}")
        return False


def hit(domain: str, op: str, **extra) -> None:
    """Log a single one-line marker, iff tracing is enabled.

    Unlike ``trace_op``, this isn't a timed BEGIN/END pair -- it's for
    confirming which branch of an if/else a call actually took (e.g.
    fast-path vs fallback, in-place vs backup-and-rewrite) so a run's LOG can
    answer "did the new code path get hit, and how often" via::

        grep -a "\\[optrace\\] HIT" LOG | awk '...'
    """
    if not _ENABLED:
        return
    _emit(f"[optrace] HIT   domain={domain} op={op} tid={threading.get_ident()} "
          f"ts={time.time():.6f}{_fmt_extra(extra)}")


def traced(domain: str, op: str = None):
    """Method decorator: ``@traced("dataframe", "dfm.upsert_df")``.

    ``op`` defaults to the wrapped function's qualified name.
    """
    def deco(fn):
        name = op or fn.__qualname__
        return trace_op(domain, name)(fn)
    return deco


class TracingDuckDBConn:
    """Transparent proxy around a duckdb connection: traces ``execute``/
    ``sql``, delegates everything else (register/unregister/close/...)
    untouched. Only construct this when tracing is enabled — with it off,
    keep using the raw connection so there is zero added indirection.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn):
        self._conn = conn

    def execute(self, *args, **kwargs):
        sql = sanitize(args[0]) if args else ""
        with trace_op("duckdb", "duckdb.execute", sql=sql):
            return self._conn.execute(*args, **kwargs)

    def sql(self, *args, **kwargs):
        sql = sanitize(args[0]) if args else ""
        with trace_op("duckdb", "duckdb.sql", sql=sql):
            return self._conn.sql(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def maybe_wrap_duckdb_conn(conn):
    """Wrap ``conn`` for tracing iff WL_OPTRACE is on, else return it as-is."""
    return TracingDuckDBConn(conn) if _ENABLED else conn
