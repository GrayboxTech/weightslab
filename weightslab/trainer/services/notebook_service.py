"""
services/notebook_service.py
============================
gRPC surface + shared in-process kernel for the studio notebook.

The notebook is an extension of the running backend: cells execute in ONE shared
Python namespace living inside the training process, so ``df``, the model, and the
checkpoints are the very objects the training loop uses -- not copies. Cells are
run one at a time on a single dedicated kernel thread so they never block the gRPC
worker pool and never race each other.

Rights are constrained by guardrails, not by OS isolation (Python cannot be safely
sandboxed in-process): while a cell runs, file WRITES are only permitted under the
experiment ``root_log_dir`` -- everywhere else is read-only. Every cell is recorded
through the existing AuditLogger. This is pragmatic protection for a trusted single
operator; it is explicitly NOT a defence against a determined user who already
controls the process.

The notebook document itself is persisted as ``root_log_dir/notebook.ipynb`` (nbformat
v4 JSON) so reloading an experiment and reopening the notebook restores it; a default
template is written on first use.

Wire-up (in ExperimentService):
    notebook_service = NotebookService(data_service, root_log_dir)
"""

import io
import os
import re
import ast
import json
import time
import ctypes
import queue
import shutil
import logging
import builtins
import threading
import traceback
import contextlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import weightslab.proto.experiment_service_pb2 as pb2

from weightslab.backend import ledgers
from weightslab.trainer.services.utils.tools import safe_grpc

logger = logging.getLogger(__name__)

# The notebook file lives directly under root_log_dir so it travels with the
# experiment's checkpoints/logs.
NOTEBOOK_FILENAME = "notebook.ipynb"

# Cap on a single streamed text chunk so a runaway print loop cannot buffer an
# unbounded string before the "done" chunk is sent.
_MAX_TEXT_CHARS = 200_000


def _async_raise(thread_id: int, exc_type) -> bool:
    """Best-effort: asynchronously raise `exc_type` in another thread at its
    next bytecode boundary (the canonical ctypes recipe for interrupting a
    thread we don't otherwise control). Our kernels aren't subprocesses --
    ipykernel's normal signal-based interrupt doesn't apply here (it also
    can't be installed at all on a non-main thread, which is what an embedded
    kernel always runs on) -- so this is the only way to stop a runaway cell.
    """
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(exc_type))
    if res > 1:
        # Hit more than one thread (shouldn't happen with a real ident) --
        # undo it per the canonical recipe so we don't corrupt thread state.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        return False
    return res == 1


# ---------------------------------------------------------------------------
# Write-guard
# ---------------------------------------------------------------------------

class _WriteGuard:
    """Best-effort filesystem write restriction, active only while a notebook cell
    runs on the kernel thread.

    It patches ``builtins.open`` and the common destructive ``os`` / ``shutil``
    entry points ONCE, process-wide, but each patched function is a no-op unless a
    thread-local "enforcing" flag is set -- and only the kernel thread ever sets it.
    So the training loop and gRPC workers are never affected.

    When enforcing, writes must resolve to a path under ``root_log_dir``. Relative
    paths passed to ``open`` are rewritten to live under ``root_log_dir`` so that a
    plain ``open("out.csv", "w")`` lands there instead of the process CWD.
    """

    _installed = False
    _lock = threading.Lock()
    _local = threading.local()
    # Instances keyed by nothing -- the guard is global; the active root is stored
    # per-thread so multiple experiments in one process each enforce their own root.

    _orig_open = None
    _orig_os = {}
    _orig_shutil = {}

    @classmethod
    def install(cls):
        with cls._lock:
            if cls._installed:
                return
            cls._orig_open = builtins.open

            def guarded_open(file, mode="r", *args, **kwargs):
                if cls._enforcing() and cls._is_write_mode(mode):
                    file = cls._check_and_rewrite(file)
                return cls._orig_open(file, mode, *args, **kwargs)

            builtins.open = guarded_open

            for name in ("remove", "unlink", "rmdir", "mkdir", "makedirs"):
                if hasattr(os, name):
                    cls._orig_os[name] = getattr(os, name)
                    setattr(os, name, cls._make_os_guard(cls._orig_os[name]))
            for name in ("rename", "replace"):
                if hasattr(os, name):
                    cls._orig_os[name] = getattr(os, name)
                    setattr(os, name, cls._make_os_guard2(cls._orig_os[name]))
            for name in ("rmtree", "move", "copy", "copyfile", "copytree"):
                if hasattr(shutil, name):
                    cls._orig_shutil[name] = getattr(shutil, name)
                    setattr(shutil, name, cls._make_shutil_guard(name, cls._orig_shutil[name]))

            cls._installed = True

    # -- enforcement scope -------------------------------------------------
    @classmethod
    @contextlib.contextmanager
    def enforce(cls, root_log_dir: Path):
        prev_root = getattr(cls._local, "root", None)
        prev_on = getattr(cls._local, "on", False)
        cls._local.root = Path(root_log_dir).resolve()
        cls._local.on = True
        try:
            yield
        finally:
            cls._local.on = prev_on
            cls._local.root = prev_root

    @classmethod
    def _enforcing(cls) -> bool:
        return getattr(cls._local, "on", False)

    @classmethod
    def _root(cls) -> Path:
        return getattr(cls._local, "root", None)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _is_write_mode(mode) -> bool:
        try:
            m = str(mode)
        except Exception:
            return False
        return any(ch in m for ch in ("w", "a", "x", "+"))

    @classmethod
    def _resolve(cls, path) -> Path:
        root = cls._root()
        p = Path(os.fspath(path))
        if not p.is_absolute():
            p = root / p
        return p

    @classmethod
    def _within_root(cls, path) -> bool:
        root = cls._root()
        if root is None:
            return True
        try:
            resolved = cls._resolve(path).resolve()
            resolved.relative_to(root)
            return True
        except Exception:
            return False

    @classmethod
    def _check_and_rewrite(cls, path):
        """For open(): rewrite relatives under root, reject anything outside it."""
        rewritten = cls._resolve(path)
        try:
            rewritten.resolve().relative_to(cls._root())
        except Exception:
            raise PermissionError(
                f"Notebook kernel may only write under {cls._root()} "
                f"(attempted: {os.fspath(path)})"
            )
        # Preserve the original argument type where possible.
        if not Path(os.fspath(path)).is_absolute():
            return str(rewritten)
        return os.fspath(path)

    @classmethod
    def _deny(cls, path):
        raise PermissionError(
            f"Notebook kernel may only write under {cls._root()} "
            f"(attempted: {os.fspath(path)})"
        )

    @classmethod
    def _make_os_guard(cls, orig):
        def guard(path, *a, **k):
            if cls._enforcing() and not cls._within_root(path):
                cls._deny(path)
            return orig(path, *a, **k)
        return guard

    @classmethod
    def _make_os_guard2(cls, orig):
        def guard(src, dst, *a, **k):
            if cls._enforcing() and not cls._within_root(dst):
                cls._deny(dst)
            return orig(src, dst, *a, **k)
        return guard

    @classmethod
    def _make_shutil_guard(cls, name, orig):
        def guard(*a, **k):
            if cls._enforcing() and len(a) >= (1 if name == "rmtree" else 2):
                target = a[0] if name == "rmtree" else a[1]
                if not cls._within_root(target):
                    cls._deny(target)
            return orig(*a, **k)
        return guard


# ---------------------------------------------------------------------------
# Shared namespace builder -- used by both NotebookKernel (below) and the
# embedded real-kernel bridge, so both engines bind identical objects.
# ---------------------------------------------------------------------------

def get_df(data_service):
    """Fetch a fresh live view of the ledger dataframe from ``data_service``."""
    try:
        if hasattr(data_service, "_pull_into_all_data_view_df"):
            return data_service._pull_into_all_data_view_df()
    except Exception as exc:
        logger.debug("get_df pull failed, falling back: %s", exc)
    return getattr(data_service, "_all_datasets_df", None)


def _unwrap_proxy(value):
    try:
        if ledgers.Proxy.is_proxy(value):
            return value.get()
    except Exception:
        pass
    return value


def build_notebook_namespace(data_service, root_log_dir, plt=None) -> dict:
    """The dict every execution engine seeds a cell's namespace with."""
    import numpy as np
    import pandas as pd
    import weightslab as wl

    def _safe(getter):
        try:
            return _unwrap_proxy(getter())
        except Exception as exc:
            logger.debug("namespace seed skipped (%s)", exc)
            return None

    root = Path(root_log_dir).resolve()
    return {
        "__name__": "__wl_notebook__",
        "pd": pd, "np": np, "wl": wl, "plt": plt,
        "get_df": lambda: get_df(data_service),
        "df": get_df(data_service),
        "model": _safe(ledgers.get_model),
        "cm": _safe(ledgers.get_checkpoint_manager),
        "logger": _safe(ledgers.get_logger),
        "hp": _safe(ledgers.get_hyperparams),
        "root_log_dir": str(root),
    }


# ---------------------------------------------------------------------------
# Embedded real-kernel support (optional; requires
# `pip install weightslab[notebook-kernel]`). wl.serve() calls
# configure_embedded_kernel() once, before the first NotebookService is
# constructed; NotebookService._get_kernel() attaches to the resulting real
# Jupyter kernel when available and falls back to NotebookKernel otherwise.
# ---------------------------------------------------------------------------

_EMBED_ENABLED = False
_EMBED_LOCK = threading.Lock()
_EMBED_STATE = {"started": False, "connection_file": None, "kernel_thread_id": None}
# Rebound on every NotebookService construction (i.e. every watchdog restart)
# so the one long-lived embedded kernel thread always refreshes df/root_log_dir
# against whichever data_service/root_log_dir is currently live.
_ACTIVE_BINDING = {"data_service": None, "root_log_dir": None}


def configure_embedded_kernel(enabled: bool) -> None:
    """Record wl.serve()'s enable/disable decision for the embedded kernel."""
    global _EMBED_ENABLED
    _EMBED_ENABLED = bool(enabled)


def _ipykernel_available() -> bool:
    try:
        import ipykernel  # noqa: F401
        import jupyter_client  # noqa: F401
        return True
    except Exception:
        return False


def _wait_for_connection_file(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def ensure_embedded_kernel(data_service, root_log_dir: Path) -> None:
    """Idempotent: starts the real kernel at most once per process."""
    _ACTIVE_BINDING["data_service"] = data_service
    _ACTIVE_BINDING["root_log_dir"] = Path(root_log_dir).resolve()

    if not _EMBED_ENABLED:
        return
    with _EMBED_LOCK:
        if _EMBED_STATE["started"]:
            return
        _EMBED_STATE["started"] = True  # decided either way; never retry
        if not _ipykernel_available():
            logger.warning(
                "Embedded notebook kernel requested but `ipykernel`/"
                "`jupyter_client` are not installed; the studio notebook "
                "panel will use the built-in executor. Install with "
                "`pip install weightslab[notebook-kernel]`."
            )
            return
        connection_file = _ACTIVE_BINDING["root_log_dir"] / "notebook_kernel.json"
        threading.Thread(
            target=_run_embedded_kernel, args=(connection_file,),
            name="WL-Embedded-Jupyter-Kernel", daemon=True,
        ).start()
        if _wait_for_connection_file(connection_file, timeout=15.0):
            _EMBED_STATE["connection_file"] = connection_file
        else:
            logger.warning(
                "Embedded kernel did not write its connection file within "
                "15s (%s); RunNotebookCell will fall back to the built-in "
                "executor.", connection_file)


def get_embedded_kernel_connection_file(wait_timeout: float = 8.0):
    """Returns the connection file once the embedded kernel is up, or None if
    embedding wasn't requested or definitively failed."""
    if not _EMBED_ENABLED:
        return None
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        if _EMBED_STATE["connection_file"] is not None:
            return _EMBED_STATE["connection_file"]
        if _EMBED_STATE["started"]:
            return None  # decided (ipykernel missing / embed failed) -- stop polling
        time.sleep(0.05)
    return None


def _run_embedded_kernel(connection_file: Path) -> None:
    import asyncio
    from ipykernel.kernelapp import IPKernelApp

    # Recorded so EmbeddedKernelBridge.interrupt() can target the thread that
    # actually executes cell code (NOT the client-side thread that calls
    # execute_interactive() and blocks waiting for it).
    _EMBED_STATE["kernel_thread_id"] = threading.get_ident()

    # ipykernel's IOLoop needs a running asyncio loop on THIS thread.
    asyncio.set_event_loop(asyncio.new_event_loop())

    ns = build_notebook_namespace(
        _ACTIVE_BINDING["data_service"], _ACTIVE_BINDING["root_log_dir"])

    app = IPKernelApp.instance(connection_file=str(connection_file), matplotlib="inline")
    try:
        app.initialize([])
        app.shell.colors = "NoColor"
        try:
            app.shell.enable_matplotlib("inline")
        except Exception:
            pass
        # IPython execs cell code as exec(code, user_global_ns, user_ns) --
        # user_global_ns defaults to a *different*, unrelated module's
        # __dict__ (whatever default module the shell created during
        # app.initialize() above) unless user_module/user_ns are rebuilt
        # together. A plain `kernel.user_ns = ns` (what ipykernel's own
        # observer does) only replaces user_ns, leaving user_global_ns
        # pointing at that stale empty module -- so top-level code sees `ns`
        # (LOAD_NAME checks locals first) but any `def f(): ...` defined in a
        # cell gets `__globals__ = user_global_ns` and raises NameError on
        # module-level names like `logger`/`df` the moment it's *called*.
        # prepare_user_module(None, ns) is IPython's own recipe (used for
        # normal, non-embedded interactive sessions) for making user_ns and
        # user_global_ns the SAME dict, so nested functions see everything
        # top-level code does.
        # IPython execs cell code as exec(code, user_global_ns, user_ns) --
        # user_global_ns defaults to a *different*, unrelated module's
        # __dict__ (whatever default module the shell created during
        # app.initialize() above) unless user_module/user_ns are rebuilt
        # together. A plain `kernel.user_ns = ns` (what ipykernel's own
        # observer does) only replaces user_ns, leaving user_global_ns
        # pointing at that stale empty module -- so top-level code sees `ns`
        # (LOAD_NAME checks locals first) but any `def f(): ...` defined in a
        # cell gets `__globals__ = user_global_ns` and raises NameError on
        # module-level names like `logger`/`df` the moment it's *called*.
        # prepare_user_module(None, ns) is IPython's own recipe (used for
        # normal, non-embedded interactive sessions) for making user_ns and
        # user_global_ns the SAME dict, so nested functions see everything
        # top-level code does.
        app.shell.user_module, app.shell.user_ns = app.shell.prepare_user_module(None, ns)
        app.shell.init_user_ns()
        app.shell.ns_table["user_global"] = app.shell.user_module.__dict__
        app.shell.ns_table["user_local"] = app.shell.user_ns
        app.shell.set_completer_frame()
        # ipykernel's OutStream batches stdout/stderr writes and only ships
        # them to iopub every `flush_interval` seconds (default 0.2s) -- fine
        # for a notebook server with many kernels, but sluggish for a single
        # embedded kernel where we want prints to show up close to instantly.
        import sys as _sys
        for _stream in (_sys.stdout, _sys.stderr):
            if hasattr(_stream, "flush_interval"):
                _stream.flush_interval = 0.05
        _install_kernel_hooks(app.shell)
        logger.info("Embedded Jupyter kernel connection file: %s", connection_file)
        app.start()  # blocks this thread forever (event loop)
    except Exception:
        logger.error("Embedded notebook kernel crashed during startup", exc_info=True)


def _install_kernel_hooks(shell) -> None:
    """Bracket every cell with the same write-guard + df-refresh the legacy
    kernel applies -- but as IPython events, which run ON the kernel's own
    execution thread (unlike the legacy path's executor thread)."""
    _WriteGuard.install()
    box = {"guard_cm": None}

    def _pre_execute():
        try:
            shell.user_ns["df"] = get_df(_ACTIVE_BINDING["data_service"])
        except Exception:
            pass
        cm = _WriteGuard.enforce(_ACTIVE_BINDING["root_log_dir"])
        cm.__enter__()
        box["guard_cm"] = cm

    def _post_execute():
        cm = box.pop("guard_cm", None)
        if cm is not None:
            cm.__exit__(None, None, None)

    shell.events.register("pre_execute", _pre_execute)
    shell.events.register("post_execute", _post_execute)


def _capped(kind: str, payload):
    """Truncate a single output message so a runaway print can't buffer an
    unbounded string before it reaches the client (applied per-message now
    that output streams live, rather than once over the whole cell)."""
    if kind in ("stdout", "stderr", "result_text", "error_traceback") and len(payload) > _MAX_TEXT_CHARS:
        return payload[:_MAX_TEXT_CHARS]
    return payload


def _append_iopub_output(emit, msg: dict) -> None:
    """Translate one iopub message into `emit(kind, payload)` call(s), live --
    `emit` is called as each message arrives (e.g. a queue.put), not collected
    into a list, so RunNotebookCell can stream chunks to the client as the
    cell actually produces them."""
    msg_type = msg.get("msg_type") or msg.get("header", {}).get("msg_type")
    content = msg.get("content", {})
    if msg_type == "stream":
        name = content.get("name", "stdout")
        kind = name if name in ("stdout", "stderr") else "stdout"
        emit(kind, _capped(kind, content.get("text", "")))
    elif msg_type in ("execute_result", "display_data"):
        data = content.get("data", {})
        text = data.get("text/plain")
        png_b64 = data.get("image/png")
        # Always show text/plain for execute_result (the auto-printed last
        # expression). For display_data (explicit display(...) calls), only
        # fall back to text when there's no image -- otherwise a matplotlib
        # figure's plain-text repr ("<Figure size ...>") would show up as
        # noise alongside the PNG it's already rendered as.
        if text and (msg_type == "execute_result" or not png_b64):
            emit("result_text", _capped("result_text", text))
        if png_b64:
            import base64
            emit("image_png", base64.b64decode(png_b64))
    elif msg_type == "error":
        tb = "\n".join(content.get("traceback", []))
        emit("error_traceback", _capped("error_traceback", tb))


class EmbeddedKernelBridge:
    """Adapts the real, already-embedded Jupyter kernel to a streaming
    interface: ``run_streaming(code)`` is a generator yielding ``(kind,
    payload)`` tuples LIVE as the cell produces them (iopub messages arrive
    while ``execute_interactive`` is still running), ending with a
    ``("__done__", {"ok":..., "exec_count":...})`` marker. Not safe for
    concurrent calls (jupyter_client's ZMQ channels are single-writer/
    single-reader), so it serializes with its own lock rather than relying on
    NotebookService's kernel-construction lock.
    """

    def __init__(self, connection_file: Path, startup_timeout: float = 30.0):
        from jupyter_client import BlockingKernelClient
        self._lock = threading.Lock()
        self._busy = False
        self._client = BlockingKernelClient(connection_file=str(connection_file))
        self._client.load_connection_file()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=startup_timeout)

    def run_streaming(self, code: str):
        q = queue.Queue()
        _DONE = object()

        def _work():
            self._busy = True
            try:
                with self._lock:
                    reply = self._client.execute_interactive(
                        code, allow_stdin=False, timeout=None,
                        output_hook=lambda msg: _append_iopub_output(
                            lambda kind, payload: q.put((kind, payload)), msg),
                    )
                content = reply.get("content", {})
                final = {
                    "ok": content.get("status") == "ok",
                    "exec_count": content.get("execution_count") or 0,
                }
            except Exception as exc:
                final = {"ok": False, "exec_count": 0, "error": str(exc)}
            finally:
                self._busy = False
            q.put((_DONE, final))

        threading.Thread(target=_work, daemon=True, name="WL-Notebook-Cell-Run").start()
        while True:
            item = q.get()
            if item[0] is _DONE:
                final = item[1]
                break
            yield item
        if final.get("error"):
            yield ("error_traceback", final["error"])
        yield ("__done__", {"ok": final["ok"], "exec_count": final["exec_count"]})

    def interrupt(self) -> bool:
        """Best-effort: only fires while a cell is actually executing, so an
        idle kernel's own message-processing loop is never touched."""
        if not self._busy:
            return False
        tid = _EMBED_STATE.get("kernel_thread_id")
        if tid is None:
            return False
        return _async_raise(tid, KeyboardInterrupt)

    def close(self):
        try:
            self._client.stop_channels()
        except Exception:
            pass


class _LiveStream:
    """Write-only file-like object that forwards each write directly to
    ``emit(kind, text)`` instead of buffering into a StringIO -- lets stdout/
    stderr reach the gRPC client as the cell actually prints, rather than only
    after the whole cell finishes."""

    def __init__(self, kind: str, emit):
        self._kind = kind
        self._emit = emit

    def write(self, s):
        if s:
            self._emit(self._kind, _capped(self._kind, s))
        return len(s)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class NotebookKernel:
    """A single shared Python namespace executed on one dedicated thread.

    Namespace is seeded from the ledger accessors so the notebook sees the live
    experiment objects. stdout/stderr stream live via ``run_streaming()`` as the
    cell prints; the last-expression repr, matplotlib figures, and any traceback
    are only known once the cell finishes and are yielded right before it.
    """

    def __init__(self, data_service, root_log_dir: Path):
        self._data_service = data_service
        self._root_log_dir = Path(root_log_dir).resolve()
        self._exec_count = 0
        self._ns = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="WL-Notebook-Kernel"
        )
        self._busy = False
        self._current_thread_id = None
        _WriteGuard.install()
        self._plt = self._try_import_matplotlib()
        self._seed_namespace()

    def interrupt(self) -> bool:
        """Best-effort: only fires while a cell is actually executing, so we
        never inject an exception into the worker thread while it's idly
        blocked waiting for the next submitted cell (that thread is reused
        for every future cell -- an ill-timed interrupt there could corrupt
        ThreadPoolExecutor's own internal bookkeeping)."""
        if not self._busy or self._current_thread_id is None:
            return False
        return _async_raise(self._current_thread_id, KeyboardInterrupt)

    # -- setup -------------------------------------------------------------
    @staticmethod
    def _try_import_matplotlib():
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            return plt
        except Exception as exc:
            logger.info("matplotlib unavailable in notebook kernel: %s", exc)
            return None

    @staticmethod
    def _unwrap(value):
        return _unwrap_proxy(value)

    def _get_df(self):
        """Fetch a fresh live view of the ledger dataframe."""
        return get_df(self._data_service)

    def _seed_namespace(self):
        ns = self._ns
        ns.clear()
        ns.update(build_notebook_namespace(self._data_service, self._root_log_dir, plt=self._plt))

    # -- execution ---------------------------------------------------------
    def run_streaming(self, code: str):
        """Generator: yields (kind, payload) tuples LIVE as the cell prints to
        stdout/stderr (the exec itself runs on the single kernel-worker thread;
        this generator drains a queue fed by that thread as it produces
        output), then result_text/image_png/error_traceback -- only knowable
        once the cell finishes -- and finally a
        ("__done__", {"ok":..., "exec_count":...}) marker.
        """
        q = queue.Queue()
        _DONE = object()

        def _emit(kind, payload):
            q.put((kind, payload))

        def _work():
            final = self._run_on_kernel_thread(code, _emit)
            q.put((_DONE, final))

        self._executor.submit(_work)
        while True:
            item = q.get()
            if item[0] is _DONE:
                final = item[1]
                break
            yield item

        if final["result_repr"] is not None:
            yield ("result_text", _capped("result_text", final["result_repr"]))
        for png in final["figures"]:
            yield ("image_png", png)
        if final["tb_text"]:
            yield ("error_traceback", _capped("error_traceback", final["tb_text"]))
        yield ("__done__", {"ok": final["ok"], "exec_count": final["exec_count"]})

    def _run_on_kernel_thread(self, code: str, emit):
        self._exec_count += 1
        ok = True
        tb_text = None
        self._current_thread_id = threading.get_ident()
        self._busy = True

        # Refresh the "df" convenience binding to the current view before each run
        # (users can still rebind it; get_df() always returns the freshest).
        try:
            self._ns["df"] = self._get_df()
        except Exception:
            pass

        if self._plt is not None:
            try:
                self._plt.close("all")
            except Exception:
                pass

        try:
            with _WriteGuard.enforce(self._root_log_dir):
                with contextlib.redirect_stdout(_LiveStream("stdout", emit)), \
                     contextlib.redirect_stderr(_LiveStream("stderr", emit)):
                    result_repr = self._exec_with_last_expr(code)
        except BaseException:  # noqa: BLE001 -- surface any user error (incl. an
            # interrupt() -injected KeyboardInterrupt) as a cell error, not a crash.
            ok = False
            tb_text = traceback.format_exc()
            result_repr = None
        finally:
            self._busy = False

        return {
            "ok": ok,
            "exec_count": self._exec_count,
            "result_repr": result_repr,
            "tb_text": tb_text,
            "figures": self._collect_figures(),
        }

    def _exec_with_last_expr(self, code: str):
        """Exec ``code`` in the shared namespace; if the last statement is a bare
        expression, evaluate it and return its repr (Jupyter-style)."""
        try:
            parsed = ast.parse(code, mode="exec")
        except SyntaxError:
            # Let exec raise the SyntaxError with a proper traceback.
            exec(compile(code, "<notebook-cell>", "exec"), self._ns)
            return None

        last_expr = None
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last_expr = ast.Expression(parsed.body.pop().value)

        if parsed.body:
            exec(compile(parsed, "<notebook-cell>", "exec"), self._ns)
        if last_expr is not None:
            value = eval(compile(last_expr, "<notebook-cell>", "eval"), self._ns)
            if value is not None:
                try:
                    return repr(value)
                except Exception:
                    return f"<unrepr-able {type(value).__name__}>"
        return None

    def _collect_figures(self):
        pngs = []
        if self._plt is None:
            return pngs
        try:
            for num in self._plt.get_fignums():
                fig = self._plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                pngs.append(buf.getvalue())
            self._plt.close("all")
        except Exception as exc:
            logger.debug("figure capture failed: %s", exc)
        return pngs


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class NotebookService:
    """gRPC facade for notebook cell execution, persistence, and code generation."""

    def __init__(self, data_service, root_log_dir: str = None):
        self._data_service = data_service
        self._root_log_dir = self._resolve_root_log_dir(root_log_dir)
        self._kernel = None
        self._kernel_lock = threading.Lock()
        # Base name (file stem, no .ipynb) of the notebook currently in use this
        # session. None until the first Get/Save resolves it.
        self._active_name = None
        # Eager: start (or rebind) the embedded kernel as soon as we know
        # data_service/root_log_dir, so external tools (`jupyter console
        # --existing <file>`) and the studio panel can both attach without
        # waiting on a lazy first-cell trigger.
        ensure_embedded_kernel(self._data_service, self._root_log_dir)

    # -- helpers -----------------------------------------------------------
    def _resolve_root_log_dir(self, root_log_dir) -> Path:
        candidates = [root_log_dir]
        try:
            cm = ledgers.get_checkpoint_manager()
            if ledgers.Proxy.is_proxy(cm):
                cm = cm.get()
            if cm is not None:
                candidates.append(getattr(cm, "root_log_dir", None))
        except Exception:
            pass
        candidates.append(getattr(self._data_service, "_root_log_dir", None))
        candidates.append(os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR"))
        for c in candidates:
            if c:
                return Path(c).resolve()
        return Path("./logs").resolve()

    @staticmethod
    def _sanitize_stem(name: str) -> str:
        """Reduce a user-supplied notebook name to a safe file stem (no .ipynb)."""
        stem = (name or "").strip()
        if stem.lower().endswith(".ipynb"):
            stem = stem[:-len(".ipynb")]
        stem = os.path.basename(stem)                     # drop any path components
        stem = re.sub(r"[^A-Za-z0-9._ -]", "_", stem).strip()
        return stem or Path(NOTEBOOK_FILENAME).stem       # fall back to "notebook"

    def _path_for(self, stem: str) -> Path:
        return self._root_log_dir / f"{stem}.ipynb"

    def _active_path(self) -> Path:
        """The current notebook file: the one used this session, else the most
        recently modified *.ipynb under root_log_dir, else the default."""
        if self._active_name:
            return self._path_for(self._active_name)
        try:
            existing = sorted(self._root_log_dir.glob("*.ipynb"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if existing:
                self._active_name = existing[0].stem
                return existing[0]
        except Exception:
            pass
        self._active_name = Path(NOTEBOOK_FILENAME).stem
        return self._path_for(self._active_name)

    def _unique_path(self, stem: str, keep: Path = None) -> Path:
        """A non-colliding path for ``stem``.ipynb; indexes ``-1, -2, …`` when the
        name is already taken by a *different* file. ``keep`` (the current file)
        never counts as a collision, so renaming to the same name is a no-op."""
        candidate = self._path_for(stem)
        if not candidate.exists() or (keep is not None and candidate == keep):
            return candidate
        i = 1
        while True:
            candidate = self._path_for(f"{stem}-{i}")
            if not candidate.exists() or (keep is not None and candidate == keep):
                return candidate
            i += 1

    def _get_kernel(self):
        with self._kernel_lock:
            if self._kernel is not None:
                return self._kernel
            connection_file = get_embedded_kernel_connection_file()
            if connection_file is not None:
                try:
                    self._kernel = EmbeddedKernelBridge(connection_file)
                    logger.info(
                        "NotebookService: attached to embedded Jupyter kernel (%s)",
                        connection_file)
                    return self._kernel
                except Exception as exc:
                    logger.warning(
                        "Embedded kernel connect failed (%s); falling back to "
                        "built-in executor", exc)
            self._kernel = NotebookKernel(self._data_service, self._root_log_dir)
            return self._kernel

    @property
    def _agent(self):
        return getattr(self._data_service, "_agent", None)

    def _audit(self, action_type, status, details=None, error=None):
        al = getattr(self._data_service, "audit_logger", None)
        if al is None:
            return
        try:
            al.log_event(action_type=action_type, status=status, details=details or {}, error=error)
        except Exception:
            pass

    @staticmethod
    def _default_notebook() -> dict:
        """A minimal nbformat v4 notebook shown on first open: one worked
        example (df + logger + checkpoint hash, all live ledger calls) and one
        blank cell whose placeholder text (rendered client-side, see the
        studio's CodeMirror setup) demonstrates the "> ..." agent convention."""
        def code(src):
            return {"cell_type": "code", "metadata": {}, "source": src,
                    "execution_count": None, "outputs": []}

        cells = [
            code(
                "# WeightsLab notebook: df / logger / cm are the live ledger objects\n"
                "# for this experiment -- no re-loading or re-importing needed.\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
                "\n"
                "# a) Dataframe view: how samples are distributed across the "
                "train/val/test split.\n"
                "# `origin` is normally an index level here, not a column -- read it via\n"
                "# get_level_values (df['origin'] would raise KeyError).\n"
                "origin_series = None\n"
                "if df is not None and len(df):\n"
                "    if \"origin\" in df.columns:\n"
                "        origin_series = df[\"origin\"]\n"
                "    elif \"origin\" in (df.index.names or []):\n"
                "        origin_series = df.index.get_level_values(\"origin\")\n"
                "if origin_series is not None:\n"
                "    origin_series.value_counts().plot(kind=\"bar\", ax=axes[0], color=\"#4c8bf5\")\n"
                "else:\n"
                "    axes[0].text(0.5, 0.5, \"df has no samples yet\", ha=\"center\", va=\"center\")\n"
                "axes[0].set_title(\"Samples per split (df)\")\n"
                "axes[0].set_xlabel(\"origin\")\n"
                "axes[0].set_ylabel(\"count\")\n"
                "\n"
                "# b) Logger view: distribution of a logged signal's recent values\n"
                "# (e.g. training loss) -- signal names are discovered, never hardcoded,\n"
                "# since they depend on how this experiment's training script logs them.\n"
                "signal_names = logger.get_graph_names() if logger is not None else []\n"
                "loss_like = [n for n in signal_names if \"loss\" in n.lower()]\n"
                "target_signal = loss_like[0] if loss_like else (signal_names[0] if signal_names else None)\n"
                "if target_signal:\n"
                "    history = logger.get_current_signaL_history(target_signal)\n"
                "    values = [p[\"metric_value\"] for p in history[-100:]]\n"
                "    axes[1].hist(values, bins=20, color=\"#e06c75\")\n"
                "    axes[1].set_title(f\"'{target_signal}' distribution (last {len(values)} steps)\")\n"
                "else:\n"
                "    axes[1].text(0.5, 0.5, \"no logged signals yet\", ha=\"center\", va=\"center\")\n"
                "    axes[1].set_title(\"Logger signal distribution\")\n"
                "\n"
                "plt.tight_layout()\n"
                "\n"
                "# Bonus: the current experiment/checkpoint hash, straight from the\n"
                "# checkpoint manager (also a live ledger call, like df and logger above).\n"
                "print(\"Current experiment hash:\", cm.get_current_experiment_hash() if cm is not None else None)"
            ),
            code(""),
        ]
        return {
            "cells": cells,
            "metadata": {"kernelspec": {"name": "weightslab", "display_name": "WeightsLab (shared)"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    # -- gRPC methods ------------------------------------------------------
    @safe_grpc(lambda msg: pb2.InterruptNotebookCellResponse(ok=False, error=msg))
    def InterruptNotebookCell(self, request, context):
        # Deliberately reads self._kernel directly, NOT _get_kernel() -- that
        # would lazily construct a fresh kernel just to report nothing is
        # running on it.
        kernel = self._kernel
        if kernel is None or not kernel.interrupt():
            return pb2.InterruptNotebookCellResponse(ok=False, error="No cell is currently running.")
        self._audit("notebook_interrupt", "success", {})
        return pb2.InterruptNotebookCellResponse(ok=True, error="")

    def RunNotebookCell(self, request, context):
        """Server-streaming: execute one cell, yielding each output chunk LIVE
        as the kernel produces it (not buffered until the cell finishes)."""
        cell_id = request.cell_id or ""
        code = request.code or ""
        started = time.perf_counter()
        exec_count = 0
        ok = True
        try:
            kernel = self._get_kernel()
            for kind, payload in kernel.run_streaming(code):
                if kind == "__done__":
                    exec_count = payload["exec_count"]
                    ok = payload["ok"]
                    break
                if kind == "image_png":
                    yield pb2.NotebookCellChunk(cell_id=cell_id, image_png=payload)
                else:
                    yield pb2.NotebookCellChunk(cell_id=cell_id, **{kind: payload})
        except Exception as exc:  # kernel-level failure (not a user error)
            logger.error("RunNotebookCell kernel failure: %s", exc, exc_info=True)
            self._audit("notebook_run", "error", {"cell_id": cell_id}, error=str(exc))
            yield pb2.NotebookCellChunk(cell_id=cell_id, error_traceback=str(exc))
            yield pb2.NotebookCellChunk(
                cell_id=cell_id, done=pb2.NotebookCellDone(exec_count=0, ok=False))
            return

        yield pb2.NotebookCellChunk(
            cell_id=cell_id,
            done=pb2.NotebookCellDone(exec_count=exec_count, ok=ok),
        )
        self._audit(
            "notebook_run",
            "success" if ok else "error",
            {"cell_id": cell_id, "code": code[:2000],
             "elapsed_s": round(time.perf_counter() - started, 3)},
        )

    @safe_grpc(lambda msg: pb2.NotebookResponse(ipynb_json="", existed=False, path="", name=""))
    def GetNotebook(self, request, context):
        path = self._active_path()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return pb2.NotebookResponse(ipynb_json=text, existed=True, path=str(path), name=path.stem)
        # Write the default template on first use (inside root_log_dir).
        default = json.dumps(self._default_notebook(), indent=1)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(default, encoding="utf-8")
        except Exception as exc:
            logger.warning("could not persist default notebook: %s", exc)
        return pb2.NotebookResponse(ipynb_json=default, existed=False, path=str(path), name=path.stem)

    @safe_grpc(lambda msg: pb2.SaveNotebookResponse(ok=False, path="", error=msg, name=""))
    def SaveNotebook(self, request, context):
        ipynb = request.ipynb_json or ""
        # Validate it is JSON before writing so we never persist a corrupt file.
        try:
            json.loads(ipynb)
        except Exception as exc:
            return pb2.SaveNotebookResponse(
                ok=False, path="", error=f"invalid notebook JSON: {exc}",
                name=(self._active_name or ""))

        current = self._active_path()
        requested = (getattr(request, "name", "") or "").strip()
        if requested:
            stem = self._sanitize_stem(requested)
            target = current if stem == current.stem else self._unique_path(stem, keep=current)
        else:
            target = current

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(ipynb, encoding="utf-8")
            # Rename semantics: when the name actually changed, drop the old file.
            if target != current and current.exists():
                try:
                    current.unlink()
                except Exception:
                    pass
            self._active_name = target.stem
        except Exception as exc:
            return pb2.SaveNotebookResponse(ok=False, path=str(target), error=str(exc), name=current.stem)
        self._audit("notebook_save", "success", {"path": str(target), "bytes": len(ipynb)})
        return pb2.SaveNotebookResponse(ok=True, path=str(target), error="", name=target.stem)

    @safe_grpc(lambda msg: pb2.GenerateNotebookCodeResponse(code="", explanation="", ok=False, error=msg))
    def GenerateNotebookCode(self, request, context):
        agent = self._agent
        if agent is None:
            return pb2.GenerateNotebookCodeResponse(
                code="", explanation="", ok=False, error="Agent backend is not running.")
        try:
            code, explanation = agent.generate_code(request.prompt or "", request.context_code or "")
        except Exception as exc:
            logger.info("GenerateNotebookCode failed: %s", exc)
            return pb2.GenerateNotebookCodeResponse(code="", explanation="", ok=False, error=str(exc))
        self._audit("notebook_generate_code", "success", {"prompt": (request.prompt or "")[:500]})
        return pb2.GenerateNotebookCodeResponse(code=code, explanation=explanation, ok=True, error="")
