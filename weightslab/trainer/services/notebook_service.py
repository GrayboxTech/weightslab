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
import ast
import json
import time
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
# Kernel
# ---------------------------------------------------------------------------

class NotebookKernel:
    """A single shared Python namespace executed on one dedicated thread.

    Namespace is seeded from the ledger accessors so the notebook sees the live
    experiment objects. Output (stdout / stderr / last-expression repr / matplotlib
    figures) is captured per cell and returned as a list of ``(kind, payload)``
    tuples for the service to stream.
    """

    def __init__(self, data_service, root_log_dir: Path):
        self._data_service = data_service
        self._root_log_dir = Path(root_log_dir).resolve()
        self._exec_count = 0
        self._ns = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="WL-Notebook-Kernel"
        )
        _WriteGuard.install()
        self._plt = self._try_import_matplotlib()
        self._seed_namespace()

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
        try:
            if ledgers.Proxy.is_proxy(value):
                return value.get()
        except Exception:
            pass
        return value

    def _get_df(self):
        """Fetch a fresh live view of the ledger dataframe."""
        ds = self._data_service
        try:
            if hasattr(ds, "_pull_into_all_data_view_df"):
                return ds._pull_into_all_data_view_df()
        except Exception as exc:
            logger.debug("get_df pull failed, falling back: %s", exc)
        return getattr(ds, "_all_datasets_df", None)

    def _seed_namespace(self):
        import numpy as np
        import pandas as pd
        import weightslab as wl

        def _safe(getter):
            try:
                return self._unwrap(getter())
            except Exception as exc:
                logger.debug("namespace seed skipped (%s)", exc)
                return None

        ns = self._ns
        ns.clear()
        ns["__name__"] = "__wl_notebook__"
        ns["pd"] = pd
        ns["np"] = np
        ns["wl"] = wl
        ns["plt"] = self._plt
        ns["get_df"] = self._get_df
        ns["df"] = self._get_df()
        ns["model"] = _safe(ledgers.get_model)
        ns["cm"] = _safe(ledgers.get_checkpoint_manager)
        ns["logger"] = _safe(ledgers.get_logger)
        ns["hp"] = _safe(ledgers.get_hyperparams)
        ns["root_log_dir"] = str(self._root_log_dir)

    # -- execution ---------------------------------------------------------
    def run(self, code: str):
        """Submit a cell to the kernel thread and wait for its captured output."""
        return self._executor.submit(self._run_on_kernel_thread, code).result()

    def _run_on_kernel_thread(self, code: str):
        self._exec_count += 1
        outputs = []
        ok = True
        tb_text = None

        out_buf, err_buf = io.StringIO(), io.StringIO()
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
                with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                    result_repr = self._exec_with_last_expr(code)
        except BaseException:  # noqa: BLE001 -- surface any user error as a cell error
            ok = False
            tb_text = traceback.format_exc()
            result_repr = None

        stdout = out_buf.getvalue()
        stderr = err_buf.getvalue()
        if stdout:
            outputs.append(("stdout", stdout[:_MAX_TEXT_CHARS]))
        if stderr:
            outputs.append(("stderr", stderr[:_MAX_TEXT_CHARS]))
        if result_repr is not None:
            outputs.append(("result_text", result_repr[:_MAX_TEXT_CHARS]))
        outputs.extend(("image_png", png) for png in self._collect_figures())
        if tb_text:
            outputs.append(("error_traceback", tb_text[:_MAX_TEXT_CHARS]))

        return {"outputs": outputs, "ok": ok, "exec_count": self._exec_count}

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

    def _notebook_path(self) -> Path:
        return self._root_log_dir / NOTEBOOK_FILENAME

    def _get_kernel(self) -> NotebookKernel:
        with self._kernel_lock:
            if self._kernel is None:
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
        """A minimal nbformat v4 notebook shown on first open."""
        def code(src):
            return {"cell_type": "code", "metadata": {}, "source": src,
                    "execution_count": None, "outputs": []}

        def md(src):
            return {"cell_type": "markdown", "metadata": {}, "source": src}

        cells = [
            md("# WeightsLab notebook\n"
               "This notebook runs inside the live experiment process. The following "
               "handles are pre-bound:\n\n"
               "- `df` / `get_df()` - the ledger dataframe (live view)\n"
               "- `model`, `cm` (checkpoints), `logger`, `hp` (hyperparameters)\n"
               "- `pd`, `np`, `plt`, `wl`\n\n"
               "Writes are only allowed under `root_log_dir`. Start a cell with `>` "
               "to ask the agent to propose code for you."),
            code("df.head()"),
            code("# Available checkpoints\ncm.get_all_hashes() if cm is not None else 'no checkpoint manager'"),
            code("> plot a histogram of the samples per dataset split"),
        ]
        return {
            "cells": cells,
            "metadata": {"kernelspec": {"name": "weightslab", "display_name": "WeightsLab (shared)"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    # -- gRPC methods ------------------------------------------------------
    def RunNotebookCell(self, request, context):
        """Server-streaming: execute one cell and yield its output chunks."""
        cell_id = request.cell_id or ""
        code = request.code or ""
        started = time.perf_counter()
        try:
            result = self._get_kernel().run(code)
        except Exception as exc:  # kernel-level failure (not a user error)
            logger.error("RunNotebookCell kernel failure: %s", exc, exc_info=True)
            self._audit("notebook_run", "error", {"cell_id": cell_id}, error=str(exc))
            yield pb2.NotebookCellChunk(cell_id=cell_id, error_traceback=str(exc))
            yield pb2.NotebookCellChunk(
                cell_id=cell_id, done=pb2.NotebookCellDone(exec_count=0, ok=False))
            return

        for kind, payload in result["outputs"]:
            if kind == "image_png":
                yield pb2.NotebookCellChunk(cell_id=cell_id, image_png=payload)
            else:
                yield pb2.NotebookCellChunk(cell_id=cell_id, **{kind: payload})

        yield pb2.NotebookCellChunk(
            cell_id=cell_id,
            done=pb2.NotebookCellDone(exec_count=result["exec_count"], ok=result["ok"]),
        )
        self._audit(
            "notebook_run",
            "success" if result["ok"] else "error",
            {"cell_id": cell_id, "code": code[:2000],
             "elapsed_s": round(time.perf_counter() - started, 3)},
        )

    @safe_grpc(lambda msg: pb2.NotebookResponse(ipynb_json="", existed=False, path=""))
    def GetNotebook(self, request, context):
        path = self._notebook_path()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return pb2.NotebookResponse(ipynb_json=text, existed=True, path=str(path))
        # Write the default template on first use (inside root_log_dir).
        default = json.dumps(self._default_notebook(), indent=1)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(default, encoding="utf-8")
        except Exception as exc:
            logger.warning("could not persist default notebook: %s", exc)
        return pb2.NotebookResponse(ipynb_json=default, existed=False, path=str(path))

    @safe_grpc(lambda msg: pb2.SaveNotebookResponse(ok=False, path="", error=msg))
    def SaveNotebook(self, request, context):
        path = self._notebook_path()
        ipynb = request.ipynb_json or ""
        # Validate it is JSON before writing so we never persist a corrupt file.
        try:
            json.loads(ipynb)
        except Exception as exc:
            return pb2.SaveNotebookResponse(ok=False, path=str(path), error=f"invalid notebook JSON: {exc}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ipynb, encoding="utf-8")
        except Exception as exc:
            return pb2.SaveNotebookResponse(ok=False, path=str(path), error=str(exc))
        self._audit("notebook_save", "success", {"path": str(path), "bytes": len(ipynb)})
        return pb2.SaveNotebookResponse(ok=True, path=str(path), error="")

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
