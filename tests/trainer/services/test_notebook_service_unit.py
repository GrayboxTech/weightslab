"""Unit tests for the shared-kernel notebook service.

These exercise the real NotebookKernel (no LLM required) plus the persistence and
code-generation gRPC surfaces with a lightweight fake DataService, mirroring the
style of test_agent_service_unit.py.
"""

import json
import time
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.services import notebook_service
from weightslab.trainer.services.notebook_service import NotebookService

try:
    import ipykernel  # noqa: F401
    import jupyter_client  # noqa: F401
    _IPYKERNEL_AVAILABLE = True
except ImportError:
    _IPYKERNEL_AVAILABLE = False


def _fake_data_service(df=None, agent=None):
    if df is None:
        df = pd.DataFrame({"origin": ["train", "train", "test"], "loss": [0.1, 0.2, 0.9]})
    return SimpleNamespace(
        _all_datasets_df=df,
        _pull_into_all_data_view_df=lambda: df,
        _root_log_dir=None,
        audit_logger=None,
        _agent=agent,
    )


def _run(service, code, cell_id="c1"):
    chunks = list(service.RunNotebookCell(
        pb2.RunNotebookCellRequest(code=code, cell_id=cell_id), None))
    return chunks


class _NotebookKernelContractTests:
    """Assertions both execution engines (legacy NotebookKernel and the real
    embedded-kernel bridge) must satisfy identically. Subclasses provide
    `_make_service()`; test bodies are shared so parity is proven by
    construction rather than by inspection."""

    def _make_service(self):
        raise NotImplementedError

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = self._make_service()

    def tearDown(self):
        kernel = getattr(self.service, "_kernel", None)
        if kernel is not None and hasattr(kernel, "close"):
            kernel.close()
        self._tmp.cleanup()

    def test_last_expression_repr_and_df_binding(self):
        chunks = _run(self.service, "df.shape")
        texts = [c.result_text for c in chunks if c.WhichOneof("payload") == "result_text"]
        self.assertTrue(any("(3, 2)" in t for t in texts))
        done = [c for c in chunks if c.WhichOneof("payload") == "done"]
        self.assertTrue(done and done[-1].done.ok)

    def test_stdout_capture(self):
        chunks = _run(self.service, "print('hello-notebook')")
        outs = [c.stdout for c in chunks if c.WhichOneof("payload") == "stdout"]
        self.assertTrue(any("hello-notebook" in o for o in outs))

    def test_stdout_streams_live_not_buffered_until_cell_finishes(self):
        # Regression: RunNotebookCell is a server-streaming RPC, but the
        # original implementation ran the whole cell to completion and only
        # THEN emitted every chunk -- so a slow print loop showed nothing
        # until it finished. Prove the first stdout chunk actually arrives
        # well before the cell (and thus the whole generator) completes.
        code = "import time\nprint('first')\ntime.sleep(0.3)\nprint('second')"
        request = pb2.RunNotebookCellRequest(code=code, cell_id="c1")
        start = time.monotonic()
        first_stdout_at = None
        for chunk in self.service.RunNotebookCell(request, None):
            if chunk.WhichOneof("payload") == "stdout" and first_stdout_at is None:
                first_stdout_at = time.monotonic() - start
        total = time.monotonic() - start
        self.assertIsNotNone(first_stdout_at, "no stdout chunk was ever produced")
        self.assertGreater(total, 0.25, "cell finished suspiciously fast; sleep didn't run")
        self.assertLess(
            first_stdout_at, total - 0.15,
            f"first stdout chunk ({first_stdout_at:.3f}s) arrived too close to "
            f"the end ({total:.3f}s) -- looks buffered, not streamed live",
        )

    def test_interrupt_reports_false_when_nothing_running(self):
        resp = self.service.InterruptNotebookCell(pb2.InterruptNotebookCellRequest(), None)
        self.assertFalse(resp.ok)
        self.assertTrue(resp.error)

    def test_interrupt_stops_a_running_cell(self):
        code = "import time\nfor _ in range(50):\n    time.sleep(0.05)"
        results = {}

        def _run_in_background():
            results["chunks"] = _run(self.service, code)

        runner = threading.Thread(target=_run_in_background)
        runner.start()
        # Give the cell time to actually start (and the kernel lazily
        # construct/attach) before trying to interrupt it.
        deadline = time.monotonic() + 5.0
        while self.service._kernel is None and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.2)

        resp = self.service.InterruptNotebookCell(pb2.InterruptNotebookCellRequest(), None)
        runner.join(timeout=10)
        self.assertFalse(runner.is_alive(), "cell never stopped after interrupt")
        self.assertTrue(resp.ok, resp.error)

        chunks = results.get("chunks", [])
        tbs = [c.error_traceback for c in chunks if c.WhichOneof("payload") == "error_traceback"]
        self.assertTrue(any("KeyboardInterrupt" in t for t in tbs), f"no KeyboardInterrupt seen: {tbs}")
        done = [c for c in chunks if c.WhichOneof("payload") == "done"]
        self.assertTrue(done and not done[-1].done.ok)

    def test_state_persists_across_cells(self):
        _run(self.service, "x = 41")
        chunks = _run(self.service, "x + 1")
        texts = [c.result_text for c in chunks if c.WhichOneof("payload") == "result_text"]
        self.assertIn("42", "".join(texts))

    def test_nested_function_sees_module_level_globals(self):
        # Regression: a def'd-in-cell function must see names bound at the
        # top level of the SAME cell (e.g. `df`, `model`, `logger`) when it is
        # later called via something like df.apply(fn) -- not just the code
        # that defined it. A real embedded IPython kernel exec()s cells as
        # exec(code, user_global_ns, user_ns); if those are two different
        # dicts, top-level lookups (LOAD_NAME) still work but a nested
        # function's __globals__ (LOAD_GLOBAL) silently points at the wrong,
        # near-empty dict and raises NameError only once *called*.
        code = (
            "multiplier = 10\n"
            "def scale(x):\n"
            "    return x * multiplier\n"
            "[scale(1), scale(2)]"
        )
        chunks = _run(self.service, code)
        texts = [c.result_text for c in chunks if c.WhichOneof("payload") == "result_text"]
        done = [c for c in chunks if c.WhichOneof("payload") == "done"]
        tbs = [c.error_traceback for c in chunks if c.WhichOneof("payload") == "error_traceback"]
        self.assertTrue(done and done[-1].done.ok, f"cell errored: {''.join(tbs)}")
        self.assertTrue(any("[10, 20]" in t for t in texts))

    def test_apply_calls_nested_function_referencing_top_level_constant(self):
        # Regression for a real user report: a top-level constant (their
        # TRAIN_LOSS_SIGNAL/LAST_N_STEPS) referenced inside a helper function
        # that's invoked indirectly through pandas .apply() -- not called
        # directly -- to build a new metadata column from every row. This is
        # the exact shape that originally raised
        # "NameError: name 'TRAIN_LOSS_SIGNAL' is not defined" only once the
        # function was *called* by .apply(), not when it was defined.
        code = (
            "THRESHOLD = 0.15\n"
            "def flag_high_loss(loss_value):\n"
            "    return loss_value > THRESHOLD\n"
            "df['flag'] = df['loss'].apply(flag_high_loss)\n"
            "df['flag'].tolist()"
        )
        chunks = _run(self.service, code)
        texts = [c.result_text for c in chunks if c.WhichOneof("payload") == "result_text"]
        done = [c for c in chunks if c.WhichOneof("payload") == "done"]
        tbs = [c.error_traceback for c in chunks if c.WhichOneof("payload") == "error_traceback"]
        self.assertTrue(done and done[-1].done.ok, f"cell errored: {''.join(tbs)}")
        self.assertTrue(any("[False, True, True]" in t for t in texts))

    def test_matplotlib_figure_is_captured_as_png(self):
        code = "plt.figure(); plt.plot([0, 1, 2], [2, 1, 0]); plt.title('t')"
        chunks = _run(self.service, code)
        images = [c.image_png for c in chunks if c.WhichOneof("payload") == "image_png"]
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].startswith(b"\x89PNG"))

    def test_error_is_reported_not_raised(self):
        chunks = _run(self.service, "1 / 0")
        tbs = [c.error_traceback for c in chunks if c.WhichOneof("payload") == "error_traceback"]
        self.assertTrue(any("ZeroDivisionError" in t for t in tbs))
        done = [c for c in chunks if c.WhichOneof("payload") == "done"]
        self.assertTrue(done and not done[-1].done.ok)

    def test_write_inside_root_is_allowed_and_lands_in_root(self):
        _run(self.service, "open('inside.txt', 'w').write('ok')")
        self.assertTrue((self.root / "inside.txt").exists())
        self.assertEqual((self.root / "inside.txt").read_text(), "ok")

    def test_write_outside_root_is_rejected(self):
        outside = Path(tempfile.gettempdir()) / "wl_notebook_should_not_exist.txt"
        if outside.exists():
            outside.unlink()
        chunks = _run(self.service, f"open(r'{outside}', 'w').write('nope')")
        tbs = [c.error_traceback for c in chunks if c.WhichOneof("payload") == "error_traceback"]
        self.assertTrue(any("PermissionError" in t for t in tbs))
        self.assertFalse(outside.exists())


class TestNotebookKernelLegacy(_NotebookKernelContractTests, unittest.TestCase):
    def _make_service(self):
        return NotebookService(_fake_data_service(), root_log_dir=str(self.root))


@unittest.skipUnless(_IPYKERNEL_AVAILABLE, "ipykernel/jupyter_client not installed")
class TestNotebookKernelEmbedded(_NotebookKernelContractTests, unittest.TestCase):
    """Same contract, driven through EmbeddedKernelBridge -> a real ipykernel.

    The embedded kernel is a process-wide singleton by design (one real
    Jupyter kernel per process), so it is started once for this class and
    explicitly torn down afterwards to avoid leaking `_EMBED_ENABLED=True`
    into whatever test module runs next in the same process.
    """

    @classmethod
    def setUpClass(cls):
        notebook_service.configure_embedded_kernel(True)

    @classmethod
    def tearDownClass(cls):
        notebook_service.configure_embedded_kernel(False)

    def _make_service(self):
        return NotebookService(_fake_data_service(), root_log_dir=str(self.root))


class TestNotebookPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = NotebookService(_fake_data_service(), root_log_dir=str(self.root))

    def tearDown(self):
        self._tmp.cleanup()

    def test_get_notebook_writes_default_on_first_use(self):
        resp = self.service.GetNotebook(pb2.Empty(), None)
        self.assertFalse(resp.existed)
        self.assertTrue((self.root / "notebook.ipynb").exists())
        doc = json.loads(resp.ipynb_json)
        self.assertEqual(doc["nbformat"], 4)
        self.assertTrue(len(doc["cells"]) >= 1)

    def test_save_then_get_round_trips(self):
        self.service.GetNotebook(pb2.Empty(), None)  # create default
        payload = json.dumps({"cells": [{"cell_type": "code", "source": "df.head()",
                                         "metadata": {}, "execution_count": None, "outputs": []}],
                              "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
        save = self.service.SaveNotebook(pb2.SaveNotebookRequest(ipynb_json=payload), None)
        self.assertTrue(save.ok)
        again = self.service.GetNotebook(pb2.Empty(), None)
        self.assertTrue(again.existed)
        self.assertIn("df.head()", again.ipynb_json)

    def test_save_rejects_invalid_json(self):
        save = self.service.SaveNotebook(pb2.SaveNotebookRequest(ipynb_json="{not json"), None)
        self.assertFalse(save.ok)
        self.assertIn("invalid notebook JSON", save.error)

    @staticmethod
    def _payload(marker="x"):
        return json.dumps({"cells": [{"cell_type": "code", "source": marker,
                                      "metadata": {}, "execution_count": None, "outputs": []}],
                           "metadata": {}, "nbformat": 4, "nbformat_minor": 5})

    def test_get_reports_default_name(self):
        resp = self.service.GetNotebook(pb2.Empty(), None)
        self.assertEqual(resp.name, "notebook")

    def test_rename_moves_file_and_reports_name(self):
        self.service.GetNotebook(pb2.Empty(), None)  # creates notebook.ipynb
        save = self.service.SaveNotebook(
            pb2.SaveNotebookRequest(ipynb_json=self._payload("renamed"), name="experiment_a"), None)
        self.assertTrue(save.ok)
        self.assertEqual(save.name, "experiment_a")
        self.assertTrue((self.root / "experiment_a.ipynb").exists())
        self.assertFalse((self.root / "notebook.ipynb").exists())  # old file removed
        # The renamed notebook is now the active one.
        again = self.service.GetNotebook(pb2.Empty(), None)
        self.assertEqual(again.name, "experiment_a")
        self.assertIn("renamed", again.ipynb_json)

    def test_rename_collision_gets_indexed(self):
        self.service.GetNotebook(pb2.Empty(), None)  # active = notebook.ipynb
        # A different file already occupies the requested name.
        (self.root / "taken.ipynb").write_text("{}", encoding="utf-8")
        save = self.service.SaveNotebook(
            pb2.SaveNotebookRequest(ipynb_json=self._payload(), name="taken"), None)
        self.assertTrue(save.ok)
        self.assertEqual(save.name, "taken-1")
        self.assertTrue((self.root / "taken-1.ipynb").exists())
        self.assertTrue((self.root / "taken.ipynb").exists())  # existing file untouched

    def test_rename_sanitizes_unsafe_name(self):
        self.service.GetNotebook(pb2.Empty(), None)
        save = self.service.SaveNotebook(
            pb2.SaveNotebookRequest(ipynb_json=self._payload(), name="../evil/name.ipynb"), None)
        self.assertTrue(save.ok)
        self.assertEqual(save.name, "name")            # path stripped, .ipynb dropped
        self.assertEqual(Path(save.path).name, "name.ipynb")
        # Never escapes root_log_dir.
        self.assertTrue(Path(save.path).resolve().is_relative_to(self.root.resolve()))


class TestGenerateNotebookCode(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_delegates_to_agent(self):
        agent = MagicMock()
        agent.generate_code.return_value = ("df.describe()", "Summary stats.")
        service = NotebookService(_fake_data_service(agent=agent), root_log_dir=str(self.root))
        resp = service.GenerateNotebookCode(
            pb2.GenerateNotebookCodeRequest(prompt="summarize", context_code=""), None)
        self.assertTrue(resp.ok)
        self.assertEqual(resp.code, "df.describe()")
        self.assertEqual(resp.explanation, "Summary stats.")
        agent.generate_code.assert_called_once()

    def test_reports_error_when_no_agent(self):
        service = NotebookService(_fake_data_service(agent=None), root_log_dir=str(self.root))
        resp = service.GenerateNotebookCode(
            pb2.GenerateNotebookCodeRequest(prompt="x", context_code=""), None)
        self.assertFalse(resp.ok)
        self.assertIn("Agent backend is not running", resp.error)


if __name__ == "__main__":
    unittest.main()
