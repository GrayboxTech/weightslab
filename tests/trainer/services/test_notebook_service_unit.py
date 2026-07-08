"""Unit tests for the shared-kernel notebook service.

These exercise the real NotebookKernel (no LLM required) plus the persistence and
code-generation gRPC surfaces with a lightweight fake DataService, mirroring the
style of test_agent_service_unit.py.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.services.notebook_service import NotebookService


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


class TestNotebookKernel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.service = NotebookService(_fake_data_service(), root_log_dir=str(self.root))

    def tearDown(self):
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

    def test_state_persists_across_cells(self):
        _run(self.service, "x = 41")
        chunks = _run(self.service, "x + 1")
        texts = [c.result_text for c in chunks if c.WhichOneof("payload") == "result_text"]
        self.assertIn("42", "".join(texts))

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
