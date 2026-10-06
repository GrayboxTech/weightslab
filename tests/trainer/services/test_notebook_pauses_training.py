"""Running a notebook cell pauses training first.

A cell runs in the training process, against the live model and dataframe. If
the loop kept stepping underneath it, a cell computing features or a t-SNE
would see a model that changes mid-cell.
"""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.backend import ledgers
from weightslab.components.global_monitoring import pause_controller, weightslab_rlock
from weightslab.trainer.services import notebook_service
from weightslab.trainer.services.notebook_service import (
    PAUSED_FOR_CELL_NOTE,
    NotebookService,
    pause_training_for_cell,
)


def _fake_data_service():
    df = pd.DataFrame({"origin": ["train", "test"], "loss": [0.1, 0.9]})
    return SimpleNamespace(_all_datasets_df=df, _pull_into_all_data_view_df=lambda: df,
                           _root_log_dir=None, audit_logger=None, _agent=None)


class _TrainingState(unittest.TestCase):
    def setUp(self):
        ledgers.register_hyperparams({"is_training": False, "pause_at_step": 0})
        self._was_paused = pause_controller.is_paused()

    def tearDown(self):
        # Leave the process-wide controller as found.
        if self._was_paused:
            pause_controller._event.clear()
        else:
            pause_controller._event.set()
        ledgers.clear_all()

    @staticmethod
    def _start_training():
        pause_controller._event.set()          # what Play does, minus the hash dump


class TestPauseBeforeTheCell(_TrainingState):

    def test_a_running_training_is_paused_and_stays_paused(self):
        self._start_training()
        note = pause_training_for_cell()
        self.assertEqual(note, PAUSED_FOR_CELL_NOTE)
        self.assertTrue(pause_controller.is_paused())
        # The UI reads the same flag the header's pause button sets.
        self.assertFalse(ledgers.get_hyperparams().get("is_training"))

    def test_already_paused_says_nothing(self):
        pause_controller._event.clear()
        self.assertIsNone(pause_training_for_cell())

    def test_the_cell_waits_for_the_step_in_flight(self):
        """pause() stops the NEXT step; the one holding the training lock
        must finish before the cell reads the model."""
        self._start_training()
        released = {}

        def step_in_flight():
            with weightslab_rlock:
                held.set()
                time.sleep(0.4)
                released["at"] = time.perf_counter()

        held = threading.Event()
        worker = threading.Thread(target=step_in_flight)
        worker.start()
        held.wait(2)
        pause_training_for_cell()
        returned_at = time.perf_counter()
        worker.join()
        self.assertGreaterEqual(returned_at, released["at"])

    def test_a_step_longer_than_the_wait_is_reported_not_blocking_forever(self):
        self._start_training()
        held, done = threading.Event(), threading.Event()

        def long_step():
            with weightslab_rlock:
                held.set()
                done.wait(5)

        worker = threading.Thread(target=long_step)
        worker.start()
        held.wait(2)
        try:
            note = pause_training_for_cell(timeout_s=0.2)
        finally:
            done.set()
            worker.join()
        self.assertIn("had not finished", note)
        self.assertTrue(pause_controller.is_paused())


class TestRunNotebookCellPauses(_TrainingState):

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        notebook_service.configure_embedded_kernel(False)
        self.service = NotebookService(_fake_data_service(), root_log_dir=str(Path(self._tmp.name)))

    def tearDown(self):
        super().tearDown()
        self._tmp.cleanup()

    def _run(self, code):
        return list(self.service.RunNotebookCell(
            pb2.RunNotebookCellRequest(code=code, cell_id="c1"), None))

    def test_the_cell_output_says_training_was_paused(self):
        self._start_training()
        chunks = self._run("1 + 1")
        first = chunks[0]
        self.assertEqual(first.WhichOneof("payload"), "stdout")
        self.assertIn("Training paused", first.stdout)
        self.assertTrue(pause_controller.is_paused())
        self.assertTrue(chunks[-1].done.ok)

    def test_the_cell_sees_training_already_paused(self):
        self._start_training()
        chunks = self._run(
            "from weightslab.components.global_monitoring import pause_controller\n"
            "pause_controller.is_paused()")
        results = [c.result_text for c in chunks if c.WhichOneof("payload") == "result_text"]
        self.assertEqual(results, ["True"])

    def test_no_note_when_nothing_was_running(self):
        pause_controller._event.clear()
        chunks = self._run("1 + 1")
        self.assertFalse(any(c.WhichOneof("payload") == "stdout"
                             and "Training paused" in c.stdout for c in chunks))

    def test_only_the_first_cell_announces_it(self):
        self._start_training()
        self._run("a = 1")
        chunks = self._run("a + 1")
        self.assertFalse(any(c.WhichOneof("payload") == "stdout" for c in chunks))


if __name__ == "__main__":
    unittest.main()
