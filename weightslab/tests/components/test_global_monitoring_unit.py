import unittest
from unittest.mock import patch

from weightslab.components.global_monitoring import (
    Context,
    GuardContext,
    get_current_context,
    set_current_context,
)
from weightslab.components.tracking import TrackingMode


class _DummyModel:
    def __init__(self):
        self.training = True
        self.mode_calls = []
        self.train_calls = []
        self.eval_calls = 0

    def set_tracking_mode(self, mode):
        self.mode_calls.append(mode)

    def train(self, mode=True):
        self.training = bool(mode)
        self.train_calls.append(mode)

    def eval(self):
        self.eval_calls += 1
        self.training = False


class TestGlobalMonitoringUnit(unittest.TestCase):
    def test_contextvar_set_and_restore(self):
        token = set_current_context(Context.TRAINING)
        self.assertEqual(get_current_context(), Context.TRAINING)

        from weightslab.components import global_monitoring as gm
        gm._current_context.reset(token)
        self.assertIn(get_current_context(), {Context.UNKNOWN, Context.TESTING, Context.TRAINING})

    def test_guard_context_training_non_audit(self):
        model = _DummyModel()
        gc = GuardContext(for_training=True)
        gc.model = model

        with patch("weightslab.components.global_monitoring.pause_controller.wait_if_paused"), \
             patch("weightslab.components.global_monitoring.resolve_hp_name", return_value=None), \
             patch("weightslab.components.global_monitoring.get_hyperparams", return_value={}):
            gc.__enter__()
            self.assertEqual(get_current_context(), Context.TRAINING)
            self.assertIn(TrackingMode.TRAIN, model.mode_calls)
            self.assertIn(True, model.train_calls)
            result = gc.__exit__(None, None, None)

        self.assertFalse(result)

    def test_guard_context_training_audit_uses_eval(self):
        model = _DummyModel()
        gc = GuardContext(for_training=True)
        gc.model = model

        with patch("weightslab.components.global_monitoring.pause_controller.wait_if_paused"), \
             patch("weightslab.components.global_monitoring.resolve_hp_name", return_value="hp"), \
             patch("weightslab.components.global_monitoring.get_hyperparams", return_value={"auditorMode": True}):
            gc.__enter__()
            self.assertIn(TrackingMode.TRAIN, model.mode_calls)
            self.assertEqual(model.eval_calls, 1)
            gc.__exit__(None, None, None)

    def test_guard_context_suppresses_runtime_error(self):
        gc = GuardContext(for_training=False)
        with patch("weightslab.components.global_monitoring.pause_controller.wait_if_paused"):
            gc.__enter__()
            suppressed = gc.__exit__(RuntimeError, RuntimeError("x"), None)
        self.assertTrue(suppressed)


if __name__ == "__main__":
    unittest.main()
