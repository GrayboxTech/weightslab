import unittest
from unittest.mock import patch

from weightslab.components.global_monitoring import (
    Context,
    GuardContext,
    get_current_context,
    set_current_context,
)


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

    # def test_guard_context_training_non_audit(self):
    # model = _DummyModel()
    # gc = GuardContext(for_training=True)
    # gc.model = model

    # with patch("weightslab.components.global_monitoring.pause_controller.wait_if_paused"), \
    # patch("weightslab.components.global_monitoring.resolve_hp_name", return_value=None), \
    # patch("weightslab.components.global_monitoring.get_hyperparams", return_value={}):
    # gc.__enter__()
    # self.assertEqual(get_current_context(), Context.TRAINING)
    # self.assertIn(True, model.train_calls)
    # result = gc.__exit__(None, None, None)

    # self.assertFalse(result)

    # def test_guard_context_training_audit_uses_eval(self):
    # model = _DummyModel()
    # gc = GuardContext(for_training=True)
    # gc.model = model

    # with patch("weightslab.components.global_monitoring.pause_controller.wait_if_paused"), \
    # patch("weightslab.components.global_monitoring.resolve_hp_name", return_value="hp"), \
    # patch("weightslab.components.global_monitoring.get_hyperparams", return_value={"auditorMode": True}):
    # gc.__enter__()
    # self.assertEqual(model.eval_calls, 1)
    # gc.__exit__(None, None, None)

    def test_guard_context_suppresses_runtime_error(self):
        gc = GuardContext(for_training=False)
        with patch("weightslab.components.global_monitoring.pause_controller.wait_if_paused"):
            gc.__enter__()
            suppressed = gc.__exit__(RuntimeError, RuntimeError("x"), None)
        self.assertTrue(suppressed)


class TestPauseControllerCheckpointManager(unittest.TestCase):
    """resume() dumps through the manager registered NOW, not the first one.

    The controller is a process-wide singleton. It used to keep the manager it
    first saw, so after ledgers.clear_all() and a new experiment it went on
    writing the new run's checkpoints into the previous run's manifest, and a
    reload by the current hash found nothing.
    """

    def tearDown(self):
        from weightslab.backend import ledgers
        ledgers.clear_all()

    def test_resume_follows_the_manager_registered_after_clear_all(self):
        from unittest.mock import MagicMock
        from weightslab.backend import ledgers
        from weightslab.components.global_monitoring import PauseController

        first, second = MagicMock(name="first"), MagicMock(name="second")
        first.hash_by_module = second.hash_by_module = ["aaaa", "bbbb", "cccc"]
        controller = PauseController()
        with patch("weightslab.components.global_monitoring.set_hyperparam"):
            ledgers.register_checkpoint_manager(first)
            controller.resume(force=True)
            first.save_pending_changes.assert_called()
            first.reset_mock()

            ledgers.clear_all()
            ledgers.register_checkpoint_manager(second)
            controller.resume(force=True)

        # Everything after the switch goes to the new run's manager, nothing to
        # the old one (call counts are not pinned: other code may use it too).
        second.update_experiment_hash.assert_called_with(first_time=True)
        second.save_pending_changes.assert_called()
        first.update_experiment_hash.assert_not_called()
        first.save_pending_changes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
