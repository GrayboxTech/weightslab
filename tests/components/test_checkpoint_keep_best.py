"""``checkpoint_manager.keep_best``: keep only a run's k best weight checkpoints.

Ranked by one logged signal (validation accuracy, say), rotated as training
dumps new ones. Only the trainer's own periodic dumps rotate; a save made on
purpose and the run's most recent checkpoint are never deleted.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch.nn as nn

from weightslab.backend.logger import LoggerQueue
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.components.checkpoint_retention import (
    KeepBestPolicy,
    score_checkpoints,
    steps_to_prune,
)


class TestPolicyConfig(unittest.TestCase):
    def test_unset_means_no_policy(self):
        self.assertIsNone(KeepBestPolicy.from_config({}))
        self.assertIsNone(KeepBestPolicy.from_config(None))
        self.assertIsNone(KeepBestPolicy.from_config({"keep_best": None}))

    def test_defaults(self):
        policy = KeepBestPolicy.from_config({"keep_best": {"signal": "val/accuracy"}})
        self.assertEqual((policy.signal, policy.k, policy.mode), ("val/accuracy", 5, "max"))
        # A loss is better low, unless told otherwise.
        self.assertEqual(KeepBestPolicy.from_config({"keep_best": {"signal": "val-loss"}}).mode, "min")
        self.assertEqual(KeepBestPolicy.from_config(
            {"keep_best": {"signal": "val-loss", "mode": "MAX", "k": "3"}}),
            KeepBestPolicy("val-loss", 3, "max"))

    def test_a_malformed_policy_is_an_error_not_a_silent_no_op(self):
        for bad in ({"k": 3}, {"signal": "acc", "k": 0}, {"signal": "acc", "k": "many"},
                    {"signal": "acc", "mode": "best"}, 5):
            with self.assertRaises(ValueError, msg=bad):
                KeepBestPolicy.from_config({"keep_best": bad})


class TestRanking(unittest.TestCase):
    def test_each_checkpoint_takes_the_nearest_evaluation(self):
        # Dumps at 100/200/300, evaluations one step later (the usual loop).
        history = [(101, 0.5), (201, 0.7), (301, 0.6)]
        scores = score_checkpoints([100, 200, 300], history)
        self.assertEqual(scores, {100: (0.5, 1), 200: (0.7, 1), 300: (0.6, 1)})

    def test_a_checkpoint_whose_evaluation_has_not_landed_is_not_ranked(self):
        # The dump at 300 fired before the evaluation of step 300 was written.
        scores = score_checkpoints([100, 200, 300], [(100, 0.5), (200, 0.7)])
        self.assertNotIn(300, scores)
        self.assertEqual(scores[200], (0.7, 0))

    def test_non_finite_values_are_ignored(self):
        scores = score_checkpoints([100], [(100, float("nan")), (110, 0.4)])
        self.assertEqual(scores, {100: (0.4, 10)})

    def test_keeps_the_k_best_and_never_the_resume_point(self):
        history = [(s, v) for s, v in zip(range(100, 700, 100), [0.5, 0.9, 0.6, 0.8, 0.7, 0.1])]
        policy = KeepBestPolicy("acc", 3, "max")
        # 600 is the worst but it is the latest checkpoint: kept for resume.
        self.assertEqual(steps_to_prune(policy, range(100, 700, 100), 600, history), [100, 300])

    def test_min_mode(self):
        history = [(100, 0.5), (200, 0.9), (300, 0.2)]
        policy = KeepBestPolicy("loss", 1, "min")
        self.assertEqual(steps_to_prune(policy, [100, 200, 300], 400, history), [100, 200])

    def test_ties_keep_the_closer_then_the_later_checkpoint(self):
        # One evaluation at 205 scores both 200 and 210; 210 is closer.
        policy = KeepBestPolicy("acc", 1, "max")
        self.assertEqual(steps_to_prune(policy, [200, 210], 999, [(205, 0.5), (300, 0.1)]), [200])


EXP = "aaaaaaaabbbbbbbbcccccccc"


class TestManagerRotation(unittest.TestCase):
    """End to end on a real CheckpointManager and signal history."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cm = CheckpointManager(root_log_dir=tmp.name)
        self.cm.current_exp_hash = EXP
        self.cm._write_manifest({"experiments": {EXP: {}}, "latest_hash": EXP})
        self.cm.config = {"checkpoint_manager": {"keep_best": {"signal": "val-acc", "k": 2}}}
        self.model = nn.Linear(2, 2)

        self.history = LoggerQueue(register=False)
        patcher = patch("weightslab.components.checkpoint_manager.ledgers")
        ledgers = patcher.start()
        self.addCleanup(patcher.stop)
        ledgers.get_logger.return_value = self.history
        ledgers.get_dataframe.side_effect = RuntimeError("no dataframe in this test")

    def _evaluate(self, step, accuracy):
        self.history._append_history_entry("val-acc", EXP, step, accuracy)

    def _dump(self, step, rotate=True):
        path = self.cm.save_model_checkpoint(model=self.model, save_optimizer=False, step=step, rotate=rotate)
        self.assertIsNotNone(path)
        return path

    def _steps_on_disk(self):
        model_dir = self.cm.models_dir / EXP[8:-8]
        return sorted(int(p.stem.rsplit("_", 1)[1]) for p in model_dir.glob(f"{EXP}_step_*.pt"))

    def test_rotates_down_to_the_k_best_plus_the_latest(self):
        for step, accuracy in [(100, 0.50), (200, 0.90), (300, 0.60), (400, 0.80), (500, 0.70)]:
            self._dump(step)
            self._evaluate(step, accuracy)   # lands after the dump, as in training
        self._dump(600)                      # its own evaluation is still to come

        # 200 and 400 are the two best; 600 is the resume point.
        self.assertEqual(self._steps_on_disk(), [200, 400, 600])
        entry = self.cm._load_manifest()["experiments"][EXP]
        self.assertEqual(sorted(entry["rotating_weight_checkpoints"]),
                         sorted(f"{EXP}_step_{s:06d}.pt" for s in (200, 400, 600)))

    def test_a_save_on_purpose_is_never_rotated(self):
        self._dump(100, rotate=False)        # the Studio's save button
        self._evaluate(100, 0.10)            # and a bad score
        for step, accuracy in [(200, 0.9), (300, 0.8), (400, 0.7)]:
            self._dump(step)
            self._evaluate(step, accuracy)
        self._dump(500)

        self.assertEqual(self._steps_on_disk(), [100, 200, 300, 500])

    def test_without_values_for_the_signal_nothing_is_deleted(self):
        self.cm.config = {"checkpoint_manager": {"keep_best": {"signal": "val-typo", "k": 1}}}
        for step, accuracy in [(100, 0.5), (200, 0.9), (300, 0.6)]:
            self._dump(step)
            self._evaluate(step, accuracy)
        self.assertEqual(self._steps_on_disk(), [100, 200, 300])

    def test_no_policy_keeps_everything(self):
        self.cm.config = {"checkpoint_manager": {}}
        for step, accuracy in [(100, 0.5), (200, 0.9), (300, 0.6), (400, 0.1)]:
            self._dump(step)
            self._evaluate(step, accuracy)
        self.assertEqual(self._steps_on_disk(), [100, 200, 300, 400])

    def test_a_projection_sidecar_goes_with_its_checkpoint(self):
        path = self._dump(100)
        sidecar = CheckpointManager.projection_sidecar_path(path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(b"encoder")
        self._evaluate(100, 0.1)
        for step, accuracy in [(200, 0.9), (300, 0.8)]:
            self._dump(step)
            self._evaluate(step, accuracy)
        self._dump(400)

        self.assertNotIn(100, self._steps_on_disk())
        self.assertFalse(Path(sidecar).exists())


if __name__ == "__main__":
    unittest.main()
