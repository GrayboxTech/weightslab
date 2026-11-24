"""Restoring a checkpoint whose weights no longer fit the model.

The Modeling tab can add/prune neurons, after which an older checkpoint's
weights do not fit the live model. The restore then reloads the checkpoint's
architecture -- and when none was saved, it used to register ``None`` as the
model: the restore reported success and the next training forward died with
"Proxy target not set". It must fail instead, leaving the live model alone.
"""
import tempfile
import unittest
from unittest import mock

import torch as th
import torch.nn as nn

from weightslab.backend import ledgers
from weightslab.components.checkpoint_manager import CheckpointManager


class TestRestoreAcrossArchitecture(unittest.TestCase):
    def tearDown(self):
        ledgers.clear_all()

    def _manager(self):
        return CheckpointManager(root_log_dir=tempfile.mkdtemp(), load_model=False,
                                 load_config=False, load_data=False)

    def test_missing_architecture_keeps_the_live_model(self):
        live = nn.Linear(4, 5)                     # e.g. after "add 2 neurons"
        ledgers.register_model(live)
        manager = self._manager()
        old_weights = {"model_state_dict": nn.Linear(4, 3).state_dict(), "step": 45}
        calls = []

        def fake_load_checkpoint(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:   # the requested restore: weights only
                return {"loaded_components": {"weights"}, "weights": old_weights, "model": None}
            # the forced "full" reload: no architecture file on disk
            return {"loaded_components": {"weights"}, "weights": old_weights, "model": None}

        with mock.patch.object(manager, "load_checkpoint", side_effect=fake_load_checkpoint):
            ok = manager.load_state("a" * 24, load_logger=False, load_data=False)

        self.assertFalse(ok)
        self.assertEqual(len(calls), 2)            # it did try the full reload
        model = ledgers.get_model()
        self.assertIs(object.__getattribute__(model, "_obj"), live)
        self.assertEqual(tuple(model(th.randn(2, 4)).shape), (2, 5))   # still callable
        self.assertIn("weights", manager.error_loading_checkpoint)

    def test_architecture_restore_rebinds_the_optimizer(self):
        restored = nn.Linear(4, 3)
        manager = self._manager()
        rebinds = []
        restored.update_optimizer = lambda: rebinds.append(True)
        data = {"loaded_components": {"model"}, "model": restored, "weights": None}
        # No logger is registered in this bare test; its bookkeeping is not the point.
        with mock.patch.object(manager, "load_checkpoint", return_value=data), \
                mock.patch.object(manager, "get_logger_length", return_value=0):
            manager.load_state("b" * 24, load_logger=False, load_data=False)
        self.assertIs(object.__getattribute__(ledgers.get_model(), "_obj"), restored)
        self.assertEqual(rebinds, [True])

    def test_architecture_dump_is_requested_by_structural_edits(self):
        manager = self._manager()
        self.assertFalse(manager._architecture_dump_requested)
        manager.request_architecture_dump()
        self.assertTrue(manager._architecture_dump_requested)

    # A Modeling add/prune writes nothing: both architectures are saved when
    # training resumes; until then the pre-edit one is kept in memory.
    OLD = "h" * 8 + "m" * 8 + "d" * 8
    NEW = "h" * 8 + "n" * 8 + "d" * 8

    def test_structural_edit_writes_nothing_until_the_changes_are_persisted(self):
        import dill
        manager = self._manager()
        manager.current_exp_hash = self.OLD
        manager.note_structural_edit(nn.Linear(4, 3))
        self.assertTrue(manager._architecture_dump_requested)
        self.assertFalse(manager._architecture_file("m" * 8).exists())
        self.assertFalse(manager.models_dir.exists() and any(manager.models_dir.rglob("*.pkl")))
        # A second edit before resuming keeps the FIRST (pre-edit) architecture.
        manager.note_structural_edit(nn.Linear(4, 5))
        self.assertEqual(dill.loads(manager._pending_previous_architecture[1]).out_features, 3)

        # Resume: the new hash is computed and the pending changes persisted.
        manager.current_exp_hash = self.NEW
        with mock.patch.object(manager, "flush_logger_to_disk"):
            manager._save_changes(model=nn.Linear(4, 5), config={}, changed_components={"model"})
        self.assertEqual(dill.loads(manager._architecture_file("m" * 8).read_bytes()).out_features, 3)
        self.assertEqual(dill.loads(manager._architecture_file("n" * 8).read_bytes()).out_features, 5)
        self.assertIsNone(manager._pending_previous_architecture)
        self.assertFalse(manager._architecture_dump_requested)

    def test_restore_before_resuming_uses_the_in_memory_architecture(self):
        manager = self._manager()
        manager.current_exp_hash = self.OLD
        manager.note_structural_edit(nn.Linear(4, 3))
        manifest = {"experiments": {self.OLD: {"hp_hash": "h" * 8, "model_hash": "m" * 8, "data_hash": "d" * 8}}}
        # The recorded model hash is still the pre-edit one (not resumed): the
        # architecture must be reloaded all the same.
        stale = {"hp": "h" * 8, "model": "m" * 8, "data": "d" * 8}
        with mock.patch.object(manager, "_locate_experiment", return_value=(manager.checkpoints_dir, manifest)),                 mock.patch.object(manager.hash_generator, "get_component_hashes", return_value=stale):
            result = manager.load_checkpoint(exp_hash=self.OLD, load_model=True, load_weights=False,
                                             load_config=False, load_data=False)
        self.assertIn("model", result["loaded_components"])
        self.assertEqual(result["model"].out_features, 3)
        self.assertFalse(manager._architecture_file("m" * 8).exists())   # still nothing on disk

    def test_proxy_is_unwrapped_before_serializing(self):
        import dill
        model = nn.Linear(4, 3)
        blob = CheckpointManager._serialize_architecture(ledgers.Proxy(model))
        self.assertIsInstance(dill.loads(blob), nn.Linear)


if __name__ == "__main__":
    unittest.main()
