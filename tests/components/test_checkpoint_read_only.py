"""CheckpointManager(read_only=True): a viewer over the experiments found under
one folder. The most recent run is the current experiment (no new hash), every
run is listed and can be loaded with its last per-sample state, and nothing is
ever written into any run's folder."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch.nn as nn
import yaml

from weightslab.backend import ledgers
from weightslab.backend.logger import LoggerQueue
from weightslab.components import checkpoint_manager as cm_module
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.h5_dataframe_store import H5DataFrameStore


def _fingerprint(folder: Path) -> dict:
    """Every file under folder with its size and modification time, and every folder."""
    out = {}
    for p in sorted(folder.rglob("*")):
        st = p.stat()
        out[str(p.relative_to(folder))] = None if p.is_dir() else (st.st_size, st.st_mtime_ns)
    return out


class CheckpointReadOnlyViewerTests(unittest.TestCase):

    def setUp(self):
        ledgers.clear_all()
        self._hooks = list(cm_module._EXPERIMENT_LOAD_HOOKS)
        self._tmp_ctx = tempfile.TemporaryDirectory(prefix="checkpoint_read_only_")
        self.parent = Path(self._tmp_ctx.name) / "models"
        self.root_a, self.root_b = self.parent / "net-f0", self.parent / "net-f1"
        self.hash_a = "aaaa0101" + "00000000" + "cccc0101"
        self.hash_b = "aaaa0102" + "00000000" + "cccc0102"
        self._write_run(self.root_a, self.hash_a, "2024-01-01T00:00:00", loss=[0.9, 0.8], name="net-f0")
        self._write_run(self.root_b, self.hash_b, "2024-06-01T00:00:00", loss=[0.5, 0.4], name=None)
        # Run A kept its live data store; run B was copied back with only its export.
        H5DataFrameStore(self.root_a / "checkpoints" / "data" / "data.h5").upsert("fold0_train", pd.DataFrame({
            "sample_id": [0, 1, 2], "annotation_id": [0, 0, 0],
            "signals//train/photo_loss": [0.11, 0.12, 0.13], "nb_seen": [5, 5, 5],
        }).set_index(["sample_id", "annotation_id"]))
        pd.DataFrame({
            "sample_id": [0, 1, 2], "annotation_id": [0, 0, 0], "origin": "fold1_val",
            "signals//val/photo_pq": [0.7, 0.8, 0.9], "tag:fold1": ["val", "val", "train"],
        }).to_parquet(self.root_b / "d8e01c87_dataframe.parquet")
        self.before = {r: _fingerprint(r) for r in (self.root_a, self.root_b)}

    def tearDown(self):
        cm_module._EXPERIMENT_LOAD_HOOKS[:] = self._hooks
        ledgers.clear_all()
        self._tmp_ctx.cleanup()

    def _write_run(self, root, exp_hash, timestamp, loss, name):
        manager = CheckpointManager(root_log_dir=str(root), load_model=False, load_config=False, load_data=False)
        manager.current_exp_hash = exp_hash
        manager._create_exp_hash_directories(exp_hash)
        lg = LoggerQueue(register=False, db_path=str(manager.get_logger_db_path()))
        lg.chkpt_manager = manager
        for i, value in enumerate(loss):
            lg.add_scalars("train/loss", {"train/loss": value}, global_step=i, signal_per_sample={},
                           aggregate_by_step=False)
        lg.flush_to_disk()
        lg.stop_background_flush()
        lg._conn.close()
        manifest_path = root / "checkpoints" / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["last_updated"] = timestamp
        info = manifest["experiments"][exp_hash]
        info["last_used"] = info["created"] = timestamp
        info["experiment_name"] = name
        manifest_path.write_text(yaml.dump(manifest, sort_keys=False))
        ledgers.clear_all()

    def _viewer(self):
        manager = CheckpointManager(root_log_dir=str(self.parent), read_only=True)
        ledgers.register_checkpoint_manager(manager)
        return manager

    def _assert_runs_untouched(self):
        for root, before in self.before.items():
            self.assertEqual(_fingerprint(root), before, f"{root.name} must not be written to")

    def test_most_recent_run_is_current_and_no_new_experiment_is_made(self):
        manager = self._viewer()
        self.assertEqual(manager.root_log_dir, self.root_b.absolute())
        self.assertEqual(manager.current_exp_hash, self.hash_b)
        self.assertEqual(manager.get_current_experiment_hash(), self.hash_b)

        new_hash, is_new, changed = manager.update_experiment_hash(hp_snapshot={"lr": 1.0}, force=True,
                                                                   dump_immediately=True)
        self.assertEqual((new_hash, is_new, changed), (self.hash_b, False, set()))
        self.assertFalse(manager.save_pending_changes(force=True))
        manager.save_config({"lr": 1.0})
        manager.save_data_snapshot(force_new_state=True)
        manager._save_manager_state()
        self._assert_runs_untouched()

        self.assertEqual(manager.workspace_dir, (self.parent / ".weightslab_viewer").absolute())
        again = CheckpointManager(root_log_dir=str(self.parent), read_only=True)
        self.assertEqual(len(again.discovered_root_dirs), 2, "the workspace is never taken for a run")

    def test_every_root_is_listed(self):
        runs = {r["hash"]: r for r in self._viewer().list_runs()}
        self.assertEqual(set(runs), {self.hash_a, self.hash_b})
        self.assertEqual(runs[self.hash_a]["root"], str(self.root_a.absolute()))
        self.assertTrue(runs[self.hash_b]["is_current"])
        self.assertFalse(runs[self.hash_a]["is_current"])
        self.assertEqual(runs[self.hash_a]["experiment_name"], "net-f0")
        self.assertEqual(runs[self.hash_b]["experiment_name"], "net-f1", "an unnamed run shows its folder name")
        self._assert_runs_untouched()

    def test_logger_stays_in_memory_with_every_runs_curves(self):
        self._viewer()
        lg = LoggerQueue(register=True)
        try:
            self.assertIn(lg._db_path, (None, ":memory:"), "a viewer never opens a run's database")
            curves = lg.get_signal_history().get("train/loss", {})
            self.assertEqual(set(curves), {self.hash_a, self.hash_b})
        finally:
            lg.stop_background_flush()
        self._assert_runs_untouched()

    def test_loading_a_run_brings_its_last_per_sample_state_and_runs_the_hooks(self):
        manager = self._viewer()
        dfm = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
        dfm.upsert_df(pd.DataFrame({"sample_id": [0, 1, 2], "annotation_id": [0, 0, 0], "origin": "photos",
                                    "discarded": False, "signals//pooled": [1.0, 2.0, 3.0]})
                      .set_index(["sample_id", "annotation_id"]), origin="photos")
        ledgers.register_dataframe(dfm)
        loaded = []
        cm_module.register_experiment_load_hook(lambda h, root: loaded.append((h, root)))
        lg = LoggerQueue(register=True)
        try:
            self.assertTrue(manager.load_experiment(), "the current (most recent) run")
            df = dfm.get_df_view()
            self.assertAlmostEqual(df.loc[("2", 0), "signals//val/photo_pq"], 0.9)
            self.assertEqual(str(df.loc[("2", 0), "tag:fold1"]), "train")
            self.assertEqual(loaded[-1], (self.hash_b, self.root_b.absolute()))

            self.assertTrue(manager.load_experiment(self.hash_a))
            df = dfm.get_df_view()
            self.assertEqual(manager.current_exp_hash, self.hash_a)
            self.assertAlmostEqual(df.loc[("1", 0), "signals//train/photo_loss"], 0.12)
            self.assertEqual(int(df.loc[("1", 0), "nb_seen"]), 5)
            self.assertNotIn("signals//val/photo_pq", df.columns, "run B's columns leave with run B")
            self.assertAlmostEqual(df.loc[("1", 0), "signals//pooled"], 2.0, "the viewer's own column stays")
            self.assertEqual(set(df["origin"].astype(str)), {"photos"})
            self.assertEqual(loaded[-1], (self.hash_a, self.root_a.absolute()))
        finally:
            lg.stop_background_flush()
        self._assert_runs_untouched()

    def test_a_run_with_only_its_per_sample_state_is_loadable(self):
        """No weights, no data snapshot: the live store, else the export, is the state."""
        manager = self._viewer()
        live = manager.load_checkpoint(self.hash_a, load_model=False, load_weights=False)
        self.assertEqual(live["loaded_components"], {"data_store"})
        self.assertEqual(live["data_store"], (self.root_a / "checkpoints" / "data" / "data.h5").absolute())
        export = manager.load_checkpoint(self.hash_b, load_model=False, load_weights=False)
        self.assertEqual(export["data_store"], (self.root_b / "d8e01c87_dataframe.parquet").absolute())
        self.assertIsNone(export["data_arrays"])

    def test_a_viewer_with_a_model_still_restores_weights(self):
        """Skipping weights is only for a viewer that wraps no model."""
        writer = CheckpointManager(root_log_dir=str(self.root_a), load_model=False, load_config=False,
                                   load_data=False)
        writer.current_exp_hash = self.hash_a
        model = nn.Sequential(nn.Linear(2, 2))
        writer.save_model_checkpoint(model=model, step=7, save_optimizer=False)
        ledgers.clear_all()
        self.before = {r: _fingerprint(r) for r in (self.root_a, self.root_b)}

        manager = self._viewer()
        target = nn.Sequential(nn.Linear(2, 2))
        ledgers.register_model(target)
        lg = LoggerQueue(register=True)
        try:
            self.assertTrue(manager.load_state(self.hash_a, load_model=False, load_logger=False))
        finally:
            lg.stop_background_flush()
        for name, tensor in model.state_dict().items():
            self.assertTrue((target.state_dict()[name] == tensor).all(), f"{name} restored")
        self._assert_runs_untouched()


if __name__ == "__main__":
    unittest.main()
