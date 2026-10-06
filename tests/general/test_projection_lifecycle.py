"""The projection's life around the model: wrapping, opting out, failing,
checkpoints and restarts, and projections the user computes themselves.

test_parametric_umap.py covers the algorithm and the serving contract; this
covers everything that happens to the projection BECAUSE of something done to
the model or the run.
"""

import copy
import os
import shutil
import tempfile
import unittest
from unittest import mock

import dill
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn

import weightslab as wl
from weightslab import src as wl_src
from weightslab.backend import ledgers
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.projection import (
    ENV_ENABLED,
    ProjectionTracker,
    attach_projection,
    clear_pending_restore,
    detach_projection,
    get_tracker,
    observe_batch,
    reattach_projection,
    save_projection,
    save_projection_coords,
)
from weightslab.projection import parametric_umap as pu
from weightslab.projection.registry import (
    REGISTRY_FILE,
    clear_registry,
    known_prefixes,
    register_prefix,
)
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.services import projection_service as ps

LOGGER = "weightslab.projection.parametric_umap"


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Linear(16, 8)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        return self.classifier(self.act(self.feat(x)))


class _Stub:
    """Stands in for the dataframe manager; records what would be written."""

    def __init__(self, columns=()):
        self.calls = []
        self._columns = list(columns)

    def enqueue_batch(self, **kwargs):
        self.calls.append(kwargs)

    def get_df_view(self, limit=-1, **_):
        return pd.DataFrame(columns=self._columns)

    def written(self, prefix):
        return {k for c in self.calls for k in (c["losses"] or {})
                if k.startswith(f"signals//{prefix}_")}


def projection_hooks(module):
    """The live projection hooks installed on *module*."""
    return [h for h in module._forward_hooks.values() if isinstance(h, pu._FeatureHook)]


def encoder_snapshot():
    return {k: v.detach().clone() for k, v in get_tracker()._encoder.state_dict().items()}


def same_weights(a, b):
    return all(th.equal(a[k], b[k]) for k in a)


def fit(model, steps, start=1, batch=32):
    """Run *steps* tracked training steps through the hook + observe path."""
    for step in range(start, start + steps):
        model.train()
        model(th.randn(batch, 16))
        observe_batch(list(range(batch)), step)


class _StubbedFrame(unittest.TestCase):
    def setUp(self):
        self._saved_frame = wl_src.DATAFRAME_M
        self.stub = _Stub()
        wl_src.DATAFRAME_M = self.stub
        detach_projection()
        clear_pending_restore()

    def tearDown(self):
        wl_src.DATAFRAME_M = self._saved_frame
        detach_projection()
        clear_pending_restore()


class _LedgerRun(_StubbedFrame):
    """A real ledger + checkpoint manager in a throwaway root_log_dir."""

    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp(prefix="wl-projection-")
        self._saved_env = os.environ.get(ENV_ENABLED)
        os.environ.pop(ENV_ENABLED, None)

    def tearDown(self):
        super().tearDown()
        ledgers.clear_all()
        # src caches the ledger's dataframe handle; clear_all orphans it, and a
        # later test file writing through the stale handle fails with "Proxy
        # target not set". Let it re-resolve, as a fresh process would.
        wl_src.DATAFRAME_M = None
        os.environ.pop(ENV_ENABLED, None)
        if self._saved_env is not None:
            os.environ[ENV_ENABLED] = self._saved_env
        shutil.rmtree(self.root, ignore_errors=True)

    def boot(self, **model_kwargs):
        """What a fresh training process does: ledger, HP, then the model."""
        ledgers.clear_all()
        detach_projection()
        manager = CheckpointManager(root_log_dir=self.root)
        ledgers.register_checkpoint_manager(manager)
        wl.watch_or_edit({"root_log_dir": self.root,
                          "experiment_dump_to_train_steps_ratio": 10 ** 9},
                         flag="hyperparameters")
        model_kwargs.setdefault("projection", {"every_n_steps": 1})
        model = wl.watch_or_edit(TinyNet(), flag="model", **model_kwargs)
        return manager, model

    def checkpoint(self, manager):
        manager.update_experiment_hash(first_time=True)
        return manager.save_model_checkpoint(save_optimizer=False, update_manifest=True)


# ---------------------------------------------------------------------------
# (b) checkpoints and restarts
# ---------------------------------------------------------------------------
class TestCheckpointPairing(_LedgerRun):

    def test_restart_resumes_the_checkpoints_encoder_not_a_later_one(self):
        """The start-up order: ModelInterface restores the checkpoint while it
        is being built, the projection attaches only afterwards. The restore
        used to find no tracker, do nothing, and leave the attach to pick up
        the run-level encoder -- written AFTER the checkpoint, so it did not
        match the weights just restored."""
        manager, model = self.boot()
        fit(model, 3)
        ckpt = self.checkpoint(manager)
        at_checkpoint = encoder_snapshot()
        fits_at_checkpoint = get_tracker().steps_trained

        fit(model, 5, start=10)
        save_projection()                       # the run-level copy moves on
        later = encoder_snapshot()
        self.assertFalse(same_weights(at_checkpoint, later))

        self.boot()                             # "restart the process"
        tracker = get_tracker()
        self.assertIsNotNone(tracker)
        self.assertTrue(same_weights(encoder_snapshot(), at_checkpoint))
        self.assertEqual(tracker.steps_trained, fits_at_checkpoint)
        self.assertTrue(CheckpointManager.projection_sidecar_path(ckpt).exists())

    def test_restore_while_running_rolls_the_encoder_back(self):
        """The Studio/agent order: the tracker is attached when the restore
        lands, so it applies at once -- and drops the buffered features, which
        came from the model that was just replaced."""
        manager, model = self.boot()
        fit(model, 3)
        ckpt = self.checkpoint(manager)
        at_checkpoint = encoder_snapshot()
        fit(model, 4, start=10)
        self.assertTrue(get_tracker()._buffer_ids)

        self.assertTrue(manager._load_projection_sidecar(ckpt))
        self.assertTrue(same_weights(encoder_snapshot(), at_checkpoint))
        self.assertEqual(get_tracker()._buffer_ids, [])

    def test_a_checkpoint_from_before_the_first_fit_restores_an_empty_layout(self):
        """No sidecar means no encoder existed at that step. Keeping the later
        encoder would read the old model's features through a newer map."""
        manager, model = self.boot()
        ckpt = self.checkpoint(manager)         # nothing fitted yet
        self.assertFalse(CheckpointManager.projection_sidecar_path(ckpt).exists())
        fit(model, 3)
        self.assertIsNotNone(get_tracker()._encoder)

        self.assertFalse(manager._load_projection_sidecar(ckpt))
        tracker = get_tracker()
        self.assertIsNone(tracker._encoder)
        self.assertEqual(tracker.steps_trained, 0)
        # ...and it simply starts learning again from there.
        fit(model, 1, start=20)
        self.assertIsNotNone(tracker._encoder)

    def test_the_sidecar_never_passes_for_a_weight_checkpoint(self):
        manager, model = self.boot()
        fit(model, 2)
        ckpt = self.checkpoint(manager)
        sidecar = CheckpointManager.projection_sidecar_path(ckpt)
        self.assertTrue(sidecar.exists())
        self.assertEqual(sidecar.parent.name, "projection")
        picked = manager._select_weight_checkpoint_file(manager.current_exp_hash)
        self.assertEqual(picked.name, ckpt.name)

    def test_a_parked_restore_is_dropped_when_the_model_opts_out(self):
        manager, model = self.boot()
        fit(model, 2)
        ckpt = self.checkpoint(manager)
        detach_projection()
        manager._load_projection_sidecar(ckpt)  # no tracker: parked
        self.assertIsNotNone(pu._PENDING_CHECKPOINT)
        self.boot(projection=False)
        self.assertIsNone(pu._PENDING_CHECKPOINT)
        self.assertIsNone(get_tracker())


class TestArchitecturePickling(_StubbedFrame):
    """Model architectures are saved with dill, mid-run, hook attached."""

    def setUp(self):
        super().setUp()
        self.model = TinyNet()
        self.live = attach_projection(self.model, every_n_steps=1)
        fit(self.model, 2)

    def test_the_architecture_file_does_not_carry_the_tracker(self):
        bare = len(dill.dumps(TinyNet()))
        hooked = len(dill.dumps(self.model))
        # Used to be ~160x: encoder, optimizer state and feature buffer rode
        # along in every architecture file.
        self.assertLess(hooked, bare * 1.5)

    def test_a_restored_architecture_feeds_the_live_tracker_once_reattached(self):
        clone = dill.loads(dill.dumps(self.model))
        self.assertEqual(projection_hooks(clone.classifier), [])
        self.live._pending = None
        clone(th.randn(5, 16))
        self.assertIsNone(self.live._pending, "a dead hook fed something")

        self.assertTrue(reattach_projection(clone))
        clone(th.randn(5, 16))
        self.assertEqual(self.live._pending.shape, (5, 8))
        # The placeholder hook is gone and the old model is no longer hooked.
        self.assertEqual(len(clone.classifier._forward_hooks), 1)
        self.assertEqual(projection_hooks(self.model.classifier), [])
        # The encoder survived the move: same layer, same width.
        self.assertIsNotNone(self.live._encoder)

    def test_a_deep_copy_keeps_feeding_the_live_tracker(self):
        """An EMA shadow model is a deepcopy; it should behave as before."""
        shadow = copy.deepcopy(self.model)
        self.live._pending = None
        shadow(th.randn(6, 16))
        self.assertEqual(self.live._pending.shape, (6, 8))

    def test_reattach_finds_the_same_layer_by_name(self):
        detach_projection()
        attach_projection(self.model, layer="feat", every_n_steps=1)
        clone = dill.loads(dill.dumps(self.model))
        reattach_projection(clone)
        self.assertEqual(get_tracker().layer_name, "feat")
        self.assertEqual(len(projection_hooks(clone.feat)), 1)


# ---------------------------------------------------------------------------
# (c) switching it off, and what happens when it cannot run
# ---------------------------------------------------------------------------
class TestModelWrapperSwitch(_LedgerRun):

    def test_projection_false_installs_nothing(self):
        _, model = self.boot(projection=False)
        self.assertIsNone(get_tracker())
        inner = pu.resolve_module(model)
        self.assertTrue(all(not projection_hooks(m) for m in inner.modules()))

    def test_projection_false_removes_the_previous_models_hook(self):
        _, first = self.boot()
        first_inner = pu.resolve_module(first)
        self.assertEqual(len(projection_hooks(first_inner.classifier)), 1)
        self.boot(projection=False)
        self.assertIsNone(get_tracker())
        self.assertEqual(projection_hooks(first_inner.classifier), [])

    def test_rewrapping_moves_the_hook_instead_of_adding_one(self):
        _, first = self.boot()
        _, second = self.boot()
        self.assertEqual(projection_hooks(pu.resolve_module(first).classifier), [])
        self.assertEqual(len(projection_hooks(pu.resolve_module(second).classifier)), 1)

    def test_a_string_is_shorthand_for_the_layer(self):
        self.boot(projection="feat")
        self.assertEqual(get_tracker().layer_name, "feat")

    def test_a_dict_steers_the_tracker(self):
        self.boot(projection={"every_n_steps": 7, "graph_size": 64,
                              "n_neighbors": 5, "signal_prefix": "umap_head"})
        tracker = get_tracker()
        self.assertEqual((tracker.every_n_steps, tracker.graph_size,
                          tracker.n_neighbors, tracker.signal_prefix),
                         (7, 64, 5, "umap_head"))

    def test_the_env_switch_wins_over_projection_true(self):
        os.environ[ENV_ENABLED] = "0"
        self.boot(projection=True)
        self.assertIsNone(get_tracker())

    def test_an_unknown_layer_falls_back_to_the_auto_pick(self):
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            self.boot(projection={"layer": "no.such.layer"})
        self.assertIn("not found", "\n".join(captured.output))
        self.assertEqual(get_tracker().layer_name, "classifier")

    def test_an_unknown_option_is_a_warning_not_a_crash(self):
        with self.assertLogs("weightslab.src", level="WARNING") as captured:
            _, model = self.boot(projection={"every_n_step": 5})   # typo
        self.assertIn("every_n_step", "\n".join(captured.output))
        self.assertIsNone(get_tracker())
        self.assertEqual(tuple(model(th.randn(2, 16)).shape), (2, 3))

    def test_a_projection_that_cannot_install_never_stops_the_wrap(self):
        with mock.patch.object(wl_src, "_attach_projection",
                               side_effect=RuntimeError("boom")):
            with self.assertLogs("weightslab.src", level="WARNING") as captured:
                _, model = self.boot()
        self.assertIn("Could not install live projection", "\n".join(captured.output))
        self.assertIsNone(get_tracker())
        self.assertEqual(tuple(model(th.randn(2, 16)).shape), (2, 3))


class TestFailureContract(_StubbedFrame):
    """Instrumentation may stop; it may never take the run down."""

    def setUp(self):
        super().setUp()
        self.model = TinyNet()
        self.tracker = attach_projection(self.model, every_n_steps=1)

    def _fail_fits(self):
        return mock.patch.object(self.tracker, "_umap_loss",
                                 side_effect=RuntimeError("CUDA out of memory"))

    def test_a_failing_fit_warns_once_and_training_carries_on(self):
        with self._fail_fits(), self.assertLogs(LOGGER, level="WARNING") as captured:
            fit(self.model, 3)                  # raises nothing
        warnings = [m for m in captured.output if "fit failed" in m]
        self.assertEqual(len(warnings), 1)
        self.assertIn("Training is unaffected", warnings[0])
        self.assertEqual(self.tracker.failures, 3)
        self.assertIsNotNone(self.tracker._handle)   # still trying

    def test_a_streak_of_failures_turns_the_projection_off(self):
        with self._fail_fits(), self.assertLogs(LOGGER, level="WARNING") as captured:
            fit(self.model, pu._MAX_CONSECUTIVE_FAILURES)
        self.assertIsNone(self.tracker._handle)
        self.assertIn("consecutive failed fits", self.tracker.disabled_reason)
        self.assertIn("turned off", "\n".join(captured.output))
        stats = self.tracker.stats()
        self.assertFalse(stats["enabled"])
        self.assertEqual(stats["failures"], pu._MAX_CONSECUTIVE_FAILURES)
        # Nothing is hooked any more, and a restore does not revive it.
        self.assertEqual(projection_hooks(self.model.classifier), [])
        self.assertFalse(reattach_projection(self.model))
        self.assertEqual(tuple(self.model(th.randn(2, 16)).shape), (2, 3))

    def test_one_good_fit_resets_the_streak(self):
        n = pu._MAX_CONSECUTIVE_FAILURES - 1
        with self._fail_fits():
            fit(self.model, n, start=1)
        fit(self.model, 1, start=50)
        with self._fail_fits():
            fit(self.model, n, start=100)
        self.assertIsNotNone(self.tracker._handle)
        self.assertIsNone(self.tracker.disabled_reason)

    def test_nonfinite_features_never_reach_the_encoder(self):
        """A diverged batch is dropped from the graph; the healthy samples
        around it keep training the encoder, which stays finite."""
        fit(self.model, 1)
        nan = lambda t, feature_last: th.full((t.shape[0], 8), float("nan"))
        with mock.patch.object(pu, "flatten_features", side_effect=nan):
            with self.assertLogs(LOGGER, level="WARNING") as captured:
                for step in (10, 11, 12):
                    self.model(th.randn(32, 16))
                    observe_batch(list(range(100, 132)), step)
        self.assertEqual(sum("NaN/inf" in m for m in captured.output), 1)
        self.assertTrue(all(th.isfinite(v).all() for v in encoder_snapshot().values()))
        self.assertEqual(self.tracker.skipped_nonfinite, 3 * 32)   # samples, not batches
        self.assertEqual(self.tracker.failures, 0)
        placed = {sid for c in self.stub.calls for sid in c["sample_ids"]}
        self.assertFalse(placed & {str(i) for i in range(100, 132)})

    def test_a_nonfinite_loss_is_never_stepped(self):
        fit(self.model, 1)
        before = encoder_snapshot()
        nan = th.tensor(float("nan"), requires_grad=True)
        with mock.patch.object(self.tracker, "_umap_loss", return_value=nan):
            fit(self.model, 1, start=10)
        self.assertTrue(same_weights(before, encoder_snapshot()))
        self.assertEqual(self.tracker.failures, 1)

    def test_a_broken_hook_never_breaks_the_forward_pass(self):
        with mock.patch.object(pu, "flatten_features", side_effect=ValueError("bad")):
            out = self.model(th.randn(4, 16))
        self.assertEqual(tuple(out.shape), (4, 3))


class TestLiveCoverage(_StubbedFrame):
    """Every training batch feeds the graph and gets placed, not only the one
    batch in every_n_steps that lands on a fit step."""

    BATCH = 8

    def setUp(self):
        super().setUp()
        self.model = TinyNet()

    def _step(self, step, n_batches=1):
        """One training step; ids are unique per (step, micro-batch)."""
        for k in range(n_batches):
            self.model.train()
            self.model(th.randn(self.BATCH, 16))
            base = (step * 10 + k) * self.BATCH
            observe_batch(list(range(base, base + self.BATCH)), step)

    def _placed(self):
        return {sid for c in self.stub.calls for sid in c["sample_ids"]}

    def _ids(self, step, k=0):
        base = (step * 10 + k) * self.BATCH
        return {str(i) for i in range(base, base + self.BATCH)}

    def test_non_fit_batches_reach_the_graph(self):
        tracker = attach_projection(self.model, every_n_steps=5)
        for step in range(1, 11):
            self._step(step)
        # The fit at step 10 saw the batches of steps 1-10, not just 5 and 10.
        self.assertEqual(len(tracker._buffer_ids), 10 * self.BATCH)
        self.assertTrue(self._ids(7) <= set(tracker._buffer_ids))

    def test_every_sample_is_placed_once_the_encoder_has_fit(self):
        attach_projection(self.model, every_n_steps=5)
        for step in range(1, 21):
            self._step(step)
        placed = self._placed()
        for step in range(1, 21):
            self.assertTrue(self._ids(step) <= placed, f"step {step} not placed")

    def test_no_random_cloud_before_the_first_fit(self):
        attach_projection(self.model, every_n_steps=5)
        for step in range(1, 5):
            self._step(step)
        self.assertEqual(self.stub.calls, [])

    def test_buffer_memory_stays_bounded_between_fits(self):
        tracker = attach_projection(self.model, every_n_steps=10 ** 6, graph_size=64)
        for step in range(1, 51):
            self._step(step)
        self.assertLessEqual(tracker._buffer_count, 64 + self.BATCH)
        self.assertEqual(tracker._buffer_count, len(tracker._buffer_ids))
        self.assertEqual(tracker._buffer_count,
                         sum(f.shape[0] for f in tracker._buffer_feats))

    def test_gradient_accumulation_micro_batches_all_count(self):
        """Several forwards share one optimizer step, hence one step number."""
        tracker = attach_projection(self.model, every_n_steps=10 ** 6)
        self._step(1, n_batches=3)
        self.assertEqual(len(tracker._buffer_ids), 3 * self.BATCH)

    def test_a_second_write_for_the_same_batch_is_not_a_miss(self):
        """A flag="loss" criterion and a save_signals on the same batch both
        call in; the second finds the features taken, which is not a miss."""
        attach_projection(self.model, every_n_steps=5)
        with mock.patch.object(pu.logger, "warning") as warned:
            for step in range(1, 11):
                self._step(step)
                observe_batch(list(range(self.BATCH)), step)   # the second write
        self.assertFalse(any("forward path" in str(c) for c in warned.call_args_list))

    def test_a_layer_that_never_runs_is_still_reported(self):
        tracker = attach_projection(self.model, layer="feat", every_n_steps=5)
        tracker.detach()                                       # never captures
        tracker._handle = object()                             # but looks attached
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            for step in range(1, 4):
                observe_batch(list(range(self.BATCH)), step)
        self.assertIn("forward path", "\n".join(captured.output))


class TestSeenCounters(_StubbedFrame):
    """Placing a sample is not the model seeing it."""

    def test_the_live_write_back_carries_no_step(self):
        model = TinyNet()
        attach_projection(model, every_n_steps=1)
        fit(model, 1, start=5)
        writes = [c for c in self.stub.calls if "signals//umap_x" in (c["losses"] or {})]
        self.assertTrue(writes)
        # step=None is what keeps enqueue_batch off nb_seen / last_seen.
        self.assertTrue(all(c["step"] is None for c in writes))

    def test_an_ordinary_signal_still_counts_as_seen(self):
        wl_src.save_signals(signals={"loss": [1.0, 2.0]}, batch_ids=[1, 2], step=9)
        self.assertEqual(self.stub.calls[0]["step"], 9)

    def test_enqueue_without_a_step_leaves_the_counters_alone(self):
        """The guarantee the two tests above rely on, on the real manager."""
        from weightslab.data.dataframe_manager import LedgeredDataFrameManager
        from weightslab.data.sample_stats import SampleStats
        manager = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
        manager.enqueue_batch(sample_ids=["1"], preds_raw=None, preds=None,
                              losses={"signals//umap_x": np.array([0.5])}, step=None)
        record = manager._buffer["1"]
        self.assertNotIn(SampleStats.Ex.NB_SEEN.value, record)
        self.assertNotIn(SampleStats.Ex.LAST_SEEN.value, record)


# ---------------------------------------------------------------------------
# (a) projections the user computes themselves
# ---------------------------------------------------------------------------
class TestBringYourOwnProjection(_StubbedFrame):

    def test_it_is_a_top_level_verb(self):
        self.assertIs(wl.save_projection_coords, save_projection_coords)

    def test_coordinates_land_as_signals_under_the_prefix(self):
        coords = np.arange(12, dtype=np.float32).reshape(4, 3)
        columns = save_projection_coords(coords, batch_ids=[10, 11, 12, 13], prefix="tsne")
        self.assertEqual(columns, ["signals//tsne_x", "signals//tsne_y", "signals//tsne_z"])
        call = self.stub.calls[0]
        self.assertEqual(call["sample_ids"], ["10", "11", "12", "13"])
        np.testing.assert_allclose(call["losses"]["signals//tsne_y"], [1, 4, 7, 10])
        self.assertIsNone(call["step"])     # not counted as seen

    def test_tensors_and_lists_are_accepted(self):
        save_projection_coords(th.zeros(3, 2), th.tensor([1, 2, 3]), prefix="pca")
        save_projection_coords([[0, 1], [2, 3]], ["a", "b"], prefix="pca")
        self.assertEqual(self.stub.written("pca"), {"signals//pca_x", "signals//pca_y"})

    def test_a_2d_projection_over_a_stale_3d_one_flattens_z(self):
        """The board reads x, y AND z when all three exist; new 2-D x/y would
        otherwise be drawn against the old projection's depth."""
        wl_src.DATAFRAME_M = _Stub(columns=["signals//tsne_z"])
        columns = save_projection_coords(np.ones((2, 2)), [1, 2], prefix="tsne")
        self.assertIn("signals//tsne_z", columns)
        np.testing.assert_allclose(wl_src.DATAFRAME_M.calls[0]["losses"]["signals//tsne_z"], [0, 0])

    def test_shapes_that_cannot_be_drawn_are_refused(self):
        for bad in (np.ones(4), np.ones((4, 1)), np.ones((4, 4)), np.ones((4, 2, 1))):
            with self.assertRaises(ValueError):
                save_projection_coords(bad, [1, 2, 3, 4], prefix="tsne")
        with self.assertRaises(ValueError) as ctx:
            save_projection_coords(np.ones((3, 2)), [1, 2], prefix="tsne")
        self.assertIn("aligned", str(ctx.exception))
        self.assertEqual(self.stub.calls, [])

    def test_prefixes_that_break_column_names_are_refused(self):
        for bad in ("", "  ", "a/b", "has space", "_lead", "signals//x"):
            with self.assertRaises(ValueError, msg=bad):
                save_projection_coords(np.ones((2, 2)), [1, 2], prefix=bad)
        save_projection_coords(np.ones((2, 2)), [1, 2], prefix="pca.layer3-v2")

    def test_the_live_umap_prefix_is_refused_only_while_it_runs(self):
        attach_projection(TinyNet())
        with self.assertRaises(ValueError) as ctx:
            save_projection_coords(np.ones((2, 3)), [1, 2], prefix="umap")
        self.assertIn("projection=False", str(ctx.exception))
        save_projection_coords(np.ones((2, 3)), [1, 2], prefix="tsne")
        detach_projection()
        save_projection_coords(np.ones((2, 3)), [1, 2], prefix="umap")   # now free

    def test_it_is_flushed_at_once_so_the_board_sees_it(self):
        """The board reads flushed rows; a buffered write stayed invisible for
        a whole flush interval right after the user plugged it in."""
        self.stub.flush = mock.MagicMock()
        save_projection_coords(np.ones((2, 2)), [1, 2], prefix="tsne")
        self.stub.flush.assert_called_once()

    def test_nan_rows_are_allowed_and_mean_not_placed(self):
        coords = np.array([[0, 0], [np.nan, np.nan]], dtype=np.float32)
        save_projection_coords(coords, [1, 2], prefix="tsne")
        self.assertTrue(np.isnan(self.stub.calls[0]["losses"]["signals//tsne_x"][1]))


class TestProjectDatasetWithYourMethod(_StubbedFrame):
    """``project_dataset(method=...)``: WeightsLab collects the features, your
    algorithm lays them out."""

    def setUp(self):
        super().setUp()
        from torch.utils.data import DataLoader, TensorDataset
        x = th.randn(40, 16)
        self.loader = DataLoader(TensorDataset(x, th.arange(40)), batch_size=8)
        self.model = TinyNet()

    def test_a_callable(self):
        def first_two(features):
            return features[:, :2]
        stats = wl_src.project_dataset(self.model, self.loader, method=first_two,
                                       verbose=False)
        self.assertEqual(stats["prefix"], "first_two")
        self.assertEqual(stats["method"], "first_two")
        self.assertEqual(stats["feature_dim"], 8)       # the head's input
        self.assertEqual(self.stub.written("first_two"),
                         {"signals//first_two_x", "signals//first_two_y"})

    def test_an_sklearn_style_estimator(self):
        class FakeTSNE:
            def __init__(self, n_components=3):
                self.n_components = n_components
                self.seen = None

            def fit_transform(self, features):
                self.seen = features
                return features[:, :self.n_components]

        estimator = FakeTSNE()
        stats = wl_src.project_dataset(self.model, self.loader, method=estimator,
                                       layer="feat", verbose=False)
        self.assertEqual(stats["prefix"], "faketsne")
        self.assertEqual(estimator.seen.shape, (40, 16))   # input of `feat`
        self.assertIsInstance(estimator.seen, np.ndarray)

    def test_an_explicit_prefix_and_a_lambda(self):
        stats = wl_src.project_dataset(self.model, self.loader, prefix="mine",
                                       method=lambda f: f[:, :3], verbose=False)
        self.assertEqual(stats["prefix"], "mine")
        stats = wl_src.project_dataset(self.model, self.loader,
                                       method=lambda f: f[:, :3], verbose=False)
        self.assertEqual(stats["prefix"], "custom")

    def test_no_encoder_is_saved_or_adopted(self):
        live = attach_projection(TinyNet())
        with mock.patch.object(pu, "save_projection") as saved:
            stats = wl_src.project_dataset(self.model, self.loader, prefix="pca",
                                           method=lambda f: f[:, :2], verbose=False)
        saved.assert_not_called()
        self.assertFalse(stats["adopted_by_live"])
        self.assertIs(get_tracker(), live)
        self.assertEqual(live.signal_prefix, "umap")

    def test_mistakes_are_caught_before_the_feature_sweep(self):
        attach_projection(TinyNet())
        swept = mock.MagicMock(side_effect=AssertionError("swept"))
        with mock.patch.object(pu, "tqdm", swept):
            with self.assertRaises(ValueError):
                wl_src.project_dataset(self.model, self.loader, method="tsne")
            with self.assertRaises(ValueError):           # the live prefix
                wl_src.project_dataset(self.model, self.loader, prefix="umap",
                                       method=lambda f: f[:, :2])

    def test_a_method_returning_the_wrong_shape_says_so(self):
        with self.assertRaises(ValueError) as ctx:
            wl_src.project_dataset(self.model, self.loader,
                                   method=lambda f: f[:, :5], verbose=False)
        self.assertIn("(N, 2) or (N, 3)", str(ctx.exception))

    def test_the_built_in_method_is_unchanged(self):
        stats = wl_src.project_dataset(self.model, self.loader, epochs=1, verbose=False)
        self.assertEqual((stats["prefix"], stats["method"]), ("umap", "umap"))


class _StatefulLoader:
    """A tracked WeightsLab loader's contract: ONE iterator, which a ``for``
    continues rather than restarts, with ``reset_iterator`` to start over."""

    def __init__(self, x, batch=8):
        self.batches = [(x[i:i + batch], th.arange(i, min(i + batch, len(x))))
                        for i in range(0, len(x), batch)]
        self.position = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.batches):
            self.position = 0                    # the next pass starts fresh
            raise StopIteration
        batch = self.batches[self.position]
        self.position += 1
        return batch

    def reset_iterator(self):
        self.position = 0


class _AgedNet(TinyNet):
    """A tracked model's age rule: a forward is a training step while the
    tracking mode is TRAIN -- which a training guard leaves behind on exit."""

    def __init__(self):
        super().__init__()
        from weightslab.components.tracking import TrackingMode
        self.tracking_mode = TrackingMode.TRAIN
        self.age = 0

    def set_tracking_mode(self, mode):
        self.tracking_mode = mode

    def forward(self, x):
        from weightslab.components.tracking import TrackingMode
        if self.tracking_mode == TrackingMode.TRAIN:
            self.age += 1
        return super().forward(x)


class TestProjectDatasetDoesNotAgeTheModel(_StubbedFrame):
    """The feature sweep is not training: it must not move the model's age."""

    def setUp(self):
        super().setUp()
        from torch.utils.data import DataLoader, TensorDataset
        self.loader = DataLoader(TensorDataset(th.randn(40, 16), th.arange(40)), batch_size=8)
        self.model = _AgedNet()

    def test_a_sweep_leaves_the_age_alone(self):
        wl_src.project_dataset(self.model, self.loader, method=lambda f: f[:, :2],
                               verbose=False)
        self.assertEqual(self.model.age, 0)           # five batches, no steps

    def test_the_built_in_method_leaves_it_alone_too(self):
        wl_src.project_dataset(self.model, self.loader, epochs=1, verbose=False)
        self.assertEqual(self.model.age, 0)

    def test_the_previous_tracking_mode_comes_back(self):
        from weightslab.components.tracking import TrackingMode
        wl_src.project_dataset(self.model, self.loader, method=lambda f: f[:, :2],
                               verbose=False)
        self.assertEqual(self.model.tracking_mode, TrackingMode.TRAIN)

    def test_it_comes_back_when_the_sweep_itself_fails(self):
        from weightslab.components.tracking import TrackingMode

        def breaks_after_one_batch():
            yield th.randn(8, 16), th.arange(8)
            raise RuntimeError("the loader died")

        with self.assertRaises(RuntimeError):
            wl_src.project_dataset(self.model, breaks_after_one_batch(),
                                   method=lambda f: f[:, :2], verbose=False)
        self.assertEqual(self.model.tracking_mode, TrackingMode.TRAIN)

    def test_a_model_that_does_not_track_is_untouched(self):
        plain = TinyNet()                              # no set_tracking_mode at all
        stats = wl_src.project_dataset(plain, self.loader, method=lambda f: f[:, :2],
                                       verbose=False)
        self.assertEqual(stats["samples"], 40)


class TestProjectDatasetOnALoaderMidEpoch(_StubbedFrame):
    """A training loop is nearly always mid-epoch on its loader. The sweep must
    still see the WHOLE split, not the rest of that epoch."""

    def setUp(self):
        super().setUp()
        self.loader = _StatefulLoader(th.randn(40, 16))
        self.model = TinyNet()
        for _ in range(3):                       # the training loop's three steps
            next(self.loader)

    def test_the_sweep_covers_the_whole_split(self):
        stats = wl_src.project_dataset(self.model, self.loader,
                                       method=lambda f: f[:, :2], verbose=False)
        self.assertEqual(stats["samples"], 40)   # not the 16 left of the epoch

    def test_every_sample_id_is_placed_exactly_once(self):
        wl_src.project_dataset(self.model, self.loader, prefix="pca",
                               method=lambda f: f[:, :2], verbose=False)
        placed = [i for call in self.stub.calls for i in call["sample_ids"]]
        self.assertEqual(sorted(int(i) for i in placed), list(range(40)))

    def test_a_finished_sweep_leaves_training_a_fresh_epoch(self):
        wl_src.project_dataset(self.model, self.loader,
                               method=lambda f: f[:, :2], verbose=False)
        first, _ = next(self.loader)
        self.assertEqual(self.loader.position, 1)    # the first batch of a new pass
        self.assertTrue(th.equal(first, self.loader.batches[0][0]))

    def test_stopping_early_does_not_strand_training_mid_sweep(self):
        wl_src.project_dataset(self.model, self.loader, max_samples=16,
                               method=lambda f: f[:, :2], verbose=False)
        next(self.loader)
        self.assertEqual(self.loader.position, 1)    # not 3 + the batches swept

    def test_the_built_in_method_sweeps_the_whole_split_too(self):
        stats = wl_src.project_dataset(self.model, self.loader, epochs=1,
                                       verbose=False)
        self.assertEqual(stats["samples"], 40)

    def test_the_sample_ids_follow_the_feature_rows(self):
        """``sample_ids[n]`` is the sample whose features were row ``n`` -- what
        lets a caller pair the rows its method received with labels."""
        from torch.utils.data import DataLoader, TensorDataset
        x = th.randn(40, 16)
        shuffled = DataLoader(TensorDataset(x, th.arange(40)), batch_size=8, shuffle=True)
        seen = {}

        def method(features):
            seen["features"] = features
            return features[:, :2]

        stats = wl_src.project_dataset(self.model, shuffled, layer="feat", method=method,
                                       verbose=False)
        order = [int(i) for i in stats["sample_ids"]]
        self.assertEqual(sorted(order), list(range(40)))
        self.assertNotEqual(order, list(range(40)))         # really shuffled
        np.testing.assert_allclose(seen["features"], x[order].numpy(), rtol=1e-6)

    def test_a_plain_iterable_is_left_alone(self):
        from torch.utils.data import DataLoader, TensorDataset
        plain = DataLoader(TensorDataset(th.randn(40, 16), th.arange(40)), batch_size=8)
        stats = wl_src.project_dataset(self.model, plain,
                                       method=lambda f: f[:, :2], verbose=False)
        self.assertEqual(stats["samples"], 40)


class TestServingAProjectionYouBrought(unittest.TestCase):

    def setUp(self):
        detach_projection()
        self._saved_root = os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        clear_registry()
        for prefix in ("tsne", "umap"):
            register_prefix(prefix)

    def tearDown(self):
        detach_projection()
        clear_registry()
        if self._saved_root is not None:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._saved_root

    def _frame(self, prefixes):
        n = 50
        rng = np.random.default_rng(0)
        index = pd.MultiIndex.from_arrays(
            [np.full(n, "train_loader"), np.arange(n).astype(str)],
            names=["origin", "sample_id"])
        columns = {}
        for prefix in prefixes:
            for axis in "xy":
                columns[f"signals//{prefix}_{axis}"] = rng.normal(size=n)
        return pd.DataFrame(columns, index=index)

    def test_no_preference_picks_what_exists(self):
        """A run with the built-in off and its own t-SNE used to fail the first
        request (asking for 'umap') and only draw a poll later."""
        response = ps.build_projection_response(self._frame(["tsne"]),
                                                pb2.ProjectionRequest())
        self.assertTrue(response.success, response.message)
        self.assertEqual(list(response.available_prefixes), ["tsne"])
        self.assertEqual(response.dims, 2)

    def test_the_live_projection_is_still_the_default_when_present(self):
        frame = self._frame(["tsne", "umap"])
        response = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertEqual(list(response.available_prefixes), ["umap", "tsne"])
        asked_umap = ps.build_projection_response(frame, pb2.ProjectionRequest(prefix="umap"))
        self.assertEqual(list(response.coords), list(asked_umap.coords))

    def test_fits_belong_to_the_live_prefix_only(self):
        tracker = attach_projection(TinyNet())
        tracker.steps_trained = 42
        frame = self._frame(["tsne", "umap"])
        live = ps.build_projection_response(frame, pb2.ProjectionRequest(prefix="umap"))
        mine = ps.build_projection_response(frame, pb2.ProjectionRequest(prefix="tsne"))
        self.assertEqual(live.fits, 42)
        self.assertEqual(mine.fits, 0)


class TestNoProjectionReason(_StubbedFrame):
    """The text the Studio's "No projection found" ribbon shows has to say what
    to do about it."""

    def setUp(self):
        super().setUp()
        self._saved_root = os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        clear_registry()

    def tearDown(self):
        super().tearDown()
        clear_registry()
        if self._saved_root is not None:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._saved_root

    def test_a_projection_just_written_is_about_to_appear(self):
        """The view is rebuilt in the background; right after the user plugs
        a t-SNE in, "plug one in" would be the wrong thing to tell them."""
        register_prefix("tsne")
        reason = ps.missing_reason("umap", [])
        self.assertIn("tsne was just written", reason)
        self.assertNotIn("Plug in your own", reason)

    def test_another_projection_exists(self):
        self.assertIn("Available: tsne", ps.missing_reason("umap", ["tsne"]))

    def test_the_live_umap_has_not_fitted_yet(self):
        attach_projection(TinyNet(), every_n_steps=25)
        reason = ps.missing_reason("umap", [])
        self.assertIn("first fit", reason)
        self.assertIn("every 25", reason)

    def test_the_live_umap_stopped_itself(self):
        tracker = attach_projection(TinyNet())
        tracker.disabled_reason = "5 consecutive failed fits"
        self.assertIn("stopped after 5 consecutive", ps.missing_reason("umap", []))

    def test_the_built_in_is_off(self):
        reason = ps.missing_reason("umap", [])
        self.assertIn("off for this run", reason)
        self.assertIn("save_projection_coords", reason)


class TestOnlyRealProjectionsAreOffered(_StubbedFrame):
    """A column pair alone does not make a projection (see registry)."""

    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp(prefix="wl-registry-")
        self._saved_root = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self.root
        clear_registry()

    def tearDown(self):
        super().tearDown()
        clear_registry()
        os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        if self._saved_root is not None:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._saved_root
        shutil.rmtree(self.root, ignore_errors=True)

    def _frame(self, prefixes):
        n = 20
        index = pd.MultiIndex.from_arrays(
            [np.arange(n).astype(str), np.zeros(n, dtype=int)],
            names=["sample_id", "annotation_id"])
        return pd.DataFrame({f"signals//{p}_{a}": np.random.rand(n)
                             for p in prefixes for a in "xy"}, index=index)

    def test_an_ordinary_signal_pair_is_not_a_projection(self):
        save_projection_coords(np.ones((2, 2)), [1, 2], prefix="tsne")
        frame = self._frame(["umap", "tsne", "center"])
        self.assertEqual(ps.available_prefixes(frame), ["umap", "tsne"])

    def test_the_live_projection_registers_itself(self):
        model = TinyNet()
        attach_projection(model, every_n_steps=1, signal_prefix="umap_head")
        fit(model, 1)
        self.assertIn("umap_head", known_prefixes())
        self.assertIn("umap_head", ps.available_prefixes(self._frame(["umap_head", "other"])))

    def test_project_dataset_with_your_method_registers_its_prefix(self):
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(th.randn(16, 16), th.arange(16)), batch_size=8)
        wl_src.project_dataset(TinyNet(), loader, method=lambda f: f[:, :2],
                               prefix="mine", verbose=False)
        self.assertIn("mine", known_prefixes())

    def test_the_registry_survives_a_restart(self):
        save_projection_coords(np.ones((2, 3)), [1, 2], prefix="tsne_step500")
        self.assertTrue(os.path.exists(os.path.join(self.root, "projection", REGISTRY_FILE)))
        clear_registry()                       # a new process: memory is empty
        self.assertIn("tsne_step500", known_prefixes())

    def test_a_run_from_before_the_registry_still_lists_its_projections(self):
        self.assertIsNone(known_prefixes())
        self.assertEqual(ps.available_prefixes(self._frame(["umap", "pca"])), ["umap", "pca"])

    def test_clear_all_forgets_the_registry(self):
        register_prefix("tsne", root_log_dir=None)
        os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)    # nothing persisted
        clear_registry()
        register_prefix("tsne")
        wl.clear_all()
        self.assertIsNone(known_prefixes())


# ---------------------------------------------------------------------------
# Grid Overview: GetProjection restricted to the samples the grid renders
# ---------------------------------------------------------------------------
class TestRestrictToSampleIds(unittest.TestCase):

    def _frame(self, n=20):
        index = pd.MultiIndex.from_arrays(
            [np.where(np.arange(n) % 4 == 0, "test_loader", "train_loader"),
             np.arange(n).astype(str)], names=["origin", "sample_id"])
        return pd.DataFrame({f"signals//umap_{a}": np.arange(n, dtype=float) for a in "xyz"},
                            index=index)

    def test_keeps_only_the_asked_samples_compared_as_strings(self):
        out = ps.restrict_to_sample_ids(self._frame(), [3, "5", 7])
        self.assertEqual(sorted(ps._sample_ids(out).tolist()), ["3", "5", "7"])

    def test_no_match_is_empty_not_the_whole_frame(self):
        """Serving everything on a mismatch made a broken restriction look like
        one that was never applied."""
        self.assertEqual(len(ps.restrict_to_sample_ids(self._frame(), ["nope"])), 0)

    def test_an_empty_list_means_no_restriction(self):
        self.assertEqual(len(ps.restrict_to_sample_ids(self._frame(), [])), 20)

    def test_works_on_the_ledger_layout_too(self):
        n = 10
        index = pd.MultiIndex.from_arrays(
            [np.arange(n).astype(str), np.zeros(n, dtype=int)],
            names=["sample_id", "annotation_id"])
        frame = pd.DataFrame({"signals//umap_x": np.arange(n, dtype=float)}, index=index)
        self.assertEqual(len(ps.restrict_to_sample_ids(frame, ["1", "2"])), 2)


class TestGetProjectionGridRestriction(unittest.TestCase):
    """The RPC itself, with a stand-in for the data service's state."""

    def setUp(self):
        import contextlib
        from types import SimpleNamespace
        n = 500
        index = pd.MultiIndex.from_arrays(
            [np.where(np.arange(n) % 4 == 0, "test_loader", "train_loader"),
             np.arange(n).astype(str)], names=["origin", "sample_id"])
        rng = np.random.default_rng(1)
        self.frame = pd.DataFrame(
            {f"signals//umap_{a}": rng.uniform(-10, 10, n) for a in "xyz"}, index=index)
        self.make = lambda filtered: SimpleNamespace(
            _all_datasets_df=self.frame, _is_filtered=filtered,
            _watched_lock=lambda name: contextlib.nullcontext(),
            _fastUpdateInternals=lambda: True, _slowUpdateInternals=lambda: None,
            _pull_into_all_data_view_df=lambda: self.frame)

    def ask(self, filtered, ids, follow=True):
        from weightslab.trainer.services.data_service import DataService
        request = pb2.ProjectionRequest(follow_view=follow, restrict_sample_ids=[str(i) for i in ids])
        return DataService.GetProjection(self.make(filtered), request, None)

    def test_the_cloud_is_exactly_the_grid_page(self):
        resp = self.ask(False, range(2000 // 10, 2000 // 10 + 32))
        self.assertTrue(resp.success, resp.message)
        self.assertEqual(sorted(map(int, resp.sample_ids)), list(range(200, 232)))

    def test_applies_to_a_filtered_data_view_as_well(self):
        resp = self.ask(True, range(300, 332))
        self.assertEqual(sorted(map(int, resp.sample_ids)), list(range(300, 332)))

    def test_without_follow_view_the_ids_are_ignored(self):
        resp = self.ask(False, range(10), follow=False)
        self.assertEqual(len(resp.sample_ids), 500)

    def test_an_empty_list_leaves_the_whole_view(self):
        self.assertEqual(len(self.ask(False, []).sample_ids), 500)

    def test_a_page_that_matches_nothing_says_so(self):
        resp = self.ask(False, ["x1", "x2"])
        self.assertFalse(resp.success)
        self.assertIn("None of the 2 samples", resp.message)


# ---------------------------------------------------------------------------
# Your own projections follow the checkpoint they were computed at
# ---------------------------------------------------------------------------
class _FakeFrameManager:
    """A dataframe manager whose rows really change, so a restore can be checked."""

    def __init__(self, n=6):
        index = pd.MultiIndex.from_arrays(
            [np.arange(n).astype(str), np.zeros(n, dtype=int)],
            names=["sample_id", "annotation_id"])
        self.df = pd.DataFrame(index=index)
        self.flushed = 0

    def get_df_view(self, *args, **kwargs):
        return self.df

    def flush(self):
        self.flushed += 1

    def save_signals(self, signals, batch_ids, log=False, _seen=True, **_):
        for name, values in signals.items():
            column = f"signals//{name}"
            if column not in self.df.columns:
                self.df[column] = np.nan
            for sid, value in zip(batch_ids, np.asarray(values, dtype=float)):
                self.df.loc[(str(sid), 0), column] = value


class TestCustomProjectionSnapshots(unittest.TestCase):
    """The sidecar format and the apply step, without a checkpoint manager."""

    def setUp(self):
        from weightslab.projection import snapshots
        self.snapshots = snapshots
        self.frames = _FakeFrameManager()
        self.root = tempfile.mkdtemp(prefix="wl-snap-")
        self._saved = wl_src.DATAFRAME_M
        wl_src.DATAFRAME_M = self.frames
        self._prefixes = {"tsne", "pca"}
        self._patches = [
            mock.patch.object(snapshots, "known_prefixes", lambda *a, **k: set(self._prefixes)),
            mock.patch.object(snapshots, "register_prefix", lambda p, *a, **k: self._prefixes.add(p)),
            mock.patch.object(snapshots, "unregister_prefix", lambda p, *a, **k: self._prefixes.discard(p)),
            mock.patch.object(wl_src, "save_signals", self.frames.save_signals),
        ]
        for patch in self._patches:
            patch.start()
        detach_projection()

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        wl_src.DATAFRAME_M = self._saved
        shutil.rmtree(self.root, ignore_errors=True)

    def place(self, prefix, coords, ids=None):
        coords = np.asarray(coords, dtype=np.float32)
        ids = ids if ids is not None else [str(i) for i in range(len(coords))]
        self.frames.save_signals(
            {f"{prefix}_{a}": coords[:, i] for i, a in enumerate("xyz"[:coords.shape[1]])}, ids)

    def test_snapshot_keeps_placed_samples_in_2d_and_3d(self):
        self.place("pca", [[0, 1], [2, 3], [4, 5]], ids=["0", "1", "2"])
        self.place("tsne", [[0, 1, 2], [3, 4, 5]], ids=["3", "4"])
        snap = self.snapshots.snapshot(self.frames.df)
        self.assertEqual(snap["pca"]["coords"].shape, (3, 2))
        self.assertEqual(snap["pca"]["ids"], ["0", "1", "2"])
        self.assertEqual(snap["tsne"]["coords"].shape, (2, 3))
        self.assertEqual(snap["tsne"]["ids"], ["3", "4"])

    def test_snapshot_skips_unplaced_and_half_written_prefixes(self):
        self.place("pca", [[0, 1], [np.nan, 3]], ids=["0", "1"])
        self.frames.df["signals//tsne_x"] = 1.0          # no _y: not a projection
        snap = self.snapshots.snapshot(self.frames.df)
        self.assertEqual(snap["pca"]["ids"], ["0"])
        self.assertNotIn("tsne", snap)

    def test_the_live_prefix_is_not_a_custom_projection(self):
        self._prefixes.add("umap")
        with mock.patch("weightslab.projection.parametric_umap.get_tracker",
                        return_value=mock.Mock(signal_prefix="umap")):
            self.assertEqual(self.snapshots.custom_prefixes(), ["pca", "tsne"])

    def test_sidecar_round_trip_and_location(self):
        ckpt = os.path.join(self.root, "models", "abc", "abc_step_000010.pt")
        self.place("pca", [[1, 2], [3, 4]], ids=["0", "1"])
        self.snapshots.save_sidecar(ckpt, self.frames.df)
        path = self.snapshots.sidecar_path(ckpt)
        self.assertEqual(path.parent.name, "projection")
        self.assertNotEqual(path.name, "abc_step_000010.pt")
        loaded = self.snapshots.load_sidecar(ckpt)
        np.testing.assert_allclose(loaded["pca"]["coords"], [[1, 2], [3, 4]])

    def test_a_checkpoint_without_a_sidecar_loads_as_none(self):
        self.assertIsNone(self.snapshots.load_sidecar(os.path.join(self.root, "none_step_000001.pt")))

    def test_an_empty_snapshot_is_still_written(self):
        """It is what says 'this moment had no custom projection'."""
        ckpt = os.path.join(self.root, "x_step_000001.pt")
        self.snapshots.save_sidecar(ckpt, self.frames.df)
        self.assertEqual(self.snapshots.load_sidecar(ckpt), {})

    def test_apply_restores_saved_and_clears_later_ones(self):
        self.place("pca", [[0, 0], [1, 1], [2, 2]], ids=["0", "1", "2"])
        saved = self.snapshots.snapshot(self.frames.df, prefixes=["pca"])

        # Later: pca is recomputed elsewhere and a new t-SNE appears.
        self.place("pca", [[9, 9], [9, 9], [9, 9], [9, 9]], ids=["0", "1", "2", "3"])
        self.place("tsne", [[5, 5, 5]] * 6, ids=[str(i) for i in range(6)])

        result = self.snapshots.apply(saved, self.frames.df)
        self.assertEqual(result, {"restored": ["pca"], "cleared": ["tsne"]})
        df = self.frames.df
        np.testing.assert_allclose(df["signals//pca_x"].to_numpy()[:3], [0, 1, 2])
        self.assertTrue(np.isnan(df["signals//pca_x"].to_numpy()[3]))   # not in the snapshot
        self.assertTrue(df["signals//tsne_x"].isna().all())
        self.assertNotIn("tsne", self._prefixes)
        self.assertIn("pca", self._prefixes)
        self.assertGreaterEqual(self.frames.flushed, 1)

    def test_apply_does_not_count_the_samples_as_seen(self):
        calls = []
        with mock.patch.object(wl_src, "save_signals",
                               lambda **kw: (calls.append(kw), self.frames.save_signals(**kw))):
            self.place("pca", [[0, 0]], ids=["0"])
            saved = self.snapshots.snapshot(self.frames.df, prefixes=["pca"])
            self.snapshots.apply(saved, self.frames.df)
        self.assertTrue(calls)
        self.assertTrue(all(c["_seen"] is False and c["log"] is False for c in calls))

    def test_unregister_prefix_takes_the_name_out_of_the_registry_file(self):
        from weightslab.projection import registry
        registry.clear_registry()
        registry.register_prefix("a", root_log_dir=self.root)
        registry.register_prefix("b", root_log_dir=self.root)
        registry.unregister_prefix("a", root_log_dir=self.root)
        self.assertEqual(registry.known_prefixes(self.root), {"b"})
        registry.clear_registry()                 # a new process reads the file
        self.assertEqual(registry.known_prefixes(self.root), {"b"})
        registry.unregister_prefix("never-there", root_log_dir=self.root)   # no error
        registry.clear_registry()

    def test_writes_inside_one_timestamp_tick_are_not_mistaken_for_one_state(self):
        """The file cache must not trust mtime alone: on a coarse clock (Windows)
        a write right after another shares its mtime, and the stale read made
        ``unregister_prefix`` write back an EMPTY registry."""
        from weightslab.projection import registry
        registry.clear_registry()
        with mock.patch("os.path.getmtime", return_value=1000.0):     # one tick, always
            registry.register_prefix("a", root_log_dir=self.root)
            registry.register_prefix("b", root_log_dir=self.root)
            registry.unregister_prefix("a", root_log_dir=self.root)
        registry.clear_registry()
        self.assertEqual(registry.known_prefixes(self.root), {"b"})
        registry.clear_registry()


class TestCustomProjectionsFollowCheckpoints(_LedgerRun):
    """Through the real checkpoint manager: save -> compute more -> restore."""

    def setUp(self):
        super().setUp()
        self.frames = _FakeFrameManager()
        self._patches = [
            mock.patch.object(ledgers, "get_dataframe", lambda *a, **k: self.frames),
            mock.patch.object(wl_src, "save_signals", self.frames.save_signals),
        ]
        for patch in self._patches:
            patch.start()
        clear_registry()

    def tearDown(self):
        for patch in self._patches:
            patch.stop()
        clear_registry()
        super().tearDown()

    def place(self, prefix, coords):
        coords = np.asarray(coords, dtype=np.float32)
        ids = [str(i) for i in range(len(coords))]
        self.frames.save_signals(
            {f"{prefix}_{a}": coords[:, i] for i, a in enumerate("xyz"[:coords.shape[1]])}, ids)
        register_prefix(prefix)

    def test_every_checkpoint_carries_the_projections_of_its_moment(self):
        from weightslab.projection import snapshots
        manager, model = self.boot(projection=False)
        self.place("pca", np.arange(12).reshape(6, 2))
        ckpt = self.checkpoint(manager)
        saved = snapshots.load_sidecar(ckpt)
        self.assertEqual(list(saved), ["pca"])
        self.assertEqual(saved["pca"]["coords"].shape, (6, 2))
        # The sidecar must never pass for a weight checkpoint.
        picked = manager._select_weight_checkpoint_file(manager.current_exp_hash)
        self.assertEqual(picked.name, ckpt.name)

    def test_restoring_returns_the_projections_of_that_moment(self):
        manager, model = self.boot(projection=False)
        self.place("pca", np.arange(12).reshape(6, 2))                 # computed at step A
        ckpt = self.checkpoint(manager)

        self.place("tsne", np.arange(18).reshape(6, 3))                # computed after A
        self.place("pca", np.full((6, 2), 7.0))                        # pca recomputed after A
        self.assertEqual(known_prefixes(), {"pca", "tsne"})

        manager._restore_custom_projections(ckpt)
        df = self.frames.df
        np.testing.assert_allclose(df["signals//pca_x"].to_numpy(), np.arange(0, 12, 2))
        self.assertTrue(df["signals//tsne_x"].isna().all())
        self.assertEqual(known_prefixes(), {"pca"})                    # the picker follows

    def test_a_checkpoint_from_before_any_projection_clears_them_all(self):
        manager, model = self.boot(projection=False)
        early = self.checkpoint(manager)                               # nothing computed yet
        self.place("tsne", np.ones((6, 3)))
        manager._restore_custom_projections(early)
        self.assertEqual(known_prefixes(), set())
        self.assertTrue(self.frames.df["signals//tsne_x"].isna().all())

    def test_an_old_checkpoint_without_a_sidecar_changes_nothing(self):
        manager, model = self.boot(projection=False)
        ckpt = self.checkpoint(manager)
        from weightslab.projection import snapshots
        snapshots.sidecar_path(ckpt).unlink()                          # as if written before this feature
        self.place("tsne", np.ones((6, 3)))
        manager._restore_custom_projections(ckpt)
        self.assertEqual(known_prefixes(), {"tsne"})
        self.assertFalse(self.frames.df["signals//tsne_x"].isna().any())

    def test_load_state_applies_them_after_the_per_sample_rewind(self):
        """The rewind blanks every signal column of the samples it rolls back,
        custom coordinates included; the restore has to land on top of it."""
        manager, model = self.boot(projection=False)
        self.place("pca", np.arange(12).reshape(6, 2))
        ckpt = self.checkpoint(manager)
        order = []
        with mock.patch.object(manager, "_rewind_sample_state",
                               lambda *a, **k: order.append("rewind") or 0), \
             mock.patch.object(manager, "_restore_custom_projections",
                               lambda f: order.append(("restore", f))):
            manager._last_loaded_checkpoint_file = ckpt
            manager.load_state(manager.current_exp_hash, load_weights=True,
                               load_data=False, force=True)
        self.assertEqual(order[0], "rewind")
        self.assertEqual(order[1][0], "restore")
        self.assertEqual(os.path.basename(str(order[1][1])), ckpt.name)


if __name__ == "__main__":
    unittest.main()
