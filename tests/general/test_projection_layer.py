"""Which layer the live projection reads, and saying so on the board.

The YOLO example never got a real projection: the auto-pick ("input of the
last conv") landed on the Detect head's frozen DFL box decoder, which training
never runs, and validation ran on an EMA copy built before the hook existed.
"""
import copy
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn

from weightslab import src as wl_src
from weightslab.projection import (
    attach_projection,
    clear_pending_restore,
    detach_projection,
    mirror_projection,
    observe_batch,
    pick_embedding_layer,
    save_projection_coords,
)
from weightslab.projection import parametric_umap as pu
from weightslab.projection.registry import (
    REGISTRY_FILE,
    clear_registry,
    known_prefixes,
    prefix_info,
    register_prefix,
)
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.services import projection_service as ps


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Linear(16, 8)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        return self.classifier(self.act(self.feat(x)))


class DetectLike(nn.Module):
    """A YOLO-style head: the class conv trains, and a frozen decoder conv is
    registered after it but only runs at inference."""

    def __init__(self):
        super().__init__()
        self.cls = nn.Conv2d(8, 2, 1)
        self.dfl = nn.Conv2d(16, 1, 1, bias=False).requires_grad_(False)

    def forward(self, x):
        return self.cls(x)


class YoloLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Conv2d(3, 8, 3, padding=1)
        self.head = DetectLike()

    def forward(self, x):
        return self.head(self.body(x))


class _Stub:
    """Stands in for the dataframe manager."""

    def enqueue_batch(self, **kwargs):
        pass

    def get_df_view(self, limit=-1, **_):
        return pd.DataFrame()


def projection_hooks(module):
    return [h for h in module._forward_hooks.values() if isinstance(h, pu._FeatureHook)]


class _Isolated(unittest.TestCase):
    def setUp(self):
        self._saved_frame = wl_src.DATAFRAME_M
        wl_src.DATAFRAME_M = _Stub()
        self.root = tempfile.mkdtemp(prefix="wl-projection-layer-")
        self._saved_root = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self.root
        detach_projection()
        clear_pending_restore()
        clear_registry()

    def tearDown(self):
        wl_src.DATAFRAME_M = self._saved_frame
        detach_projection()
        clear_pending_restore()
        clear_registry()
        os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        if self._saved_root is not None:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._saved_root
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# the auto-pick
# ---------------------------------------------------------------------------
class TestFrozenLayersAreNotTheHead(unittest.TestCase):

    def test_a_frozen_decoder_after_the_head_is_skipped(self):
        model = YoloLike()
        module, use_input, _ = pick_embedding_layer(model)
        self.assertIs(module, model.head.cls)
        self.assertTrue(use_input)

    def test_the_picked_layer_runs_in_a_training_forward(self):
        model = YoloLike().train()
        tracker = attach_projection(model, every_n_steps=1)
        try:
            model(th.randn(4, 3, 8, 8))
            self.assertEqual(tracker._pending.shape, (4, 8))
        finally:
            detach_projection()

    def test_a_model_frozen_whole_still_gets_a_pick(self):
        model = YoloLike().requires_grad_(False)
        module, _, _ = pick_embedding_layer(model)
        self.assertIs(module, model.head.dfl)

    def test_trainable_models_pick_as_before(self):
        model = TinyNet()
        self.assertIs(pick_embedding_layer(model)[0], model.classifier)


# ---------------------------------------------------------------------------
# signals logged from inside the caller's no_grad
# ---------------------------------------------------------------------------
class TestFitsUnderTheCallersGradMode(_Isolated):
    """Ultralytics ships every per-sample signal under ``no_grad``; the fit
    used to inherit it, fail five times, and turn the projection off."""

    def _run(self, mode):
        model = TinyNet()
        tracker = attach_projection(model, every_n_steps=1)
        for step in range(1, 4):
            model.train()
            model(th.randn(16, 16))
            with mode():
                observe_batch(list(range(16)), step)
        return tracker

    def test_fits_under_no_grad(self):
        tracker = self._run(th.no_grad)
        self.assertEqual(tracker.failures, 0)
        self.assertEqual(tracker.steps_trained, 3)
        self.assertGreater(tracker.samples_written, 0)

    def test_fits_under_inference_mode(self):
        tracker = self._run(th.inference_mode)
        self.assertEqual(tracker.failures, 0)
        self.assertEqual(tracker.steps_trained, 3)

    def test_features_captured_under_inference_mode_still_fit(self):
        model = TinyNet()
        tracker = attach_projection(model, every_n_steps=1)
        model.train()
        with th.inference_mode():
            model(th.randn(16, 16))
        observe_batch(list(range(16)), 1)
        self.assertEqual((tracker.failures, tracker.steps_trained), (0, 1))

    def test_the_callers_grad_mode_is_left_as_it_was(self):
        model = TinyNet()
        attach_projection(model, every_n_steps=1)
        model.train()
        model(th.randn(16, 16))
        with th.no_grad():
            observe_batch(list(range(16)), 1)
            self.assertFalse(th.is_grad_enabled())


# ---------------------------------------------------------------------------
# a copy made before the hook (Ultralytics' EMA shadow)
# ---------------------------------------------------------------------------
class TestMirror(_Isolated):

    def setUp(self):
        super().setUp()
        self.model = TinyNet()
        # Made BEFORE the attach, like UL's EMA: deepcopy cannot carry a hook
        # that does not exist yet.
        self.shadow = copy.deepcopy(self.model).eval()
        self.live = attach_projection(self.model, every_n_steps=1)

    def test_a_copy_made_before_the_attach_feeds_nothing_on_its_own(self):
        self.shadow(th.randn(5, 16))
        self.assertIsNone(self.live._pending)

    def test_a_mirrored_copy_feeds_the_live_tracker_as_eval(self):
        self.assertTrue(mirror_projection(self.shadow))
        self.shadow(th.randn(5, 16))
        self.assertEqual(self.live._pending.shape, (5, 8))
        self.assertFalse(self.live._pending_training)

    def test_mirroring_twice_adds_one_hook(self):
        mirror_projection(self.shadow)
        mirror_projection(self.shadow)
        self.assertEqual(len(projection_hooks(self.shadow.classifier)), 1)

    def test_detach_removes_the_mirror_too(self):
        mirror_projection(self.shadow)
        detach_projection()
        self.assertEqual(projection_hooks(self.shadow.classifier), [])

    def test_reattaching_the_primary_keeps_the_mirror(self):
        mirror_projection(self.shadow)
        self.live.attach(self.model.classifier, use_input=True, feature_last=True)
        self.assertEqual(len(projection_hooks(self.shadow.classifier)), 1)

    def test_a_copy_without_the_layer_is_refused(self):
        self.assertFalse(mirror_projection(nn.Sequential(nn.Linear(16, 3))))

    def test_nothing_to_mirror_without_a_live_projection(self):
        detach_projection()
        self.assertFalse(mirror_projection(self.shadow))


# ---------------------------------------------------------------------------
# recording the layer, and serving it to the board
# ---------------------------------------------------------------------------
def _frame(prefix="umap", n=32):
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_arrays(
        [np.arange(n), np.zeros(n, dtype=int)], names=["sample_id", "annotation_id"])
    return pd.DataFrame({f"signals//{prefix}_{a}": rng.normal(size=n) for a in "xyz"},
                        index=index)


class TestTheBoardIsToldTheLayer(_Isolated):

    def _live_fit(self):
        model = TinyNet()
        tracker = attach_projection(model, every_n_steps=1)
        model.train()
        model(th.randn(16, 16))
        observe_batch(list(range(16)), 1)
        return tracker

    def test_the_live_projection_records_its_layer(self):
        self._live_fit()
        self.assertEqual(prefix_info("umap"),
                         {"layer": "classifier", "layer_detail": "Linear input"})

    def test_the_layer_survives_a_restart(self):
        self._live_fit()
        detach_projection()
        clear_registry()                     # a new process: memory is empty
        self.assertEqual(prefix_info("umap")["layer"], "classifier")

    def test_the_response_names_the_live_layer(self):
        self._live_fit()
        response = ps.build_projection_response(_frame("umap"), pb2.ProjectionRequest(max_points=100))
        self.assertTrue(response.success)
        self.assertEqual(response.layer, "classifier")
        self.assertEqual(response.layer_detail, "Linear input")

    def test_the_live_tracker_wins_over_a_stale_registry_entry(self):
        tracker = self._live_fit()
        tracker.layer_name, tracker.layer_detail = "feat", "Linear input"
        self.assertEqual(ps.projection_layer("umap"), ("feat", "Linear input"))

    def test_your_own_projection_has_no_layer(self):
        save_projection_coords(np.ones((2, 3)), [1, 2], prefix="tsne")
        response = ps.build_projection_response(
            _frame("tsne"), pb2.ProjectionRequest(prefix="tsne", max_points=100))
        self.assertEqual((response.layer, response.layer_detail), ("", ""))

    def test_project_dataset_records_the_layer_it_read(self):
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(th.randn(16, 16), th.arange(16)), batch_size=8)
        wl_src.project_dataset(TinyNet(), loader, layer="feat", method=lambda f: f[:, :2],
                               prefix="mine", verbose=False)
        self.assertEqual(prefix_info("mine"), {"layer": "feat", "layer_detail": "Linear input"})

    def test_registering_without_a_layer_keeps_the_recorded_one(self):
        register_prefix("snap", layer="model.22", layer_detail="C3k2 output")
        register_prefix("snap")
        self.assertEqual(prefix_info("snap")["layer"], "model.22")

    def test_a_registry_file_from_before_layers_still_reads(self):
        folder = os.path.join(self.root, "projection")
        os.makedirs(folder)
        with open(os.path.join(folder, REGISTRY_FILE), "w", encoding="utf-8") as handle:
            json.dump(["umap", "tsne"], handle)
        self.assertEqual(known_prefixes(), {"umap", "tsne"})
        self.assertEqual(prefix_info("tsne"), {})
        register_prefix("pca", layer="feat")
        clear_registry()
        self.assertEqual(known_prefixes(), {"umap", "tsne", "pca"})
        self.assertEqual(prefix_info("pca"), {"layer": "feat"})

    def test_an_older_reader_still_finds_every_prefix(self):
        """Older versions build the set with ``for n in json.load(...)``; on
        the new object that iterates the keys."""
        register_prefix("umap", layer="classifier", layer_detail="Linear input")
        register_prefix("tsne")
        with open(os.path.join(self.root, "projection", REGISTRY_FILE), encoding="utf-8") as handle:
            self.assertEqual({str(n) for n in json.load(handle)}, {"umap", "tsne"})


# ---------------------------------------------------------------------------
# Ultralytics
# ---------------------------------------------------------------------------
# unittest's own skip, not pytest.importorskip: the release workflow runs this
# suite with `python -m unittest`, which reports pytest's Skipped as an error.
@unittest.skipUnless(importlib.util.find_spec("ultralytics") is not None,
                     "ultralytics not installed")
class TestUltralyticsLayer(_Isolated):

    def setUp(self):
        super().setUp()
        from ultralytics.nn.tasks import DetectionModel
        self.model = DetectionModel("yolo11n.yaml", nc=2, verbose=False)

    def test_the_auto_pick_avoids_the_dfl(self):
        module, _, _ = pick_embedding_layer(self.model)
        name = next(n for n, m in self.model.named_modules() if m is module)
        self.assertNotIn("dfl", name)

    def test_the_trainer_hooks_the_neck_output_and_it_trains(self):
        from weightslab.integrations.ultralytics.trainer import _embedding_layer
        layer = _embedding_layer(self.model)
        tracker = attach_projection(self.model, layer=layer, every_n_steps=1)
        self.assertEqual(tracker.layer_name, f"model.{len(self.model.model) - 2}")
        self.assertTrue(tracker.layer_detail.endswith("output"))

        self.model.train()
        self.model(th.randn(2, 3, 64, 64))
        self.assertIsNotNone(tracker._pending, "the hooked layer did not run in training")
        self.assertEqual(tracker._pending.shape[0], 2)
        self.assertTrue(tracker._pending_training)

    def test_the_ema_shadow_feeds_validation(self):
        from ultralytics.utils.torch_utils import ModelEMA
        from weightslab.integrations.ultralytics.trainer import _embedding_layer
        ema = ModelEMA(self.model)       # built before the hook, as UL does
        tracker = attach_projection(self.model, layer=_embedding_layer(self.model),
                                    every_n_steps=1)
        self.assertTrue(mirror_projection(ema.ema))
        with th.no_grad():
            ema.ema(th.randn(2, 3, 64, 64))
        self.assertIsNotNone(tracker._pending)
        self.assertFalse(tracker._pending_training)


if __name__ == "__main__":
    unittest.main()
