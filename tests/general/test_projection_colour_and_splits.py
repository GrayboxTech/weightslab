"""Colour source, and the eval split reaching the cloud at all.

Both of these were user-visible: the projection showed the train split only,
and colouring by class ran the class ids through a continuous ramp.
"""
import numpy as np
import pandas as pd

from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.services import projection_service as ps


def _frame(n=64, targets=None, origins=None):
    rng = np.random.default_rng(0)
    sid = np.arange(n)
    index = pd.MultiIndex.from_arrays(
        [sid, np.zeros(n, dtype=int)], names=["sample_id", "annotation_id"])
    return pd.DataFrame({
        "signals//umap_x": rng.normal(size=n),
        "signals//umap_y": rng.normal(size=n),
        "signals//umap_z": rng.normal(size=n),
        "target": np.arange(n) % 10 if targets is None else targets,
        "origin": (["train_loader"] * n) if origins is None else origins,
        "discarded": np.zeros(n, dtype=bool),
    }, index=index)


def _request(**kwargs):
    return pb2.ProjectionRequest(max_points=1000, **kwargs)


class TestColourSource:
    def test_class_ids_are_categorical_not_a_ramp(self):
        """A class id is a label, not a measurement.

        Through the continuous ramp, class 4 sits "between" 3 and 5 and
        neighbouring classes come out near-identical -- the one thing a
        projection coloured by class must not say.
        """
        response = ps.build_projection_response(_frame(), _request(color_column="target"))
        assert response.success
        assert len(response.color_labels) == response.returned
        assert not response.color_values

    def test_a_continuous_target_still_ramps(self):
        # 100 distinct values: past the low-cardinality threshold, so this is a
        # regression target and the ramp is the right reading of it.
        frame = _frame(n=100, targets=np.linspace(0.0, 1.0, 100))
        response = ps.build_projection_response(frame, _request(color_column="target"))
        assert response.success
        assert len(response.color_values) == response.returned
        assert not response.color_labels

    def test_colouring_by_split_is_left_to_the_client(self):
        """The Studio owns the split palette, so a sample is the same colour
        here as in the data grid. Sending labels would give the viewer a
        second, different palette for the same thing."""
        frame = _frame(origins=["train_loader"] * 32 + ["test_loader"] * 32)
        response = ps.build_projection_response(frame, _request(color_column="origin"))
        assert response.success
        assert not response.color_labels
        assert not response.color_values
        # ...and the split membership it needs to do that is still sent.
        assert sorted(response.origins) == ["test_loader", "train_loader"]

    def test_a_metadata_column_still_works(self):
        frame = _frame()
        frame["signals//loss"] = np.linspace(0.0, 2.0, len(frame))
        response = ps.build_projection_response(
            frame, _request(color_column="signals//loss"))
        assert response.success
        assert len(response.color_values) == response.returned


class TestBothSplitsAreServed:
    def test_test_split_points_are_returned_alongside_train(self):
        frame = _frame(origins=["train_loader"] * 40 + ["test_loader"] * 24)
        response = ps.build_projection_response(frame, _request())
        assert response.success
        names = {response.origins[i] for i in response.origin_ids}
        assert names == {"train_loader", "test_loader"}
        assert response.returned == 64


class TestEvalBatchesArePlaced:
    """The gate that kept the test split out of the cloud.

    ``every_n_steps`` rations FITTING, and only training batches fit. Applying
    it to eval batches too meant a validation pass -- which does not advance the
    step, so every batch of it arrives carrying the same number -- had its first
    batch rejected unless that step happened to be a multiple of every_n_steps,
    and all the rest rejected by ``step_i == self._last_step_run``.
    """

    def _tracker(self):
        from weightslab.projection.parametric_umap import ProjectionTracker
        tracker = ProjectionTracker(every_n_steps=50)
        # Enough of a hook for observe_batch to believe one is attached.
        tracker._handle = object()
        return tracker

    def test_an_eval_batch_is_not_rationed_by_every_n_steps(self):
        import torch as th
        tracker = self._tracker()
        placed = []
        tracker._write_coords = lambda ids, coords, step: placed.append(list(ids)) or True

        # A step that is NOT a multiple of every_n_steps, three times over, the
        # way a validation pass arrives.
        for _ in range(3):
            tracker._pending = th.randn(8, 16)
            tracker._pending_training = False
            tracker.observe_batch(list(range(8)), step=137)

        assert len(placed) == 3, (
            "every eval batch should be placed; only the first got through")

    def test_a_training_batch_is_still_rationed(self):
        import torch as th
        tracker = self._tracker()
        fits = []
        tracker._write_coords = lambda ids, coords, step: fits.append(list(ids)) or True

        tracker._pending = th.randn(8, 16)
        tracker._pending_training = True
        tracker.observe_batch(list(range(8)), step=137)   # 137 % 50 != 0
        assert not fits

        tracker._pending = th.randn(8, 16)
        tracker._pending_training = True
        tracker.observe_batch(list(range(8)), step=150)   # 150 % 50 == 0
        assert len(fits) == 1

    def test_an_eval_batch_does_not_lock_out_the_training_batch_at_that_step(self):
        """Only a fit claims the step. An eval batch that claimed it would shut
        out the training batch arriving at the same one."""
        import torch as th
        tracker = self._tracker()
        seen = []
        tracker._write_coords = lambda ids, coords, step: seen.append(list(ids)) or True

        tracker._pending = th.randn(8, 16)
        tracker._pending_training = False
        tracker.observe_batch(list(range(8)), step=150)

        tracker._pending = th.randn(8, 16)
        tracker._pending_training = True
        tracker.observe_batch(list(range(100, 108)), step=150)

        assert len(seen) == 2


class TestTheEncoderIsCheckpointedOncePerFit:
    """A full evaluation pass must not rewrite the encoder once per batch.

    The checkpoint is keyed on ``steps_trained``, which only advances on a fit
    -- but the write-back it lives in now also runs for eval batches, which are
    placed rather than learned from. With the counter sitting on a multiple of
    _SAVE_EVERY_FITS, every batch of an evaluation pass saved the same weights
    again: a burst of saves a second apart for one fit's worth of encoder.
    """

    def test_repeated_eval_batches_save_at_most_once(self, monkeypatch):
        import torch as th
        from weightslab.projection import parametric_umap as pu

        saves = []
        monkeypatch.setattr(pu, "save_projection", lambda tracker: saves.append(1))

        tracker = pu.ProjectionTracker(every_n_steps=50)
        tracker._handle = object()
        # Sitting exactly on a save boundary is the state that triggered it.
        tracker.steps_trained = pu._SAVE_EVERY_FITS
        # _write_coords imports it from weightslab.src at call time.
        import weightslab.src as wl_src
        monkeypatch.setattr(wl_src, "save_signals", lambda **kwargs: None)

        for _ in range(8):
            tracker._pending = th.randn(8, 16)
            tracker._pending_training = False
            tracker.observe_batch(list(range(8)), step=137)

        assert len(saves) <= 1, f"encoder written {len(saves)} times for one fit"

    def test_a_later_fit_still_checkpoints(self, monkeypatch):
        import torch as th
        from weightslab.projection import parametric_umap as pu

        saves = []
        monkeypatch.setattr(pu, "save_projection", lambda tracker: saves.append(1))
        import weightslab.src as wl_src
        monkeypatch.setattr(wl_src, "save_signals", lambda **kwargs: None)

        tracker = pu.ProjectionTracker(every_n_steps=50)
        tracker._handle = object()

        tracker.steps_trained = pu._SAVE_EVERY_FITS
        tracker._pending = th.randn(8, 16)
        tracker._pending_training = False
        tracker.observe_batch(list(range(8)), step=137)

        # The next save boundary is a different fit, and must still be written.
        tracker.steps_trained = pu._SAVE_EVERY_FITS * 2
        tracker._pending = th.randn(8, 16)
        tracker._pending_training = False
        tracker.observe_batch(list(range(8)), step=139)

        assert len(saves) == 2


class TestAuxiliaryHeadsAreNotHooked:
    """Which layer the projection reads, per task family.

    torchvision's FCN/DeepLab register ``aux_classifier`` AFTER ``classifier``,
    so "the last conv" chose the auxiliary head. That is wrong twice over: the
    aux head branches off an earlier backbone stage, so its input is a
    shallower representation than the one the model predicts from; and
    torchvision skips it outside training, so on an evaluation pass the hook
    produced nothing and the cloud silently stopped gaining test-split points.
    """

    def _picked(self, model):
        from weightslab.projection import pick_embedding_layer
        module, use_input, feature_last = pick_embedding_layer(model)
        name = next((n for n, m in model.named_modules() if m is module), "?")
        return name, module, use_input, feature_last

    def test_a_segmentation_model_with_an_aux_head_hooks_the_main_one(self):
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 4, 1)

            def forward(self, x):
                return self.conv(x)

        class SegWithAux(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv2d(3, 8, 3, padding=1)
                self.classifier = Head()
                # Registered last, exactly as torchvision does it.
                self.aux_classifier = Head()

            def forward(self, x):
                return self.classifier(self.backbone(x))

        name, _, use_input, feature_last = self._picked(SegWithAux())
        assert name == "classifier.conv", f"hooked {name}"
        assert use_input is True and feature_last is False

    def test_the_same_holds_for_the_mmseg_spelling(self):
        import torch.nn as nn

        class SegWithAux(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Conv2d(3, 8, 3, padding=1)
                self.decode_head = nn.Conv2d(8, 4, 1)
                self.auxiliary_head = nn.Conv2d(8, 4, 1)

            def forward(self, x):
                return self.decode_head(self.backbone(x))

        name, _, _, _ = self._picked(SegWithAux())
        assert name == "decode_head", f"hooked {name}"

    def test_a_model_that_is_only_an_aux_head_still_gets_hooked(self):
        """The filter is a preference, not a veto: a model with nothing but an
        auxiliary head must still project rather than fall through to a leaf."""
        import torch.nn as nn

        class OnlyAux(nn.Module):
            def __init__(self):
                super().__init__()
                self.aux_classifier = nn.Conv2d(3, 4, 1)

            def forward(self, x):
                return self.aux_classifier(x)

        name, module, _, _ = self._picked(OnlyAux())
        assert isinstance(module, nn.Conv2d)
        assert name == "aux_classifier"

    def test_the_zoo_heads_per_task_family(self):
        """The real models, by task: classification heads are Linear, the
        segmentation and detection heads are the last conv of the MAIN head."""
        import inspect
        import torch.nn as nn
        try:
            from weightslab.examples.utils.baseline_models.pytorch import models as zoo
        except Exception:
            import pytest
            pytest.skip("baseline model zoo unavailable")

        names = {n: o for n, o in vars(zoo).items()
                 if inspect.isclass(o) and issubclass(o, nn.Module)
                 and o.__module__ == zoo.__name__}
        expected = {
            # classification -> the classifier Linear
            "ResNet18": "model.fc",
            "FashionCNN": "fc4",
            # 2-D segmentation -> the output conv of the main head
            "UNet": "outc",
            "TinyUNet": "outc",
            "UNet3p": "outc",
            "FCNResNet50": "model.classifier.4",
            # 3-D segmentation
            "UNet3D": "out_conv",
            # detection
            "TinyYOLO": "head.1",
        }
        for model_name, want in expected.items():
            if model_name not in names:
                continue
            got, _, _, _ = self._picked(names[model_name]())
            assert got == want, f"{model_name}: hooked {got}, expected {want}"
