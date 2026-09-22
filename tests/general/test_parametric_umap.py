"""Parametric UMAP: the projection itself, and the view-box serving around it."""

import os
import unittest

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn

from weightslab import src as wl_src
from weightslab.projection import (
    ENV_ENABLED,
    ProjectionTracker,
    attach_projection,
    detach_projection,
    find_ab_params,
    first_tensor,
    flatten_features,
    get_tracker,
    membership_high_dim,
    pick_embedding_layer,
    projection_enabled,
)
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.services import projection_service as ps


class TinyNet(nn.Module):
    """conv-ish trunk -> features -> classifier, so the auto-picked layer has
    an obvious right answer (the classifier's INPUT, i.e. `feat`'s output)."""

    def __init__(self, in_dim=16, feat_dim=8, classes=3):
        super().__init__()
        self.feat = nn.Linear(in_dim, feat_dim)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(feat_dim, classes)

    def forward(self, x):
        return self.classifier(self.act(self.feat(x)))


def clustered(n_per=40, dim=16, clusters=3, seed=0):
    """Well-separated blobs -- structure the projection is expected to keep."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(clusters, dim))
    xs, ys = [], []
    for c in range(clusters):
        xs.append(centers[c] + rng.normal(scale=0.35, size=(n_per, dim)))
        ys.append(np.full(n_per, c))
    return (th.tensor(np.concatenate(xs), dtype=th.float32),
            np.concatenate(ys))


class TestAbParams(unittest.TestCase):
    def test_matches_umap_learn_reference(self):
        a, b = find_ab_params(spread=1.0, min_dist=0.1)
        # umap-learn's scipy fit for these inputs.
        self.assertAlmostEqual(a, 1.5769434, delta=0.08)
        self.assertAlmostEqual(b, 0.8950608, delta=0.05)

    def test_smaller_min_dist_tightens_the_curve(self):
        a_tight, _ = find_ab_params(spread=1.0, min_dist=0.0)
        a_loose, _ = find_ab_params(spread=1.0, min_dist=0.5)
        self.assertGreater(a_tight, a_loose)


class TestMembership(unittest.TestCase):
    def test_rows_sum_to_log2_k_before_symmetrisation(self):
        feats, _ = clustered(n_per=20)
        k = 10
        # Re-derive the one-sided memberships the way the function does, since
        # the symmetrised output deliberately no longer sums to log2(k).
        dist = th.cdist(feats, feats)
        dist = dist + th.eye(len(feats)) * 1e12
        knn_d, _ = th.topk(dist, k, dim=1, largest=False)
        mu = membership_high_dim(feats, k)
        self.assertEqual(mu.shape, (len(feats), len(feats)))
        # Symmetric, in [0,1], zero on the diagonal.
        self.assertTrue(th.allclose(mu, mu.t(), atol=1e-6))
        self.assertGreaterEqual(float(mu.min()), 0.0)
        self.assertLessEqual(float(mu.max()), 1.0 + 1e-6)
        self.assertLess(float(mu.diagonal().abs().max()), 1e-6)

    def test_near_neighbours_score_above_far_ones(self):
        feats, labels = clustered(n_per=25, clusters=3)
        mu = membership_high_dim(feats, 15).numpy()
        same = np.equal.outer(labels, labels)
        np.fill_diagonal(same, False)
        self.assertGreater(mu[same].mean(), mu[~same].mean())


class TestTracker(unittest.TestCase):
    def test_fitting_preserves_cluster_structure(self):
        """The point of the whole feature: separated in feature space should
        come out separated in 3-D."""
        th.manual_seed(0)
        feats, labels = clustered(n_per=40, dim=16, clusters=3)
        tracker = ProjectionTracker(out_dim=3, n_neighbors=12, inner_steps=1)
        tracker._ensure_encoder(feats.shape[1], feats.device)
        target = membership_high_dim(feats, 12)

        first = None
        for step in range(220):
            tracker._optimizer.zero_grad(set_to_none=True)
            loss = tracker._umap_loss(feats, target)
            loss.backward()
            tracker._optimizer.step()
            if step == 0:
                first = float(loss)
        self.assertLess(float(loss), first)

        with th.no_grad():
            emb = tracker._encoder(feats).numpy()
        within, between = [], []
        for i in range(3):
            centroid = emb[labels == i].mean(axis=0)
            within.append(np.linalg.norm(emb[labels == i] - centroid, axis=1).mean())
            for j in range(i + 1, 3):
                between.append(np.linalg.norm(centroid - emb[labels == j].mean(axis=0)))
        self.assertGreater(np.mean(between), 2.0 * np.mean(within))

    def test_hook_captures_penultimate_features_not_logits(self):
        model = TinyNet(in_dim=16, feat_dim=8, classes=3)
        layer, use_input, feature_last = pick_embedding_layer(model)
        self.assertIs(layer, model.classifier)
        self.assertTrue(use_input)
        self.assertTrue(feature_last)

        tracker = ProjectionTracker()
        tracker.attach(layer, use_input=use_input, feature_last=feature_last)
        try:
            model(th.randn(5, 16))
            self.assertIsNotNone(tracker._pending)
            # 8 (features), not 3 (classes).
            self.assertEqual(tuple(tracker._pending.shape), (5, 8))
        finally:
            tracker.detach()

    def test_hook_never_backpropagates_into_the_model(self):
        """Instrumentation must not touch the training graph."""
        model = TinyNet()
        tracker = ProjectionTracker()
        tracker.attach(model.classifier, use_input=True)
        try:
            model(th.randn(6, 16))
            self.assertFalse(tracker._pending.requires_grad)
        finally:
            tracker.detach()

    def test_conv_shaped_features_are_pooled_to_2d(self):
        conv = nn.Conv2d(3, 7, 3, padding=1)
        tracker = ProjectionTracker()
        tracker.attach(conv, use_input=False)
        try:
            conv(th.randn(4, 3, 8, 8))
            self.assertEqual(tuple(tracker._pending.shape), (4, 7))
        finally:
            tracker.detach()

    def test_mismatched_batch_ids_are_refused(self):
        """Pairing is positional; a length mismatch would silently attach one
        sample's features to another's id, so it must be dropped."""
        model = TinyNet()
        tracker = ProjectionTracker(every_n_steps=1)
        tracker.attach(model.classifier)
        try:
            model(th.randn(8, 16))
            self.assertFalse(tracker.observe_batch(["a", "b"], step=1))
        finally:
            tracker.detach()


class TestSaveSignalsCoercion(unittest.TestCase):
    """save_signals must never create a column it then writes nothing into."""

    class _Stub:
        def __init__(self):
            self.calls = []

        def enqueue_batch(self, **kwargs):
            self.calls.append(kwargs)

    def setUp(self):
        self._saved = wl_src.DATAFRAME_M
        self.stub = self._Stub()
        wl_src.DATAFRAME_M = self.stub

    def tearDown(self):
        wl_src.DATAFRAME_M = self._saved

    def test_numpy_arrays_reach_the_dataframe(self):
        """The regression that motivated this: a bare ndarray used to normalize
        to None, so the column appeared and stayed empty forever."""
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        wl_src.save_signals(signals={"metric": values}, batch_ids=[1, 2, 3], step=0)
        losses = self.stub.calls[0]["losses"]
        self.assertIn("signals//metric", losses)
        self.assertIsNotNone(losses["signals//metric"])
        np.testing.assert_allclose(losses["signals//metric"], values)

    def test_tensors_still_reach_the_dataframe(self):
        wl_src.save_signals(signals={"metric": th.tensor([1.0, 2.0])},
                            batch_ids=[1, 2], step=0)
        np.testing.assert_allclose(
            self.stub.calls[0]["losses"]["signals//metric"], [1.0, 2.0])

    def test_lists_still_reach_the_dataframe(self):
        wl_src.save_signals(signals={"metric": [1.0, 2.0]}, batch_ids=[1, 2], step=0)
        self.assertIsNotNone(self.stub.calls[0]["losses"]["signals//metric"])

    def test_multidim_values_reduce_to_one_per_sample(self):
        wl_src.save_signals(signals={"metric": np.ones((3, 4), dtype=np.float32)},
                            batch_ids=[1, 2, 3], step=0)
        self.assertEqual(self.stub.calls[0]["losses"]["signals//metric"].shape, (3,))

    def test_unsupported_type_raises_instead_of_writing_nothing(self):
        with self.assertRaises(TypeError) as ctx:
            wl_src.save_signals(signals={"metric": {"not": "a signal"}},
                                batch_ids=[1, 2], step=0)
        message = str(ctx.exception)
        self.assertIn("metric", message)        # names the offending signal
        self.assertIn("batch_ids", message)     # says what a good value is
        self.assertEqual(self.stub.calls, [])   # nothing half-written


class TestAnyTorchModel(unittest.TestCase):
    """The hook has to land on a usable (B, F) embedding for whatever the user
    actually wrapped -- not just for a toy classifier."""

    def test_flatten_keeps_the_feature_axis_for_each_layout(self):
        # Linear INPUT: features are the last axis whatever the rank, because
        # that is the axis Linear acts on.
        self.assertEqual(
            tuple(flatten_features(th.randn(4, 7, 16), feature_last=True).shape), (4, 16))
        self.assertEqual(
            tuple(flatten_features(th.randn(4, 16), feature_last=True).shape), (4, 16))
        # Conv OUTPUT: channels-first, so pool the spatial axes.
        self.assertEqual(
            tuple(flatten_features(th.randn(4, 16, 8, 8), feature_last=False).shape), (4, 16))
        self.assertEqual(
            tuple(flatten_features(th.randn(4, 16, 8, 8, 3), feature_last=False).shape), (4, 16))

    def test_first_tensor_unwraps_real_model_outputs(self):
        tensor = th.randn(2, 3)
        self.assertIs(first_tensor(tensor), tensor)
        self.assertIs(first_tensor((tensor, None)), tensor)
        self.assertIs(first_tensor([None, [tensor]]), tensor)
        self.assertIs(first_tensor({"last_hidden_state": tensor}), tensor)
        self.assertIs(first_tensor({"anything": tensor}), tensor)
        self.assertIsNone(first_tensor({"nope": "text"}))

    def _assert_projects(self, model, sample_input, expected_dim):
        layer, use_input, feature_last = pick_embedding_layer(model)
        self.assertIsNotNone(layer)
        tracker = ProjectionTracker(every_n_steps=1)
        tracker.attach(layer, use_input=use_input, feature_last=feature_last)
        try:
            model(sample_input)
            self.assertIsNotNone(tracker._pending, "hook captured nothing")
            self.assertEqual(tracker._pending.ndim, 2)
            self.assertEqual(tracker._pending.shape[0], sample_input.shape[0])
            self.assertEqual(tracker._pending.shape[1], expected_dim)
        finally:
            tracker.detach()

    def test_tabular_mlp(self):
        model = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 2))
        self._assert_projects(model, th.randn(8, 12), 32)

    def test_vision_cnn(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(8, 5))
        self._assert_projects(model, th.randn(6, 3, 16, 16), 8)

    def test_sequence_model_pools_tokens_not_channels(self):
        """(B, T, C) into the head: keep C, pool T. Getting this backwards
        would hand the projection a per-token scalar."""
        class Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.TransformerEncoderLayer(
                    d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
                self.head = nn.Linear(32, 3)

            def forward(self, x):
                return self.head(self.encoder(x))

        self._assert_projects(Seq(), th.randn(5, 9, 32), 32)

    def test_fully_convolutional_segmentation_head(self):
        """No Linear anywhere: the last CONV is the head, and its INPUT is the
        representation. Reading its output would give 4 class-logit channels --
        well-shaped, and completely the wrong thing to lay out."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 4, 1))
        layer, use_input, feature_last = pick_embedding_layer(model)
        self.assertIsInstance(layer, nn.Conv2d)
        self.assertTrue(use_input, "must read the conv head's INPUT, not its logits")
        self.assertFalse(feature_last, "conv activations are channels-first")
        # 16 (decoder features), not 4 (classes).
        self._assert_projects(model, th.randn(4, 3, 12, 12), 16)

    def test_model_returning_a_tuple(self):
        class TupleOut(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Linear(8, 16)
                self.head = nn.Linear(16, 2)

            def forward(self, x):
                hidden = self.trunk(x)
                return self.head(hidden), hidden

        self._assert_projects(TupleOut(), th.randn(3, 8), 16)

    def test_eval_batches_are_placed_but_not_fit(self):
        """Held-out samples belong in the picture; letting them shape the
        projection would leak the eval set into it."""
        model = TinyNet()
        tracker = ProjectionTracker(every_n_steps=1)
        tracker.attach(model.classifier)
        try:
            model.train()
            model(th.randn(8, 16))
            tracker.observe_batch(list(range(8)), step=1)
            fits_after_train = tracker.steps_trained
            self.assertGreater(fits_after_train, 0)

            model.eval()
            with th.no_grad():
                model(th.randn(8, 16))
            tracker.observe_batch(list(range(8, 16)), step=2)
            self.assertEqual(tracker.steps_trained, fits_after_train)
        finally:
            tracker.detach()

    def test_survives_a_live_architecture_edit(self):
        """Editing a running model's width is a headline WeightsLab feature. An
        encoder pinned to the old width raises on every batch afterwards, and
        the broad except around the fit turns that into a projection that
        silently stops updating from the user's first prune onward."""
        model = TinyNet(in_dim=16, feat_dim=8)
        tracker = ProjectionTracker(every_n_steps=1)
        tracker.attach(model.classifier)
        try:
            model(th.randn(8, 16))
            tracker.observe_batch(list(range(8)), step=1)
            self.assertEqual(tracker.steps_trained, 1)
            self.assertEqual(tracker.stats()["feature_dim"], 8)

            # Prune the feature layer 8 -> 5, the way a neuron op does: the
            # module object stays, its parameters are replaced.
            model.feat = nn.Linear(16, 5)
            model.classifier.weight = nn.Parameter(th.randn(3, 5))
            model.classifier.in_features = 5

            model(th.randn(8, 16))
            tracker.observe_batch(list(range(8)), step=2)
            self.assertEqual(tracker.steps_trained, 2, "projection died after the edit")
            self.assertEqual(tracker.stats()["feature_dim"], 5)
            self.assertEqual(tracker.rebuilds, 1)
        finally:
            tracker.detach()

    def test_hook_survives_a_parameter_swap(self):
        """The hook is on the module, and WeightsLab's ops replace parameters
        rather than modules -- so a prune must not need a re-attach."""
        model = TinyNet()
        tracker = ProjectionTracker()
        tracker.attach(model.classifier)
        try:
            model.classifier.weight = nn.Parameter(th.randn(3, 8))
            tracker._pending = None
            model(th.randn(4, 16))
            self.assertIsNotNone(tracker._pending)
        finally:
            tracker.detach()

    def test_encoder_follows_the_model_to_a_new_device(self):
        """.cuda() after wrapping must not strand the encoder on the CPU."""
        tracker = ProjectionTracker()
        tracker._ensure_encoder(8, th.device("cpu"))
        first = tracker._encoder
        tracker._ensure_encoder(8, th.device("cpu"))
        self.assertIs(tracker._encoder, first, "same width must not rebuild")
        self.assertEqual(tracker.rebuilds, 0)

    def test_write_back_does_not_recurse(self):
        """save_signals calls observe_batch, and the write-back calls
        save_signals -- the guard is what stops that looping."""
        tracker = ProjectionTracker(every_n_steps=1)
        tracker._writing_back = True
        self.assertFalse(tracker.observe_batch([1, 2, 3, 4], step=1))


class TestPerInstancePairing(unittest.TestCase):
    """Detection / segmentation: the loss wrapper rebinds batch_ids to one entry
    per ANNOTATION, but the forward hook saw one feature row per SAMPLE. The
    projection must be handed the sample-level ids, or every detection run
    silently projects nothing (the length check rejects the mismatch)."""

    def setUp(self):
        from weightslab.backend import ledgers
        self.ledgers = ledgers
        self.observed = []

        # Two samples, three annotations between them -- the layout the
        # per-instance branch reads back out of the dataframe.
        frame = pd.DataFrame(
            {"x": [0, 1, 2, 3, 4]},
            index=pd.MultiIndex.from_tuples(
                [(7, 0), (7, 1), (7, 2), (9, 0), (9, 1)],
                names=["sample_id", "annotation_id"]),
        )

        class _DfStub:
            _df = frame

        self._saved_get_df = ledgers.get_dataframe
        self._saved_observe = wl_src._projection_observe_batch
        self._saved_instance = wl_src.save_instance_signals
        ledgers.get_dataframe = lambda *a, **k: _DfStub()
        wl_src._projection_observe_batch = lambda ids, step: self.observed.append(ids)
        wl_src.save_instance_signals = lambda **kwargs: None

    def tearDown(self):
        self.ledgers.get_dataframe = self._saved_get_df
        wl_src._projection_observe_batch = self._saved_observe
        wl_src.save_instance_signals = self._saved_instance

    def test_projection_receives_sample_ids_not_annotation_ids(self):
        sample_ids = [7, 9]

        def fake_forward(preds, targets):
            return th.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # one value per instance

        wl_src.wrappered_fwd(
            fake_forward,
            {"per_sample": False, "per_instance": True, "log": False},
            "det_loss",
            th.randn(2, 4), th.randn(2, 4),
            batch_ids=sample_ids,
        )

        self.assertEqual(len(self.observed), 1)
        # The sample ids, exactly as the hook saw them -- NOT the five
        # annotation-level ids the per-instance branch rebinds batch_ids to.
        self.assertEqual(list(self.observed[0]), sample_ids)


class TestOfflineReprojection(unittest.TestCase):
    """``wl.project_dataset`` -- re-projecting a trained model from a different
    layer, with no training step involved. The answer to "the live projection
    picked the wrong layer"."""

    class _Stub:
        def __init__(self):
            self.calls = []

        def enqueue_batch(self, **kwargs):
            self.calls.append(kwargs)

    def setUp(self):
        from torch.utils.data import DataLoader, TensorDataset
        self._saved = wl_src.DATAFRAME_M
        self.stub = self._Stub()
        wl_src.DATAFRAME_M = self.stub
        detach_projection()   # no live tracker unless a test attaches one

        feats, labels = clustered(n_per=24, dim=16, clusters=4)
        self.loader = DataLoader(
            TensorDataset(feats, th.arange(len(feats)), th.tensor(labels)),
            batch_size=16)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.early = nn.Sequential(nn.Linear(16, 12), nn.ReLU())
                self.late = nn.Sequential(nn.Linear(12, 9), nn.ReLU())
                self.classifier = nn.Linear(9, 4)

            def forward(self, x):
                return self.classifier(self.late(self.early(x)))

        self.model = Net()

    def tearDown(self):
        wl_src.DATAFRAME_M = self._saved
        detach_projection()

    def _written(self, prefix):
        return {k for call in self.stub.calls for k in call["losses"]
                if k.startswith(f"signals//{prefix}_")}

    def test_projects_without_any_training_step(self):
        stats = wl_src.project_dataset(self.model, self.loader, epochs=2, verbose=False)
        self.assertEqual(stats["samples"], 96)
        self.assertEqual(stats["feature_dim"], 9)       # classifier's input
        self.assertEqual(stats["prefix"], "umap")
        self.assertEqual(
            self._written("umap"),
            {"signals//umap_x", "signals//umap_y", "signals//umap_z"})

    def test_named_layer_overrides_the_auto_pick(self):
        stats = wl_src.project_dataset(
            self.model, self.loader, layer="late.0", epochs=2, verbose=False)
        # late.0 is Linear(12, 9) -> its INPUT is 12-wide, not the auto-picked 9.
        self.assertEqual(stats["feature_dim"], 12)

    def test_prefixes_coexist_so_layers_can_be_compared(self):
        """The point of re-projecting: keep both and switch between them."""
        wl_src.project_dataset(self.model, self.loader, epochs=1, verbose=False)
        wl_src.project_dataset(self.model, self.loader, layer="early.0",
                               prefix="umap_early", epochs=1, verbose=False)
        self.assertTrue(self._written("umap"))
        self.assertTrue(self._written("umap_early"))
        self.assertIn("signals//umap_early_x", self._written("umap_early"))

    def test_model_weights_and_mode_are_untouched(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train()
        wl_src.project_dataset(self.model, self.loader, epochs=2, verbose=False)
        self.assertTrue(self.model.training, "training mode must be restored")
        for key, value in self.model.state_dict().items():
            self.assertTrue(th.equal(before[key], value), f"{key} was modified")

    def test_no_hook_left_behind(self):
        wl_src.project_dataset(self.model, self.loader, epochs=1, verbose=False)
        self.assertEqual(len(self.model.classifier._forward_hooks), 0)

    def test_unknown_layer_name_lists_what_is_available(self):
        with self.assertRaises(ValueError) as ctx:
            wl_src.project_dataset(self.model, self.loader, layer="nope")
        self.assertIn("nope", str(ctx.exception))
        self.assertIn("classifier", str(ctx.exception))

    def test_layer_off_the_forward_path_says_so(self):
        self.model.orphan = nn.Linear(4, 4)  # never called in forward
        with self.assertRaises(RuntimeError) as ctx:
            wl_src.project_dataset(self.model, self.loader, layer="orphan",
                                   epochs=1, verbose=False)
        self.assertIn("forward path", str(ctx.exception))

    def test_accepts_the_wrapped_handle_users_actually_hold(self):
        """``wl.watch_or_edit(net, flag="model")`` returns a ledger Proxy, not an
        nn.Module -- and that is what every training script keeps a reference to.
        A layer lookup against it used to report that the model had no layers at
        all, which is a baffling thing to be told about a model you can see."""
        class FakeProxy:
            def __init__(self, inner):
                self.model = inner

            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        proxy = FakeProxy(self.model)
        self.assertEqual(list(getattr(proxy, "named_modules", lambda: [])()), [])

        stats = wl_src.project_dataset(proxy, self.loader, layer="late.0",
                                       epochs=1, verbose=False)
        self.assertEqual(stats["feature_dim"], 12)

    def test_resolve_module_unwraps_and_passes_through(self):
        from weightslab.projection import resolve_module

        class Wrapper:
            def __init__(self, inner):
                self.model = inner

        self.assertIs(resolve_module(self.model), self.model)
        self.assertIs(resolve_module(Wrapper(self.model)), self.model)
        self.assertIs(resolve_module(Wrapper(Wrapper(self.model))), self.model)
        self.assertIsNone(resolve_module(object()))

    def test_offline_then_resume_adopts_by_default_on_a_prefix_collision(self):
        """Writing to the prefix the LIVE projection owns and then resuming
        training would overwrite these coordinates sample by sample with a
        freshly-initialised encoder -- half this layout, half another, in one
        cloud. The default is therefore to hand the fit over."""
        attach_projection(self.model, every_n_steps=1)
        live = get_tracker()
        try:
            self.assertIsNone(live._encoder, "live encoder should not exist yet")
            stats = wl_src.project_dataset(
                self.model, self.loader, layer="late.0", epochs=1, verbose=False)

            self.assertTrue(stats["adopted_by_live"])
            # The live projection now owns the offline encoder AND is hooked to
            # the layer it was fitted on -- adopting one without the other just
            # triggers a rebuild and throws the fit away.
            self.assertIsNotNone(live._encoder)
            self.assertEqual(live._encoder_dim, 12)
            self.assertEqual(live.signal_prefix, "umap")

            # Resuming training refines that same encoder instead of restarting.
            fits_before = live.steps_trained
            for step, (x, ids, _) in enumerate(self.loader, 1):
                self.model(x)
                live.observe_batch(ids, step=step)
            self.assertGreater(live.steps_trained, fits_before)
            self.assertEqual(live._encoder_dim, 12, "rebuilt instead of continuing")
        finally:
            detach_projection()

    def test_a_distinct_prefix_leaves_the_live_projection_alone(self):
        """A snapshot under its own name is never written again, so there is
        nothing to hand over."""
        attach_projection(self.model, every_n_steps=1)
        live = get_tracker()
        try:
            stats = wl_src.project_dataset(
                self.model, self.loader, layer="late.0", prefix="umap_late",
                epochs=1, verbose=False)
            self.assertFalse(stats["adopted_by_live"])
            self.assertIsNone(live._encoder, "live projection must be untouched")
            self.assertEqual(live.signal_prefix, "umap")
        finally:
            detach_projection()

    def test_adopt_false_on_a_collision_warns_instead_of_corrupting_silently(self):
        attach_projection(self.model, every_n_steps=1)
        try:
            with self.assertLogs("weightslab.projection.parametric_umap",
                                 level="WARNING") as captured:
                stats = wl_src.project_dataset(
                    self.model, self.loader, epochs=1, verbose=False, adopt=False)
            self.assertFalse(stats["adopted_by_live"])
            message = "\n".join(captured.output)
            self.assertIn("overwrite", message)
            self.assertIn("adopt=True", message)
        finally:
            detach_projection()

    def test_adopt_true_switches_the_live_projection_to_a_new_prefix(self):
        attach_projection(self.model, every_n_steps=1)
        live = get_tracker()
        try:
            stats = wl_src.project_dataset(
                self.model, self.loader, layer="early.0", prefix="umap_early",
                epochs=1, verbose=False, adopt=True)
            self.assertTrue(stats["adopted_by_live"])
            self.assertEqual(live.signal_prefix, "umap_early")
            self.assertEqual(live._encoder_dim, 16)  # early.0's input width
        finally:
            detach_projection()

    @unittest.skipIf(not th.cuda.is_available(), "needs a second device")
    def test_adoption_across_devices_keeps_fitting(self):
        """An adopted encoder can arrive on another device. Moving it with
        .to() leaves Adam's state behind, so step() then raises on every batch
        -- swallowed by observe_batch's broad except into a projection that
        silently stops. The optimizer has to be rebuilt with it."""
        model = self.model.cuda()
        attach_projection(model, every_n_steps=1)
        live = get_tracker()
        try:
            wl_src.project_dataset(model, self.loader, layer="late.0",
                                   epochs=1, verbose=False, device="cpu")
            self.assertIsNotNone(live._encoder)
            fits_before = live.steps_trained
            for step, (x, ids, _) in enumerate(self.loader, 1):
                model(x.cuda())
                live.observe_batch(ids, step=step)
                break
            self.assertGreater(live.steps_trained, fits_before,
                               "adopted encoder stopped fitting after the move")
            self.assertEqual(
                next(live._encoder.parameters()).device.type,
                live._optimizer.param_groups[0]["params"][0].device.type)
        finally:
            detach_projection()
            self.model = model.cpu()

    def test_no_live_projection_means_nothing_to_adopt(self):
        detach_projection()
        stats = wl_src.project_dataset(self.model, self.loader, epochs=1,
                                       verbose=False)
        self.assertFalse(stats["adopted_by_live"])

    def test_max_samples_caps_collection(self):
        stats = wl_src.project_dataset(self.model, self.loader, epochs=1,
                                       max_samples=32, verbose=False)
        self.assertEqual(stats["samples"], 32)

    def test_offline_fit_preserves_cluster_structure(self):
        """Same bar the live path is held to, on the offline fitter."""
        feats, labels = clustered(n_per=40, dim=16, clusters=3)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(
            TensorDataset(feats, th.arange(len(feats)), th.tensor(labels)),
            batch_size=60)
        model = nn.Sequential(nn.Identity(), nn.Linear(16, 4))
        wl_src.project_dataset(model, loader, epochs=40, fit_batch=120,
                               verbose=False)

        coords, ids = [], []
        for call in self.stub.calls:
            if "signals//umap_x" not in call["losses"]:
                continue
            coords.append(np.stack(
                [call["losses"][f"signals//umap_{a}"] for a in "xyz"], axis=1))
            ids.extend(int(s) for s in call["sample_ids"])
        coords = np.concatenate(coords)
        got = labels[np.array(ids)]

        centroids = np.stack([coords[got == c].mean(axis=0) for c in range(3)])
        within = np.mean([np.linalg.norm(coords[got == c] - centroids[c], axis=1).mean()
                          for c in range(3)])
        between = np.mean([np.linalg.norm(centroids[i] - centroids[j])
                           for i in range(3) for j in range(i + 1, 3)])
        self.assertGreater(between, 2.0 * within)


class TestFollowView(unittest.TestCase):
    """Whose filter the projection obeys.

    A filter from the DATA side (typed, or asked of the agent) should narrow
    the cloud with it. A filter the projection itself produced must not, or a
    lasso collapses the view to the handful of points just picked and there is
    nothing left to select from.
    """

    def _frame(self, n=200):
        rng = np.random.default_rng(3)
        index = pd.MultiIndex.from_arrays(
            [np.where(np.arange(n) % 4 == 0, "test_loader", "train_loader"),
             np.arange(n).astype(str)],
            names=["origin", "sample_id"])
        return pd.DataFrame(
            {f"signals//{a}": rng.random(n) for a in
             ("umap_x", "umap_y", "umap_z")}, index=index)

    def test_request_carries_the_flag(self):
        req = pb2.ProjectionRequest(follow_view=True)
        self.assertTrue(req.follow_view)
        self.assertFalse(pb2.ProjectionRequest().follow_view)

    def test_serving_is_indifferent_to_the_flag(self):
        """The flag selects WHICH frame data_service hands over; the builder
        itself just renders whatever it is given, either way."""
        frame = self._frame(120)
        for follow in (True, False):
            resp = ps.build_projection_response(
                frame, pb2.ProjectionRequest(follow_view=follow))
            self.assertTrue(resp.success)
            self.assertEqual(resp.total_available, 120)

    def test_a_filtered_frame_yields_a_filtered_cloud(self):
        """What "the projection matches the current view" actually means: hand
        it the narrowed frame and only those points come back."""
        full = self._frame(200)
        narrowed = full.iloc[:40]
        wide = ps.build_projection_response(full, pb2.ProjectionRequest(follow_view=True))
        thin = ps.build_projection_response(narrowed, pb2.ProjectionRequest(follow_view=True))
        self.assertEqual(wide.total_available, 200)
        self.assertEqual(thin.total_available, 40)
        self.assertTrue(set(thin.sample_ids).issubset(set(wide.sample_ids)))

    def test_discarded_flags_are_sent_for_greying(self):
        frame = self._frame(50)
        frame["discarded"] = [i % 5 == 0 for i in range(50)]
        resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertEqual(len(resp.discarded), 50)
        self.assertEqual(sum(resp.discarded), 10)

    def test_target_colours_points_without_being_asked(self):
        """On a classification set the classes ARE the clusters; colouring by
        split paints the whole train set one colour and says nothing."""
        frame = self._frame(60)
        frame["target"] = np.arange(60) % 10
        resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertEqual(len(resp.color_values) + len(resp.color_labels), 60)


class TestStratifiedDecimation(unittest.TestCase):
    """No cluster may vanish because the budget is small."""

    def _blobs(self, sizes):
        ids, groups = [], []
        for name, count in sizes.items():
            ids.extend(f"{name}-{i}" for i in range(count))
            groups.extend([name] * count)
        return np.array(ids), np.array(groups)

    def test_small_clusters_keep_their_floor(self):
        """A flat global sample gives a 200-point cluster ~2 points out of a
        100k cloud -- it reads as noise, or disappears, and the user concludes
        the model has no such cluster."""
        sizes = {"big": 100_000, "mid": 5_000, "tiny": 200}
        ids, groups = self._blobs(sizes)

        budget = 20_000                      # >= 10% of the cloud, so every
        keep = ps.stratified_keep(ids, groups, budget)   # floor is payable
        self.assertLessEqual(len(keep), budget)

        kept = groups[keep]
        for name, count in sizes.items():
            got = int((kept == name).sum())
            floor = int(np.ceil(ps.MIN_GROUP_FRACTION * count))
            self.assertGreaterEqual(
                got, min(floor, count),
                f"{name}: kept {got} of {count}, below the 10% floor")

    def test_tight_budget_still_protects_the_small_clusters(self):
        """When 10% of everything exceeds the budget the floor cannot hold for
        every group -- the big one absorbs the shortfall, the small ones keep
        their share. Losing a little resolution on a 100k blob costs nothing;
        losing a 200-point cluster loses the finding."""
        sizes = {"big": 100_000, "mid": 5_000, "tiny": 200}
        ids, groups = self._blobs(sizes)
        budget = 5_000
        keep = ps.stratified_keep(ids, groups, budget)
        self.assertLessEqual(len(keep), budget)

        kept = groups[keep]
        for name in ("tiny", "mid"):
            got = int((kept == name).sum())
            floor = int(np.ceil(ps.MIN_GROUP_FRACTION * sizes[name]))
            self.assertGreaterEqual(got, floor, f"{name} lost its floor")
        self.assertGreater(int((kept == "big").sum()), 0)

    def test_nothing_to_do_under_budget(self):
        ids = np.array([f"s{i}" for i in range(50)])
        groups = np.array(["a"] * 25 + ["b"] * 25)
        keep = ps.stratified_keep(ids, groups, 100)
        self.assertEqual(len(keep), 50)

    def test_more_groups_than_budget_never_overshoots(self):
        """500 groups into a 400-point budget: not every group can appear, and
        quietly returning 500 points would blow the budget the client told us
        it could draw. Spend it across as many groups as it holds."""
        ids = np.array([f"s{i}" for i in range(2000)])
        groups = np.array([f"g{i % 500}" for i in range(2000)])
        keep = ps.stratified_keep(ids, groups, 400)
        self.assertLessEqual(len(keep), 400)
        self.assertGreater(len(np.unique(groups[keep])), 300)

    def test_selection_is_deterministic(self):
        ids = np.array([f"s{i}" for i in range(5000)])
        groups = np.array([f"g{i % 7}" for i in range(5000)])
        a = ps.stratified_keep(ids, groups, 500)
        b = ps.stratified_keep(ids, groups, 500)
        np.testing.assert_array_equal(a, b)

    def test_a_bigger_budget_only_adds_points(self):
        """Zooming in raises the effective budget per region; a point on screen
        must stay on screen."""
        ids = np.array([f"s{i}" for i in range(5000)])
        groups = np.array([f"g{i % 7}" for i in range(5000)])
        small = set(ps.stratified_keep(ids, groups, 300).tolist())
        large = set(ps.stratified_keep(ids, groups, 1200).tolist())
        self.assertTrue(small.issubset(large))

    def test_grouping_prefers_the_colour_column(self):
        n = 60
        frame = pd.DataFrame(
            {"target": np.arange(n) % 3, "origin": ["train_loader"] * n},
            index=pd.Index(np.arange(n).astype(str), name="sample_id"))
        coords = np.random.default_rng(0).random((n, 3))
        groups = ps.grouping_key(frame, coords, "target")
        self.assertEqual(len(np.unique(groups)), 3)

    def test_grouping_falls_back_to_spatial_cells(self):
        """No label at all -- the guarantee still has to mean something, so
        group by where the points actually are."""
        n = 400
        frame = pd.DataFrame(
            {"unrelated": np.arange(n)},
            index=pd.Index(np.arange(n).astype(str), name="sample_id"))
        rng = np.random.default_rng(1)
        coords = np.vstack([rng.normal(0, 0.2, (n // 2, 3)),
                            rng.normal(6, 0.2, (n // 2, 3))])
        groups = ps.grouping_key(frame, coords, "")
        self.assertGreater(len(np.unique(groups)), 1)
        # The two blobs must land in different cells.
        self.assertNotEqual(groups[0], groups[-1])


class TestPrefixDiscovery(unittest.TestCase):
    """The board can only offer an offline re-projection if the server reports
    it -- only the server knows which projections a dataset holds."""

    def _frame(self, prefixes):
        n = 40
        index = pd.MultiIndex.from_arrays(
            [np.arange(n), np.zeros(n, dtype=int)],
            names=["sample_id", "annotation_id"])
        data = {}
        for prefix in prefixes:
            for axis in "xyz":
                data[f"signals//{prefix}_{axis}"] = np.random.rand(n)
        return pd.DataFrame(data, index=index)

    def test_lists_every_stored_projection_with_the_live_one_first(self):
        frame = self._frame(["umap_layer3", "umap", "umap_early"])
        self.assertEqual(ps.available_prefixes(frame),
                         ["umap", "umap_early", "umap_layer3"])

    def test_response_carries_the_list(self):
        frame = self._frame(["umap", "umap_layer3"])
        resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertEqual(list(resp.available_prefixes), ["umap", "umap_layer3"])

    def test_a_named_prefix_selects_that_projection(self):
        frame = self._frame(["umap", "umap_layer3"])
        resp = ps.build_projection_response(
            frame, pb2.ProjectionRequest(prefix="umap_layer3"))
        self.assertTrue(resp.success)
        self.assertEqual(resp.returned, 40)

    def test_missing_prefix_reports_what_is_available(self):
        frame = self._frame(["umap", "umap_layer3"])
        resp = ps.build_projection_response(
            frame, pb2.ProjectionRequest(prefix="gone"))
        self.assertFalse(resp.success)
        self.assertIn("umap_layer3", resp.message)
        self.assertEqual(list(resp.available_prefixes), ["umap", "umap_layer3"])

    def test_half_written_projection_is_not_offered(self):
        """A prefix with only _x would fail the moment it was selected."""
        frame = self._frame(["umap"])
        frame["signals//broken_x"] = np.random.rand(len(frame))
        self.assertEqual(ps.available_prefixes(frame), ["umap"])


class TestEnvSwitch(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(ENV_ENABLED)

    def tearDown(self):
        os.environ.pop(ENV_ENABLED, None)
        if self._saved is not None:
            os.environ[ENV_ENABLED] = self._saved
        detach_projection()

    def test_default_is_on(self):
        os.environ.pop(ENV_ENABLED, None)
        self.assertTrue(projection_enabled())

    def test_off_values(self):
        for value in ("0", "false", "No", "OFF"):
            os.environ[ENV_ENABLED] = value
            self.assertFalse(projection_enabled(), value)

    def test_disabled_attaches_no_hook(self):
        os.environ[ENV_ENABLED] = "0"
        model = TinyNet()
        self.assertIsNone(attach_projection(model))
        # Nothing captured, because nothing is hooked.
        model(th.randn(4, 16))
        self.assertEqual(len(model.classifier._forward_hooks), 0)


class TestProjectionServing(unittest.TestCase):
    """The view-box/level-of-detail contract."""

    # The two index layouts that actually occur. The ledger frame is
    # (sample_id, annotation_id); the data service's _all_datasets_df -- which
    # is what GetProjection is handed -- is (origin, sample_id), with NO
    # annotation level. Addressing levels positionally works on the first and
    # silently returns zero rows on the second, so every serving test runs
    # against both.
    LAYOUTS = ("ledger", "view")

    def _index(self, n, layout):
        if layout == "ledger":
            return pd.MultiIndex.from_arrays(
                [np.arange(n).astype(str), np.zeros(n, dtype=int)],
                names=["sample_id", "annotation_id"])
        return pd.MultiIndex.from_arrays(
            [np.where(np.arange(n) % 4 == 0, "test_loader", "train_loader"),
             np.arange(n).astype(str)],
            names=["origin", "sample_id"])

    def _frame(self, n=500, prefix="umap", layout="view"):
        rng = np.random.default_rng(7)
        index = self._index(n, layout)
        return pd.DataFrame({
            f"signals//{prefix}_x": rng.uniform(-10, 10, n),
            f"signals//{prefix}_y": rng.uniform(-10, 10, n),
            f"signals//{prefix}_z": rng.uniform(-10, 10, n),
            "origin": rng.choice(["train_loader", "test_loader"], n),
            "signals//loss": rng.uniform(0, 3, n),
        }, index=index)

    def test_returns_packed_xyz_triples(self):
        resp = ps.build_projection_response(self._frame(120), pb2.ProjectionRequest())
        self.assertTrue(resp.success)
        self.assertEqual(resp.dims, 3)
        self.assertEqual(len(resp.sample_ids), 120)
        self.assertEqual(len(resp.coords), 120 * 3)
        self.assertEqual(resp.total_available, 120)

    def test_missing_columns_report_cleanly(self):
        frame = self._frame(10).drop(columns=["signals//umap_x", "signals//umap_y",
                                              "signals//umap_z"])
        resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertFalse(resp.success)
        self.assertIn("no projection for prefix", resp.message)

    def test_view_box_filters_to_the_box(self):
        frame = self._frame(800)
        req = pb2.ProjectionRequest(has_bounds=True, min_x=0, min_y=0, min_z=0,
                                    max_x=10, max_y=10, max_z=10)
        resp = ps.build_projection_response(frame, req)
        self.assertTrue(resp.success)
        xyz = np.array(resp.coords).reshape(-1, 3)
        self.assertTrue((xyz >= 0).all() and (xyz <= 10).all())
        self.assertLess(resp.total_in_view, resp.total_available)

    def test_extent_describes_the_whole_cloud_not_the_box(self):
        """The client frames the full cloud from this, so a zoomed-in request
        must not shrink it."""
        frame = self._frame(400)
        wide = ps.build_projection_response(frame, pb2.ProjectionRequest())
        req = pb2.ProjectionRequest(has_bounds=True, min_x=0, min_y=0, min_z=0,
                                    max_x=1, max_y=1, max_z=1)
        zoomed = ps.build_projection_response(frame, req)
        self.assertAlmostEqual(wide.extent_min_x, zoomed.extent_min_x, places=5)
        self.assertAlmostEqual(wide.extent_max_z, zoomed.extent_max_z, places=5)

    def test_budget_is_respected(self):
        resp = ps.build_projection_response(
            self._frame(5000), pb2.ProjectionRequest(max_points=250))
        self.assertEqual(len(resp.sample_ids), 250)
        self.assertEqual(resp.total_in_view, 5000)
        self.assertEqual(resp.total_available, 5000)

    def test_decimation_is_stable_across_requests(self):
        """Two identical requests must pick the SAME points, or the cloud boils
        under every camera move."""
        frame = self._frame(3000)
        req = pb2.ProjectionRequest(max_points=300)
        a = ps.build_projection_response(frame, req)
        b = ps.build_projection_response(frame, req)
        self.assertEqual(list(a.sample_ids), list(b.sample_ids))

    def test_zooming_in_only_adds_points(self):
        """A point drawn while zoomed out stays drawn when you zoom into it --
        the property that makes zoom feel like revealing detail, not resampling.
        """
        frame = self._frame(4000)
        wide = ps.build_projection_response(frame, pb2.ProjectionRequest(max_points=200))
        wide_xyz = np.array(wide.coords).reshape(-1, 3)
        inside = ((wide_xyz >= 0).all(axis=1) & (wide_xyz <= 10).all(axis=1))
        kept_when_zoomed_out = {sid for sid, keep
                                in zip(wide.sample_ids, inside) if keep}

        zoomed = ps.build_projection_response(frame, pb2.ProjectionRequest(
            has_bounds=True, min_x=0, min_y=0, min_z=0,
            max_x=10, max_y=10, max_z=10, max_points=200))
        self.assertTrue(kept_when_zoomed_out.issubset(set(zoomed.sample_ids)))

    def test_origins_are_interned(self):
        resp = ps.build_projection_response(self._frame(300), pb2.ProjectionRequest())
        self.assertEqual(sorted(resp.origins), ["test_loader", "train_loader"])
        self.assertEqual(len(resp.origin_ids), 300)
        self.assertTrue(all(0 <= i < len(resp.origins) for i in resp.origin_ids))

    def test_numeric_color_column(self):
        resp = ps.build_projection_response(
            self._frame(150), pb2.ProjectionRequest(color_column="signals//loss"))
        self.assertEqual(len(resp.color_values), 150)
        self.assertEqual(len(resp.color_labels), 0)

    def test_categorical_color_column(self):
        frame = self._frame(150)
        frame["cohort"] = np.where(np.arange(150) % 3 == 0, "alpha", "beta")
        resp = ps.build_projection_response(
            frame, pb2.ProjectionRequest(color_column="cohort"))
        self.assertEqual(len(resp.color_labels), 150)
        self.assertEqual(len(resp.color_values), 0)

    def test_colouring_by_origin_is_left_to_the_client(self):
        """"Colour by split" is the one categorical column the server does NOT
        answer. The Studio holds the split palette, so a sample is the same
        colour in the cloud as in the data grid; sending labels here would give
        the viewer a second, different palette for the same thing.

        See test_projection_colour_and_splits for the rest of this contract.
        """
        resp = ps.build_projection_response(
            self._frame(150), pb2.ProjectionRequest(color_column="origin"))
        self.assertEqual(len(resp.color_labels), 0)
        self.assertEqual(len(resp.color_values), 0)
        # The split membership the client needs to do it is still sent.
        self.assertEqual(sorted(resp.origins), ["test_loader", "train_loader"])

    def test_instance_rows_are_excluded(self):
        """Detection frames carry annotation_id >= 1 rows with no coordinates."""
        base = self._frame(100, layout="ledger")
        extra = base.copy()
        extra.index = pd.MultiIndex.from_arrays(
            [np.arange(100).astype(str), np.ones(100, dtype=int)],
            names=["sample_id", "annotation_id"])
        for axis in ("x", "y", "z"):
            extra[f"signals//umap_{axis}"] = np.nan
        resp = ps.build_projection_response(pd.concat([base, extra]),
                                            pb2.ProjectionRequest())
        self.assertEqual(resp.total_available, 100)

    def test_every_index_layout_is_served(self):
        """THE regression: the data service hands GetProjection a frame indexed
        (origin, sample_id) -- no annotation level, and sample ids on level 1.
        Addressing levels positionally read 'train_loader' as a sample id and
        filtered every row away, so a working projection reported "no finite
        coordinates yet" and the board never appeared."""
        for layout in self.LAYOUTS:
            with self.subTest(layout=layout):
                frame = self._frame(120, layout=layout)
                resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
                self.assertTrue(resp.success, f"{layout}: {resp.message}")
                self.assertEqual(resp.total_available, 120)
                self.assertEqual(len(resp.sample_ids), 120)
                # Sample ids, not split names.
                self.assertNotIn("train_loader", resp.sample_ids)
                self.assertTrue(all(s.isdigit() for s in resp.sample_ids))

    def test_single_level_index_is_served(self):
        """A flat frame (no MultiIndex at all) must not be special-cased away."""
        n = 40
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {f"signals//umap_{a}": rng.random(n) for a in "xyz"},
            index=pd.Index(np.arange(n).astype(str), name="sample_id"))
        resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertTrue(resp.success, resp.message)
        self.assertEqual(resp.total_available, n)

    def test_unprojected_samples_are_skipped(self):
        frame = self._frame(100)
        frame.iloc[:40, frame.columns.get_loc("signals//umap_y")] = np.nan
        resp = ps.build_projection_response(frame, pb2.ProjectionRequest())
        self.assertEqual(resp.total_available, 60)


if __name__ == "__main__":
    unittest.main()
