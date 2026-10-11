"""Instance-level projections: one point per BOX (wl.project_instances).

Covers the pure parts (matching, per-box features, the stored table), the
collection sweep through an adapter, and serving: the projection service turns
the table into a per-box frame that goes through the usual index, colours,
grid restriction and lasso -- and a lasso still selects SAMPLES.
"""
import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn

from weightslab.projection import instances as inst
from weightslab.projection.registry import clear_registry, prefix_info
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.trainer.services import projection_service as ps


class _Isolated(unittest.TestCase):
    """A fresh root_log_dir (where tables and the prefix registry live)."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="wl-instances-")
        self._saved_root = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self.root
        clear_registry()

    def tearDown(self):
        clear_registry()
        os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        if self._saved_root is not None:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._saved_root
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
class TestMatching(unittest.TestCase):
    def test_iou(self):
        iou = inst.box_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
        np.testing.assert_allclose(iou, [[1.0, 1 / 3, 0.0]], atol=1e-6)
        self.assertEqual(inst.box_iou(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape, (0, 1))

    def test_one_to_one_best_iou_first(self):
        gt = [[0, 0, 10, 10], [1, 0, 11, 10]]
        # Prediction 0 overlaps both labels; it goes to the one it fits best,
        # and the other label takes prediction 1.
        pred = [[1, 0, 11, 10], [0, 0, 9, 10]]
        g2p, p2g, g_iou = inst.match_boxes(gt, pred, 0.5)
        self.assertEqual(g2p.tolist(), [1, 0])
        self.assertEqual(p2g.tolist(), [1, 0])
        self.assertTrue(np.all(g_iou > 0.5))

    def test_threshold_is_strict_and_unmatched_is_minus_one(self):
        g2p, p2g, g_iou = inst.match_boxes([[0, 0, 10, 10]], [[5, 0, 15, 10]], 1 / 3)
        self.assertEqual(g2p.tolist(), [-1])
        self.assertEqual(p2g.tolist(), [-1])
        self.assertTrue(np.isnan(g_iou[0]))

    def test_matching_ignores_class_so_a_wrong_class_is_seen(self):
        g2p, p2g, _ = inst.match_boxes([[0, 0, 10, 10]], [[0, 0, 10, 10]], 0.5)
        labels, preds = inst.match_categories([2], [7], g2p, p2g)
        self.assertEqual(labels, [inst.WRONG_CLASS])
        self.assertEqual(preds, [inst.WRONG_CLASS])

    def test_the_four_categories(self):
        gt = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]]
        gt_cls = [0, 1, 2]
        pred = [[0, 0, 10, 10], [20, 0, 30, 10], [60, 0, 70, 10]]
        pred_cls = [0, 5, 3]
        g2p, p2g, _ = inst.match_boxes(gt, pred, 0.5)
        labels, preds = inst.match_categories(gt_cls, pred_cls, g2p, p2g)
        self.assertEqual(labels, [inst.CONSISTENT, inst.WRONG_CLASS, inst.NO_PREDICTION])
        self.assertEqual(preds, [inst.CONSISTENT, inst.WRONG_CLASS, inst.MISSING_LABEL])


# ---------------------------------------------------------------------------
# per-box features
# ---------------------------------------------------------------------------
class TestRoiFeatures(unittest.TestCase):
    def _maps(self):
        th.manual_seed(0)
        # A 64x64 input read at strides 8 / 16 / 32, channels 4 / 6 / 8.
        return [th.randn(2, 4, 8, 8), th.randn(2, 6, 4, 4), th.randn(2, 8, 2, 2)], [8, 16, 32]

    def test_one_unit_vector_per_box_over_every_level(self):
        maps, strides = self._maps()
        rois = th.tensor([[0, 0, 0, 32, 32], [1, 10, 10, 60, 60], [0, 40, 40, 44, 44.]])
        feats = inst.roi_features(maps, strides, rois)
        self.assertEqual(tuple(feats.shape), (3, 4 + 6 + 8))
        np.testing.assert_allclose(feats.norm(dim=1).numpy(), 1.0, atol=1e-5)

    def test_the_same_box_reads_the_same_vector_and_another_image_does_not(self):
        maps, strides = self._maps()
        rois = th.tensor([[0, 8, 8, 40, 40], [0, 8, 8, 40, 40], [1, 8, 8, 40, 40.]])
        feats = inst.roi_features(maps, strides, rois)
        np.testing.assert_allclose(feats[0].numpy(), feats[1].numpy(), atol=1e-6)
        self.assertGreater(float((feats[0] - feats[2]).abs().sum()), 1e-3)

    def test_no_boxes(self):
        maps, strides = self._maps()
        self.assertEqual(tuple(inst.roi_features(maps, strides, th.zeros((0, 5))).shape), (0, 18))


# ---------------------------------------------------------------------------
# the stored table
# ---------------------------------------------------------------------------
def _table(n_samples=6, boxes_per_sample=3, with_predictions=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_samples):
        for a in range(1, boxes_per_sample + 1):
            rows.append((str(s), a, (s + a) % 3, inst.CONSISTENT if a != 2 else inst.WRONG_CLASS))
        if with_predictions:
            rows.append((str(s), -1, 1, inst.MISSING_LABEL))
    n = len(rows)
    x1 = rng.uniform(0, 0.5, n)
    y1 = rng.uniform(0, 0.5, n)
    return inst.InstanceTable(
        sample_ids=np.array([r[0] for r in rows], dtype=object),
        instance_ids=np.array([r[1] for r in rows], dtype=np.int64),
        class_ids=np.array([r[2] for r in rows], dtype=np.int32),
        pred_class_ids=np.array([r[2] for r in rows], dtype=np.int32),
        confidences=rng.uniform(0.3, 1, n).astype(np.float32),
        ious=rng.uniform(0.5, 1, n).astype(np.float32),
        matches=np.array([r[3] for r in rows], dtype=object),
        boxes=np.stack([x1, y1, x1 + 0.2, y1 + 0.3], axis=1).astype(np.float32),
        origins=np.array(["train_loader" if int(r[0]) < n_samples // 2 else "val_loader"
                          for r in rows], dtype=object),
        coords=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        class_names=["car", "truck", "bus"],
    )


class TestStoredTable(_Isolated):
    def test_round_trip(self):
        table = _table()
        path = os.path.join(self.root, "t.npz")
        table.save(path)
        back = inst.InstanceTable.load(path)
        self.assertEqual(back.sample_ids.tolist(), table.sample_ids.tolist())
        self.assertEqual(back.instance_ids.tolist(), table.instance_ids.tolist())
        self.assertEqual(back.matches.tolist(), table.matches.tolist())
        self.assertEqual(back.class_names, ["car", "truck", "bus"])
        np.testing.assert_allclose(back.coords, table.coords)
        self.assertEqual(back.kinds.tolist().count(inst.PREDICTION), 6)

    def test_saving_bring_your_own_registers_an_instance_prefix(self):
        table = _table()
        inst.save_instance_projection_coords(
            table.coords[:, :2], table.sample_ids, table.instance_ids, prefix="boxes_pca",
            class_ids=table.class_ids, class_names={0: "car", 1: "truck", 2: "bus"},
            matches=table.matches, boxes=table.boxes)
        self.assertEqual(prefix_info("boxes_pca").get("level"), "instance")
        self.assertTrue(inst.is_instance_prefix("boxes_pca"))
        self.assertEqual(inst.instance_prefixes(), ["boxes_pca"])
        stored = inst.load_instance_table("boxes_pca")
        self.assertEqual(len(stored), len(table))
        # A 2-D projection is drawn on the z = 0 plane.
        self.assertTrue(np.all(stored.coords[:, 2] == 0))
        self.assertEqual(stored.class_name(1), "truck")

    def test_misaligned_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            inst.save_instance_projection_coords(np.zeros((3, 3)), ["1", "2"], [1, 2], prefix="x")

    def test_a_bad_prefix_is_refused(self):
        with self.assertRaises(ValueError):
            inst.save_instance_projection_coords(np.zeros((1, 3)), ["1"], [1], prefix="a/b")


# ---------------------------------------------------------------------------
# the sweep, through an adapter
# ---------------------------------------------------------------------------
class ToyDetector(nn.Module):
    """Two feature levels; no head -- the adapter below 'predicts'."""

    def __init__(self):
        super().__init__()
        self.p3 = nn.Conv2d(3, 4, 3, stride=8, padding=1)
        self.p4 = nn.Conv2d(3, 6, 3, stride=16, padding=1)

    def forward(self, x):
        return [self.p3(x), self.p4(x)]


class ToyLoader:
    """A loader of sample ids (a list would read as a list of loaders)."""

    def __init__(self, *ids):
        self.ids = [str(i) for i in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.ids[item]


class ToyAdapter(inst.InstanceAdapter):
    """Image 0: a label of class 0 with a matching class-0 prediction, and a
    prediction with no label. Image 1: a label predicted as class 1, and a
    label nothing was predicted on."""

    def batches(self, loader):
        for start in range(0, len(loader), 2):
            ids = loader[start:start + 2]
            gt = []
            for _ in ids:
                gt.append((np.array([[4, 4, 28, 28], [36, 36, 60, 60]], np.float32),
                           np.array([0, 2]), np.array([1, 2])))
            yield inst.InstanceBatch(inputs=th.rand(len(ids), 3, 64, 64), sample_ids=ids,
                                     gt=gt, image_size=(64, 64))

    def forward(self, model, batch, *, conf):
        maps = model(batch.inputs)
        preds = []
        for b in range(len(batch.sample_ids)):
            boxes = np.array([[4, 4, 28, 28], [36, 4, 60, 28], [36, 36, 60, 60]], np.float32)
            scores = np.array([0.9, 0.8, 0.1 if b == 1 else 0.7], np.float32)
            classes = np.array([0, 1, 2 if b == 0 else 1])
            keep = scores > conf
            preds.append((boxes[keep], scores[keep], classes[keep]))
        return inst.InstanceForward(feature_maps=maps, strides=[8, 16], predictions=preds)

    def class_names(self, model, loaders):
        return {0: "car", 1: "truck", 2: "bus"}

    def layer(self, model):
        return "p3", "toy"


class TestCollect(_Isolated):
    def test_every_label_and_every_unlabelled_prediction(self):
        feats, table = inst.collect_instances(
            ToyDetector(), {"train_loader": ToyLoader(10, 11), "val_loader": ToyLoader(20)}, ToyAdapter(),
            conf=0.25, verbose=False)
        self.assertEqual(feats.shape[0], len(table))
        self.assertEqual(feats.shape[1], 4 + 6)
        # Per image: 2 labels + 1 prediction on empty ground (class 1 at 36,4).
        self.assertEqual(len(table), 3 * 3)
        mine = table.sample_ids == "10"
        self.assertEqual(sorted(table.instance_ids[mine].tolist()), [-1, 1, 2])
        self.assertEqual(set(table.origins.tolist()), {"train_loader", "val_loader"})
        self.assertEqual(table.class_names, ["car", "truck", "bus"])
        self.assertEqual((table.layer, table.layer_detail), ("p3", "toy"))
        # Labels carry their matched prediction.
        first = np.flatnonzero(mine & (table.instance_ids == 1))[0]
        self.assertEqual(table.matches[first], inst.CONSISTENT)
        self.assertAlmostEqual(float(table.confidences[first]), 0.9, places=5)
        self.assertAlmostEqual(float(table.ious[first]), 1.0, places=5)
        np.testing.assert_allclose(table.boxes[first], [4 / 64, 4 / 64, 28 / 64, 28 / 64], atol=1e-6)
        pred = np.flatnonzero(mine & (table.instance_ids == -1))[0]
        self.assertEqual(table.matches[pred], inst.MISSING_LABEL)
        self.assertEqual(int(table.class_ids[pred]), 1)

    def test_categories_follow_the_predictions(self):
        _, table = inst.collect_instances(ToyDetector(), ToyLoader(10, 11), ToyAdapter(),
                                          conf=0.25, verbose=False)
        by_id = {(s, i): m for s, i, m in zip(table.sample_ids, table.instance_ids, table.matches)}
        self.assertEqual(by_id[("10", 2)], inst.CONSISTENT)       # class 2 predicted 2
        self.assertEqual(by_id[("11", 2)], inst.NO_PREDICTION)    # its prediction is under conf

    def test_predictions_can_be_left_out(self):
        _, table = inst.collect_instances(ToyDetector(), ToyLoader(10), ToyAdapter(), conf=0.25,
                                          include_predictions=False, verbose=False)
        self.assertTrue((table.instance_ids > 0).all())

    def test_a_list_is_several_loaders(self):
        _, table = inst.collect_instances(ToyDetector(), [ToyLoader(1), ToyLoader(2)], ToyAdapter(),
                                          conf=0.25, verbose=False)
        self.assertEqual(sorted(set(table.sample_ids.tolist())), ["1", "2"])

    def test_max_instances(self):
        _, table = inst.collect_instances(ToyDetector(), ToyLoader(1, 2, 3, 4), ToyAdapter(),
                                          conf=0.25, max_instances=5, verbose=False)
        self.assertEqual(len(table), 5)

    def test_the_model_is_left_as_it_was(self):
        model = ToyDetector().train()
        inst.collect_instances(model, ToyLoader(1), ToyAdapter(), verbose=False)
        self.assertTrue(model.training)

    def test_project_instances_writes_a_registered_table(self):
        stats = inst.project_instances(ToyDetector(), {"train_loader": ToyLoader(1, 2, 3)},
                                       adapter=ToyAdapter(), prefix="boxes", method="pca",
                                       verbose=False)
        self.assertEqual(stats["instances"], 9)
        self.assertEqual(stats["labels"], 6)
        self.assertEqual(stats["predictions"], 3)
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(sum(stats["matches"].values()), 9)
        table = inst.load_instance_table("boxes")
        self.assertEqual(table.coords.shape, (9, 3))
        self.assertTrue(np.isfinite(table.coords).all())
        self.assertEqual(prefix_info("boxes").get("layer"), "p3")

    def test_the_built_in_umap_fits_too(self):
        stats = inst.project_instances(ToyDetector(), ToyLoader(*range(6)),
                                       adapter=ToyAdapter(), prefix="boxes_umap",
                                       method="umap", epochs=2, verbose=False)
        self.assertEqual(stats["instances"], 18)
        self.assertTrue(np.isfinite(inst.load_instance_table("boxes_umap").coords).all())

    def test_your_own_method(self):
        inst.project_instances(ToyDetector(), ToyLoader(1, 2), adapter=ToyAdapter(), prefix="mine",
                               method=lambda f: f[:, :2], verbose=False)
        self.assertEqual(inst.load_instance_table("mine").coords.shape[1], 3)

    def test_too_few_boxes_is_an_error_not_an_empty_projection(self):
        with self.assertRaises(RuntimeError):
            inst.project_instances(ToyDetector(), ToyLoader(1), adapter=ToyAdapter(),
                                   include_predictions=False, method="pca", verbose=False)

    def test_no_adapter_for_an_unknown_model(self):
        with self.assertRaises(TypeError):
            inst.resolve_adapter(nn.Linear(2, 2))


# ---------------------------------------------------------------------------
# serving
# ---------------------------------------------------------------------------
class TestServing(_Isolated):
    CAMERA = list(np.eye(4).T.ravel())
    VIEWPORT = [100, 100]
    LEFT = [0, 0, 50, 0, 50, 100, 0, 100]          # NDC x in [-1, 0]

    def setUp(self):
        super().setUp()
        self.table = _table()
        inst.write_instance_table(self.table, "boxes")
        # The data service's view: one row per sample, (origin, sample_id).
        n = 6
        index = pd.MultiIndex.from_arrays(
            [["train_loader"] * 3 + ["val_loader"] * 3, np.arange(n)], names=["origin", "sample_id"])
        self.view = pd.DataFrame({
            "signals//umap_x": np.linspace(-1, 1, n), "signals//umap_y": np.zeros(n),
            "signals//umap_z": np.zeros(n),
            "discarded": [False, True, False, False, False, False],
            "signals//loss": np.arange(n, dtype=float),
        }, index=index)
        clear_registry()
        from weightslab.projection.registry import register_prefix
        register_prefix("umap")
        register_prefix("boxes", level="instance")

    def serve(self, cache=None, view=None, filtered=False, **fields):
        cache = cache or ps.ProjectionCache()
        frame = self.view if view is None else view
        request = pb2.ProjectionRequest(prefix="boxes", max_points=1000, **fields)
        return cache.serve(request, view=lambda: frame, is_filtered=lambda: filtered,
                           pull_full=lambda: self.view)

    def test_one_point_per_box_with_its_box(self):
        resp = self.serve()
        self.assertTrue(resp.success, resp.message)
        self.assertTrue(resp.instance_level)
        self.assertEqual(resp.returned, len(self.table))
        self.assertEqual(len(resp.instance_ids), resp.returned)
        self.assertEqual(len(resp.instance_boxes), 4 * resp.returned)
        self.assertIn("instance_match", list(resp.instance_columns))
        # Each point is (sample, box) and the box is the one stored for it.
        stored = {(s, int(i)): b for s, i, b in
                  zip(self.table.sample_ids, self.table.instance_ids, self.table.boxes)}
        for k, (sid, iid) in enumerate(zip(resp.sample_ids, resp.instance_ids)):
            np.testing.assert_allclose(resp.instance_boxes[4 * k:4 * k + 4], stored[(sid, iid)],
                                       atol=1e-6)

    def test_coloured_by_class_by_default(self):
        resp = self.serve()
        self.assertEqual(set(resp.color_labels), {"car", "truck", "bus"})

    def test_coloured_by_match_or_by_a_sample_column(self):
        by_match = self.serve(color_column="instance_match")
        self.assertEqual(set(by_match.color_labels),
                         {inst.CONSISTENT, inst.WRONG_CLASS, inst.MISSING_LABEL})
        by_loss = self.serve(color_column="signals//loss")
        # The image's value, repeated on each of its boxes.
        for sid, value in zip(by_loss.sample_ids, by_loss.color_values):
            self.assertEqual(value, float(sid))

    def test_split_and_discarded_come_from_the_sample(self):
        resp = self.serve()
        origins = [resp.origins[i] for i in resp.origin_ids]
        for sid, origin, discarded in zip(resp.sample_ids, origins, resp.discarded):
            self.assertEqual(origin, "train_loader" if int(sid) < 3 else "val_loader")
            self.assertEqual(discarded, sid == "1")

    def test_the_picker_lists_both_kinds(self):
        resp = self.serve()
        self.assertEqual(list(resp.available_prefixes), ["umap", "boxes"])
        sample_level = ps.ProjectionCache().serve(
            pb2.ProjectionRequest(max_points=100), view=lambda: self.view,
            is_filtered=lambda: False)
        self.assertFalse(sample_level.instance_level)
        self.assertIn("boxes", list(sample_level.available_prefixes))

    def test_a_filtered_view_keeps_only_its_images_boxes(self):
        subset = self.view.iloc[[0, 4]]
        resp = self.serve(view=subset, filtered=True, follow_view=True)
        self.assertEqual(set(resp.sample_ids), {"0", "4"})
        self.assertEqual(resp.returned, 8)

    def test_grid_overview_draws_the_pages_boxes(self):
        resp = self.serve(follow_view=True, restrict_sample_ids=["2", "5"])
        self.assertTrue(resp.success, resp.message)
        self.assertEqual(set(resp.sample_ids), {"2", "5"})
        self.assertTrue(resp.instance_level)
        self.assertEqual(list(resp.available_prefixes), ["umap", "boxes"])

    def test_a_lasso_selects_the_images_its_boxes_are_in(self):
        cache = ps.ProjectionCache()
        resp = self.serve(cache=cache, lasso_mode=pb2.PROJECTION_LASSO_REPLACE,
                          lasso_px=self.LEFT, lasso_viewport=self.VIEWPORT,
                          view_projection=self.CAMERA)
        self.assertTrue(resp.success, resp.message)
        self.assertIn("boxes in", resp.message)
        inside = self.table.coords[:, 0] < -0.02
        want = set(int(s) for s in self.table.sample_ids[inside])
        got = set(int(s) for s in cache.selection(resp.selection_token))
        self.assertTrue(want <= got)
        self.assertEqual(resp.selected, len(got))
        # Applied to the data board's view: those samples' rows.
        rows = cache.selection_rows(resp.selection_token, self.view)
        self.assertEqual(set(self.view.index.get_level_values("sample_id")[rows]), got)
        # And the board's "whole dataset" is the sample frame, not the boxes.
        self.assertIs(cache.full_frame(), self.view)

    def test_drawn_points_say_whether_they_are_selected(self):
        cache = ps.ProjectionCache()
        lasso = self.serve(cache=cache, lasso_mode=pb2.PROJECTION_LASSO_REPLACE,
                           lasso_px=self.LEFT, lasso_viewport=self.VIEWPORT,
                           view_projection=self.CAMERA)
        self.assertTrue(lasso.success, lasso.message)
        view = self.serve(cache=cache)
        self.assertEqual(len(view.in_selection), view.returned)
        # Box-precise: the boxes inside the loop, not every box of their images.
        x = np.asarray(view.coords, dtype=np.float32).reshape(-1, 3)[:, 0]
        flags = np.asarray(view.in_selection)
        self.assertTrue(flags.any())
        self.assertTrue((x[flags] <= 0.025).all())
        self.assertTrue((x[~flags] >= -0.025).all())

    def test_a_rewritten_table_is_served(self):
        cache = ps.ProjectionCache()
        self.assertEqual(self.serve(cache=cache).returned, len(self.table))
        smaller = self.table.take(np.arange(10))
        inst.write_instance_table(smaller, "boxes")
        ps.ProjectionCache.REFRESH_SECONDS, saved = 0.0, ps.ProjectionCache.REFRESH_SECONDS
        try:
            fresh = ps.ProjectionCache()
            self.assertEqual(self.serve(cache=fresh).returned, 10)
        finally:
            ps.ProjectionCache.REFRESH_SECONDS = saved

    def test_a_missing_table_says_so(self):
        os.remove(inst.table_path("boxes"))
        resp = self.serve()
        self.assertFalse(resp.success)
        self.assertIn("boxes", resp.message)

    def test_string_sample_ids_on_the_frame(self):
        view = self.view.copy()
        view.index = pd.MultiIndex.from_arrays(
            [view.index.get_level_values("origin"),
             view.index.get_level_values("sample_id").astype(str)], names=["origin", "sample_id"])
        resp = self.serve(view=view)
        self.assertEqual(resp.returned, len(self.table))


# ---------------------------------------------------------------------------
# Ultralytics adapter
# ---------------------------------------------------------------------------
@unittest.skipUnless(importlib.util.find_spec("ultralytics") is not None,
                     "ultralytics not installed")
class TestUltralyticsAdapter(_Isolated):
    def _model(self):
        from ultralytics.nn.tasks import DetectionModel
        th.manual_seed(0)
        return DetectionModel("yolo11n.yaml", nc=3, verbose=False).eval()

    def test_features_come_from_the_detect_inputs(self):
        from weightslab.integrations.ultralytics.instances import UltralyticsInstanceAdapter
        model = self._model()
        batch = inst.InstanceBatch(inputs=th.rand(2, 3, 128, 128), sample_ids=["a", "b"],
                                   gt=[(np.zeros((0, 4)), np.zeros(0), np.zeros(0))] * 2,
                                   image_size=(128, 128))
        out = UltralyticsInstanceAdapter().forward(model, batch, conf=0.0)
        self.assertEqual([tuple(f.shape[-2:]) for f in out.feature_maps], [(16, 16), (8, 8), (4, 4)])
        self.assertEqual(out.strides, [8.0, 16.0, 32.0])
        self.assertEqual(len(out.predictions), 2)
        boxes, scores, classes = out.predictions[0]
        self.assertEqual(boxes.shape[1], 4)
        self.assertTrue(((classes >= 0) & (classes < 3)).all())

    def test_the_batch_dict_becomes_boxes_in_pixels_in_label_order(self):
        from weightslab.integrations.ultralytics.instances import UltralyticsInstanceAdapter
        batch = {
            "img": th.zeros(2, 3, 100, 200, dtype=th.uint8),
            "ids": th.tensor([7, 9]),
            "batch_idx": th.tensor([0., 1., 1.]),
            "cls": th.tensor([[2.], [0.], [1.]]),
            "bboxes": th.tensor([[0.5, 0.5, 0.5, 0.5], [0.25, 0.5, 0.1, 0.2], [0.75, 0.5, 0.1, 0.2]]),
        }
        ib = UltralyticsInstanceAdapter._to_instance_batch(batch)
        self.assertEqual(ib.sample_ids, ["7", "9"])
        self.assertEqual(ib.image_size, (100, 200))
        boxes, cls, ann = ib.gt[1]
        np.testing.assert_allclose(boxes[0], [40, 40, 60, 60], atol=1e-4)
        self.assertEqual(cls.tolist(), [0, 1])
        self.assertEqual(ann.tolist(), [1, 2])

    def test_the_trainer_section_turns_it_on(self):
        from weightslab.integrations.ultralytics.instances import instance_projection_config
        self.assertFalse(instance_projection_config({})["enabled"])
        self.assertFalse(instance_projection_config(None)["enabled"])
        cfg = instance_projection_config({"instance_projection": {"prefix": "boxes", "method": "pca"}})
        self.assertTrue(cfg["enabled"])                       # present means on
        self.assertEqual((cfg["prefix"], cfg["method"], cfg["every_n_epochs"]), ("boxes", "pca", 1))
        self.assertFalse(instance_projection_config(
            {"instance_projection": {"enabled": False}})["enabled"])
        self.assertTrue(instance_projection_config({"instance_projection": True})["enabled"])

    def test_predictions_decode_xywh_and_nms(self):
        from weightslab.integrations.ultralytics.instances import UltralyticsInstanceAdapter

        class Head:
            nc, xyxy, end2end = 2, False, False

        # Three anchors: two overlapping class-0 boxes, one class-1 box.
        y = th.tensor([[[10., 11., 50.], [10., 10., 50.], [8., 8., 8.], [8., 8., 8.],
                        [0.9, 0.8, 0.1], [0.1, 0.1, 0.7]]])
        (boxes, scores, classes), = UltralyticsInstanceAdapter()._predictions(Head(), (y, {}), 0.25)
        self.assertEqual(classes.tolist(), [0, 1])
        np.testing.assert_allclose(boxes[0], [6, 6, 14, 14])
        np.testing.assert_allclose(scores, [0.9, 0.7], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
