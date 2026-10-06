"""The Fashion-MNIST "live UMAP, then a PCA of your own" example, and the docs
that point at it.

``examples/PyTorch/wl-fashion-mnist-umap/main.py`` drives a run in stages:
train with the built-in parametric UMAP, keep a checkpoint, reload it, discard
the lowest-loss training samples, project what is left with a PCA written in
plain torch, compare the two layouts, retrain.

Three things are held here:

* the building blocks (loss cut, PCA, neighbour purity, layout comparison) on
  small synthetic inputs;
* the whole script, run for real on a synthetic Fashion-MNIST (no download),
  against the real ledger, checkpoint manager and projection;
* the docs: the page, the gallery entry, and no link left to the sandbox.
"""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from PIL import Image

import weightslab as wl
from weightslab import src as wl_src
from weightslab.backend import ledgers
from weightslab.projection import detach_projection
from weightslab.projection.registry import clear_registry, known_prefixes

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "weightslab" / "examples" / "PyTorch" / "wl-fashion-mnist-umap"
DOCS = REPO_ROOT / "docs"

LOSS = "signals//train-loss-CE"


def load_example():
    """Import the example's main.py by path (its directory is not a package)."""
    spec = importlib.util.spec_from_file_location("wl_fashion_mnist_umap_example",
                                                  EXAMPLE_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load_example()


# =============================================================================
# The building blocks
# =============================================================================
def sample_rows(n=10, **columns):
    """A per-sample table the way ``sample_table()`` returns it."""
    return pd.DataFrame(columns, index=pd.Index(range(n), name="sample_id"))


class TestLowestLossIds(unittest.TestCase):

    def test_picks_the_lowest_fraction(self):
        table = sample_rows(10, **{LOSS: [9, 1, 8, 2, 7, 3, 6, 4, 5, 0.5]})
        self.assertEqual(sorted(example.lowest_loss_ids(table, 0.3)), [1, 3, 9])

    def test_returns_plain_ints(self):
        table = sample_rows(4, **{LOSS: [0.4, 0.1, 0.3, 0.2]})
        ids = example.lowest_loss_ids(table, 0.5)
        self.assertTrue(all(type(i) is int for i in ids))
        self.assertEqual(ids, [1, 3])                       # lowest first

    def test_a_sample_with_no_loss_is_never_picked(self):
        """Unseen is not easy: it must not be discarded as if the model knew it."""
        table = sample_rows(4, **{LOSS: [np.nan, 0.2, np.nan, 0.1]})
        self.assertEqual(example.lowest_loss_ids(table, 1.0), [3, 1])

    def test_a_fraction_of_zero_or_a_missing_column_picks_nothing(self):
        table = sample_rows(4, **{LOSS: [0.4, 0.1, 0.3, 0.2]})
        self.assertEqual(example.lowest_loss_ids(table, 0), [])
        self.assertEqual(example.lowest_loss_ids(table, 0.5, column="signals//nope"), [])

    def test_a_fraction_above_one_means_all_of_them(self):
        table = sample_rows(4, **{LOSS: [0.4, 0.1, 0.3, 0.2]})
        self.assertEqual(len(example.lowest_loss_ids(table, 7.0)), 4)

    def test_losses_stored_as_objects_still_compare_as_numbers(self):
        table = sample_rows(3, **{LOSS: np.array([10.0, 9.0, 100.0], dtype=object)})
        self.assertEqual(example.lowest_loss_ids(table, 0.34), [1])   # not '100.0' < '9.0'


class TestCheckpointFailuresAreLoud(unittest.TestCase):
    """A checkpoint that was not written, or not reloaded, must stop the run
    there -- not surface two stages later as a confusing result."""

    def test_a_checkpoint_that_could_not_be_written_is_an_error(self):
        manager = mock.MagicMock()
        manager.save_model_checkpoint.return_value = None          # what the manager gives
        with mock.patch.object(example.ledgers, "get_checkpoint_manager", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "could not write a checkpoint"):
                example.save_checkpoint()

    def test_a_checkpoint_that_could_not_be_reloaded_is_an_error(self):
        manager = mock.MagicMock()
        manager.load_state.return_value = False
        with mock.patch.object(example.ledgers, "get_checkpoint_manager", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "step 40"):
                example.reload_checkpoint(40)
        self.assertEqual(manager.load_state.call_args.kwargs["target_step"], 40)


class TestPcaCoordinates(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        # 400 points in 5-D: almost all variance along two known directions.
        scale = np.array([10.0, 4.0, 0.05, 0.05, 0.05])
        self.features = rng.normal(size=(400, 5)) * scale + 100.0   # off-centre on purpose

    def test_shape_for_two_and_three_components(self):
        for dim in (2, 3):
            coords, _ = example.pca_coordinates(self.features, dim)
            self.assertEqual(coords.shape, (400, dim))

    def test_the_coordinates_are_centred(self):
        coords, _ = example.pca_coordinates(self.features, 2)
        np.testing.assert_allclose(coords.mean(0), 0.0, atol=1e-3)

    def test_the_variance_share_matches_the_covariance_spectrum(self):
        _, share = example.pca_coordinates(self.features, 2)
        eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(self.features.T)))[::-1]
        np.testing.assert_allclose(share, eigenvalues[:2].sum() / eigenvalues.sum(), rtol=1e-3)
        self.assertGreater(share, 0.99)

    def test_the_first_axis_carries_the_most_variance(self):
        coords, _ = example.pca_coordinates(self.features, 3)
        variances = coords.var(0)
        self.assertTrue(variances[0] > variances[1] > variances[2])

    def test_distances_never_grow(self):
        """A projection can only lose information: pairwise distances shrink."""
        coords, _ = example.pca_coordinates(self.features, 2)
        original = np.linalg.norm(self.features[0] - self.features[1])
        self.assertLessEqual(np.linalg.norm(coords[0] - coords[1]), original + 1e-4)

    def test_accepts_a_tensor(self):
        coords, _ = example.pca_coordinates(torch.as_tensor(self.features), 2)
        self.assertEqual(coords.shape, (400, 2))


class TestKnnPurity(unittest.TestCase):

    def test_separated_classes_are_perfectly_pure(self):
        coords = np.concatenate([np.zeros((30, 2)), np.full((30, 2), 100.0)])
        coords += np.random.default_rng(0).normal(size=coords.shape)
        labels = np.array([0] * 30 + [1] * 30)
        self.assertEqual(example.knn_purity(coords, labels, k=5), 1.0)

    def test_unrelated_labels_sit_near_chance(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(1500, 2))
        labels = rng.integers(0, 10, size=1500)
        self.assertLess(example.knn_purity(coords, labels, k=10), 0.2)

    def test_a_point_is_not_its_own_neighbour(self):
        """Two points, two classes: if a point counted itself, purity would be 1."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        self.assertEqual(example.knn_purity(coords, np.array([0, 1]), k=1), 0.0)

    def test_k_larger_than_the_set_is_clipped(self):
        coords = np.array([[0.0], [1.0], [2.0]])
        value = example.knn_purity(coords, np.array([0, 0, 0]), k=50)
        self.assertEqual(value, 1.0)

    def test_a_single_point_has_no_neighbours(self):
        self.assertTrue(np.isnan(example.knn_purity(np.zeros((1, 2)), np.array([0]), k=3)))

    def test_works_across_the_row_blocks_it_computes_in(self):
        """More rows than one block (2048): the blocks must line up."""
        rng = np.random.default_rng(1)
        n = 2100
        labels = rng.integers(0, 3, size=n)
        coords = np.stack([labels * 50.0, np.zeros(n)], axis=1) + rng.normal(size=(n, 2))
        self.assertGreater(example.knn_purity(coords, labels, k=5), 0.99)

    def test_three_dimensional_layouts_work_too(self):
        coords = np.concatenate([np.zeros((20, 3)), np.full((20, 3), 50.0)])
        coords += np.random.default_rng(0).normal(size=coords.shape)
        self.assertEqual(example.knn_purity(coords, np.array([0] * 20 + [1] * 20), k=4), 1.0)


class TestLayoutAndCompare(unittest.TestCase):

    def setUp(self):
        n = 12
        rng = np.random.default_rng(0)
        a = rng.normal(size=(n, 2))
        b = rng.normal(size=(n, 3))
        a[10:] = np.nan                                  # "a" never placed 10 and 11
        b[:3] = np.nan                                   # "b" never placed 0..2
        discarded = np.zeros(n, dtype=bool)
        discarded[5] = True
        self.table = sample_rows(
            n, target=[0, 1] * 6, discarded=discarded,
            **{"signals//a_x": a[:, 0], "signals//a_y": a[:, 1],
               "signals//b_x": b[:, 0], "signals//b_y": b[:, 1], "signals//b_z": b[:, 2]})

    def test_layout_leaves_out_discarded_and_unplaced_samples(self):
        ids, coords = example.layout(self.table, "a")
        self.assertEqual(ids, [0, 1, 2, 3, 4, 6, 7, 8, 9])   # not 5 (discarded), 10, 11
        self.assertEqual(coords.shape, (9, 2))

    def test_layout_follows_the_columns_that_exist(self):
        ids, coords = example.layout(self.table, "b")
        self.assertEqual(coords.shape[1], 3)
        self.assertNotIn(5, ids)
        self.assertEqual(ids[0], 3)

    def test_an_unknown_prefix_places_nothing(self):
        ids, coords = example.layout(self.table, "nope")
        self.assertEqual(ids, [])
        self.assertEqual(coords.size, 0)

    def test_compare_scores_the_samples_every_layout_places(self):
        scores, shared = example.compare(self.table, ["a", "b"], k=2)
        self.assertEqual(shared, 6)                         # 3,4,6,7,8,9
        self.assertEqual(set(scores), {"a", "b"})
        for value in scores.values():
            self.assertTrue(0.0 <= value <= 1.0)


# =============================================================================
# The whole script, for real
# =============================================================================
class FakeFashionMNIST:
    """``torchvision.datasets.FashionMNIST``'s interface -- ``(PIL image, label)``
    items and ``.targets`` -- with no download. Class ``c`` is a bright band at
    rows ``2c+3 .. 2c+5``, so even a few steps tell the classes apart."""

    def __init__(self, root=None, train=True, download=False, transform=None):
        count = 1200 if train else 300
        rng = np.random.default_rng(0 if train else 1)
        self.targets = torch.as_tensor(np.arange(count) % 10)
        self.images = rng.integers(0, 40, size=(count, 28, 28), dtype=np.uint8)
        for index, label in enumerate(self.targets.tolist()):
            self.images[index, 2 * label + 3:2 * label + 6, :] += 150

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return Image.fromarray(self.images[index]), int(self.targets[index])


TRAIN_SIZE, TEST_SIZE = 400, 100
STEPS_INITIAL, STEPS_RETRAIN, DISCARD = 24, 12, 0.25


def tiny_parameters(root):
    return {
        "experiment_name": "test_fashion_mnist_umap",
        "device": "cpu",
        "root_log_dir": root,
        "data_root": os.path.join(root, "data"),
        "projection": {"every_n_steps": 4, "graph_size": 128},
        "exploration": {"steps_initial": STEPS_INITIAL, "steps_retrain": STEPS_RETRAIN,
                        "discard_fraction": DISCARD, "knn_k": 5},
        "eval_full_to_train_steps_ratio": 12,
        "experiment_dump_to_train_steps_ratio": 10 ** 9,   # only the explicit checkpoints
        "tqdm_display": False,
        "start_timeout": 0,
        "serving_grpc": False,
        "ledger_enable_h5_persistence": False,
        "data": {"train_loader": {"batch_size": 32, "max_samples": TRAIN_SIZE},
                 "test_loader": {"batch_size": 50, "max_samples": TEST_SIZE}},
    }


class TestTheStagedRun(unittest.TestCase):
    """One real run of ``run()``; each test reads a different part of it."""

    @classmethod
    def setUpClass(cls):
        ledgers.clear_all()
        detach_projection()
        clear_registry()
        cls.root = tempfile.mkdtemp(prefix="wl-fmnist-umap-")
        cls.saved_checkpoints = {}
        real_save = example.save_checkpoint

        def spy():
            step = real_save()
            # What the train table looked like when this checkpoint was written.
            cls.saved_checkpoints.setdefault(
                step, int(example.sample_table()["nb_seen"].astype(float).sum()))
            return step

        real_cut = example.lowest_loss_ids

        def cut_spy(table, fraction, **kwargs):
            ids = real_cut(table, fraction, **kwargs)
            cls.cut = (table.copy(), ids)           # the table the cut was made from
            return ids

        cls.served = mock.MagicMock()
        try:
            with mock.patch.object(example.datasets, "FashionMNIST", FakeFashionMNIST), \
                    mock.patch.object(example, "save_checkpoint", spy),                     mock.patch.object(example, "lowest_loss_ids", cut_spy), \
                    mock.patch.object(wl, "serve", cls.served), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                cls.summary = example.run(tiny_parameters(cls.root))
            cls.output = out.getvalue()
            # The end of the run, kept before the reload below rewinds it.
            cls.table = example.sample_table()
            cls.prefixes = set(known_prefixes())
            cls.reload = cls.reload_the_checkpoint()
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def reload_the_checkpoint(cls):
        """Train has gone on to step 36; reload step 24 and read what came back."""
        model = ledgers.get_model()
        found = {"age_before": model.get_age()}

        # The UMAP encoder and the custom projections are saved beside each
        # checkpoint under the same step number: the weights are the one file
        # that holds a model_state_dict.
        manager = ledgers.get_checkpoint_manager()
        files = sorted(Path(manager.models_dir).rglob(f"*_step_{STEPS_INITIAL:06d}.pt"))
        loaded = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
        weights = [c["model_state_dict"] for c in loaded
                   if isinstance(c, dict) and "model_state_dict" in c]
        found["checkpointed_weights"] = weights[-1] if weights else None

        found["age_after"] = example.reload_checkpoint(STEPS_INITIAL)
        found["weights_after"] = {k: v.detach().cpu().clone()
                                  for k, v in ledgers.get_model().state_dict().items()
                                  if torch.is_tensor(v)}
        found["seen_after"] = int(example.sample_table()["nb_seen"].astype(float).sum())
        return found

    @classmethod
    def tearDownClass(cls):
        wl.pause_training()                      # a fresh process starts paused
        detach_projection()
        clear_registry()
        ledgers.clear_all()
        wl_src.DATAFRAME_M = None                # clear_all orphans the cached handle
        shutil.rmtree(cls.root, ignore_errors=True)

    # -- the stages ------------------------------------------------------------
    def test_trains_to_the_checkpoint_step_and_keeps_a_checkpoint(self):
        self.assertEqual(self.summary["checkpoint_step"], STEPS_INITIAL)

    def test_the_reload_lands_on_the_checkpoint(self):
        self.assertEqual(self.summary["reloaded_age"], STEPS_INITIAL)

    def test_discards_the_requested_share_of_the_training_samples(self):
        self.assertEqual(self.summary["discarded"], int(TRAIN_SIZE * DISCARD))
        table = self.table
        self.assertEqual(int(table["discarded"].astype(bool).sum()), self.summary["discarded"])
        self.assertEqual(len(table), TRAIN_SIZE)                  # train split only

    def test_what_was_discarded_is_what_the_model_knew_best(self):
        """At the moment of the cut, every discarded sample has a loss no larger
        than every sample that stayed. (Not checkable on the end state: kept
        samples keep training afterwards, discarded ones are frozen.)"""
        table, ids = self.cut
        loss = pd.to_numeric(table[LOSS], errors="coerce")
        gone, kept = loss.loc[ids], loss.drop(index=ids).dropna()
        self.assertEqual(len(ids), self.summary["discarded"])
        self.assertLessEqual(gone.max(), kept.min())

    def test_the_pca_covers_exactly_what_was_kept(self):
        """The loader skips discarded samples, and the sweep sees the WHOLE split
        even though training was mid-epoch on it (the stateful-loader trap)."""
        kept = TRAIN_SIZE - self.summary["discarded"]
        self.assertEqual(self.summary["pca_samples"], kept)
        table = self.table
        placed = table[f"signals//pca2d_step{STEPS_INITIAL}_x"].notna()
        self.assertEqual(int(placed.sum()), kept)
        self.assertFalse((placed & table["discarded"].astype(bool)).any())

    def test_the_pca_is_two_dimensional(self):
        table = self.table
        self.assertIn(f"signals//pca2d_step{STEPS_INITIAL}_y", table.columns)
        self.assertNotIn(f"signals//pca2d_step{STEPS_INITIAL}_z", table.columns)

    def test_the_live_umap_placed_the_training_samples(self):
        table = self.table
        for axis in "xyz":
            self.assertGreater(int(table[f"signals//umap_{axis}"].notna().sum()), TRAIN_SIZE // 2)

    def test_both_layouts_are_scored(self):
        for value in (*self.summary["purity"].values(), self.summary["purity_features"]):
            self.assertTrue(0.0 <= value <= 1.0, value)
        self.assertEqual(set(self.summary["purity"]), {"umap", f"pca2d_step{STEPS_INITIAL}"})

    def test_the_figure_is_written(self):
        figure = Path(self.summary["log_dir"]) / "umap_vs_pca.png"
        self.assertTrue(figure.is_file())
        self.assertGreater(figure.stat().st_size, 5_000)

    def test_the_picker_lists_every_projection(self):
        self.assertIn("umap", self.prefixes)
        self.assertIn(f"pca2d_step{STEPS_INITIAL}", self.prefixes)
        self.assertIn(f"pca2d_step{STEPS_INITIAL + STEPS_RETRAIN}", self.prefixes)

    # -- the retraining ----------------------------------------------------------
    def test_neither_sweeps_nor_projections_age_the_model(self):
        """Five sweeps ran (PCA, PCA again, the UMAP's own placements): the model
        trained exactly 24 + 12 steps, not one more."""
        self.assertEqual(self.summary["final_age"], STEPS_INITIAL + STEPS_RETRAIN)

    def test_the_second_pca_is_a_new_projection_not_an_overwrite(self):
        after = f"pca2d_step{STEPS_INITIAL + STEPS_RETRAIN}"
        self.assertIn(after, self.summary["purity_after"])
        table = self.table
        self.assertEqual(int(table[f"signals//{after}_x"].notna().sum()),
                         TRAIN_SIZE - self.summary["discarded"])
        self.assertEqual(int(table[f"signals//pca2d_step{STEPS_INITIAL}_x"].notna().sum()),
                         TRAIN_SIZE - self.summary["discarded"])

    # -- the reload -------------------------------------------------------------
    def test_reload_brings_back_the_age_of_the_checkpoint(self):
        self.assertEqual(self.reload["age_before"], STEPS_INITIAL + STEPS_RETRAIN)
        self.assertEqual(self.reload["age_after"], STEPS_INITIAL)

    def test_reload_brings_back_the_weights_in_the_checkpoint_file(self):
        saved = self.reload["checkpointed_weights"]
        self.assertIsNotNone(saved, "no weights checkpoint was written at the checkpoint step")
        checked = 0
        for name, value in saved.items():
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                self.assertTrue(torch.equal(value, self.reload["weights_after"][name]), name)
                checked += 1
        self.assertGreater(checked, 4)

    def test_reload_rewinds_the_sample_counters_to_the_checkpoint(self):
        """The per-sample seen-counts go back too, instead of describing steps
        the reload just undid. They land within ONE batch of the checkpoint, not
        exactly on it: signals are logged at model age - 1 (see the TODO on
        ``src._get_step``), so a restore to step N still includes the signals the
        (N+1)-th batch logged at step N."""
        at_checkpoint = self.saved_checkpoints[STEPS_INITIAL]
        at_end = int(self.table["nb_seen"].astype(float).sum())
        after = self.reload["seen_after"]
        self.assertLess(after, at_end)
        self.assertGreaterEqual(after, at_checkpoint)
        self.assertLessEqual(after - at_checkpoint, 32)           # one batch

    # -- wiring ------------------------------------------------------------------
    def test_serves_the_studio_backend_as_configured(self):
        self.served.assert_called_once_with(serving_grpc=False)

    def test_announces_each_stage(self):
        for stage in ("[1. Train", "[2. Reload", "[3. Discard", "[4. PCA", "[5. Compare",
                      "[6. Retrain"):
            self.assertIn(stage, self.output)


# =============================================================================
# The docs
# =============================================================================
@unittest.skipUnless(DOCS.is_dir(), "the docs are not part of this checkout")
class TestTheDocs(unittest.TestCase):

    def read(self, *parts):
        return (DOCS.joinpath(*parts)).read_text(encoding="utf-8")

    def test_the_example_has_a_page_in_the_pytorch_section(self):
        page = self.read("examples", "pytorch", "fashion_mnist_umap.rst")
        self.assertIn("weightslab/examples/PyTorch/wl-fashion-mnist-umap/main.py", page)
        self.assertRegex(self.read("examples", "pytorch", "index.rst"),
                         r"(?m)^\s+fashion_mnist_umap\s*$")

    def test_the_usecase_page_points_at_the_script(self):
        page = self.read("examples", "usecases", "projection_fashion_mnist.rst")
        self.assertIn("PyTorch/wl-fashion-mnist-umap", page)

    def test_the_gallery_lists_the_example(self):
        gallery = self.read("_static", "examples-gallery.js")
        entry = re.search(r"\{[^{}]*url:\s*'examples/pytorch/fashion_mnist_umap\.html'[^{}]*\}",
                          gallery)
        self.assertIsNotNone(entry, "no gallery entry for the example")
        self.assertIn("badge: 'PyTorch'", entry.group(0))
        self.assertTrue((DOCS / "examples" / "pytorch" / "fashion_mnist_umap.rst").is_file())

    def test_no_link_to_the_sandbox_is_left(self):
        """The hosted sandbox is not linked from the docs for now."""
        for path in (DOCS / "index.rst", DOCS / "examples" / "index.rst",
                     DOCS / "_static" / "custom.css", REPO_ROOT / "README.md"):
            text = path.read_text(encoding="utf-8")
            with self.subTest(file=path.name):
                self.assertNotIn("sandbox.graybx.com", text)
                self.assertNotIn("wl-hero-cta-sandbox", text)
                self.assertNotIn("Open Sandbox", text)


if __name__ == "__main__":
    unittest.main()
