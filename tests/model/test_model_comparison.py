"""Compare A vs B (the Modeling tab): per-layer divergence and per-neuron
effects, in pair mode (one sample each) and group mode (sets of samples)."""
import threading
import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn

from weightslab.components import model_comparison as mc
from weightslab.components import model_representation as mr


def setUpModule():
    # A real lock, as in production: a comparison that replaces a running
    # one must not run its forwards at the same time as the old one (both
    # would hook the same modules from two threads).
    global _lock_patch
    _lock_patch = mock.patch(
        "weightslab.components.global_monitoring.weightslab_rlock", threading.RLock())
    _lock_patch.start()


def tearDownModule():
    _lock_patch.stop()


class _Tracked:
    """The parts of a tracked dataset the sample loader touches."""

    def __init__(self, items, split, first_id):
        self.items = items
        self._dataset_split = split
        self.unique_ids = [str(first_id + i) for i in range(len(items))]

    def get_index_from_sample_id(self, sid):
        return self.unique_ids.index(str(sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i], self.unique_ids[i], 0


class _Loader:
    def __init__(self, items, split, first_id):
        self.tracked_dataset = _Tracked(items, split, first_id)

    def get_dataloader(self):
        return None


class _SignNet(nn.Module):
    """conv channel 0 copies the input, channel 1 negates it, channel 2 is
    dead; then ReLU, pooling and a head. Positive inputs fire channel 0 only,
    negative ones channel 1 only."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 1, bias=False)
        with th.no_grad():
            self.conv.weight.copy_(th.tensor([1.0, -1.0, 0.0]).view(3, 1, 1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))


def _wait(manager, status):
    assert status["state"] == "running", status
    session = manager.wait(status["comparison_id"], timeout=60)
    assert session.state == "done", session.message
    return session


class TestGroupMode(unittest.TestCase):
    def setUp(self):
        g = th.Generator().manual_seed(0)
        positive = [th.rand(1, 6, 6, generator=g) + 0.1 for _ in range(30)]       # ids 1000..
        negative = [-(th.rand(1, 6, 6, generator=g) + 0.1) for _ in range(30)]    # ids 2000..
        mixed = [th.rand(1, 6, 6, generator=g) - 0.5 for _ in range(80)]          # ids 3000..
        self.loaders = {
            "pos": _Loader(positive, "train", 1000),
            "neg": _Loader(negative, "train", 2000),
            "mix": _Loader(mixed, "test", 3000),
        }
        self.rep = mr.ModelRepresentationManager()
        self.manager = mc.ComparisonManager(self.rep)
        self.model = _SignNet()

    @staticmethod
    def _ids(first, n):
        return [str(first + i) for i in range(n)]

    def test_a_layer_that_separates_a_from_b_scores_high_and_names_its_neurons(self):
        status = self.manager.start(self.model, self.loaders,
                                    {"sample_ids": self._ids(1000, 30), "label": "positive"},
                                    {"sample_ids": self._ids(2000, 30)})
        self.assertEqual(status["mode"], "group")
        session = _wait(self.manager, status)
        self.assertEqual(session.done, 60)
        self.assertEqual(session.a.label, "positive")
        self.assertEqual(session.b.label, "30 samples")
        diffs = {d["node_id"]: d for d in self.manager.diffs(session)}
        for node in (mr.INPUT_NODE, "conv", "relu", "pool", "fc", mr.OUTPUT_NODE):
            self.assertIn(node, diffs)

        relu = diffs["relu"]
        self.assertEqual(relu["channels"], 3)
        self.assertEqual(relu["only_a"], 1)          # channel 0
        self.assertEqual(relu["only_b"], 1)          # channel 1
        self.assertEqual(relu["strong_channels"], 2)  # channel 2 is dead on both sides
        self.assertGreater(relu["score"], 3.0)
        self.assertGreater(relu["rate_a"], 0.3)

        layer = self.manager.layer(session.id, "relu")
        self.assertTrue(layer["success"], layer["message"])
        self.assertEqual(layer["neuron_axis"], "channel")
        n0, n1, n2 = layer["neurons"]
        self.assertGreater(n0["effect"], 3.0)        # stronger on A
        self.assertLess(n1["effect"], -3.0)          # stronger on B
        self.assertEqual(n2["effect"], 0.0)          # dead on both: no difference
        self.assertEqual((n0["rate_a"], n0["rate_b"]), (1.0, 0.0))
        self.assertEqual((n1["rate_a"], n1["rate_b"]), (0.0, 1.0))
        self.assertGreater(n0["std_a"], 0.0)
        self.assertEqual(self.manager.layer(session.id, "nope")["success"], False)

    def test_two_draws_of_one_distribution_look_alike(self):
        status = self.manager.start(self.model, self.loaders,
                                    {"sample_ids": self._ids(3000, 40)},
                                    {"sample_ids": self._ids(3040, 40)})
        session = _wait(self.manager, status)
        diffs = {d["node_id"]: d for d in self.manager.diffs(session)}
        for node in ("conv", "relu", "fc"):
            self.assertLess(diffs[node]["score"], 0.6, node)
            self.assertEqual(diffs[node]["only_a"] + diffs[node]["only_b"], 0, node)

    def test_overlap_is_left_out_of_both_sides_and_sides_are_capped(self):
        status = self.manager.start(self.model, self.loaders,
                                    {"sample_ids": self._ids(3000, 10)},
                                    {"sample_ids": self._ids(3006, 30)}, max_samples=8)
        session = _wait(self.manager, status)
        self.assertEqual(status["overlap_dropped"], 4)
        self.assertEqual(status["a"]["requested"], 10)
        self.assertEqual(status["a"]["count"], 6)
        self.assertEqual(status["b"]["count"], 8)            # 26 left, capped
        self.assertFalse(set(status["a"]["sample_ids"]) & set(status["b"]["sample_ids"]))
        self.assertEqual(session.total, 14)
        # The cap keeps an even, reproducible subset, not the first 8.
        again = self.manager.start(self.model, self.loaders,
                                   {"sample_ids": self._ids(3000, 10)},
                                   {"sample_ids": self._ids(3006, 30)}, max_samples=8)
        self.assertEqual(again["b"]["sample_ids"], status["b"]["sample_ids"])
        self.assertNotEqual(status["b"]["sample_ids"], self._ids(3010, 8))
        self.manager.wait(again["comparison_id"])

    def test_errors_empty_and_identical_sides(self):
        same = self.manager.start(self.model, self.loaders,
                                  {"sample_ids": self._ids(1000, 3)},
                                  {"sample_ids": self._ids(1000, 3)})
        self.assertEqual(same["state"], "error")
        self.assertIn("also in the other side", same["message"])
        missing = self.manager.start(self.model, self.loaders,
                                     {"sample_ids": ["nope"]}, {"sample_ids": ["1000"]})
        self.assertEqual(missing["state"], "error")
        self.assertIn("could be loaded", missing["message"])
        empty = self.manager.start(self.model, self.loaders, {}, {"sample_ids": ["1000"]})
        self.assertIn("A has no samples", empty["message"])

    def test_tags_resolve_through_the_dataframe(self):
        tagged = {"pos": (self._ids(1000, 5), ["train"] * 5), "neg": (self._ids(2000, 5), ["train"] * 5)}
        with mock.patch.object(mc, "samples_with_tag", side_effect=lambda tag, origin="": tagged[tag]):
            status = self.manager.start(self.model, self.loaders, {"tag": "pos"}, {"tag": "neg"})
        session = _wait(self.manager, status)
        self.assertEqual((session.a.label, session.b.label), ("tag 'pos'", "tag 'neg'"))
        self.assertEqual(status["a"]["tag"], "pos")
        self.assertEqual(status["b"]["count"], 5)

    def test_staleness_follows_the_model_edits(self):
        session = _wait(self.manager, self.manager.start(
            self.model, self.loaders, {"sample_ids": ["1000", "1001"]}, {"sample_ids": ["2000"]}))
        self.assertFalse(self.manager.is_stale(session))
        self.rep.note_edit(structural=False)
        self.assertTrue(self.manager.is_stale(session))

    def test_a_new_comparison_replaces_the_running_one(self):
        first = self.manager.start(self.model, self.loaders,
                                   {"sample_ids": self._ids(3000, 40)}, {"sample_ids": self._ids(3040, 40)})
        second = self.manager.start(self.model, self.loaders,
                                    {"sample_ids": ["1000"]}, {"sample_ids": ["2000"]})
        self.manager.wait(first["comparison_id"])
        self.assertIn(self.manager.get(first["comparison_id"]).state, ("cancelled", "done"))
        _wait(self.manager, second)
        self.assertEqual(self.manager.get().id, second["comparison_id"])
        self.manager.clear()
        self.assertIsNone(self.manager.get())


class TestPairMode(unittest.TestCase):
    def setUp(self):
        g = th.Generator().manual_seed(1)
        base = th.zeros(1, 8, 8)
        base[0, 1:4, 1:4] = th.rand(3, 3, generator=g) + 0.5
        shifted = th.roll(base, shifts=(4, 4), dims=(1, 2))   # same content, elsewhere
        other = th.rand(1, 8, 8, generator=g)
        self.loaders = {"train": _Loader([base, base.clone(), shifted, other], "train", 10)}
        self.manager = mc.ComparisonManager(mr.ModelRepresentationManager())
        self.model = _SignNet()

    def _pair(self, a, b):
        status = self.manager.start(self.model, self.loaders, {"sample_ids": [a]}, {"sample_ids": [b]})
        self.assertEqual(status["mode"], "pair")
        self.assertEqual(status["a"]["count"], 1)
        session = _wait(self.manager, status)
        return session, {d["node_id"]: d for d in self.manager.diffs(session)}

    def test_identical_samples_do_not_diverge_anywhere(self):
        session, diffs = self._pair("10", "11")
        self.assertEqual(session.a.label, "sample 10")
        for node, diff in diffs.items():
            self.assertAlmostEqual(diff["score"], 0.0, places=6, msg=node)

    def test_where_things_fire_matters_not_only_how_much(self):
        # Shifted copy: per-channel means are equal (the pooled vectors agree),
        # but the activation maps do not overlap -> the conv/relu score is
        # ~1 (unrelated), and the pooling layer is where they meet again.
        _, diffs = self._pair("10", "12")
        self.assertAlmostEqual(diffs["relu"]["cosine_distance"], 0.0, places=5)
        self.assertGreater(diffs["relu"]["score"], 0.9)
        self.assertGreater(diffs[mr.INPUT_NODE]["score"], 0.9)
        self.assertAlmostEqual(diffs["pool"]["score"], 0.0, places=5)
        self.assertAlmostEqual(diffs["fc"]["score"], 0.0, places=5)

    def test_pair_neurons_fire_or_not(self):
        session, diffs = self._pair("10", "13")
        layer = self.manager.layer(session.id, "relu")
        self.assertTrue(layer["success"])
        self.assertEqual(layer["mode"], "pair")
        rates = [(n["rate_a"], n["rate_b"]) for n in layer["neurons"]]
        self.assertEqual(rates[0], (1.0, 1.0))      # both positive inputs: channel 0 fires
        self.assertEqual(rates[1], (0.0, 0.0))      # neither negative
        self.assertEqual(layer["neurons"][2]["effect"], 0.0)


class TestScores(unittest.TestCase):
    def test_normalized_squared_distance(self):
        a = np.array([1.0, 0.0])
        self.assertAlmostEqual(mc._normalized_sq_distance(a, a), 0.0)
        self.assertAlmostEqual(mc._normalized_sq_distance(a, np.array([0.0, 1.0])), 1.0)
        self.assertAlmostEqual(mc._normalized_sq_distance(a, -a), 2.0)
        self.assertAlmostEqual(mc._normalized_sq_distance(a, 0.5 * a), 0.2)   # weaker = farther
        self.assertEqual(mc._normalized_sq_distance(np.zeros(3), np.zeros(3)), 0.0)

    def test_effects_are_clipped_and_signed(self):
        a, b = mc.SideStats(), mc.SideStats()
        a.update(th.tensor([[1.0, 5.0], [1.0, 5.2]]), th.ones(2, 2), 0.0, 0)
        b.update(th.tensor([[1.0, 0.0], [1.0, 0.1]]), th.ones(2, 2), 0.0, 0)
        effect = mc.neuron_effects(a, b, "group")
        self.assertEqual(effect[0], 0.0)
        self.assertEqual(effect[1], mc.EFFECT_CLIP)


class TestService(unittest.TestCase):
    def test_status_and_diffs_convert_to_protobuf(self):
        from weightslab.trainer.services.model_service import ModelService
        loaders = {"train": _Loader([th.rand(1, 4, 4) for _ in range(4)], "train", 0)}
        manager = mc.ComparisonManager(mr.ModelRepresentationManager())
        status = manager.start(_SignNet(), loaders, {"sample_ids": ["0", "1"]}, {"sample_ids": ["2"]})
        session = _wait(manager, status)
        pb = ModelService._comparison_status_pb(session.status(False))
        self.assertEqual(pb.mode, "group")
        self.assertEqual(list(pb.a.sample_ids), ["0", "1"])
        self.assertEqual(pb.b.count, 1)
        diffs = [ModelService._node_diff_pb(d) for d in manager.diffs(session)]
        self.assertTrue(any(d.node_id == "relu" and d.channels == 3 for d in diffs))


if __name__ == "__main__":
    unittest.main()
