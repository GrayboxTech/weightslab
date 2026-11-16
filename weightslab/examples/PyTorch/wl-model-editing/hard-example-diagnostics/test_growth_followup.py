"""Exercise the actual public add/prune API used in the blog experiment."""
import tempfile
import unittest
from pathlib import Path

import torch
import weightslab as wl
from run_growth_followup import grow, learned_parameters, prune_added, register


class GrowthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = {"device": "cpu", "num_threads": 1,
                       "backbone": {"feature_width": 4},
                       "head": {"hidden_width": 3, "classes": 2, "negative_slope": .1},
                       "optimizer": {"name": "SGD", "lr": .1, "momentum": 0}}
        _, self.model, self.optimizer = register(self.config, Path(self.directory.name), "test", 17)
        self.x = torch.randn(12, 4)

    def tearDown(self):
        wl.clear_all()
        self.directory.cleanup()

    def test_preserving_growth_and_prune_roundtrip(self):
        with torch.no_grad():
            before = self.model(self.x).clone()
        event = grow(self.model, self.optimizer, 2, "zero_outgoing")
        self.assertEqual(event["after_width"], 5)
        with torch.no_grad():
            self.assertTrue(torch.allclose(before, self.model(self.x), atol=1e-6))
        prune_added(self.model, self.optimizer, 3, 2)
        with torch.no_grad():
            self.assertTrue(torch.allclose(before, self.model(self.x), atol=1e-6))

    def test_new_units_can_learn_after_zero_outgoing_initialization(self):
        grow(self.model, self.optimizer, 2, "zero_outgoing")
        before = learned_parameters(self.model)
        for _ in range(3):
            with wl.guard_training_context:
                self.optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(self.x), torch.arange(12) % 2)
                loss.backward()
                self.optimizer.step()
        after = learned_parameters(self.model)
        self.assertGreater(float(after["2"]["weight"][:, 3:].abs().sum()), 0)
        self.assertFalse(torch.equal(before["0"]["weight"][3:], after["0"]["weight"][3:]))

    def test_removal_cannot_touch_original_units(self):
        grow(self.model, self.optimizer, 2, "random")
        with self.assertRaises(ValueError):
            prune_added(self.model, self.optimizer, 3, 3)
        with self.assertRaises(ValueError):
            prune_added(self.model, self.optimizer, 3, -1)


if __name__ == "__main__":
    unittest.main()
