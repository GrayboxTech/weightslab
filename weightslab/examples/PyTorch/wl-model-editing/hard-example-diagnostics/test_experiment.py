"""Small CPU checks for the numerical helpers; no dataset or pretrained download."""

import unittest

import torch
from attribute_waterbirds import integrated_gradients
from run_waterbirds import schedule


class ExperimentTests(unittest.TestCase):
    def test_original_schedule_is_float_valid_and_reproducible(self):
        rows = [{"group": group} for group in ("0:0", "0:1", "1:0", "1:1")]
        first = schedule(rows, 17, "original", 10, 8)
        self.assertEqual(tuple(first.shape), (10, 8))
        self.assertTrue(torch.equal(first, schedule(rows, 17, "original", 10, 8)))
        self.assertFalse(torch.equal(first, schedule(rows, 29, "original", 10, 8)))
        self.assertTrue(bool(((first >= 0) & (first < len(rows))).all()))

    def test_balanced_schedule_reweights_unequal_groups(self):
        rows = [{"group": group} for group, count in (("0:0", 100), ("0:1", 10), ("1:0", 2), ("1:1", 30)) for _ in range(count)]
        sampled = schedule(rows, 17, "group_balanced", 100, 64).flatten().tolist()
        for group in ("0:0", "0:1", "1:0", "1:1"):
            fraction = sum(rows[i]["group"] == group for i in sampled) / len(sampled)
            self.assertLess(abs(fraction - 0.25), 0.03)

    def test_integrated_gradients_matches_linear_ground_truth(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2))
        model.requires_grad_(False)
        image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) / 12
        attribution, delta, _ = integrated_gradients(model, image, 1, 8, 3)
        expected = image * model[1].weight[1].reshape(1, 3, 2, 2)
        self.assertTrue(torch.allclose(attribution, expected, atol=1e-6))
        self.assertLess(abs(delta), 1e-6)

    def test_integrated_gradients_zero_input_has_zero_attribution(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2), torch.nn.Tanh())
        model.requires_grad_(False)
        attribution, delta, difference = integrated_gradients(model, torch.zeros(1, 3, 2, 2), 0, 8, 3)
        self.assertEqual(float(attribution.abs().sum()), 0)
        self.assertEqual(delta, 0)
        self.assertEqual(difference, 0)


if __name__ == "__main__":
    unittest.main()
