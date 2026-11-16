"""Small CPU checks for the numerical helpers; no dataset or pretrained download."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from attribute_waterbirds import integrated_gradients
from prepare_waterbirds import sha256
from run_waterbirds import schedule


class ExperimentTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(hashlib, "file_digest"), "file_digest requires Python 3.11+")
    def test_sha256_uses_file_digest_when_available(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            for data in (b"", b"waterbirds", b"x" * (2 * 1024 * 1024 + 7)):
                with self.subTest(size=len(data)):
                    path.write_bytes(data)
                    with mock.patch("hashlib.file_digest", wraps=hashlib.file_digest) as digest:
                        self.assertEqual(sha256(path), hashlib.sha256(data).hexdigest())
                        digest.assert_called_once()
                        self.assertEqual(digest.call_args.args[1], "sha256")

    def test_sha256_streams_without_file_digest_on_python_310(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            # Remove the API entirely to reproduce Python 3.10, even on newer Python.
            with mock.patch.object(hashlib, "file_digest", create=True):
                del hashlib.file_digest
                for data in (b"", b"waterbirds", b"x" * (2 * 1024 * 1024 + 7)):
                    with self.subTest(size=len(data)):
                        path.write_bytes(data)
                        self.assertEqual(sha256(path), hashlib.sha256(data).hexdigest())

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
