"""Reject unsupported live edits before loading or modifying a model."""
import unittest
from live_growth_lab import validate_request


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(seed=17, sampling="original", initialization="zero_outgoing",
                            stage="immediate", prune=0, split="validation")

    def test_supported_request(self):
        self.assertEqual(validate_request(self.request, [17, 29, 43]), self.request)

    def test_invalid_values(self):
        for key, value in (("seed", True), ("seed", 99), ("prune", True), ("prune", -1),
                           ("prune", 17), ("sampling", "test"), ("initialization", "unknown"),
                           ("stage", "train_now"), ("split", "train")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_request({**self.request, key: value}, [17, 29, 43])

    def test_no_growth_has_nothing_to_prune(self):
        with self.assertRaises(ValueError):
            validate_request({**self.request, "initialization": "none", "prune": 4}, [17])

    def test_no_extra_fields_or_arbitrary_paths(self):
        for request in (None, [], {}, {**self.request, "checkpoint": "/tmp/model.pt"}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_request(request, [17])


if __name__ == "__main__":
    unittest.main()
