"""Unit tests for tabular task_type auto-detection in data_service.py.

Datasets whose model input is a 1-D feature vector (no image) should be
auto-detected as task_type "tabular" — distinct from "classification" — even
when the dataset/model never sets an explicit `task_type` attribute. See
`peek_is_tabular_sample` (used as the last step of the "Robust Task Type
Detection" heuristic in `DataService._process_sample_row`).
"""

import unittest

import numpy as np
import torch

from weightslab.trainer.services.data_service import peek_is_tabular_sample


class _TabularDataset:
    """Minimal stand-in for a tabular Dataset (ledger get_items contract).

    ``__getitem__`` is required only because ``load_raw_image_array`` uses its
    presence as an "is this dataset-like" gate before consulting ``get_items``.
    """

    def __init__(self, n=8, n_features=16):
        self.features = torch.randn(n, n_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], idx, 0

    def get_index_from_sample_id(self, sample_id):
        return sample_id if 0 <= sample_id < len(self.features) else None

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        image = self.features[idx] if include_images else None
        return image, idx, 0, None


class _ImageDataset:
    """Minimal stand-in for a vision Dataset (H×W×C image input)."""

    def __init__(self, n=8):
        self.images = torch.randint(0, 255, (n, 32, 32, 3), dtype=torch.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], idx, 0

    def get_index_from_sample_id(self, sample_id):
        return sample_id if 0 <= sample_id < len(self.images) else None

    def get_items(self, idx, include_metadata=False, include_labels=False, include_images=False):
        image = self.images[idx] if include_images else None
        return image, idx, 0, None


class TestPeekIsTabularSample(unittest.TestCase):
    def test_true_for_1d_feature_vector_input(self):
        ds = _TabularDataset()
        self.assertTrue(peek_is_tabular_sample(ds, 0))
        self.assertTrue(peek_is_tabular_sample(ds, 3))

    def test_false_for_image_input(self):
        ds = _ImageDataset()
        self.assertFalse(peek_is_tabular_sample(ds, 0))

    def test_false_when_sample_id_unresolvable(self):
        ds = _TabularDataset(n=4)
        self.assertFalse(peek_is_tabular_sample(ds, 999))

    def test_false_on_dataset_without_expected_lookup_methods(self):
        # No get_physical_location / get_index_from_sample_id / __getitem__
        # that behaves as expected -> best-effort heuristic, never raises.
        class Broken:
            pass

        self.assertFalse(peek_is_tabular_sample(Broken(), 0))

    def test_respects_get_physical_location_when_present(self):
        calls = []

        class GroupedTabularDataset(_TabularDataset):
            def get_physical_location(self, sample_id):
                calls.append(sample_id)
                return sample_id, 0

        ds = GroupedTabularDataset()
        self.assertTrue(peek_is_tabular_sample(ds, 2))
        self.assertEqual(calls, [2])


if __name__ == "__main__":
    unittest.main()
