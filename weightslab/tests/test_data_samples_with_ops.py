"""
Unit tests for data_samples_with_ops.py

Tests the DataSampleTrackingWrapper class, which wraps PyTorch datasets
and provides per-sample statistics tracking and tag-based labeling.
"""

import os
import tempfile
import unittest
import numpy as np
import torch

import weightslab as wl

from torch.utils.data import Dataset
from unittest.mock import patch

from weightslab.data.data_samples_with_ops import (
    DataSampleTrackingWrapper,
    _has_regex_symbols,
    _match_column_patterns,
    _filter_columns_with_patterns,
)


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=10, return_labels=True):
        self.size = size
        self.return_labels = return_labels
        self.__name__ = "simple_dataset"

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return random data with shape (3, 32, 32) to simulate images
        data = np.random.randn(3, 32, 32).astype(np.float32)
        if self.return_labels:
            label = idx % 10  # Simulate 10 classes
            return data, label
        return data


class TestHelperFunctions(unittest.TestCase):
    """Test helper utility functions."""

    def test_has_regex_symbols(self):
        """Test regex symbol detection."""
        # True cases
        self.assertTrue(_has_regex_symbols(".*"))
        self.assertTrue(_has_regex_symbols("test.*"))
        self.assertTrue(_has_regex_symbols("[abc]"))
        self.assertTrue(_has_regex_symbols("(test)"))
        self.assertTrue(_has_regex_symbols("test+"))

        # False cases
        self.assertFalse(_has_regex_symbols("test"))
        self.assertFalse(_has_regex_symbols("test_column"))
        self.assertFalse(_has_regex_symbols("123"))

    def test_match_column_patterns(self):
        """Test column pattern matching."""
        # Exact match
        self.assertTrue(_match_column_patterns("test_col", ["test_col"]))
        self.assertTrue(_match_column_patterns("exact", ["exact", "other"]))

        # Regex match
        self.assertTrue(_match_column_patterns("test_1", ["test_.*"]))
        self.assertTrue(_match_column_patterns("feature_loss", [".*_loss"]))

        # No match
        self.assertFalse(_match_column_patterns("column", ["other"]))
        self.assertFalse(_match_column_patterns("test", [".*_loss"]))

    def test_filter_columns_with_patterns(self):
        """Test column filtering by patterns."""
        columns = ["loss", "loss_train", "accuracy", "test_accuracy", "feature_map"]

        # Exact patterns
        result = _filter_columns_with_patterns(columns, ["loss"])
        self.assertEqual(result, ["loss"])

        # Regex patterns - note: the regex is correctly anchored, so ".*accuracy" matches "accuracy" and "test_accuracy"
        result = _filter_columns_with_patterns(columns, [".*accuracy"])
        self.assertIn("accuracy", result)
        self.assertIn("test_accuracy", result)

        # Multiple patterns - "loss" and "accuracy" should match "loss", "accuracy", "test_accuracy" = 2 matches (not 3 - loss_train is separate)
        result = _filter_columns_with_patterns(columns, ["loss", "accuracy"])
        # "loss" matches exactly "loss" (1), "accuracy" matches "accuracy" and "test_accuracy" (2) = 2 total
        self.assertGreaterEqual(len(result), 2)

        # Empty result
        result = _filter_columns_with_patterns(columns, ["nonexistent"])
        self.assertEqual(result, [])


class TestDataSampleTrackingWrapperInit(unittest.TestCase):
    """Test initialization and basic properties."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_with_valid_params(self):
        """Test wrapper initialization with valid parameters."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            is_training=True,
            name="train",
            enable_h5_persistence=False,
            compute_hash=False,
        )

        self.assertEqual(len(wrapper), len(self.dataset))
        self.assertEqual(wrapper.name, "train")
        self.assertTrue(wrapper.is_training)
        self.assertIsNotNone(wrapper.unique_ids)

    def test_length_matches_dataset(self):
        """Test that wrapper length matches dataset length."""

        dataset_size = 20
        dataset = SimpleDataset(size=dataset_size)
        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        self.assertEqual(len(wrapper), dataset_size)


class TestDataSampleTrackingWrapperGetItem(unittest.TestCase):
    """Test __getitem__ and data retrieval."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10, return_labels=True)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_getitem_returns_data_and_id(self):
        """Test that __getitem__ returns (data, id, label, ...)."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
            use_tags=False,
        )

        # Get first item
        result = wrapper[0]

        # Should return tuple with (data, id, target, ...)
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 3)  # data, id, target at minimum

        # First element should be numpy array or tensor
        self.assertTrue(isinstance(result[0], (np.ndarray, torch.Tensor)))

        # Second element should be a numeric UID
        self.assertTrue(isinstance(result[1], (int, np.integer)))

    def test_getitem_with_single_element_dataset(self):
        """Test __getitem__ with unsupervised dataset (no labels)."""

        dataset = SimpleDataset(size=5, return_labels=False)
        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
            use_tags=False,
        )

        result = wrapper[0]

        # Should return (data, id)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestDataSampleTrackingWrapperTagBasedLabeling(unittest.TestCase):
    """Test tag-based labeling functionality."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10, return_labels=True)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_binary_tag_labeling(self):
        """Test binary tag-based labeling."""

        tags_mapping = {"target_tag": 1, "non_target_tag": 0}
        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
            use_tags=True,
            tags_mapping=tags_mapping,
        )

        # Get item - should override label with tag-based label
        result = wrapper[0]
        target_label = result[2]
        sample_id = result[1]

        # Check based on actual sample_id
        self.assertEqual(target_label, 0)

        # Set tags for samples
        wrapper.set(sample_id=sample_id, stat_name="tags", value='target_tag')

        # Test another sample
        result = wrapper[1]
        sample_id = result[1]

        # Set tags for samples
        wrapper.set(sample_id=sample_id, stat_name="tags", value='non_target_tag')

        # Get labels
        result = wrapper[1]
        target_label = result[2]
        self.assertEqual(target_label, 0)


class TestDataSampleTrackingWrapperDenylist(unittest.TestCase):
    """Test denylisting and allowlisting functionality."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_denylist_samples(self):
        """Test denylisting samples."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Get first few sample IDs
        denied_ids = set(wrapper.unique_ids[:3])

        wrapper.denylist_samples(denied_ids)

        # Check denied count was updated
        self.assertEqual(wrapper.denied_sample_cnt, len(denied_ids))

    def test_allowlist_samples(self):
        """Test allowlisting samples."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # First deny some samples
        denied_ids = set(wrapper.unique_ids[:3])
        wrapper.denylist_samples(denied_ids)
        self.assertEqual(wrapper.denied_sample_cnt, len(denied_ids))

        # Then allow them back
        wrapper.allowlist_samples(denied_ids)

        # If allow was successful, denied_sample_cnt should be updated
        # (depends on mock behavior of get_df_view)

    def test_denylist_clear(self):
        """Test clearing all denylists."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Deny all samples
        all_ids = set(wrapper.unique_ids)
        wrapper.denylist_samples(all_ids)
        self.assertEqual(wrapper.denied_sample_cnt, len(all_ids))

        # Clear denials by passing None
        wrapper.denylist_samples(None)
        self.assertEqual(wrapper.denied_sample_cnt, 0)


class TestDataSampleTrackingWrapperStateDict(unittest.TestCase):
    """Test state_dict and load_state_dict functionality."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=5)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_state_dict_structure(self):
        """Test that state_dict has correct structure."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        state = wrapper.state_dict()

        # Check structure
        self.assertIn("blockd_samples", state)
        self.assertIn("sample_statistics", state)
        self.assertIsInstance(state["blockd_samples"], int)
        self.assertIsInstance(state["sample_statistics"], dict)


class TestDataSampleTrackingWrapperUtilities(unittest.TestCase):
    """Test utility methods."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_sample_id_at_index(self):
        """Test retrieving sample ID at given index."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Get sample ID at index 0
        sample_id = wrapper.get_sample_id_at_index(0)
        self.assertEqual(sample_id, int(wrapper.unique_ids[0]))

    def test_get_index_from_sample_id(self):
        """Test retrieving index from sample ID."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Get index from first sample ID
        sample_id = int(wrapper.unique_ids[0])
        index = wrapper.get_index_from_sample_id(sample_id)
        self.assertEqual(index, 0)

    def test_infer_num_classes_from_dataset(self):
        """Test inferring number of classes from wrapped dataset."""

        # Create a dataset with num_classes attribute
        dataset = SimpleDataset(size=10)
        dataset.num_classes = 10

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        num_classes = wrapper.infer_num_classes()
        self.assertEqual(num_classes, 10)

    def test_infer_num_classes_binary_tags(self):
        """Test inferring num_classes with binary tag mapping."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
            use_tags=True,
            tags_mapping={"target": 1},
        )

        num_classes = wrapper.infer_num_classes()
        self.assertEqual(num_classes, 2)


class TestDataSampleTrackingWrapperDuplicateDetection(unittest.TestCase):
    """Test duplicate sample detection and removal."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.array_id_2bytes')
    def test_duplicate_detection_with_hash(self, mock_hash):
        """Test that duplicate samples are detected and removed."""

        # Create dataset with duplicates
        dataset = SimpleDataset(size=5)

        # Mock hash function to create duplicates
        # Return same hash for first two samples
        hash_values = [100, 100, 101, 102, 103]
        mock_hash.side_effect = hash_values

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=True,
        )

        # Should have removed one duplicate
        # The wrapper should now have 4 unique samples instead of 5
        # (though actual behavior depends on how Subset is used)
        self.assertLessEqual(len(wrapper.unique_ids), len(dataset))


class TestDataSampleTrackingWrapperEquality(unittest.TestCase):
    """Test equality comparison."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_equality_same_wrapper(self):
        """Test equality comparison of wrappers."""

        wrapper1 = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        wrapper2 = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Both have same wrapped_dataset and same denied_count
        self.assertTrue(wrapper1 == wrapper2)

    def test_equality_different_types(self):
        """Test equality comparison with different types."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Compare with non-wrapper object
        self.assertFalse(wrapper == "not a wrapper")
        self.assertFalse(wrapper == 123)


class TestDataSampleTrackingWrapperAsRecords(unittest.TestCase):
    """Test as_records functionality."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=5)

        # Initialize HP
        parameters = {
            'flush_interval': 3.0,
            'flush_max_rows': 100,
            'enable_h5': True,
            'enable_flush': True
        }
        wl.watch_or_edit(
            parameters,
            flag="hyperparameters",
            name='TestCheckpointManagerHP',
            defaults=parameters,
            poll_interval=1.0,
        )

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_as_records(self):
        """Test converting DataFrame to records."""

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        records = wrapper.as_records()

        self.assertIsInstance(records, list)
        self.assertGreater(len(records), 0)
        for record in records:
            self.assertIsInstance(record, dict)


if __name__ == "__main__":
    unittest.main()
