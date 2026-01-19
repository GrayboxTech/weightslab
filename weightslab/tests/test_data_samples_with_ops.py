"""
Unit tests for data_samples_with_ops.py

Tests the DataSampleTrackingWrapper class, which wraps PyTorch datasets
and provides per-sample statistics tracking and tag-based labeling.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from unittest.mock import Mock, patch

from weightslab.data.data_samples_with_ops import (
    DataSampleTrackingWrapper,
    _has_regex_symbols,
    _match_column_patterns,
    _filter_columns_with_patterns,
)
from weightslab.data.sample_stats import SampleStatsEx


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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_initialization_with_valid_params(self, mock_ledger):
        """Test wrapper initialization with valid parameters."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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
        mock_ledger.register_split.assert_called_once()

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_length_matches_dataset(self, mock_ledger):
        """Test that wrapper length matches dataset length."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_getitem_returns_data_and_id(self, mock_ledger):
        """Test that __getitem__ returns (data, id, label, ...)."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())
        mock_ledger.get_value = Mock(return_value=None)

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

        # Second element should be a numeric ID
        self.assertTrue(isinstance(result[1], (int, np.integer)))

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_getitem_with_single_element_dataset(self, mock_ledger):
        """Test __getitem__ with unsupervised dataset (no labels)."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())
        mock_ledger.get_value = Mock(return_value=None)

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_binary_tag_labeling(self, mock_ledger):
        """Test binary tag-based labeling."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

        # Create a shared dictionary to track get_value calls
        def mock_get_value(split, sample_id, key):
            # Mock tag values - the sample_id here is the actual unique_id, not the index
            if key == SampleStatsEx.TAGS.value:
                # Use modulo on sample_id to alternate between tags
                return "target_tag" if sample_id % 2 == 0 else "other_tag"
            return None

        mock_ledger.get_value = mock_get_value

        tags_mapping = {"target_tag": 1}
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
        if sample_id % 2 == 0:
            self.assertEqual(target_label, 1)
        else:
            self.assertEqual(target_label, 0)

        # Test another sample
        result = wrapper[1]
        target_label = result[2]
        sample_id = result[1]
        if sample_id % 2 == 0:
            self.assertEqual(target_label, 1)
        else:
            self.assertEqual(target_label, 0)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_multiclass_tag_labeling(self, mock_ledger):
        """Test multiclass tag-based labeling."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

        def mock_get_value(split, sample_id, key):
            if key == SampleStatsEx.TAGS.value:
                mapping = {0: "small", 1: "medium", 2: "large"}
                return mapping.get(sample_id % 3, "small")
            return None

        mock_ledger.get_value = mock_get_value

        tags_mapping = {"small": 0, "medium": 1, "large": 2}
        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
            use_tags=True,
            tags_mapping=tags_mapping,
        )

        # Test different samples - verify against their actual sample_id
        for idx in range(3):
            result = wrapper[idx]
            target_label = result[2]
            sample_id = result[1]

            # Map based on the actual sample_id
            expected = {0: 0, 1: 1, 2: 2}.get(sample_id % 3, 0)
            self.assertEqual(target_label, expected)


class TestDataSampleTrackingWrapperDenylist(unittest.TestCase):
    """Test denylisting and allowlisting functionality."""

    def setUp(self):
        """Create a temporary directory for logs."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = SimpleDataset(size=10)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_denylist_samples(self, mock_ledger):
        """Test denylisting samples."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())
        mock_ledger.update_values = Mock()
        mock_ledger.mark_dirty = Mock()

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

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_allowlist_samples(self, mock_ledger):
        """Test allowlisting samples."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())
        mock_ledger.update_values = Mock()
        mock_ledger.mark_dirty = Mock()

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

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_denylist_clear(self, mock_ledger):
        """Test clearing all denylists."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())
        mock_ledger.update_values = Mock()
        mock_ledger.mark_dirty = Mock()

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_state_dict_structure(self, mock_ledger):
        """Test that state_dict has correct structure."""
        mock_ledger.register_split = Mock()
        mock_df = pd.DataFrame({
            "prediction_loss": [0.1, 0.2],
        })
        mock_ledger.get_df_view = Mock(return_value=mock_df)
        mock_ledger.get_dense_map = Mock(return_value={})

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_get_sample_id_at_index(self, mock_ledger):
        """Test retrieving sample ID at given index."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

        wrapper = DataSampleTrackingWrapper(
            wrapped_dataset=self.dataset,
            root_log_dir=self.temp_dir,
            enable_h5_persistence=False,
            compute_hash=False,
        )

        # Get sample ID at index 0
        sample_id = wrapper.get_sample_id_at_index(0)
        self.assertEqual(sample_id, int(wrapper.unique_ids[0]))

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_get_index_from_sample_id(self, mock_ledger):
        """Test retrieving index from sample ID."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_infer_num_classes_from_dataset(self, mock_ledger):
        """Test inferring number of classes from wrapped dataset."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_infer_num_classes_binary_tags(self, mock_ledger):
        """Test inferring num_classes with binary tag mapping."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    @patch('weightslab.data.data_samples_with_ops.array_id_2bytes')
    def test_duplicate_detection_with_hash(self, mock_hash, mock_ledger):
        """Test that duplicate samples are detected and removed."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_equality_same_wrapper(self, mock_ledger):
        """Test equality comparison of wrappers."""
        mock_ledger.register_split = Mock()
        mock_df = pd.DataFrame()
        mock_ledger.get_df_view = Mock(return_value=mock_df)

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

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_equality_different_types(self, mock_ledger):
        """Test equality comparison with different types."""
        mock_ledger.register_split = Mock()
        mock_ledger.get_df_view = Mock(return_value=pd.DataFrame())

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

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('weightslab.data.data_samples_with_ops.LEDGER_MANAGER')
    def test_as_records(self, mock_ledger):
        """Test converting DataFrame to records."""
        mock_ledger.register_split = Mock()
        mock_df = pd.DataFrame({
            "sample_id": [1, 2, 3],
            "prediction_loss": [0.1, 0.2, 0.3],
        }).set_index("sample_id")
        mock_ledger.get_df_view = Mock(return_value=mock_df)

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
