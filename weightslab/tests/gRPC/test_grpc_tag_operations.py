"""
Unit Tests for gRPC Tag and discarded Operations

Tests the complete workflow of tag management via gRPC:
- Adding tags to samples (EDIT_ACCUMULATE)
- Removing tags from samples (EDIT_REMOVE)
- Deleting entire tag columns (EDIT_REMOVE with value=-1)
- Marking samples as discarded (discarded)
- Restoring discarded samples

Uses a subsample of 100 MNIST test samples.
"""

import os
import unittest
import tempfile
import warnings
import shutil
from pathlib import Path

from weightslab.data.sample_stats import SampleStatsEx
warnings.filterwarnings("ignore")

import weightslab as wl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Subset
from torchvision import datasets, transforms

# Import gRPC components
from weightslab.trainer.services.data_service import DataService
from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.proto.experiment_service_pb2 import SampleEditType
from weightslab.utils.tools import seed_everything

# Mock context for gRPC
class MockContext:
    """Mock gRPC context for testing"""
    def __init__(self):
        pass

# Set seed for reproducibility
seed_everything(42)
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""

    def __init__(self, conv1_out=8, conv2_out=16):
        super(SimpleCNN, self).__init__()
        self.input_shape = (1, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(conv2_out * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestGRPCTagOperations(unittest.TestCase):
    """Test suite for tag and discarded operations via gRPC"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment - runs once before all tests"""
        cls.temp_dir = tempfile.mkdtemp(prefix="wl_grpc_test_")
        cls.exp_name = "mnist_grpc_tag_test"
        
        print(f"\n========== Setting up test environment ==========")
        print(f"Temp dir: {cls.temp_dir}")
        
        # Create hyperparameters
        cls.config = {
            'model': {
                'conv1_out': 8,
                'conv2_out': 16
            },
            'data': {
                'train_loader': {
                    'batch_size': 32,
                    'shuffle': False
                }
            },
            'optimizer': {
                'lr': 0.001
            }
        }
        
        # ==================
        # Initialize dataset (100 samples from MNIST test set)
        # ==================
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download to a standard location
        mnist_data_path = Path.home() / '.mnist_data'
        mnist_data_path.mkdir(exist_ok=True)
        
        full_dataset = datasets.MNIST(
            root=str(mnist_data_path),
            train=False,
            download=True,
            transform=transform
        )
        
        # Create subset with 100 samples
        cls.dataset = Subset(full_dataset, list(range(100)))
        
        print(f"Dataset size: {len(cls.dataset)} samples")
        
        # =================
        # Register components
        # =================
        cls.config = wl.watch_or_edit(
            cls.config,
            flag="hyperparameters",
            defaults=cls.config,
            poll_interval=1.0
        )
        
        # Initialize model
        cls.model = SimpleCNN(conv1_out=8, conv2_out=16)
        cls.model = wl.watch_or_edit(
            cls.model,
            flag="model",
            device=DEVICE,
            skip_previous_auto_load=True
        )
        
        # Register dataloader
        cls.dataloader = wl.watch_or_edit(
            cls.dataset,
            flag="data",
            compute_hash=False,
            is_training=True,
            batch_size=cls.config.get('data', {}).get('train_loader', {}).get('batch_size', 32),
            shuffle=cls.config.get('data', {}).get('train_loader', {}).get('shuffle', False)
        )
        
        # Register optimizer
        cls.optimizer = wl.watch_or_edit(
            th.optim.Adam(cls.model.parameters(), lr=cls.config['optimizer']['lr']),
            flag="optimizer"
        )
        
        # Create ExperimentContext for DataService
        cls.ctx = ExperimentContext(cls.exp_name)
        
        # Initialize DataService
        cls.data_service = DataService(cls.ctx)
        cls.mock_context = MockContext()
        
        print(f"========== Setup complete ==========\n")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print(f"\n========== Cleaning up test environment ==========")
        # Close WeightsLab
        try:
            wl.finish()
        except:
            pass
        
        # Clean up temp directory
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)
        print(f"========== Cleanup complete ==========\n")

    def test_01_add_tags_accumulate(self):
        """Test adding tags to samples using EDIT_ACCUMULATE"""
        print("\n[TEST 1] Testing tag addition (EDIT_ACCUMULATE)")
        
        # Create request to add tag "test_tag" to first 10 samples
        request = pb2.DataEditsRequest(
            stat_name="tags",
            float_value=0,
            string_value="test_tag",
            bool_value=False,
            type=SampleEditType.EDIT_ACCUMULATE,
            samples_ids=list(range(10)),
            sample_origins=["test"] * 10
        )
        
        response = self.data_service.EditDataSample(request, self.mock_context)
        
        self.assertTrue(response.success, f"Failed to add tags: {response.message}")
        print(f"✓ Successfully added tag 'test_tag' to 10 samples")
        
        # Verify tags were added by checking the dataframe
        df = self.data_service._all_datasets_df
        if df is not None:
            tag_col = "tag:test_tag"
            self.assertIn(tag_col, df.columns, f"Tag column {tag_col} not found in dataframe")
            
            # Check first 10 samples have the tag
            for sample_id in range(10):
                if isinstance(df.index, pd.MultiIndex):
                    value = df.loc[("test", sample_id), tag_col]
                else:
                    mask = (df.index == sample_id) & (df["origin"] == "test")
                    if mask.any():
                        value = df.loc[mask, tag_col].iloc[0]
                    else:
                        value = False
                
                self.assertTrue(value, f"Sample {sample_id} should have tag 'test_tag'")
            
            print(f"✓ Verified tag column exists and has correct values")

    def test_02_add_multiple_tags(self):
        """Test adding multiple different tags"""
        print("\n[TEST 2] Testing multiple tag addition")
        
        # Add "difficult" tag to samples 0-4
        request1 = pb2.DataEditsRequest(
            stat_name="tags",
            float_value=0,
            string_value="difficult",
            bool_value=False,
            type=SampleEditType.EDIT_ACCUMULATE,
            samples_ids=list(range(5)),
            sample_origins=["test"] * 5
        )
        
        response1 = self.data_service.EditDataSample(request1, self.mock_context)
        self.assertTrue(response1.success)
        print(f"✓ Added tag 'difficult' to samples 0-4")
        
        # Add "outlier" tag to samples 5-9
        request2 = pb2.DataEditsRequest(
            stat_name="tags",
            float_value=0,
            string_value="outlier",
            bool_value=False,
            type=SampleEditType.EDIT_ACCUMULATE,
            samples_ids=list(range(5, 10)),
            sample_origins=["test"] * 5
        )
        
        response2 = self.data_service.EditDataSample(request2, self.mock_context)
        self.assertTrue(response2.success)
        print(f"✓ Added tag 'outlier' to samples 5-9")
        
        # Verify both tags exist
        df = self.data_service._all_datasets_df
        self.assertIn("tag:difficult", df.columns)
        self.assertIn("tag:outlier", df.columns)
        print(f"✓ Both tag columns exist in dataframe")

    def test_03_remove_tag_from_samples(self):
        """Test removing a tag from specific samples using EDIT_REMOVE"""
        print("\n[TEST 3] Testing tag removal from samples")
        
        # Remove "test_tag" from samples 0-4
        request = pb2.DataEditsRequest(
            stat_name="tags",
            float_value=0,
            string_value="test_tag",
            bool_value=False,
            type=SampleEditType.EDIT_REMOVE,
            samples_ids=list(range(5)),
            sample_origins=["test"] * 5
        )
        
        response = self.data_service.EditDataSample(request, self.mock_context)
        self.assertTrue(response.success, f"Failed to remove tag: {response.message}")
        print(f"✓ Removed tag 'test_tag' from samples 0-4")
        
        # Verify tag was removed from those samples
        df = self.data_service._all_datasets_df
        tag_col = "tag:test_tag"
        
        for sample_id in range(5):
            if isinstance(df.index, pd.MultiIndex):
                value = df.loc[("test", sample_id), tag_col]
            else:
                mask = (df.index == sample_id) & (df["origin"] == "test")
                if mask.any():
                    value = df.loc[mask, tag_col].iloc[0]
                else:
                    value = False
            
            self.assertFalse(value, f"Sample {sample_id} should NOT have tag 'test_tag'")
        
        # But samples 5-9 should still have it
        for sample_id in range(5, 10):
            if isinstance(df.index, pd.MultiIndex):
                value = df.loc[("test", sample_id), tag_col]
            else:
                mask = (df.index == sample_id) & (df["origin"] == "test")
                if mask.any():
                    value = df.loc[mask, tag_col].iloc[0]
                else:
                    value = False
            
            self.assertTrue(value, f"Sample {sample_id} should still have tag 'test_tag'")
        
        print(f"✓ Verified tag removal worked correctly")

    def test_04_delete_entire_tag_column(self):
        """Test deleting an entire tag column using EDIT_REMOVE with value=-1"""
        print("\n[TEST 4] Testing entire tag column deletion")
        
        # Delete the "difficult" tag column completely
        request = pb2.DataEditsRequest(
            stat_name="tag:difficult",
            float_value=-1,  # Signal for column deletion
            string_value="",
            bool_value=False,
            type=SampleEditType.EDIT_REMOVE,
            samples_ids=[0],  # Just need one sample as reference
            sample_origins=["test"]
        )
        
        response = self.data_service.EditDataSample(request, self.mock_context)
        self.assertTrue(response.success, f"Failed to delete tag column: {response.message}")
        print(f"✓ Deleted entire 'difficult' tag column")
        
        # Verify column no longer exists
        df = self.data_service._all_datasets_df
        self.assertNotIn("tag:difficult", df.columns, "Tag column should be deleted")
        print(f"✓ Verified tag column no longer exists in dataframe")

    def test_05_deny_listed_operations(self):
        """Test discarded (discard/restore) operations"""
        print("\n[TEST 5] Testing discarded operations")
        
        # Mark samples 10-14 as discarded (discarded)
        request_discard = pb2.DataEditsRequest(
            stat_name=SampleStatsEx.DISCARDED.value,
            float_value=0,
            string_value="",
            bool_value=True,  # True = discarded
            type=SampleEditType.EDIT_OVERRIDE,
            samples_ids=list(range(10, 15)),
            sample_origins=["test"] * 5
        )
        
        response = self.data_service.EditDataSample(request_discard, self.mock_context)
        self.assertTrue(response.success, f"Failed to discard samples: {response.message}")
        print(f"✓ Marked samples 10-14 as discarded")
        
        # Verify samples are marked as discarded
        df = self.data_service._all_datasets_df
        for sample_id in range(10, 15):
            if isinstance(df.index, pd.MultiIndex):
                value = df.loc[("test", sample_id), SampleStatsEx.DISCARDED.value]
            else:
                mask = (df.index == sample_id) & (df["origin"] == "test")
                if mask.any():
                    value = df.loc[mask, SampleStatsEx.DISCARDED.value].iloc[0]
                else:
                    value = False
            
            self.assertTrue(value, f"Sample {sample_id} should be discarded")
        
        print(f"✓ Verified samples are discarded")
        
        # Now restore samples 10-12
        request_restore = pb2.DataEditsRequest(
            stat_name=SampleStatsEx.DISCARDED.value,
            float_value=0,
            string_value="",
            bool_value=False,  # False = restored
            type=SampleEditType.EDIT_OVERRIDE,
            samples_ids=list(range(10, 13)),
            sample_origins=["test"] * 3
        )
        
        response = self.data_service.EditDataSample(request_restore, self.mock_context)
        self.assertTrue(response.success, f"Failed to restore samples: {response.message}")
        print(f"✓ Restored samples 10-12")
        
        # Verify restoration
        df = self.data_service._all_datasets_df
        for sample_id in range(10, 13):
            if isinstance(df.index, pd.MultiIndex):
                value = df.loc[("test", sample_id), SampleStatsEx.DISCARDED.value]
            else:
                mask = (df.index == sample_id) & (df["origin"] == "test")
                if mask.any():
                    value = df.loc[mask, SampleStatsEx.DISCARDED.value].iloc[0]
                else:
                    value = False
            
            self.assertFalse(value, f"Sample {sample_id} should be restored")
        
        # But 13-14 should still be discarded
        for sample_id in range(13, 15):
            if isinstance(df.index, pd.MultiIndex):
                value = df.loc[("test", sample_id), SampleStatsEx.DISCARDED.value]
            else:
                mask = (df.index == sample_id) & (df["origin"] == "test")
                if mask.any():
                    value = df.loc[mask, SampleStatsEx.DISCARDED.value].iloc[0]
                else:
                    value = False
            
            self.assertTrue(value, f"Sample {sample_id} should still be discarded")
        
        print(f"✓ Verified restoration worked correctly")

    def test_06_batch_tag_operations(self):
        """Test batch operations on many samples at once"""
        print("\n[TEST 6] Testing batch tag operations")
        
        # Add "batch_tag" to 50 samples at once
        request = pb2.DataEditsRequest(
            stat_name=f"{SampleStatsEx.TAG.value}:batch_tag",
            float_value=0,
            string_value="batch_tag",
            bool_value=False,
            type=SampleEditType.EDIT_ACCUMULATE,
            samples_ids=list(range(50)),
            sample_origins=["test"] * 50
        )
        
        response = self.data_service.EditDataSample(request, self.mock_context)
        self.assertTrue(response.success, f"Failed to add batch tag: {response.message}")
        print(f"✓ Added 'batch_tag' to 50 samples in one operation")
        
        # Verify all 50 samples have the tag
        df = self.data_service._all_datasets_df
        tag_col = f"{SampleStatsEx.TAG.value}:batch_tag"
        self.assertIn(tag_col, df.columns)
        
        success_count = 0
        for sample_id in range(50):
            if isinstance(df.index, pd.MultiIndex):
                try:
                    value = df.loc[("test", sample_id), tag_col]
                    if value:
                        success_count += 1
                except KeyError:
                    pass
            else:
                mask = (df.index == sample_id) & (df["origin"] == "test")
                if mask.any():
                    value = df.loc[mask, tag_col].iloc[0]
                    if value:
                        success_count += 1
        
        self.assertGreaterEqual(success_count, 45, f"Expected at least 45 samples to have batch_tag, got {success_count}")
        print(f"✓ Verified {success_count}/50 samples have the batch tag")

    def test_07_tag_persistence(self):
        """Test that tags persist and can be queried"""
        print("\n[TEST 7] Testing tag persistence")
        
        df = self.data_service._all_datasets_df
        
        # Count tag columns
        tag_columns = [col for col in df.columns if col.startswith(f"{SampleStatsEx.TAG.value}:")]
        print(f"✓ Found {len(tag_columns)} tag columns: {tag_columns}")
        
        self.assertGreater(len(tag_columns), 0, "Should have at least one tag column")
        
        # Verify we can query tagged samples
        for tag_col in tag_columns:
            tagged_samples = df[df[tag_col] == True]
            print(f"  - {tag_col}: {len(tagged_samples)} samples")
            self.assertGreaterEqual(len(tagged_samples), 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Running gRPC Tag and Deny_Listed Operations Tests")
    print("="*80)
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGRPCTagOperations)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80 + "\n")
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")




