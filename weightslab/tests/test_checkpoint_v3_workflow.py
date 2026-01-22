"""
Comprehensive Unit Tests for Checkpoint System V3

Tests the complete workflow of experiment checkpointing with:
- Model architecture changes
- Hyperparameter updates
- Data state changes (tags, discard)
- Checkpoint reloading and branching

Tests are separated into individual methods (init, testA, B, C, D, E)
with state preserved in class variables between tests.
"""

import os
import sys
import time
import shutil
import random
import unittest
import tempfile
import warnings
import pandas as pd
from sympy import true
import yaml
import dill
warnings.filterwarnings("ignore")

import weightslab as wl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pyexpat import model
from tqdm import trange

# Import components directly to avoid full weightslab initialization
from weightslab.components.checkpoint_manager_v2 import CheckpointManagerV2
from weightslab.backend import ledgers
from weightslab.components.global_monitoring import (
    guard_training_context,
    pause_controller
)
from weightslab.utils.tools import seed_everything


# Helper function to register objects in ledger directly
def register_in_ledger(obj, flag, name, device='cpu', **kwargs):
    """Register an object in the ledger."""
    try:
        if flag == "hyperparameters":
            return wl.watch_or_edit(
                obj,
                flag="hyperparameters",
                name=name,
                defaults=obj,
                poll_interval=1.0,
                **kwargs
            )
        elif flag == "model":
            return wl.watch_or_edit(
                obj,
                flag="model",
                name=name,
                device=device,
                **kwargs
            )
        elif flag == "dataloader":
           return wl.watch_or_edit(
                obj,
                flag="data",
                name="train_loader",
                **kwargs
            )
        elif flag == "optimizer":
            return wl.watch_or_edit(
                obj,
                flag="optimizer",
                name=name,
                **kwargs
            )
        elif flag == "signal":
            return wl.watch_or_edit(
                obj,
                flag="signal",
                name=name,
                **kwargs
            )
    except Exception as e:
        # If direct registration fails, silently continue
        pass


# Set seed for reproducibility
seed_everything()
DEVICE = "cuda" if th.cuda.is_available() else "cpu"


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""

    def __init__(self, conv1_out=8, conv2_out=16):
        super(SimpleCNN, self).__init__()
        self.input_shape = (1, 28, 28)  # MNIST input shape
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


class TaggableDataset:
    """Wrapper for dataset with tagging and discard functionality"""

    def __init__(self, dataset):
        self.dataset = dataset
        self._tags = {}
        self._discarded = set()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self._discarded:
            # Return next non-discarded sample
            for i in range(idx + 1, len(self.dataset)):
                if i not in self._discarded:
                    return self.dataset[i]
            # Wrap around
            for i in range(0, idx):
                if i not in self._discarded:
                    return self.dataset[i]
        return (
            self.dataset[idx][0],  # Data
            # th.Tensor([idx]).to(int),   # UID
            self.dataset[idx][1]   # Label
        )

    def get_sample_uids(self):
        """Return list of sample UIDs"""
        return [i for i in range(len(self.dataset))]

    def is_discarded(self, uid):
        """Check if sample is discarded"""
        idx = int(uid.split('_')[1])
        return idx in self._discarded

    def discard(self, uid):
        """Discard a sample"""
        idx = int(uid.split('_')[1])
        self._discarded.add(idx)

    def add_tag(self, uid, tag):
        """Add tag to sample"""
        if uid not in self._tags:
            self._tags[uid] = []
        if tag not in self._tags[uid]:
            self._tags[uid].append(tag)

    def get_tags(self, uid):
        """Get tags for sample"""
        return self._tags.get(uid, [])

    def get_data_state(self):
        """Get complete data state for checkpointing"""
        uids = self.get_sample_uids()
        return {
            'uids': uids,
            'discarded': self._discarded.copy(),
            'tags': self._tags.copy()
        }


class CheckpointSystemV3Tests(unittest.TestCase):
    """Comprehensive tests for checkpoint system V3 with separated test methods"""

    # Class variables to preserve state across tests
    temp_dir = None
    log_dir = None
    dataset = None
    config = None
    manager = None

    # State tracking for each experiment
    state = {
        'exp_hash_a': None,
        'exp_hash_b': None,
        'exp_hash_c': None,
        'exp_hash_d': None,
        'exp_hash_e': None,
    }

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        print("\n" + "="*80)
        print("CHECKPOINT SYSTEM V3 - COMPREHENSIVE TESTS (SEPARATED)")
        print("="*80 + "\n")

        # Init pause controller
        pause_controller.pause()

        # Create temporary directory (used for all tests)
        # cls.temp_dir = tempfile.mkdtemp(prefix="checkpoint_v3_test_")
        cls.temp_dir = r'C:/Users/GuillaumePelluet/Desktop/trash/cls_usecase/'
        cls.log_dir = os.path.join(cls.temp_dir, "experiments")

        # Load MNIST subset (100 samples for all tests)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_dataset = datasets.MNIST(
            root=os.path.join(cls.temp_dir, 'data'),
            train=False,
            download=True,
            transform=transform
        )

        # Create subset with 100 samples
        subset_indices = list(range(100))
        mnist_subset = Subset(full_dataset, subset_indices)

        # Wrap in taggable dataset
        cls.dataset = TaggableDataset(mnist_subset)

        # Initialize config from YAML-like dict (similar to ws-classification)
        cls.config = {
            'experiment_name': 'mnist_checkpoint_test',
            'device': DEVICE,
            'root_log_dir': cls.log_dir,

            # Data parameters
            'data': {
                'train_loader': {
                    'batch_size': 2,
                    'shuffle': True
                },
            },

            'experiment_dump_to_train_steps_ratio': 5,
            'skip_checkpoint_load': False,

            # Configure global dataframe storage
            'ledger_enable_flushing_threads': True,
            'ledger_enable_h5_persistence': True,
            'ledger_flush_max_rows': 750,
            'ledger_flush_interval': 30,

            # Configure clients
            'serving_grpc': False,
            'serving_cli': False,

            # Training parameters
            'training': {
                'num_epochs': 11,
            },

            # Optimizer parameters
            'optimizer': {
                'lr': 0.001
            }
        }

        # Initialize checkpoint manager for everyone
        cls.manager = CheckpointManagerV2(root_log_dir=cls.log_dir)

        # Register in ledger so model_interface can use it
        cls.manager = ledgers.register_checkpoint_manager(cls.config['experiment_name'], cls.manager)

        # Print setup info
        print(f"[OK] Created MNIST subset: {len(cls.dataset)} samples")
        print(f"[OK] Temporary directory: {cls.temp_dir}")
        print(f"[OK] Config initialized")
        print(f"[OK] Checkpoint manager initialized at {cls.log_dir}\n")

    def train_epochs(self, model, loader, optimizer, criterion, num_epochs):
        """Train model for specified epochs with checkpointing"""
        losses = []

        for _ in trange(num_epochs, desc="Training"):
            with guard_training_context:
                epoch_loss = 0.0
                batch_count = 0

                (inputs, ids, labels) = next(loader)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                preds_raw = model(inputs)

                # Preds
                if preds_raw.ndim == 1:
                    preds = (preds_raw > 0.0).long()
                else:
                    preds = preds_raw.argmax(dim=1, keepdim=True)

                # Loss and backward
                loss = criterion(
                    preds_raw,
                    labels,
                    batch_ids=ids,
                    preds=preds
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                losses.append(avg_loss)

        return losses

    # ========================================================================
    # Test: 00_initialize_experiment
    # ========================================================================
    def test_00_initialize_experiment(self):
        """Initialize experiment with configuration and first model"""
        print(f"\n{'='*80}")
        print("TEST 00: Initialize Experiment Configuration")
        print(f"{'='*80}\n")

        # Initialize hyperparameters with model_age
        hp = self.config
        exp_name = hp['experiment_name']

        # Register HP in ledger
        register_in_ledger(hp, flag="hyperparameters", name=exp_name)

        # Create and register model
        model = SimpleCNN(conv1_out=8, conv2_out=16)
        model = register_in_ledger(model, flag="model", name=exp_name, device=DEVICE)

        # Create and register dataloader
        dataloader = register_in_ledger(
            self.dataset,
            flag="dataloader",
            name=exp_name,
            compute_hash=False,
            is_training=True,
            batch_size=hp.get('data', {}).get('train_loader', {}).get('batch_size', 32),
            shuffle=hp.get('data', {}).get('train_loader', {}).get('shuffle', True)
        )

        # Optimizer and criterion
        # # Create and register optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=hp.get('optimizer', {}).get('lr', 0.001))
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        # # Create and register signal (criterion)
        criterion = nn.CrossEntropyLoss()
        criterion = register_in_ledger(criterion, flag="signal", name=exp_name)

        print("Configuration:")
        print(f"  Experiment: {hp['experiment_name']}")
        print(f"  Device: {hp.get('device', 'cpu')}")
        print(f"  Model: SimpleCNN(conv1_out=8, conv2_out=16)")
        print(f"  LR: {hp.get('optimizer', {}).get('lr', 0.001)}")
        print(f"  Batch size: {hp.get('data', {}).get('train_loader', {}).get('batch_size', 32)}")
        print(f"  Checkpoint freq: {hp.get('experiment_dump_to_train_steps_ratio', 5)}")

        # Update experiment hash
        exp_hash_a, is_new, changed = self.manager.update_experiment_hash()

        print(f"\n[OK] Experiment hash A: {exp_hash_a}")
        print(f"[OK] Changed components: {changed}")

        # Verify directory structure
        self.assertTrue(os.path.exists(self.manager.models_dir))
        self.assertTrue(os.path.exists(self.manager.hp_dir))
        self.assertTrue(os.path.exists(self.manager.data_checkpoint_dir))
        self.assertTrue(os.path.exists(self.manager.manifest_file))

        # Store in state for next tests
        self.state['exp_hash_a'] = exp_hash_a

        print(f"\n[OK] TEST 00 PASSED - Experiment initialized")

    # ========================================================================
    # Test: 01_train_A
    # ========================================================================
    def test_01_train_A(self):
        """Train initial model for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST A: Initialize and First Training")
        print(f"{'='*80}\n")

        # Get stored state from previous test
        exp_hash_a = self.state['exp_hash_a']
        exp_name = self.config['experiment_name']

        # Model
        model = ledgers.get_model(exp_name)

        # Dataloader
        dataloader = ledgers.get_dataloader('train_loader')

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer(exp_name)
        criterion = ledgers.get_signal(exp_name)

        # Whole configuration at initialization
        # # Model add neurons
        model.operate(0, {-1, -2}, 1)  # conv1 out channels: 8 -> 10
        # # Data discard samples
        dfm = ledgers.get_dataframe('sample_stats')  # Get dataframe manager
        rows = []
        for idx in random.sample(range(100), 20):
            uid = dfm._df.index[idx]
            rows.append(
                {
                    "sample_id": uid,
                    "tags": f"ugly_{random.randint(0, 100)}",
                    "deny_listed": bool(1 - dfm._df['deny_listed'].iloc[idx])
                }
            )
        # # # Updates data - Simulate adding tags and discarding samples in dataset
        # # # upsert_df updates the ledger's dataframe immediately
        dfm.upsert_df(pd.DataFrame(rows).set_index("sample_id"), origin='train_loader', force_flush=True)
        # # Hyperparameters change
        self.config['optimizer']['lr'] = 0.0005  # Change learning rate
        self.config['data']['train_loader']['batch_size'] = 8  # Change batch size
        register_in_ledger(self.config, flag="hyperparameters", name=exp_name)
        print("Initial modifications before training:")
        print(f"  Model: conv1 out channels 8 -> 10")
        print(f"  Data: Discarded 20 random samples and added tags")
        print(f"  Hyperparameters: LR 0.001 -> 0.0005, Batch size 2 -> 8")

        # Training
        print("Training for 11 epochs with checkpoint frequency 5...")
        pause_controller.resume()
        self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
        )
        pause_controller.pause()
        exp_hash_a_1, is_new, changed = self.manager.update_experiment_hash()
        print("\nTraining completed.")

        # Verify checkpoints
        model_dir_a = self.manager.models_dir / exp_hash_a_1
        self.assertTrue(model_dir_a.exists(), "Model checkpoint directory should exist")

        # Check for weight checkpoints
        weight_files = list(model_dir_a.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files), 2, "Should have at least 2 weight checkpoints")

        # Check HP directory
        hp_dir_a = self.manager.hp_dir / exp_hash_a_1
        self.assertTrue(hp_dir_a.exists(), "HP checkpoint directory should exist")

        # Check data directory
        data_dir_a = self.manager.data_checkpoint_dir / exp_hash_a_1
        self.assertTrue(data_dir_a.exists(), "Data checkpoint directory should exist")

        self.state['exp_hash_a'] = exp_hash_a_1
        print(f"\n[OK] TEST A PASSED - Initial training completed")
        print(f"  Final model_age: {model.current_step}")

    # ========================================================================
    # Test: 02_train_B_model_change
    # ========================================================================
    def test_02_train_B_model_change(self):
        """Modify model architecture and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST B: Modify Model Architecture")
        print(f"{'='*80}\n")

        # Get stored state from previous test
        exp_name = self.config['experiment_name']
        data_state = self.dataset.get_data_state()

        # Model
        model = ledgers.get_model(exp_name)

        # Dataloader
        dataloader = ledgers.get_dataloader('train_loader')

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer(exp_name)
        criterion = ledgers.get_signal(exp_name)

        print("Modifying model architecture...")

        # Modify model architecture
        model.operate(0, {-1, -2, -3}, 1)  # Increase conv1 out channels by 2
        model.operate(2, 1, 2)  # Decrease conv2 out channels by 1
        model.operate(-2, {}, 3)  # Freeze fc1 layer
        model.operate(-1, {1}, 4)  # Reset fc2 layer

        print(f"  Conv1: 8 -> 10 channels")
        print(f"  Conv2: 16 -> 15 channels")
        print(f"  FC1: Frozen")
        print(f"  FC2: Reset")

        # Update hash
        exp_hash_b, is_new, changed = self.manager.update_experiment_hash(
            model_snapshot=model,
            hp_snapshot=self.config,
            dfm_snapshot=data_state,
            dump_immediately=False
        )

        print(f"\n[OK] New experiment hash B: {exp_hash_b}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('model', changed, "Model should have changed")
        self.assertNotEqual(self.state['exp_hash_a'], exp_hash_b, "Hash should be different")

        print("\nResuming training for 11 epochs...")
        model.operate(0, {-1, -1}, 1)  # Model change from UI during pause
        pause_controller.resume()
        self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs']
        )
        pause_controller.pause()
        exp_hash_b_1, is_new, changed = self.manager.update_experiment_hash(
            model_snapshot=model,
            hp_snapshot=self.config,
            dfm_snapshot=data_state,
            dump_immediately=False
        )
        self.assertNotEqual(exp_hash_b_1, exp_hash_b, "Hash should be different")

        print("\nTraining completed.")

        # Verify new model directory
        model_dir_b = self.manager.models_dir / exp_hash_b_1
        self.assertTrue(model_dir_b.exists(), "New model checkpoint directory should exist")
        weight_files_b = list(model_dir_b.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_b)} weight checkpoint files in new directory")
        self.assertGreaterEqual(len(weight_files_b), 2, "Should have at least 2 new weight checkpoints")

        # Store state
        self.state['exp_hash_b'] = exp_hash_b_1

        print(f"\n[OK] TEST B PASSED - Model architecture updated")
        print(f"  Final model_age: {model.current_step}")

    # ========================================================================
    # Test: 03_train_C_hyperparams_change
    # ========================================================================
    def test_03_train_C_hyperparams_change(self):
        """Change hyperparameters and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST C: Change Hyperparameters")
        print(f"{'='*80}\n")

        # Get stored state from previous test
        exp_name = self.config['experiment_name']

        # Model
        model = ledgers.get_model(exp_name)

        # Dataloader
        dataloader = ledgers.get_dataloader('train_loader')

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer(exp_name)
        criterion = ledgers.get_signal(exp_name)

        print("Changing hyperparameters...")

        # Change batch size
        self.config['data']['train_loader']['batch_size'] = 4
        print(f"  Batch size: 2 -> 4")

        # Update in ledger
        register_in_ledger(self.config, flag="hyperparameters", name=exp_name)

        # Update hash
        exp_hash_c, is_new, changed = self.manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash C: {exp_hash_c}")
        self.assertNotEqual(self.state['exp_hash_b'], exp_hash_c, "Hash should be different sa hp changedcd")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume()
        self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs']
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify new HP directory
        hp_dir_c = self.manager.hp_dir / exp_hash_c
        self.assertTrue(hp_dir_c.exists(), "New HP checkpoint directory should exist")

        # Verify model weights still being saved
        model_dir_c = self.manager.models_dir / exp_hash_c
        weight_files_c = list(model_dir_c.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_c)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_c), 2, "Should have at least 2 weight checkpoints")

        # Store state
        self.state['exp_hash_c'] = exp_hash_c

        print(f"\n[OK] TEST C PASSED - Hyperparameters updated")
        print(f"  Final model_age: {model.current_step}")

    # ========================================================================
    # Test: 04_train_D_data_change
    # ========================================================================
    def test_04_train_D_data_change(self):
        """Change data state (tags and discard) and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST D: Change Data State (Tags and Discard)")
        print(f"{'='*80}\n")

        # Get stored state from previous test
        exp_name = self.config['experiment_name']

        # Model
        model = ledgers.get_model(exp_name)

        # Data
        dataloader = ledgers.get_dataloader('train_loader')  # Get dataloader
        dfm = ledgers.get_dataframe('sample_stats')  # Get dataframe manager

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer(exp_name)
        criterion = ledgers.get_signal(exp_name)

        print("Modifying data...")

        # Add 20 random tags with 'ugly'
        tagged_samples = random.sample(range(100), 20)
        rows = []
        for idx in tagged_samples:
            uid = dfm._df.index[idx]
            rows.append(
                {
                    "sample_id": uid,
                    "tags": f"ugly_{random.randint(0, 100)}",
                    "deny_listed": bool(1 - dfm._df['deny_listed'].iloc[idx])
                }
            )

        # Updates data - Simulate adding tags and discarding samples in dataset
        df_update = pd.DataFrame(rows).set_index("sample_id")
        # upsert_df updates the ledger's dataframe immediately
        dfm.upsert_df(df_update, origin='train_loader', force_flush=True)

        # Changes will be pending
        print(f"  Added 'ugly' tag to 20 samples")
        print(f"  Discarded 20 samples")

        # Update hash
        exp_hash_d, is_new, changed = self.manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash D: {exp_hash_d}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('data', changed, "Data should have changed")
        self.assertNotEqual(self.state['exp_hash_c'], exp_hash_d, "Hash should be different")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume()  # Pending changes to dump: data state
        self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs']
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify new data directory
        data_dir_d = self.manager.data_checkpoint_dir / exp_hash_d
        self.assertTrue(data_dir_d.exists(), "New data checkpoint directory should exist")

        # Verify model weights still being saved
        model_dir_d = self.manager.models_dir / exp_hash_d
        weight_files_d = list(model_dir_d.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_d)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_d), 2, "Should have at least 2 weight checkpoints")

        # Store state
        self.state['exp_hash_d'] = exp_hash_d

        print(f"\n[OK] TEST D PASSED - Data state updated")
        print(f"  Final model_age: {model.current_step}")

    # ========================================================================
    # Test: 05_train_E_reload_and_branch
    # ========================================================================
    def test_05_train_E_reload_and_branch(self):
        """Reload state B and branch with modified HP and data"""
        print(f"\n{'='*80}")
        print("TEST E: Reload State B and Branch")
        print(f"{'='*80}\n")

        # Get hp from original training
        hp_original = self.config
        exp_name = hp_original['experiment_name']

        print("Experiment paused. Analyzing experiment history...")

        # Get all hashes
        all_hashes = self.manager.get_all_hashes(sort_by='created')
        print(f"\n[OK] Found {len(all_hashes)} experiment states:")
        for i, entry in enumerate(all_hashes):
            print(f"  {i+1}. {entry['hash'][:16]}... (created: {entry['created'][:19]})")

        # Reload state B (second state created)
        hash_a_from_manifest = self.state['exp_hash_a']

        print(f"\n[OK] Reloading state B: {hash_a_from_manifest[:16]}...")

        # Use new load_state method to load and apply checkpoint in-place
        success = self.manager.load_state(exp_hash=hash_a_from_manifest)
        self.assertTrue(success, "State should be loaded successfully")

        # Get components from ledger (they were updated in-place)
        model_reloaded = ledgers.get_model(exp_name)
        hp_reloaded = ledgers.get_hyperparams(exp_name)

        print(f"[OK] State applied successfully")
        print(f"[OK] Loaded HP: {hp_reloaded}")

        # Modify HP and data
        print("\nModifying HP and data (not training yet)...")

        # Handle nested config structure
        if 'data' in hp_reloaded and 'train_loader' in hp_reloaded['data']:
            hp_reloaded['data']['train_loader']['batch_size'] = 1
            old_batch_size = hp_original.get('data', {}).get('train_loader', {}).get('batch_size', 2)
            print(f"  Batch size: {old_batch_size} -> 1")

        if 'optimizer' in hp_reloaded:
            hp_reloaded['optimizer']['lr'] = hp_reloaded['optimizer'].get('lr', 0.001) / 10
            old_lr = hp_original.get('optimizer', {}).get('lr', 0.001)
            print(f"  LR: {old_lr} -> {hp_reloaded['optimizer']['lr']}")

        # Discard more data
        # Add 20 random tags with 'ugly'
        tagged_samples = random.sample(range(100), 20)
        rows = []
        dfm = ledgers.get_dataframe()  # Get dataframe manager
        for idx in tagged_samples:
            uid = dfm._df.index[idx]
            rows.append(
                {
                    "sample_id": uid,
                    "tags": f"hugly_{random.randint(0, 100)}",
                    "deny_listed": bool(1 - dfm._df['deny_listed'].iloc[idx])
                }
            )
        # # # Updates data - Simulate adding tags and discarding samples in dataset
        # # # upsert_df updates the ledger's dataframe immediately
        dfm.upsert_df(pd.DataFrame(rows).set_index("sample_id"), origin='train_loader', force_flush=True)

        # Update hash with all changes
        exp_hash_e, is_new, changed = self.manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash E (branch): {exp_hash_e}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('hp', changed, "HP should have changed")
        self.assertIn('data', changed, "Data should have changed")

        # Update ledger
        # Ledger is already registered as proxy are used
        pass

        # Setting training environment from loader
        dataloader = ledgers.get_dataloader('train_loader')
        model = ledgers.get_model(exp_name)
        optimizer = ledgers.get_optimizer(exp_name)
        criterion = ledgers.get_signal(exp_name)

        print("\nResuming training for 21 epochs...")
        pause_controller.resume()
        self.train_epochs(
            model_reloaded, dataloader, optimizer, criterion,
            num_epochs=21
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify checkpoints for E
        model_dir_e = self.manager.models_dir / exp_hash_e
        weight_files_e = list(model_dir_e.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_e)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_e), 4, "Should have at least 4 weight checkpoints for 21 epochs")

        # Store state
        self.state['exp_hash_e'] = exp_hash_e

        print(f"\n[OK] TEST E PASSED - Reloaded and branched successfully")
        print(f"  Final model_age: {model.current_step}")

    # ========================================================================
    # Test: 06_final_verification
    # ========================================================================
    def test_06_final_verification(self):
        """Verify all experiment states and directory structure"""
        print(f"\n{'='*80}")
        print("TEST 06: Final Verification")
        print(f"{'='*80}\n")

        # Verify manifest
        final_hashes = self.manager.get_all_hashes(sort_by='created')
        print(f"[OK] Total experiment states: {len(final_hashes)}")
        self.assertEqual(len(final_hashes), 5, "Should have 5 total experiment states (A, B, C, D, E)")

        # Verify directory structure
        print("\n[OK] Directory structure:")
        for exp_hash in [self.state['exp_hash_a'], self.state['exp_hash_b'],
                         self.state['exp_hash_c'], self.state['exp_hash_d'],
                         self.state['exp_hash_e']]:
            short_hash = exp_hash[:8]
            print(f"\n  Hash {short_hash}...")

            model_dir = self.manager.models_dir / exp_hash
            hp_dir = self.manager.hp_dir / exp_hash
            data_dir = self.manager.data_checkpoint_dir / exp_hash

            print(f"    Models: {model_dir.exists()} ({len(list(model_dir.glob('*.pt')))} checkpoints)")
            print(f"    HP: {hp_dir.exists()}")
            print(f"    Data: {data_dir.exists()}")

        print("\n" + "="*80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80 + "\n")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
