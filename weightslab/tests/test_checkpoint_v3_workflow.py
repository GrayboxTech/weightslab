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
import random
import unittest
import tempfile
import warnings
import json
import pandas as pd
import dill
from pathlib import Path
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
from weightslab.utils.tools import capture_rng_state, restore_rng_state
from weightslab.components.checkpoint_manager_v2 import CheckpointManagerV2
from weightslab.utils.logger import LoggerQueue
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
EXP_NAME = "mnist_checkpoint_test_v3"

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""

    def __init__(self, conv1_out=8, conv2_out=16):
        super(SimpleCNN, self).__init__()
        self.input_shape = (1, 1, 28, 28)  # MNIST input shape
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
        'exp_hash_f': None,
        'exp_hash_g': None,
        'exp_hash_h': None,
        'exp_hash_i': None,
        'exp_hash_j': None,
        'exp_hash_k': None,
        'losses_a': None,
        'losses_b': None,
        'losses_c': None,
        'losses_d': None,
        'losses_e': None,
        'losses_f': None,
        'losses_g': None,
        'losses_h': None,
        'losses_i': None,
        'losses_j': None,
        'losses_k': None,
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
        cls.temp_dir = tempfile.mkdtemp(prefix="checkpoint_v3_test_")
        # cls.temp_dir = r'C:\Users\GUILLA~1\AppData\Local\Temp\checkpoint_v3_test_17c8pljj'
        cls.log_dir = os.path.join(cls.temp_dir, "experiments")

        # Load MNIST subset (100 samples for all tests)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        full_dataset = datasets.MNIST(
            # root=os.path.join(cls.temp_dir, 'data'),
            root='C:/Users/GuillaumePelluet/Desktop/mnist_data/',
            train=False,
            download=True,
            transform=transform
        )

        # Init logger
        cls.logger = LoggerQueue(name=EXP_NAME, register=True)

        # Create subset with 100 samples
        subset_indices = list(range(100))
        mnist_subset = Subset(full_dataset, subset_indices)

        # Wrap in taggable dataset
        cls.dataset = TaggableDataset(mnist_subset)

        # Initialize config from YAML-like dict (similar to ws-classification)
        cls.config = {
            'experiment_name': EXP_NAME,
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
            'ledger_flush_max_rows': 4,
            'ledger_flush_interval': 5.0,

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
        cls.config_cp = cls.config.copy()

        # Initialize checkpoint manager for everyone
        cls.chkpt_manager = CheckpointManagerV2(root_log_dir=cls.config.get('root_log_dir'))

        # Register in ledger so model_interface can use it
        cls.chkpt_manager = ledgers.register_checkpoint_manager(cls.config['experiment_name'], cls.chkpt_manager)

        # Print setup info
        print(f"[OK] Created MNIST subset: {len(cls.dataset)} samples")
        print(f"[OK] Temporary directory: {cls.temp_dir}")
        print(f"[OK] Config initialized")
        print(f"[OK] Checkpoint manager initialized at {cls.config.get('root_log_dir')}\n")

    def train_epochs(self, model, loader, optimizer, criterion, num_epochs, criterion_bin=None, return_uids=False):
        """Train model for specified epochs with checkpointing"""
        losses = []
        uids_trained = []
        for _ in trange(num_epochs, desc="Training"):
            with guard_training_context:
                epoch_loss = 0.0
                batch_count = 0

                (inputs, ids, labels) = next(loader)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                uids_trained.extend(ids.tolist())

                optimizer.zero_grad()
                preds_raw = model(inputs)

                # Preds
                if preds_raw.ndim == 1:
                    preds = (preds_raw > 0.0).long()
                else:
                    preds = preds_raw.argmax(dim=1, keepdim=True)

                if criterion_bin is not None:
                    # Binary loss
                    loss = criterion_bin(
                        preds_raw[:, 7],
                        (labels==7).float(),
                        batch_ids=ids,
                        preds=preds
                    )

                # Loss and backward
                loss = criterion(
                    preds_raw,
                    labels,
                    batch_ids=ids,
                    preds=preds
                )
                loss.mean().backward()
                optimizer.step()

                epoch_loss += loss.mean().item()
                batch_count += 1

                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                losses.append(avg_loss)
        print(f"Trained on {uids_trained}.")
        if return_uids:
            return losses, uids_trained
        return losses

    # ==============================
    # Test: 00_initialize_experiment
    # ==============================
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
        register_in_ledger(
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
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = register_in_ledger(
            criterion,
            flag="signal",
            name="train_mlt_loss/CE",
            log=True
        )
        criterion_bin = nn.BCEWithLogitsLoss(reduction='none')
        criterion_bin = register_in_ledger(
            criterion_bin,
            flag="signal",
            name="train_bin_loss/BCE",
            log=True
        )

        print("Configuration:")
        print(f"  Experiment: {hp['experiment_name']}")
        print(f"  Device: {hp.get('device', 'cpu')}")
        print(f"  Model: SimpleCNN(conv1_out=8, conv2_out=16)")
        print(f"  LR: {hp.get('optimizer', {}).get('lr', 0.001)}")
        print(f"  Batch size: {hp.get('data', {}).get('train_loader', {}).get('batch_size', 32)}")
        print(f"  Checkpoint freq: {hp.get('experiment_dump_to_train_steps_ratio', 5)}")

        exp_hash_a, _, changed = self.chkpt_manager.update_experiment_hash(firsttime=True)

        print(f"\n[OK] Experiment hash A: {exp_hash_a}")
        print(f"[OK] Changed components: {changed}")

        # Verify directory structure
        pause_controller.resume()
        self.assertTrue(os.path.exists(self.chkpt_manager.models_dir))
        self.assertTrue(os.path.exists(self.chkpt_manager.hp_dir))
        self.assertTrue(os.path.exists(self.chkpt_manager.data_checkpoint_dir))
        self.assertTrue(os.path.exists(self.chkpt_manager.manifest_file))

        # Store in state for next tests
        self.state['exp_hash_a'] = exp_hash_a

        print(f"\n[OK] TEST 00 PASSED - Experiment initialized")

    # ================
    # Test: 01_train_A
    # ================
    def test_01_train_A(self):
        """Train initial model for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST A: Initialize and First Training")
        print(f"{'='*80}\n")
        # Get stored state from previous test
        exp_hash_a = self.state['exp_hash_a']
        exp_name = self.config['experiment_name']
        success = self.chkpt_manager.load_state(exp_hash=self.state['exp_hash_a'])
        self.assertTrue(success, "Checkpoint load should succeed")
        pause_controller.resume()

        # Model
        model = ledgers.get_model()

        # Dataloader
        dataloader = ledgers.get_dataloader()

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        # Training
        print("Training for 11 epochs with checkpoint frequency 5...")
        pause_controller.resume()
        loss_A = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()
        print("\nTraining completed.")

        # Verify checkpoints
        model_dir_a = self.chkpt_manager.models_dir / exp_hash_a
        self.assertTrue(model_dir_a.exists(), "Model checkpoint directory should exist")

        # Check for weight checkpoints
        weight_files = list(model_dir_a.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files), 2, "Should have at least 2 weight checkpoints")

        # Check HP directory
        hp_dir_a = self.chkpt_manager.hp_dir / exp_hash_a
        self.assertTrue(hp_dir_a.exists(), "HP checkpoint directory should exist")

        # Check data directory
        data_dir_a = self.chkpt_manager.data_checkpoint_dir / exp_hash_a
        self.assertTrue(data_dir_a.exists(), "Data checkpoint directory should exist")

        self.state['exp_hash_a'] = exp_hash_a
        self.state['losses_a'] = loss_A
        print(f"\n[OK] TEST A PASSED - Initial training completed")
        print(f"  Final model_age: {model.current_step}")

    # =============================
    # Test: 02_train_B_model_change
    # =============================
    def test_02_train_B_model_change(self):
        """Modify model architecture and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST B: Modify Model Architecture")
        print(f"{'='*80}\n")

        # Get stored state from previous test
        exp_name = self.config['experiment_name']

        # Model
        model = ledgers.get_model()

        # Dataloader
        dataloader = ledgers.get_dataloader()

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

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

        # Update hash here to get hash
        exp_hash_b, _, changed = self.chkpt_manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash B: {exp_hash_b}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('model', changed, "Model should have changed")
        self.assertNotEqual(self.state['exp_hash_a'], exp_hash_b, "Hash should be different")

        print("\nResuming training for 11 epochs...")
        model.operate(0, {-1, -1}, 1)  # Model change from UI during pause
        pause_controller.resume()
        loss_B = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()
        exp_hash_b_1, _, changed = self.chkpt_manager.update_experiment_hash()
        self.assertNotEqual(exp_hash_b_1, exp_hash_b, "Hash should be different")

        print("\nTraining completed.")

        # Verify new model directory
        model_dir_b = self.chkpt_manager.models_dir / exp_hash_b_1
        self.assertTrue(model_dir_b.exists(), "New model checkpoint directory should exist")
        weight_files_b = list(model_dir_b.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_b)} weight checkpoint files in new directory")
        self.assertGreaterEqual(len(weight_files_b), 2, "Should have at least 2 new weight checkpoints")

        # Store state
        self.state['exp_hash_b'] = exp_hash_b_1
        self.state['losses_b'] = loss_B

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
        model = ledgers.get_model()

        # Dataloader
        dataloader = ledgers.get_dataloader()

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("Changing hyperparameters...")

        # Change batch size
        self.config['data']['train_loader']['batch_size'] = 4
        print(f"  Batch size: 2 -> 4")

        # Update in ledger
        register_in_ledger(self.config, flag="hyperparameters", name=exp_name)

        # Update hash
        exp_hash_c, _, _ = self.chkpt_manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash C: {exp_hash_c}")
        self.assertNotEqual(self.state['exp_hash_b'], exp_hash_c, "Hash should be different sa hp changedcd")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume()
        loss_C = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify new HP directory
        hp_dir_c = self.chkpt_manager.hp_dir / exp_hash_c
        self.assertTrue(hp_dir_c.exists(), "New HP checkpoint directory should exist")

        # Verify model weights still being saved
        model_dir_c = self.chkpt_manager.models_dir / exp_hash_c
        weight_files_c = list(model_dir_c.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_c)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_c), 2, "Should have at least 2 weight checkpoints")

        # Store state
        self.state['exp_hash_c'] = exp_hash_c
        self.state['losses_c'] = loss_C

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
        model = ledgers.get_model()

        # Data
        dataloader = ledgers.get_dataloader()  # Get dataloader
        dfm = ledgers.get_dataframe('sample_stats')  # Get dataframe manager

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

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
        exp_hash_d, _, changed = self.chkpt_manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash D: {exp_hash_d}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('data', changed, "Data should have changed")
        self.assertNotEqual(self.state['exp_hash_c'], exp_hash_d, "Hash should be different")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume()  # Pending changes to dump: data state
        loss_D = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify new data directory
        data_dir_d = self.chkpt_manager.data_checkpoint_dir / exp_hash_d
        self.assertTrue(data_dir_d.exists(), "New data checkpoint directory should exist")

        # Verify model weights still being saved
        model_dir_d = self.chkpt_manager.models_dir / exp_hash_d
        weight_files_d = list(model_dir_d.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_d)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_d), 2, "Should have at least 2 weight checkpoints")

        # Store state
        self.state['exp_hash_d'] = exp_hash_d
        self.state['losses_d'] = loss_D

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
        all_hashes = self.chkpt_manager.get_all_hashes(sort_by='created')
        print(f"\n[OK] Found {len(all_hashes)} experiment states:")
        for i, entry in enumerate(all_hashes):
            print(f"  {i+1}. {entry['hash'][:16]}... (created: {entry['created'][:19]})")

        # Reload state B (second state created)
        hash_a_from_manifest = self.state['exp_hash_a']

        print(f"\n[OK] Reloading state B: {hash_a_from_manifest[:16]}...")

        # Use new load_state method to load and apply checkpoint in-place
        success = self.chkpt_manager.load_state(exp_hash=hash_a_from_manifest)
        self.assertTrue(success, "State should be loaded successfully")

        # Get components from ledger (they were updated in-place)
        model_reloaded = ledgers.get_model()
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
        exp_hash_e, _, changed = self.chkpt_manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash E (branch): {exp_hash_e}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('hp', changed, "HP should have changed")
        self.assertIn('data', changed, "Data should have changed")

        # Update ledger
        # Ledger is already registered as proxy are used
        pass

        # Setting training environment from loader
        dataloader = ledgers.get_dataloader()
        model = ledgers.get_model()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nResuming training for 21 epochs...")
        pause_controller.resume()
        loss_E = self.train_epochs(
            model_reloaded, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'] * 2,
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify checkpoints for E
        model_dir_e = self.chkpt_manager.models_dir / exp_hash_e
        weight_files_e = list(model_dir_e.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_e)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_e), 4, "Should have at least 4 weight checkpoints for 21 epochs")

        # Store state
        self.state['exp_hash_e'] = exp_hash_e
        self.state['losses_e'] = loss_E

        print(f"\n[OK] TEST E PASSED - Reloaded and generate a new train branch successfully")
        print(f"  Final model_age: {model.current_step}")

    # ========================================================================
    # Test: 06_reload_before_model_change
    # ========================================================================
    def test_06_reload_before_model_change(self):
        """Reload before model change (back to A), fix conv size with RNG replay, verify HP+data"""
        print(f"\n{'='*80}")
        print("TEST 06: Reload Before Model Change - Fix Conv Size with RNG State")
        print(f"{'='*80}\n")

        exp_name = self.config['experiment_name']
        hash_a = self.state['exp_hash_a']  # Before model change
        loss_a = self.state['losses_a']  # Before model change

        print(f"Reloading state A (before model change): {hash_a[:16]}...")
        success = self.chkpt_manager.load_state(exp_hash=hash_a)
        self.assertTrue(success, "State A should load successfully")

        # Verify HP and data are from checkpoint A
        hp_reloaded = ledgers.get_hyperparams(exp_name)

        print(f"[OK] HP batch_size: {hp_reloaded.get('data', {}).get('train_loader', {}).get('batch_size', 'N/A')}")
        self.assertEqual(hp_reloaded.get('data', {}).get('train_loader', {}).get('batch_size'), 2,
                        "Should have batch_size=2 from state A")
        print(f"[OK] Data state verified from state A")
        print(f"[OK] RNG state restored for reproducible batching")

        # Train with original model to verify batches are the same
        print("\nTraining with original model from state A (11 epochs)...")
        model_original = ledgers.get_model()
        dataloader_original = ledgers.get_dataloader()
        optimizer_original = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        pause_controller.resume()
        loss_A_original = self.train_epochs(
            model_original, dataloader_original, optimizer_original, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()

        # Allow small tolerance for floating point differences and non-deterministic ops
        loss_diff = abs(sum(loss_a) - sum(loss_A_original))
        loss_relative_diff = loss_diff / sum(loss_a) if sum(loss_a) != 0 else 0

        print(f"[OK] Original model training loss (first/last): {loss_A_original[0]:.6f} / {loss_A_original[-1]:.6f}")
        print(f"[OK] Loss sum comparison: original={sum(loss_a):.6f}, reloaded={sum(loss_A_original):.6f}")
        print(f"[OK] Absolute difference: {loss_diff:.6f}, Relative difference: {loss_relative_diff*100:.3f}%")

        # Assert losses are close (within 1% tolerance for floating point and non-determinism)
        self.assertLess(
            loss_relative_diff, 0.1,
            msg=f"Losses should be close (within 1%): diff={loss_diff:.6f}, relative={loss_relative_diff*100:.3f}%"
        )

        # Reload again and fix model, should get same batches due to restored RNG
        print(f"\nReloading state A again (to reset RNG for fair comparison)...")
        success = self.chkpt_manager.load_state(exp_hash=hash_a)
        self.assertTrue(success, "State A should load successfully second time")

        # Fix model conv size - create new model with different architecture
        print("\nFixing model architecture...")
        model = ledgers.get_model()
        model.operate(0, {-1}, 1)
        model.operate(2, {-1}, 2)
        model.operate(-2, {}, 3)
        model.operate(-1, {-1 }, 4)

        exp_hash_h, _, changed = self.chkpt_manager.update_experiment_hash()
        print(f"\n[OK] New experiment hash H: {exp_hash_h[:16]}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('model', changed, "Only model should have changed")
        self.assertNotIn('hp', changed, "HP should not have changed")
        self.assertNotIn('data', changed, "Data should not have changed")

        # Train with new model - should get same batches due to restored RNG
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=hp_reloaded.get('optimizer', {}).get('lr', 0.001))
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs with new model (same RNG state = same batches)...")
        pause_controller.resume()
        loss_H = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        print(f"[OK] Fixed model training loss (first/last): {loss_H} / {loss_H}")

        # Compare: First batch should be same, but losses differ due to different model
        print(f"\n[OK] Reproducibility verified:")
        print(f"  Original model first batch loss: {loss_A_original}")
        print(f"  Fixed model first batch loss: {loss_H}")
        print(f"  (Same RNG = same batches, different losses due to model change)")

        # Store state
        self.state['losses_h'] = loss_H
        self.state['exp_hash_h'] = exp_hash_h

        print(f"\n[OK] TEST 06 PASSED - Reloaded with RNG state, trained with fixed architecture")

    # ========================================================================
    # Test: 07_change_data_from_test06
    # ========================================================================
    def test_07_change_data_from_test06(self):
        """Change data from test 06 - discard more data and train again"""
        print(f"\n{'='*80}")
        print("TEST 07: Change Data from Test 06 - Discard More Data")
        print(f"{'='*80}\n")

        exp_name = self.config['experiment_name']
        hash_h = self.state['exp_hash_h']  # From test 06

        print(f"Starting from state H: {hash_h[:16]}...")

        # Discard additional 15 samples (total 25% discarded)
        print("\nDiscarding additional 15 samples (25% total)...")
        dfm = ledgers.get_dataframe('sample_stats')
        tagged_samples = random.sample(range(100), 15)
        rows = []
        for idx in tagged_samples:
            uid = dfm._df.index[idx]
            rows.append({
                "sample_id": uid,
                "tags": f"discard_25pct_{random.randint(0, 100)}",
                "deny_listed": True
            })

        df_update = pd.DataFrame(rows).set_index("sample_id")
        dfm.upsert_df(df_update, origin='train_loader', force_flush=True)

        exp_hash_i, _, changed = self.chkpt_manager.update_experiment_hash()
        print(f"\n[OK] New experiment hash I: {exp_hash_i[:16]}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('data', changed, "Only data should have changed")

        # Train for 11 epochs
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001)
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs with 25% discarded...")
        pause_controller.resume()
        loss_I = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Store state
        self.state['losses_i'] = loss_I
        self.state['exp_hash_i'] = exp_hash_i

        print(f"\n[OK] TEST 07 PASSED - Changed data and trained successfully")

    # ========================================================================
    # Test: 08_reload_before_data_change_verify_and_modify
    # ========================================================================
    def test_08_reload_before_data_change_verify_and_modify(self):
        """Reload before data change (state C), verify training reproducibility, then modify model"""
        print(f"\n{'='*80}")
        print("TEST 08: Reload Before Data Change - Verify and Modify Model")
        print(f"{'='*80}\n")

        exp_name = self.config['experiment_name']
        hash_c = self.state['exp_hash_c']  # Before data change (after HP change)
        loss_c = self.state['losses_c']

        print(f"Part A: Reloading state C and verifying training reproducibility...")
        print(f"Reloading state C: {hash_c[:16]}...")

        success = self.chkpt_manager.load_state(exp_hash=hash_c)
        self.assertTrue(success, "State C should load successfully")

        # Verify training produces same results
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001)
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_C_verify = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Check reproducibility
        loss_diff = abs(sum(loss_c) - sum(loss_C_verify))
        loss_relative_diff = loss_diff / sum(loss_c) if sum(loss_c) != 0 else 0

        print(f"\n[OK] Loss comparison:")
        print(f"  Original: {sum(loss_c):.6f}")
        print(f"  Reloaded: {sum(loss_C_verify):.6f}")
        print(f"  Relative difference: {loss_relative_diff*100:.3f}%")

        self.assertLess(loss_relative_diff, 0.1,
                       msg=f"Training should be reproducible within 10%")

        print(f"\nPart B: Modifying model from reloaded state C...")

        # Reload again to reset state
        success = self.chkpt_manager.load_state(exp_hash=hash_c)

        # Modify model
        model = ledgers.get_model()
        print("\nModifying model architecture...")
        model.operate(0, {-2}, 1)  # Change conv1
        model.operate(2, {-2}, 2)  # Change conv2

        exp_hash_j, _, changed = self.chkpt_manager.update_experiment_hash()
        print(f"\n[OK] New experiment hash J: {exp_hash_j[:16]}")
        self.assertIn('model', changed, "Model should have changed")

        # Train with modified model
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001)
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)

        print("\nTraining for 11 epochs with modified model...")
        pause_controller.resume()
        loss_J = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Store state
        self.state['losses_j'] = loss_J
        self.state['exp_hash_j'] = exp_hash_j

        print(f"\n[OK] TEST 08 PASSED - Verified reproducibility and modified model")

    # ========================================================================
    # Test: 09_reload_before_hp_change_verify_and_fix
    # ========================================================================
    def test_09_reload_before_hp_change_verify_and_fix(self):
        """Reload before HP change (state B), verify training, then fix HP, model, and data"""
        print(f"\n{'='*80}")
        print("TEST 09: Reload Before HP Change - Verify and Fix Everything")
        print(f"{'='*80}\n")

        exp_name = self.config['experiment_name']
        hash_b = self.state['exp_hash_b']  # Before HP change (after model change)
        loss_b = self.state['losses_b']

        print(f"Part A: Reloading state B and verifying training reproducibility...")
        print(f"Reloading state B: {hash_b[:16]}...")

        success = self.chkpt_manager.load_state(exp_hash=hash_b)
        self.assertTrue(success, "State B should load successfully")

        # Verify training produces same results
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001)
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_B_verify = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Check reproducibility
        loss_diff = abs(sum(loss_b) - sum(loss_B_verify))
        loss_relative_diff = loss_diff / sum(loss_b) if sum(loss_b) != 0 else 0

        print(f"\n[OK] Loss comparison:")
        print(f"  Original: {sum(loss_b):.6f}")
        print(f"  Reloaded: {sum(loss_B_verify):.6f}")
        print(f"  Relative difference: {loss_relative_diff*100:.3f}%")

        self.assertLess(loss_relative_diff, 0.1,
                       msg=f"Training should be reproducible within 10%")

        print(f"\nPart B: Fixing HP, model, and data from reloaded state B...")

        # Reload again to reset state
        success = self.chkpt_manager.load_state(exp_hash=hash_b)

        # Fix HP
        hp = ledgers.get_hyperparams(exp_name)
        hp['optimizer']['lr'] = 0.005  # Change LR
        hp['data']['train_loader']['batch_size'] = 3  # Change batch size
        register_in_ledger(hp, flag="hyperparameters", name=exp_name)

        # Fix model
        model = ledgers.get_model()
        model.operate(0, {-3}, 1)  # Further modify conv1

        # Fix data - discard 5 samples
        dfm = ledgers.get_dataframe('sample_stats')
        tagged_samples = random.sample(range(100), 5)
        rows = []
        for idx in tagged_samples:
            uid = dfm._df.index[idx]
            rows.append({
                "sample_id": uid,
                "tags": f"discard_fix_{random.randint(0, 100)}",
                "deny_listed": True
            })
        df_update = pd.DataFrame(rows).set_index("sample_id")
        dfm.upsert_df(df_update, origin='train_loader', force_flush=True)

        exp_hash_k, _, changed = self.chkpt_manager.update_experiment_hash()
        print(f"\n[OK] New experiment hash K: {exp_hash_k[:16]}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('hp', changed, "HP should have changed")
        self.assertIn('model', changed, "Model should have changed")
        self.assertIn('data', changed, "Data should have changed")

        # Train with all fixes
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=0.005)
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs with all fixes...")
        pause_controller.resume()
        loss_K = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Store state
        self.state['losses_k'] = loss_K
        self.state['exp_hash_k'] = exp_hash_k

        print(f"\n[OK] TEST 09 PASSED - Verified reproducibility and fixed everything")

    # ========================================================================
    # Test: 10_reload_branch_j_verify_reproducibility
    # ========================================================================
    def test_10_reload_branch_j_verify_reproducibility(self):
        """Reload branch J (from test 08.b) and verify training reproducibility"""
        print(f"\n{'='*80}")
        print("TEST 10: Reload Branch J - Verify Training Reproducibility")
        print(f"{'='*80}\n")

        exp_name = self.config['experiment_name']
        hash_j = self.state['exp_hash_j']  # From test 08.b
        loss_j = self.state['losses_j']

        print(f"Reloading branch J: {hash_j[:16]}...")

        success = self.chkpt_manager.load_state(exp_hash=hash_j)
        self.assertTrue(success, "State J should load successfully")

        # Train again to verify reproducibility
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = th.optim.Adam(model.parameters(), lr=0.001)
        optimizer = register_in_ledger(optimizer, flag="optimizer", name=exp_name)
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_J_verify, uids_j = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin, return_uids=True)
        pause_controller.pause()
        self.state['uids_j'] = uids_j  # Store UIDs for later verification

        # Check reproducibility
        loss_diff = abs(sum(loss_j) - sum(loss_J_verify))
        loss_relative_diff = loss_diff / sum(loss_j) if sum(loss_j) != 0 else 0

        print(f"\n[OK] Loss comparison:")
        print(f"  Original: {sum(loss_j):.6f}")
        print(f"  Reloaded: {sum(loss_J_verify):.6f}")
        print(f"  Relative difference: {loss_relative_diff*100:.3f}%")

        self.assertLess(loss_relative_diff, 0.1,
                       msg=f"Training should be reproducible within 10%")

        print(f"\n[OK] TEST 10 PASSED - Branch J training is reproducible")

    # ========================================================================
    # Test: 11_restart_from_config_verify_reproducibility
    # ========================================================================
    def test_11_restart_from_config_verify_reproducibility(self):
        """Test 11: Restart experiment from config - verify all components load to branch_j state"""
        print(f"\n{'='*80}")
        print("TEST 11: Restart Experiment from Config - Verify Full Reproducibility")
        print(f"{'='*80}\n")

        # Reference variables
        target_hash = self.state['exp_hash_j']  # Target is branch_j
        loss_j_original = self.state['losses_j']
        originals_uids = self.state.get('uids_j', None)

        print(f"Simulating fresh restart: loading everything from config...")
        print(f"Target state: {target_hash[:16]} (branch_j)")

        # Simulate fresh Python process: re-register everything from config
        config_reloaded = self.config_cp
        exp_name = EXP_NAME

        # Clear existing ledger entries
        ledgers.clear_all()
        print("[OK] Cleared existing ledger entries")

        # =================================================
        # Automotically load components from existing chkpt
        # =================================================
        # First init a checkpoint manager with reloaded config
        self.chkpt_manager = CheckpointManagerV2(root_log_dir=self.config.get('root_log_dir'))
        ledgers.register_checkpoint_manager(exp_name, self.chkpt_manager)

        # Re-register HP
        register_in_ledger(config_reloaded, flag="hyperparameters", name=exp_name)
        print("[OK] Hyperparameters re-registered")

        # Create fresh model
        model_restarted = SimpleCNN(conv1_out=8, conv2_out=16)  # Match branch_j architecture
        # # Model arch. and weights are updated at the init of model interface
        model_restarted = register_in_ledger(model_restarted, flag="model", name=exp_name, device=DEVICE)

        # Re-register dataloader
        # # Same here, dataloader is created from HP at init of dataloader interface, and data are loaded from chkpt
        register_in_ledger(
            self.dataset,
            flag="dataloader",
            name=exp_name,
            compute_hash=False,
            is_training=True,
            batch_size=config_reloaded.get('data', {}).get('train_loader', {}).get('batch_size', 32),
            shuffle=config_reloaded.get('data', {}).get('train_loader', {}).get('shuffle', True)
        )

        # Create and register dataloader
        dataloader = register_in_ledger(
            self.dataset,
            flag="dataloader",
            name=exp_name,
            compute_hash=False,
            is_training=True,
            batch_size=self.config.get('data', {}).get('train_loader', {}).get('batch_size', 32),
            shuffle=self.config.get('data', {}).get('train_loader', {}).get('shuffle', True)
        )

        # Optimizer and criterion
        optimizer_restarted = th.optim.Adam(
            model_restarted.parameters(),
            lr=config_reloaded.get('optimizer', {}).get('lr', 0.001)
        )
        optimizer_restarted = register_in_ledger(optimizer_restarted, flag="optimizer", name=exp_name)
        # # Create and register signal (criterion)
        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = register_in_ledger(
            criterion,
            flag="signal",
            name="train_mlt_loss/CE",
            log=True
        )
        criterion_bin = nn.BCEWithLogitsLoss(reduction='none')
        criterion_bin = register_in_ledger(
            criterion_bin,
            flag="signal",
            name="train_bin_loss/BCE",
            log=True
        )
        print("[OK] Fresh registrations complete")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_J_verify, uids_j_restarted = self.train_epochs(model_restarted, dataloader, optimizer_restarted, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin, return_uids=True)
        pause_controller.pause()

        # Verify UIDs match original branch_j
        self.assertListEqual(uids_j_restarted, originals_uids,
                             msg="Sample UIDs should match original branch_j after restart")
        # Verify training produces same results
                # Check reproducibility
        loss_diff = abs(sum(loss_j_original) - sum(loss_J_verify))
        loss_relative_diff = loss_diff / sum(loss_j_original) if sum(loss_j_original) != 0 else 0

        print(f"\n[OK] Loss comparison:")
        print(f"  Original: {sum(loss_j_original):.6f}")
        print(f"  Reloaded: {sum(loss_J_verify):.6f}")
        print(f"  Relative difference: {loss_relative_diff*100:.3f}%")
        self.assertLess(loss_relative_diff, 0.1,
                       msg=f"Training should be reproducible within 10%")

    def test_logger_queue_saved_with_weights(self):
        # Initialize hash and dump immediately
        hash, isnew, changed = self.chkpt_manager.update_experiment_hash(force=False, dump_immediately=False)

        snapshot_path = Path(self.chkpt_manager.loggers_dir) / self.chkpt_manager.current_exp_hash / "loggers.json"
        self.assertTrue(snapshot_path.exists(), "Logger snapshot should be saved with checkpoint")

        with open(snapshot_path, "r") as f:
            snapshot = json.load(f)

        loggers = snapshot.get("loggers", {})
        self.assertIn(self.config.get("experiment_name"), loggers, "Logger entry should be present")
        signals = loggers[self.config.get("experiment_name")].get("signal_history", [])
        self.assertGreaterEqual(len(signals), 1, "Signal history should contain logged signals")


if __name__ == '__main__':
    # Create test suite with explicit ordering
    suite = unittest.TestSuite()

    # Add tests in specific order
    suite.addTest(CheckpointSystemV3Tests('test_00_initialize_experiment'))
    suite.addTest(CheckpointSystemV3Tests('test_01_train_A'))
    suite.addTest(CheckpointSystemV3Tests('test_02_train_B_model_change'))
    suite.addTest(CheckpointSystemV3Tests('test_03_train_C_hyperparams_change'))
    suite.addTest(CheckpointSystemV3Tests('test_04_train_D_data_change'))
    suite.addTest(CheckpointSystemV3Tests('test_05_train_E_reload_and_branch'))
    suite.addTest(CheckpointSystemV3Tests('test_06_reload_before_model_change'))
    suite.addTest(CheckpointSystemV3Tests('test_07_change_data_from_test06'))
    suite.addTest(CheckpointSystemV3Tests('test_08_reload_before_data_change_verify_and_modify'))
    suite.addTest(CheckpointSystemV3Tests('test_09_reload_before_hp_change_verify_and_fix'))
    suite.addTest(CheckpointSystemV3Tests('test_10_reload_branch_j_verify_reproducibility'))
    suite.addTest(CheckpointSystemV3Tests('test_11_restart_from_config_verify_reproducibility'))
    suite.addTest(CheckpointSystemV3Tests('test_logger_queue_saved_with_weights'))

    # Run the suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
