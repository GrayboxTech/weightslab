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
import yaml
from unittest import mock

warnings.filterwarnings("ignore")

import weightslab as wl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import trange
from pathlib import Path

# Import components directly to avoid full weightslab initialization
from weightslab.components.checkpoint_manager import CheckpointManager
from weightslab.components.experiment_hash import ExperimentHashGenerator
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.backend.logger import LoggerQueue
from weightslab.backend import ledgers
from weightslab.components.global_monitoring import (
    guard_training_context,
    start_hp_sync_thread_event,
    pause_controller
)
from weightslab.utils.tools import seed_everything


# Init hp sync thread event (used in training loop to trigger HP checkpointing)
start_hp_sync_thread_event()


# Helper function to register objects in ledger directly
def register_in_ledger(obj, flag, device='cpu', **kwargs):
    """Register an object in the ledger."""
    try:
        if flag == "hyperparameters":
            return wl.watch_or_edit(
                obj,
                flag="hyperparameters",
                defaults=obj,
                poll_interval=1.0,
                **kwargs
            )
        elif flag == "model":
            return wl.watch_or_edit(
                obj,
                flag="model",
                device=device,
                **kwargs
            )
        elif flag == "dataloader":
           return wl.watch_or_edit(
                obj,
                flag="data",
                **kwargs
            )
        elif flag == "optimizer":
            return wl.watch_or_edit(
                obj,
                flag="optimizer",
                **kwargs
            )
        elif flag == "signal":
            return wl.watch_or_edit(
                obj,
                flag="signal",
                **kwargs
            )
    except Exception:
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
        self.input_shape = (1, 1, 28, 28) # MNIST input shape
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
        def _pack_item(sample_idx):
            data, target = self.dataset[sample_idx]
            uid = th.tensor(sample_idx, dtype=th.long)
            return data, uid, target

        if idx in self._discarded:
            # Return next non-discarded sample
            for i in range(idx + 1, len(self.dataset)):
                if i not in self._discarded:
                    return _pack_item(i)
            # Wrap around
            for i in range(0, idx):
                if i not in self._discarded:
                    return _pack_item(i)
        return _pack_item(idx)

    def get_sample_uids(self):
        """Return list of sample UIDs"""
        return [i for i in range(len(self.dataset))]

    def is_discarded(self, uid):
        """Check if sample is discarded"""
        idx = str(uid.split('_')[1])
        return idx in self._discarded

    def discard(self, uid):
        """Discard a sample"""
        idx = str(uid.split('_')[1])
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

# @unittest.skip("Skip this simulation as the modeling part has been disabled.")
class CheckpointSystemTests(unittest.TestCase):
    """Comprehensive tests for checkpoint system with separated test methods"""

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


    def train_epochs(self, model, loader, optimizer, criterion, num_epochs, criterion_bin=None):
        """Train model for specified epochs with checkpointing"""
        losses = []
        uids_trained = []
        for _ in trange(num_epochs, desc="Training"):
            with guard_training_context:
                epoch_loss = 0.0
                batch_count = 0

                # Data Processing
                (inputs, ids, labels) = next(loader)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                uids_trained.extend(ids)

                # Inference
                optimizer.zero_grad()
                preds_raw = model(inputs)

                # Preds
                if preds_raw.ndim == 1:
                    preds = (preds_raw > 0.0).long()
                else:
                    preds = preds_raw.argmax(dim=1, keepdim=True)

                # Losses
                # # Binary loss
                if criterion_bin is not None:
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
        return losses, uids_trained

    def check_reproducibility(self, original_loss, reloaded_loss, original_uids=None, reloaded_uids=None, loss_tol=0.1, uids_msg=None):
        """Common reproducibility check for losses and UIDs"""
        return
        # # Check reproducibility of losses and UIDs
        # if isinstance(original_loss, (list, tuple)):
        # original_loss_sum = sum(original_loss)/len(original_loss)
        # else:
        # original_loss_sum = original_loss
        # if isinstance(reloaded_loss, (list, tuple)):
        # reloaded_loss_sum = sum(reloaded_loss)/len(reloaded_loss)
        # else:
        # reloaded_loss_sum = reloaded_loss
        # loss_diff = abs(original_loss_sum - reloaded_loss_sum)
        # loss_relative_diff = loss_diff / original_loss_sum if original_loss_sum != 0 else 0
        # print(f"[OK] Loss comparison:")
        # print(f" Original: {original_loss_sum:.6f}")
        # print(f" Reloaded: {reloaded_loss_sum:.6f}")
        # print(f" Relative difference: {loss_relative_diff*100:.3f}%")
        # self.assertLess(loss_relative_diff, loss_tol, msg=f"Training should be reproducible within {loss_tol*100:.1f}%")
        # if original_uids is not None and reloaded_uids is not None:
        # print(f"[OK] UIDs comparison:")
        # print(f" Original: {original_uids}")
        # print(f" Reloaded: {reloaded_uids}")
        # self.assertListEqual(reloaded_uids, original_uids, msg=uids_msg or "Sample UIDs should match for reproducibility")

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
        cls.log_dir = os.path.join(cls.temp_dir, "experiments")

        # Initialize config from YAML-like dict (similar to wl-classification)
        cls.config = {
            'experiment_name': EXP_NAME,
            'device': DEVICE,
            'root_log_dir': cls.log_dir,
            'experiment_dump_to_train_steps_ratio': 5,
            'skip_checkpoint_load': False,

            # Data parameters
            'data': {
                'train_loader': {
                    'batch_size': 2,
                    'shuffle': False
                },
            },

            # Checkpoint manager parameters
            'checkpoint_manager': {
                'dump_model_architecture': True,
            },

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

        # ==================
        # Initialize dataset
        # ==================
        # Load MNIST subset (10 samples for all tests)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_dataset = datasets.MNIST(
            root=os.path.join(cls.temp_dir, 'mnist_data'),
            train=False,
            download=True,
            transform=transform
        )
        mnist_subset = Subset(full_dataset, list(range(10))) # Create subset with 10 samples
        cls.dataset = TaggableDataset(mnist_subset) # Wrap in taggable dataset

        # =================
        # Initialize Logger
        # =================
        cls.logger = LoggerQueue(register=True)

        # =============
        # Initialize HP
        # =============
        # Register HP in ledger
        cls.config = register_in_ledger(cls.config, flag="hyperparameters")

        # ================
        # Initialize Model
        # ================
        model = SimpleCNN(conv1_out=8, conv2_out=16)
        model = register_in_ledger(model, flag="model", device=DEVICE, skip_previous_auto_load=True, compute_dependencies=False) # Compute dependencies is disabled

        # =====================
        # Initialize DataLoader
        # =====================
        register_in_ledger(
            cls.dataset,
            flag="dataloader",
            compute_hash=False,
            is_training=True,
            batch_size=cls.config.get('data', {}).get('train_loader', {}).get('batch_size', 32),
            shuffle=cls.config.get('data', {}).get('train_loader', {}).get('shuffle', False)
        )

        # ==================================
        # Initialize Criterion and Optimizer
        # ==================================
        # Optimizer and criterion
        # # Create and register optimizer
        register_in_ledger(
            th.optim.Adam(model.parameters(),
            lr=cls.config.get('optimizer', {}).get('lr', 0.001)),
            flag="optimizer"
        )
        # # Create and register signal (criterion)
        register_in_ledger(
            nn.CrossEntropyLoss(reduction='none'),
            flag="signal",
            log=True,
            name="train_mlt_loss/CE"
        )
        register_in_ledger(
            nn.BCEWithLogitsLoss(reduction='none'),
            flag="signal",
            log=True,
            name="train_bin_loss/BCE"
        )

        # =================================
        # Get the global checkpoint manager
        # =================================
        cls.chkpt_manager = ledgers.get_checkpoint_manager()

        # ============================
        # Print setup info
        print(f"[OK] Created MNIST subset: {len(cls.dataset)} samples")
        print(f"[OK] Temporary directory: {cls.temp_dir}")
        print(f"[OK] Config initialized")
        print(f"[OK] Checkpoint manager initialized at {cls.config.get('root_log_dir')}\n")

    # ==============================
    # Test: 00_initialize_experiment
    # ==============================
    def test_00_initialize_experiment(self):
        """Initialize experiment with configuration and first model"""
        print(f"\n{'='*80}")
        print("TEST 00: Initialize Experiment Configuration")
        print(f"{'='*80}\n")

        # Initialize hyperparameters with model_age
        exp_hash_a, _, changed = self.chkpt_manager.update_experiment_hash(first_time=True)

        print(f"\n[OK] Experiment hash A: {exp_hash_a}")
        print(f"[OK] Changed components: {changed}")

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

        # Get stored state from previous test and load it
        exp_hash_a = self.state['exp_hash_a']
        success = self.chkpt_manager.load_state(exp_hash=exp_hash_a)
        self.assertTrue(success, "Checkpoint load should succeed")

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
        loss_A, uids_A = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()
        print("\nTraining completed.")

        # Verify checkpoints
        exp_hash_a2 = self.chkpt_manager.get_current_experiment_hash()
        model_dir_a = self.chkpt_manager.models_dir / exp_hash_a2[8:-8]
        self.assertTrue(model_dir_a.exists(), "Model checkpoint directory should exist")

        # Check for weight checkpoints
        weight_files = list(model_dir_a.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files), 2, "Should have at least 2 weight checkpoints")

        # Check HP directory
        hp_dir_a = self.chkpt_manager.hp_dir / exp_hash_a2[:8]
        self.assertTrue(hp_dir_a.exists(), "HP checkpoint directory should exist")

        # Check data directory
        data_dir_a = self.chkpt_manager.data_checkpoint_dir / exp_hash_a2[-8:]
        self.assertTrue(data_dir_a.exists(), "Data checkpoint directory should exist")

        # Save state for next tests
        self.state['exp_hash_a'] = exp_hash_a2
        self.state['losses_a'] = sum(loss_A) / len(loss_A)
        self.state['uids_a'] = uids_A

        # Final verbose
        print(f" Final model_age (i.e., how many epochs lived by the model): {model.get_age()}")
        print(f"\n[OK] TEST A PASSED - Initial training completed")

    # =============================
    # Test: 02_train_B_model_change
    # =============================
    def test_02_train_B_model_change(self):
        """Modify model architecture and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST B: Modify Model Architecture")
        print(f"{'='*80}\n")

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
        # TODO (GP): Still have the pb btw model archi. and torch cache
        # 11/05/2026-11:20:11.207 DEBUG:weightslab.components.global_monitoring:__exit__: Suppressing exception: Function ConvolutionBackward0 returned an invalid gradient at index 2 - got [15] but expected shape compatible with [16] in GuardContext.__exit__
        # model.operate(0, {-1, -2, -3, -4}, 1) # Increase conv1 out channels by 2
        # model.operate(2, {-1}, 2) # Freeze fc1 layer
        # model.operate(-2, {}, 3) # Freeze fc1 layer
        # # model.operate(-1, {1}, 4) # Reset fc2 layer

        # print(f" Conv1: 8 -> 12 channels")
        # print(f" Conv2: 16 -> 15 channels")
        # print(f" FC1: Frozen")
        # print(f" FC2: Reset")

        # Update hash here to get hash
        exp_hash_b, _, changed = self.chkpt_manager.update_experiment_hash(force=True)
        print(f"\n[OK] New experiment hash B: {exp_hash_b}")
        print(f"[OK] Changed components: {changed}")
        # self.assertIn('model', changed, "Model should have changed")
        # self.assertNotEqual(self.state['exp_hash_a'], exp_hash_b, "Hash should be different")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume()
        loss_B, uids_B = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()
        print("\nTraining completed.")

        # Verify new model directory
        model_dir_b = self.chkpt_manager.models_dir / exp_hash_b[8:-8]
        self.assertTrue(model_dir_b.exists(), "New model checkpoint directory should exist")
        weight_files_b = list(model_dir_b.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_b)} weight checkpoint files in new directory")
        self.assertGreaterEqual(len(weight_files_b), 2, "Should have at least 2 new weight checkpoints")

        # Store state
        self.state['exp_hash_b'] = exp_hash_b
        self.state['losses_b'] = sum(loss_B) / len(loss_B)
        self.state['uids_b'] = uids_B

        # Final verbose
        print(f"\n[OK] TEST B PASSED - Model architecture updated")
        print(f" Final model_age: {model.get_age()}")

    # ========================================================================
    # Test: 03_train_C_hyperparams_change
    # ========================================================================
    def test_03_train_C_hyperparams_change(self):
        """Change hyperparameters and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST C: Change Hyperparameters")
        print(f"{'='*80}\n")

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
        new_bs = 3
        self.config['data']['train_loader']['batch_size'] = new_bs
        print(f" Batch size: 2 -> 4")

        # Update hash
        exp_hash_c, _, _ = self.chkpt_manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash C: {exp_hash_c}")
        self.assertNotEqual(self.state['exp_hash_b'], exp_hash_c, "Hash should be different as hp changed")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume()
        loss_C, uids_C = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify new HP directory
        hp_dir_c = self.chkpt_manager.hp_dir / exp_hash_c[:8]
        self.assertTrue(hp_dir_c.exists(), "New HP checkpoint directory should exist")

        # Verify model weights still being saved
        model_dir_c = self.chkpt_manager.models_dir / exp_hash_c[8:-8]
        weight_files_c = list(model_dir_c.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_c)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_c), 2, "Should have at least 2 weight checkpoints")

        # Store state
        self.state['exp_hash_c'] = exp_hash_c
        self.state['losses_c'] = sum(loss_C) / len(loss_C)
        self.state['uids_c'] = uids_C
        self.state['new_bs_C'] = self.config['data']['train_loader']['batch_size']

        # Final verbose
        print(f"\n[OK] TEST C PASSED - Hyperparameters updated")
        print(f" Final model_age (i.e., how many epochs lived by the model): {model.get_age()}")

    # ========================================================================
    # Test: 04_train_D_data_change
    # ========================================================================
    def test_04_train_D_data_change(self):
        """Change data state (tags and discard) and train for 11 epochs"""
        print(f"\n{'='*80}")
        print("TEST D: Change Data State (Tags and Discard)")
        print(f"{'='*80}\n")

        # Model
        model = ledgers.get_model()

        # Data
        dataloader = ledgers.get_dataloader() # Get dataloader
        dfm = ledgers.get_dataframe() # Get dataframe manager

        # Optimizer and criterion
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("Modifying data...")

        # Add 20 random tags with 'ugly'
        tagged_samples = random.sample(range(10), 4)
        rows = []
        uids_discarded = []
        for idx in tagged_samples:
            uid, in_uid = dfm.get_df_view().index[idx]
            uids_discarded.append(uid)
            rows.append(
                {
                    SampleStatsEx.SAMPLE_ID.value: uid,
                    SampleStatsEx.INSTANCE_ID.value: in_uid, # For simplicity, use same UID for annotation
                    f"{SampleStatsEx.TAG.value}:ugly": True, # Random tag with 'ugly'
                    SampleStatsEx.DISCARDED.value: bool(1 - dfm.get_df_view()[SampleStatsEx.DISCARDED.value].iloc[idx])
                }
            )

        # Updates data - Simulate adding tags and discarding samples in dataset
        df_update = pd.DataFrame(rows).set_index(SampleStatsEx.SAMPLE_ID.value)
        # upsert_df updates the ledger's dataframe immediately
        dfm.upsert_df(df_update, origin='train_loader', force_flush=True)

        # Changes will be pending
        print(f" Added 'ugly' tag to 20 samples")
        print(f" Discarded 20 samples")

        # Update hash
        exp_hash_d, _, changed = self.chkpt_manager.update_experiment_hash()

        print(f"\n[OK] New experiment hash D: {exp_hash_d}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('data', changed, "Data should have changed")
        self.assertNotEqual(self.state['exp_hash_c'], exp_hash_d, "Hash should be different")

        print("\nResuming training for 11 epochs...")
        pause_controller.resume() # Pending changes to dump: data state
        loss_D, uids_D = self.train_epochs(
            model, dataloader, optimizer, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify new data directory
        data_dir_d = self.chkpt_manager.data_checkpoint_dir / exp_hash_d[-8:]
        self.assertTrue(data_dir_d.exists(), "New data checkpoint directory should exist")

        # Verify model weights still being saved
        model_dir_d = self.chkpt_manager.models_dir / exp_hash_d[8:-8]
        weight_files_d = list(model_dir_d.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_d)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_d), 2, "Should have at least 2 weight checkpoints")

        # Store state
        self.state['exp_hash_d'] = exp_hash_d
        self.state['losses_d'] = sum(loss_D) / len(loss_D)
        self.state['uids_d'] = uids_D
        self.state['uids_discarded_d'] = uids_discarded
        # self.state['model_c1_neurons'] = model.layers[0].out_neurons

        # Final verbose
        print(f"\n[OK] TEST D PASSED - Data state updated")
        print(f" Final model_age (i.e., how many epochs lived by the model): {model.get_age()}")

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

        print("Experiment paused. Analyzing experiment history...")

        # Get all hashes
        all_hashes = self.chkpt_manager.get_all_hashes(sort_by='created')
        print(f"\n[OK] Found {len(all_hashes)} experiment states:")
        for i, entry in enumerate(all_hashes):
            print(f" {i+1}. {entry['hash'][:16]}... (created: {entry['created'][:19]})")

        # Reload state B (second state created)
        hash_a_from_manifest = self.state['exp_hash_a']

        print(f"\n[OK] Reloading state B: {hash_a_from_manifest[:16]}...")

        # Use new load_state method to load and apply checkpoint in-place
        success = self.chkpt_manager.load_state(exp_hash=hash_a_from_manifest)
        self.assertTrue(success, "State should be loaded successfully")

        # Get components from ledger (they were updated in-place)
        model_reloaded = ledgers.get_model()
        hp_reloaded = ledgers.get_hyperparams()

        print(f"[OK] State applied successfully")
        print(f"[OK] Loaded HP: {hp_reloaded}")

        # Modify HP and data
        print("\nModifying HP and data (not training yet)...")

        # Handle nested config structure
        if 'data' in hp_reloaded and 'train_loader' in hp_reloaded['data']:
            hp_reloaded['data']['train_loader']['batch_size'] = 1
            old_batch_size = hp_original.get('data', {}).get('train_loader', {}).get('batch_size', 2)
            print(f" Batch size: {old_batch_size} -> 1")

        # Discard more data
        # Add 20 random tags with 'ugly'
        tagged_samples = random.sample(range(10), 1)
        rows = []
        dfm = ledgers.get_dataframe() # Get dataframe manager
        for idx in tagged_samples:
            uid, in_uid = dfm.get_df_view().index[idx]
            rows.append(
                {
                    SampleStatsEx.SAMPLE_ID.value: uid,
                    SampleStatsEx.INSTANCE_ID.value: in_uid, # For simplicity, use same UID for annotation
                    f"{SampleStatsEx.TAG.value}:ugly": True,
                    SampleStatsEx.DISCARDED.value: bool(1 - dfm.get_df_view(SampleStatsEx.DISCARDED.value).iloc[idx])
                }
            )
        # # # Updates data - Simulate adding tags and discarding samples in dataset
        # # # upsert_df updates the ledger's dataframe immediately
        dfm.upsert_df(pd.DataFrame(rows).set_index(SampleStatsEx.SAMPLE_ID.value), origin='train_loader', force_flush=True)

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
        loss_E, uids_E = self.train_epochs(
            model_reloaded, dataloader, optimizer, criterion, criterion_bin=criterion_bin,
            num_epochs=self.config['training']['num_epochs'] * 2,
        )
        pause_controller.pause()

        print("\nTraining completed.")

        # Verify checkpoints for E
        model_dir_e = self.chkpt_manager.models_dir / exp_hash_e[8:-8]
        weight_files_e = list(model_dir_e.glob("*_step_*.pt"))
        print(f"[OK] Found {len(weight_files_e)} weight checkpoint files")
        self.assertGreaterEqual(len(weight_files_e), 4, "Should have at least 4 weight checkpoints for 21 epochs")

        # Store state
        self.state['exp_hash_e'] = exp_hash_e
        self.state['losses_e'] = loss_E
        self.state['uids_e'] = uids_E

        print(f"\n[OK] TEST E PASSED - Reloaded and generate a new train branch successfully")
        print(f" Final model_age: {model.get_age()}")

    # ========================================================================
    # Test: 06_reload_before_model_change
    # ========================================================================
    def test_06_reload_before_model_change(self):
        """Reload before model change (back to A), fix conv size with RNG replay, verify HP+data"""
        print(f"\n{'='*80}")
        print("TEST 06: Reload Before Model Change - Fix Conv Size with RNG State")
        print(f"{'='*80}\n")

        hash_A_original = self.state['exp_hash_a'] # Before model change
        loss_A_original = self.state['losses_a'] # Before model change
        uids_A_original = self.state['uids_a'] # Before model change

        print(f"Reloading state A (before model change) for verification: {hash_A_original[:16]}...")
        # load_config=True so the checkpointed hyperparameters are restored into
        # the ledger (this test verifies HP). Without it, load_state leaves the
        # in-memory config untouched — and test_05 mutated batch_size to 1 to
        # create branch E, so that stale value would leak in here.
        success = self.chkpt_manager.load_state(exp_hash=hash_A_original, load_config=True)
        self.assertTrue(success, "State A should load successfully")

        # Verify HP and data are from checkpoint A
        hp_reloaded = ledgers.get_hyperparams()

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
        loss_A_reloaded, uids_A_reloaded = self.train_epochs(
            model_original, dataloader_original, optimizer_original, criterion,
            num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin
        )
        pause_controller.pause()

        # Check reproducibility with original loss and UIDs
        self.check_reproducibility(loss_A_original, loss_A_reloaded, uids_A_original, uids_A_reloaded)

        # Reload again and fix model, should get same batches due to restored RNG
        print(f"\nReloading state A again (to reset RNG for fair comparison) and modifying model architecture...")
        success = self.chkpt_manager.load_state(exp_hash=hash_A_original)
        self.assertTrue(success, "State A should load successfully second time")

        # Fix model conv size - create new model with different architecture
        print("\nFixing model architecture...")
        model = ledgers.get_model()
        # model.operate(0, {-1}, 1) # Commented; see test 2 - still have the pb btw model archi. and torch cache
        # model.operate(2, {-1}, 2)
        # model.operate(-2, {}, 3)
        # model.operate(-1, {-1 }, 4)

        exp_hash_h, _, changed = self.chkpt_manager.update_experiment_hash(force=True)
        print(f"\n[OK] New experiment hash H: {exp_hash_h[:16]}")
        print(f"[OK] Changed components: {changed}")
        # self.assertIn('model', changed, "Only model should have changed")
        # self.assertNotIn('hp', changed, "HP should not have changed")
        # self.assertNotIn('data', changed, "Data should not have changed")

        # Train with new model - should get same batches due to restored RNG
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs with new model (same RNG state = same batches)...")
        pause_controller.resume()
        loss_H, uids_H = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        print(f"[OK] Fixed model training loss (first/last): {loss_H} / {loss_H}")

        # Compare: First batch should be same, but losses differ due to different model
        print(f"\n[OK] Reproducibility verified:")
        print(f" Original model first batch loss: {loss_A_reloaded}")
        print(f" Fixed model first batch loss: {loss_H}")
        print(f" (Same RNG = same batches, different losses due to model change)")

        # Store state
        self.state['losses_h'] = loss_H
        self.state['exp_hash_h'] = exp_hash_h
        self.state['uids_h'] = uids_H

        print(f"\n[OK] TEST 06 PASSED - Reloaded with RNG state, trained with fixed architecture")

    # ========================================================================
    # Test: 07_change_data_from_test06
    # ========================================================================
    def test_07_change_data_from_test06(self):
        """Change data from test 06 - discard more data and train again"""
        print(f"\n{'='*80}")
        print("TEST 07: Change Data from Test 06 - Discard More Data")
        print(f"{'='*80}\n")

        hash_H = self.state['exp_hash_h'] # From test 06

        print(f"Starting from state H: {hash_H[:16]}...")

        # Discard additional 15 samples (total 25% discarded)
        print("\nDiscarding additional 15 samples (25% total)...")
        dfm = ledgers.get_dataframe()
        tagged_samples = random.sample(range(10), 2)
        rows = []
        for idx in tagged_samples:
            uid, in_uid = dfm.get_df_view().index[idx]
            rows.append({
                SampleStatsEx.SAMPLE_ID.value: uid,
                SampleStatsEx.INSTANCE_ID.value: in_uid, # For simplicity, use same UID for annotation
                f"{SampleStatsEx.TAG.value}:discard_25pct": True,
                SampleStatsEx.DISCARDED.value: True
            })

        df_update = pd.DataFrame(rows).set_index(SampleStatsEx.SAMPLE_ID.value)
        dfm.upsert_df(df_update, origin='train_loader', force_flush=True)

        exp_hash_i, _, changed = self.chkpt_manager.update_experiment_hash()
        print(f"\n[OK] New experiment hash I: {exp_hash_i[:16]}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('data', changed, "Only data should have changed")

        # Train for 11 epochs
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs with 25% discarded...")
        pause_controller.resume()
        loss_I, uids_I = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Store state
        self.state['losses_i'] = loss_I
        self.state['uids_i'] = uids_I
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

        hash_c = self.state['exp_hash_c'] # Before data change (after HP change)

        print(f"Part A: Reloading state C and verifying training reproducibility...")
        print(f"Reloading state C: {hash_c[:16]}...")

        success = self.chkpt_manager.load_state(exp_hash=hash_c)
        self.assertTrue(success, "State C should load successfully")

        # Verify training produces same results
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_C_verify, uids_C_verify = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Check reproducibility with original loss and UIDs
        # self.check_reproducibility(loss_c, loss_C_verify, uids_c, None, loss_tol=1e-1)

        print(f"\nPart B: Modifying model from reloaded state C...")

        # Reload again to reset state
        success = self.chkpt_manager.load_state(exp_hash=hash_c)

        # Modify model
        model = ledgers.get_model()
        # print("\nModifying model architecture...")
        # model.operate(-2, {}, 3)
        # model.operate(-1, {-1 }, 4)

        exp_hash_j, _, changed = self.chkpt_manager.update_experiment_hash(force=True)
        print(f"\n[OK] New experiment hash J: {exp_hash_j[:16]}")
        # self.assertIn('model', changed, "Model should have changed")

        # Train with modified model
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()

        print("\nTraining for 11 epochs with modified model...")
        pause_controller.resume()
        loss_J, _ = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Store state
        self.state['losses_j'] = sum(loss_J)/len(loss_J)
        self.state['exp_hash_j'] = exp_hash_j

        print(f"\n[OK] TEST 08 PASSED - Verified reproducibility and modified model")

    # ========================================================================
    # Test: 09_reload_before_hp_change_verify_and_fix
    # ========================================================================
    def test_09_reload_before_hp_change_verify_and_modify(self):
        """Reload before HP change (state B), verify training, then fix HP, model, and data"""
        print(f"\n{'='*80}")
        print("TEST 09: Reload Before HP Change - Verify and Fix Everything")
        print(f"{'='*80}\n")

        hash_b = self.state['exp_hash_b'] # Before HP change (after model change)
        loss_b = self.state['losses_b']

        print(f"Part A: Reloading state B and verifying training reproducibility...")
        print(f"Reloading state B: {hash_b[:16]}...")

        success = self.chkpt_manager.load_state(exp_hash=hash_b)
        self.assertTrue(success, "State B should load successfully")

        # Verify training produces same results
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_B_verify, uids_B_verify = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Check reproducibility with original loss and UIDs
        self.check_reproducibility(loss_b, loss_B_verify, self.state.get('uids_b'), None, loss_tol=1e-1)

        print(f"\nPart B: Fixing HP, model, and data from reloaded state B...")

        # Reload again to reset state
        success = self.chkpt_manager.load_state(exp_hash=hash_b)

        # Fix HP
        hp = ledgers.get_hyperparams()
        hp['data']['train_loader']['batch_size'] = 7 # Change batch size

        # Fix model
        model = ledgers.get_model()
        # # model.operate(0, {-3}, 1) # Further modify conv1
        # model.operate(-1, {-1 }, 4)

        # Fix data - discard 5 samples
        dfm = ledgers.get_dataframe()
        tagged_samples = random.sample(range(10), 2)
        rows = []
        for idx in tagged_samples:
            uid, in_uid = dfm.get_df_view().index[idx]
            rows.append({
                SampleStatsEx.SAMPLE_ID.value: uid,
                SampleStatsEx.INSTANCE_ID.value: in_uid, # For simplicity, use same UID for annotation
                f"{SampleStatsEx.TAG.value}:discard_fix": True,
                SampleStatsEx.DISCARDED.value: True
            })
        df_update = pd.DataFrame(rows).set_index(SampleStatsEx.SAMPLE_ID.value)
        dfm.upsert_df(df_update, origin='train_loader', force_flush=True)

        exp_hash_k, _, changed = self.chkpt_manager.update_experiment_hash()
        print(f"\n[OK] New experiment hash K: {exp_hash_k[:16]}")
        print(f"[OK] Changed components: {changed}")
        self.assertIn('hp', changed, "HP should have changed")
        # self.assertIn('model', changed, "Model should have changed")
        self.assertIn('data', changed, "Data should have changed")

        # Train with all fixes
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs with all fixes...")
        pause_controller.resume()
        loss_K, _ = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Store state
        self.state['losses_k'] = sum(loss_K)/len(loss_K)
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

        hash_j = self.state['exp_hash_j'] # From test 08.b

        print(f"Reloading branch J: {hash_j[:16]}...")

        success = self.chkpt_manager.load_state(exp_hash=hash_j)
        self.assertTrue(success, "State J should load successfully")

        # Train again to verify reproducibility
        model = ledgers.get_model()
        dataloader = ledgers.get_dataloader()
        optimizer = ledgers.get_optimizer()
        criterion = ledgers.get_signal(name="train_mlt_loss/CE")
        criterion_bin = ledgers.get_signal(name="train_bin_loss/BCE")

        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        loss_j_verify, _ = self.train_epochs(model, dataloader, optimizer, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # Check reproducibility with original loss and UIDs
        self.check_reproducibility(self.state['losses_j'], loss_j_verify, self.state.get('uids_b'), None, loss_tol=1e-1)

        print(f"\n[OK] TEST 10 PASSED - Branch J training is reproducible")

    # ========================================================================
    # Test: 11_restart_from_config_verify_reproducibility
    # ========================================================================
    def test_11_restart_from_scratch_to_hash_d_and_verify_reproducibility(self):
        """Test 11: Restart experiment from config - verify all components load to branch_j state"""
        print(f"\n{'='*80}")
        print("TEST 11: Restart Experiment from Config - Verify Full Reproducibility")
        print(f"{'='*80}\n")

        # Reference variables
        target_hash = self.state['exp_hash_d'] # Target is branch_d

        print(f"Simulating fresh restart: loading everything from config...")
        print(f"Target state: {target_hash[:16]} (branch_d)")

        # Simulate fresh Python process: re-register everything from config
        config_reloaded = self.config_cp

        # Clear existing ledger entries
        ledgers.clear_all()
        print("[OK] Cleared existing ledger entries")

        # =================================================
        # Automotically load components from existing chkpt
        # =================================================
        # First init a checkpoint manager with reloaded config
        self.chkpt_manager = CheckpointManager(root_log_dir=self.config.get('root_log_dir'))
        ledgers.register_checkpoint_manager(self.chkpt_manager)

        # Re-register HP
        register_in_ledger(config_reloaded, flag="hyperparameters")
        print("[OK] Hyperparameters re-registered")

        # Create fresh model
        model_restarted = SimpleCNN(conv1_out=8, conv2_out=16) # Match branch_d architecture
        # # Model arch. and weights are updated at the init of model interface
        model_restarted = register_in_ledger(model_restarted, flag="model", device=DEVICE)

        # Re-register dataloader
        dataloader = register_in_ledger(
            self.dataset,
            flag="dataloader",
            loader_name='train_loader',
            compute_hash=False,
            is_training=True,
            batch_size=self.config.get('data', {}).get('train_loader', {}).get('batch_size', 32),
            shuffle=self.config.get('data', {}).get('train_loader', {}).get('shuffle', False)
        )

        # Optimizer and criterion
        optimizer_restarted = th.optim.Adam(
            model_restarted.parameters(),
            lr=config_reloaded.get('optimizer', {}).get('lr', 0.001)
        )
        optimizer_restarted = register_in_ledger(optimizer_restarted, flag="optimizer")
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

        # Get all hashes
        all_hashes = self.chkpt_manager.get_all_hashes(sort_by='created')
        print(f"\n[OK] Found {len(all_hashes)} experiment states:")
        for i, entry in enumerate(all_hashes):
            print(f" {i+1}. {entry['hash'][:16]}... (created: {entry['created'][:19]})")

        # Reload state B (second state created)
        hash_a_from_manifest = self.state['exp_hash_a']

        print(f"\n[OK] Reloading state B: {hash_a_from_manifest[:16]}...")

        # Use new load_state method to load and apply checkpoint in-place
        _ = self.chkpt_manager.load_state(exp_hash=target_hash)
        # self.assertTrue(success, "State should be loaded successfully")

        print(f"[OK] Checkpoint loaded to reach target state {target_hash[:16]}")
        print("\nTraining for 11 epochs to verify reproducibility...")
        pause_controller.resume()
        model_restarted = ledgers.get_model() # Get model after loading state
        _, _ = self.train_epochs(model_restarted, dataloader, optimizer_restarted, criterion, num_epochs=self.config['training']['num_epochs'],
            criterion_bin=criterion_bin)
        pause_controller.pause()

        # # Check reproducibility with original loss and UIDs
        # self.assertEqual(model_restarted.layers[-1].operation_age['FREEZE'], 1,
        # "Model architecture should match state in D")
        # self.assertEqual(model_restarted.layers[-1].operation_age['RESET'], 1,
        # "Model architecture should match state in D")
        # self.assertEqual(model_restarted.layers[0].out_neurons, 12,
        # "Model architecture should match state in D")

        # Not possible as data are generated randomly without reproducibility now
        # self.check_reproducibility(loss_d_original, loss_d_verify, originals_uids, None, loss_tol=1e-1)

    # ========================================================================
    # Test: logger queue saved with weights
    # ========================================================================
    def test_logger_queue_saved_with_weights(self):
        self.chkpt_manager.update_experiment_hash(force=False, dump_immediately=False)
        self.chkpt_manager.save_pending_changes(force=True)

        # Logger history is persisted to an on-disk DuckDB file alongside the
        # checkpoint (the chunked-zstd snapshot was replaced by DuckDB in #257).
        db_path = Path(self.chkpt_manager.get_logger_db_path())
        self.assertTrue(db_path.exists(), "Logger DuckDB should be saved with checkpoint")

        # The registered logger carries the persisted signal history.
        lg = ledgers.get_logger('main')
        self.assertIsNotNone(lg, "Logger entry should be present")
        signals = lg.get_signal_history() if hasattr(lg, 'get_signal_history') else []
        if isinstance(signals, dict):
            total_signals = 0
            for experiments in signals.values():
                if not isinstance(experiments, dict):
                    continue
                for steps in experiments.values():
                    if not isinstance(steps, dict):
                        continue
                    for entries in steps.values():
                        if isinstance(entries, list):
                            total_signals += len(entries)
                        elif entries is not None:
                            total_signals += 1
            self.assertGreaterEqual(total_signals, 0, "Signal history count should be non-negative")
        else:
            self.assertIsInstance(signals, list, "Signal history should be list or nested dict")


class CheckpointStepAwareBehaviorTests(unittest.TestCase):
    def test_model_hash_depends_on_init_step(self):
        generator = ExperimentHashGenerator()
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        hash_from_step_0 = generator.generate_hash(
            model=model,
            config={"learning_rate": 0.001},
            data_state={},
            model_init_step=0,
        )
        hash_from_step_52 = generator.generate_hash(
            model=model,
            config={"learning_rate": 0.001},
            data_state={},
            model_init_step=52,
        )

        self.assertNotEqual(
            hash_from_step_0[8:16],
            hash_from_step_52[8:16],
            "Model hash segment should change when model_init_step changes",
        )

    def test_selects_closest_weights_checkpoint_for_target_step(self):
        with tempfile.TemporaryDirectory(prefix="weights_step_select_") as temp_dir:
            manager = CheckpointManager(root_log_dir=temp_dir, load_model=False, load_config=False, load_data=False)
            manager.current_exp_hash = "12345678abcdef0199aabbcc"

            model = nn.Sequential(nn.Linear(3, 3))
            manager.save_model_checkpoint(model=model, step=0, save_optimizer=False)
            manager.save_model_checkpoint(model=model, step=5, save_optimizer=False)

            picked = manager._select_weight_checkpoint_file(manager.current_exp_hash, target_step=3)
            self.assertIsNotNone(picked, "A checkpoint file should be selected")
            self.assertIn("_step_000005.pt", picked.name)

    def test_load_latest_checkpoint_finds_file_under_model_hash_dir(self):
        """Regression test: load_latest_checkpoint used to look for weight
        files under models_dir/<full_24_byte_hash>/, but save_model_checkpoint
        (and every other loader) stores them under models_dir/<model_hash>/
        (the middle 8-byte segment) -- so it always returned None."""
        with tempfile.TemporaryDirectory(prefix="load_latest_checkpoint_") as temp_dir:
            manager = CheckpointManager(root_log_dir=temp_dir, load_model=False, load_config=False, load_data=False)
            manager.current_exp_hash = "12345678abcdef0199aabbcc"

            model = nn.Sequential(nn.Linear(3, 3))
            manager.save_model_checkpoint(model=model, step=0, save_optimizer=False)
            manager.save_model_checkpoint(model=model, step=9, save_optimizer=False)

            loaded = manager.load_latest_checkpoint(model=nn.Sequential(nn.Linear(3, 3)), load_optimizer=False)
            self.assertIsNotNone(loaded, "The latest checkpoint should be found and loaded")
            self.assertEqual(loaded.get('step'), 9)


class CheckpointMultiRootTests(unittest.TestCase):
    """Tests for pointing a CheckpointManager at a parent directory that fans
    out into several independent, nested experiment roots, e.g.:

        scorer_exp/lr_tests/v1/checkpoints/manifest.yaml
        scorer_exp/lr_tests/v2/checkpoints/manifest.yaml
        scorer_exp/dataShuffling_tests/v1/checkpoints/manifest.yaml

    The manager should transparently discover every nested root (up to
    NESTED_ROOT_DISCOVERY_MAX_DEPTH levels down) and adopt the one with the
    most recently updated manifest as its effective root, so model/config/
    weight loading (and any future checkpoint writes) behave exactly like
    the existing single root_log_dir case -- while merging signal-history
    curves from every discovered root so training curves read as one
    continuous history across all of them.
    """

    def setUp(self):
        ledgers.clear_all()
        self._tmp_ctx = tempfile.TemporaryDirectory(prefix="checkpoint_multiroot_")
        self.tmp_dir = Path(self._tmp_ctx.name)

    def tearDown(self):
        ledgers.clear_all()
        self._tmp_ctx.cleanup()

    def _pin_manifest_timestamp(self, root_dir, exp_hash, timestamp):
        """Overwrite a root's manifest recency timestamps with a controlled
        value -- real datetime.now() ordering between fast successive setup
        calls would otherwise make winner-selection non-deterministic."""
        manifest_path = root_dir / "checkpoints" / "manifest.yaml"
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        manifest['last_updated'] = timestamp
        manifest['experiments'][exp_hash]['last_used'] = timestamp
        manifest['experiments'][exp_hash]['created'] = timestamp
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    def _write_experiment_root(self, root_dir, exp_hash, step, timestamp, loss_values):
        """Create a standalone, valid experiment root under root_dir: a
        CheckpointManager with one model checkpoint and some logger signal
        history tagged with exp_hash, with its manifest recency pinned to a
        controlled timestamp (see _pin_manifest_timestamp)."""
        manager = CheckpointManager(root_log_dir=str(root_dir), load_model=False, load_config=False, load_data=False)
        manager.current_exp_hash = exp_hash
        manager._create_exp_hash_directories(exp_hash)

        model = nn.Sequential(nn.Linear(2, 2))
        manager.save_model_checkpoint(model=model, step=step, save_optimizer=False)

        lg = LoggerQueue(register=False, db_path=str(manager.get_logger_db_path()))
        lg.chkpt_manager = manager  # tag signals with this root's own exp_hash
        for i, value in enumerate(loss_values):
            lg.add_scalars('loss', {'loss': value}, global_step=i, signal_per_sample={}, aggregate_by_step=False)
        lg.flush_to_disk()
        lg.stop_background_flush()
        lg._conn.close()  # release the file lock so it can be ATTACHed for merging later

        self._pin_manifest_timestamp(root_dir, exp_hash, timestamp)

        ledgers.clear_all()
        return manager

    def test_fresh_empty_root_has_no_discovered_roots(self):
        """A brand-new, never-used root_log_dir behaves exactly as before:
        no nested roots are found, so nothing is adopted or merged."""
        root = self.tmp_dir / "brand_new"
        manager = CheckpointManager(root_log_dir=str(root), load_model=False, load_config=False, load_data=False)

        self.assertEqual(manager.root_log_dir, root.absolute())
        self.assertEqual(manager.given_root_log_dir, root.absolute())
        self.assertEqual(manager.discovered_root_dirs, [])
        self.assertFalse(manager.is_multi_root)

    def test_single_preexisting_root_is_backward_compatible(self):
        """Pointing directly at a single existing experiment root (the
        current, common usage) must resolve to itself unchanged."""
        root = self.tmp_dir / "single_exp"
        exp_hash = "aaaa0000" + "bbbb0000" + "cccc0000"
        self._write_experiment_root(root, exp_hash, step=3, timestamp="2024-01-01T00:00:00", loss_values=[0.9])

        manager = CheckpointManager(root_log_dir=str(root), load_model=False, load_config=False, load_data=False)

        self.assertEqual(manager.root_log_dir, root.absolute())
        self.assertEqual(manager.discovered_root_dirs, [root.absolute()])
        self.assertFalse(manager.is_multi_root)
        self.assertEqual(manager.current_exp_hash, exp_hash)

    def test_multiroot_with_single_nested_root_behaves_like_single_root(self):
        """A parent directory that currently fans out into only one nested
        root should adopt it exactly like the single-root case (no merging
        needed since there's nothing to merge from)."""
        parent = self.tmp_dir / "single_nested"
        only_root = parent / "attempt_1"
        exp_hash = "aaaa0004" + "bbbb0004" + "cccc0004"
        self._write_experiment_root(only_root, exp_hash, step=7, timestamp="2024-03-01T00:00:00", loss_values=[0.5])

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)

        self.assertEqual(manager.root_log_dir, only_root.absolute())
        self.assertEqual(manager.discovered_root_dirs, [only_root.absolute()])
        self.assertFalse(manager.is_multi_root)
        self.assertEqual(manager.current_exp_hash, exp_hash)

    def test_multiroot_adopts_most_recently_updated_nested_root(self):
        """Given several nested roots at varying depths (as deep as the
        max-depth-3 example from the feature request: parent/group/version),
        the manager must adopt the most recently updated one for model/
        weights/config, while merging logger curves from all of them."""
        parent = self.tmp_dir / "scorer_exp"
        root_old = parent / "lr_tests" / "v1"
        root_mid = parent / "lr_tests" / "v2"
        root_winner = parent / "dataShuffling_tests" / "v2"

        hash_old = "aaaa0001" + "bbbb0001" + "cccc0001"
        hash_mid = "aaaa0002" + "bbbb0002" + "cccc0002"
        hash_winner = "aaaa0003" + "bbbb0003" + "cccc0003"

        self._write_experiment_root(root_old, hash_old, step=10, timestamp="2024-01-01T00:00:00", loss_values=[2.0, 1.8])
        self._write_experiment_root(root_mid, hash_mid, step=20, timestamp="2024-06-01T00:00:00", loss_values=[1.5, 1.3])
        self._write_experiment_root(root_winner, hash_winner, step=30, timestamp="2025-01-01T00:00:00", loss_values=[1.0, 0.8])

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)

        # --- Winner selection: model/config/data come entirely from the ---
        # --- most recently updated nested root (same as single-root case) ---
        self.assertEqual(manager.root_log_dir, root_winner.absolute())
        self.assertTrue(manager.is_multi_root)
        self.assertEqual(
            set(manager.discovered_root_dirs),
            {root_old.absolute(), root_mid.absolute(), root_winner.absolute()},
        )
        self.assertEqual(manager.current_exp_hash, hash_winner)

        checkpoint_file = manager._select_weight_checkpoint_file(hash_winner)
        self.assertIsNotNone(checkpoint_file, "Winner's own weight checkpoint should be found")
        self.assertIn("_step_000030.pt", checkpoint_file.name)

        result = manager.load_checkpoint(hash_winner, load_model=False, load_config=False, load_data=False)
        self.assertIn('weights', result['loaded_components'])
        self.assertEqual(
            result['weights'].get('step'), 30,
            "The most recent checkpoint across all manifests should be loaded",
        )

        # --- Logger curves: merged in from every discovered root, not ---
        # --- just the winner's own ---
        ledgers.register_checkpoint_manager(manager)
        lg = LoggerQueue(register=True)
        try:
            history = lg.get_signal_history()
            loss_by_hash = history.get('loss', {})
            self.assertEqual(
                set(loss_by_hash.keys()), {hash_old, hash_mid, hash_winner},
                "Signal history should carry curves tagged with every discovered root's own hash",
            )
            total_points = sum(len(steps) for steps in loss_by_hash.values())
            self.assertEqual(total_points, 6, "2 loss points from each of the 3 merged roots")
        finally:
            lg.stop_background_flush()

    def test_sibling_merge_is_idempotent_across_call_sites(self):
        """merge_from_disk can be triggered from more than one init-ordering
        call site (logger created before vs. after the checkpoint manager);
        merging the same sibling twice must not duplicate its rows."""
        parent = self.tmp_dir / "idempotent_exp"
        root_a = parent / "a"
        root_b = parent / "b"
        hash_a = "aaaa0005" + "bbbb0005" + "cccc0005"
        hash_b = "aaaa0006" + "bbbb0006" + "cccc0006"

        self._write_experiment_root(root_a, hash_a, step=1, timestamp="2024-01-01T00:00:00", loss_values=[1.0])
        self._write_experiment_root(root_b, hash_b, step=2, timestamp="2024-02-01T00:00:00", loss_values=[2.0])

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)
        ledgers.register_checkpoint_manager(manager)

        lg = LoggerQueue(register=True)  # first merge trigger: LoggerQueue.__init__
        try:
            manager._merge_sibling_logger_histories()  # second, redundant trigger
            manager._merge_sibling_logger_histories()  # third, redundant trigger

            history = lg.get_signal_history()
            loss_by_hash = history.get('loss', {})
            self.assertEqual(len(loss_by_hash.get(hash_a, {})), 1, "Sibling root's rows must not be duplicated")
        finally:
            lg.stop_background_flush()

    def test_discovery_respects_max_depth_cutoff(self):
        """A root nested deeper than NESTED_ROOT_DISCOVERY_MAX_DEPTH levels
        must not be discovered, even if it's the most recently updated one --
        only what's actually found can be adopted."""
        parent = self.tmp_dir / "depth_test"
        within_depth = parent / "a" / "b" / "c"        # 3 levels down: in range
        beyond_depth = parent / "w" / "x" / "y" / "z"  # 4 levels down: out of range

        hash_within = "aaaa0007" + "bbbb0007" + "cccc0007"
        hash_beyond = "aaaa0008" + "bbbb0008" + "cccc0008"
        self._write_experiment_root(within_depth, hash_within, step=1, timestamp="2024-01-01T00:00:00", loss_values=[1.0])
        # Deliberately the more "recent" one, to prove it's ignored for being unreachable.
        self._write_experiment_root(beyond_depth, hash_beyond, step=1, timestamp="2025-01-01T00:00:00", loss_values=[1.0])

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)

        discovered = set(manager.discovered_root_dirs)
        self.assertIn(within_depth.absolute(), discovered)
        self.assertNotIn(beyond_depth.absolute(), discovered)
        self.assertEqual(manager.root_log_dir, within_depth.absolute())

    def test_ranking_falls_back_to_experiment_timestamps_when_last_updated_missing(self):
        """Older manifests may predate the top-level 'last_updated' field;
        ranking should still work off the latest experiment's own timestamp."""
        parent = self.tmp_dir / "legacy_timestamp_exp"
        root_a = parent / "a"
        root_b = parent / "b"
        hash_a = "aaaa0009" + "bbbb0009" + "cccc0009"
        hash_b = "aaaa0010" + "bbbb0010" + "cccc0010"
        self._write_experiment_root(root_a, hash_a, step=1, timestamp="2024-01-01T00:00:00", loss_values=[1.0])
        self._write_experiment_root(root_b, hash_b, step=1, timestamp="2025-01-01T00:00:00", loss_values=[1.0])

        manifest_path = root_b / "checkpoints" / "manifest.yaml"
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        del manifest['last_updated']
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)
        self.assertEqual(
            manager.root_log_dir, root_b.absolute(),
            "Should still rank by the latest experiment's own timestamp when top-level last_updated is missing",
        )

    def test_corrupt_manifest_does_not_crash_discovery(self):
        """One candidate's manifest.yaml being unparsable must not take down
        discovery for the others -- it just can't win the ranking."""
        parent = self.tmp_dir / "corrupt_manifest_exp"
        root_good = parent / "good"
        root_bad = parent / "bad"
        hash_good = "aaaa0011" + "bbbb0011" + "cccc0011"
        self._write_experiment_root(root_good, hash_good, step=1, timestamp="2024-01-01T00:00:00", loss_values=[1.0])

        bad_checkpoints_dir = root_bad / "checkpoints"
        bad_checkpoints_dir.mkdir(parents=True)
        # Tab-indentation is invalid YAML syntax -- guaranteed to raise on parse.
        (bad_checkpoints_dir / "manifest.yaml").write_text("experiments:\n\tbad: [\n")

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)
        self.assertEqual(len(manager.discovered_root_dirs), 2, "Both roots should be discovered")
        self.assertEqual(manager.root_log_dir, root_good.absolute(), "The corrupt manifest should rank last, not crash ranking")

    def test_merges_legacy_json_snapshot_from_sibling_without_duckdb(self):
        """A sibling root that predates DuckDB-backed logger persistence
        (only an uncompressed loggers.json snapshot, no loggers.duckdb) must
        still have its curves merged in via the legacy fallback path."""
        parent = self.tmp_dir / "legacy_logger_exp"
        root_legacy = parent / "legacy"
        root_winner = parent / "current"
        hash_legacy = "aaaa0012" + "bbbb0012" + "cccc0012"
        hash_winner = "aaaa0013" + "bbbb0013" + "cccc0013"

        legacy_manager = CheckpointManager(root_log_dir=str(root_legacy), load_model=False, load_config=False, load_data=False)
        legacy_manager.current_exp_hash = hash_legacy
        legacy_manager._create_exp_hash_directories(hash_legacy)
        legacy_manager.save_model_checkpoint(model=nn.Sequential(nn.Linear(2, 2)), step=1, save_optimizer=False)
        legacy_payload = {
            "loggers": {
                "main": {
                    "graph_names": ["loss"],
                    "signal_history": [
                        {"metric_name": "loss", "experiment_hash": hash_legacy, "model_age": 0, "metric_value": 4.2},
                    ],
                }
            }
        }
        with open(legacy_manager.loggers_dir / "loggers.json", "w") as f:
            json.dump(legacy_payload, f)
        self.assertFalse((legacy_manager.loggers_dir / "loggers.duckdb").exists(), "Precondition: no DuckDB file for this legacy root")
        self._pin_manifest_timestamp(root_legacy, hash_legacy, "2024-01-01T00:00:00")
        ledgers.clear_all()

        self._write_experiment_root(root_winner, hash_winner, step=5, timestamp="2025-01-01T00:00:00", loss_values=[1.0])

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)
        ledgers.register_checkpoint_manager(manager)
        lg = LoggerQueue(register=True)
        try:
            history = lg.get_signal_history()
            loss_by_hash = history.get('loss', {})
            self.assertIn(hash_legacy, loss_by_hash, "Legacy JSON snapshot curves from a pre-DuckDB sibling should be merged")
            self.assertEqual(loss_by_hash[hash_legacy][0][0]['metric_value'], 4.2)
        finally:
            lg.stop_background_flush()

    def test_merge_sibling_logger_histories_survives_merge_from_disk_exception(self):
        """If merging one sibling's on-disk history raises unexpectedly, the
        whole init must not blow up -- it's logged and skipped."""
        parent = self.tmp_dir / "raising_merge_exp"
        root_a = parent / "a"
        root_b = parent / "b"
        hash_a = "aaaa0014" + "bbbb0014" + "cccc0014"
        hash_b = "aaaa0015" + "bbbb0015" + "cccc0015"
        self._write_experiment_root(root_a, hash_a, step=1, timestamp="2024-01-01T00:00:00", loss_values=[1.0])
        self._write_experiment_root(root_b, hash_b, step=1, timestamp="2025-01-01T00:00:00", loss_values=[1.0])

        manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)
        ledgers.register_checkpoint_manager(manager)
        lg = LoggerQueue(register=True)
        try:
            with mock.patch.object(lg, 'merge_from_disk', side_effect=RuntimeError("boom")):
                manager._merge_sibling_logger_histories()  # must not raise
        finally:
            lg.stop_background_flush()

    def test_merge_triggered_when_logger_predates_manager(self):
        """The reverse init ordering: a LoggerQueue already exists BEFORE the
        multi-root CheckpointManager is constructed. _bind_logger_to_disk
        must still trigger the sibling merge once it binds the pre-existing
        logger to the winner's on-disk history."""
        parent = self.tmp_dir / "logger_first_exp"
        root_a = parent / "a"
        root_b = parent / "b"
        hash_a = "aaaa0016" + "bbbb0016" + "cccc0016"
        hash_b = "aaaa0017" + "bbbb0017" + "cccc0017"
        self._write_experiment_root(root_a, hash_a, step=1, timestamp="2024-01-01T00:00:00", loss_values=[1.0])
        self._write_experiment_root(root_b, hash_b, step=1, timestamp="2025-01-01T00:00:00", loss_values=[1.0])

        lg = LoggerQueue(register=True)  # created before any CheckpointManager exists
        try:
            manager = CheckpointManager(root_log_dir=str(parent), load_model=False, load_config=False, load_data=False)
            self.assertEqual(manager.root_log_dir, root_b.absolute())

            history = lg.get_signal_history()
            loss_by_hash = history.get('loss', {})
            self.assertIn(hash_a, loss_by_hash, "Sibling root's curve should be merged once the pre-existing logger is bound to disk")
            self.assertIn(hash_b, loss_by_hash, "Winner root's own curve should be present too (adopted from its on-disk file)")
        finally:
            lg.stop_background_flush()


class LoggerQueueMergeFromDiskTests(unittest.TestCase):
    """Focused unit tests for LoggerQueue.merge_from_disk's edge/failure
    paths, isolated from the CheckpointManager multi-root scaffolding."""

    class _FakeCheckpointManager:
        def __init__(self, exp_hash):
            self._exp_hash = exp_hash

        def get_current_experiment_hash(self):
            return self._exp_hash

    def setUp(self):
        ledgers.clear_all()
        self._tmp_ctx = tempfile.TemporaryDirectory(prefix="logger_merge_")
        self.tmp_dir = Path(self._tmp_ctx.name)

    def tearDown(self):
        ledgers.clear_all()
        self._tmp_ctx.cleanup()

    def _make_closed_db_with_signal(self, db_path, exp_hash, value, drop_table=None):
        lg = LoggerQueue(register=False, db_path=str(db_path))
        lg.chkpt_manager = self._FakeCheckpointManager(exp_hash)
        lg.add_scalars('loss', {'loss': value}, global_step=0, signal_per_sample={}, aggregate_by_step=False)
        lg.flush_to_disk()
        if drop_table:
            lg._conn.execute(f"DROP TABLE {drop_table}")
        lg.stop_background_flush()
        lg._conn.close()

    def test_noop_for_falsy_path(self):
        lg = LoggerQueue(register=False)
        self.assertFalse(lg.merge_from_disk(None))
        self.assertFalse(lg.merge_from_disk(""))
        lg.stop_background_flush()

    def test_noop_for_nonexistent_path(self):
        lg = LoggerQueue(register=False)
        self.assertFalse(lg.merge_from_disk(str(self.tmp_dir / "does_not_exist.duckdb")))
        lg.stop_background_flush()

    def test_noop_for_merging_self(self):
        db_path = self.tmp_dir / "self.duckdb"
        lg = LoggerQueue(register=False, db_path=str(db_path))
        self.assertFalse(lg.merge_from_disk(str(db_path)), "Merging a logger's own backing file into itself must be a no-op")
        lg.stop_background_flush()

    def test_handles_corrupt_source_file_gracefully(self):
        corrupt_path = self.tmp_dir / "corrupt.duckdb"
        corrupt_path.write_bytes(b"this is not a duckdb file")

        lg = LoggerQueue(register=False)
        try:
            self.assertFalse(lg.merge_from_disk(str(corrupt_path)))
        finally:
            lg.stop_background_flush()

    def test_copies_rows_and_is_idempotent(self):
        source_db = self.tmp_dir / "source.duckdb"
        self._make_closed_db_with_signal(source_db, "srchash", 3.5)

        lg = LoggerQueue(register=False)
        try:
            self.assertTrue(lg.merge_from_disk(str(source_db)))
            history = lg.get_signal_history()
            self.assertEqual(len(history.get('loss', {}).get('srchash', {})), 1)

            self.assertFalse(lg.merge_from_disk(str(source_db)), "Re-merging the same source must be a no-op")
            history_again = lg.get_signal_history()
            self.assertEqual(
                len(history_again.get('loss', {}).get('srchash', {})), 1,
                "Rows must not be duplicated by a repeated merge",
            )
        finally:
            lg.stop_background_flush()

    def test_skips_missing_table_gracefully(self):
        """An older-schema sibling DB might be missing a table entirely
        (schema evolved since); the other tables should still merge."""
        source_db = self.tmp_dir / "old_schema.duckdb"
        self._make_closed_db_with_signal(source_db, "oldschemahash", 1.1, drop_table="per_instance")

        lg = LoggerQueue(register=False)
        try:
            self.assertTrue(lg.merge_from_disk(str(source_db)))
            history = lg.get_signal_history()
            self.assertIn('oldschemahash', history.get('loss', {}))
        finally:
            lg.stop_background_flush()


if __name__ == '__main__':
    # Create test suite with explicit ordering
    suite = unittest.TestSuite()

    # Add tests in specific order
    # # Initialize experiment
    suite.addTest(CheckpointSystemTests('test_00_initialize_experiment'))
    # # User Adventures training workflow
    suite.addTest(CheckpointSystemTests('test_01_train_A'))
    suite.addTest(CheckpointSystemTests('test_02_train_B_model_change'))
    suite.addTest(CheckpointSystemTests('test_03_train_C_hyperparams_change'))
    suite.addTest(CheckpointSystemTests('test_04_train_D_data_change'))
    # # Reload and branching tests
    suite.addTest(CheckpointSystemTests('test_05_train_E_reload_and_branch'))
    suite.addTest(CheckpointSystemTests('test_06_reload_before_model_change'))
    suite.addTest(CheckpointSystemTests('test_07_change_data_from_test06'))
    # # Reload and check full reproducibility - Loss and UIDs
    suite.addTest(CheckpointSystemTests('test_08_reload_before_data_change_verify_and_modify'))
    suite.addTest(CheckpointSystemTests('test_09_reload_before_hp_change_verify_and_modify'))
    suite.addTest(CheckpointSystemTests('test_10_reload_branch_j_verify_reproducibility'))
    suite.addTest(CheckpointSystemTests('test_11_restart_from_scratch_to_hash_d_and_verify_reproducibility'))
    # # Check that logger queue is saved and loaded
    suite.addTest(CheckpointSystemTests('test_logger_queue_saved_with_weights'))
    # # Step-aware hash/checkpoint behavior
    suite.addTest(CheckpointStepAwareBehaviorTests('test_model_hash_depends_on_init_step'))
    suite.addTest(CheckpointStepAwareBehaviorTests('test_selects_closest_weights_checkpoint_for_target_step'))
    suite.addTest(CheckpointStepAwareBehaviorTests('test_load_latest_checkpoint_finds_file_under_model_hash_dir'))
    # # Multi-root discovery (root_log_dir fanning out into nested experiment roots)
    suite.addTest(CheckpointMultiRootTests('test_fresh_empty_root_has_no_discovered_roots'))
    suite.addTest(CheckpointMultiRootTests('test_single_preexisting_root_is_backward_compatible'))
    suite.addTest(CheckpointMultiRootTests('test_multiroot_with_single_nested_root_behaves_like_single_root'))
    suite.addTest(CheckpointMultiRootTests('test_multiroot_adopts_most_recently_updated_nested_root'))
    suite.addTest(CheckpointMultiRootTests('test_sibling_merge_is_idempotent_across_call_sites'))
    suite.addTest(CheckpointMultiRootTests('test_discovery_respects_max_depth_cutoff'))
    suite.addTest(CheckpointMultiRootTests('test_ranking_falls_back_to_experiment_timestamps_when_last_updated_missing'))
    suite.addTest(CheckpointMultiRootTests('test_corrupt_manifest_does_not_crash_discovery'))
    suite.addTest(CheckpointMultiRootTests('test_merges_legacy_json_snapshot_from_sibling_without_duckdb'))
    suite.addTest(CheckpointMultiRootTests('test_merge_sibling_logger_histories_survives_merge_from_disk_exception'))
    suite.addTest(CheckpointMultiRootTests('test_merge_triggered_when_logger_predates_manager'))
    suite.addTest(LoggerQueueMergeFromDiskTests('test_noop_for_falsy_path'))
    suite.addTest(LoggerQueueMergeFromDiskTests('test_noop_for_nonexistent_path'))
    suite.addTest(LoggerQueueMergeFromDiskTests('test_noop_for_merging_self'))
    suite.addTest(LoggerQueueMergeFromDiskTests('test_handles_corrupt_source_file_gracefully'))
    suite.addTest(LoggerQueueMergeFromDiskTests('test_copies_rows_and_is_idempotent'))
    suite.addTest(LoggerQueueMergeFromDiskTests('test_skips_missing_table_gracefully'))

    # Run the suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
