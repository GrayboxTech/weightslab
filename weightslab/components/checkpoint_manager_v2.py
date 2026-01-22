"""
Checkpoint Manager V2 - Structured Checkpoint Management


This module implements checkpoint management with separated component directories:

Directory Structure:
    root_log_dir/
        data/           # Data-related files (global)
        logs/           # Training logs (global)
        checkpoints/
            manifest.yaml   # Tracks all hashes with timestamps
            models/
                {hash}/     # 24-byte hash: HP_MODEL_DATA
                    {hash}_step_000100.pt
                    {hash}_architecture.pkl
            HP/
                {hash}/
                    {hash}_config.yaml
            data/
                {hash}/
                    {hash}_data_state.yaml

Hash format: HP(8) + MODEL(8) + DATA(8) = 24 bytes

Manifest tracks hash chronology for loading most recent experiments.
"""

import os
import json
import yaml
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import pandas as pd

import torch as th
import dill
import pickle

from weightslab.components.experiment_hash import ExperimentHashGenerator
from weightslab.backend.ledgers import (
    get_model,
    get_optimizer,
    get_dataloader,
    get_dataloaders,
)
from weightslab.backend import ledgers
from weightslab.data.sample_stats import SampleStatsEx


# Init logger
logger = logging.getLogger(__name__)


class CheckpointManagerV2:
    """Structured checkpoint manager with experiment hash-based organization.

    This manager creates a well-organized checkpoint structure where each
    unique experiment configuration gets its own directory identified by
    a deterministic hash.

    Attributes:
        root_log_dir (Path): Root directory for all experiment outputs
        checkpoints_dir (Path): Base checkpoints directory
        hash_generator (ExperimentHashGenerator): Hash generation utility
        current_exp_hash (str): Current experiment hash
        _step_counter (int): Global step counter for model checkpoints
    """

    def __init__(self, root_log_dir: str = 'root_experiment'):
        """Initialize the checkpoint manager.

        Args:
            root_log_dir: Root directory for experiment outputs
        """
        self.root_log_dir = Path(root_log_dir).absolute()
        self.root_log_dir.mkdir(parents=True, exist_ok=True)

        # Create main subdirectories
        self.data_dir = self.root_log_dir / "data"
        self.logs_dir = self.root_log_dir / "logs"
        self.checkpoints_dir = self.root_log_dir / "checkpoints"

        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Create checkpoint subdirectories for different components
        self.models_dir = self.checkpoints_dir / "models"
        self.hp_dir = self.checkpoints_dir / "HP"
        self.data_checkpoint_dir = self.checkpoints_dir / "data"

        self.models_dir.mkdir(exist_ok=True)
        self.hp_dir.mkdir(exist_ok=True)
        self.data_checkpoint_dir.mkdir(exist_ok=True)

        # Manifest file for tracking hash chronology
        self.manifest_file = self.checkpoints_dir / "manifest.yaml"

        # Hash management
        self.hash_generator = ExperimentHashGenerator()
        self.current_exp_hash: Optional[str] = None

        # Step tracking
        self._step_counter = 0

        # Pending changes tracking
        self._pending_model = None
        self._pending_config = None
        self._pending_data_state = None
        self._has_pending_changes = False
        self._pending_components = set()

        # Load existing state if available
        self._load_manager_state()

        logger.info(f"CheckpointManagerV2 initialized at {self.root_log_dir}")

    def __repr__(self) -> str:
        return (
            f"CheckpointManagerV2(\n"
            f"  root_log_dir={self.root_log_dir}\n"
            f"  current_exp_hash={self.current_exp_hash}\n"
            f"  step_counter={self._step_counter}\n"
            f")"
        )

    def _get_data_state_snapshot(self, dfm):
        """Return a combined data state from registered dataloaders, if present."""
        try:
            if isinstance(dfm, dict):
                return dfm

            collected_discarded = {}
            collected_tags = {}

            collected_discarded.update(dfm.get_df_view(SampleStatsEx.DENY_LISTED.value).to_dict())
            collected_tags.update(dfm.get_df_view(SampleStatsEx.TAGS.value).to_dict())

            if not collected_tags and not collected_discarded:
                return None

            return {
                'discarded': collected_discarded,
                'tags': collected_tags,
            }
        except Exception:
            return None

    def update_experiment_hash(
        self,
        model_snapshot: Optional[th.nn.Module] = None,
        hp_snapshot: Optional[Dict[str, Any]] = None,
        dfm_snapshot: Optional[Dict[str, Any]] = None,
        force: bool = False,
        dump_immediately: bool = False
    ) -> tuple[str, bool, Set[str]]:
        """Update experiment hash and track changes (pending or immediate).

        Changes can be:
        1. Pending: Tracked but not dumped until training resumes or manual dump
        2. Immediate: Dumped right away if dump_immediately=True

        Args:
            model_snapshot: PyTorch model
            hp_snapshot: Dictionary of hyperparameters
            dfm_snapshot: Dictionary with 'uids', 'discarded', 'tags'
            force: Force hash regeneration even if nothing changed
            dump_immediately: If True, dump changes immediately. If False, mark as pending.

        Returns:
            tuple: (exp_hash: str, is_new: bool, changed_components: Set[str])
        """

        # Get ledgered components
        hp_snapshot = ledgers.get_hyperparams()
        dfm_snapshot = ledgers.get_dataframe()
        model_snapshot = ledgers.get_model()

        # Process dataframe snapshot
        data_state = self._get_data_state_snapshot(dfm_snapshot)

        # Check what changed
        has_changed, changed_components = self.hash_generator.has_changed(
            model=model_snapshot,
            config=hp_snapshot,
            data_state=data_state
        )

        if not has_changed and not force:
            return self.current_exp_hash, False, set()

        # Generate new hash with all components
        new_hash = self.hash_generator.generate_hash(
            model=model_snapshot,
            config=hp_snapshot,
            data_state=data_state
        )

        is_new = (new_hash != self.current_exp_hash) or force

        if is_new:
            logger.info(f"New experiment hash: {new_hash} (previous: {self.current_exp_hash})")
            logger.info(f"Changed components: {changed_components}")

            # Update hash
            old_hash = self.current_exp_hash
            self.current_exp_hash = new_hash

            # Create checkpoint subdirectories for this hash
            self._create_exp_hash_directories(new_hash)

            if dump_immediately:
                # Dump changes immediately
                self._dump_changes(
                    model=model_snapshot,
                    config=hp_snapshot,
                    data_state=data_state,
                    changed_components=changed_components
                )
                self._has_pending_changes = False
                self._pending_components = set()
            else:
                # Mark as pending
                self._pending_model = model_snapshot
                self._pending_config = hp_snapshot
                self._pending_data_state = data_state
                self._has_pending_changes = True
                self._pending_components = changed_components
                logger.info(f"Changes pending (not dumped yet): {changed_components}")

            # Save manager state
            self._save_manager_state()

        return new_hash, is_new, changed_components

    def _create_exp_hash_directories(self, exp_hash: str):
        """Create directory structure for an experiment hash in separate component folders.

        Args:
            exp_hash: 24-byte experiment hash (HP_MODEL_DATA)
        """
        model_hash_dir = self.models_dir / exp_hash
        hp_hash_dir = self.hp_dir / exp_hash
        data_hash_dir = self.data_checkpoint_dir / exp_hash

        model_hash_dir.mkdir(exist_ok=True)
        hp_hash_dir.mkdir(exist_ok=True)
        data_hash_dir.mkdir(exist_ok=True)

        logger.debug(f"Created checkpoint directories for {exp_hash}")
        self._update_manifest(exp_hash)

    def _update_manifest(self, exp_hash: str):
        """Update manifest file with new or updated hash."""
        try:
            manifest = self._load_manifest()
            component_hashes = self.hash_generator.get_component_hashes()

            if exp_hash not in manifest['experiments']:
                manifest['experiments'][exp_hash] = {
                    'hp_hash': component_hashes.get('hp', exp_hash[0:8]),
                    'model_hash': component_hashes.get('model', exp_hash[8:16]),
                    'data_hash': component_hashes.get('data', exp_hash[16:24]),
                    'created': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat()
                }
            else:
                manifest['experiments'][exp_hash]['last_used'] = datetime.now().isoformat()

            manifest['latest_hash'] = exp_hash
            manifest['last_updated'] = datetime.now().isoformat()

            with open(self.manifest_file, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to update manifest: {e}")

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return yaml.safe_load(f) or {'experiments': {}, 'latest_hash': None}
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return {'experiments': {}, 'latest_hash': None}

    def get_latest_hash(self) -> Optional[str]:
        """Get the most recent experiment hash."""
        manifest = self._load_manifest()
        return manifest.get('latest_hash')

    def get_all_hashes(self, sort_by: str = 'created') -> List[Dict[str, Any]]:
        """Get all hashes sorted by timestamp."""
        manifest = self._load_manifest()
        experiments = manifest.get('experiments', {})
        hash_list = [{'hash': h, **info} for h, info in experiments.items()]
        if sort_by in ['created', 'last_used']:
            hash_list.sort(key=lambda x: x.get(sort_by, ''), reverse=True)
        return hash_list

    def get_hashes_by_component(self, hp_hash: Optional[str] = None,
                                model_hash: Optional[str] = None,
                                data_hash: Optional[str] = None) -> List[str]:
        """Find hashes matching component hash(es)."""
        manifest = self._load_manifest()
        experiments = manifest.get('experiments', {})
        matching = []
        for exp_hash, info in experiments.items():
            if hp_hash and info.get('hp_hash') != hp_hash:
                continue
            if model_hash and info.get('model_hash') != model_hash:
                continue
            if data_hash and info.get('data_hash') != data_hash:
                continue
            matching.append(exp_hash)
        return matching

    def dump_pending_changes(self, force: bool = False) -> bool:
        """Dump any pending changes to disk.

        This is called when:
        - Training resumes after model/config/data changes
        - Manual checkpoint is requested with force=True

        Args:
            force: Force dump even if no pending changes

        Returns:
            bool: True if changes were dumped, False otherwise
        """
        if not self._has_pending_changes and not force:
            logger.debug("No pending changes to dump")
            return False

        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Cannot dump pending changes.")
            return False

        logger.info(f"Dumping pending changes: {self._pending_components}")

        self._dump_changes(
            model=self._pending_model,
            config=self._pending_config,
            data_state=self._pending_data_state,
            changed_components=self._pending_components
        )

        # Clear pending state
        self._pending_model = None
        self._pending_config = None
        self._pending_data_state = None
        self._has_pending_changes = False
        self._pending_components = set()

        self._save_manager_state()
        return True

    def _dump_changes(
        self,
        model: Optional[th.nn.Module],
        config: Optional[Dict[str, Any]],
        data_state: Optional[Dict[str, Any]],
        changed_components: Set[str]
    ):
        """Internal method to dump changes to disk.

        Args:
            model: PyTorch model
            config: Hyperparameters config
            data_state: Data state dict
            changed_components: Set of changed components ('model', 'config', 'data')
        """
        if 'model' in changed_components and model is not None:
            logger.info("Dumping model architecture...")
            self.save_model_architecture(model)
            self.save_model_checkpoint(model)

        if ('hp' in changed_components or 'config' in changed_components) and config is not None:
            logger.info("Dumping hyperparameters config...")
            self.save_config(config)

        if 'data' in changed_components:
            logger.info("Dumping data snapshot...")
            self.save_data_snapshot()

    def has_pending_changes(self) -> tuple[bool, Set[str]]:
        """Check if there are pending changes.

        Returns:
            tuple: (has_pending: bool, pending_components: Set[str])
        """
        return self._has_pending_changes, self._pending_components.copy()

    def save_model_checkpoint(
        self,
        model: Optional[th.nn.Module] = None,
        model_name: Optional[str] = None,
        step: Optional[int] = None,
        save_optimizer: bool = True,
        optimizer_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        force_dump_pending: bool = False
    ) -> Optional[Path]:
        """Save model weights checkpoint.

        This saves only the model weights (state_dict) for fast checkpointing
        during training. The architecture is saved separately when the hash changes.

        Args:
            model: PyTorch model (or get from ledger if None)
            model_name: Name to get model from ledger
            step: Training step number (uses internal counter if None)
            save_optimizer: Whether to also save optimizer state
            optimizer_name: Name to get optimizer from ledger
            metadata: Additional metadata to save
            force_dump_pending: If True, dump any pending changes before saving checkpoint

        Returns:
            Path: Path to saved checkpoint file, or None if failed
        """
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        # Dump pending changes if requested
        if force_dump_pending and self._has_pending_changes:
            logger.info("Force dumping pending changes before checkpoint...")
            self.dump_pending_changes(force=True)

        # Get model from ledger if not provided
        if model is None:
            try:
                model = get_model(model_name)
                # Unwrap proxy if needed
                if hasattr(model, 'get') and callable(model.get):
                    model = model.get()
            except Exception as e:
                logger.error(f"Could not get model from ledger: {e}")
                return None

        if model is None:
            logger.error("No model available to checkpoint")
            return None

        # Determine step
        if step is None:
            step = self._step_counter
        self._step_counter = max(self._step_counter, step + 1)

        # Prepare checkpoint data
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'exp_hash': self.current_exp_hash,
        }

        # Add optimizer state if requested
        if save_optimizer:
            try:
                optimizer = get_optimizer(optimizer_name)
                if hasattr(optimizer, 'get') and callable(optimizer.get):
                    optimizer = optimizer.get()
                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            except Exception as e:
                logger.warning(f"Could not save optimizer state: {e}")

        # Add metadata
        if metadata:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        model_dir = self.models_dir / self.current_exp_hash
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = model_dir / f"{self.current_exp_hash}_step_{step:06d}.pt"

        try:
            th.save(checkpoint, checkpoint_file)
            logger.info(f"Saved model checkpoint: {checkpoint_file.name}")
            return checkpoint_file
        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}")
            return None

    def save_model_architecture(
        self,
        model: th.nn.Module,
        model_name: Optional[str] = None
    ) -> Optional[Path]:
        """Save full model architecture (structure + code).

        This is saved once per experiment hash when the architecture changes.
        Uses dill for serialization to handle custom modules.

        Args:
            model: PyTorch model
            model_name: Optional name for the model

        Returns:
            Path: Path to saved architecture file, or None if failed
        """
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        model_dir = self.models_dir / self.current_exp_hash
        model_dir.mkdir(parents=True, exist_ok=True)
        arch_file = model_dir / f"{self.current_exp_hash}_architecture.pkl"

        # Don't overwrite if already exists
        if arch_file.exists():
            logger.debug(f"Architecture already saved for {self.current_exp_hash}")
            return arch_file

        try:
            # Try dill first (better for custom classes)
            if dill is not None:
                with open(arch_file, 'wb') as f:
                    dill.dump(model, f)
            else:
                with open(arch_file, 'wb') as f:
                    pickle.dump(model, f)

            logger.info(f"Saved model architecture: {arch_file.name}")

            # Also save a text representation
            arch_txt = model_dir / f"{self.current_exp_hash}_architecture.txt"
            with open(arch_txt, 'w') as f:
                f.write(str(model))

            return arch_file
        except Exception as e:
            logger.error(f"Failed to save model architecture: {e}")
            return None

    def save_config(
        self,
        config: Dict[str, Any],
        config_name: str = "config"
    ) -> Optional[Path]:
        """Save hyperparameter configuration."""
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        hp_hash_dir = self.hp_dir / self.current_exp_hash
        config_file = hp_hash_dir / f"{self.current_exp_hash}_{config_name}.yaml"

        try:
            config_with_meta = {
                'hyperparameters': config,
                'exp_hash': self.current_exp_hash,
                'last_updated': datetime.now().isoformat()
            }

            with open(config_file, 'w') as f:
                yaml.dump(config_with_meta, f, default_flow_style=False)

            logger.info(f"Saved config: {config_file.name}")
            return config_file
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return None

    def save_data_snapshot(self) -> Optional[Path]:
        """Save lightweight JSON snapshot of data state (sample_id, tags, deny_listed).

        H5 files (data.h5 and arrays.h5) are saved in parent directory (shared).
        Only checkpoint-specific metadata is saved here as JSON.
        """
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        try:
            # Get dataframe manager
            dfm = ledgers.get_dataframe('sample_stats')
            if dfm is None:
                return None

            # Trigger H5 flush to parent directory (shared)
            dfm.flush_if_needed_nonblocking(force=True)

            # Extract only sample_id, tags, deny_listed for this checkpoint
            df = dfm.get_df_view()
            if df.empty:
                return None

            if 'sample_id' not in df.columns:
                df = df.reset_index()

            # Keep only checkpoint-specific columns
            snapshot_cols = ['sample_id', 'tags', 'deny_listed']
            available_cols = [col for col in snapshot_cols if col in df.columns]

            snapshot_df = df[available_cols]

            # Convert to JSON-serializable format
            snapshot_data = {
                'exp_hash': self.current_exp_hash,
                'timestamp': datetime.now().isoformat(),
                'data': snapshot_df.to_dict(orient='records')
            }

            # Save to hash-specific directory
            data_hash_dir = self.data_checkpoint_dir / self.current_exp_hash
            data_hash_dir.mkdir(parents=True, exist_ok=True)
            json_file = data_hash_dir / f"{self.current_exp_hash}_data_snapshot.json"

            with open(json_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2)

            logger.info(f"Saved data snapshot: {json_file.name} ({len(snapshot_df)} rows)")
            return json_file

        except Exception as e:
            logger.error(f"Failed to save data snapshot: {e}")
        return None

    def save_data_backup(
        self,
        data_h5_path: Path,
        backup_name: Optional[str] = None
    ) -> Optional[Path]:
        """Backup data h5 file for this experiment.

        This creates a copy of the data h5 file in the experiment's data directory.
        Should only be called for main data h5 files, not large array h5 files.

        Args:
            data_h5_path: Path to source h5 file
            backup_name: Optional name for backup (uses source name if None)

        Returns:
            Path: Path to backup file, or None if failed
        """
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        data_dir = self.checkpoints_dir / self.current_exp_hash / "data"

        if backup_name is None:
            backup_name = data_h5_path.name

        backup_file = data_dir / f"{self.current_exp_hash}_{backup_name}"

        try:
            shutil.copy2(data_h5_path, backup_file)
            logger.info(f"Backed up data: {backup_file.name}")
            return backup_file
        except Exception as e:
            logger.error(f"Failed to backup data: {e}")
            return None

    def load_latest_checkpoint(
        self,
        model: Optional[th.nn.Module] = None,
        model_name: Optional[str] = None,
        load_optimizer: bool = True,
        optimizer_name: Optional[str] = None,
        exp_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the latest model checkpoint.

        Args:
            model: PyTorch model to load weights into
            model_name: Name to get model from ledger
            load_optimizer: Whether to load optimizer state
            optimizer_name: Name to get optimizer from ledger
            exp_hash: Specific experiment hash (uses current if None)

        Returns:
            dict: Checkpoint data including step, metadata, etc., or None if failed
        """
        target_hash = exp_hash or self.current_exp_hash

        if target_hash is None:
            logger.warning("No experiment hash specified")
            return None

        # Find latest checkpoint
        model_dir = self.models_dir / target_hash

        if not model_dir.exists():
            logger.warning(f"No checkpoints found for hash {target_hash}")
            return None

        checkpoint_files = sorted(model_dir.glob(f"{target_hash}_step_*.pt"))

        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {model_dir}")
            return None

        latest_checkpoint = checkpoint_files[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint.name}")

        try:
            checkpoint = th.load(latest_checkpoint, weights_only=False)

            # Load model state
            if model is None:
                try:
                    model = get_model(model_name)
                    if hasattr(model, 'get') and callable(model.get):
                        model = model.get()
                except Exception as e:
                    logger.error(f"Could not get model from ledger: {e}")
                    return checkpoint

            if model is not None and 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("Loaded model state")
                except Exception as e:
                    logger.error(f"Failed to load model state: {e}")

            # Load optimizer state
            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer = get_optimizer(optimizer_name)
                    if hasattr(optimizer, 'get') and callable(optimizer.get):
                        optimizer = optimizer.get()
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        logger.info("Loaded optimizer state")
                except Exception as e:
                    logger.warning(f"Could not load optimizer state: {e}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    def load_config(self, exp_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load hyperparameter configuration.

        Args:
            exp_hash: Specific experiment hash (uses current if None)

        Returns:
            dict: Configuration dictionary, or None if not found
        """
        target_hash = exp_hash or self.current_exp_hash

        if target_hash is None:
            logger.warning("No experiment hash specified")
            return None

        hp_dir = self.checkpoints_dir / target_hash / "hp"
        config_file = hp_dir / f"{target_hash}_config.yaml"

        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return None

        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            return config_data.get('hyperparameters', config_data)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return None

    def list_experiment_hashes(self) -> List[str]:
        """List all experiment hashes with checkpoints.

        Returns:
            list: List of experiment hash strings
        """
        if not self.checkpoints_dir.exists():
            return []

        hashes = [d.name for d in self.checkpoints_dir.iterdir() if d.is_dir()]
        return sorted(hashes)

    def get_checkpoint_info(self, exp_hash: Optional[str] = None) -> Dict[str, Any]:
        """Get information about checkpoints for an experiment.

        Args:
            exp_hash: Specific experiment hash (uses current if None)

        Returns:
            dict: Information about checkpoints, configs, data backups
        """
        target_hash = exp_hash or self.current_exp_hash

        if target_hash is None:
            return {}

        exp_dir = self.checkpoints_dir / target_hash

        if not exp_dir.exists():
            return {}

        info = {
            'exp_hash': target_hash,
            'model_checkpoints': [],
            'architecture_saved': False,
            'configs': [],
            'data_backups': []
        }

        model_dir = exp_dir / "model"
        if model_dir.exists():
            info['model_checkpoints'] = [
                f.name for f in sorted(model_dir.glob(f"{target_hash}_step_*.pt"))
            ]

        # Configs
        hp_dir = exp_dir / "hp"
        if hp_dir.exists():
            info['configs'] = [f.name for f in hp_dir.glob("*.yaml")]

        # Data backups
        data_dir = exp_dir / "data"
        if data_dir.exists():
            info['data_backups'] = [f.name for f in data_dir.glob("*.h5")]

        return info

    def _save_manager_state(self):
        """Save manager state (current hash, step counter, etc.)"""
        state_file = self.root_log_dir / ".checkpoint_manager_state.json"

        state = {
            'current_exp_hash': self.current_exp_hash,
            'step_counter': self._step_counter,
            'last_updated': datetime.now().isoformat()
        }

        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save manager state: {e}")

    def _load_manager_state(self):
        """Load manager state if available"""
        state_file = self.root_log_dir / ".checkpoint_manager_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            self.current_exp_hash = state.get('current_exp_hash')
            self._step_counter = state.get('step_counter', 0)

            logger.info(f"Loaded manager state: hash={self.current_exp_hash}, step={self._step_counter}")
        except Exception as e:
            logger.warning(f"Failed to load manager state: {e}")

    def load_checkpoint(self, exp_hash: str,
                       load_model: bool = True,
                       load_weights: bool = True,
                       load_config: bool = True,
                       load_data: bool = True) -> Dict[str, Any]:
        """Load a complete checkpoint state by experiment hash.

        This method intelligently loads only the components that differ from
        the current state by comparing component hashes.

        Args:
            exp_hash: The 24-byte experiment hash to load (HP_MODEL_DATA)
            load_model: Whether to load model architecture if different
            load_weights: Whether to load model weights
            load_config: Whether to load hyperparameters if different
            load_data: Whether to load data state if different

        Returns:
            dict: Dictionary with keys:
                - 'model': Loaded model (if changed and load_model=True)
                - 'weights': Checkpoint dict with weights and metadata
                - 'config': Loaded config (if changed and load_config=True)
                - 'data_state': Loaded data state (if changed and load_data=True)
                - 'loaded_components': Set of components that were loaded
                - 'exp_hash': The experiment hash that was loaded
        """
        result = {
            'model': None,
            'weights': None,
            'config': None,
            'data_state': None,
            'loaded_components': set(),
            'exp_hash': exp_hash
        }

        # Load manifest to get component hashes
        manifest = self._load_manifest()
        if exp_hash not in manifest.get('experiments', {}):
            logger.error(f"Experiment hash {exp_hash} not found in manifest")
            return result

        exp_info = manifest['experiments'][exp_hash]
        target_hp_hash = exp_info.get('hp_hash')
        target_model_hash = exp_info.get('model_hash')
        target_data_hash = exp_info.get('data_hash')

        # Get current component hashes
        current_hashes = self.hash_generator.get_component_hashes()
        current_hp_hash = current_hashes.get('hp', '')
        current_model_hash = current_hashes.get('model', '')
        current_data_hash = current_hashes.get('data', '')

        logger.info(f"Loading checkpoint {exp_hash[:16]}...")
        logger.info(f"  Target: HP={target_hp_hash} MODEL={target_model_hash} DATA={target_data_hash}")
        logger.info(f"  Current: HP={current_hp_hash} MODEL={current_model_hash} DATA={current_data_hash}")

        # Load model architecture if different
        if load_model and target_model_hash != current_model_hash:
            model_dir = self.models_dir / exp_hash
            arch_file = model_dir / f"{exp_hash}_architecture.pkl"

            if arch_file.exists():
                try:
                    with open(arch_file, 'rb') as f:
                        result['model'] = dill.load(f)
                    result['loaded_components'].add('model')
                    logger.info(f"  [OK] Loaded model architecture (hash changed)")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load model architecture: {e}")
            else:
                logger.warning(f"  [WARNING] Model architecture file not found: {arch_file}")
        else:
            logger.info(f"  [-] Model architecture unchanged, using current model")

        # Load model weights (always if requested)
        if load_weights:
            model_dir = self.models_dir / exp_hash
            weight_files = sorted(model_dir.glob(f"{exp_hash}_step_*.pt"))

            if weight_files:
                latest_weights = weight_files[-1]
                try:
                    result['weights'] = th.load(latest_weights, weights_only=False)
                    result['loaded_components'].add('weights')
                    step = result['weights'].get('step', -1)
                    logger.info(f"  [OK] Loaded weights from step {step}")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load weights: {e}")
            else:
                logger.warning(f"  [WARNING] No weight files found for {exp_hash}")

        # Load config if different
        if load_config and target_hp_hash != current_hp_hash:
            hp_dir = self.hp_dir / exp_hash
            config_file = hp_dir / f"{exp_hash}_config.yaml"

            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    result['config'] = config_data.get('hyperparameters', config_data)
                    result['loaded_components'].add('config')
                    logger.info(f"  [OK] Loaded config (hash changed)")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load config: {e}")
            else:
                logger.warning(f"  [WARNING] Config file not found: {config_file}")
        else:
            logger.info(f"  [-] Config unchanged, using current config")

        # Load data snapshot if different
        if load_data and target_data_hash != current_data_hash:
            data_dir = self.data_checkpoint_dir / exp_hash
            json_file = data_dir / f"{exp_hash}_data_snapshot.json"

            if json_file.exists():
                try:
                    # Load JSON snapshot
                    with open(json_file, 'r') as f:
                        snapshot_data = json.load(f)

                    # Convert to DataFrame
                    snapshot_df = pd.DataFrame(snapshot_data.get('data', []))

                    if not snapshot_df.empty:
                        result['data_state'] = {'snapshot': snapshot_df}
                        result['loaded_components'].add('data')
                        logger.info(f"  [OK] Loaded data snapshot ({len(snapshot_df)} rows)")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load data snapshot: {e}")
            else:
                logger.warning(f"  [WARNING] Data snapshot file not found: {json_file}")
        else:
            logger.info(f"  [-] Data state unchanged, using current data")

        logger.info(f"Loaded components: {result['loaded_components']}")
        return result

    def load_state(self, exp_hash: str) -> bool:
        """Load and apply a complete checkpoint state by experiment hash.

        This method loads all components and updates the system state in-place:
        - Updates model in ledger (architecture + weights)
        - Updates config in ledger
        - Updates dataframe manager with loaded data
        - Updates current experiment hash

        Args:
            exp_hash: The 24-byte experiment hash to load and apply

        Returns:
            bool: True if state was successfully loaded and applied
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading and applying state: {exp_hash[:16]}...")
        logger.info(f"{'='*60}")

        # Load checkpoint data
        checkpoint_data = self.load_checkpoint(
            exp_hash=exp_hash,
            load_model=True,
            load_weights=True,
            load_config=True,
            load_data=True
        )

        if not checkpoint_data['loaded_components']:
            logger.warning("No components were loaded")
            return False

        success = True

        # Apply model (architecture + weights)
        if 'model' in checkpoint_data['loaded_components']:
            try:
                model = checkpoint_data['model']
                weights = checkpoint_data.get('weights')

                # Apply weights if available
                if weights and 'model_state_dict' in weights:
                    model.load_state_dict(weights['model_state_dict'], strict=False)
                    step = weights.get('step', -1)
                    logger.info(f"[OK] Applied model architecture + weights (step {step})")
                else:
                    logger.info(f"[OK] Applied model architecture (no weights)")

                # Register in ledger
                ledgers.register_model(ledgers.resolve_hp_name(), model)
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply model: {e}")
                success = False
        elif 'weights' in checkpoint_data['loaded_components']:
            # Only weights changed, apply to existing model
            try:
                model = ledgers.get_model()
                weights = checkpoint_data['weights']
                if model and weights and 'model_state_dict' in weights:
                    model.load_state_dict(weights['model_state_dict'], strict=False)
                    step = weights.get('step', -1)
                    logger.info(f"[OK] Applied weights to existing model (step {step})")
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply weights: {e}")
                success = False

        # Apply config
        if 'config' in checkpoint_data['loaded_components']:
            try:
                config = checkpoint_data['config']
                ledgers.register_hyperparams(ledgers.resolve_hp_name(), config)
                logger.info(f"[OK] Applied hyperparameters config")
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply config: {e}")
                success = False

# Apply data (merge snapshot columns into current dataframe)
        if 'data' in checkpoint_data['loaded_components']:
            try:
                data_state = checkpoint_data.get('data_state', {})
                snapshot_df = data_state.get('snapshot')

                if snapshot_df is not None and not snapshot_df.empty:
                    dfm = ledgers.get_dataframe('sample_stats')
                    if dfm is not None:
                        # Set index if needed
                        if 'sample_id' in snapshot_df.columns:
                            snapshot_df = snapshot_df.set_index('sample_id')

                        # Merge only the checkpoint-specific columns (tags, deny_listed)
                        # This updates existing rows without replacing all data
                        dfm.upsert_df(snapshot_df, force_flush=True)
                        logger.info(f"[OK] Applied data snapshot ({len(snapshot_df)} rows)")
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply data: {e}")
                success = False

        # Update current experiment hash
        if success:
            self.current_exp_hash = exp_hash
            self._save_manager_state()
            logger.info(f"\n[OK] Successfully loaded and applied state: {exp_hash[:16]}")
        else:
            logger.warning(f"\n[WARNING] State loaded with errors")

        logger.info(f"{'='*60}\n")
        return success
