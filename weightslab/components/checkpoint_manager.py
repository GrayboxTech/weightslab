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
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import pandas as pd

import torch as th
import dill
import pickle
import zstandard as zstd

from weightslab.components.global_monitoring import guard_training_context, guard_testing_context
from weightslab.components.experiment_hash import ExperimentHashGenerator
from weightslab.backend.ledgers import (
    get_model,
    get_optimizer,
    get_dataloader,
    get_dataloaders,
)
from weightslab.backend import ledgers
from weightslab.utils.logger import LoggerQueue
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.utils.tools import capture_rng_state, restore_rng_state
from weightslab.components.global_monitoring import pause_controller as pause_ctrl

# Init logger
logger = logging.getLogger(__name__)


class CheckpointManager:
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

    LOGGER_SNAPSHOT_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
    LOGGER_SNAPSHOT_CHUNK_PREFIX = "loggers.part"
    LOGGER_SNAPSHOT_CHUNK_SUFFIX = ".json.zst"
    LOGGER_SNAPSHOT_MANIFEST_FILE = "loggers.manifest.json"

    def __init__(self, root_log_dir: str = 'root_experiment', load_model: bool = True, load_config: bool = True, load_data: bool = True):
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
        self.loggers_dir = self.checkpoints_dir / "loggers"

        self.models_dir.mkdir(exist_ok=True)
        self.hp_dir.mkdir(exist_ok=True)
        self.data_checkpoint_dir.mkdir(exist_ok=True)
        self.loggers_dir.mkdir(exist_ok=True)

        # Manifest file for tracking hash chronology
        self.manifest_file = self.checkpoints_dir / "manifest.yaml"

        # Hash management
        self.hash_generator = ExperimentHashGenerator()
        self.current_exp_hash: Optional[str] = None
        self.previous_exp_hash: Optional[str] = None
        self.hash_by_module: list = [None, None, None]  # HP, MODEL, DATA

        # Step tracking
        self._step_counter = None
        self._model_init_step = 0

        # First time only
        self.firsttime = True

        # Pending changes tracking
        self._pending_model = None
        self._pending_config = None
        self._pending_data_state = None
        self._has_pending_changes = False
        self._pending_components = set()

        # Load existing state if available
        self._load_manager_state()

        # Load any existing logger snapshots for visibility when starting
        self._load_all_logger_snapshots()

        # Automatically resume latest state when an existing root_log_dir is provided
        self._bootstrap_latest_state(load_model=load_model, load_config=load_config, load_data=load_data)

        logger.info(f"CheckpointManager initialized at {self.root_log_dir}")

    def __repr__(self) -> str:
        return (
            f"CheckpointManager(\n"
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

            # Collect discarded series
            collected_discarded.update(
                dfm.get_df_view(
                    SampleStatsEx.DISCARDED.value
                ).to_dict()
            )
            # Collect tag series
            df_tag_columns = [col for col in dfm.get_df_view().columns if col.startswith(f"{SampleStatsEx.TAG.value}:")]
            for col in df_tag_columns:
                collected_tags.update(
                    {
                        col: dfm.get_df_view(
                            col
                        ).to_dict()
                    }
                )

            # if nothing found
            if not collected_tags and not collected_discarded:
                return None

            return {
                SampleStatsEx.DISCARDED.value: collected_discarded,
                SampleStatsEx.TAG.value: collected_tags,
            }
        except Exception:
            return None

    def _get_logger_snapshot_dir(self) -> Path:
        return self.loggers_dir

    def _get_logger_snapshot_path(self) -> Path:
        return self._get_logger_snapshot_dir() / "loggers.json"

    def _get_logger_snapshot_manifest_path(self) -> Path:
        return self._get_logger_snapshot_dir() / self.LOGGER_SNAPSHOT_MANIFEST_FILE

    def _get_logger_snapshot_chunk_path(self, chunk_index: int) -> Path:
        fname = f"{self.LOGGER_SNAPSHOT_CHUNK_PREFIX}{chunk_index:04d}{self.LOGGER_SNAPSHOT_CHUNK_SUFFIX}"
        return self._get_logger_snapshot_dir() / fname

    def _list_logger_snapshot_chunks(self) -> List[Path]:
        snapshot_dir = self._get_logger_snapshot_dir()
        if not snapshot_dir.exists():
            return []
        return sorted(snapshot_dir.glob(f"{self.LOGGER_SNAPSHOT_CHUNK_PREFIX}*{self.LOGGER_SNAPSHOT_CHUNK_SUFFIX}"))

    def _load_logger_snapshot_payload(self) -> Dict[str, Any]:
        snapshot_dir = self._get_logger_snapshot_dir()
        manifest_path = self._get_logger_snapshot_manifest_path()

        chunk_paths: List[Path] = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f) or {}
                chunk_names = manifest.get("chunks", [])
                chunk_paths = [snapshot_dir / name for name in chunk_names if name]
            except Exception as e:
                logger.warning(f"Failed to read logger snapshot manifest for {manifest_path}: {e}")

        if not chunk_paths:
            chunk_paths = self._list_logger_snapshot_chunks()

        if chunk_paths:
            snapshot = {"timestamp": datetime.now().isoformat(), "loggers": {}}
            dctx = zstd.ZstdDecompressor()
            for path in chunk_paths:
                if not path.exists():
                    continue
                try:
                    with open(path, "rb") as f:
                        raw_payload = dctx.decompress(f.read())
                    lines = raw_payload.decode("utf-8").splitlines()
                    for line in lines:
                        if not line:
                            continue
                        record = json.loads(line)
                        lname = record.get("logger_name")
                        payload = record.get("payload")
                        if lname:
                            snapshot["loggers"][lname] = payload or {}
                except Exception as e:
                    logger.warning(f"Failed to load logger snapshot chunk {path}: {e}")
            if snapshot["loggers"]:
                return snapshot

        # Legacy layout fallback (new per-exp first, then old global file)
        legacy_candidates = [
            self._get_logger_snapshot_path(),
            self.loggers_dir / "loggers.json",
        ]
        for legacy_path in legacy_candidates:
            if not legacy_path.exists():
                continue
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    payload = json.load(f) or {}
                if payload.get("loggers"):
                    return payload
            except Exception as e:
                logger.warning(f"Failed to read legacy logger snapshot at {legacy_path}: {e}")

        return {}

    def _create_exp_hash_directories(
            self,
            exp_hash: str,
            create_model_dir: bool = True,
            create_hp_dir: bool = True,
            create_data_dir: bool = True
    ):
        """Create directory structure for an experiment hash in separate component folders.

        Args:
            exp_hash: 24-byte experiment hash (HP_MODEL_DATA)
        """
        model_hash_dir = self.models_dir / exp_hash[:8]
        hp_hash_dir = self.hp_dir / exp_hash[8:16]
        data_hash_dir = self.data_checkpoint_dir / exp_hash[16:24]

        if create_model_dir:
            model_hash_dir.mkdir(exist_ok=True)
        if create_hp_dir:
            hp_hash_dir.mkdir(exist_ok=True)
        if create_data_dir:
            data_hash_dir.mkdir(exist_ok=True)

        logger.debug(f"Created checkpoint directories for {exp_hash}: hp_dir={hp_hash_dir}, model_dir={model_hash_dir}, data_dir={data_hash_dir}")
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
                    'last_used': datetime.now().isoformat(),
                    'latest_weight_checkpoint': None,
                    'latest_weight_step': None
                }
            else:
                manifest['experiments'][exp_hash]['last_used'] = datetime.now().isoformat()

            manifest['latest_hash'] = exp_hash
            manifest['last_updated'] = datetime.now().isoformat()

            # Ensure parent directory exists
            self.manifest_file.parent.mkdir(parents=True, exist_ok=True)

            # Write manifest file (overwrites if exists)
            with open(self.manifest_file, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.warning(f"Failed to update manifest: {e}")

    def _update_manifest_weight_checkpoint(self, exp_hash: str, checkpoint_filename: str, step: int):
        """Update manifest with latest weight checkpoint for given experiment hash."""
        try:
            manifest = self._load_manifest()
            if exp_hash in manifest['experiments']:
                manifest['experiments'][exp_hash]['latest_weight_checkpoint'] = checkpoint_filename
                manifest['experiments'][exp_hash]['latest_weight_step'] = step
                manifest['experiments'][exp_hash]['last_used'] = datetime.now().isoformat()

                # Write updated manifest
                with open(self.manifest_file, 'w') as f:
                    yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
                logger.debug(f"Updated manifest with weight checkpoint: {checkpoint_filename} (step {step})")
        except Exception as e:
            logger.warning(f"Failed to update manifest weight checkpoint: {e}")

    def _bootstrap_latest_state(self, load_model: bool = True, load_config: bool = True, load_data: bool = True):
        """If a current hash is known (or manifest has one), load and apply it.

        This enables auto-resume when instantiating the manager on an existing
        root_log_dir without requiring an explicit load_state call by the user.
        """
        target = self.current_exp_hash or self.get_latest_hash()
        if not target:
            return
        try:
            self.load_state(target, load_model=load_model, load_config=load_config, load_data=load_data)
        except Exception as e:
            logger.warning(f"Auto-resume failed for {target}: {e}")

    def _extract_step_from_checkpoint_name(self, filename: str) -> Optional[int]:
        match = re.search(r"_step_(\d+)\.pt$", filename)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _select_weight_checkpoint_file(self, exp_hash: str, target_step: Optional[int] = None) -> Optional[Path]:
        """Select weight checkpoint file for an experiment hash.

        - If target_step is None: returns latest checkpoint.
        - If target_step is provided: returns closest step; tie breaks toward higher step.
        """
        model_dir = self.models_dir / exp_hash[8:-8]
        if not model_dir.exists():
            return None

        weight_files = sorted(model_dir.glob(f"{exp_hash}_step_*.pt"))
        if not weight_files:
            weight_files = sorted(model_dir.glob(f"{exp_hash[8:-8]}_step_*.pt"))
        if not weight_files:
            return None

        if target_step is None:
            return weight_files[-1]

        target = int(target_step)
        parsed: List[tuple[Path, int]] = []
        for path in weight_files:
            step = self._extract_step_from_checkpoint_name(path.name)
            if step is not None:
                parsed.append((path, step))

        if not parsed:
            return weight_files[-1]

        parsed.sort(key=lambda item: item[1])
        best_path, _ = min(parsed, key=lambda item: (abs(item[1] - target), -item[1]))
        return best_path

    def get_current_experiment_hash(self) -> Optional[str]:
        """Get the current experiment hash. Computes it on-demand if not yet set."""
        if self.current_exp_hash is None:
            # Compute hash on first access to ensure logger has valid hash from step 0
            model_snapshot = self.get_model_snapshot()
            hp_snapshot = self.get_HP_snapshot()
            data_snapshot = self.get_dataframe_snapshot()

            if model_snapshot is not None or hp_snapshot or data_snapshot:
                self.current_exp_hash = self.hash_generator.generate_hash(
                    model=model_snapshot,
                    config=hp_snapshot,
                    data_state=data_snapshot,
                    model_init_step=self._model_init_step,
                )
                logger.info(f"Initial experiment hash computed on-demand: {self.current_exp_hash}")

        return self.current_exp_hash

    def get_HP_snapshot(self) -> Dict[str, Any]:
        """Get current hyperparameters snapshot from ledger."""
        try:
            hp = ledgers.get_hyperparams()
            if hp is None:
                return {}
            if isinstance(hp, ledgers.Proxy) and hasattr(hp, 'get') and callable(hp.get):
                hp = hp.get()
            if isinstance(hp, dict):
                return hp
            elif hasattr(hp, '__dict__'):
                return vars(hp)
            else:
                return {}
        except Exception:
            return {}

    def get_model_snapshot(self) -> Optional[th.nn.Module]:
        """Get current model snapshot from ledger."""
        try:
            model = ledgers.get_model()
            if model is None:
                return None
            if isinstance(model, ledgers.Proxy) and hasattr(model, 'get') and callable(model.get):
                model = model.get()
            if isinstance(model, th.nn.Module):
                return model
            else:
                return None
        except Exception:
            return None

    def get_dataframe_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get current dataframe snapshot from registered dataloaders."""
        try:
            dfm = ledgers.get_dataframe()
            if isinstance(dfm, ledgers.Proxy) and hasattr(dfm, 'get') and callable(dfm.get):
                dfm = dfm.get()
            if dfm is None:
                return None
            return self._get_data_state_snapshot(dfm)
        except Exception:
            return None

    def get_hp_hash(self) -> Optional[str]:
        """Get hash of hyperparameters snapshot."""
        if ledgers.get_hyperparams() == None:
            return "None"
        else:
            return self.hash_generator.get_component_hashes().get('hp')

    def get_model_hash(self) -> Optional[str]:
        """Get hash of model snapshot."""
        if ledgers.get_model() == None:
            return "None"
        else:
            return self.hash_generator.get_component_hashes().get('model')

    def get_data_hash(self) -> Optional[str]:
        """Get hash of dataframe snapshot."""
        if ledgers.get_dataframe() == None:
            return "None"
        else:
            return self.hash_generator.get_component_hashes().get('data')

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

    def update_experiment_hash(
        self,
        model_snapshot: Optional[th.nn.Module] = None,
        hp_snapshot: Optional[Dict[str, Any]] = None,
        dfm_snapshot: Optional[Dict[str, Any]] = None,
        force: bool = False,
        firsttime: bool = False,
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
        # Init first time saving the init state when first resumes
        if firsttime and self.firsttime and not force:
            logger.info("First time initialization; skipping hash update.")
            self.firsttime = False
            force = True
            dump_immediately = True

        # Get ledgered components
        hp_snapshot = self.get_HP_snapshot() if hp_snapshot is None else hp_snapshot
        data_snapshot = self.get_dataframe_snapshot() if dfm_snapshot is None else dfm_snapshot
        model_snapshot = self.get_model_snapshot() if model_snapshot is None else model_snapshot

        # Check what changed
        has_changed, changed_components = self.hash_generator.has_changed(
            model=model_snapshot,
            config=hp_snapshot,
            data_state=data_snapshot,
            model_init_step=self._model_init_step,
            force=force
        )

        if not has_changed and not force:
            return self.current_exp_hash, False, set()

        # Generate new hash with all components
        new_hash = self.hash_generator.generate_hash(
            model=model_snapshot,
            config=hp_snapshot,
            data_state=data_snapshot,
            model_init_step=self._model_init_step,
        )

        is_new = (new_hash != self.current_exp_hash) or (force or dump_immediately)

        if is_new:
            if self.current_exp_hash is not None:
                logger.info(f"New experiment hash: {new_hash[:8]}-{new_hash[8:-8]}-{new_hash[-8:]} (previous: {self.current_exp_hash[:8]}-{self.current_exp_hash[8:-8]}-{self.current_exp_hash[-8:]})")
            else:
                logger.info(f"Initial experiment hash set: {new_hash[:8]}-{new_hash[8:-8]}-{new_hash[-8:]}")
            logger.info(f"Changed components: {changed_components}")

            # Update hash
            old_hash = self.current_exp_hash
            self.current_exp_hash = new_hash
            self.previous_exp_hash = old_hash
            self.hash_by_module[0] = self.hash_generator.get_component_hashes().get('hp', None)
            self.hash_by_module[1] = self.hash_generator.get_component_hashes().get('model', None)
            self.hash_by_module[2] = self.hash_generator.get_component_hashes().get('data', None)

            if dump_immediately:
                # Dump changes immediately
                self._save_changes(
                    model=model_snapshot,
                    config=hp_snapshot,
                    changed_components=changed_components
                )
                self._has_pending_changes = False
                self._pending_components = set()

                # Sync RNG and data state from just-dumped checkpoint
                try:
                    # Restore RNG state
                    rng_state = capture_rng_state()
                    restore_rng_state(rng_state)

                    # Reset dataloader iterators to sync with new state
                    for loader_name in get_dataloaders():
                        loader = get_dataloader(loader_name)
                        if hasattr(loader, 'reset_iterator') and callable(loader.reset_iterator):
                            loader.reset_iterator()
                            logger.debug(f"Reset iterator for dataloader: {loader_name}")
                except Exception as e:
                    logger.warning(f"Failed to sync RNG/data state after dump: {e}")
            else:
                # Mark as pending
                self._pending_model = model_snapshot
                self._pending_config = hp_snapshot
                self._pending_data_state = data_snapshot
                self._has_pending_changes = True
                self._pending_components = changed_components
                logger.info(f"Changes pending (not dumped yet): {changed_components}")

            # Save manager state
            self._save_manager_state()

        return new_hash, is_new, changed_components

    def list_experiment_hashes(self) -> List[str]:
        """List all experiment hashes with checkpoints.

        Returns:
            list: List of experiment hash strings
        """
        if not self.checkpoints_dir.exists():
            return []

        hashes = [d.name for d in self.checkpoints_dir.iterdir() if d.is_dir()]
        return sorted(hashes)

    def has_pending_changes(self) -> tuple[bool, Set[str]]:
        """Check if there are pending changes.

        Returns:
            tuple: (has_pending: bool, pending_components: Set[str])
        """
        return self._has_pending_changes, self._pending_components.copy()

    # ================
    # SAVING FUNCTIONS
    # ================
    def _save_architecture_reference_if_needed(self):
        """Save architecture reference file if architecture doesn't exist in current hash.

        This handles the case where weights are saved to a new hash (due to HP or data changes)
        but the model architecture hasn't changed. Instead of duplicating the architecture file,
        we save a JSON reference pointing to the hash that contains the actual architecture.
        """
        if self.current_exp_hash is None:
            return

        model_dir = self.models_dir / self.current_exp_hash[8:-8]
        arch_file = model_dir / f"{self.current_exp_hash[8:-8]}_architecture.pkl"
        arch_ref_file = model_dir / f"{self.current_exp_hash[8:-8]}_architecture_ref.json"

        # If architecture file already exists here, no need for reference
        if arch_file.exists():
            return

        # If reference file already exists, no need to create it again
        if arch_ref_file.exists():
            return

        # Find the most recent hash with the same model hash that has the architecture
        try:
            component_hashes = self.hash_generator.get_component_hashes()
            current_model_hash = component_hashes.get('model')

            if not current_model_hash:
                return

            # Get all hashes with the same model hash
            matching_hashes = self.get_hashes_by_component(model_hash=current_model_hash)

            # Find the most recent one that has the architecture file
            for hash_candidate in sorted(matching_hashes, reverse=True):
                arch_candidate = self.models_dir / hash_candidate / f"{hash_candidate}_architecture.pkl"
                if arch_candidate.exists():
                    # Save reference to this hash
                    ref_data = {
                        'architecture_hash': hash_candidate,
                        'current_hash': self.current_exp_hash,
                        'model_hash': current_model_hash,
                        'reason': 'Model architecture unchanged, reference points to hash where it is stored',
                        'created': datetime.now().isoformat()
                    }

                    os.makedirs(model_dir, exist_ok=True)
                    with open(arch_ref_file, 'w') as f:
                        json.dump(ref_data, f, indent=2)

                    logger.info(f"Saved architecture reference: {self.current_exp_hash[:16]} → {hash_candidate[:16]}")
                    return
        except Exception as e:
            logger.debug(f"Could not save architecture reference: {e}")

    def _save_manager_state(self):
        """Save manager state (current hash, step counter, etc.)"""
        state_file = self.root_log_dir / ".checkpoint_manager_state.json"

        state = {
            'current_exp_hash': self.current_exp_hash,
            'previous_exp_hash': self.previous_exp_hash,
            'step_counter': self._step_counter,
            'model_init_step': self._model_init_step,
            'last_updated': datetime.now().isoformat(),
            'component_hashes': self.hash_generator.get_component_hashes(),
        }

        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save manager state: {e}")

    def _save_changes(
        self,
        model: Optional[th.nn.Module],
        config: Optional[Dict[str, Any]],
        changed_components: Set[str]
    ):
        """Internal method to dump changes to disk.

        Args:
            model: PyTorch model
            config: Hyperparameters config
            data_state: Data state dict
            changed_components: Set of changed components ('model', 'config', 'data')
        """

        # Get checkpoint manager hp
        manager_hp = config.get('checkpoint_manager', {}) if config else {}
        enable_checkpoints = manager_hp.get('enable_checkpoints', True)
        dump_model_architecture = manager_hp.get('dump_model_architecture', True)
        dump_model_state = manager_hp.get('dump_model_state', True)
        dump_optimizer_state = manager_hp.get('dump_optimizer_state', True)
        dump_data_state = manager_hp.get('dump_data_state', True)
        dump_config_state = manager_hp.get('dump_config_state', True)

        # Do not save checkpoints if disabled
        if not enable_checkpoints:
            logger.info("Checkpoint dumping is disabled in config; skipping dump.")
            return

        # Create checkpoint subdirectories for this hash
        self._create_exp_hash_directories(
            self.current_exp_hash,
            create_data_dir='data' in changed_components and dump_data_state,
            create_hp_dir='config' in changed_components and dump_config_state,
            create_model_dir='model' in changed_components and (
                dump_model_architecture or
                dump_model_state or
                dump_optimizer_state
            )
        )

        # Track if we need to save weights
        should_save_weights = False
        weights_model = None

        if 'model' in changed_components and model is not None and dump_model_architecture:
            logger.info("Dumping model architecture...")
            self.save_model_architecture(model)
            should_save_weights = True
            weights_model = model

        if ('hp' in changed_components or 'config' in changed_components) and config is not None and dump_config_state:
            logger.info("Dumping hyperparameters config...")
            self.save_config(config)
            should_save_weights = True

        if 'data' in changed_components and dump_data_state:
            logger.info("Dumping data snapshot...")
            self.save_data_snapshot()
            should_save_weights = True

        # Save weights whenever any component changes to preserve complete state
        if should_save_weights and (dump_model_state or dump_optimizer_state):
            try:
                # Get model from ledger if not provided
                if weights_model is None:
                    try:
                        weights_model = get_model()
                        if hasattr(weights_model, 'get') and callable(weights_model.get):
                            weights_model = weights_model.get()
                    except Exception:
                        pass

                if weights_model is not None:
                    logger.info("Saving model weights checkpoint with component changes...")
                    self.save_model_checkpoint(
                        save_optimizer=dump_optimizer_state,
                        save_model_checkpoint=dump_model_state
                    )
                else:
                    logger.warning("Could not save weights: no model available")
            except Exception as e:
                logger.warning(f"Failed to save weights with component changes: {e}")

        # Always save logger snapshot alongside other components (same hash)
        if dump_model_architecture or dump_model_state or dump_optimizer_state or dump_config_state:
            self.save_logger_snapshot()

    def save_model_checkpoint(
        self,
        model: Optional[th.nn.Module] = None,
        save_optimizer: bool = True,
        save_model_checkpoint: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        force_dump_pending: bool = False,
        update_manifest: bool = True,
        step: Optional[int] = None
    ) -> Optional[Path]:
        """Save model weights checkpoint.

        This saves only the model weights (state_dict) for fast checkpointing
        during training. The architecture is saved separately when the hash changes.

        Args:
            model: PyTorch model (or get from ledger if None)
            save_optimizer: Whether to also save optimizer state
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
            self.save_pending_changes(force=True)

        # Get model from ledger if not provided
        if model is None:
            model = get_model()

        if model is None:
            logger.error("No model available to checkpoint")
            return None

        # Determine step
        self._step_counter = model.get_age() if step is None else step

        # Prepare checkpoint data
        checkpoint = {
            'step': self._step_counter,
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
            'exp_hash': self.current_exp_hash,
            'rng_state': capture_rng_state(),  # Capture RNG state for reproducible training
        }

        # Capture dataloader iteration state(s) for reproducible resume (support multiple loaders)
        try:
            loader_states = {}
            for loader_name in get_dataloaders():
                dataloader = get_dataloader(loader_name)
                if dataloader is not None and hasattr(dataloader, 'capture_iteration_state'):
                    loader_states[loader_name] = dataloader.capture_iteration_state()
            if loader_states:
                checkpoint['dataloader_iteration_state'] = loader_states
                logger.debug(f"Captured dataloader iteration states: {loader_states}")
        except Exception as e:
            logger.debug(f"Could not capture dataloader iteration state: {e}")

        # Add optimizer state if requested
        if save_optimizer:
            try:
                optimizer = get_optimizer()
                if hasattr(optimizer, 'get') and callable(optimizer.get):
                    optimizer = optimizer.get()
                if optimizer is not None:
                    checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            except Exception as e:
                logger.warning(f"Could not save optimizer state: {e}")

        # Save checkpoint only if enabled in config to avoid unnecessary disk usage when only architecture/config changes
        if not save_model_checkpoint:
            return None

        # Add metadata
        if metadata:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        model_dir = self.models_dir / self.current_exp_hash[8:-8]
        os.makedirs(model_dir, exist_ok=True)
        # Use full exp_hash in filename for clarity and uniqueness
        checkpoint_file = model_dir / f"{self.current_exp_hash}_step_{self._step_counter:06d}.pt"

        try:
            th.save(checkpoint, checkpoint_file)
            logger.info(f"Saved model checkpoint: {checkpoint_file.name}")

            # Update manifest with latest weight checkpoint for this experiment
            if update_manifest:
                self._update_manifest_weight_checkpoint(self.current_exp_hash, checkpoint_file.name, self._step_counter)

            # If model architecture doesn't exist in this hash directory, save a reference to where it is
            self._save_architecture_reference_if_needed()

            # Persist logger queues alongside weight checkpoints
            try:
                self.save_logger_snapshot()
            except Exception as e:
                logger.debug(f"Could not save logger snapshot with checkpoint: {e}")
            return checkpoint_file
        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}")
            return None

    def save_model_architecture(
        self,
        model: th.nn.Module,
    ) -> Optional[Path]:
        """Save full model architecture (structure + code).

        This is saved once per experiment hash when the architecture changes.
        Uses dill for serialization to handle custom modules.

        Args:
            model: PyTorch model

        Returns:
            Path: Path to saved architecture file, or None if failed
        """
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        model_dir = self.models_dir / self.current_exp_hash[8:-8]
        os.makedirs(model_dir, exist_ok=True)
        arch_file = model_dir / f"{self.current_exp_hash[8:-8]}_architecture.pkl"

        # Don't overwrite if already exists
        if arch_file.exists():
            logger.debug(f"Architecture already saved for {self.current_exp_hash[8:-8]}")
            return arch_file

        # Remove links to lock objects in the model
        if hasattr(model, 'guard_training_context'):
            del model.guard_training_context
        if hasattr(model, 'guard_testing_context'):
            del model.guard_testing_context

        # Save
        try:
            tmp_arch_file = arch_file.with_suffix(arch_file.suffix)

            # Try dill first (better for custom classes)
            if dill is not None:
                with open(tmp_arch_file, 'wb') as f:
                    dill.dump(model, f)
                    f.flush()
                    os.fsync(f.fileno())
            else:
                with open(tmp_arch_file, 'wb') as f:
                    pickle.dump(model, f)
                    f.flush()
                    os.fsync(f.fileno())

            os.replace(tmp_arch_file, arch_file)
            logger.info(f"Saved model architecture: {arch_file.name}")

            # Also save a text representation
            arch_txt = model_dir / f"{self.current_exp_hash[8:-8]}_architecture.txt"
            with open(arch_txt, 'w') as f:
                f.write(str(model))

            return arch_file
        except Exception as e:
            import traceback
            logger.error(f"Failed to save model architecture: {e}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
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

        hp_hash_dir = self.hp_dir / self.current_exp_hash[:8]
        os.makedirs(hp_hash_dir, exist_ok=True)
        config_file = hp_hash_dir / f"{self.current_exp_hash[:8]}_{config_name}.yaml"

        try:
            config_with_meta = {
                'hyperparameters': config,
                'exp_hash': self.current_exp_hash[:8],
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
        """Save lightweight JSON snapshot of data state (sample_id, tags, discarded) + RNG state.

        H5 files (data.h5 and arrays.h5) are saved in parent directory (shared).
        Only checkpoint-specific metadata is saved here as JSON, including random state
        for reproducible data generation.
        """
        if self.current_exp_hash is None:
            logger.warning("No experiment hash set. Call update_experiment_hash first.")
            return None

        try:
            # Get dataframe manager
            dfm = ledgers.get_dataframe()
            if dfm is None:
                return None

            # Trigger H5 flush to parent directory (shared)
            dfm.flush_if_needed_nonblocking(force=True)

            # Extract only sample_id, tags, discarded for this checkpoint
            df = dfm.get_df_view()
            if df.empty:
                return None

            if 'sample_id' not in df.columns:
                df = df.reset_index()

            # Keep only checkpoint-specific columns
            available_cols = [
                col for col in df.columns if col in [
                    SampleStatsEx.SAMPLE_ID.value,
                    SampleStatsEx.DISCARDED.value
                ] or col.startswith(SampleStatsEx.TAG.value)
            ]

            # Get dataframe snapshot with only relevant columns for checkpoint metadata (sample_id, tags cols, discarded)
            snapshot_df = df[available_cols]

            # Capture current RNG states for reproducibility using tool function
            rng_state = capture_rng_state()

            # Convert to JSON-serializable format
            snapshot_data = {
                'exp_hash': self.current_exp_hash,
                'timestamp': datetime.now().isoformat(),
                'data': snapshot_df.to_dict(orient='records'),
                'rng_state': rng_state
            }

            # Capture dataloader iteration state(s) for reproducible resume (support multiple loaders)
            try:
                loader_states = {}
                for loader_name in get_dataloaders():
                    dataloader = get_dataloader(loader_name)
                    if dataloader is not None and hasattr(dataloader, 'capture_iteration_state'):
                        loader_states[loader_name] = dataloader.capture_iteration_state()
                if loader_states:
                    snapshot_data['dataloader_iteration_state'] = loader_states
                    logger.debug(f"Captured dataloader iteration states: {loader_states}")
            except Exception as e:
                logger.debug(f"Could not capture dataloader iteration state: {e}")

            # Save to hash-specific directory
            data_hash_dir = self.data_checkpoint_dir / self.current_exp_hash[-8:]
            os.makedirs(data_hash_dir, exist_ok=True)
            json_file = data_hash_dir / f"{self.current_exp_hash[-8:]}_data_snapshot.json"

            with open(json_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)

            logger.info(f"Saved data snapshot: {json_file.name} ({len(snapshot_df)} rows) with RNG state")
            return json_file

        except Exception as e:
            logger.error(f"Failed to save data snapshot: {e}")
        return None

    def save_logger_snapshot(self, exp_hash: Optional[str] = None) -> Optional[Path]:
        """Persist logger queues for the given experiment hash.

        Uses the same hash as model/hp/data; does not affect hashing.
        """
        exp = exp_hash or self.current_exp_hash
        if exp is None:
            return None

        try:
            logger_names = ledgers.list_loggers()
            if not logger_names:
                return None

            snapshot = {"exp_hash": exp, "timestamp": datetime.now().isoformat(), "loggers": {}}
            for lname in logger_names:
                lg = ledgers.get_logger(lname)
                if lg is None:
                    continue

                # Expect LoggerQueue interface
                signal_history = lg.get_signal_history() if hasattr(lg, "get_signal_history") else []
                signal_history_per_sample = lg.get_signal_history_per_sample() if hasattr(lg, "get_signal_history_per_sample") else {}
                graphs = lg.get_graph_names() if hasattr(lg, "get_graph_names") else []

                # Get final snapshot for this logger
                snapshot["loggers"][lname] = {
                    "signal_history": signal_history,
                    "signal_history_per_sample": signal_history_per_sample,
                    "graph_names": graphs,
                }

            if not snapshot["loggers"]:
                return None

            snapshot_dir = self._get_logger_snapshot_dir()
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            # Merge existing payload to avoid dropping loggers not currently registered
            try:
                existing = self._load_logger_snapshot_payload()
                existing_loggers = existing.get("loggers", {}) if isinstance(existing, dict) else {}
                if existing_loggers:
                    existing_loggers.update(snapshot.get("loggers", {}))
                    snapshot["loggers"] = existing_loggers
            except Exception as e:
                logger.warning(f"Failed to merge existing logger snapshot: {e}")

            records: List[bytes] = []
            for lname, payload in snapshot["loggers"].items():
                line = json.dumps({"logger_name": lname, "payload": payload}, default=str) + "\n"
                records.append(line.encode("utf-8"))

            chunks: List[bytes] = []
            current_chunk = bytearray()
            for record in records:
                if current_chunk and (len(current_chunk) + len(record) > self.LOGGER_SNAPSHOT_MAX_FILE_SIZE_BYTES):
                    chunks.append(bytes(current_chunk))
                    current_chunk = bytearray()
                current_chunk.extend(record)
            if current_chunk:
                chunks.append(bytes(current_chunk))

            old_chunks = self._list_logger_snapshot_chunks()
            for old_chunk in old_chunks:
                try:
                    old_chunk.unlink()
                except Exception:
                    pass

            compressor = zstd.ZstdCompressor(level=3)
            chunk_names: List[str] = []
            for idx, raw_chunk in enumerate(chunks, start=1):
                chunk_path = self._get_logger_snapshot_chunk_path(idx)
                tmp_chunk_path = chunk_path.with_name(chunk_path.name + ".tmp")
                with open(tmp_chunk_path, "wb") as f:
                    f.write(compressor.compress(raw_chunk))
                os.replace(tmp_chunk_path, chunk_path)
                chunk_names.append(chunk_path.name)

            manifest = {
                "exp_hash": exp,
                "timestamp": datetime.now().isoformat(),
                "format": "ndjson+zstd",
                "max_file_size_bytes": self.LOGGER_SNAPSHOT_MAX_FILE_SIZE_BYTES,
                "chunks": chunk_names,
            }
            manifest_path = self._get_logger_snapshot_manifest_path()
            tmp_manifest_path = manifest_path.with_name(manifest_path.name + ".tmp")
            with open(tmp_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(tmp_manifest_path, manifest_path)

            logger.info(f"Saved logger snapshot: {manifest_path} ({len(chunk_names)} chunks)")
            return manifest_path
        except Exception as e:
            logger.warning(f"Failed to save logger snapshot: {e}")
            return None

    def save_pending_changes(self, force: bool = False) -> bool:
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

        self._save_changes(
            model=self._pending_model,
            config=self._pending_config,
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

    # =================
    # LOADING FUNCTIONS
    # =================
    def _load_architecture_with_retry(self, arch_file: Path, max_retries: int = 5, base_delay: float = 0.2):
        """Load a model architecture file with retry/backoff for transient file locks."""
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                if not arch_file.exists() or arch_file.stat().st_size == 0:
                    raise EOFError("Architecture file is empty or missing")

                with open(arch_file, 'rb') as f:
                    if dill is not None:
                        try:
                            return dill.load(f)
                        except Exception as e:
                            if 'cannot acquire lock' in str(e).lower():
                                pass
                            else:
                                raise

                # Retry with a fresh handle using dill ignore=True when lock was hit
                with open(arch_file, 'rb') as f:
                    if dill is not None:
                        return dill.load(f, ignore=True)
                    return pickle.load(f)
            except (PermissionError, OSError) as e:
                last_error = e
                msg = str(e).lower()
                if 'lock' not in msg and 'permission' not in msg:
                    break
                sleep_time = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"  [WARN] Architecture load locked (attempt {attempt}/{max_retries}). "
                    f"Retrying in {sleep_time:.2f}s..."
                )
                time.sleep(sleep_time)
            except Exception as e:
                last_error = e
                if isinstance(e, EOFError):
                    sleep_time = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"  [WARN] Architecture load incomplete (attempt {attempt}/{max_retries}). "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    continue
                break
        if last_error:
            raise last_error

    def _load_all_logger_snapshots(self):
        """Load all logger snapshots found under loggers/ for visibility when starting."""
        if not self.loggers_dir.exists():
            return
        has_snapshot = (
            (self.loggers_dir / self.LOGGER_SNAPSHOT_MANIFEST_FILE).exists()
            or (self.loggers_dir / "loggers.json").exists()
            or any(self.loggers_dir.glob(f"{self.LOGGER_SNAPSHOT_CHUNK_PREFIX}*{self.LOGGER_SNAPSHOT_CHUNK_SUFFIX}"))
        )
        if has_snapshot:
            self.load_logger_snapshot(self.loggers_dir.name)

    def _load_manifest(self) -> Dict[str, Any]:
        """Load manifest file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return yaml.safe_load(f) or {'experiments': {}, 'latest_hash': None}
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return {'experiments': {}, 'latest_hash': None}

    def _load_manager_state(self):
        """Load manager state if available"""
        state_file = self.root_log_dir / ".checkpoint_manager_state.json"

        if not state_file.exists():
            # handled above
            # No explicit state file; try to derive from manifest
            manifest = self._load_manifest()
            latest = manifest.get('latest_hash')
            if latest:
                self.current_exp_hash = latest
                exp_info = manifest.get('experiments', {}).get(latest, {})
                component_hashes = {
                    'hp': exp_info.get('hp_hash'),
                    'model': exp_info.get('model_hash'),
                    'data': exp_info.get('data_hash'),
                    'combined': latest,
                }
                self.hash_generator.restore_hashes(component_hashes, combined_hash=latest)
                logger.info(f"Derived manager state from manifest: hash={latest}")
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            self.current_exp_hash = state.get('current_exp_hash')
            self.previous_exp_hash = state.get('previous_exp_hash')
            self._step_counter = state.get('step_counter', 0)
            self._model_init_step = state.get('model_init_step', 0)

            component_hashes = state.get('component_hashes')

            # Fallback: derive component hashes from manifest when missing
            if not component_hashes and self.current_exp_hash:
                manifest = self._load_manifest()
                exp_info = manifest.get('experiments', {}).get(self.current_exp_hash, {})
                if exp_info:
                    component_hashes = {
                        'hp': exp_info.get('hp_hash'),
                        'model': exp_info.get('model_hash'),
                        'data': exp_info.get('data_hash'),
                        'combined': self.current_exp_hash
                    }

            self.hash_generator.restore_hashes(
                component_hashes,
                combined_hash=self.current_exp_hash
            )

            logger.info(f"Loaded manager state: hash={self.current_exp_hash}, step={self._step_counter}")
        except Exception as e:
            logger.warning(f"Failed to load manager state: {e}")

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

    def load_logger_snapshot(self, exp_hash: str) -> bool:
        """Load logger queues from snapshot for a specific experiment hash."""
        try:
            snapshot = self._load_logger_snapshot_payload()
            if not snapshot:
                return False

            loggers_payload = snapshot.get("loggers", {})
            for lname, payload in loggers_payload.items():
                try:
                    lg = ledgers.get_logger(lname) if lname in ledgers.list_loggers() else None
                    if lg is None or not hasattr(lg, "load_snapshot"):
                        lg = LoggerQueue(register=False)
                        ledgers.register_logger(lg, name=lname)
                    lg.load_snapshot(payload)
                except Exception as inner_e:
                    logger.warning(f"Failed to restore logger '{lname}': {inner_e}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load logger snapshot for {exp_hash}: {e}")
            return False

    def load_checkpoint(self,
                        exp_hash: str,
                        load_model: bool = True,
                        load_weights: bool = True,
                        load_config: bool = True,
                        load_data: bool = True,
                        target_step: Optional[int] = None,
                        force: bool = False
    ) -> Dict[str, Any]:
        """Load a complete checkpoint state by experiment hash.

        This method intelligently loads only the components that differ from
        the current state by comparing component hashes.

        Args:
            exp_hash: The 24-byte experiment hash to load (HP_MODEL_DATA)
            load_model: Whether to load model architecture if different
            load_weights: Whether to load model weights
            load_config: Whether to load hyperparameters if different
            load_data: Whether to load data state if different
            force: If True, force reload of all components regardless of hash comparison

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
            'rng_state': None,
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

        # Logger
        logger.info(f"Loading checkpoint {exp_hash[:16]}...")
        logger.info(f"  Target: HP={target_hp_hash} MODEL={target_model_hash} DATA={target_data_hash}")
        logger.info(f"  Current: HP={current_hp_hash} MODEL={current_model_hash} DATA={current_data_hash}")

        # Load model architecture if different, or load only RNG state for reproducibility if model hash is unchanged
        model_rng_loaded = False
        if load_model and (target_model_hash != current_model_hash or force):
            model_dir = self.models_dir / exp_hash[8:-8]
            arch_ref_file = model_dir / f"{exp_hash[8:-8]}_architecture_ref.json"

            # First check if this is a reference to architecture in another hash
            actual_arch_hash = exp_hash[8:-8]
            if arch_ref_file.exists():
                try:
                    with open(arch_ref_file, 'r') as f:
                        ref_data = json.load(f)
                    actual_arch_hash = ref_data.get('architecture_hash', exp_hash[8:-8])
                    logger.debug(f"  Architecture reference found: pointing to hash {actual_arch_hash}")
                except Exception as e:
                    logger.warning(f"Failed to load architecture reference: {e}")

            # Now load from actual location
            actual_arch_file = self.models_dir / actual_arch_hash / f"{actual_arch_hash}_architecture.pkl"

            if actual_arch_file.exists():
                try:
                    result['model'] = self._load_architecture_with_retry(actual_arch_file)
                    # Remove links to lock objects in the model
                    if not hasattr(result['model'], 'guard_training_context'):
                        result['model'].guard_training_context = guard_training_context
                    if not hasattr(result['model'], 'guard_testing_context'):
                        result['model'].guard_testing_context = guard_testing_context

                    result['loaded_components'].add('model')
                    logger.info(f"  [OK] Loaded model architecture from hash {actual_arch_hash[:16]}")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load model architecture: {e}")
            else:
                logger.warning(f"  [WARNING] Model architecture file not found: {actual_arch_file}")

        elif load_model and (target_model_hash == current_model_hash and not force):
            # Try to load only the RNG state from the latest model checkpoint for reproducibility
            model_dir = self.models_dir / exp_hash[8:-8]
            checkpoint_files = sorted(model_dir.glob(f"{exp_hash}_step_*.pt"))
            if not checkpoint_files:
                checkpoint_files = sorted(model_dir.glob(f"{exp_hash[8:-8]}_step_*.pt"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                try:
                    checkpoint = th.load(latest_checkpoint, weights_only=False)
                    rng_state = checkpoint.get('rng_state')
                    if rng_state:
                        result['rng_state'] = rng_state
                        model_rng_loaded = True
                        logger.info(f"  [OK] Loaded model RNG state for reproducibility (model unchanged)")
                except Exception as e:
                    logger.debug(f"  [WARNING] Could not load model RNG state: {e}")
            if not model_rng_loaded:
                logger.info(f"  [-] Model architecture unchanged, using current model")
        else:
            logger.info(f"  [-] Model architecture unchanged, using current model")

        # Load model weights (always if requested)
        if load_weights:
            model_dir = self.models_dir / exp_hash[8:-8]

            # First, try to get the weight checkpoint from manifest for this specific experiment
            checkpoint_file_to_load = None
            exp_info = manifest['experiments'][exp_hash]
            manifest_weight_checkpoint = exp_info.get('latest_weight_checkpoint')

            if manifest_weight_checkpoint and target_step is None:
                checkpoint_path = model_dir / manifest_weight_checkpoint
                if checkpoint_path.exists():
                    checkpoint_file_to_load = checkpoint_path
                    logger.debug(f"  Using weight checkpoint from manifest: {manifest_weight_checkpoint}")

            # Fallback: scan for weight files (old behavior for backward compatibility)
            if checkpoint_file_to_load is None:
                checkpoint_file_to_load = self._select_weight_checkpoint_file(exp_hash, target_step=target_step)
                if checkpoint_file_to_load is not None:
                    if target_step is None:
                        logger.debug(f"  Using latest weight checkpoint from directory scan: {checkpoint_file_to_load.name}")
                    else:
                        logger.debug(f"  Using closest weight checkpoint for target step {target_step}: {checkpoint_file_to_load.name}")

            if checkpoint_file_to_load:
                try:
                    result['weights'] = th.load(checkpoint_file_to_load, weights_only=False)
                    result['loaded_components'].add('weights')
                    step = result['weights'].get('step', -1)

                    # Extract RNG state from model checkpoint if available
                    checkpoint_rng_state = result['weights'].get('rng_state')
                    if checkpoint_rng_state:
                        result['rng_state'] = checkpoint_rng_state
                        logger.info(f"  [OK] Loaded weights from step {step} with RNG state")
                    else:
                        logger.info(f"  [OK] Loaded weights from step {step}")

                    # Extract dataloader iteration state if available
                    dataloader_iter_state = result['weights'].get('dataloader_iteration_state')
                    if dataloader_iter_state:
                        # Normalize to mapping of loader_name -> state for backward compatibility
                        if isinstance(dataloader_iter_state, dict) and 'samples_yielded' in dataloader_iter_state:
                            iter_state_map = {'default': dataloader_iter_state}
                        elif isinstance(dataloader_iter_state, dict):
                            iter_state_map = dataloader_iter_state
                        else:
                            iter_state_map = {'default': dataloader_iter_state}

                        result['dataloader_iteration_state'] = iter_state_map
                        logger.debug(f"  [OK] Found dataloader iteration state(s): {iter_state_map}")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load weights: {e}")
            else:
                logger.warning(f"  [WARNING] No weight files found for {exp_hash[8:-8]}")

        # Load config if different
        if load_config and (target_hp_hash != current_hp_hash or force):
            hp_dir = self.hp_dir / exp_hash[:8]
            config_file = hp_dir / f"{exp_hash[:8]}_config.yaml"

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

        # Load data snapshot if different, or if only RNG state changed (for reproducibility)
        if load_data:
            data_dir = self.data_checkpoint_dir / exp_hash[-8:]
            json_file = data_dir / f"{exp_hash[-8:]}_data_snapshot.json"

            # Always try to load RNG state for reproducibility, even if data hash is unchanged
            load_data_snapshot = (target_data_hash != current_data_hash or force)
            load_rng_only = (target_data_hash == current_data_hash and not force)

            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        snapshot_data = json.load(f)

                    rng_state = snapshot_data.get('rng_state', {})

                    if load_data_snapshot:
                        snapshot_df = pd.DataFrame(snapshot_data.get('data', []))
                        if not snapshot_df.empty:
                            result['data_state'] = {'snapshot': snapshot_df}
                            result['loaded_components'].add('data')
                            if rng_state:
                                result['rng_state'] = rng_state
                                logger.info(f"  [OK] Loaded data snapshot ({len(snapshot_df)} rows) with RNG state")
                            else:
                                logger.info(f"  [OK] Loaded data snapshot ({len(snapshot_df)} rows)")
                    elif load_rng_only and rng_state:
                        # Only RNG state is needed for reproducibility
                        result['rng_state'] = rng_state
                        logger.info(f"  [OK] Loaded RNG state for reproducibility (data unchanged)")
                    else:
                        logger.info(f"  [-] Data state unchanged, using current data")
                except Exception as e:
                    logger.error(f"  [ERROR] Failed to load data snapshot: {e}")
            else:
                logger.warning(f"  [WARNING] Data snapshot file not found: {json_file}")

        logger.info(f"Loaded components: {result['loaded_components']}")
        return result

    def load_state(
        self,
        exp_hash: str,
        force: bool = False,
        load_model: bool = True,
        load_weights: bool = True,
        load_config: bool = True,
        load_data: bool = True,
        target_step: Optional[int] = None,
    ) -> bool:
        """Load and apply a complete checkpoint state by experiment hash.

        This method loads all components and updates the system state in-place:
        - Updates model in ledger (architecture + weights)
        - Updates config in ledger
        - Updates dataframe manager with loaded data
        - Updates current experiment hash

        Args:
            exp_hash: The 24-byte experiment hash to load and apply
            force: If True, force reload of all components regardless of hash comparison

        Returns:
            bool: True if state was successfully loaded and applied
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading and applying state: {exp_hash[:16]}...")
        logger.info(f"{'='*60}")

        # Load checkpoint data
        checkpoint_data = self.load_checkpoint(
            exp_hash=exp_hash,
            load_model=load_model,
            load_weights=load_weights,
            load_config=load_config,
            load_data=load_data,
            target_step=target_step,
            force=force
        )
        if not checkpoint_data['loaded_components']:
            logger.warning("No components were loaded")
            return False
        success = True

        # Apply model (architecture + weights)
        if 'model' in checkpoint_data['loaded_components']:
            try:
                model = checkpoint_data['model']

                # Register in ledger
                ledgers.register_model(model)

                # Set Model Training Guard
                guard_training_context.model = model  # Train
                guard_testing_context.model = model  # Eval

                loaded_step = None
                if checkpoint_data.get('weights') is not None:
                    loaded_step = checkpoint_data['weights'].get('step', None)
                if loaded_step is not None:
                    self._model_init_step = int(loaded_step)
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply model: {e}")
                success = False

        elif 'weights' in checkpoint_data['loaded_components']:
            # Only weights changed, apply to existing model
            try:
                model = ledgers.get_model()
                weights = checkpoint_data['weights']
                if model and weights and 'model_state_dict' in weights:
                    model.load_state_dict(weights['model_state_dict'])
                    step = weights.get('step', -1)
                    logger.info(f"[OK] Applied weights to existing model (step {step})")
                    self._model_init_step = int(step)

                # Set Model Training Guard
                guard_training_context.model = model  # Train
                guard_testing_context.model = model  # Eval
            except Exception:
                if 'model' not in checkpoint_data['loaded_components']:
                    logger.info("Attempting to reload full checkpoint to recover...")
                    # Load checkpoint data
                    model_data = self.load_checkpoint(
                        exp_hash=exp_hash,
                        load_model=True,
                        load_weights=True,
                        load_config=False,
                        load_data=False,
                        force=True
                    )

                    if not model_data['loaded_components']:
                        logger.warning("No components were loaded")
                        return False
                try:
                    model = model_data['model']
                    ledgers.register_model(model)
                    weights = checkpoint_data['weights']
                    if model and weights and 'model_state_dict' in weights:
                        model.load_state_dict(weights['model_state_dict'])
                        step = weights.get('step', -1)
                        logger.info(f"[OK] Applied weights to reloaded model (step {step})")
                        self._model_init_step = int(step)

                    # Set Model Training Guard
                    guard_training_context.model = model  # Train
                    guard_testing_context.model = model  # Eval
                except Exception as e:
                    logger.error(f"[ERROR] Failed to apply weights: {e}")
                    success = False

        # Apply config
        if 'config' in checkpoint_data['loaded_components']:
            try:
                config = checkpoint_data['config']
                ledgers.register_hyperparams(config)
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
                    dfm = ledgers.get_dataframe()
                    if dfm is not None:
                        # Set index if needed
                        if 'sample_id' in snapshot_df.columns:
                            snapshot_df = snapshot_df.set_index('sample_id')

                        # Merge only the checkpoint-specific columns (tags, discarded)
                        # This updates existing rows without replacing all data
                        dfm.upsert_df(snapshot_df, force_flush=True)
                        logger.info(f"[OK] Applied data snapshot ({len(snapshot_df)} rows)")
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply data: {e}")
                success = False

        # Restore RNG state if provided and not already restored
        if checkpoint_data.get('rng_state'):
            try:
                restore_rng_state(checkpoint_data['rng_state'])
                logger.debug(f"Restored RNG state from checkpoint")

                # Reset dataloaders iterators to ensure reproducibility
                for loader_name in ledgers.get_dataloaders():
                    loader = ledgers.get_dataloader(loader_name)

                    if loader is not None:
                        # Resume loader state
                        if hasattr(loader, 'reset_iterator') and callable(loader.reset_iterator):
                            loader.reset_iterator()
                            logger.debug(f"Reset iterator for dataloader: {loader}")

                # Restore RNG state again after resetting dataloaders
                restore_rng_state(checkpoint_data['rng_state'])
                logger.debug(f"Restored RNG state from checkpoint")

            except Exception as e:
                logger.error(f"[ERROR] Failed to restore RNG state: {e}")
                pause_ctrl.pause()
                success = False

        # Restore dataloader iteration state if provided
        if checkpoint_data.get('dataloader_iteration_state'):
            try:
                iter_state_raw = checkpoint_data['dataloader_iteration_state']

                # Normalize to mapping loader_name -> state for backward compatibility
                if isinstance(iter_state_raw, dict) and 'samples_yielded' in iter_state_raw:
                    state_map = {'default': iter_state_raw}
                elif isinstance(iter_state_raw, dict):
                    state_map = iter_state_raw
                else:
                    state_map = {'default': iter_state_raw}

                restored_any = False
                for loader_name in ledgers.get_dataloaders():
                    loader = ledgers.get_dataloader(loader_name)
                    if loader is None or not hasattr(loader, 'restore_iteration_state'):
                        continue

                    state_for_loader = state_map.get(loader_name) or state_map.get('default')
                    if state_for_loader:
                        try:
                            loader.restore_iteration_state(state_for_loader)
                            # Resume loader state
                            if hasattr(loader, 'reset_iterator') and callable(loader.reset_iterator):
                                loader.reset_iterator()
                                logger.debug(f"Reset iterator for dataloader: {loader}")
                            logger.info(f"[OK] Restored dataloader iteration state for {loader_name}: {state_for_loader}")
                            restored_any = True
                        except Exception as inner_e:
                            logger.warning(f"[WARNING] Failed to restore iteration state for {loader_name}: {inner_e}")

                if not restored_any:
                    logger.warning("No dataloader iteration state could be applied to registered loaders")
            except Exception as e:
                logger.error(f"[ERROR] Failed to restore dataloader iteration state: {e}")
                success = False

        # Restore logger snapshot for this experiment if available
        try:
            self.load_logger_snapshot(exp_hash)
        except Exception as e:
            logger.warning(f"Failed to restore logger snapshot for {exp_hash}: {e}")

        # Update current experiment hash
        if success:
            old_hash = self.current_exp_hash
            self.current_exp_hash = exp_hash
            self.previous_exp_hash = old_hash

            # Keep hash generator in sync with loaded experiment
            manifest = self._load_manifest()
            exp_info = manifest.get('experiments', {}).get(exp_hash, {})
            component_hashes = {
                'hp': exp_info.get('hp_hash'),
                'model': exp_info.get('model_hash'),
                'data': exp_info.get('data_hash'),
                'combined': exp_hash
            }
            self.hash_generator.restore_hashes(component_hashes, combined_hash=exp_hash)

            self._save_manager_state()
            logger.info(f"\n[OK] Successfully loaded and applied state: {exp_hash[:16]}")
        else:
            logger.warning(f"\n[WARNING] State loaded with errors")

        logger.info(f"{'='*60}\n")
        return success
