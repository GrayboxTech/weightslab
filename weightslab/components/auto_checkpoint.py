"""
Automatic Checkpoint System - Ledger-Integrated

This module provides a fully automatic checkpoint management system that:
1. Registers itself in the ledger
2. Auto-initializes on first model/dataloader registration
3. Automatically saves checkpoints every N steps
4. Detects and responds to:
   - Model architecture changes → triggers new hash (pending until resume)
   - Hyperparameter updates → triggers new hash (pending until resume)
   - Data changes (discard, tags) → triggers new hash (pending until resume)
   - Model state changes (freeze/reset) → saves metadata

Changes are marked as "pending" until training resumes or manual dump is forced.

The system is completely transparent to the user - no manual calls needed.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set
from datetime import datetime

import torch as th

from weightslab.components.checkpoint_manager_v2 import CheckpointManagerV2
from weightslab.components.experiment_hash import ExperimentHashGenerator
from weightslab.backend import ledgers


logger = logging.getLogger(__name__)


class AutomaticCheckpointSystem:
    """Automatic checkpoint system that integrates with the ledger.

    This system:
    - Monitors model, optimizer, and hyperparameter registrations
    - Automatically saves checkpoints every N steps
    - Detects configuration changes and creates new checkpoint directories
    - Completely transparent to the user

    Attributes:
        checkpoint_manager (CheckpointManagerV2): Core checkpoint manager
        checkpoint_frequency (int): Save checkpoints every N steps
        _initialized (bool): Whether system has been initialized
        _step_counter (int): Current training step
        _lock (threading.Lock): Thread safety lock
    """

    def __init__(
        self,
        root_log_dir: str = 'root_experiment',
        checkpoint_frequency: int = 100,
        auto_register: bool = True
    ):
        """Initialize the automatic checkpoint system.

        Args:
            root_log_dir: Root directory for experiments
            checkpoint_frequency: Save checkpoints every N steps
            auto_register: Auto-register in ledger on init
        """
        self.checkpoint_manager = CheckpointManagerV2(root_log_dir=root_log_dir)
        self.checkpoint_frequency = checkpoint_frequency

        self._initialized = False
        self._step_counter = 0
        self._lock = threading.Lock()

        # Track last known states for change detection
        self._last_model_id = None
        self._last_config = None
        self._last_data_state = None
        self._last_checkpoint_step = -1
        self._training_resumed = False

        logger.info(f"AutomaticCheckpointSystem initialized (freq={checkpoint_frequency})")

    def initialize_from_ledger(self):
        """Initialize checkpoint system from current ledger state.

        This is called automatically on first checkpoint or can be called
        manually to sync with ledger.
        """
        with self._lock:
            if self._initialized:
                logger.debug("Checkpoint system already initialized")
                return

            try:
                model = self._get_model_from_ledger()
                config = self._get_config_from_ledger()
                data = self._get_dfm_from_ledger()
                if model is not None or config is not None:
                    exp_hash, is_new, changed = self.checkpoint_manager.update_experiment_hash(
                        model_snapshot=model,
                        hp_snapshot=config,
                        dfm_snapshot=data,
                    )

                    self._last_config = config.copy() if config else None
                    self._initialized = True

                    logger.info(f"Checkpoint system initialized with hash: {exp_hash}")
                else:
                    logger.warning("No model or config found in ledger for initialization")

            except Exception as e:
                logger.error(f"Failed to initialize from ledger: {e}")

    def on_training_step(self, step: Optional[int] = None, force_dump: bool = False):
        """Called after each training step to potentially save checkpoint.

        This:
        1. Dumps pending changes if training resumed
        2. Checks if it's time for a periodic checkpoint
        3. Can force dump pending changes if requested

        Args:
            step: Training step number (auto-increments if None)
            force_dump: Force dump pending changes regardless of frequency
        """
        with self._lock:
            if not self._initialized:
                self.initialize_from_ledger()

            if step is not None:
                self._step_counter = step
            else:
                self._step_counter += 1

            current_step = self._step_counter

            if not self._training_resumed:
                has_pending, pending_comps = self.checkpoint_manager.has_pending_changes()
                if has_pending:
                    logger.info(f"Training resumed, dumping pending changes: {pending_comps}")
                    self.checkpoint_manager.dump_pending_changes(force=True)
                self._training_resumed = True

            if force_dump:
                self.checkpoint_manager.dump_pending_changes(force=True)

            if current_step % self.checkpoint_frequency == 0:
                if current_step != self._last_checkpoint_step:
                    self._save_checkpoint(step=current_step, force_dump_pending=force_dump)
                    self._last_checkpoint_step = current_step

    def on_model_change(self, model: Optional[th.nn.Module] = None, dump_immediately: bool = False):
        """Called when model architecture changes (add/prune layers).

        By default, changes are marked as pending until training resumes.
        Set dump_immediately=True to dump right away.

        Args:
            model: New model (gets from ledger if None)
            dump_immediately: If True, dump immediately instead of marking pending
        """
        with self._lock:
            if model is None:
                model = self._get_model_from_ledger()

            if model is None:
                return

            config = self._get_config_from_ledger()
            data_state = self._get_data_state_from_ledger()

            exp_hash, is_new, changed_components = self.checkpoint_manager.update_experiment_hash(
                model_snapshot=model,
                hp_snapshot=config,
                dfm_snapshot=data_state,
                dump_immediately=dump_immediately
            )

            if is_new:
                logger.info(f"Model architecture changed, new hash: {exp_hash}")
                if dump_immediately:
                    logger.info("Changes dumped immediately")
                else:
                    logger.info("Changes marked as pending (will dump on training resume)")

            self._training_resumed = False

    def on_config_change(self, config: Optional[Dict[str, Any]] = None, dump_immediately: bool = False):
        """Called when hyperparameters change.

        By default, changes are marked as pending until training resumes.
        Set dump_immediately=True to dump right away.

        Args:
            config: New config (gets from ledger if None)
            dump_immediately: If True, dump immediately instead of marking pending
        """
        with self._lock:
            if config is None:
                config = self._get_config_from_ledger()

            if config is None:
                return

            if self._last_config is not None and config == self._last_config:
                logger.debug("Config unchanged, skipping checkpoint")
                return

            model = self._get_model_from_ledger()
            data_state = self._get_data_state_from_ledger()

            exp_hash, is_new, changed_components = self.checkpoint_manager.update_experiment_hash(
                model_snapshot=model,
                hp_snapshot=config,
                dfm_snapshot=data_state,
                dump_immediately=dump_immediately
            )

            if is_new:
                logger.info(f"Hyperparameters changed, new hash: {exp_hash}")
                if dump_immediately:
                    logger.info("Changes dumped immediately")
                else:
                    logger.info("Changes marked as pending (will dump on training resume)")

            self._last_config = config.copy() if config else None
            self._training_resumed = False

    def on_data_change(self, data_state: Optional[Dict[str, Any]] = None, dump_immediately: bool = False):
        """Called when data state changes (discard, tags).

        By default, changes are marked as pending until training resumes.
        Set dump_immediately=True to dump right away.

        Args:
            data_state: Dict with 'uids', 'discarded', 'tags' (gets from ledger if None)
            dump_immediately: If True, dump immediately instead of marking pending
        """
        with self._lock:
            if data_state is None:
                data_state = self._get_data_state_from_ledger()

            if data_state is None:
                return

            if self._last_data_state is not None and data_state == self._last_data_state:
                logger.debug("Data state unchanged, skipping checkpoint")
                return

            model = self._get_model_from_ledger()
            config = self._get_config_from_ledger()

            exp_hash, is_new, changed_components = self.checkpoint_manager.update_experiment_hash(
                model_snapshot=model,
                hp_snapshot=config,
                dfm_snapshot=data_state,
                dump_immediately=dump_immediately
            )

            if is_new:
                logger.info(f"Data state changed, new hash: {exp_hash}")
                if dump_immediately:
                    logger.info("Changes dumped immediately")
                else:
                    logger.info("Changes marked as pending (will dump on training resume)")

            self._last_data_state = data_state if data_state else None
            self._training_resumed = False

    def on_change(self, dump_immediately: bool = False):
        """Called when any tracked component changes.

        This is a convenience method that checks model, config, and data.
        """
        self.on_model_change(dump_immediately=dump_immediately)
        self.on_config_change(dump_immediately=dump_immediately)
        self.on_data_change(dump_immediately=dump_immediately)

    def on_model_state_change(self, event_type: str):
        """Called when model state changes (freeze, reset, etc.).

        This triggers a checkpoint save with metadata.

        Args:
            event_type: Type of state change ('freeze', 'reset', etc.)
        """
        with self._lock:
            logger.info(f"Model state change: {event_type}")

            model = self._get_model_from_ledger()
            if model is not None:
                self.checkpoint_manager.save_model_checkpoint(
                    model=model,
                    step=self._step_counter,
                    save_optimizer=True,
                    metadata={
                        'trigger': 'state_change',
                        'event_type': event_type,
                        'timestamp': datetime.now().isoformat()
                    }
                )

    def _save_checkpoint(self, step: int, force_dump_pending: bool = False):
        """Internal method to save a checkpoint.

        Args:
            step: Training step number
            force_dump_pending: Force dump pending changes before saving
        """
        try:
            model = self._get_model_from_ledger()

            if model is None:
                logger.warning("No model found in ledger, skipping checkpoint")
                return

            checkpoint_path = self.checkpoint_manager.save_model_checkpoint(
                model=model,
                step=step,
                save_optimizer=True,
                metadata={'step': step},
                force_dump_pending=force_dump_pending
            )

            if checkpoint_path:
                logger.info(f"Saved checkpoint at step {step}")
            else:
                logger.warning(f"Failed to save checkpoint at step {step}")

        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _get_dfm_from_ledger(self) -> Optional[th.nn.Module]:
        """Get DataFrame Manager from ledger, handling proxies.
        Returns:
            Dataframe Manager
        """
        try:
            dfm = ledgers.get_dataframe()

            if dfm is not None:
                return dfm
            return None

        except Exception as e:
            logger.debug(f"Could not get dataframe manager from ledger: {e}")
            return None

    def _get_model_from_ledger(self) -> Optional[th.nn.Module]:
        """Get model from ledger, handling proxies.

        Returns:
            PyTorch model or None
        """
        try:
            model = ledgers.get_model()

            if model is not None:
                return model

            return None

        except Exception as e:
            logger.debug(f"Could not get model from ledger: {e}")
            return None

    def _get_config_from_ledger(self) -> Optional[Dict[str, Any]]:
        """Get hyperparameters from ledger, handling proxies.

        Returns:
            Config dict or None
        """
        try:
            config = ledgers.get_hyperparams()

            if config is not None:
                return config

            return None

        except Exception as e:
            logger.debug(f"Could not get config from ledger: {e}")
            return None

    def _get_data_state_from_ledger(self) -> Optional[Dict[str, Any]]:
        """Get data state from dataloaders in ledger.

        Aggregates UIDs, discard status, and tags from all registered dataloaders.

        Returns:
            Dict with 'uids', 'discarded', 'tags' or None
        """
        try:
            dfm = ledgers.get_dataframe()

            if dfm is not None:
                return dfm

            return None

        except Exception as e:
            logger.debug(f"Could not get dfm from ledger: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the checkpoint system.

        Returns:
            dict: Status information
        """
        with self._lock:
            return {
                'initialized': self._initialized,
                'current_step': self._step_counter,
                'last_checkpoint_step': self._last_checkpoint_step,
                'checkpoint_frequency': self.checkpoint_frequency,
                'current_exp_hash': self.checkpoint_manager.current_exp_hash,
                'root_log_dir': str(self.checkpoint_manager.root_log_dir)
            }


_GLOBAL_CHECKPOINT_SYSTEM: Optional[AutomaticCheckpointSystem] = None
_SYSTEM_LOCK = threading.Lock()


def get_checkpoint_system(
    root_log_dir: Optional[str] = None,
    checkpoint_frequency: int = 100,
    auto_init: bool = True
) -> AutomaticCheckpointSystem:
    """Get or create the global automatic checkpoint system.

    Args:
        root_log_dir: Root directory (only used on first call)
        checkpoint_frequency: Checkpoint frequency (only used on first call)
        auto_init: Auto-initialize from ledger

    Returns:
        AutomaticCheckpointSystem: Global checkpoint system instance
    """
    global _GLOBAL_CHECKPOINT_SYSTEM

    with _SYSTEM_LOCK:
        if _GLOBAL_CHECKPOINT_SYSTEM is None:
            if root_log_dir is None:
                try:
                    hp = ledgers.get_hyperparams()
                    root_log_dir = hp.get('root_log_dir', 'root_experiment') if hp else 'root_experiment'
                except Exception:
                    root_log_dir = 'root_experiment'

            _GLOBAL_CHECKPOINT_SYSTEM = AutomaticCheckpointSystem(
                root_log_dir=root_log_dir,
                checkpoint_frequency=checkpoint_frequency,
                auto_register=True
            )

            if auto_init:
                _GLOBAL_CHECKPOINT_SYSTEM.initialize_from_ledger()

        return _GLOBAL_CHECKPOINT_SYSTEM


def checkpoint_on_step(step: Optional[int] = None, force_dump: bool = False):
    """Convenience function to trigger checkpoint on training step.

    This can be called from training loops. Use force_dump=True to
    immediately dump any pending changes.

    Args:
        step: Training step number
        force_dump: Force dump pending changes
    """
    system = get_checkpoint_system()
    system.on_training_step(step=step, force_dump=force_dump)


def checkpoint_on_model_change(model: Optional[th.nn.Module] = None, dump_immediately: bool = False):
    """Convenience function to trigger checkpoint on model architecture change.

    Args:
        model: New model (gets from ledger if None)
        dump_immediately: Dump changes immediately instead of marking pending
    """
    system = get_checkpoint_system()
    system.on_model_change(model=model, dump_immediately=dump_immediately)


def checkpoint_on_config_change(config: Optional[Dict[str, Any]] = None, dump_immediately: bool = False):
    """Convenience function to trigger checkpoint on config change.

    Args:
        config: New config (gets from ledger if None)
        dump_immediately: Dump changes immediately instead of marking pending
    """
    system = get_checkpoint_system()
    system.on_config_change(config=config, dump_immediately=dump_immediately)


def checkpoint_on_data_change(data_state: Optional[Dict[str, Any]] = None, dump_immediately: bool = False):
    """Convenience function to trigger checkpoint on data state change.

    Args:
        data_state: Dict with 'uids', 'discarded', 'tags' (gets from ledger if None)
        dump_immediately: Dump changes immediately instead of marking pending
    """
    system = get_checkpoint_system()
    system.on_data_change(data_state=data_state, dump_immediately=dump_immediately)


def checkpoint_on_state_change(event_type: str):
    """Convenience function to trigger checkpoint on model state change.

    Args:
        event_type: Type of state change ('freeze', 'reset', etc.)
    """
    system = get_checkpoint_system()
    system.on_model_state_change(event_type=event_type)

def checkpoint_on_change(dump_immediately: bool = False):
    """Convenience function to trigger checkpoint on any tracked component change.
    """
    logger.info('\nCheck if changes to dump.')
    system = get_checkpoint_system(auto_init=False)
    system.on_change(dump_immediately=dump_immediately)
