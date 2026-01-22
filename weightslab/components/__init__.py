"""
Weightslab Components Module

This module contains core components for experiment tracking, checkpointing,
and monitoring in Weightslab.
"""

# Legacy checkpoint manager (deprecated)
from weightslab.components.checkpoint import CheckpointManager

# New structured checkpoint system
from weightslab.components.checkpoint_manager_v2 import CheckpointManagerV2
from weightslab.components.experiment_hash import ExperimentHashGenerator

# Automatic checkpoint system (recommended)
from weightslab.components.auto_checkpoint import (
    AutomaticCheckpointSystem,
    get_checkpoint_system,
    checkpoint_on_step,
    checkpoint_on_model_change,
    checkpoint_on_config_change,
    checkpoint_on_data_change,
    checkpoint_on_state_change,
)

# Other components
from weightslab.components.tracking import Tracker, TrackingMode
# from weightslab.components.global_monitoring import GlobalMonitoring  # TODO: Fix missing GlobalMonitoring class

__all__ = [
    # Checkpoint management
    'CheckpointManager',  # Legacy - deprecated
    'CheckpointManagerV2',  # Manual checkpoint system
    'ExperimentHashGenerator',

    # Automatic checkpoint system (recommended)
    'AutomaticCheckpointSystem',
    'get_checkpoint_system',
    'checkpoint_on_step',
    'checkpoint_on_model_change',
    'checkpoint_on_config_change',
    'checkpoint_on_data_change',
    'checkpoint_on_state_change',

    # Tracking
    'Tracker',
    'TrackingMode',

    # Monitoring - commented out until GlobalMonitoring is implemented
    # 'GlobalMonitoring',
]
