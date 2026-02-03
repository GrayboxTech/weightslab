"""
Weightslab Components Module

This module contains core components for experiment tracking, checkpointing,
and monitoring in Weightslab.
"""

# New structured checkpoint system
from weightslab.components.checkpoint_manager_v2 import CheckpointManagerV2
from weightslab.components.experiment_hash import ExperimentHashGenerator

# Other components
from weightslab.components.tracking import Tracker, TrackingMode
# from weightslab.components.global_monitoring import GlobalMonitoring  # TODO: Fix missing GlobalMonitoring class


__all__ = [
    # Checkpoint management
    'CheckpointManagerV2',  # Manual checkpoint system
    'ExperimentHashGenerator',

    # Tracking
    'Tracker',
    'TrackingMode',

    # Monitoring - commented out until GlobalMonitoring is implemented
    # 'GlobalMonitoring',
]
