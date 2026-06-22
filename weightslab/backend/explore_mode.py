"""Process-wide read-only "explore" mode.

When the backend is launched to browse a finished experiment loaded from disk
(``weightslab --logdir <root_log_dir>``), it runs in *explore mode*: there is no
training loop, and the experiment is reconstructed from the checkpoints/logs on
disk so a user can inspect it in the UI while training continues elsewhere
(e.g. on a cluster).

In this mode the backend refuses the actions that would mutate the model or the
training run — starting/resuming training, changing hyperparameters, and
loading/restoring/saving weights or checkpoints. Local **data management**
(tagging, discarding, queries, plot notes) and all **reads** stay available,
since the whole point is to manage and explore the data locally.

This is a simple process-wide flag: a given backend process is either a live
training server or a read-only explorer for its whole lifetime.
"""

import logging

logger = logging.getLogger(__name__)

_EXPLORE_MODE = False

# Returned by guarded RPC handlers when a forbidden (mutating) action is attempted.
EXPLORE_BLOCKED_MESSAGE = (
    "This experiment is open in read-only explore mode (loaded from --logdir). "
    "Training, hyperparameter changes, and weight/checkpoint loading are disabled."
)


def set_explore_mode(enabled: bool) -> None:
    """Enable/disable the process-wide read-only explore mode."""
    global _EXPLORE_MODE
    _EXPLORE_MODE = bool(enabled)
    logger.info(
        "Explore (read-only) mode %s", "ENABLED" if _EXPLORE_MODE else "disabled"
    )


def is_explore_mode() -> bool:
    """True when the backend is serving a read-only experiment from disk."""
    return _EXPLORE_MODE
