"""Friendly two-word experiment names (adjective-noun), e.g. "wl-brisk-otter".

Shared by ``weightslab.cli`` (naming a fresh experiment directory for
``weightslab start``) and ``CheckpointManager`` (naming a fresh run when no
``experiment_name`` was supplied via config/hyperparameters).
"""

import random
from typing import Callable, Optional

_EXP_ADJECTIVES = (
    "brisk", "calm", "clever", "bold", "bright", "keen", "lucid", "nimble",
    "quiet", "swift", "warm", "zesty", "amber", "cobalt", "coral", "jade",
)
_EXP_NOUNS = (
    "otter", "falcon", "maple", "cedar", "comet", "delta", "ember", "harbor",
    "lark", "meadow", "pixel", "quartz", "ridge", "tide", "vertex", "willow",
)


def generate_experiment_name(is_taken: Optional[Callable[[str], bool]] = None) -> str:
    """Return a fresh adjective-noun name, e.g. ``wl-brisk-otter``.

    ``is_taken`` — optional predicate used to retry on collision (e.g. against
    directory names already on disk, or experiment names already recorded in
    a manifest). Without it, a single name is returned unconditionally.
    """
    name = f"wl-{random.choice(_EXP_ADJECTIVES)}-{random.choice(_EXP_NOUNS)}"
    if is_taken is None:
        return name
    while is_taken(name):
        name = f"wl-{random.choice(_EXP_ADJECTIVES)}-{random.choice(_EXP_NOUNS)}"
    return name
