"""Keep only a run's best weight checkpoints, ranked by one signal.

Configured under ``checkpoint_manager`` in the hyperparameters::

    checkpoint_manager:
      keep_best:
        signal: metric-ACC   # the signal's name, as the Studio plots it
        k: 5                 # default 5
        mode: max            # max | min; default min when the name has "loss"

Only checkpoints the trainer dumps on its own rotate (every
``experiment_dump_to_train_steps_ratio`` steps, or on a state change). One saved
on purpose -- the Studio's save button, the agent, a direct
``CheckpointManager.save_model_checkpoint()`` call -- is never deleted, and
neither is a run's most recent checkpoint, which is where a restart resumes.

The pure ranking lives here; ``CheckpointManager._apply_keep_best`` reads the
config, the signal history and the files, and deletes what this returns.
"""

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class KeepBestPolicy:
    signal: str
    k: int
    mode: str  # "max" or "min"

    @classmethod
    def from_config(cls, manager_config: Optional[Mapping]) -> Optional["KeepBestPolicy"]:
        """The policy ``checkpoint_manager.keep_best`` asks for; None when unset.

        Raises ValueError when it is set but unusable, so the caller can say so
        instead of silently keeping every checkpoint.
        """
        raw = (manager_config or {}).get("keep_best")
        if not raw:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("keep_best must be a mapping with a 'signal' (and optionally 'k', 'mode')")
        signal = str(raw.get("signal") or "").strip()
        if not signal:
            raise ValueError("keep_best needs a 'signal' to rank checkpoints by")
        try:
            k = int(raw.get("k", 5))
        except (TypeError, ValueError):
            raise ValueError(f"keep_best 'k' must be an integer, got {raw.get('k')!r}")
        if k < 1:
            raise ValueError(f"keep_best 'k' must be at least 1, got {k}")
        mode = str(raw.get("mode") or ("min" if "loss" in signal.lower() else "max")).strip().lower()
        if mode not in ("max", "min"):
            raise ValueError(f"keep_best 'mode' must be 'max' or 'min', got {raw.get('mode')!r}")
        return cls(signal=signal, k=k, mode=mode)


def score_checkpoints(
    steps: Iterable[int],
    history: Iterable[Tuple[int, float]],
) -> Dict[int, Tuple[float, int]]:
    """``{step: (value, distance)}``: each checkpoint step scored by the signal's
    value at the nearest logged step.

    Evaluations and dumps rarely land on the same step (a dump fires right after
    the optimizer step; a loop may evaluate one step later), so an exact match
    would leave most checkpoints unranked. The nearest evaluation is the closest
    thing to "how good were these weights".

    A step with nothing logged at or after it is left out: its evaluation may
    still be coming, and a checkpoint deleted on a stale score cannot come back.
    """
    points = sorted(
        (int(step), float(value)) for step, value in history
        if value is not None and math.isfinite(float(value))
    )
    logged_steps = [step for step, _ in points]
    scores: Dict[int, Tuple[float, int]] = {}
    for step in steps:
        i = bisect_left(logged_steps, step)
        if i == len(points):
            continue
        after_step, after_value = points[i]
        best = (after_value, after_step - step)
        if i > 0:
            before_step, before_value = points[i - 1]
            if step - before_step < best[1]:
                best = (before_value, step - before_step)
        scores[step] = best
    return scores


def steps_to_prune(
    policy: KeepBestPolicy,
    rotating_steps: Collection[int],
    latest_step: Optional[int],
    history: Iterable[Tuple[int, float]],
) -> List[int]:
    """The rotating checkpoint steps outside the ``policy.k`` best.

    Ties go to the checkpoint closer to its evaluation, then to the later one.
    Never returns ``latest_step`` (the resume point) nor a step that cannot be
    scored yet.
    """
    scores = score_checkpoints(rotating_steps, history)
    sign = -1.0 if policy.mode == "max" else 1.0
    ranked = sorted(scores, key=lambda s: (sign * scores[s][0], scores[s][1], -s))
    return sorted(s for s in ranked[policy.k:] if s != latest_step)
