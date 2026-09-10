"""Shared pytest setup.

Keeps the suite out of the developer's own WeightsLab state.

``weightslab.utils.active_experiment`` records the active experiment directory
in a real per-user file (``~/.weightslab/active_experiment.json``) so a
training run started in another terminal lands in the experiment the UI
established. That is deliberate at runtime -- and poison in tests: a run of
this suite while a ``weightslab start`` was up resolved its root_log_dir into
that live experiment, found the config and checkpoints of whatever was running
there, and failed in setUp with an unrelated config (seen for real:
tests/gRPC/test_grpc_tag_operations.py loading a segmentation example's
hyperparameters).

So every test session gets its own throwaway state directory. Tests that
exercise the handoff itself set ``WEIGHTSLAB_STATE_DIR`` to their own temp
directory anyway; this only changes the default.
"""

import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolate_weightslab_state():
    previous = os.environ.get("WEIGHTSLAB_STATE_DIR")
    with tempfile.TemporaryDirectory(prefix="wl-test-state-") as state_dir:
        os.environ["WEIGHTSLAB_STATE_DIR"] = state_dir
        try:
            yield state_dir
        finally:
            if previous is None:
                os.environ.pop("WEIGHTSLAB_STATE_DIR", None)
            else:
                os.environ["WEIGHTSLAB_STATE_DIR"] = previous
