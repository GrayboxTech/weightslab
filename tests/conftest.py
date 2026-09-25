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


@pytest.fixture(scope="session", autouse=True)
def _disable_resource_monitor():
    """Keep the real resource monitor out of the suite.

    ``grpc_serve`` ends by calling ``start_resource_monitor_from_config()``, and
    none of the tests that exercise it stub that out -- so the first such test
    starts a REAL ResourceMonitor: a process-wide singleton whose sampling
    thread then polls CPU/memory/disk/network/GPU for the rest of the session,
    logging as it goes. Nothing stops it, because the singleton makes every
    later call a no-op that returns the already-running instance.

    That background load is not free on a 2-core CI runner. It showed up as
    tests/trainer/services/test_trainer_services_server.py's
    test_grpc_serve_honors_explicit_port_without_force_parameters timing out
    against its 30s cap -- while the SAME commit passed on the push run
    (ac405334: PR run 35738307562 failed, push run 35738302340 passed), which
    is the signature of contention rather than a real defect.

    The monitor's own tests are unaffected: they patch
    ``load_resource_monitoring_config`` directly, or set this same variable
    inside their own ``_patched_env``.
    """
    previous = os.environ.get("WEIGHTSLAB_DISABLE_RESOURCE_MONITORING")
    os.environ["WEIGHTSLAB_DISABLE_RESOURCE_MONITORING"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WEIGHTSLAB_DISABLE_RESOURCE_MONITORING", None)
        else:
            os.environ["WEIGHTSLAB_DISABLE_RESOURCE_MONITORING"] = previous
