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


@pytest.fixture(scope="session", autouse=True)
def _no_automatic_studio():
    """Keep ``wl.serve()``'s automatic Weights Studio out of the suite.

    ``wl.serve()`` now serves the UI itself (weightslab/ui/embedded.py) and
    opens it in the browser. Every test that serves gRPC would otherwise bind a
    UI port and pop a browser tab on the developer's machine. Tests of the
    automatic UI turn it back on for themselves (monkeypatch).
    """
    previous = os.environ.get("WEIGHTSLAB_SERVE_UI")
    os.environ["WEIGHTSLAB_SERVE_UI"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WEIGHTSLAB_SERVE_UI", None)
        else:
            os.environ["WEIGHTSLAB_SERVE_UI"] = previous


# The test modules that spawn (fake) OpenCode servers.
_OPENCODE_SPAWNING_MODULES = (
    "test_opencode_process",
    "test_opencode_shared_server_integration",
    "test_server_agent",
)


@pytest.fixture(autouse=True)
def _private_opencode_machine_lock(request, monkeypatch):
    """Give each OpenCode-spawning test its own machine-wide lock.

    opencode_process publishes every server it spawns in
    ``~/.weightslab/.wl_opencode.json`` (the machine-wide rendezvous), and the
    fake servers these tests start are detached on purpose, so they outlive
    the test. Against the real home directory, the next test found the
    previous one's fake still healthy and adopted it instead of spawning
    (``'machine-lockfile' != 'spawned'``), and running the suite overwrote the
    developer's own lock with a test server's address.

    test_opencode_rendezvous isolates the home directory itself and asserts the
    real path layout, so it is left alone here.
    """
    if request.module.__name__.rsplit(".", 1)[-1] not in _OPENCODE_SPAWNING_MODULES:
        yield
        return
    from weightslab import opencode_process
    lock = request.getfixturevalue("tmp_path") / ".weightslab" / opencode_process.LOCK_FILENAME
    monkeypatch.setattr(opencode_process, "machine_lock_path", lambda: lock)
    yield


def _ledger_model():
    from weightslab.backend import ledgers
    try:
        model = ledgers.get_model()
    except Exception:
        return None
    try:
        return object.__getattribute__(model, "_obj")      # unwrap the ledger Proxy
    except AttributeError:
        return model


@pytest.fixture(scope="module", autouse=True)
def _no_model_leaks_between_modules():
    """Put the ledger's model back as each test module found it.

    ``ModelInterface`` registers itself in the global ledger (even with
    ``register=False``: the checkpoint auto-load needs it there), and a test
    that ends with that model in eval mode leaves it in *audit mode*.
    WeightsLab patches ``Tensor.backward`` to do nothing while the registered
    model is in audit mode -- so every later module in the process silently
    lost its gradients (tests/model/test_neuron_graph.py after the zoo
    sweep), and LoggerQueue tagged its signals ``audit_mode=True``
    (tests/model/test_logger.py after test_dependency_patterns.py). Order
    dependent, so green alone and red in CI.
    """
    before = _ledger_model()
    yield
    after = _ledger_model()
    if after is None or after is before:
        return
    from weightslab.backend import ledgers
    ledgers.GLOBAL_LEDGER.unregister_model()
    if before is not None:
        ledgers.register_model(before)
