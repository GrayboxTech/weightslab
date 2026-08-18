"""Unit tests for weightslab.monitoring.resource_monitor.

Covers:
  - Config resolution: hardcoded defaults, env var overrides, YAML overriding env vars
  - ResourceMonitor category gating (only enabled categories get sampled/logged)
  - Individual psutil-backed samplers return the expected metric keys
  - GPU sampling degrades gracefully with no NVIDIA driver present
  - start()/stop() lifecycle of the background thread
  - Process-wide singleton helpers
"""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from weightslab.monitoring.resource_monitor import (
    DEFAULT_CATEGORIES,
    DEFAULT_STEP_SOURCE,
    ResourceMonitor,
    load_resource_monitoring_config,
    start_resource_monitor_from_config,
    get_resource_monitor,
    stop_resource_monitor,
)

_ENV_KEYS = (
    "WEIGHTSLAB_DISABLE_RESOURCE_MONITORING",
    "WL_RESOURCE_MONITOR_INTERVAL_SECONDS",
    "WL_RESOURCE_MONITOR_CATEGORIES",
    "WL_RESOURCE_MONITOR_DISK_PATH",
    "WL_RESOURCE_MONITOR_CONFIG_PATH",
    "WL_RESOURCE_MONITOR_STEP_SOURCE",
)


class TestResourceMonitorStepAxis(unittest.TestCase):
    """The x value samples are logged against.

    Resource curves used to be plotted against the monitor's own uptime in
    seconds, so restarting training from step 0 left them running on from
    wherever they had reached -- on an axis no other plot shared. They now follow
    the model's age.
    """

    def _monitor(self, **kwargs):
        cats = {name: False for name in DEFAULT_CATEGORIES}
        cats["cpu"] = True
        monitor = ResourceMonitor(interval_seconds=1.0, categories=cats, **kwargs)
        monitor._start_monotonic = 0.0
        return monitor

    def _steps_for_ages(self, monitor, ages):
        """Steps logged for a model reporting each age in turn."""
        steps = []
        monitor._log_metric = lambda reg_name, value, step: steps.append(step)
        for age in ages:
            monitor._model_age = lambda age=age: age
            monitor._tick()
        return steps

    def test_step_follows_model_age(self):
        monitor = self._monitor()
        self.assertEqual(self._steps_for_ages(monitor, [0, 1, 2]), [0, 1, 2])

    def test_restarted_training_restarts_the_curve_at_zero(self):
        # The reported bug: after a restart the age goes back to 0 and the
        # resource curve has to follow it there instead of carrying on.
        monitor = self._monitor()
        self.assertEqual(self._steps_for_ages(monitor, [40, 41, 0, 1]), [40, 41, 0, 1])

    def test_a_step_that_has_not_moved_is_sampled_once(self):
        # Paused training, or two steps further apart than the sampling
        # interval: repeating the same x would stack points instead of
        # extending the curve.
        monitor = self._monitor()
        self.assertEqual(self._steps_for_ages(monitor, [7, 7, 7, 8]), [7, 8])

    def test_samples_before_any_model_land_on_the_baseline_step(self):
        # The monitor starts with the gRPC server, before a model is registered:
        # one baseline point at 0 rather than a pile of them.
        monitor = self._monitor()
        logged = []
        monitor._log_metric = lambda reg_name, value, step: logged.append(step)
        monitor._model_age = lambda: None
        monitor._tick()
        monitor._tick()
        self.assertEqual(logged, [0])

    def test_model_age_lookup_survives_a_raising_ledger(self):
        # The ledger proxy raises for a missing attribute rather than returning
        # None (and get_model itself can raise before registration).
        monitor = self._monitor()
        with mock.patch("weightslab.backend.ledgers.get_model", side_effect=RuntimeError("no model")):
            self.assertIsNone(monitor._model_age())

    def test_seconds_mode_keeps_the_old_uptime_axis(self):
        monitor = self._monitor(step_source="seconds")
        logged = []
        monitor._log_metric = lambda reg_name, value, step: logged.append(step)
        # Never consults the model, and repeats are not de-duplicated: the clock
        # advancing is what makes each sample distinct.
        monitor._model_age = lambda: (_ for _ in ()).throw(AssertionError("model asked in seconds mode"))
        with mock.patch("time.monotonic", return_value=12.4):
            monitor._tick()
            monitor._tick()
        self.assertEqual(logged, [12, 12])

    def test_an_unknown_step_source_falls_back_instead_of_raising(self):
        monitor = self._monitor(step_source="fortnights")
        self.assertEqual(monitor._step_source, DEFAULT_STEP_SOURCE)


class TestLoadResourceMonitoringConfig(unittest.TestCase):

    def setUp(self):
        # Point config resolution at an empty directory so no real
        # resource_monitoring.yaml (repo root, cwd, ...) leaks into these tests.
        self._empty_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._empty_dir.cleanup)

    def _patched_env(self, **overrides):
        # The loader falls back to Path.cwd()/resource_monitoring.yaml (same
        # design as agent_config.yaml's lookup chain) -- patch cwd too so the
        # real repo-root config (pytest's cwd) can't leak into these tests.
        env = {"WL_RESOURCE_MONITOR_CONFIG_PATH": self._empty_dir.name}
        env.update(overrides)
        return self._enter_isolated(env)

    def _enter_isolated(self, env):
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, env, clear=False))
        stack.enter_context(mock.patch("pathlib.Path.cwd", return_value=Path(self._empty_dir.name)))
        return stack

    def test_defaults_when_no_overrides(self):
        with self._patched_env():
            for key in _ENV_KEYS:
                os.environ.pop(key, None)
            os.environ["WL_RESOURCE_MONITOR_CONFIG_PATH"] = self._empty_dir.name
            config = load_resource_monitoring_config()

        self.assertTrue(config["enabled"])
        self.assertEqual(config["interval_seconds"], 15.0)
        self.assertEqual(set(config["categories"]), set(DEFAULT_CATEGORIES))
        self.assertTrue(all(config["categories"].values()))

    def test_env_var_disables_monitoring(self):
        with self._patched_env(WEIGHTSLAB_DISABLE_RESOURCE_MONITORING="1"):
            config = load_resource_monitoring_config()
        self.assertFalse(config["enabled"])

    def test_step_source_defaults_to_model_age(self):
        with self._patched_env():
            self.assertEqual(load_resource_monitoring_config()["step_source"], "model_age")

    def test_env_var_selects_the_legacy_seconds_axis(self):
        with self._patched_env(WL_RESOURCE_MONITOR_STEP_SOURCE="seconds"):
            self.assertEqual(load_resource_monitoring_config()["step_source"], "seconds")

    def test_unknown_step_source_falls_back_to_the_default(self):
        with self._patched_env(WL_RESOURCE_MONITOR_STEP_SOURCE="epochs"):
            self.assertEqual(load_resource_monitoring_config()["step_source"], DEFAULT_STEP_SOURCE)

    def test_env_var_restricts_categories(self):
        with self._patched_env(WL_RESOURCE_MONITOR_CATEGORIES="cpu,gpu"):
            config = load_resource_monitoring_config()
        self.assertTrue(config["categories"]["cpu"])
        self.assertTrue(config["categories"]["gpu"])
        self.assertFalse(config["categories"]["memory"])
        self.assertFalse(config["categories"]["disk"])
        self.assertFalse(config["categories"]["network"])
        self.assertFalse(config["categories"]["process"])

    def test_yaml_overrides_env_vars(self):
        yaml_dir = tempfile.TemporaryDirectory()
        self.addCleanup(yaml_dir.cleanup)
        yaml_path = Path(yaml_dir.name) / "resource_monitoring.yaml"
        yaml_path.write_text(
            "resource_monitoring:\n"
            "  enabled: false\n"
            "  interval_seconds: 5\n"
            "  disk_path: /data\n"
            "  categories:\n"
            "    cpu: false\n"
        )

        # Env says "enabled" but YAML (which takes precedence) says disabled.
        with self._patched_env(
            WEIGHTSLAB_DISABLE_RESOURCE_MONITORING="0",
            WL_RESOURCE_MONITOR_CONFIG_PATH=yaml_dir.name,
        ):
            config = load_resource_monitoring_config()

        self.assertFalse(config["enabled"])
        self.assertEqual(config["interval_seconds"], 5.0)
        self.assertEqual(config["disk_path"], "/data")
        self.assertFalse(config["categories"]["cpu"])
        # Categories not mentioned in the YAML keep their (default) value.
        self.assertTrue(config["categories"]["memory"])


class TestResourceMonitorTick(unittest.TestCase):

    def _monitor(self, categories):
        return ResourceMonitor(interval_seconds=1.0, categories=categories)

    def test_tick_only_logs_enabled_categories(self):
        categories = {name: False for name in DEFAULT_CATEGORIES}
        categories["cpu"] = True
        monitor = self._monitor(categories)
        monitor._start_monotonic = 0.0

        logged = []
        monitor._log_metric = lambda reg_name, value, step: logged.append(reg_name)

        monitor._tick()

        self.assertTrue(logged, "expected at least one metric to be logged")
        self.assertTrue(all(name.startswith("resource/cpu/") for name in logged))

    def test_tick_skips_gpu_when_unavailable(self):
        categories = {name: True for name in DEFAULT_CATEGORIES}
        monitor = self._monitor(categories)
        monitor._start_monotonic = 0.0
        monitor._gpu_available = False # simulate "no NVIDIA driver"

        logged = []
        monitor._log_metric = lambda reg_name, value, step: logged.append(reg_name)

        monitor._tick()

        self.assertFalse(any(name.startswith("resource/gpu/") for name in logged))

    def test_log_metric_delegates_to_signal_logger(self):
        monitor = self._monitor({name: True for name in DEFAULT_CATEGORIES})
        with mock.patch("weightslab.src._log_signal") as mock_log_signal:
            monitor._log_metric("resource/cpu/utilization_percent", 42.0, 7)
        mock_log_signal.assert_called_once_with(
            42.0, None, "resource/cpu/utilization_percent", step=7
        )

    def test_log_metric_swallows_exceptions(self):
        monitor = self._monitor({name: True for name in DEFAULT_CATEGORIES})
        with mock.patch("weightslab.src._log_signal", side_effect=RuntimeError("boom")):
            monitor._log_metric("resource/cpu/utilization_percent", 42.0, 7) # must not raise


class TestResourceMonitorSamplers(unittest.TestCase):

    def setUp(self):
        self.monitor = ResourceMonitor(interval_seconds=1.0)

    def test_sample_cpu_keys(self):
        metrics = self.monitor._sample_cpu()
        self.assertIn("resource/cpu/utilization_percent", metrics)
        self.assertIsInstance(metrics["resource/cpu/utilization_percent"], float)

    def test_sample_memory_keys(self):
        metrics = self.monitor._sample_memory()
        self.assertIn("resource/memory/system_utilization_percent", metrics)

    def test_sample_process_keys(self):
        metrics = self.monitor._sample_process()
        for key in (
            "resource/process/cpu_utilization_percent",
            "resource/process/cpu_threads_in_use",
            "resource/process/memory_in_use_mb",
            "resource/process/memory_in_use_percent",
            "resource/process/memory_available_mb",
        ):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)
        self.assertGreater(metrics["resource/process/cpu_threads_in_use"], 0)

    def test_sample_disk_keys(self):
        metrics = self.monitor._sample_disk()
        # At least the usage percent/GB pair should always resolve on any OS.
        self.assertIn("resource/disk/utilization_percent", metrics)
        self.assertIn("resource/disk/utilization_gb", metrics)

    def test_sample_network_keys(self):
        metrics = self.monitor._sample_network()
        if metrics: # some sandboxed/CI environments expose no network counters
            self.assertIn("resource/network/bytes_sent", metrics)
            self.assertIn("resource/network/bytes_received", metrics)

    def test_init_gpu_never_raises_without_driver(self):
        # No assertion on the outcome (depends on the test machine's hardware) --
        # only that a missing/absent NVIDIA driver degrades gracefully.
        self.monitor._init_gpu()
        self.assertIsInstance(self.monitor._gpu_available, bool)
        if not self.monitor._gpu_available:
            self.assertEqual(self.monitor._gpu_handles, [])
        self.monitor._shutdown_gpu()
        self.assertFalse(self.monitor._gpu_available)


class TestResourceMonitorLifecycle(unittest.TestCase):

    def test_start_stop_clean_thread_teardown(self):
        monitor = ResourceMonitor(
            interval_seconds=1.0,
            categories={name: False for name in DEFAULT_CATEGORIES},
        )
        monitor.start()
        try:
            self.assertIsInstance(monitor._thread, threading.Thread)
            self.assertTrue(monitor._thread.is_alive())
        finally:
            monitor.stop()
        self.assertFalse(monitor._thread.is_alive())


class TestResourceMonitorSingleton(unittest.TestCase):

    def tearDown(self):
        stop_resource_monitor()

    def test_start_from_config_returns_none_when_disabled(self):
        with mock.patch(
            "weightslab.monitoring.resource_monitor.load_resource_monitoring_config",
            return_value={
                "enabled": False,
                "interval_seconds": 15.0,
                "categories": {name: True for name in DEFAULT_CATEGORIES},
                "disk_path": os.sep,
            },
        ):
            monitor = start_resource_monitor_from_config()
        self.assertIsNone(monitor)
        self.assertIsNone(get_resource_monitor())

    def test_start_from_config_is_idempotent(self):
        with mock.patch(
            "weightslab.monitoring.resource_monitor.load_resource_monitoring_config",
            return_value={
                "enabled": True,
                "interval_seconds": 1.0,
                "categories": {name: False for name in DEFAULT_CATEGORIES},
                "disk_path": os.sep,
            },
        ):
            first = start_resource_monitor_from_config()
            second = start_resource_monitor_from_config()

        self.assertIsNotNone(first)
        self.assertIs(first, second)
        self.assertIs(get_resource_monitor(), first)

        stop_resource_monitor()
        self.assertIsNone(get_resource_monitor())


if __name__ == "__main__":
    unittest.main()
