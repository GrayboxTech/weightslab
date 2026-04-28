import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

import weightslab.trainer.trainer_services as trainer_services

# Default per-test timeout in seconds.  Override with WL_TEST_TIMEOUT env var.
import os
_TEST_TIMEOUT = int(os.getenv("WL_TEST_TIMEOUT", "30"))


class _TimeoutMixin:
    """Mixin that wraps every test with a hard timeout so stuck gRPC
    threads / infinite loops cannot block the whole CI run."""

    def run(self, result=None):
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(super().run, result)
        try:
            fut.result(timeout=_TEST_TIMEOUT)
        except FuturesTimeoutError:
            if result is not None:
                result.addError(self, (TimeoutError, TimeoutError(
                    f"Test timed out after {_TEST_TIMEOUT}s"), None))
        finally:
            pool.shutdown(wait=False)


class TestExperimentServiceServicerDelegation(_TimeoutMixin, unittest.TestCase):
    def test_servicer_delegates_to_subservices(self):
        exp_service = MagicMock()
        exp_service.model_service = MagicMock()
        exp_service.data_service = MagicMock()

        servicer = trainer_services.ExperimentServiceServicer(exp_service=exp_service)
        req = object()
        ctx = object()

        servicer.GetSamples(req, ctx)
        servicer.GetWeights(req, ctx)
        servicer.GetActivations(req, ctx)
        servicer.ApplyDataQuery(req, ctx)
        servicer.GetDataSamples(req, ctx)
        servicer.EditDataSample(req, ctx)
        servicer.GetDataSplits(req, ctx)
        servicer.CheckAgentHealth(req, ctx)
        servicer.GetLatestLoggerData(req, ctx)
        servicer.ExperimentCommand(req, ctx)
        servicer.ManipulateWeights(req, ctx)
        servicer.RestoreCheckpoint(req, ctx)

        exp_service.model_service.GetSamples.assert_called_once_with(req, ctx)
        exp_service.model_service.GetWeights.assert_called_once_with(req, ctx)
        exp_service.model_service.GetActivations.assert_called_once_with(req, ctx)
        exp_service.data_service.ApplyDataQuery.assert_called_once_with(req, ctx)
        exp_service.data_service.GetDataSamples.assert_called_once_with(req, ctx)
        exp_service.data_service.EditDataSample.assert_called_once_with(req, ctx)
        exp_service.data_service.GetDataSplits.assert_called_once_with(req, ctx)
        exp_service.data_service.CheckAgentHealth.assert_called_once_with(req, ctx)
        exp_service.GetLatestLoggerData.assert_called_once_with(req, ctx)
        exp_service.ExperimentCommand.assert_called_once_with(req, ctx)
        exp_service.model_service.ManipulateWeights.assert_called_once_with(req, ctx)
        exp_service.RestoreCheckpoint.assert_called_once_with(req, ctx)


class TestTlsPathResolution(_TimeoutMixin, unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_grpc_tls_path_defaults_to_user_home_certs_dir(self):
        resolved = trainer_services._resolve_grpc_tls_path(
            {},
            "grpc_tls_cert_file",
            "GRPC_TLS_CERT_FILE",
            "backend-server.crt",
        )

        expected = os.path.join(
            os.path.expanduser(os.path.join("~", "certs")),
            "backend-server.crt",
        )
        self.assertEqual(resolved, expected)

    @patch.dict("os.environ", {"GRPC_TLS_CERT_DIR": "~/custom-certs"}, clear=True)
    def test_resolve_grpc_tls_path_uses_shared_cert_dir_override(self):
        resolved = trainer_services._resolve_grpc_tls_path(
            {},
            "grpc_tls_key_file",
            "GRPC_TLS_KEY_FILE",
            "backend-server.key",
        )

        expected = os.path.join(
            os.path.expanduser("~/custom-certs"),
            "backend-server.key",
        )
        self.assertEqual(resolved, expected)

    @patch.dict("os.environ", {"GRPC_TLS_CERT_DIR": "~/env-certs"}, clear=True)
    def test_resolve_grpc_tls_path_prefers_config_dir_over_env_dir(self):
        resolved = trainer_services._resolve_grpc_tls_path(
            {"grpc_tls_cert_dir": "~/config-certs"},
            "grpc_tls_key_file",
            "GRPC_TLS_KEY_FILE",
            "backend-server.key",
        )

        expected = os.path.join(
            os.path.expanduser("~/config-certs"),
            "backend-server.key",
        )
        self.assertEqual(resolved, expected)

    @patch.dict("os.environ", {"GRPC_TLS_CA_FILE": "~/override/ca.pem"}, clear=True)
    def test_resolve_grpc_tls_path_prefers_config_file_over_env_file(self):
        resolved = trainer_services._resolve_grpc_tls_path(
            {"grpc_tls_ca_file": "~/from-config/ca.crt", "grpc_tls_cert_dir": "~/config-certs"},
            "grpc_tls_ca_file",
            "GRPC_TLS_CA_FILE",
            "ca.crt",
        )

        self.assertEqual(resolved, os.path.expanduser("~/from-config/ca.crt"))

    @patch.dict("os.environ", {"GRPC_TLS_CA_FILE": "~/override/ca.pem"}, clear=True)
    def test_resolve_grpc_tls_path_uses_env_file_when_config_file_missing(self):
        resolved = trainer_services._resolve_grpc_tls_path(
            {"grpc_tls_cert_dir": "~/config-certs"},
            "grpc_tls_ca_file",
            "GRPC_TLS_CA_FILE",
            "ca.crt",
        )

        self.assertEqual(resolved, os.path.expanduser("~/override/ca.pem"))

    def test_resolve_bool_setting_prefers_config_over_env(self):
        with patch.dict("os.environ", {"GRPC_TLS_ENABLED": "0"}, clear=True):
            self.assertTrue(
                trainer_services._resolve_bool_setting(
                    {"grpc_tls_enabled": True},
                    "grpc_tls_enabled",
                    "GRPC_TLS_ENABLED",
                    "1",
                )
            )

    @patch.dict("os.environ", {}, clear=True)
    def test_resolve_tls_client_auth_defaults_to_tls_enabled_when_undefined(self):
        self.assertTrue(trainer_services._resolve_tls_client_auth_setting({}, True))
        self.assertFalse(trainer_services._resolve_tls_client_auth_setting({}, False))

    @patch.dict("os.environ", {"GRPC_TLS_REQUIRE_CLIENT_AUTH": "0"}, clear=True)
    def test_resolve_tls_client_auth_uses_env_when_defined(self):
        self.assertFalse(trainer_services._resolve_tls_client_auth_setting({}, True))

    @patch.dict("os.environ", {"GRPC_TLS_REQUIRE_CLIENT_AUTH": "0"}, clear=True)
    def test_resolve_tls_client_auth_prefers_config_over_env(self):
        self.assertTrue(
            trainer_services._resolve_tls_client_auth_setting(
                {"grpc_tls_require_client_auth": True},
                False,
            )
        )


class TestGrpcServe(_TimeoutMixin, unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "GRPC_TLS_ENABLED": "1",
            "GRPC_TLS_CERT_FILE": "missing/backend-server.crt",
            "GRPC_TLS_KEY_FILE": "missing/backend-server.key",
            "GRPC_TLS_CA_FILE": "missing/ca.crt",
            "GRPC_TLS_REQUIRE_CLIENT_AUTH": "1",
        },
        clear=False,
    )
    def test_grpc_serve_fails_fast_when_tls_files_are_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            trainer_services.grpc_serve(
                n_workers_grpc=1,
                grpc_host="127.0.0.1",
                grpc_port=50099,
                force_parameters=True,
            )

        self.assertIn("GRPC_TLS_KEY_FILE", str(ctx.exception))

    @patch.dict("os.environ", {"GRPC_TLS_ENABLED": "0", "GRPC_AUTH_TOKEN": ""}, clear=False)
    def test_grpc_serve_starts_thread_and_server(self):
        fake_server = MagicMock()
        # add_insecure_port must return non-zero to indicate successful binding.
        fake_server.add_insecure_port.return_value = 50099

        # stop() returns an event-like object whose wait() returns True (clean stop).
        fake_stop_event = MagicMock()
        fake_stop_event.wait.return_value = True
        fake_server.stop.return_value = fake_stop_event

        # server_manager: should_restart() returns True on the first call so the
        # inner wait-loop exits immediately (no spin on time.sleep(0.5)).
        # time.sleep is mocked to raise _StopOuter on the first call, which happens
        # at time.sleep(2) after the graceful restart sequence, breaking the outer
        # ``while True`` before a second server is created.
        class _StopOuter(Exception):
            pass

        fake_server_manager = MagicMock()
        fake_server_manager.should_restart.return_value = True

        def _sleep_stop(_duration):
            raise _StopOuter("stop outer restart loop")

        fake_watchdog = MagicMock()
        fake_watchdog.server_manager = fake_server_manager
        fake_watchdog.rpc_state = MagicMock()

        class _InstantThread:
            """Mock Thread that runs the serving callback synchronously
            but skips the watchdog callback (infinite ``while True`` loop)."""
            def __init__(self, target=None, daemon=None, name=None):
                self._target = target
                self.daemon = daemon
                self.name = name
                self.ident = 12345

            def start(self):
                # Only execute the gRPC serving thread; the watchdog
                # contains an infinite loop that would block forever.
                if self._target is not None and self.name == "WL-gRPC_Server":
                    self._target()

        with patch("weightslab.trainer.trainer_services.grpc.server", return_value=fake_server), \
             patch("weightslab.trainer.trainer_services.pb2_grpc.add_ExperimentServiceServicer_to_server") as add_servicer, \
             patch("weightslab.trainer.trainer_services.Thread", _InstantThread), \
             patch("weightslab.trainer.trainer_services.WeighlabsWatchdog", return_value=fake_watchdog), \
             patch("weightslab.trainer.trainer_services.time.sleep", side_effect=_sleep_stop), \
             patch("weightslab.trainer.trainer_services.ExperimentServiceServicer") as servicer_cls:
            trainer_services.grpc_serve(n_workers_grpc=2, grpc_host="127.0.0.1", grpc_port=50099, force_parameters=True)

        servicer_cls.assert_called_once()
        add_servicer.assert_called_once()
        fake_server.add_insecure_port.assert_called_once_with("127.0.0.1:50099")
        fake_server.start.assert_called_once()
        # Graceful stop is now triggered by the watchdog restart loop, not wait_for_termination.
        fake_server.stop.assert_called_once()



    @patch.dict("os.environ", {"WEIGHTSLAB_DISABLE_WATCHDOGS": "1", "GRPC_TLS_ENABLED": "0", "GRPC_AUTH_TOKEN": ""}, clear=False)
    def test_grpc_serve_can_disable_watchdogs_via_env(self):
        fake_server = MagicMock()
        fake_server.add_insecure_port.return_value = 50099

        fake_stop_event = MagicMock()
        fake_stop_event.wait.return_value = True
        fake_server.stop.return_value = fake_stop_event

        class _StopOuter(Exception):
            pass

        fake_server_manager = MagicMock()
        fake_server_manager.should_restart.return_value = True

        def _sleep_stop(_duration):
            raise _StopOuter("stop outer restart loop")

        class _InstantThread:
            def __init__(self, target=None, daemon=None, name=None):
                self._target = target
                self.daemon = daemon
                self.name = name
                self.ident = 12345

            def start(self):
                if self._target is not None and self.name == "WL-gRPC_Server":
                    self._target()

        with patch("weightslab.trainer.trainer_services.grpc.server", return_value=fake_server), \
             patch("weightslab.trainer.trainer_services.pb2_grpc.add_ExperimentServiceServicer_to_server") as add_servicer, \
             patch("weightslab.trainer.trainer_services.Thread", _InstantThread), \
             patch("weightslab.trainer.trainer_services.GrpcServerManager", return_value=fake_server_manager), \
             patch("weightslab.trainer.trainer_services.WeighlabsWatchdog") as watchdog_cls, \
             patch("weightslab.trainer.trainer_services.time.sleep", side_effect=_sleep_stop), \
             patch("weightslab.trainer.trainer_services.ExperimentServiceServicer") as servicer_cls:
            trainer_services.grpc_serve(n_workers_grpc=2, grpc_host="127.0.0.1", grpc_port=50099, force_parameters=True)

        watchdog_cls.assert_not_called()
        servicer_cls.assert_called_once()
        add_servicer.assert_called_once()
        fake_server.add_insecure_port.assert_called_once_with("127.0.0.1:50099")
        fake_server.start.assert_called_once()
        fake_server.stop.assert_called_once()
if __name__ == "__main__":
    unittest.main()
