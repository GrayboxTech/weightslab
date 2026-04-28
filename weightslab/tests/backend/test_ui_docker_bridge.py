import argparse
import os
import unittest
import logging
from unittest.mock import patch, MagicMock

from weightslab.ui_docker_bridge import (
    _check_docker,
    _compose_cmd,
    main,
    ui_drop,
    ui_launch,
    ui_stop,
    ui_secure_environment,
    ui_docker_secure_environment,
    ui_launch_secure,
    docker_launch_secure,
    docker_info,
    docker_stop,
)


class TestCheckDocker(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.shutil.which", return_value=None)
    def test_exits_when_docker_not_found(self, _mock_which):
        with self.assertRaises(SystemExit) as ctx:
            _check_docker()
        self.assertEqual(ctx.exception.code, 1)

    @patch(
        "weightslab.ui_docker_bridge.subprocess.run",
        side_effect=__import__("subprocess").CalledProcessError(1, "docker info"),
    )
    @patch("weightslab.ui_docker_bridge.shutil.which", return_value="/usr/bin/docker")
    def test_exits_when_daemon_not_running(self, _mock_which, _mock_run):
        with self.assertRaises(SystemExit) as ctx:
            _check_docker()
        self.assertEqual(ctx.exception.code, 1)

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    @patch("weightslab.ui_docker_bridge.shutil.which", return_value="/usr/bin/docker")
    def test_passes_when_docker_available(self, _mock_which, mock_run):
        _check_docker()
        mock_run.assert_called_once()


class TestComposeCmd(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_runs_docker_compose_with_env(self, mock_run):
        _compose_cmd("/path/to/compose.yml", "/path/to/envoy.yaml", ["up", "-d"])
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(
            args[0],
            ["docker", "compose", "-f", "/path/to/compose.yml", "up", "-d"],
        )
        self.assertEqual(kwargs["env"]["WS_ENVOY_CONFIG"], "/path/to/envoy.yaml")
        self.assertTrue(kwargs["check"])


class TestUiLaunch(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_prints_url_default_port(self, _gc, _ge, mock_check, mock_compose):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        mock_check.assert_called_once()
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["up", "-d", "--pull", "always"],
        )
        self.assertTrue(any("https://localhost:5173" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_respects_custom_port(self, _gc, _ge, _mock_check, _mock_compose):
        with patch.dict(os.environ, {"VITE_PORT": "3000"}):
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        self.assertTrue(any("https://localhost:3000" in msg for msg in log_context.output))


class TestUiStop(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_stop_prints_message(self, _gc, _ge, _mock_check, mock_compose):
        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_stop(argparse.Namespace())
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["stop"],
        )
        self.assertTrue(any("Weights Studio UI stopped" in msg for msg in log_context.output))


class TestUiDrop(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_drop_prints_message(self, _gc, _ge, _mock_check, mock_compose):
        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_drop(argparse.Namespace())
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["down", "--rmi", "all"],
        )
        self.assertTrue(any("containers and images removed" in msg for msg in log_context.output))


class TestUiSecureEnvironment(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_ui_secure_environment_success(self, mock_gen_certs, mock_cert_manager):
        """Test successful secure environment setup."""
        from pathlib import Path
        from unittest.mock import MagicMock

        mock_manager_instance = MagicMock()
        # Make certs_dir a proper MagicMock that acts like a Path
        mock_certs_dir = MagicMock()
        mock_certs_dir.mkdir = MagicMock()
        mock_manager_instance.certs_dir = mock_certs_dir

        mock_manager_instance.get_or_create_auth_token.return_value = "fake_token"
        mock_manager_instance.setup_tls_environment.return_value = {"TLS_KEY": "/fake/key"}
        mock_manager_instance.setup_auth_environment.return_value = {"AUTH_TOKEN": "token123"}
        mock_cert_manager.return_value = mock_manager_instance

        args = argparse.Namespace(
            certs_dir="/fake/certs",
            no_auth=False,
            force_certs=False,
        )

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_secure_environment(args)

        self.assertTrue(any("Certificates generated successfully" in msg for msg in log_context.output))
        self.assertTrue(any("gRPC auth token created" in msg for msg in log_context.output))
        mock_gen_certs.assert_called_once_with(force_certs=False)
        mock_manager_instance.certs_dir.mkdir.assert_called_once()

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=1)
    def test_ui_secure_environment_cert_failure(self, mock_gen_certs):
        """Test secure environment setup failure."""
        args = argparse.Namespace(
            certs_dir="/fake/certs",
            no_auth=False,
            force_certs=False,
        )

        with self.assertRaises(SystemExit) as ctx:
            ui_secure_environment(args)
        self.assertEqual(ctx.exception.code, 1)


class TestUiDockerSecureEnvironment(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.ui_launch_secure")
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_docker_secure_environment_success(self, mock_gen_certs, mock_cert_manager, mock_launch):
        """Test Docker secure environment setup with successful certs."""
        mock_manager_instance = MagicMock()
        # certs_dir should be a Path-like object with mkdir method
        mock_certs_dir = MagicMock()
        mock_certs_dir.mkdir = MagicMock()
        mock_manager_instance.certs_dir = mock_certs_dir
        mock_manager_instance.setup_tls_environment.return_value = {"TLS_KEY": "/fake/key"}
        mock_manager_instance.setup_auth_environment.return_value = {"AUTH_TOKEN": "token123"}
        mock_cert_manager.from_env_or_default.return_value = mock_manager_instance

        args = argparse.Namespace(
            no_auth=False,
            force_certs=False,
            dev=False,
            test=False,
        )

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_docker_secure_environment(args)

        self.assertTrue(any("Certificates and auth token generated" in msg for msg in log_context.output))
        mock_launch.assert_called_once()

    @patch("weightslab.ui_docker_bridge.ui_launch_secure")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=1)
    def test_docker_secure_environment_cert_failure_fallback(self, mock_gen_certs, mock_launch):
        """Test Docker secure environment falls back to unsecured launch on cert failure."""
        args = argparse.Namespace(
            no_auth=False,
            force_certs=False,
            dev=False,
            test=False,
        )

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_docker_secure_environment(args)

        self.assertTrue(any("Certificate generation failed" in msg for msg in log_context.output))
        self.assertTrue(any("unsecured mode" in msg for msg in log_context.output))
        mock_launch.assert_called_once()


class TestUiLaunchSecure(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.Path")
    @patch("weightslab.ui_docker_bridge._run_powershell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script")
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_secure_with_valid_certs(self, mock_ui_launch, mock_cert_manager,
                                             mock_bootstrap_script, mock_ps_script, mock_path):
        """Test launching with secure certs on Windows."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.check_and_apply.return_value = (True, "Secure env ready")
        mock_cert_manager.from_env_or_default.return_value = mock_manager_instance
        mock_bootstrap_script.return_value = "/fake/bootstrap.ps1"

        # Mock Path.exists() to return True for the bootstrap script
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        args = argparse.Namespace(
            no_auth=False,
            dev=False,
            test=False,
            unsecure=False,
        )

        with patch("weightslab.ui_docker_bridge._is_windows", return_value=True):
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch_secure(args)

        self.assertTrue(any("✓ Secure environment configured" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge.ui_launch")
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    def test_launch_secure_with_unsecure_flag(self, mock_cert_manager, mock_ui_launch):
        """Test launching with --unsecure flag bypasses cert check."""
        args = argparse.Namespace(
            unsecure=True,
            no_auth=False,
            dev=False,
            test=False,
        )

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_launch_secure(args)

        self.assertTrue(any("Forcing unsecured mode" in msg for msg in log_context.output))
        mock_ui_launch.assert_called_once_with(args)

    @patch("weightslab.ui_docker_bridge.ui_launch")
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    def test_launch_secure_without_certs_fallback(self, mock_cert_manager, mock_ui_launch):
        """Test launching without certs falls back to unsecured mode."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.check_and_apply.return_value = (False, "Certs not found")
        mock_cert_manager.from_env_or_default.return_value = mock_manager_instance

        args = argparse.Namespace(
            unsecure=False,
            no_auth=False,
            dev=False,
            test=False,
        )

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_launch_secure(args)

        self.assertTrue(any("Secure certs not found" in msg for msg in log_context.output))
        self.assertTrue(any("falling back to unsecured mode" in msg for msg in log_context.output))
        mock_ui_launch.assert_called_once_with(args)


class TestDockerLaunchSecure(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.ui_launch_secure")
    def test_docker_launch_secure(self, mock_launch_secure):
        """Test docker_launch_secure delegates to ui_launch_secure."""
        args = argparse.Namespace(
            dev=False,
            no_auth=False,
            test=False,
        )
        docker_launch_secure(args)
        mock_launch_secure.assert_called_once_with(args)


class TestDockerStop(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.ui_stop")
    def test_docker_stop(self, mock_ui_stop):
        """Test docker_stop delegates to ui_stop."""
        args = argparse.Namespace()
        docker_stop(args)
        mock_ui_stop.assert_called_once_with(args)


class TestDockerInfo(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    def test_docker_info(self, mock_cert_manager):
        """Test docker_info displays configuration."""
        mock_manager_instance = MagicMock()
        mock_manager_instance.certs_dir = "/fake/certs"
        mock_manager_instance.has_valid_certs.return_value = True
        mock_manager_instance.enable_auth = True
        mock_manager_instance.setup_tls_environment.return_value = {"TLS_KEY": "/fake/key"}
        mock_manager_instance.setup_auth_environment.return_value = {"AUTH_TOKEN": "token123"}
        mock_cert_manager.from_env_or_default.return_value = mock_manager_instance

        args = argparse.Namespace()

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            docker_info(args)

        self.assertTrue(any("Weights Studio Docker Configuration" in msg for msg in log_context.output))
        self.assertTrue(any("/fake/certs" in msg for msg in log_context.output))




class TestMainCLI(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_main_dispatches_ui_launch(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch"]):
            main()
        mock_launch.assert_called_once()

    @patch("weightslab.ui_docker_bridge.ui_stop")
    def test_main_dispatches_ui_stop(self, mock_stop):
        with patch("sys.argv", ["weightslab", "ui", "stop"]):
            main()
        mock_stop.assert_called_once()

    @patch("weightslab.ui_docker_bridge.ui_drop")
    def test_main_dispatches_ui_drop(self, mock_drop):
        with patch("sys.argv", ["weightslab", "ui", "drop"]):
            main()
        mock_drop.assert_called_once()

    @patch("weightslab.ui_docker_bridge.ui_secure_environment")
    def test_main_dispatches_secure_environment(self, mock_se):
        """Test 'se' command (secured environment)."""
        with patch("sys.argv", ["weightslab", "se"]):
            main()
        mock_se.assert_called_once()

    @patch("weightslab.ui_docker_bridge.ui_docker_secure_environment")
    def test_main_dispatches_docker_secure_environment(self, mock_docker_se):
        """Test 'ui docker se' command."""
        with patch("sys.argv", ["weightslab", "ui", "docker", "se"]):
            main()
        mock_docker_se.assert_called_once()

    @patch("weightslab.ui_docker_bridge.docker_launch_secure")
    def test_main_dispatches_docker_launch(self, mock_launch):
        """Test 'ui docker launch' command."""
        with patch("sys.argv", ["weightslab", "ui", "docker", "launch"]):
            main()
        mock_launch.assert_called_once()

    @patch("weightslab.ui_docker_bridge.docker_stop")
    def test_main_dispatches_docker_stop(self, mock_stop):
        """Test 'ui docker stop' command."""
        with patch("sys.argv", ["weightslab", "ui", "docker", "stop"]):
            main()
        mock_stop.assert_called_once()

    @patch("weightslab.ui_docker_bridge.docker_info")
    def test_main_dispatches_docker_info(self, mock_info):
        """Test 'ui docker info' command."""
        with patch("sys.argv", ["weightslab", "ui", "docker", "info"]):
            main()
        mock_info.assert_called_once()

    def test_main_help_does_not_crash(self):
        with patch("sys.argv", ["weightslab", "help"]):
            main()  # should not raise

    def test_main_no_args_does_not_crash(self):
        with patch("sys.argv", ["weightslab"]):
            main()  # should not raise


if __name__ == "__main__":
    unittest.main()
