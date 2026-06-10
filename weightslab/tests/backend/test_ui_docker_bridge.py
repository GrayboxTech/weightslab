import argparse
import contextlib
import io
import os
import unittest
from unittest.mock import patch, MagicMock

from weightslab.ui_docker_bridge import (
    _check_docker,
    _clean_stale_docker_resources,
    _compose_cmd,
    _ensure_certificates,
    _generate_certs_with_fallback,
    _remove_docker_image,
    _strip_derived_deploy_env,
    _DERIVED_DEPLOY_ENV_VARS,
    _FRONTEND_IMAGE,
    _STACK_CONTAINERS,
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
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        _compose_cmd("/path/to/compose.yml", "/path/to/envoy.yaml", ["up", "-d"])
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(
            args[0],
            ["docker", "compose", "-f", "/path/to/compose.yml", "up", "-d"],
        )
        self.assertEqual(kwargs["env"]["WS_ENVOY_CONFIG"], "/path/to/envoy.yaml")
        self.assertTrue(kwargs["stdout"])
        self.assertTrue(kwargs["text"])


class TestUiLaunch(unittest.TestCase):
    """ui_launch: auto-generate certs (unless --no-certs), clean stale Docker, launch."""

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_default_generates_certs_cleans_and_launches(
        self, _gc, _ge, mock_check, mock_compose, mock_clean, mock_ensure,
        _mock_shell, _gb, mock_mgr,
    ):
        mock_mgr.from_env_or_default.return_value = MagicMock(certs_dir="/fake/certs")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        mock_check.assert_called_once()
        mock_ensure.assert_called_once()              # certs ensured by default
        mock_clean.assert_called_once()               # stale cleanup ran
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["up", "-d", "--pull", "always"],
        )
        self.assertTrue(any("https://localhost:5173" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_respects_custom_port(
        self, _gc, _ge, _mock_check, _mock_compose, _mock_clean, _mock_ensure,
        _mock_shell, _gb, mock_mgr,
    ):
        mock_mgr.from_env_or_default.return_value = MagicMock(certs_dir="/fake/certs")
        with patch.dict(os.environ, {"VITE_PORT": "3000"}):
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        self.assertTrue(any("https://localhost:3000" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_no_certs_skips_cert_gen_and_runs_unsecured(
        self, _gc, _ge, _mock_check, _mock_compose, _mock_clean, mock_ensure,
        mock_shell, mock_mgr,
    ):
        # Uses the real (existing) bootstrap path so the --unsecure arg path runs.
        # certs_dir is a MagicMock so the --no-certs mkdir (for a valid mount) works.
        mock_mgr.from_env_or_default.return_value = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace(no_certs=True))
        mock_ensure.assert_not_called()
        # bootstrap invoked with --unsecure, and a NON-empty certs dir for the mount
        self.assertIsNotNone(mock_shell.call_args)
        script_args = mock_shell.call_args.args[1]
        bootstrap_env = mock_shell.call_args.args[2]
        self.assertIn("--unsecure", script_args)
        self.assertTrue(bootstrap_env["WEIGHTSLAB_CERTS_DIR"],
                        "certs dir must be non-empty so the bind-mount has a real source")
        self.assertTrue(any("http://localhost:5173" in msg for msg in log_context.output))
        self.assertFalse(any("https://localhost:5173" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_no_clean_skips_cleanup(
        self, _gc, _ge, _mock_check, _mock_compose, mock_clean, _mock_ensure,
        _mock_shell, _gb, mock_mgr,
    ):
        mock_mgr.from_env_or_default.return_value = MagicMock(certs_dir="/fake/certs")
        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_launch(argparse.Namespace(no_clean=True))
        mock_clean.assert_not_called()
        self.assertTrue(any("Skipping stale Docker resource cleanup" in msg for msg in log_context.output))


class TestEnsureCertificates(unittest.TestCase):
    """_ensure_certificates only generates files; it never exports TLS/auth env."""

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback")
    def test_uses_existing_certs_without_generating(self, mock_gen):
        manager = MagicMock()
        manager.has_valid_certs.return_value = True
        result = _ensure_certificates(manager, force_certs=False)
        self.assertTrue(result)
        mock_gen.assert_not_called()
        manager.get_or_create_auth_token.assert_called_once()
        # Derived TLS env must NOT be set here — the deploy pipeline derives it.
        manager.setup_tls_environment.assert_not_called()

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_generates_when_missing_and_forwards_certs_dir(self, mock_gen):
        manager = MagicMock()
        # Missing at the gate, present after generation.
        manager.has_valid_certs.side_effect = [False, True]
        result = _ensure_certificates(manager, force_certs=False)
        self.assertTrue(result)
        mock_gen.assert_called_once_with(force_certs=False, certs_dir=manager.certs_dir)
        manager.setup_tls_environment.assert_not_called()

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_force_regenerates_even_when_present(self, mock_gen):
        manager = MagicMock()
        manager.has_valid_certs.return_value = True
        _ensure_certificates(manager, force_certs=True)
        mock_gen.assert_called_once_with(force_certs=True, certs_dir=manager.certs_dir)

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=1)
    def test_returns_false_on_generation_failure(self, mock_gen):
        manager = MagicMock()
        manager.has_valid_certs.return_value = False
        result = _ensure_certificates(manager, force_certs=False)
        self.assertFalse(result)


class TestRemoveDockerImage(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_removes_when_present(self, mock_run):
        mock_run.side_effect = [MagicMock(stdout="abc123\nabc123\ndef456\n"), MagicMock()]
        _remove_docker_image(_FRONTEND_IMAGE)
        self.assertEqual(mock_run.call_count, 2)
        rmi_call = mock_run.call_args_list[1].args[0]
        self.assertEqual(rmi_call[:3], ["docker", "rmi", "-f"])
        self.assertIn("abc123", rmi_call)
        self.assertIn("def456", rmi_call)

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_noop_when_absent(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        _remove_docker_image(_FRONTEND_IMAGE)
        mock_run.assert_called_once()  # only the 'docker images -q' query, no rmi


class TestCleanStaleDockerResources(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge._remove_docker_image")
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_clean_tears_down_and_removes_image(self, _gc, _ge, mock_compose, mock_run, mock_rmimg):
        _clean_stale_docker_resources()
        # 1. compose down --remove-orphans --volumes
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["down", "--remove-orphans", "--volumes"],
        )
        # 2. docker rm -f for each known stack container
        removed = [c.args[0] for c in mock_run.call_args_list]
        for container in _STACK_CONTAINERS:
            self.assertTrue(
                any(call[:3] == ["docker", "rm", "-f"] and container in call for call in removed),
                f"expected 'docker rm -f {container}'",
            )
        # 3. cached frontend image removed
        mock_rmimg.assert_called_once_with(_FRONTEND_IMAGE)

    @patch("weightslab.ui_docker_bridge._remove_docker_image")
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    @patch("weightslab.ui_docker_bridge._compose_cmd",
           side_effect=__import__("subprocess").CalledProcessError(1, "down"))
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_clean_tolerates_compose_down_failure(self, _gc, _ge, _mc, _mr, mock_rmimg):
        # Should not raise even when 'compose down' returns non-zero (nothing to remove).
        _clean_stale_docker_resources()
        mock_rmimg.assert_called_once_with(_FRONTEND_IMAGE)


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
        # Generation is pointed at the chosen certs dir (single source of truth).
        mock_gen_certs.assert_called_once_with(force_certs=False, certs_dir=mock_manager_instance.certs_dir)
        mock_manager_instance.certs_dir.mkdir.assert_called_once()
        # se exports WEIGHTSLAB_CERTS_DIR for the process.
        self.assertTrue(any("WEIGHTSLAB_CERTS_DIR exported" in msg for msg in log_context.output))

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
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge.Path")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script")
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_secure_with_valid_certs(self, mock_ui_launch, mock_cert_manager,
                                             mock_bootstrap_script, mock_path, mock_run_shell):
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


class TestUserOnboardingFlow(unittest.TestCase):
    """Integration-like tests for the user onboarding flow."""

    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    def test_complete_onboarding_workflow(self, mock_cert_manager, mock_gen_certs,
                                          mock_run_shell, mock_check, mock_compose):
        """A complete onboarding flow: setup -> launch -> check -> stop, hermetic.

        Low-level Docker (`_compose_cmd`, `_check_docker`) and the bootstrap
        (`_run_shell_script`) are mocked so nothing touches a real daemon.
        """
        mock_manager_instance = MagicMock()  # certs_dir is a MagicMock (supports .mkdir)
        mock_manager_instance.check_and_apply.return_value = (True, "Secure env ready")
        mock_cert_manager.return_value = mock_manager_instance
        mock_cert_manager.from_env_or_default.return_value = mock_manager_instance

        args_se = argparse.Namespace(certs_dir=None, no_auth=False, force_certs=False)
        args_launch = argparse.Namespace(
            no_auth=False, force_certs=False, dev=False, test=False, unsecure=False)
        args_info = argparse.Namespace()
        args_stop = argparse.Namespace()

        # Isolate os.environ — `se` exports WEIGHTSLAB_CERTS_DIR for the process.
        with patch.dict(os.environ, {}, clear=False):
            try:
                ui_secure_environment(args_se)
                docker_launch_secure(args_launch)
                docker_info(args_info)
                docker_stop(args_stop)
            except Exception as e:
                self.fail(f"Onboarding workflow failed: {e}")

    @patch("sys.argv", ["weightslab", "se"])
    @patch("weightslab.ui_docker_bridge.ui_secure_environment")
    def test_cli_secure_environment_command(self, mock_se):
        """Test CLI command: weightslab se"""
        main()
        mock_se.assert_called_once()

    @patch("sys.argv", ["weightslab", "ui", "docker", "launch", "--test"])
    @patch("weightslab.ui_docker_bridge.docker_launch_secure")
    def test_cli_docker_launch_with_test(self, mock_launch):
        """Test CLI command: weightslab ui docker launch --test"""
        main()
        mock_launch.assert_called_once()
        # Verify --test flag was passed
        call_args = mock_launch.call_args[0][0]
        self.assertTrue(call_args.test)

    @patch("sys.argv", ["weightslab", "ui", "docker", "info"])
    @patch("weightslab.ui_docker_bridge.docker_info")
    def test_cli_docker_info_command(self, mock_info):
        """Test CLI command: weightslab ui docker info"""
        main()
        mock_info.assert_called_once()

    @patch("sys.argv", ["weightslab", "ui", "docker", "launch", "--unsecure"])
    @patch("weightslab.ui_docker_bridge.docker_launch_secure")
    def test_cli_docker_launch_unsecure(self, mock_launch):
        """Test CLI command: weightslab ui docker launch --unsecure"""
        main()
        mock_launch.assert_called_once()
        call_args = mock_launch.call_args[0][0]
        self.assertTrue(call_args.unsecure)

    @patch("sys.argv", ["weightslab", "ui", "docker", "se", "--force-certs"])
    @patch("weightslab.ui_docker_bridge.ui_docker_secure_environment")
    def test_cli_docker_secure_environment_with_force_certs(self, mock_docker_se):
        """Test CLI command: weightslab ui docker se --force-certs"""
        main()
        mock_docker_se.assert_called_once()
        call_args = mock_docker_se.call_args[0][0]
        self.assertTrue(call_args.force_certs)


class TestBackendConnectionDetection(unittest.TestCase):
    """Test backend connection detection functionality."""

    @patch("socket.socket")
    def test_backend_connection_success(self, mock_socket_class):
        """Test successful backend connection detection."""
        from weightslab.ui_docker_bridge import _test_backend_connection

        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket

        result = _test_backend_connection()
        self.assertTrue(result)

    @patch("socket.socket")
    def test_backend_connection_failure(self, mock_socket_class):
        """Test failed backend connection detection."""
        from weightslab.ui_docker_bridge import _test_backend_connection

        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 1  # Connection refused
        mock_socket_class.return_value = mock_socket

        result = _test_backend_connection()
        self.assertFalse(result)

    @patch("socket.socket")
    def test_backend_connection_timeout(self, mock_socket_class):
        """Test backend connection timeout handling."""
        from weightslab.ui_docker_bridge import _test_backend_connection

        mock_socket = MagicMock()
        mock_socket.connect_ex.side_effect = Exception("Connection timeout")
        mock_socket_class.return_value = mock_socket

        result = _test_backend_connection()
        self.assertFalse(result)

    @patch("socket.socket")
    def test_backend_connection_with_custom_host_port(self, mock_socket_class):
        """Test backend connection with custom host and port."""
        from weightslab.ui_docker_bridge import _test_backend_connection

        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket

        result = _test_backend_connection(host='192.168.1.1', port=8080, timeout=10.0)
        self.assertTrue(result)
        mock_socket.connect_ex.assert_called_once_with(('192.168.1.1', 8080))
        mock_socket.settimeout.assert_called_once_with(10.0)


class TestPathConversion(unittest.TestCase):
    """Test Windows path to Git Bash conversion."""

    def test_windows_path_conversion(self):
        """Test converting Windows path to Git Bash format."""
        from weightslab.ui_docker_bridge import _convert_to_git_bash_path

        # Test Windows path
        win_path = r"C:\Users\testuser\.weightslab-certs"
        bash_path = _convert_to_git_bash_path(win_path)
        self.assertEqual(bash_path, "/mnt/c/Users/testuser/.weightslab-certs")

    def test_unix_path_passthrough(self):
        """Test Unix paths pass through unchanged."""
        from weightslab.ui_docker_bridge import _convert_to_git_bash_path

        unix_path = "/home/testuser/.weightslab-certs"
        bash_path = _convert_to_git_bash_path(unix_path)
        self.assertEqual(bash_path, "/home/testuser/.weightslab-certs")


class TestSingleSourceOfTruth(unittest.TestCase):
    """WEIGHTSLAB_CERTS_DIR is the only input; everything else derives from it."""

    def test_strip_derived_deploy_env_removes_all(self):
        sentinel = {k: "stale" for k in _DERIVED_DEPLOY_ENV_VARS}
        with patch.dict(os.environ, sentinel, clear=False):
            _strip_derived_deploy_env()
            for key in _DERIVED_DEPLOY_ENV_VARS:
                self.assertNotIn(key, os.environ, f"{key} should have been stripped")

    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    def test_generate_certs_forwards_certs_dir(self, mock_shell):
        rc = _generate_certs_with_fallback(force_certs=False, certs_dir="/custom/certs")
        self.assertEqual(rc, 0)
        # The certs dir is forwarded to the generation script as WEIGHTSLAB_CERTS_DIR.
        env_vars = mock_shell.call_args.args[2]
        self.assertIsInstance(env_vars, dict)
        self.assertIn("WEIGHTSLAB_CERTS_DIR", env_vars)
        self.assertIn("custom/certs", env_vars["WEIGHTSLAB_CERTS_DIR"].replace("\\", "/"))

    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    def test_generate_certs_without_dir_passes_no_env(self, mock_shell):
        # No certs_dir -> scripts fall back to their ~/.weightslab-certs default.
        _generate_certs_with_fallback(force_certs=False)
        self.assertIsNone(mock_shell.call_args.args[2])

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_ui_launch_strips_derived_env(
        self, _gc, _ge, _mock_check, _mock_compose, _mock_clean, _mock_ensure, _gb, mock_mgr,
    ):
        mock_mgr.from_env_or_default.return_value = MagicMock(certs_dir="/fake/certs")
        # Simulate a stale derived env var leaking in (e.g. from import-time check).
        with patch.dict(os.environ, {"ENVOY_DOWNSTREAM_TLS": "on", "VITE_SERVER_PROTOCOL": "https"}, clear=False):
            ui_launch(argparse.Namespace())
            self.assertNotIn("ENVOY_DOWNSTREAM_TLS", os.environ)
            self.assertNotIn("VITE_SERVER_PROTOCOL", os.environ)

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_ui_launch_url_https_only_when_certs_present(
        self, _gc, _ge, _mock_check, _mock_compose, _mock_clean, _mock_ensure, _gb, mock_mgr,
    ):
        # Certs absent on disk -> URL must be http even though we didn't pass --no-certs.
        mgr = MagicMock(certs_dir="/fake/certs")
        mgr.has_valid_certs.return_value = False
        mock_mgr.from_env_or_default.return_value = mgr
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        self.assertTrue(any("http://localhost:5173" in m for m in log_context.output))
        self.assertFalse(any("https://localhost:5173" in m for m in log_context.output))


class TestBannerAndHelp(unittest.TestCase):
    """`weightslab`, `weightslab help`, `-h`, `--help` should show banner + commands."""

    @staticmethod
    def _capture_main(argv):
        buf = io.StringIO()
        with patch("sys.argv", argv):
            with contextlib.redirect_stdout(buf):
                try:
                    main()
                except SystemExit:
                    pass  # argparse -h/--help exits 0
        return buf.getvalue()

    def test_dash_h_shows_banner_and_command_reference(self):
        out = self._capture_main(["weightslab", "-h"])
        self.assertIn("WeightsLab", out)          # tagline from description
        self.assertIn("ui launch", out)           # command reference
        self.assertIn("--no-certs", out)          # documented flag

    def test_long_help_flag_shows_command_reference(self):
        out = self._capture_main(["weightslab", "--help"])
        self.assertIn("ui launch", out)
        self.assertIn("--force-certs", out)

    def test_help_subcommand_shows_command_reference(self):
        out = self._capture_main(["weightslab", "help"])
        self.assertIn("ui launch", out)
        self.assertIn("se [DIR]", out)

    def test_no_args_shows_command_reference(self):
        out = self._capture_main(["weightslab"])
        self.assertIn("ui launch", out)


class TestLaunchCliFlags(unittest.TestCase):
    """The `ui launch` subcommand parses the new flags onto the args namespace."""

    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_no_certs_flag_parsed(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch", "--no-certs"]):
            main()
        mock_launch.assert_called_once()
        self.assertTrue(mock_launch.call_args.args[0].no_certs)

    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_force_certs_and_no_clean_flags_parsed(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch", "--force-certs", "--no-clean"]):
            main()
        mock_launch.assert_called_once()
        ns = mock_launch.call_args.args[0]
        self.assertTrue(ns.force_certs)
        self.assertTrue(ns.no_clean)

    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_defaults_are_false(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch"]):
            main()
        ns = mock_launch.call_args.args[0]
        self.assertFalse(ns.no_certs)
        self.assertFalse(ns.no_clean)
        self.assertFalse(ns.force_certs)


if __name__ == "__main__":
    unittest.main()
