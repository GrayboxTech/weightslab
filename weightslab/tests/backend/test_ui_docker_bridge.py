import argparse
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from weightslab.ui_docker_bridge import (
    _check_docker,
    _clean_stale_docker_resources,
    _compose_cmd,
    _ensure_certificates,
    _ensure_scripts_executable,
    _generate_certs_with_fallback,
    _get_example_dir,
    _install_example_requirements,
    _make_executable,
    _remove_docker_image,
    _strip_derived_deploy_env,
    _DERIVED_DEPLOY_ENV_VARS,
    _FRONTEND_IMAGE,
    _STACK_CONTAINERS,
    example_start,
    main,
    ui_launch,
    ui_secure_environment,
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
    @patch("weightslab.ui_docker_bridge._compose_base_cmd", return_value=["docker", "compose"])
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_runs_docker_compose_with_env(self, mock_run, _mock_base):
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

    @patch("weightslab.ui_docker_bridge._compose_base_cmd", return_value=["docker-compose"])
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_v1_translates_up_pull_into_pull_then_up(self, mock_run, _mock_base):
        """Compose v1 has no `up --pull`; it must `pull` first, then `up` without the flag."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        _compose_cmd("/c.yml", "/e.yaml", ["up", "-d", "--pull", "always"])
        cmds = [c.args[0] for c in mock_run.call_args_list]
        # A separate `docker-compose -f /c.yml pull` ran first.
        self.assertIn(["docker-compose", "-f", "/c.yml", "pull"], cmds)
        # The final `up` carries no --pull flag (v1 would reject it).
        up_cmd = cmds[-1]
        self.assertEqual(up_cmd, ["docker-compose", "-f", "/c.yml", "up", "-d"])
        self.assertNotIn("--pull", up_cmd)

    @patch("weightslab.ui_docker_bridge._compose_base_cmd", return_value=None)
    def test_exits_when_no_compose_cli(self, _mock_base):
        with self.assertRaises(SystemExit) as ctx:
            _compose_cmd("/c.yml", "/e.yaml", ["up", "-d"])
        self.assertEqual(ctx.exception.code, 1)


class TestComposeDetection(unittest.TestCase):
    """Prefer Compose v2 (`docker compose`), fall back to v1 (`docker-compose`)."""

    @staticmethod
    def _fake_run(ok_keys):
        """Return a subprocess.run stub: rc 0 when ' '.join(cmd[:2]) is in ok_keys."""
        def run(cmd, *a, **k):
            return MagicMock(returncode=0 if " ".join(cmd[:2]) in ok_keys else 1)
        return run

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_prefers_v2(self, mock_run):
        from weightslab.ui_docker_bridge import _detect_compose_cmd
        mock_run.side_effect = self._fake_run({"docker compose"})
        self.assertEqual(_detect_compose_cmd(), ["docker", "compose"])

    @patch("weightslab.ui_docker_bridge.shutil.which", return_value="/usr/bin/docker-compose")
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_falls_back_to_v1_when_v2_absent(self, mock_run, _which):
        from weightslab.ui_docker_bridge import _detect_compose_cmd
        # v2 probe fails (rc 1); v1 probe (`docker-compose version`) succeeds.
        mock_run.side_effect = self._fake_run({"docker-compose version"})
        self.assertEqual(_detect_compose_cmd(), ["docker-compose"])

    @patch("weightslab.ui_docker_bridge.shutil.which", return_value=None)
    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_returns_none_when_neither_available(self, mock_run, _which):
        from weightslab.ui_docker_bridge import _detect_compose_cmd
        mock_run.side_effect = self._fake_run(set())
        self.assertIsNone(_detect_compose_cmd())


class TestScriptsExecutable(unittest.TestCase):
    """Bundled .sh scripts are made executable so users skip the manual `chmod +x`."""

    @unittest.skipIf(sys.platform == "win32", "execute bit is POSIX-only")
    def test_make_executable_adds_exec_bits(self):
        import stat as _stat
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
            path = f.name
        try:
            os.chmod(path, 0o644)
            _make_executable(path)
            mode = os.stat(path).st_mode
            self.assertTrue(mode & _stat.S_IXUSR)
            self.assertTrue(mode & _stat.S_IXGRP)
            self.assertTrue(mode & _stat.S_IXOTH)
        finally:
            os.unlink(path)

    def test_make_executable_is_noop_on_windows(self):
        with patch("weightslab.ui_docker_bridge._is_windows", return_value=True):
            with patch("weightslab.ui_docker_bridge.os.chmod") as mock_chmod:
                _make_executable("/whatever/path.sh")
                mock_chmod.assert_not_called()

    def test_make_executable_swallows_oserror(self):
        # A non-chmod-able path (e.g. root-owned system install) must not raise.
        with patch("weightslab.ui_docker_bridge.os.stat", side_effect=OSError("denied")):
            _make_executable("/root/owned.sh")  # should not raise

    @unittest.skipIf(sys.platform == "win32", "execute bit is POSIX-only")
    def test_ensure_scripts_executable_marks_bundled_scripts(self):
        import stat as _stat
        from weightslab.ui_docker_bridge import _get_bootstrap_script
        _ensure_scripts_executable()
        bootstrap = _get_bootstrap_script()
        self.assertTrue(bootstrap.exists(), bootstrap)
        self.assertTrue(os.stat(bootstrap).st_mode & _stat.S_IXUSR)

    def test_ensure_scripts_executable_noop_on_windows(self):
        with patch("weightslab.ui_docker_bridge._is_windows", return_value=True):
            with patch("weightslab.ui_docker_bridge._make_executable") as mock_mk:
                _ensure_scripts_executable()
                mock_mk.assert_not_called()


class TestUiLaunch(unittest.TestCase):
    """ui_launch: unsecured by default (no cert gen); --certs opts into TLS."""

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_default_no_cert_gen_cleans_and_launches_unsecured(
        self, _gc, _ge, mock_check, mock_compose, mock_clean, mock_ensure,
        _mock_shell, _gb, mock_mgr,
    ):
        mgr = MagicMock(certs_dir="/fake/certs")
        mgr.has_valid_certs.return_value = False   # no certs on disk -> unsecured
        mock_mgr.from_env_or_default.return_value = mgr
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        mock_check.assert_called_once()
        mock_ensure.assert_not_called()               # certs NOT generated by default
        mock_clean.assert_called_once()               # stale cleanup ran
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["up", "-d", "--pull", "always"],
        )
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
    def test_launch_respects_custom_port(
        self, _gc, _ge, _mock_check, _mock_compose, _mock_clean, _mock_ensure,
        _mock_shell, _gb, mock_mgr,
    ):
        mgr = MagicMock(certs_dir="/fake/certs")
        mgr.has_valid_certs.return_value = True
        mock_mgr.from_env_or_default.return_value = mgr
        with patch.dict(os.environ, {"VITE_PORT": "3000"}):
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace(certs=True))
        self.assertTrue(any("https://localhost:3000" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._get_bootstrap_script", return_value="/does/not/exist.sh")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_certs_flag_generates_and_runs_secured(
        self, _gc, _ge, _mock_check, _mock_compose, _mock_clean, mock_ensure,
        _mock_shell, _gb, mock_mgr,
    ):
        mgr = MagicMock(certs_dir="/fake/certs")
        mgr.has_valid_certs.return_value = True   # certs present after generation
        mock_mgr.from_env_or_default.return_value = mgr
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace(certs=True))
        mock_ensure.assert_called_once()          # --certs generates certs
        self.assertTrue(any("https://localhost:5173" in msg for msg in log_context.output))


class TestEnsureCertificates(unittest.TestCase):
    """_ensure_certificates only generates files; it never exports TLS/auth env."""

    @patch("weightslab.ui_docker_bridge._install_ca_trust")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback")
    def test_uses_existing_certs_without_generating(self, mock_gen, _mock_trust):
        manager = MagicMock()
        # Gate is has_any_credentials(); existing creds short-circuit generation.
        manager.has_any_credentials.return_value = True
        manager.has_valid_certs.return_value = True
        result = _ensure_certificates(manager, force_certs=False)
        self.assertTrue(result)
        mock_gen.assert_not_called()
        manager.get_or_create_auth_token.assert_called_once()
        # Derived TLS env must NOT be set here — the deploy pipeline derives it.
        manager.setup_tls_environment.assert_not_called()

    @patch("weightslab.ui_docker_bridge._install_ca_trust")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_generates_when_missing_and_forwards_certs_dir(self, mock_gen, _mock_trust):
        manager = MagicMock()
        # No credentials at the gate -> generate; certs valid afterwards.
        manager.has_any_credentials.return_value = False
        manager.has_valid_certs.return_value = True
        result = _ensure_certificates(manager, force_certs=False)
        self.assertTrue(result)
        mock_gen.assert_called_once_with(force_certs=False, certs_dir=manager.certs_dir)
        manager.setup_tls_environment.assert_not_called()

    @patch("weightslab.ui_docker_bridge._install_ca_trust")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_force_regenerates_even_when_present(self, mock_gen, _mock_trust):
        manager = MagicMock()
        manager.has_any_credentials.return_value = True
        manager.has_valid_certs.return_value = True
        _ensure_certificates(manager, force_certs=True)
        mock_gen.assert_called_once_with(force_certs=True, certs_dir=manager.certs_dir)

    @patch("weightslab.ui_docker_bridge._install_ca_trust")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=1)
    def test_returns_false_on_generation_failure(self, mock_gen, _mock_trust):
        manager = MagicMock()
        manager.has_any_credentials.return_value = False
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


class TestUiSecureEnvironment(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_ui_secure_environment_success(self, mock_gen_certs, mock_cert_manager):
        """`weightslab se`: generate certs + token, export WEIGHTSLAB_CERTS_DIR."""
        mock_manager_instance = MagicMock()       # certs_dir is a MagicMock (supports .mkdir)
        mock_manager_instance.get_or_create_auth_token.return_value = "fake_token"
        mock_cert_manager.return_value = mock_manager_instance

        args = argparse.Namespace(force_certs=False)

        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            ui_secure_environment(args)

        self.assertTrue(any("Certificates generated successfully" in msg for msg in log_context.output))
        self.assertTrue(any("gRPC auth token created" in msg for msg in log_context.output))
        # Generation is pointed at the chosen certs dir (single source of truth).
        mock_gen_certs.assert_called_once_with(force_certs=False, certs_dir=mock_manager_instance.certs_dir)
        mock_manager_instance.certs_dir.mkdir.assert_called_once()
        # se exports WEIGHTSLAB_CERTS_DIR for the process.
        self.assertTrue(any("WEIGHTSLAB_CERTS_DIR exported" in msg for msg in log_context.output))

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    def test_ui_secure_environment_force_certs(self, mock_gen_certs):
        """`weightslab se --force-certs` forwards force_certs to generation."""
        args = argparse.Namespace(force_certs=True)
        with patch.dict(os.environ, {}, clear=False):
            ui_secure_environment(args)
        self.assertTrue(mock_gen_certs.call_args.kwargs["force_certs"])

    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=1)
    def test_ui_secure_environment_cert_failure(self, mock_gen_certs):
        """Certificate generation failure exits non-zero."""
        args = argparse.Namespace(force_certs=False)
        with self.assertRaises(SystemExit) as ctx:
            ui_secure_environment(args)
        self.assertEqual(ctx.exception.code, 1)


class TestMainCLI(unittest.TestCase):
    """The CLI exposes exactly: se, ui launch, start example, help."""

    @patch("weightslab.ui_docker_bridge.ui_secure_environment")
    def test_main_dispatches_se(self, mock_se):
        with patch("sys.argv", ["weightslab", "se"]):
            main()
        mock_se.assert_called_once()

    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_main_dispatches_ui_launch(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch"]):
            main()
        mock_launch.assert_called_once()

    @patch("weightslab.ui_docker_bridge.example_start")
    def test_main_dispatches_start_example(self, mock_example):
        with patch("sys.argv", ["weightslab", "start", "example"]):
            main()
        mock_example.assert_called_once()

    def test_main_ui_without_action_does_not_crash(self):
        with patch("sys.argv", ["weightslab", "ui"]):
            main()  # should print ui help, not raise

    def test_main_start_without_target_does_not_crash(self):
        with patch("sys.argv", ["weightslab", "start"]):
            main()  # should print start help, not raise

    def test_main_help_does_not_crash(self):
        with patch("sys.argv", ["weightslab", "help"]):
            main()  # should not raise

    def test_main_no_args_does_not_crash(self):
        with patch("sys.argv", ["weightslab"]):
            main()  # should not raise


class TestUserOnboardingFlow(unittest.TestCase):
    """Integration-like test of the full onboarding flow, fully hermetic."""

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    @patch("weightslab.ui_docker_bridge._clean_stale_docker_resources")
    @patch("weightslab.ui_docker_bridge._ensure_certificates", return_value=True)
    @patch("weightslab.ui_docker_bridge._generate_certs_with_fallback", return_value=0)
    @patch("weightslab.ui_docker_bridge.CertAuthManager")
    def test_complete_onboarding_workflow(
        self, mock_cert_manager, mock_gen, mock_ensure, mock_clean,
        mock_shell, mock_check, mock_compose, mock_subproc,
    ):
        """se -> ui launch -> start example, with no real Docker/daemon/subprocess."""
        mgr = MagicMock()
        mock_cert_manager.return_value = mgr
        mock_cert_manager.from_env_or_default.return_value = mgr
        mock_subproc.return_value = MagicMock(returncode=0)

        with patch.dict(os.environ, {}, clear=False):
            try:
                ui_secure_environment(argparse.Namespace(force_certs=False))
                ui_launch(argparse.Namespace())
                example_start(argparse.Namespace())
            except Exception as e:
                self.fail(f"Onboarding workflow failed: {e}")

    @patch("sys.argv", ["weightslab", "se", "--force-certs"])
    @patch("weightslab.ui_docker_bridge.ui_secure_environment")
    def test_cli_se_force_certs(self, mock_se):
        main()
        mock_se.assert_called_once()
        self.assertTrue(mock_se.call_args.args[0].force_certs)


class TestBackendConnectionDetection(unittest.TestCase):
    """Test backend connection detection utility."""

    @patch("socket.socket")
    def test_backend_connection_success(self, mock_socket_class):
        from weightslab.ui_docker_bridge import _test_backend_connection
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket
        self.assertTrue(_test_backend_connection())

    @patch("socket.socket")
    def test_backend_connection_failure(self, mock_socket_class):
        from weightslab.ui_docker_bridge import _test_backend_connection
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 1
        mock_socket_class.return_value = mock_socket
        self.assertFalse(_test_backend_connection())

    @patch("socket.socket")
    def test_backend_connection_timeout(self, mock_socket_class):
        from weightslab.ui_docker_bridge import _test_backend_connection
        mock_socket = MagicMock()
        mock_socket.connect_ex.side_effect = Exception("Connection timeout")
        mock_socket_class.return_value = mock_socket
        self.assertFalse(_test_backend_connection())


class TestPathConversion(unittest.TestCase):
    """Test Windows path to Git Bash conversion."""

    def test_windows_path_conversion(self):
        from weightslab.ui_docker_bridge import _convert_to_git_bash_path
        win_path = r"C:\Users\testuser\.weightslab-certs"
        bash_path = _convert_to_git_bash_path(win_path)
        self.assertEqual(bash_path, "/mnt/c/Users/testuser/.weightslab-certs")

    def test_unix_path_passthrough(self):
        from weightslab.ui_docker_bridge import _convert_to_git_bash_path
        unix_path = "/home/testuser/.weightslab-certs"
        self.assertEqual(_convert_to_git_bash_path(unix_path), unix_path)


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
        env_vars = mock_shell.call_args.args[2]
        self.assertIsInstance(env_vars, dict)
        self.assertIn("WEIGHTSLAB_CERTS_DIR", env_vars)
        self.assertIn("custom/certs", env_vars["WEIGHTSLAB_CERTS_DIR"].replace("\\", "/"))

    @patch("weightslab.ui_docker_bridge._run_shell_script", return_value=0)
    def test_generate_certs_without_dir_passes_no_env(self, mock_shell):
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
        # Certs absent on disk -> URL must be http (default unsecured launch).
        mgr = MagicMock(certs_dir="/fake/certs")
        mgr.has_valid_certs.return_value = False
        mock_mgr.from_env_or_default.return_value = mgr
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
                ui_launch(argparse.Namespace())
        self.assertTrue(any("http://localhost:5173" in m for m in log_context.output))
        self.assertFalse(any("https://localhost:5173" in m for m in log_context.output))


class TestExampleStart(unittest.TestCase):
    """`weightslab start example` runs the bundled classification example."""

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_example_start_runs_classification(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            example_start(argparse.Namespace())
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], sys.executable)
        main_py = cmd[1].replace("\\", "/")
        self.assertTrue(main_py.endswith("examples/PyTorch/ws-classification/main.py"), main_py)
        cwd = mock_run.call_args.kwargs["cwd"].replace("\\", "/")
        self.assertTrue(cwd.endswith("examples/PyTorch/ws-classification"), cwd)
        self.assertTrue(any("classification (cls) example" in m for m in log_context.output))

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_example_start_propagates_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=3)
        with self.assertRaises(SystemExit) as ctx:
            example_start(argparse.Namespace())
        self.assertEqual(ctx.exception.code, 3)

    @patch("weightslab.ui_docker_bridge._get_example_dir", return_value=Path("/does/not/exist"))
    def test_example_start_errors_when_missing(self, _mock_dir):
        with self.assertRaises(SystemExit) as ctx:
            example_start(argparse.Namespace())
        self.assertEqual(ctx.exception.code, 1)

    def test_example_dir_points_at_bundled_example(self):
        # The bundled classification example must actually ship with the package.
        self.assertTrue((_get_example_dir("ws-classification") / "main.py").exists())

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_example_start_seg_runs_segmentation(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with self.assertLogs("weightslab.ui_docker_bridge", level="INFO") as log_context:
            example_start(argparse.Namespace(example_kind="seg"))
        main_py = mock_run.call_args.args[0][1].replace("\\", "/")
        self.assertTrue(main_py.endswith("examples/PyTorch/ws-segmentation/main.py"), main_py)
        self.assertTrue(any("segmentation (seg) example" in m for m in log_context.output))

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_example_start_defaults_to_cls_when_flag_absent(self, mock_run):
        # A Namespace without example_kind (e.g. older call sites) still defaults to cls.
        mock_run.return_value = MagicMock(returncode=0)
        example_start(argparse.Namespace())
        main_py = mock_run.call_args.args[0][1].replace("\\", "/")
        self.assertTrue(main_py.endswith("examples/PyTorch/ws-classification/main.py"), main_py)


class TestInstallExampleRequirements(unittest.TestCase):
    """Requirements install is non-interactive and only runs when a file is present."""

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_installs_requirements_non_interactively_when_present(self, mock_run):
        import tempfile
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            req = Path(tmp) / "requirements.txt"
            req.write_text("numpy\n")
            _install_example_requirements(Path(tmp))
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[:5], [sys.executable, "-m", "pip", "install", "-r"])
        self.assertIn("--no-input", cmd)  # never prompts
        self.assertTrue(mock_run.call_args.kwargs.get("check"))

    @patch("weightslab.ui_docker_bridge.subprocess.run")
    def test_skips_when_no_requirements_file(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _install_example_requirements(Path(tmp))
        mock_run.assert_not_called()


class TestBannerAndHelp(unittest.TestCase):
    """`weightslab`, `weightslab help`, `-h`, `--help` show banner + the command set."""

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
        self.assertIn("ui launch", out)
        self.assertIn("--certs", out)
        self.assertIn("start example", out)

    def test_long_help_flag_shows_command_reference(self):
        out = self._capture_main(["weightslab", "--help"])
        self.assertIn("ui launch", out)
        self.assertIn("--force-certs", out)

    def test_help_subcommand_shows_command_reference(self):
        out = self._capture_main(["weightslab", "help"])
        self.assertIn("se", out)
        self.assertIn("ui launch", out)
        self.assertIn("start example", out)

    def test_no_args_shows_command_reference(self):
        out = self._capture_main(["weightslab"])
        self.assertIn("ui launch", out)

    def test_help_does_not_mention_removed_commands(self):
        out = self._capture_main(["weightslab", "help"])
        for removed in ("ui stop", "ui drop", "ui docker", "--no-auth", "--no-clean"):
            self.assertNotIn(removed, out, f"help should not mention removed '{removed}'")


class TestLaunchCliFlags(unittest.TestCase):
    """`ui launch` accepts only --certs (TLS is opt-in; default unsecured)."""

    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_certs_flag_parsed(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch", "--certs"]):
            main()
        mock_launch.assert_called_once()
        self.assertTrue(mock_launch.call_args.args[0].certs)

    @patch("weightslab.ui_docker_bridge.ui_launch")
    def test_launch_default_certs_false(self, mock_launch):
        with patch("sys.argv", ["weightslab", "ui", "launch"]):
            main()
        self.assertFalse(mock_launch.call_args.args[0].certs)

    def test_launch_removed_flags_are_rejected(self):
        for flag in ("--no-auth", "--force-certs", "--no-clean", "--dev"):
            with patch("sys.argv", ["weightslab", "ui", "launch", flag]):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertNotEqual(ctx.exception.code, 0, flag)


if __name__ == "__main__":
    unittest.main()
