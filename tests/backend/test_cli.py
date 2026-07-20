"""Tests for the Docker-free WeightsLab CLI (weightslab.cli).

Covers the command set that survived the Docker removal: se (secure env),
start (native UI), start example, cli, tunnel, banner/help — plus cert
generation and the small path/script helpers. There is no Docker, Envoy, or
compose code left to test.
"""
import argparse
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from weightslab.cli import (
    _convert_to_git_bash_path,
    _ensure_scripts_executable,
    _generate_certs_with_fallback,
    _get_cert_script,
    _get_example_dir,
    _install_example_requirements,
    _make_executable,
    example_start,
    main,
    ui_secure_environment,
    ui_start_native,
)


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
        with patch("weightslab.cli._is_windows", return_value=True):
            with patch("weightslab.cli.os.chmod") as mock_chmod:
                _make_executable("/whatever/path.sh")
                mock_chmod.assert_not_called()

    def test_make_executable_swallows_oserror(self):
        with patch("weightslab.cli.os.stat", side_effect=OSError("denied")):
            _make_executable("/root/owned.sh")  # should not raise

    def test_ensure_scripts_executable_noop_on_windows(self):
        with patch("weightslab.cli._is_windows", return_value=True):
            with patch("weightslab.cli._make_executable") as mock_mk:
                _ensure_scripts_executable()
                mock_mk.assert_not_called()

    def test_cert_script_is_bundled_under_ui_utils(self):
        cert_script = _get_cert_script()
        self.assertTrue(str(cert_script).replace("\\", "/").endswith(
            "weightslab/ui/utils/generate-certs-auth-token.sh"), cert_script)


class TestPathConversion(unittest.TestCase):
    def test_windows_path_conversion(self):
        self.assertEqual(
            _convert_to_git_bash_path(r"C:\Users\testuser\.weightslab-certs"),
            "/mnt/c/Users/testuser/.weightslab-certs",
        )

    def test_unix_path_passthrough(self):
        self.assertEqual(
            _convert_to_git_bash_path("/home/testuser/.weightslab-certs"),
            "/home/testuser/.weightslab-certs",
        )


class TestCertGeneration(unittest.TestCase):
    @patch("weightslab.cli._run_shell_script", return_value=0)
    def test_generate_certs_forwards_certs_dir(self, mock_shell):
        rc = _generate_certs_with_fallback(force_certs=False, certs_dir="/custom/certs")
        self.assertEqual(rc, 0)
        env_vars = mock_shell.call_args.args[2]
        self.assertIsInstance(env_vars, dict)
        self.assertIn("WEIGHTSLAB_CERTS_DIR", env_vars)
        self.assertIn("custom/certs", env_vars["WEIGHTSLAB_CERTS_DIR"].replace("\\", "/"))

    @patch("weightslab.cli._run_shell_script", return_value=0)
    def test_generate_certs_without_dir_passes_no_env(self, mock_shell):
        _generate_certs_with_fallback(force_certs=False)
        self.assertIsNone(mock_shell.call_args.args[2])


class TestUiSecureEnvironment(unittest.TestCase):
    @patch("weightslab.cli.CertAuthManager")
    @patch("weightslab.cli._generate_certs_with_fallback", return_value=0)
    def test_ui_secure_environment_success(self, mock_gen_certs, mock_cert_manager):
        """`weightslab se`: generate certs + token, export WEIGHTSLAB_CERTS_DIR."""
        mgr = MagicMock()
        mgr.get_or_create_auth_token.return_value = "fake_token"
        mock_cert_manager.return_value = mgr
        with self.assertLogs("weightslab.cli", level="INFO") as log_context:
            ui_secure_environment(argparse.Namespace(force_certs=False))
        self.assertTrue(any("Certificates generated successfully" in m for m in log_context.output))
        mock_gen_certs.assert_called_once_with(force_certs=False, certs_dir=mgr.certs_dir)
        mgr.certs_dir.mkdir.assert_called_once()
        self.assertTrue(any("WEIGHTSLAB_CERTS_DIR exported" in m for m in log_context.output))

    @patch("weightslab.cli._generate_certs_with_fallback", return_value=0)
    def test_ui_secure_environment_force_certs(self, mock_gen_certs):
        with patch.dict(os.environ, {}, clear=False):
            ui_secure_environment(argparse.Namespace(force_certs=True))
        self.assertTrue(mock_gen_certs.call_args.kwargs["force_certs"])

    @patch("weightslab.cli._generate_certs_with_fallback", return_value=1)
    def test_ui_secure_environment_cert_failure(self, _mock_gen_certs):
        with self.assertRaises(SystemExit) as ctx:
            ui_secure_environment(argparse.Namespace(force_certs=False))
        self.assertEqual(ctx.exception.code, 1)


class TestUiStartNative(unittest.TestCase):
    """`weightslab start` serves the bundled SPA + gRPC-Web proxy (no Docker)."""

    @patch("weightslab.ui.server.serve_ui")
    def test_start_invokes_serve_ui_with_defaults(self, mock_serve):
        with patch.dict(os.environ, {}, clear=False):
            for k in ("WEIGHTSLAB_UI_HOST", "WEIGHTSLAB_UI_PORT",
                      "GRPC_BACKEND_HOST", "GRPC_BACKEND_PORT"):
                os.environ.pop(k, None)
            args = argparse.Namespace(port=9123, host=None, backend_host=None,
                                      backend_port=None, no_browser=True, certs=False)
            ui_start_native(args)
        mock_serve.assert_called_once()
        kwargs = mock_serve.call_args.kwargs
        self.assertEqual(kwargs["backend_port"], 50051)
        self.assertFalse(kwargs["open_browser"])
        self.assertIsNone(kwargs["certs_dir"])  # no --certs -> unsecured

    @patch("weightslab.ui.server.serve_ui")
    @patch("weightslab.cli.CertAuthManager")
    def test_start_certs_without_valid_certs_falls_back_to_http(self, mock_mgr, mock_serve):
        mgr = MagicMock()
        mgr.has_valid_certs.return_value = False
        mock_mgr.from_env_or_default.return_value = mgr
        args = argparse.Namespace(port=9124, host=None, backend_host=None,
                                  backend_port=None, no_browser=True, certs=True)
        with self.assertLogs("weightslab.cli", level="WARNING"):
            ui_start_native(args)
        self.assertIsNone(mock_serve.call_args.kwargs["certs_dir"])


class TestExampleStart(unittest.TestCase):
    @patch("weightslab.cli.subprocess.run")
    def test_example_start_runs_classification(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with self.assertLogs("weightslab.cli", level="INFO") as log_context:
            example_start(argparse.Namespace())
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[0], sys.executable)
        main_py = cmd[1].replace("\\", "/")
        self.assertTrue(main_py.endswith("examples/PyTorch/wl-classification/main.py"), main_py)
        self.assertTrue(any("classification (cls) example" in m for m in log_context.output))

    @patch("weightslab.cli.subprocess.run")
    def test_example_start_propagates_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=3)
        with self.assertRaises(SystemExit) as ctx:
            example_start(argparse.Namespace())
        self.assertEqual(ctx.exception.code, 3)

    @patch("weightslab.cli._get_example_dir", return_value=Path("/does/not/exist"))
    def test_example_start_errors_when_missing(self, _mock_dir):
        with self.assertRaises(SystemExit) as ctx:
            example_start(argparse.Namespace())
        self.assertEqual(ctx.exception.code, 1)

    def test_example_dir_points_at_bundled_example(self):
        self.assertTrue((_get_example_dir("wl-classification") / "main.py").exists())

    @patch("weightslab.cli.subprocess.run")
    def test_example_start_seg_runs_segmentation(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        example_start(argparse.Namespace(example_kind="seg"))
        main_py = mock_run.call_args.args[0][1].replace("\\", "/")
        self.assertTrue(main_py.endswith("examples/PyTorch/wl-segmentation/main.py"), main_py)


class TestInstallExampleRequirements(unittest.TestCase):
    @patch("weightslab.cli.subprocess.run")
    def test_installs_requirements_non_interactively_when_present(self, mock_run):
        import tempfile
        mock_run.return_value = MagicMock(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "requirements.txt").write_text("numpy\n")
            _install_example_requirements(Path(tmp))
        cmd = mock_run.call_args.args[0]
        self.assertEqual(cmd[:5], [sys.executable, "-m", "pip", "install", "-r"])
        self.assertIn("--no-input", cmd)

    @patch("weightslab.cli.subprocess.run")
    def test_skips_when_no_requirements_file(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _install_example_requirements(Path(tmp))
        mock_run.assert_not_called()


class TestMainCLI(unittest.TestCase):
    """The CLI exposes exactly: se, start (+example), cli, tunnel, help. No `ui`."""

    @patch("weightslab.cli.ui_secure_environment")
    def test_main_dispatches_se(self, mock_se):
        with patch("sys.argv", ["weightslab", "se"]):
            main()
        mock_se.assert_called_once()

    @patch("weightslab.cli.ui_start_native")
    def test_main_bare_start_launches_native_ui(self, mock_start):
        with patch("sys.argv", ["weightslab", "start"]):
            main()
        mock_start.assert_called_once()

    @patch("weightslab.cli.example_start")
    def test_main_dispatches_start_example(self, mock_example):
        with patch("sys.argv", ["weightslab", "start", "example"]):
            main()
        mock_example.assert_called_once()

    def test_main_ui_command_is_gone(self):
        # `ui` is no longer a valid subcommand (Docker launcher removed).
        with patch("sys.argv", ["weightslab", "ui", "launch"]):
            with self.assertRaises(SystemExit):
                main()

    def test_main_help_does_not_crash(self):
        with patch("sys.argv", ["weightslab", "help"]):
            main()

    def test_main_no_args_does_not_crash(self):
        with patch("sys.argv", ["weightslab"]):
            main()


class TestBannerAndHelp(unittest.TestCase):
    @staticmethod
    def _capture_main(argv):
        buf = io.StringIO()
        with patch("sys.argv", argv):
            with contextlib.redirect_stdout(buf):
                try:
                    main()
                except SystemExit:
                    pass
        return buf.getvalue()

    def test_help_shows_new_command_set(self):
        out = self._capture_main(["weightslab", "help"])
        self.assertIn("se", out)
        self.assertIn("start", out)
        self.assertIn("start example", out)

    def test_help_does_not_mention_removed_docker_artifacts(self):
        # "Docker-free" / "no Docker" phrasing is intentional; the removed
        # commands and Docker artifacts must be gone.
        out = self._capture_main(["weightslab", "-h"]).lower()
        for removed in ("ui launch", "envoy", "nginx", "graybx/weightslab",
                        "docker compose", "docker-compose"):
            self.assertNotIn(removed, out, f"help should not mention '{removed}'")


if __name__ == "__main__":
    unittest.main()
