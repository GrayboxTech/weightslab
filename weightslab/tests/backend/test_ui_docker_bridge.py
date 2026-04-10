import argparse
import os
import unittest
from unittest.mock import MagicMock, patch

from weightslab.ui_docker_bridge import (
    _check_docker,
    _compose_cmd,
    main,
    ui_drop,
    ui_launch,
    ui_stop,
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
    def test_launch_prints_url_default_port(self, _gc, _ge, mock_check, mock_compose, capsys=None):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VITE_PORT", None)
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                ui_launch(argparse.Namespace())
        mock_check.assert_called_once()
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["up", "-d", "--pull", "always"],
        )
        self.assertIn("http://localhost:5173", mock_stdout.getvalue())

    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_launch_respects_custom_port(self, _gc, _ge, _mock_check, _mock_compose):
        with patch.dict(os.environ, {"VITE_PORT": "3000"}):
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                ui_launch(argparse.Namespace())
        self.assertIn("http://localhost:3000", mock_stdout.getvalue())


class TestUiStop(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_stop_prints_message(self, _gc, _ge, _mock_check, mock_compose):
        from io import StringIO
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            ui_stop(argparse.Namespace())
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["stop"],
        )
        self.assertIn("Weights Studio UI stopped", mock_stdout.getvalue())


class TestUiDrop(unittest.TestCase):
    @patch("weightslab.ui_docker_bridge._compose_cmd")
    @patch("weightslab.ui_docker_bridge._check_docker")
    @patch("weightslab.ui_docker_bridge._get_envoy_config", return_value="/fake/envoy.yaml")
    @patch("weightslab.ui_docker_bridge._get_compose_file", return_value="/fake/docker-compose.yml")
    def test_drop_prints_message(self, _gc, _ge, _mock_check, mock_compose):
        from io import StringIO
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            ui_drop(argparse.Namespace())
        mock_compose.assert_called_once_with(
            "/fake/docker-compose.yml",
            "/fake/envoy.yaml",
            ["down", "--rmi", "all"],
        )
        self.assertIn("containers and images removed", mock_stdout.getvalue())


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

    def test_main_help_does_not_crash(self):
        with patch("sys.argv", ["weightslab", "help"]):
            main()  # should not raise

    def test_main_no_args_does_not_crash(self):
        with patch("sys.argv", ["weightslab"]):
            main()  # should not raise


if __name__ == "__main__":
    unittest.main()
