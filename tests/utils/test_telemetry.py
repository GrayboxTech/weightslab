"""Tests for weightslab.utils.telemetry.

Covers the CLI-launch detection that keeps the `import` ping from firing
on every `weightslab <subcommand>` invocation (only `ping_ui_launch` should
fire for `weightslab start`), plus the resulting `ping_import` gating.
"""
import unittest
from unittest.mock import patch

from weightslab.utils import telemetry


class TestLaunchedViaCli(unittest.TestCase):
    def test_true_for_console_script(self):
        with patch.object(telemetry.sys, "argv", ["/usr/local/bin/weightslab", "start"]):
            self.assertTrue(telemetry._launched_via_cli())

    def test_true_for_windows_exe(self):
        with patch.object(telemetry.sys, "argv", [r"C:\Python\Scripts\weightslab.exe", "start"]):
            self.assertTrue(telemetry._launched_via_cli())

    def test_true_for_module_invocation(self):
        cli_path = r"C:\Python\site-packages\weightslab\cli.py"
        with patch.object(telemetry.sys, "argv", [cli_path, "start"]):
            self.assertTrue(telemetry._launched_via_cli())

    def test_false_for_user_script(self):
        with patch.object(telemetry.sys, "argv", ["/home/user/train.py"]):
            self.assertFalse(telemetry._launched_via_cli())

    def test_false_for_notebook_kernel(self):
        with patch.object(telemetry.sys, "argv", ["/usr/bin/python3", "-m", "ipykernel_launcher"]):
            self.assertFalse(telemetry._launched_via_cli())


class TestPingImportGating(unittest.TestCase):
    def setUp(self):
        # `ping_import` short-circuits after its first call per process;
        # reset that guard so each test observes a fresh decision.
        telemetry._import_pinged_this_process = False

    @patch("weightslab.utils.telemetry._fire")
    @patch("weightslab.utils.telemetry._launched_via_cli", return_value=True)
    def test_no_ping_when_launched_via_cli(self, _mock_launched, mock_fire):
        telemetry.ping_import("1.2.3")
        mock_fire.assert_not_called()

    @patch("weightslab.utils.telemetry._fire")
    @patch("weightslab.utils.telemetry._ping_due", return_value=True)
    @patch("weightslab.utils.telemetry._is_ci", return_value=False)
    @patch("weightslab.utils.telemetry._disabled", return_value=False)
    @patch("weightslab.utils.telemetry._launched_via_cli", return_value=False)
    def test_ping_when_not_launched_via_cli(self, _mock_launched, _mock_disabled, _mock_ci, _mock_due, mock_fire):
        telemetry.ping_import("1.2.3")
        mock_fire.assert_called_once_with("import", "1.2.3")


if __name__ == "__main__":
    unittest.main()
