"""Tests for the `weightslab agent` CLI surface and the agent-config gating that
keeps an unconfigured run agent-free and error-free (info hint only)."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from weightslab import cli


class AgentConfiguredTests(unittest.TestCase):
    def test_not_configured_when_no_url_no_env_no_auth(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENCODE_URL", None)
            with patch.object(cli, "_opencode_auth_paths",
                              return_value=[Path("/nonexistent/auth.json")]), \
                    patch.object(cli, "_agent_env_files",
                                 return_value=[Path("/nonexistent/.env")]):
                self.assertFalse(cli.agent_is_configured())

    def test_configured_when_opencode_url_set(self):
        with patch.dict(os.environ, {"OPENCODE_URL": "http://127.0.0.1:4096"}, clear=False):
            self.assertTrue(cli.agent_is_configured())

    def test_configured_when_env_file_present(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENCODE_URL", None)
            with patch.object(cli, "_agent_env_files") as envs, \
                    patch.object(cli, "_opencode_auth_paths", return_value=[]), \
                    patch.object(Path, "is_file", return_value=True):
                envs.return_value = [Path("/proj/.env")]
                self.assertTrue(cli.agent_is_configured())

    def test_configured_when_auth_file_present(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENCODE_URL", None)
            with patch.object(cli, "_agent_env_files", return_value=[Path("/nonexistent/.env")]), \
                    patch.object(cli, "_opencode_auth_paths") as paths, \
                    patch.object(Path, "is_file", return_value=True):
                paths.return_value = [Path("/whatever/auth.json")]
                self.assertTrue(cli.agent_is_configured())


class PrewarmGateTests(unittest.TestCase):
    def test_install_always_no_hint_when_configured(self):
        # Installing the binary is unconditional; only the sign-in hint is gated.
        with patch.object(cli, "agent_is_configured", return_value=True), \
                patch.object(cli, "_prewarm_opencode") as prewarm, \
                patch.object(cli, "_log_agent_config_hint") as hint:
            cli._prewarm_opencode_or_hint()
            prewarm.assert_called_once()
            hint.assert_not_called()

    def test_install_and_hint_when_not_configured(self):
        with patch.object(cli, "agent_is_configured", return_value=False), \
                patch.object(cli, "_prewarm_opencode") as prewarm, \
                patch.object(cli, "_log_agent_config_hint") as hint:
            cli._prewarm_opencode_or_hint()
            prewarm.assert_called_once()
            hint.assert_called_once()


class AgentParserTests(unittest.TestCase):
    def test_agent_init_parsed(self):
        args = cli._build_parser().parse_args(["agent", "init", "--provision-only"])
        self.assertEqual(args.command, "agent")
        self.assertEqual(args.agent_action, "init")
        self.assertTrue(args.provision_only)

    def test_agent_init_defaults_no_provision_only(self):
        args = cli._build_parser().parse_args(["agent", "init"])
        self.assertFalse(args.provision_only)


if __name__ == "__main__":
    unittest.main()
