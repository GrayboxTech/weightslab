"""Tests for the container/tunnel-enabling knobs:
  * opencode_process.opencode_bind_host() -- what OpenCode binds to
  * ui.server._client_is_trusted -- who may hit the local-only control routes

Both default to the safe, loopback-only behavior; the env overrides only widen
things for the container-behind-a-tunnel deployment.
"""

import ipaddress
import os
import unittest
from unittest.mock import patch

from weightslab import opencode_process
from weightslab.ui import server as ui_server


class BindHostTests(unittest.TestCase):
    def test_default_is_loopback(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(opencode_process.HOST_ENV_VAR, None)
            self.assertEqual(opencode_process.opencode_bind_host(), "127.0.0.1")

    def test_env_override(self):
        with patch.dict(os.environ, {opencode_process.HOST_ENV_VAR: "0.0.0.0"}, clear=False):
            self.assertEqual(opencode_process.opencode_bind_host(), "0.0.0.0")

    def test_blank_env_falls_back_to_default(self):
        with patch.dict(os.environ, {opencode_process.HOST_ENV_VAR: "  "}, clear=False):
            self.assertEqual(opencode_process.opencode_bind_host(), "127.0.0.1")


class TrustedClientTests(unittest.TestCase):
    def test_loopback_always_trusted(self):
        with patch.object(ui_server, "_TRUSTED_CLIENT_NETS", []):
            self.assertTrue(ui_server._client_is_trusted("127.0.0.1"))
            self.assertTrue(ui_server._client_is_trusted("::1"))

    def test_non_loopback_rejected_by_default(self):
        with patch.object(ui_server, "_TRUSTED_CLIENT_NETS", []):
            self.assertFalse(ui_server._client_is_trusted("172.17.0.1"))

    def test_trusted_net_allows_gateway(self):
        nets = [ipaddress.ip_network("172.16.0.0/12")]
        with patch.object(ui_server, "_TRUSTED_CLIENT_NETS", nets):
            self.assertTrue(ui_server._client_is_trusted("172.17.0.1"))
            self.assertFalse(ui_server._client_is_trusted("8.8.8.8"))

    def test_garbage_addr_is_not_trusted(self):
        nets = [ipaddress.ip_network("172.16.0.0/12")]
        with patch.object(ui_server, "_TRUSTED_CLIENT_NETS", nets):
            self.assertFalse(ui_server._client_is_trusted("not-an-ip"))


class TrustedNetsParsingTests(unittest.TestCase):
    def test_parse_multiple_and_ignore_invalid(self):
        with patch.dict(os.environ,
                        {"WEIGHTSLAB_UI_TRUSTED_HOSTS": "172.16.0.0/12, bad, 10.1.2.3"},
                        clear=False):
            nets = ui_server._parse_trusted_client_nets()
        self.assertEqual(len(nets), 2)

    def test_empty_env_is_empty_list(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WEIGHTSLAB_UI_TRUSTED_HOSTS", None)
            self.assertEqual(ui_server._parse_trusted_client_nets(), [])


if __name__ == "__main__":
    unittest.main()
