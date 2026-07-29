"""Tests for weightslab.tunnel's `weightslab tunnel` CLI handler.

Covers the telemetry ping added to `tunnel_connect` (mirrors `ui_start_native`'s
`ping_ui_launch`) and that a broken ping never blocks the tunnel from running.
"""
import argparse
import unittest
from unittest.mock import patch

from weightslab.tunnel import tunnel_connect


class TestTunnelTelemetry(unittest.TestCase):
    @patch("weightslab.utils.telemetry.ping_tunnel_launch")
    @patch("weightslab.tunnel.run_tunnel", return_value=0)
    def test_tunnel_connect_fires_tunnel_launch_ping(self, _mock_run, mock_ping):
        from weightslab import __version__ as expected_version
        args = argparse.Namespace(endpoint="bore.pub:12345", remote_port=None,
                                  listen_host=None, listen_port=None)
        with self.assertRaises(SystemExit):
            tunnel_connect(args)
        mock_ping.assert_called_once_with(expected_version)

    @patch("weightslab.utils.telemetry.ping_tunnel_launch", side_effect=RuntimeError("boom"))
    @patch("weightslab.tunnel.run_tunnel", return_value=0)
    def test_tunnel_connect_survives_telemetry_failure(self, mock_run, _mock_ping):
        args = argparse.Namespace(endpoint="bore.pub:12345", remote_port=None,
                                  listen_host=None, listen_port=None)
        with self.assertRaises(SystemExit) as ctx:
            tunnel_connect(args)
        self.assertEqual(ctx.exception.code, 0)
        mock_run.assert_called_once()

    @patch("weightslab.utils.telemetry.ping_tunnel_launch")
    def test_tunnel_connect_pings_even_without_endpoint(self, mock_ping):
        """The ping fires before argument validation, same as ui_start_native."""
        args = argparse.Namespace(endpoint=None, remote_port=None,
                                  listen_host=None, listen_port=None)
        with patch.dict("os.environ", {}, clear=False):
            import os as _os
            _os.environ.pop("WEIGHTSLAB_TUNNEL_ENDPOINT", None)
            with self.assertRaises(SystemExit) as ctx:
                tunnel_connect(args)
        self.assertEqual(ctx.exception.code, 2)
        mock_ping.assert_called_once()


if __name__ == "__main__":
    unittest.main()
