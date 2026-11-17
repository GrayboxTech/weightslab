"""Both halves of WeightsLab must land on ONE OpenCode server.

Observed failure: the studio's model picker wrote one server's config while
the training backend read another's, so a model chosen in the UI never
reached the backend and a `/model` from the CLI never reached the UI — with
no error on either side. The two had different workspace directories (the UI
server generates its own when started without one; the backend falls back to
its cwd), so the per-workspace lock never matched and each spawned its own.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from weightslab import opencode_process as op


class TestMachineWideRendezvous(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._home = tempfile.TemporaryDirectory()
        self._ws = tempfile.TemporaryDirectory()
        patcher = patch.object(Path, "home", return_value=Path(self._home.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._home.cleanup)
        self.addCleanup(self._ws.cleanup)

    def test_machine_lock_lives_under_the_user_home(self):
        self.assertEqual(
            op.machine_lock_path(),
            Path(self._home.name) / ".weightslab" / op.LOCK_FILENAME)

    def test_absent_machine_lock_reads_as_none(self):
        self.assertIsNone(op.read_machine_lock())

    def test_writing_a_workspace_lock_also_publishes_machine_wide(self):
        # This is what lets the OTHER side find this server at all.
        op.write_lock(self._ws.name, "http://127.0.0.1:51447", pid=4242)

        shared = op.read_machine_lock()
        self.assertIsNotNone(shared)
        self.assertEqual(shared["url"], "http://127.0.0.1:51447")
        self.assertEqual(shared["pid"], 4242)

    def test_workspace_lock_still_written(self):
        op.write_lock(self._ws.name, "http://127.0.0.1:51447")

        with open(op.lock_path(self._ws.name), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["url"], "http://127.0.0.1:51447")

    def test_a_different_workspace_adopts_the_running_server(self):
        """The actual bug: two workspaces, one server."""
        op.write_lock(self._ws.name, "http://127.0.0.1:51447", pid=1)

        import tempfile
        with tempfile.TemporaryDirectory() as other_ws:
            with patch.object(op, "opencode_healthy", return_value=True), \
                 patch.object(op, "ensure_workspace_agent_files"), \
                 patch.dict("os.environ", {}, clear=False):
                import os
                os.environ.pop("OPENCODE_URL", None)
                result = op.resolve_or_spawn_opencode(other_ws)

            self.assertTrue(result["ok"])
            self.assertEqual(result["url"], "http://127.0.0.1:51447")
            self.assertEqual(result["source"], "machine-lockfile")

    def test_an_unhealthy_shared_server_is_not_adopted(self):
        op.write_lock(self._ws.name, "http://127.0.0.1:9", pid=1)

        import os, tempfile
        with tempfile.TemporaryDirectory() as other_ws:
            with patch.object(op, "opencode_healthy", return_value=False), \
                 patch.object(op, "ensure_workspace_agent_files"), \
                 patch.object(op, "resolve_opencode_argv", return_value=None):
                os.environ.pop("OPENCODE_URL", None)
                result = op.resolve_or_spawn_opencode(other_ws)

            # Falls through to spawning (which we stubbed to fail) rather than
            # adopting a dead address.
            self.assertFalse(result["ok"])

    def test_explicit_opencode_url_still_wins(self):
        """An operator who set OPENCODE_URL opted out of discovery."""
        op.write_lock(self._ws.name, "http://127.0.0.1:51447", pid=1)

        import os, tempfile
        with tempfile.TemporaryDirectory() as other_ws:
            with patch.object(op, "opencode_healthy", return_value=True), \
                 patch.object(op, "ensure_workspace_agent_files"), \
                 patch.dict(os.environ, {"OPENCODE_URL": "http://127.0.0.1:4096"}):
                result = op.resolve_or_spawn_opencode(other_ws)

            self.assertEqual(result["url"], "http://127.0.0.1:4096")
            self.assertEqual(result["source"], "env")


if __name__ == "__main__":
    unittest.main()
