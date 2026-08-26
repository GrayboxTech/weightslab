"""Tests for weightslab/opencode_binary.py -- the on-demand provisioner that
makes ``pip install weightslab`` ship a working OpenCode with no Node.js.

Everything here is offline: the one network call (download_managed_binary) is
exercised by patching ``urllib.request.urlopen`` to hand back an in-memory npm
tarball, so the extract/chmod/atomic-rename path is covered without touching the
real registry. Platform selection is exercised by patching the tiny set of
host probes (``sys.platform``, ``platform.machine``, AVX2/musl detection).
"""

import io
import os
import stat
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from weightslab import opencode_binary, opencode_process


def _fake_npm_tarball(binary_name: str = "opencode", body: bytes = b"#!/bin/sh\necho ok\n") -> bytes:
    """Build an in-memory .tgz laid out like an opencode-<platform> npm package
    (``package/bin/<binary_name>``), the exact shape _extract_binary looks for."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=f"package/bin/{binary_name}")
        info.size = len(body)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


class ManagedPathTests(unittest.TestCase):
    def test_home_env_override_and_version_scoping(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {opencode_binary.HOME_ENV_VAR: tmp, opencode_binary.VERSION_ENV_VAR: "9.9.9"}
            with patch.dict(os.environ, env, clear=False):
                path = opencode_binary.managed_binary_path()
                self.assertEqual(Path(path).parent.parent, Path(tmp) / "9.9.9")
                self.assertEqual(Path(path).parent.name, "bin")

    def test_pinned_version_env_override(self):
        with patch.dict(os.environ, {opencode_binary.VERSION_ENV_VAR: "1.2.3"}, clear=False):
            self.assertEqual(opencode_binary.pinned_version(), "1.2.3")
        with patch.dict(os.environ, {opencode_binary.VERSION_ENV_VAR: ""}, clear=False):
            self.assertEqual(opencode_binary.pinned_version(),
                             opencode_binary.DEFAULT_OPENCODE_VERSION)

    def test_autodownload_toggle(self):
        for val, expected in [("0", False), ("false", False), ("no", False),
                              ("off", False), ("1", True), ("", True)]:
            with patch.dict(os.environ, {opencode_binary.AUTODOWNLOAD_ENV_VAR: val}, clear=False):
                self.assertEqual(opencode_binary.autodownload_enabled(), expected)


class CandidatePackageTests(unittest.TestCase):
    def _candidates(self, plat, machine, avx2, musl):
        with patch.object(opencode_binary.sys, "platform", plat), \
                patch.object(opencode_binary.platform, "machine", return_value=machine), \
                patch.object(opencode_binary, "_supports_avx2", return_value=avx2), \
                patch.object(opencode_binary, "_is_musl", return_value=musl):
            return opencode_binary.candidate_packages()

    def test_linux_x64_avx2_glibc(self):
        got = self._candidates("linux", "x86_64", avx2=True, musl=False)
        self.assertEqual(got[0], "opencode-linux-x64")
        self.assertIn("opencode-linux-x64-baseline", got)

    def test_linux_x64_no_avx2_prefers_baseline(self):
        got = self._candidates("linux", "x86_64", avx2=False, musl=False)
        self.assertEqual(got[0], "opencode-linux-x64-baseline")

    def test_linux_musl_prefers_musl(self):
        got = self._candidates("linux", "x86_64", avx2=True, musl=True)
        self.assertEqual(got[0], "opencode-linux-x64-musl")

    def test_linux_arm64(self):
        got = self._candidates("linux", "aarch64", avx2=False, musl=False)
        self.assertEqual(got[0], "opencode-linux-arm64")

    def test_darwin_arm64(self):
        got = self._candidates("darwin", "arm64", avx2=False, musl=False)
        self.assertEqual(got, ["opencode-darwin-arm64"])

    def test_windows_x64_avx2(self):
        got = self._candidates("win32", "AMD64", avx2=True, musl=False)
        self.assertEqual(got[0], "opencode-windows-x64")


class FindAndEnsureTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = patch.dict(
            os.environ,
            {opencode_binary.HOME_ENV_VAR: self._tmp.name,
             opencode_binary.VERSION_ENV_VAR: "1.2.3"},
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _install_fake(self):
        path = opencode_binary.managed_binary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def test_find_missing_returns_none(self):
        self.assertIsNone(opencode_binary.find_managed_binary())

    def test_find_present_returns_path(self):
        path = self._install_fake()
        self.assertEqual(opencode_binary.find_managed_binary(), path)

    def test_ensure_returns_existing_without_download(self):
        path = self._install_fake()
        with patch.object(opencode_binary, "download_managed_binary") as dl:
            self.assertEqual(opencode_binary.ensure_managed_binary(), path)
            dl.assert_not_called()

    def test_ensure_respects_autodownload_disabled(self):
        with patch.dict(os.environ, {opencode_binary.AUTODOWNLOAD_ENV_VAR: "0"}, clear=False):
            with patch.object(opencode_binary, "download_managed_binary") as dl:
                self.assertIsNone(opencode_binary.ensure_managed_binary())
                dl.assert_not_called()

    def test_ensure_downloads_when_missing(self):
        sentinel = self._tmp.name + "/sentinel"
        with patch.object(opencode_binary, "download_managed_binary", return_value=sentinel) as dl:
            self.assertEqual(opencode_binary.ensure_managed_binary(), sentinel)
            dl.assert_called_once()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env = patch.dict(
            os.environ,
            {opencode_binary.HOME_ENV_VAR: self._tmp.name,
             opencode_binary.VERSION_ENV_VAR: "1.2.3"},
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_download_extracts_and_marks_executable(self):
        tgz = _fake_npm_tarball(binary_name=opencode_binary._binary_filename())

        def fake_urlopen(req, timeout=None):
            return io.BytesIO(tgz)

        with patch.object(opencode_binary.urllib.request, "urlopen", side_effect=fake_urlopen):
            path = opencode_binary.download_managed_binary()

        self.assertIsNotNone(path)
        self.assertTrue(Path(path).is_file())
        self.assertTrue(os.access(str(path), os.X_OK))
        self.assertEqual(Path(path).read_bytes()[:2], b"#!")

    def test_download_all_candidates_fail_returns_none(self):
        def boom(req, timeout=None):
            raise OSError("network down")

        with patch.object(opencode_binary.urllib.request, "urlopen", side_effect=boom):
            self.assertIsNone(opencode_binary.download_managed_binary())


class BackgroundInstallTests(unittest.TestCase):
    def setUp(self):
        # Reset the once-per-process guard so each test starts clean.
        opencode_binary._bg_started = False
        self.addCleanup(setattr, opencode_binary, "_bg_started", False)

    def test_noop_when_already_installed(self):
        with patch.object(opencode_binary, "find_managed_binary", return_value=Path("/x/opencode")), \
                patch.object(opencode_binary, "download_managed_binary") as dl:
            opencode_binary.ensure_managed_binary_in_background(reason="test")
            dl.assert_not_called()

    def test_noop_when_autodownload_disabled(self):
        with patch.dict(os.environ, {opencode_binary.AUTODOWNLOAD_ENV_VAR: "0"}, clear=False), \
                patch.object(opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_binary, "download_managed_binary") as dl:
            opencode_binary.ensure_managed_binary_in_background(reason="test")
            dl.assert_not_called()

    def test_downloads_in_background_when_missing(self):
        done = threading.Event()

        def fake_download(version=None):
            done.set()
            return Path("/x/opencode")

        with patch.object(opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_binary, "download_managed_binary", side_effect=fake_download):
            opencode_binary.ensure_managed_binary_in_background(reason="test")
            self.assertTrue(done.wait(timeout=5), "background download did not run")

    def test_only_first_call_starts(self):
        with patch.object(opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_binary, "download_managed_binary",
                             return_value=Path("/x/opencode")) as dl:
            opencode_binary.ensure_managed_binary_in_background(reason="a")
            opencode_binary.ensure_managed_binary_in_background(reason="b")
            # second call must be a no-op regardless of thread timing
            import time as _t
            _t.sleep(0.5)
            self.assertLessEqual(dl.call_count, 1)


class ResolverPrecedenceTests(unittest.TestCase):
    """opencode_process.resolve_opencode_argv order:
    managed-present -> PATH -> managed-download -> npx -> None."""

    def test_managed_present_wins(self):
        with patch.object(opencode_process.opencode_binary, "find_managed_binary",
                          return_value=Path("/mgd/opencode")), \
                patch.object(opencode_process.shutil, "which", return_value="/usr/bin/opencode"):
            self.assertEqual(opencode_process.resolve_opencode_argv(), ["/mgd/opencode"])

    def test_path_used_before_download(self):
        with patch.object(opencode_process.opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_process.opencode_binary, "ensure_managed_binary") as ensure, \
                patch.object(opencode_process.shutil, "which",
                             side_effect=lambda n: "/usr/bin/opencode" if n == "opencode" else None):
            self.assertEqual(opencode_process.resolve_opencode_argv(), ["/usr/bin/opencode"])
            ensure.assert_not_called()

    def test_download_when_no_path(self):
        with patch.object(opencode_process.opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_process.opencode_binary, "ensure_managed_binary",
                             return_value=Path("/mgd/opencode")), \
                patch.object(opencode_process.shutil, "which", return_value=None):
            self.assertEqual(opencode_process.resolve_opencode_argv(), ["/mgd/opencode"])

    def test_npx_last_resort(self):
        def which(name):
            return "/usr/bin/npx" if name == "npx" else None
        with patch.object(opencode_process.opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_process.opencode_binary, "ensure_managed_binary", return_value=None), \
                patch.object(opencode_process.shutil, "which", side_effect=which):
            self.assertEqual(opencode_process.resolve_opencode_argv(),
                             ["/usr/bin/npx", "--yes", "opencode-ai@latest"])

    def test_none_when_nothing_available(self):
        with patch.object(opencode_process.opencode_binary, "find_managed_binary", return_value=None), \
                patch.object(opencode_process.opencode_binary, "ensure_managed_binary", return_value=None), \
                patch.object(opencode_process.shutil, "which", return_value=None):
            self.assertIsNone(opencode_process.resolve_opencode_argv())


if __name__ == "__main__":
    unittest.main()
