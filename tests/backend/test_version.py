"""Tests for weightslab._resolve_version's committed-release fallback.

weightslab/VERSION holds the last release tag (kept current by release.yml's
sync-version-file job) and is the source of __version__ — and so of the
telemetry version — only once git, _version.py and dist metadata all fail.
"""
import os
import re
import sys
import types
import unittest
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import weightslab

_VERSION_FILE = os.path.join(os.path.dirname(weightslab.__file__), "VERSION")
# Normalized PEP 440 public version, as release.yml writes it (no "v", no local).
_PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def _read_version_file() -> str:
    with open(_VERSION_FILE, encoding="utf-8") as fh:
        return fh.read().strip()


class TestReleaseVersionFile(unittest.TestCase):
    def test_version_file_is_a_normalized_version(self):
        self.assertRegex(_read_version_file(), _PEP440)

    def test_used_when_git_build_and_metadata_all_fail(self):
        # Not the main process -> the git step is skipped; a None module entry
        # makes `from ._version import ...` raise ImportError.
        with patch.object(weightslab, "_IS_MAIN_PROCESS", False), \
                patch.dict(sys.modules, {"weightslab._version": None}), \
                patch("importlib.metadata.version", side_effect=PackageNotFoundError):
            self.assertEqual(weightslab._resolve_version(), _read_version_file())

    def test_not_used_while_a_build_version_exists(self):
        built = types.ModuleType("weightslab._version")
        built.__version__ = "9.9.9.dev1"
        with patch.object(weightslab, "_IS_MAIN_PROCESS", False), \
                patch.dict(sys.modules, {"weightslab._version": built}):
            self.assertEqual(weightslab._resolve_version(), "9.9.9.dev1")


if __name__ == "__main__":
    unittest.main()
