"""The licence is stated the same way everywhere it is declared.

WeightsLab is free for noncommercial use under PolyForm Noncommercial 1.0.0;
commercial use needs a paid licence (see LICENSE). These checks keep the
package metadata, ``weightslab.__license__`` and the licence files in step.
"""
import tomllib
import unittest
from pathlib import Path

import weightslab

REPO_ROOT = Path(__file__).resolve().parents[2]
SPDX_ID = "PolyForm-Noncommercial-1.0.0"


class LicenseMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
            cls.project = tomllib.load(fh)["project"]

    def test_pyproject_declares_the_spdx_expression(self):
        self.assertEqual(self.project["license"], SPDX_ID)

    def test_package_states_the_same_licence(self):
        self.assertEqual(weightslab.__license__, SPDX_ID)

    def test_licence_files_exist(self):
        for name in self.project["license-files"]:
            self.assertTrue((REPO_ROOT / name).is_file(), f"missing licence file {name}")

    def test_no_license_classifier_next_to_the_expression(self):
        # PEP 639: a licence expression replaces the License :: classifiers.
        self.assertFalse(
            [c for c in self.project.get("classifiers", []) if c.startswith("License ::")]
        )

    def test_notice_points_commercial_use_to_a_paid_licence(self):
        notice = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("PolyForm Noncommercial License 1.0.0", notice)
        self.assertIn("commercial licence", notice)
        self.assertIn("Required Notice: Copyright Graybox SAS", notice)


if __name__ == "__main__":
    unittest.main()
