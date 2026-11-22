"""Offline page refresh safeguards; no dataset, GPU or network required."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_demo


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.chart = b"/* chart test fixture */"
        self.pin = patch.object(build_demo, "CHART_SHA256", hashlib.sha256(self.chart).hexdigest())
        self.pin.start()
        self.addCleanup(self.pin.stop)
        snapshot = {"prediction": 1, "confidence": .9, "true_label_margin": -2}
        self.attribution = {"protocol_sha256": "protocol", "seed": 17, "target": "dataset class", "cases": {
            "bird.jpg": {"label": 0, "group": "0:1", "crop": "same-image", "snapshots": {"continue": snapshot}}
        }}
        self.data = {"protocol_sha256": "protocol", "media": {}, "metrics": {"accuracy": .7}, "attribution": self.attribution}
        self.existing = self.root / "old.html"
        self.existing.write_text("const DATA=" + json.dumps(self.data) + ";")
        self.output = self.root / "new.html"

    def test_refresh_preserves_metrics_and_embeds_raw_maps_and_renderer(self):
        updated = copy.deepcopy(self.attribution)
        updated["cases"]["bird.jpg"]["snapshots"]["continue"]["attribution"] = {"encoding": "float32-le-base64", "shape": [1, 1], "data": "AAAAAA=="}
        build_demo.refresh(self.existing, updated, self.chart, self.output)
        html = self.output.read_text()
        data, _ = json.JSONDecoder().raw_decode(html.partition("const DATA=")[2])
        self.assertEqual(data["metrics"], self.data["metrics"])
        self.assertEqual(data["attribution"], updated)
        self.assertNotIn("/*__ATTRIBUTION_JS__*/", html)
        self.assertIn("root.AttributionView = api", html)

    def test_refresh_rejects_changed_evidence(self):
        for field, value in (("protocol_sha256", "other"), ("seed", 29), ("target", "predicted class")):
            with self.subTest(field=field):
                updated = copy.deepcopy(self.attribution)
                updated[field] = value
                with self.assertRaises(ValueError):
                    build_demo.refresh(self.existing, updated, self.chart, self.output)
        for field, value in (("label", 1), ("crop", "different-image"), ("group", "1:0")):
            with self.subTest(field=field):
                updated = copy.deepcopy(self.attribution)
                updated["cases"]["bird.jpg"][field] = value
                with self.assertRaises(ValueError):
                    build_demo.refresh(self.existing, updated, self.chart, self.output)
        updated = copy.deepcopy(self.attribution)
        updated["cases"]["bird.jpg"]["snapshots"]["continue"]["prediction"] = 0
        with self.assertRaises(ValueError):
            build_demo.refresh(self.existing, updated, self.chart, self.output)

    def test_does_not_overwrite_existing_page(self):
        original = self.existing.read_bytes()
        with self.assertRaises(FileExistsError):
            build_demo.refresh(self.existing, None, self.chart, self.existing)
        self.assertEqual(original, self.existing.read_bytes())

    def test_rejects_unpinned_chart_and_escapes_script_data(self):
        with self.assertRaises(ValueError):
            build_demo.render(self.data, b"wrong chart", self.output)
        self.data["note"] = "</script><script>alert(1)</script>"
        build_demo.render(self.data, self.chart, self.output)
        self.assertNotIn(self.data["note"], self.output.read_text())


if __name__ == "__main__":
    unittest.main()
