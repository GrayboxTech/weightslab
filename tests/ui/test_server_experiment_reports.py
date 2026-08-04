"""Tests for weightslab/ui/server.py's experiment-report browsing endpoints:

- GET /experiment-report/list  -- reports under <experiment_dir>/reports/,
  sorted newest-first.
- GET /experiment-report/view/<name> -- raw HTML content of one report.

These are plain same-origin HTTP endpoints (no gRPC involved), backing the
report button's right-click "list existing reports" / "open a report"
behavior in the connected Weights Studio UI -- mirrors the local Jupyter
notebook picker's /local-notebook/list and /local-notebook endpoints.

Uses a real serve_ui() instance on an ephemeral port (no existing unit-test
harness exists for this HTTP-handler file to build on) rather than mocking
the handler, since the class under test is a BaseHTTPRequestHandler subclass
that's awkward to instantiate directly outside a real request.
"""

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from weightslab.ui import server as ui_server


class _ServerTestCase(unittest.TestCase):
    """Spins up a real serve_ui() on 127.0.0.1:<ephemeral> per test, rooted
    at a fresh temp dir passed as experiment_dir."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd = ui_server.serve_ui(
            ui_host="127.0.0.1", ui_port=0,
            backend_host="localhost", backend_port=50051,
            open_browser=False, block=False,
            experiment_dir=self.tmp,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)

    def _get(self, path):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5)

    def _write_report(self, name, body="<html><body>report</body></html>"):
        reports_dir = os.path.join(self.tmp, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        full_path = os.path.join(reports_dir, name)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(body)
        return full_path


class TestListExperimentReports(_ServerTestCase):

    def test_no_reports_dir_returns_empty_list(self):
        with self._get("/experiment-report/list") as r:
            data = json.loads(r.read().decode())
        self.assertEqual(data, {"reports": []})

    def test_lists_only_html_files(self):
        self._write_report("experiment_report_20260101_000000.html")
        reports_dir = os.path.join(self.tmp, "reports")
        with open(os.path.join(reports_dir, "notes.txt"), "w") as f:
            f.write("not a report")
        with self._get("/experiment-report/list") as r:
            data = json.loads(r.read().decode())
        names = [e["name"] for e in data["reports"]]
        self.assertEqual(names, ["experiment_report_20260101_000000.html"])

    def test_sorted_newest_first(self):
        self._write_report("old.html")
        time.sleep(0.02)
        self._write_report("new.html")
        with self._get("/experiment-report/list") as r:
            data = json.loads(r.read().decode())
        names = [e["name"] for e in data["reports"]]
        self.assertEqual(names, ["new.html", "old.html"])

    def test_entries_include_name_path_and_modified_at(self):
        full_path = self._write_report("r.html")
        with self._get("/experiment-report/list") as r:
            data = json.loads(r.read().decode())
        entry = data["reports"][0]
        self.assertEqual(entry["name"], "r.html")
        self.assertEqual(os.path.normcase(entry["path"]), os.path.normcase(full_path))
        self.assertIsInstance(entry["modified_at"], (int, float))


class TestServeExperimentReport(_ServerTestCase):

    def test_serves_html_content_with_correct_content_type(self):
        self._write_report("r.html", body="<html><body>HELLO</body></html>")
        with self._get("/experiment-report/view/r.html") as r:
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers.get("Content-Type", ""))
            self.assertIn("HELLO", r.read().decode())

    def test_missing_report_returns_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/experiment-report/view/does_not_exist.html")
        self.assertEqual(ctx.exception.code, 404)

    def test_non_html_extension_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/experiment-report/view/r.txt")
        self.assertEqual(ctx.exception.code, 400)

    def test_path_traversal_is_contained_to_reports_dir(self):
        # A secret file OUTSIDE reports_dir (directly in experiment_dir) must
        # never be reachable via "..": os.path.basename strips path
        # separators, so this resolves to a bare filename inside reports_dir
        # (which doesn't exist there) rather than escaping it.
        secret_path = os.path.join(self.tmp, "secret.html")
        with open(secret_path, "w") as f:
            f.write("TOP SECRET")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/experiment-report/view/..%2fsecret.html")
        self.assertEqual(ctx.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
