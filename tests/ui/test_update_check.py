"""The studio's "new WeightsLab version" notice: the UI server compares the
backend version it is given with the latest stable release on PyPI."""
import json
import os
import threading
import time
import unittest
import unittest.mock
import urllib.request

from weightslab.ui import server as ui_server
from weightslab.ui import update_check


def _index(*versions, yanked=(), info=None):
    releases = {}
    for v in versions:
        releases[v] = [{"filename": f"weightslab-{v}.tar.gz", "yanked": v in yanked}]
    return {"info": {"version": info or (versions[-1] if versions else "")}, "releases": releases}


class TestLatestStableRelease(unittest.TestCase):
    def test_highest_final_release_wins(self):
        self.assertEqual(update_check.latest_stable_release(_index("2.0.2", "2.0.10", "2.0.3")), "2.0.10")

    def test_prereleases_and_dev_builds_are_ignored(self):
        index = _index("2.0.3", "2.1.0rc1", "2.1.0b2", "2.0.4.dev7", "2.1.0a1")
        self.assertEqual(update_check.latest_stable_release(index), "2.0.3")

    def test_a_fully_yanked_release_is_ignored(self):
        self.assertEqual(update_check.latest_stable_release(_index("2.0.3", "2.0.4", yanked={"2.0.4"})), "2.0.3")

    def test_falls_back_to_info_version(self):
        self.assertEqual(update_check.latest_stable_release({"info": {"version": "2.0.3"}}), "2.0.3")
        self.assertIsNone(update_check.latest_stable_release({"info": {"version": "2.1.0rc1"}}))

    def test_only_the_public_index_is_used(self):
        self.assertTrue(update_check.PYPI_JSON_URL.startswith("https://pypi.org/"))
        self.assertNotIn("test.pypi", update_check.PYPI_JSON_URL)


class TestUpdateInfo(unittest.TestCase):
    def setUp(self):
        update_check._reset_cache()
        self._env = os.environ.pop("WEIGHTSLAB_UPDATE_CHECK", None)

    def tearDown(self):
        update_check._reset_cache()
        if self._env is not None:
            os.environ["WEIGHTSLAB_UPDATE_CHECK"] = self._env
        else:
            os.environ.pop("WEIGHTSLAB_UPDATE_CHECK", None)

    def test_older_backend_gets_the_notice(self):
        info = update_check.update_info("2.0.3", fetch=lambda: _index("2.0.3", "2.0.5"))
        self.assertTrue(info["update_available"])
        self.assertEqual(info["latest"], "2.0.5")
        self.assertEqual(info["url"], "https://pypi.org/project/weightslab/2.0.5/")

    def test_up_to_date_or_newer_backend_gets_none(self):
        fetch = lambda: _index("2.0.3", "2.0.5")
        self.assertFalse(update_check.update_info("2.0.5", fetch=fetch)["update_available"])
        update_check._reset_cache()
        # A dev build past the release (e.g. an editable install of dev).
        self.assertFalse(update_check.update_info("2.0.6.dev0", fetch=fetch)["update_available"])

    def test_unknown_versions_never_notify(self):
        fetch = lambda: _index("2.0.5")
        self.assertFalse(update_check.update_info("", fetch=fetch)["update_available"])
        self.assertFalse(update_check.update_info("not-a-version", fetch=fetch)["update_available"])

    def test_offline_means_no_notice_and_no_retry_per_page_load(self):
        calls = []

        def offline():
            calls.append(1)
            raise OSError("network unreachable")

        for _ in range(3):
            info = update_check.update_info("2.0.3", fetch=offline)
            self.assertFalse(info["update_available"])
            self.assertIsNone(info["latest"])
        self.assertEqual(len(calls), 1, "the failure is cached too")

    def test_pypi_is_asked_once_per_cache_period(self):
        calls = []

        def fetch():
            calls.append(1)
            return _index("2.0.5")

        update_check.latest_release(fetch, now=1000.0)
        update_check.latest_release(fetch, now=1000.0 + update_check.CACHE_SECONDS - 1)
        self.assertEqual(len(calls), 1)
        update_check.latest_release(fetch, now=1000.0 + update_check.CACHE_SECONDS + 1)
        self.assertEqual(len(calls), 2)

    def test_can_be_turned_off(self):
        os.environ["WEIGHTSLAB_UPDATE_CHECK"] = "off"
        fetch = unittest.mock.Mock(return_value=_index("9.9.9"))
        info = update_check.update_info("2.0.3", fetch=fetch)
        self.assertFalse(info["enabled"])
        self.assertFalse(info["update_available"])
        fetch.assert_not_called()


class TestLatestVersionEndpoint(unittest.TestCase):
    def setUp(self):
        update_check._reset_cache()
        self.httpd = ui_server.serve_ui(
            ui_host="127.0.0.1", ui_port=0, backend_host="localhost", backend_port=50051,
            open_browser=False, block=False,
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5)
        update_check._reset_cache()

    def test_answers_for_the_backend_version_it_is_given(self):
        with unittest.mock.patch.object(update_check, "_fetch_index", return_value=_index("2.0.3", "2.0.5")), \
                unittest.mock.patch.object(update_check.latest_release, "__defaults__", (update_check._fetch_index, None)):
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/weightslab/latest-version?installed=2.0.3", timeout=10) as r:
                info = json.loads(r.read().decode())
        self.assertEqual(info["installed"], "2.0.3")
        self.assertEqual(info["latest"], "2.0.5")
        self.assertTrue(info["update_available"])


if __name__ == "__main__":
    unittest.main()
