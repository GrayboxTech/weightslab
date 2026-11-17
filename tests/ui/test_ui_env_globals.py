"""Runtime UI settings: env vars read by `weightslab start` reach the page as
window.WS_* globals (the new-user tour's on/off, video and notebook here)."""
import os
import unittest
import unittest.mock

from weightslab.ui import server as ui_server

_TUTORIAL_VARS = [
    name
    for window_key, candidates in ui_server._UI_ENV_GLOBALS
    if "TUTORIAL" in window_key
    for name in candidates
]


class TutorialSettingsTest(unittest.TestCase):
    def _globals_js(self, env):
        clean = {k: v for k, v in os.environ.items() if k not in _TUTORIAL_VARS}
        clean.update(env)
        with unittest.mock.patch.dict("os.environ", clean, clear=True):
            return ui_server._ui_env_globals_js()

    def test_the_three_tutorial_settings_are_mapped(self):
        keys = [k for k, _ in ui_server._UI_ENV_GLOBALS]
        for key in ("WS_ENABLE_TUTORIAL", "WS_TUTORIAL_VIDEO_URL", "WS_TUTORIAL_COLAB_URL"):
            self.assertIn(key, keys)

    def test_settings_reach_the_page(self):
        js = self._globals_js({
            "ENABLE_TUTORIAL": "0",
            "TUTORIAL_VIDEO_URL": "https://youtu.be/abcdefghijk",
            "WS_TUTORIAL_COLAB_URL": "https://colab.research.google.com/drive/it's-mine",
        })
        self.assertIn("window.WS_ENABLE_TUTORIAL='0';", js)
        self.assertIn("window.WS_TUTORIAL_VIDEO_URL='https://youtu.be/abcdefghijk';", js)
        # Quotes are escaped: the value cannot break out of the JS string.
        self.assertIn("window.WS_TUTORIAL_COLAB_URL='https://colab.research.google.com/drive/it\\'s-mine';", js)

    def test_the_plain_name_wins_over_the_prefixed_ones(self):
        js = self._globals_js({"ENABLE_TUTORIAL": "1", "WS_ENABLE_TUTORIAL": "0", "VITE_ENABLE_TUTORIAL": "0"})
        self.assertIn("window.WS_ENABLE_TUTORIAL='1';", js)

    def test_unset_settings_are_left_to_the_page_defaults(self):
        js = self._globals_js({})
        self.assertNotIn("TUTORIAL", js)


if __name__ == "__main__":
    unittest.main()
