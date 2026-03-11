import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from weightslab.utils import logs


class TestLogsUnit(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            try:
                handler.close()
            except Exception:
                pass
            root.removeHandler(handler)

        logs._LOG_FILE_PATH = None
        logs._TMP_DIR_PATH = None
        logs._FILE_HANDLER = None

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            try:
                handler.close()
            except Exception:
                pass
            root.removeHandler(handler)

    def test_setup_logging_with_file_and_print_location(self):
        logs.setup_logging("INFO", log_to_file=True)
        self.assertIsNotNone(logs._LOG_FILE_PATH)
        self.assertTrue(os.path.exists(logs._LOG_FILE_PATH))

        with patch("weightslab.utils.logs.print") as p:
            logs._print_log_location()
        p.assert_called_once()

    def test_set_log_directory_moves_log_and_reopens_handler(self):
        logs.setup_logging("DEBUG", log_to_file=True)
        old_path = logs._LOG_FILE_PATH

        tmpdir = tempfile.mkdtemp()
        try:
            logs.set_log_directory(tmpdir)
            self.assertNotEqual(old_path, logs._LOG_FILE_PATH)
            self.assertTrue(logs._LOG_FILE_PATH.startswith(tmpdir))
            self.assertTrue(os.path.exists(logs._LOG_FILE_PATH))
        finally:
            if logs._FILE_HANDLER is not None:
                try:
                    logs._FILE_HANDLER.flush()
                    logs._FILE_HANDLER.close()
                except Exception:
                    pass
            logging.getLogger().handlers = []
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_custom_print_routes_to_levels(self):
        with patch("logging.info") as info_mock, \
             patch("logging.warning") as warning_mock:
            logs.print("hello", "world")
            logs.print("warn-msg", level="WARNING")

        info_mock.assert_called_once()
        warning_mock.assert_called_once()

    def test_set_log_directory_without_setup_is_noop(self):
        with patch("logging.warning") as warn_mock:
            logs.set_log_directory("dummy")
        warn_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
