import io
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from weightslab.utils import logs


class LogsTestBase(unittest.TestCase):
    """Isolates the root logger and the module globals from the test runner's."""

    def setUp(self):
        self._tmpdirs = []
        self._saved_env = {
            key: os.environ.pop(key, None)
            for key in ("WEIGHTSLAB_ROOT_LOG_DIR", "WEIGHTSLAB_LOG_FILE_LEVEL",
                        "WEIGHTSLAB_TQDM_LOG_TO_TERMINAL")
        }
        self._reset_logging()

    def tearDown(self):
        self._reset_logging()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for path in self._tmpdirs:
            shutil.rmtree(path, ignore_errors=True)

    def _reset_logging(self):
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
        logs._CONSOLE_HANDLER = None
        logs._CONSOLE_LEVEL = logging.INFO
        logs._EXIT_HOOK_REGISTERED = False
        logs._FILE_LEVEL = logging.NOTSET

    def _mkdtemp(self):
        path = tempfile.mkdtemp()
        self._tmpdirs.append(path)
        return path

    def _log_file_contents(self):
        logs.flush_logs()
        with open(logs._LOG_FILE_PATH, encoding="utf-8") as handle:
            return handle.read()


class TestLogsUnit(LogsTestBase):
    def test_setup_logging_with_file_and_print_location(self):
        logs.setup_logging("INFO", log_to_file=True)
        self.assertIsNotNone(logs._LOG_FILE_PATH)
        self.assertTrue(os.path.exists(logs._LOG_FILE_PATH))

        # builtins.print, not the module's logging shim: the notice has to reach
        # the terminal even once logging is being torn down at interpreter exit.
        with patch("weightslab.utils.logs.builtins.print") as p:
            logs._print_log_location()
        p.assert_called_once()
        self.assertIn(logs._LOG_FILE_PATH, p.call_args.args[0])

    def test_set_log_directory_moves_log_and_reopens_handler(self):
        logs.setup_logging("DEBUG", log_to_file=True)
        old_path = logs._LOG_FILE_PATH

        tmpdir = self._mkdtemp()
        logs.set_log_directory(tmpdir)
        self.assertNotEqual(old_path, logs._LOG_FILE_PATH)
        self.assertTrue(logs._LOG_FILE_PATH.startswith(tmpdir))
        self.assertTrue(os.path.exists(logs._LOG_FILE_PATH))
        self.assertFalse(os.path.exists(old_path))

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


class TestLevelResolution(LogsTestBase):
    def test_resolve_level_accepts_names_numbers_and_garbage(self):
        self.assertEqual(logs._resolve_level("debug"), logging.DEBUG)
        self.assertEqual(logs._resolve_level("WARNING"), logging.WARNING)
        self.assertEqual(logs._resolve_level(" Error "), logging.ERROR)
        self.assertEqual(logs._resolve_level(25), 25)
        self.assertEqual(logs._resolve_level("25"), 25)
        # Unknown / empty / None fall back instead of raising, so a typo in
        # WEIGHTSLAB_LOG_LEVEL degrades logging rather than killing the import.
        self.assertEqual(logs._resolve_level("nonsense"), logging.INFO)
        self.assertEqual(logs._resolve_level(""), logging.INFO)
        self.assertEqual(logs._resolve_level(None), logging.INFO)
        self.assertEqual(logs._resolve_level(None, default=logging.NOTSET), logging.NOTSET)

    def test_resolve_level_knows_the_custom_watchdog_level(self):
        from weightslab.watchdog.log_level import WATCHDOG
        self.assertEqual(logs._resolve_level("WATCHDOG"), WATCHDOG)


class TestConsoleAndFileLevelsAreIndependent(LogsTestBase):
    """The terminal honours WEIGHTSLAB_LOG_LEVEL; the file keeps everything."""

    def test_file_keeps_debug_when_console_is_info(self):
        logs.setup_logging("INFO", log_to_file=True)

        # The root logger gates records before any handler sees them, so it has
        # to be wide open or the file handler's own level is moot.
        self.assertEqual(logging.getLogger().level, logging.NOTSET)
        self.assertEqual(logs._CONSOLE_HANDLER.level, logging.INFO)
        self.assertEqual(logs._FILE_HANDLER.level, logging.NOTSET)

        logging.getLogger("test.independent").debug("debug-marker")
        logging.getLogger("test.independent").info("info-marker")

        contents = self._log_file_contents()
        self.assertIn("debug-marker", contents)
        self.assertIn("info-marker", contents)

    def test_terminal_filters_what_the_file_still_records(self):
        logs.setup_logging("WARNING", log_to_file=True)
        terminal = io.StringIO()
        logs._CONSOLE_HANDLER.setStream(terminal)

        log = logging.getLogger("test.console")
        log.debug("debug-marker")
        log.info("info-marker")
        log.warning("warning-marker")

        printed = terminal.getvalue()
        self.assertNotIn("debug-marker", printed)
        self.assertNotIn("info-marker", printed)
        self.assertIn("warning-marker", printed)

        # All three are on disk regardless of what the terminal showed.
        contents = self._log_file_contents()
        for marker in ("debug-marker", "info-marker", "warning-marker"):
            self.assertIn(marker, contents)

    def test_file_level_can_be_capped_by_env_var(self):
        os.environ["WEIGHTSLAB_LOG_FILE_LEVEL"] = "WARNING"
        logs.setup_logging("DEBUG", log_to_file=True)

        self.assertEqual(logs._FILE_HANDLER.level, logging.WARNING)
        # Root sits at the most permissive of the two sinks (DEBUG here).
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

        logging.getLogger("test.capped").debug("debug-marker")
        logging.getLogger("test.capped").warning("warning-marker")

        contents = self._log_file_contents()
        self.assertNotIn("debug-marker", contents)
        self.assertIn("warning-marker", contents)

    def test_file_level_argument_overrides_env_var(self):
        os.environ["WEIGHTSLAB_LOG_FILE_LEVEL"] = "WARNING"
        logs.setup_logging("INFO", log_to_file=True, file_level="DEBUG")
        self.assertEqual(logs._FILE_HANDLER.level, logging.DEBUG)

    def test_without_file_logging_root_keeps_the_console_level(self):
        logs.setup_logging("INFO", log_to_file=False)
        self.assertEqual(logging.getLogger().level, logging.INFO)
        self.assertIsNone(logs._FILE_HANDLER)


class TestLogDirectoryLayout(LogsTestBase):
    """A session log must never move between two directory conventions."""

    def test_experiment_log_dir_is_the_root_plus_subdir(self):
        self.assertEqual(
            logs.experiment_log_dir(os.path.join("a", "b")),
            os.path.join("a", "b", logs.LOG_SUBDIR))

    def test_setup_logging_uses_the_root_log_dir_env_var(self):
        root = self._mkdtemp()
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = root
        logs.setup_logging("INFO", log_to_file=True)

        self.assertEqual(
            os.path.dirname(logs._LOG_FILE_PATH), logs.experiment_log_dir(root))

    def test_relocation_lands_in_the_same_subdir_setup_logging_uses(self):
        logs.setup_logging("INFO", log_to_file=True)
        root = self._mkdtemp()

        logs.set_log_directory(logs.experiment_log_dir(root))

        self.assertEqual(
            os.path.dirname(logs._LOG_FILE_PATH), logs.experiment_log_dir(root))
        self.assertTrue(os.path.exists(logs._LOG_FILE_PATH))

    def test_relocation_carries_history_over_and_keeps_appending(self):
        logs.setup_logging("INFO", log_to_file=True)
        logging.getLogger("test.move").info("before-marker")

        logs.set_log_directory(self._mkdtemp())
        logging.getLogger("test.move").info("after-marker")

        contents = self._log_file_contents()
        self.assertIn("before-marker", contents)
        self.assertIn("after-marker", contents)

    def test_relocation_preserves_the_file_level(self):
        logs.setup_logging("INFO", log_to_file=True)
        logs.set_log_directory(self._mkdtemp())

        self.assertEqual(logs._FILE_HANDLER.level, logging.NOTSET)
        logging.getLogger("test.level").debug("debug-after-move")
        self.assertIn("debug-after-move", self._log_file_contents())

    def test_relocating_to_the_current_directory_is_a_noop(self):
        logs.setup_logging("INFO", log_to_file=True)
        path = logs._LOG_FILE_PATH
        handler = logs._FILE_HANDLER

        logs.set_log_directory(os.path.dirname(path))

        # Same file, same handler: no churn and no repeated "updated" lines.
        self.assertEqual(logs._LOG_FILE_PATH, path)
        self.assertIs(logs._FILE_HANDLER, handler)

    def test_failed_move_still_leaves_a_working_handler(self):
        logs.setup_logging("INFO", log_to_file=True)
        target = self._mkdtemp()

        with patch("weightslab.utils.logs.shutil.move", side_effect=OSError("locked")):
            logs.set_log_directory(target)

        self.assertEqual(os.path.dirname(logs._LOG_FILE_PATH), target)
        logging.getLogger("test.failed_move").info("still-logging")
        self.assertIn("still-logging", self._log_file_contents())


class TestSurvivesExternalLoggingReconfiguration(LogsTestBase):
    """``logging.config.dictConfig`` closes every handler in the process.

    traitlets runs it while building any ``Application`` — which is what
    ipykernel does every time the studio's embedded notebook kernel starts.
    ``_clearExistingHandlers`` closes the handlers but leaves them attached to
    the root logger, so a dead one goes on accepting records and dropping them.
    """

    @staticmethod
    def _external_dictconfig():
        import logging.config
        logging.config.dictConfig({
            "version": 1, "handlers": {}, "loggers": {},
            "disable_existing_loggers": False,
        })

    def test_session_log_keeps_recording_without_any_repair_call(self):
        logs.setup_logging("INFO", log_to_file=True)
        logging.getLogger("test.reconfig").info("before-marker")

        self._external_dictconfig()
        logging.getLogger("test.reconfig").info("after-marker")

        contents = self._log_file_contents()
        self.assertIn("before-marker", contents)
        self.assertIn("after-marker", contents)

    def test_a_mode_w_handler_would_have_lost_the_record(self):
        # Guards the reason _make_file_handler appends: CPython refuses to
        # reopen a closed mode='w' FileHandler (bpo-42378), so the old handler
        # died silently right here.
        logs.setup_logging("INFO", log_to_file=True)
        root = logging.getLogger()
        root.removeHandler(logs._FILE_HANDLER)
        logs._FILE_HANDLER.close()
        truncating = logging.FileHandler(logs._LOG_FILE_PATH, mode="w", encoding="utf-8")
        truncating.setFormatter(logging.Formatter(logs.FORMAT, datefmt=logs.DATE_FORMAT))
        root.addHandler(truncating)
        logs._FILE_HANDLER = truncating

        self._external_dictconfig()
        logging.getLogger("test.reconfig").info("after-marker")

        self.assertIn(truncating, root.handlers)   # still attached...
        self.assertIsNone(truncating.stream)       # ...but stream gone
        self.assertNotIn("after-marker", self._log_file_contents())

    def test_ensure_logging_intact_reopens_a_closed_file_handler(self):
        logs.setup_logging("INFO", log_to_file=True)
        logs._FILE_HANDLER.close()
        self.assertIsNone(logs._FILE_HANDLER.stream)

        self.assertTrue(logs.ensure_logging_intact())

        logging.getLogger("test.repair").info("repaired-marker")
        self.assertIn("repaired-marker", self._log_file_contents())

    def test_ensure_logging_intact_reattaches_detached_handlers(self):
        logs.setup_logging("INFO", log_to_file=True)
        root = logging.getLogger()
        root.handlers = []

        self.assertTrue(logs.ensure_logging_intact())

        self.assertIn(logs._CONSOLE_HANDLER, root.handlers)
        self.assertIn(logs._FILE_HANDLER, root.handlers)

    def test_ensure_logging_intact_restores_the_root_level(self):
        logs.setup_logging("INFO", log_to_file=True)
        logging.getLogger().setLevel(logging.CRITICAL)

        self.assertTrue(logs.ensure_logging_intact())
        self.assertEqual(logging.getLogger().level, logging.NOTSET)

    def test_ensure_logging_intact_is_a_noop_when_nothing_is_broken(self):
        logs.setup_logging("INFO", log_to_file=True)
        self.assertFalse(logs.ensure_logging_intact())

    def test_ensure_logging_intact_without_setup_does_not_raise(self):
        self.assertFalse(logs.ensure_logging_intact())


class TestProgressChannel(LogsTestBase):
    """The tqdm mirror belongs in the file, not on top of the live bar."""

    def test_progress_records_reach_the_file_but_not_the_terminal(self):
        logs.setup_logging("INFO", log_to_file=True)
        terminal = io.StringIO()
        logs._CONSOLE_HANDLER.setStream(terminal)

        logging.getLogger(logs.PROGRESS_LOGGER_NAME).info("Training: 120 steps | loss=0.4")
        logging.getLogger("test.ordinary").info("ordinary-marker")

        printed = terminal.getvalue()
        self.assertNotIn("Training: 120 steps", printed)
        self.assertIn("ordinary-marker", printed)

        contents = self._log_file_contents()
        self.assertIn("Training: 120 steps", contents)

    def test_the_terminal_handler_never_prints_progress_itself(self):
        # The terminal copy comes from tqdm.write (see tqdm_logging), which
        # redraws the bar around it. A handler write would land mid-bar, so the
        # filter stays on regardless of WEIGHTSLAB_TQDM_LOG_TO_TERMINAL.
        os.environ["WEIGHTSLAB_TQDM_LOG_TO_TERMINAL"] = "1"
        logs.setup_logging("INFO", log_to_file=True)
        terminal = io.StringIO()
        logs._CONSOLE_HANDLER.setStream(terminal)

        logging.getLogger(logs.PROGRESS_LOGGER_NAME).info("Training: 120 steps")
        self.assertNotIn("Training: 120 steps", terminal.getvalue())
        self.assertIn("Training: 120 steps", self._log_file_contents())


class TestFlushLogs(LogsTestBase):
    def test_flush_logs_without_a_handler_is_a_noop(self):
        logs.flush_logs()  # must not raise

    def test_flush_logs_pushes_records_to_disk(self):
        logs.setup_logging("INFO", log_to_file=True)
        logging.getLogger("test.flush").info("flushed-marker")
        logs.flush_logs()

        with open(logs._LOG_FILE_PATH, encoding="utf-8") as handle:
            self.assertIn("flushed-marker", handle.read())


class TestSessionLogSurvivesItsDirectory(unittest.TestCase):
    """A log moved into a temporary root_log_dir that is then deleted must not
    make every later log call in the process raise FileNotFoundError."""

    def test_a_deleted_log_directory_never_fails_the_caller(self):
        root = tempfile.mkdtemp(prefix="wl-log-gone-")
        self.addCleanup(shutil.rmtree, root, True)
        path = os.path.join(root, "weightslab_logs", "session.log")
        os.makedirs(os.path.dirname(path))
        handler = logs._make_file_handler(path)
        self.addCleanup(handler.close)
        logger = logging.getLogger("test.log_dir_gone")
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        logger.propagate = False
        self.addCleanup(setattr, logger, "propagate", True)
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)

        logger.info("before")
        handler.close()                       # set_log_directory / dictConfig close it
        shutil.rmtree(os.path.dirname(path))  # the test's tearDown removes the dir

        logger.info("after")                  # used to raise FileNotFoundError here
        handler.flush()
        with open(path, encoding="utf-8") as fh:
            self.assertIn("after", fh.read())


if __name__ == "__main__":
    unittest.main()
