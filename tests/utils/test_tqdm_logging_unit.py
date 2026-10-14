"""Tests for the tqdm -> session log mirror.

A tqdm bar paints itself onto a terminal and never touches ``logging``, so the
log file had no record of a run's own progress. These cover the sampler that
puts it there.
"""

import logging
import os
import unittest
from unittest.mock import patch

from weightslab.utils import tqdm_logging
from weightslab.utils.logs import PROGRESS_LOGGER_NAME


class _FakeBar:
    """Stands in for a tqdm instance: only ``format_dict`` is read."""

    def __init__(self, **fields):
        base = {"prefix": "Training", "n": 0, "total": None,
                "elapsed": 0.0, "rate": None, "postfix": None}
        base.update(fields)
        self.format_dict = base


_ENV_KEYS = ("WEIGHTSLAB_TQDM_LOG_INTERVAL", "WEIGHTSLAB_TQDM_LOG_TO_TERMINAL")


class TqdmLoggingTestBase(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.pop(key, None) for key in _ENV_KEYS}
        # Off by default in tests: the echo is verified explicitly below, and
        # every other test would otherwise print into the runner's output.
        os.environ["WEIGHTSLAB_TQDM_LOG_TO_TERMINAL"] = "0"
        self.addCleanup(self._restore_env)
        self.addCleanup(tqdm_logging.stop_tqdm_log_mirror)

    def _restore_env(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestRender(TqdmLoggingTestBase):
    def test_renders_a_compact_line_without_the_drawn_bar(self):
        bar = _FakeBar(n=2750, elapsed=1752.0, rate=1.57,
                       postfix="train_loss=0.3421 | test_acc=88.3%")
        line = tqdm_logging._render(bar)

        self.assertIn("Training", line)
        self.assertIn("2750 steps", line)
        self.assertIn("1.57 it/s", line)
        self.assertIn("train_loss=0.3421", line)
        # The block-drawing bar carries nothing in a log file.
        self.assertNotIn("█", line)
        self.assertNotIn("\r", line)

    def test_shows_a_percentage_when_the_total_is_known(self):
        line = tqdm_logging._render(_FakeBar(n=25, total=100, elapsed=10.0))
        self.assertIn("25/100", line)
        self.assertIn("25%", line)

    def test_falls_back_to_a_generic_label_without_a_description(self):
        self.assertTrue(tqdm_logging._render(_FakeBar(prefix=None)).startswith("progress:"))

    def test_a_bar_that_cannot_be_read_is_skipped_not_raised(self):
        class Broken:
            @property
            def format_dict(self):
                raise RuntimeError("gone")

        self.assertIsNone(tqdm_logging._render(Broken()))


class TestSampling(TqdmLoggingTestBase):
    def test_logs_one_line_per_bar_to_the_progress_channel(self):
        bar = _FakeBar(n=10, elapsed=5.0, postfix="train_loss=0.5")
        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]), \
             self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO) as captured:
            tqdm_logging._sample_once({})

        self.assertEqual(len(captured.records), 1)
        self.assertIn("train_loss=0.5", captured.output[0])

    def test_an_unchanged_bar_is_not_logged_twice(self):
        bar = _FakeBar(n=10, elapsed=5.0)
        seen = {}
        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]):
            with self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO):
                tqdm_logging._sample_once(seen)
            # A paused run must not fill the log with identical lines.
            with patch.object(tqdm_logging.progress_logger, "info") as info:
                tqdm_logging._sample_once(seen)
        info.assert_not_called()

    def test_a_paused_bar_is_not_relogged_just_because_time_passed(self):
        # Regression: the skip used to compare rendered lines, which carry
        # elapsed time. A stopped run therefore wrote a near-identical entry
        # every interval forever.
        bar = _FakeBar(n=453, elapsed=148.0, rate=2.90, postfix="train_loss=0.1861")
        seen = {}
        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]):
            with self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO):
                tqdm_logging._sample_once(seen)
            for extra in (30.0, 60.0, 90.0):
                bar.format_dict["elapsed"] = 148.0 + extra
                with patch.object(tqdm_logging.progress_logger, "info") as info:
                    tqdm_logging._sample_once(seen)
                info.assert_not_called()

    def test_a_bar_that_moved_is_logged_again(self):
        bar = _FakeBar(n=10, elapsed=5.0)
        seen = {}
        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]):
            with self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO):
                tqdm_logging._sample_once(seen)
            bar.format_dict["n"] = 20
            with self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO) as second:
                tqdm_logging._sample_once(seen)
        self.assertIn("20 steps", second.output[0])

    def test_closed_bars_are_forgotten(self):
        bar = _FakeBar(n=10)
        seen = {}
        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]), \
             self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO):
            tqdm_logging._sample_once(seen)
        self.assertEqual(len(seen), 1)

        with patch.object(tqdm_logging, "_live_bars", return_value=[]):
            tqdm_logging._sample_once(seen)
        self.assertEqual(seen, {})

    def test_no_bars_logs_nothing(self):
        with patch.object(tqdm_logging, "_live_bars", return_value=[]), \
             patch.object(tqdm_logging.progress_logger, "info") as info:
            tqdm_logging._sample_once({})
        info.assert_not_called()


class TestTerminalEcho(TqdmLoggingTestBase):
    """Progress also reaches the terminal, without corrupting the live bar."""

    def test_echoes_through_tqdm_write_by_default(self):
        os.environ.pop("WEIGHTSLAB_TQDM_LOG_TO_TERMINAL", None)
        bar = _FakeBar(n=10, elapsed=5.0, postfix="train_loss=0.5")
        import tqdm as tqdm_mod

        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]), \
             patch.object(tqdm_mod.tqdm, "write") as write, \
             self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO):
            tqdm_logging._sample_once({})

        # tqdm.write, not a bare print: it lifts the bar, prints, and redraws.
        write.assert_called_once()
        self.assertIn("train_loss=0.5", write.call_args.args[0])

    def test_echo_can_be_turned_off(self):
        os.environ["WEIGHTSLAB_TQDM_LOG_TO_TERMINAL"] = "0"
        bar = _FakeBar(n=10, elapsed=5.0)
        import tqdm as tqdm_mod

        with patch.object(tqdm_logging, "_live_bars", return_value=[bar]), \
             patch.object(tqdm_mod.tqdm, "write") as write, \
             self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO):
            tqdm_logging._sample_once({})

        write.assert_not_called()

    def test_a_failing_echo_does_not_stop_the_log_line(self):
        import tqdm as tqdm_mod
        os.environ.pop("WEIGHTSLAB_TQDM_LOG_TO_TERMINAL", None)

        with patch.object(tqdm_logging, "_live_bars", return_value=[_FakeBar(n=1)]), \
             patch.object(tqdm_mod.tqdm, "write", side_effect=OSError("closed")), \
             self.assertLogs(PROGRESS_LOGGER_NAME, level=logging.INFO) as captured:
            tqdm_logging._sample_once({})

        self.assertEqual(len(captured.records), 1)


class TestLifecycle(TqdmLoggingTestBase):
    def test_starts_and_is_idempotent(self):
        self.assertTrue(tqdm_logging.start_tqdm_log_mirror(interval=30))
        first = tqdm_logging._thread
        self.assertTrue(tqdm_logging.start_tqdm_log_mirror(interval=30))
        self.assertIs(tqdm_logging._thread, first)

    def test_a_non_positive_interval_disables_it(self):
        self.assertFalse(tqdm_logging.start_tqdm_log_mirror(interval=0))
        self.assertIsNone(tqdm_logging._thread)

    def test_the_interval_comes_from_the_environment(self):
        os.environ["WEIGHTSLAB_TQDM_LOG_INTERVAL"] = "0"
        self.assertFalse(tqdm_logging.start_tqdm_log_mirror())

        os.environ["WEIGHTSLAB_TQDM_LOG_INTERVAL"] = "not-a-number"
        self.assertEqual(tqdm_logging._interval_from_env(),
                         tqdm_logging.DEFAULT_INTERVAL_SECONDS)

    def test_stop_is_safe_when_never_started(self):
        tqdm_logging.stop_tqdm_log_mirror()  # must not raise

    def test_stop_ends_the_thread(self):
        tqdm_logging.start_tqdm_log_mirror(interval=30)
        thread = tqdm_logging._thread
        tqdm_logging.stop_tqdm_log_mirror()
        self.assertIsNone(tqdm_logging._thread)
        self.assertFalse(thread.is_alive())


class TestLiveBars(TqdmLoggingTestBase):
    def test_reads_real_tqdm_instances(self):
        import tqdm as tqdm_mod

        with open(os.devnull, "w") as sink:
            bar = tqdm_mod.tqdm(total=10, desc="RealBar", file=sink)
            try:
                bar.update(3)
                rendered = [tqdm_logging._render(b) for b in tqdm_logging._live_bars()]
            finally:
                bar.close()

        self.assertTrue(any(line and "RealBar" in line for line in rendered))


if __name__ == "__main__":
    unittest.main()
