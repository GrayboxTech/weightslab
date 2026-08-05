"""Tests for the ways to ask for an experiment report other than clicking the
Studio button:

- ``wl.ai_report_generation(...)`` — the public Python API (weightslab.src)
- ``report`` — the interactive CLI console command (weightslab.backend.cli)
- ``DataService._agent_generate_experiment_report`` — the agent action the
  chat bar and the Studio button both trigger

All three are thin resolvers over ``reporting.generate_report`` (covered in
test_reporting.py), so what matters here is what they resolve and forward:
the experiment dir, the logger, the dataframe, and the agent's LLM as the
narrative writer — plus their degradation when any of those is missing.
"""

import os
import tempfile
import types
import unittest
from unittest import mock
from unittest.mock import MagicMock

import weightslab.backend.cli as cli_backend
from weightslab import src as wl_src
from weightslab.backend.cli import _handle_command
from weightslab.backend.logger import LoggerQueue


def _lg_with(*signal_names) -> LoggerQueue:
    """A standalone logger seeded with a few points per signal. Same
    checkpoint-manager isolation as tests/test_reporting.py's ``_lg`` — a
    logger built while another test's manager is registered would otherwise
    adopt that experiment's on-disk history."""
    with mock.patch("weightslab.backend.logger.get_checkpoint_manager",
                    return_value=None):
        lg = LoggerQueue(register=False)
    lg.chkpt_manager = None
    for name in signal_names:
        for step, value in enumerate([1.0, 0.8, 0.6, 0.4]):
            lg.add_scalars(name, {"agg": value}, step, {}, aggregate_by_step=False)
    return lg


class _ReportEntryPointCase(unittest.TestCase):
    """Patches src's resolution points so no live experiment is needed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logger_q = _lg_with("train_loss")

        self.agent = MagicMock()
        self.agent.generate_report_narrative.return_value = "Training looks healthy."
        self.service = MagicMock()
        self.service._root_log_dir = self.tmp.name
        self.service._agent = self.agent

        patches = [
            mock.patch.object(wl_src, "_live_data_service", return_value=self.service),
            mock.patch.object(wl_src, "get_logger", return_value=self.logger_q),
            mock.patch.object(wl_src, "get_dataframe", return_value=None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class TestAiReportGeneration(_ReportEntryPointCase):

    def test_writes_a_report_and_returns_its_path(self):
        path = wl_src.ai_report_generation()

        self.assertTrue(os.path.isfile(path))
        self.assertEqual(os.path.basename(os.path.dirname(path)), "reports")
        self.assertTrue(path.startswith(os.path.abspath(self.tmp.name))
                        or path.startswith(self.tmp.name))

    def test_narrative_comes_from_the_live_agent(self):
        path = wl_src.ai_report_generation()

        self.agent.generate_report_narrative.assert_called_once()
        summary = self.agent.generate_report_narrative.call_args[0][0]
        self.assertIn("train_loss", summary)
        self.assertNotIn("plot_b64", summary)  # bounded, plot-free payload
        self.assertIn("Training looks healthy.", open(path, encoding="utf-8").read())

    def test_use_agent_false_skips_the_llm_entirely(self):
        path = wl_src.ai_report_generation(use_agent=False)

        self.agent.generate_report_narrative.assert_not_called()
        self.assertIn("No narrative was generated", open(path, encoding="utf-8").read())

    def test_agent_failure_still_produces_a_report(self):
        self.agent.generate_report_narrative.side_effect = RuntimeError("Agent not configured")

        path = wl_src.ai_report_generation()

        self.assertTrue(os.path.isfile(path))
        self.assertIn("No narrative was generated", open(path, encoding="utf-8").read())

    def test_no_live_service_reports_without_an_analysis(self):
        with mock.patch.object(wl_src, "_live_data_service", return_value=None):
            path = wl_src.ai_report_generation(root_log_dir=self.tmp.name)

        self.assertTrue(os.path.isfile(path))
        self.agent.generate_report_narrative.assert_not_called()

    def test_explicit_output_path_and_signals_are_honored(self):
        self.logger_q = _lg_with("train_loss", "val_accuracy")
        with mock.patch.object(wl_src, "get_logger", return_value=self.logger_q):
            out = os.path.join(self.tmp.name, "chosen.html")
            path = wl_src.ai_report_generation(signals=["train_loss"], output_path=out)

        self.assertEqual(os.path.abspath(path), os.path.abspath(out))
        html = open(path, encoding="utf-8").read()
        self.assertIn("train_loss", html)
        self.assertNotIn("val_accuracy", html)

    def test_dataframe_stats_are_included_when_a_manager_is_registered(self):
        import pandas as pd

        df = pd.DataFrame({
            "sample_id": [1, 2, 3],
            "origin": ["train", "train", "val"],
            "discarded": [False, False, True],
        }).set_index("sample_id")
        manager = MagicMock()
        manager.get_combined_df.return_value = df

        with mock.patch.object(wl_src, "get_dataframe", return_value=manager):
            path = wl_src.ai_report_generation(use_agent=False)

        html = open(path, encoding="utf-8").read()
        self.assertIn("3</b> total samples", html)
        self.assertIn("train: 2", html)

    def test_no_experiment_dir_raises(self):
        with mock.patch.object(wl_src, "_live_data_service", return_value=None), \
             mock.patch.object(wl_src, "get_checkpoint_manager", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                wl_src.ai_report_generation()
        self.assertIn("no experiment directory", str(ctx.exception))

    def test_no_logger_raises(self):
        with mock.patch.object(wl_src, "get_logger", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                wl_src.ai_report_generation()
        self.assertIn("no experiment logger", str(ctx.exception))

    def test_exported_on_the_package(self):
        import weightslab as wl

        self.assertIn("ai_report_generation", wl.__all__)
        self.assertTrue(callable(wl.ai_report_generation))


class TestCliReportCommand(_ReportEntryPointCase):

    def setUp(self):
        super().setUp()
        cli_backend.set_cli_agent(None)
        cli_backend.set_cli_data_service(None)
        self.addCleanup(cli_backend.set_cli_data_service, None)

    def test_report_writes_a_file_and_summarizes_it(self):
        result = _handle_command("report")

        self.assertTrue(result["ok"])
        self.assertTrue(os.path.isfile(result["path"]))
        self.assertEqual(result["signals"], 1)
        self.assertTrue(result["analysis"])
        self.assertIn("Report written to", result["message"])

    def test_positional_and_comma_separated_signal_filters(self):
        self.logger_q = _lg_with("train_loss", "val_accuracy")
        with mock.patch.object(wl_src, "get_logger", return_value=self.logger_q):
            positional = _handle_command("report train_loss")
            comma = _handle_command("report --signals train_loss,val_accuracy")

        self.assertEqual(positional["signals"], 1)
        self.assertEqual(comma["signals"], 2)

    def test_no_agent_flag_skips_the_analysis(self):
        result = _handle_command("report --no-agent")

        self.assertTrue(result["ok"])
        self.assertFalse(result["analysis"])
        self.assertIn("no written analysis", result["message"])
        self.agent.generate_report_narrative.assert_not_called()

    def test_output_flag_selects_the_file(self):
        out = os.path.join(self.tmp.name, "cli_report.html")
        result = _handle_command(f"report --output {out}")

        self.assertTrue(result["ok"])
        self.assertEqual(os.path.abspath(result["path"]), os.path.abspath(out))

    def test_unknown_flag_returns_usage(self):
        result = _handle_command("report --nope")

        self.assertFalse(result["ok"])
        self.assertIn("usage: report", result["error"])

    def test_failure_is_reported_not_raised(self):
        with mock.patch.object(wl_src, "get_logger", return_value=None):
            result = _handle_command("report")

        self.assertFalse(result["ok"])
        self.assertIn("no experiment logger", result["error"])

    def test_report_is_listed_in_console_help(self):
        help_payload = _handle_command("help")

        self.assertIn("report", help_payload["commands"])
        self.assertIn("report_examples", help_payload)


class TestAgentActionSharesTheSamePath(unittest.TestCase):
    """The chat/Studio-button action must keep producing the same artifact as
    the Python and CLI entry points — it is the same ``generate_report`` call,
    only with the live service's own logger/dataframe/agent."""

    def _run_action(self, root_log_dir, logger_q, narrative_fn):
        from weightslab.trainer.services.data_service import DataService

        service = types.SimpleNamespace(
            _resolve_checkpoint_manager=lambda: None,
            _root_log_dir=root_log_dir,
            _df_manager=None,
            _agent=types.SimpleNamespace(generate_report_narrative=narrative_fn),
        )
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=logger_q):
            return DataService._agent_generate_experiment_report(service)

    def test_action_writes_the_report_and_names_it_in_the_reply(self):
        logger_q = _lg_with("train_loss")
        with tempfile.TemporaryDirectory() as tmp:
            message = self._run_action(tmp, logger_q, lambda _s: "Healthy run.")

            self.assertIn("generated experiment report (1 signal(s))", message)
            path = message.split(" at ", 1)[1]
            self.assertTrue(os.path.isfile(path))
            self.assertIn("Healthy run.", open(path, encoding="utf-8").read())

    def test_action_flags_a_missing_analysis_instead_of_failing(self):
        logger_q = _lg_with("train_loss")

        def boom(_summary):
            raise RuntimeError("Agent not configured")

        with tempfile.TemporaryDirectory() as tmp:
            message = self._run_action(tmp, logger_q, boom)

        self.assertIn("no written analysis", message)

    def test_action_reports_failures_as_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            message = self._run_action(tmp, None, lambda _s: "unused")
        self.assertIn("no experiment logger available", message)


if __name__ == "__main__":
    unittest.main()
