"""Unit tests for the on-demand per-signal trajectory path (GetSignalTrajectory
+ _signal_trajectory_curves): dynamic signal-name resolution, the 3-point
minimum, and downsampling to max_points. No always-on `loss_trajectory` stat and
no hardcoded `"loss"` signal name remain."""
import unittest
from types import SimpleNamespace
from unittest import mock

from weightslab.backend.logger import LoggerQueue
from weightslab.trainer.services.data_service import (
    DataService, SIGNAL_TRAJ_MIN_POINTS, SIGNAL_TRAJ_MAX_POINTS,
)


def _logger_with(rows_by_sample, metric="loss_sample"):
    """An isolated in-memory LoggerQueue with per-sample rows staged: {sid: [values...]}.

    ``get_checkpoint_manager`` is stubbed out during construction: otherwise a
    global checkpoint manager left over from an earlier test binds the logger to
    a shared on-disk ``loggers.duckdb`` (see LoggerQueue.__init__), and per-sample
    rows accumulate across tests instead of staying isolated to this one.
    """
    with mock.patch("weightslab.backend.logger.get_checkpoint_manager", return_value=None):
        lg = LoggerQueue(register=False)
    for sid, vals in rows_by_sample.items():
        for step, v in enumerate(vals):
            lg._stage_sample_row(metric, "h", sid, step, float(v))
    return lg


class TestSignalTrajectoryCurves(unittest.TestCase):
    def _service(self):
        return DataService.__new__(DataService)

    def test_omits_samples_below_min_points(self):
        lg = _logger_with({
            "1": [5, 4, 3, 2, 1],   # 5 points -> kept
            "2": [9, 8],            # 2 points -> dropped (< 3)
            "3": [7, 6, 5],         # 3 points -> kept (== min)
        })
        svc = self._service()
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=lg):
            resolved, curves = svc._signal_trajectory_curves("loss_sample", ["1", "2", "3"])
        self.assertEqual(resolved, "loss_sample")
        self.assertEqual(set(curves), {"1", "3"})
        self.assertNotIn("2", curves)
        self.assertTrue(all(len(c) >= SIGNAL_TRAJ_MIN_POINTS for c in curves.values()))
        self.assertEqual(curves["1"], [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_downsamples_to_max_points(self):
        lg = _logger_with({"1": list(range(50))})
        svc = self._service()
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=lg):
            _, curves = svc._signal_trajectory_curves("loss_sample", ["1"], max_points=10)
        self.assertEqual(len(curves["1"]), 10)
        # Endpoints preserved by the even downsample.
        self.assertEqual(curves["1"][0], 0.0)
        self.assertEqual(curves["1"][-1], 49.0)

    def test_default_cap_is_signal_traj_max_points(self):
        n = SIGNAL_TRAJ_MAX_POINTS + 37
        lg = _logger_with({"1": list(range(n))})
        svc = self._service()
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=lg):
            _, curves = svc._signal_trajectory_curves("loss_sample", ["1"])
        self.assertLessEqual(len(curves["1"]), SIGNAL_TRAJ_MAX_POINTS)

    def test_empty_when_no_logger(self):
        svc = self._service()
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=None):
            resolved, curves = svc._signal_trajectory_curves("loss_sample", ["1"])
        self.assertEqual(curves, {})

    def test_dynamic_name_resolution(self):
        # The registered/aggregated graph name is used to resolve the UI spelling;
        # here the stored metric is the resolved name and query matches it.
        lg = _logger_with({"1": [3, 2, 1]}, metric="train-loss-CE")
        svc = self._service()
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=lg):
            resolved, curves = svc._signal_trajectory_curves("train-loss-CE", ["1"])
        self.assertIn("1", curves)


class TestGetSignalTrajectoryRPC(unittest.TestCase):
    def _service(self):
        return DataService.__new__(DataService)

    def test_returns_trajectories(self):
        lg = _logger_with({"1": [5, 4, 3, 2, 1]})
        svc = self._service()
        req = SimpleNamespace(signal_name="loss_sample", sample_ids=["1"], max_points=0)
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=lg):
            resp = svc.GetSignalTrajectory(req, None)
        self.assertTrue(resp.success)
        self.assertEqual(resp.signal_name, "loss_sample")
        self.assertEqual(len(resp.trajectories), 1)
        self.assertEqual(resp.trajectories[0].sample_id, "1")
        self.assertEqual(len(resp.trajectories[0].value), 5)

    def test_requires_signal_name(self):
        svc = self._service()
        req = SimpleNamespace(signal_name="", sample_ids=[], max_points=0)
        resp = svc.GetSignalTrajectory(req, None)
        self.assertFalse(resp.success)

    def test_empty_success_when_insufficient_history(self):
        lg = _logger_with({"1": [9, 8]})  # only 2 points -> below the 3-point min
        svc = self._service()
        req = SimpleNamespace(signal_name="loss_sample", sample_ids=["1"], max_points=0)
        with mock.patch("weightslab.backend.ledgers.get_logger", return_value=lg):
            resp = svc.GetSignalTrajectory(req, None)
        self.assertTrue(resp.success)
        self.assertEqual(len(resp.trajectories), 0)


if __name__ == "__main__":
    unittest.main()
