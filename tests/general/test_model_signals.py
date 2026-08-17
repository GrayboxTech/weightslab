"""Per-step model signals: `wl.save_model_signals` + `wl.track_model_signals`.

Split into two halves that need very different setups:

  TestSaveModelSignals   the write path alone, against a mock logger. Pins the
                         contract that makes these curves work at all --
                         `aggregate_by_step=False` with no per-sample map, which
                         is what appends one point per step instead of bucketing.
  TestTrackModelSignals  the collector, against a real 3-layer model and a real
                         training step. Asserts on values, not just on calls:
                         the global norm has an arithmetic answer, and getting
                         it right is the whole point of combining layers in
                         quadrature rather than summing their norms.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

import weightslab as wl
from weightslab.components.model_signals import (
    METRICS,
    ModelSignalTracker,
    _iter_layers,
)
from weightslab.components.tracking import TrackingMode


class TestSaveModelSignals(unittest.TestCase):
    def _capture(self, signals, step=7):
        """Run save_model_signals against a mock logger; return its calls."""
        mock_logger = MagicMock()
        with patch("weightslab.src.get_logger", return_value=mock_logger), \
             patch("weightslab.src._get_step", side_effect=lambda step=None: step):
            wl.save_model_signals(signals, step=step)
        return mock_logger.add_scalars.call_args_list

    def test_emits_one_point_per_signal_in_immediate_mode(self):
        calls = self._capture({
            "metrics/global/grad_norm": 1.5,
            "metrics/layer/3/weights_norm": torch.tensor(2.5),
        })
        self.assertEqual(len(calls), 2)
        for call in calls:
            name = call.args[0]
            self.assertEqual(call.args[1], {name: unittest.mock.ANY})
            self.assertEqual(call.kwargs["global_step"], 7)
            # The contract that makes a step-keyed curve behave: no per-sample
            # map, and no per-step aggregation bucket to be averaged into.
            self.assertIsNone(call.kwargs["signal_per_sample"])
            self.assertFalse(call.kwargs["aggregate_by_step"])

        by_name = {c.args[0]: c.args[1][c.args[0]] for c in calls}
        self.assertAlmostEqual(by_name["metrics/global/grad_norm"], 1.5, places=6)
        self.assertAlmostEqual(by_name["metrics/layer/3/weights_norm"], 2.5, places=6)

    def test_reduces_a_tensor_to_one_scalar(self):
        calls = self._capture({"metrics/global/x": torch.tensor([1.0, 2.0, 3.0])})
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0].args[1]["metrics/global/x"], 2.0, places=6)

    def test_drops_non_finite_points_without_raising(self):
        """A diverged run should break the curve, not rescale the whole axis."""
        calls = self._capture({
            "metrics/global/exploded": float("inf"),
            "metrics/global/nan": float("nan"),
            "metrics/global/fine": 0.25,
        })
        self.assertEqual([c.args[0] for c in calls], ["metrics/global/fine"])

    def test_empty_and_non_numeric_are_no_ops(self):
        self.assertEqual(self._capture({}), [])
        self.assertEqual(self._capture({"metrics/global/text": "not a number"}), [])


class _ThreeLayer(nn.Module):
    """Two parameterized layers with a ReLU between them.

    Small enough that the expected global gradient norm can be written out by
    hand, which is what test_global_norm_combines_layers_in_quadrature needs.
    """

    def __init__(self):
        super().__init__()
        self.input_shape = (1, 4)
        self.fc1 = nn.Linear(4, 3)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TestTrackModelSignals(unittest.TestCase):
    def setUp(self):
        self.model = _ThreeLayer()
        # The tracker reads tracking_mode to tell a training step from an eval
        # pass; a bare nn.Module has no such attribute, so stand in for what
        # guard_training_context would set.
        self.model.tracking_mode = TrackingMode.TRAIN
        self.model.get_age = lambda: 0
        self.trackers = []

    def tearDown(self):
        for tracker in self.trackers:
            tracker.remove()

    def _tracker(self, **kwargs):
        # `_ensure_flush_hooked` looks up a watched optimizer to wrap; there is
        # no ledger in this test, so short it out and drive flush() by hand.
        tracker = ModelSignalTracker(self.model, **kwargs)
        tracker._flush_hooked = True
        self.trackers.append(tracker)
        return tracker

    def _train_step(self, tracker, flush=True):
        """One real forward/backward, then flush. Returns the emitted map."""
        emitted = {}
        with patch("weightslab.src.save_model_signals",
                   side_effect=lambda signals, step=None: emitted.update(signals)):
            self.model.zero_grad()
            out = self.model(torch.randn(5, 4))
            out.pow(2).mean().backward()
            if flush:
                tracker.flush()
        return emitted

    def test_layer_ids_come_from_the_module_id_the_rest_of_wl_uses(self):
        rows = _iter_layers(self.model)
        self.assertEqual([type(m).__name__ for _, m in rows],
                         ["Linear", "ReLU", "Linear"])

    def test_emits_every_metric_for_every_eligible_layer(self):
        emitted = self._train_step(self._tracker())

        # Parameterized layers get norms; the ReLU gets activations only.
        for metric in ("grad_norm", "weights_norm"):
            names = [n for n in emitted if n.endswith("/" + metric)
                     and n.startswith("metrics/layer/")]
            self.assertEqual(len(names), 2, f"{metric}: {names}")
        for metric in ("activation_mean", "activation_std",
                       "activation_max", "activation_min"):
            names = [n for n in emitted if n.endswith("/" + metric)]
            self.assertEqual(len(names), 3, f"{metric}: {names}")

        self.assertIn("metrics/global/grad_norm", emitted)
        self.assertIn("metrics/global/weights_norm", emitted)

    def test_global_norm_combines_layers_in_quadrature(self):
        """sqrt(sum of squares), not a sum of per-layer norms.

        The wrong version (summing norms) is an L1 over L2s and always reads
        high, so it silently misreports every run rather than failing loudly.
        """
        emitted = self._train_step(self._tracker(metrics=["grad_norm"]))

        per_layer = [v for k, v in emitted.items()
                     if k.startswith("metrics/layer/") and k.endswith("/grad_norm")]
        expected = sum(v ** 2 for v in per_layer) ** 0.5
        self.assertAlmostEqual(emitted["metrics/global/grad_norm"], expected, places=5)
        # And it is genuinely below the naive sum, so this asserts something.
        self.assertLess(emitted["metrics/global/grad_norm"], sum(per_layer))

    def test_relu_activation_min_is_exactly_zero(self):
        """A cheap end-to-end sanity check on the activation values themselves."""
        emitted = self._train_step(self._tracker(metrics=["activation_min"]))
        relu_id = next(lid for lid, m in _iter_layers(self.model)
                       if isinstance(m, nn.ReLU))
        self.assertEqual(emitted[f"metrics/layer/{relu_id}/activation_min"], 0.0)

    def test_collects_nothing_outside_a_training_context(self):
        tracker = self._tracker()
        self.model.tracking_mode = TrackingMode.EVAL
        self.assertEqual(self._train_step(tracker), {})

    def test_every_n_steps_skips_unsampled_steps(self):
        tracker = self._tracker(every_n_steps=10)

        self.model.get_age = lambda: 3
        self.assertEqual(self._train_step(tracker), {})

        self.model.get_age = lambda: 20
        self.assertTrue(self._train_step(tracker))

    def test_layer_ids_filter_restricts_what_is_tracked(self):
        first_id = _iter_layers(self.model)[0][0]
        emitted = self._train_step(self._tracker(layer_ids=[first_id]))
        layer_names = {n for n in emitted if n.startswith("metrics/layer/")}
        self.assertTrue(layer_names)
        self.assertEqual({n.split("/")[2] for n in layer_names}, {first_id})

    def test_include_global_false_drops_only_the_global_curves(self):
        emitted = self._train_step(self._tracker(include_global=False))
        self.assertFalse([n for n in emitted if n.startswith("metrics/global/")])
        self.assertTrue([n for n in emitted if n.startswith("metrics/layer/")])

    def test_unknown_metric_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as ctx:
            ModelSignalTracker(self.model, metrics=["grad_norm", "nope"])
        self.assertIn("nope", str(ctx.exception))

    def test_flush_resets_so_a_skipped_step_emits_nothing(self):
        tracker = self._tracker()
        self.assertTrue(self._train_step(tracker))
        # No new forward/backward: everything was consumed by the first flush,
        # so only the always-available weight norms remain.
        emitted = {}
        with patch("weightslab.src.save_model_signals",
                   side_effect=lambda signals, step=None: emitted.update(signals)):
            tracker.flush()
        self.assertFalse([n for n in emitted if "grad_norm" in n])
        self.assertFalse([n for n in emitted if "activation" in n])

    def test_remove_is_idempotent_and_stops_collection(self):
        tracker = self._tracker()
        tracker.remove()
        tracker.remove()
        self.assertEqual(self._train_step(tracker), {})

    def test_metrics_constant_matches_what_the_tracker_accepts(self):
        """Guards the docs: METRICS is the published list of signal names."""
        emitted = self._train_step(self._tracker(metrics=METRICS))
        suffixes = {n.rsplit("/", 1)[1] for n in emitted}
        self.assertEqual(suffixes, set(METRICS))


if __name__ == "__main__":
    unittest.main()
