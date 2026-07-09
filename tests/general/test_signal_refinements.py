"""Unit tests for the signal-refinements changes: per-sample query cache,
value-at-step read + staging fast path, cycle detection, ctx.logits, freshness,
and the reactive-derived gather skip."""
import unittest

import numpy as np
import torch

import weightslab as wl
from weightslab.backend.logger import LoggerQueue
from weightslab.src import (
    _REGISTERED_SIGNALS, _detect_signal_cycles, _gather_inputs_fresh,
    BatchSignalContext, SignalContext, StaleSignalError,
)


def _lg():
    return LoggerQueue(register=False)


class TestQueryCache(unittest.TestCase):
    def test_cached_and_fresh_copy(self):
        lg = _lg()
        lg._stage_sample_row("m", "h", "1", 0, 1.5)
        lg._stage_sample_row("m", "h", "2", 0, 2.5)
        r1 = lg.query_per_sample("m", sample_ids=["1", "2"])
        r2 = lg.query_per_sample("m", sample_ids=["1", "2"])
        self.assertEqual(len(r1), 2)
        self.assertIsNot(r1, r2)                       # fresh list each call
        self.assertGreaterEqual(lg._qps_cache.cache_info().hits, 1)  # 2nd read hit

    def test_invalidated_on_write(self):
        lg = _lg()
        lg._stage_sample_row("m", "h", "1", 0, 1.0)
        self.assertEqual(len(lg.query_per_sample("m", sample_ids=["1"])), 1)
        lg._stage_sample_row("m", "h", "1", 1, 9.0)    # new row bumps version
        self.assertEqual(len(lg.query_per_sample("m", sample_ids=["1"])), 2)

    def test_step_scoped_clear(self):
        lg = _lg()
        lg._stage_sample_row("m", "h", "1", 0, 1.0)
        lg.query_per_sample("m", sample_ids=["1"])
        lg._stage_sample_row("m", "h", "2", 1, 2.0)    # step advance -> clear
        self.assertEqual(lg._qps_cache_step, 1)


class TestValueAtStep(unittest.TestCase):
    def test_staging_fast_path(self):
        lg = _lg()
        lg._stage_sample_row("m", "h", "5", 3, 4.0)    # staged, not flushed
        at = dict((int(s), v) for s, v in lg.query_per_sample_at_step("m", [5], 3))
        self.assertEqual(at, {5: 4.0})

    def test_absent_at_other_step(self):
        lg = _lg()
        lg._stage_sample_row("m", "h", "5", 3, 4.0)
        self.assertEqual(lg.query_per_sample_at_step("m", [5], 99), [])


class TestCycleDetection(unittest.TestCase):
    def setUp(self): _REGISTERED_SIGNALS.clear()
    def tearDown(self): _REGISTERED_SIGNALS.clear()

    def test_self_loop_and_two_cycle(self):
        @wl.signal(name="sig/self", inputs=["sig/self"], batched=True)
        def _s(b): return b
        @wl.signal(name="sig/X", inputs=["sig/Y"], batched=True)
        def _x(b): return b
        @wl.signal(name="sig/Y", inputs=["sig/X"], batched=True)
        def _y(b): return b
        keys = {frozenset(c) for c in _detect_signal_cycles()}
        self.assertIn(frozenset(["sig/self"]), keys)
        self.assertIn(frozenset(["sig/X", "sig/Y"]), keys)

    def test_dag_has_no_cycle(self):
        @wl.signal(name="sig/a", inputs=["train/loss"], batched=True)   # base leaf
        def _a(b): return b
        @wl.signal(name="sig/b", inputs=["sig/a"], batched=True)
        def _b(b): return b
        self.assertEqual(_detect_signal_cycles(), [])


class TestContexts(unittest.TestCase):
    def test_batch_context_carries_logits(self):
        b = BatchSignalContext(sample_ids=[1, 2], subscribed_values=[0.1, 0.2],
                               logits=torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        self.assertEqual(tuple(b.logits.shape), (2, 3))

    def test_sample_context_carries_logits(self):
        c = SignalContext(sample_id=1, dataframe=None, logits=torch.tensor([1., 2., 3.]))
        self.assertEqual(tuple(c.logits.shape), (3,))

    def test_stale_signal_raises(self):
        b = BatchSignalContext(sample_ids=[1], subscribed_values=[0.0], logger=_lg(), step=5)
        with self.assertRaises(StaleSignalError):
            b.latest("sig/never", require_fresh=True)


class TestGatherFreshCache(unittest.TestCase):
    def setUp(self): _REGISTERED_SIGNALS.clear()
    def tearDown(self): _REGISTERED_SIGNALS.clear()

    def test_reactive_derived_not_queried_when_absent(self):
        @wl.signal(name="sig/derived", inputs=["x"], batched=True)   # reactive
        def _d(b): return b
        lg = _lg()
        calls = {"n": 0}
        orig = lg.query_per_sample_at_step
        lg.query_per_sample_at_step = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1]
        res = _gather_inputs_fresh(lg, ["sig/derived"], [1], 0, fresh_cache={})
        self.assertIsNone(res)          # not fired yet -> skip
        self.assertEqual(calls["n"], 0)  # and NOT a ledger query (no flush)

    def test_reactive_derived_read_from_cache(self):
        @wl.signal(name="sig/derived", inputs=["x"], batched=True)
        def _d(b): return b
        res = _gather_inputs_fresh(_lg(), ["sig/derived"], [1, 2], 0,
                                   fresh_cache={"sig/derived": np.array([1.0, 2.0])})
        self.assertIsNotNone(res)
        np.testing.assert_array_equal(res["sig/derived"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
