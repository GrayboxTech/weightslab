"""Focused unit tests for the DDP outbox DELTA optimization (no DDP spawn).

These prove the three correctness properties of the delta change directly,
in milliseconds, instead of relying on the multi-minute YOLO scenarios:

  1. local_df_writes emits only CHANGED rows since the last flush (the delta),
     and nothing when nothing moved.
  2. merge_df_writes seeds rank-0's current value first, so a stale/lower delta
     CANNOT regress a MAX column (last_seen), while a higher delta still raises
     it and LATEST resolves to the newest delta.
  3. local_signal_triples advances a per-(graph, exp_hash) cursor, emitting only
     triples appended since the last flush (and re-sending all if the buffer
     shrank under it, e.g. after a restore).
"""
import unittest

import pandas as pd

from weightslab.backend import ledgers
import weightslab.components.parallel_state as ps


class _FakeDFM:
    """Minimal stand-in for the dataframe manager: holds a df, records upserts,
    and applies them cumulatively so seed-reads see prior merges."""
    def __init__(self, df):
        self._df = df.copy()
        self.upserts = []
        self._outbox_dirty = set()

    def mark_dirty(self, *sids):
        self._outbox_dirty.update(str(s) for s in sids)

    def drain_outbox_dirty(self):
        d = self._outbox_dirty
        self._outbox_dirty = set()
        return d

    def get_combined_df(self, return_proxies=False):
        return self._df.copy()

    def upsert_df(self, df, force_flush=False):
        self.upserts.append(df.copy())
        # Emulate cell overwrite keyed by sample_id (df index = sample_id).
        for sid, row in df.iterrows():
            mask = self._df["sample_id"] == str(sid)
            for col, val in row.items():
                if pd.notna(val):
                    if col not in self._df.columns:
                        self._df[col] = None
                    self._df.loc[mask, col] = val


class _FakeLogger:
    def __init__(self):
        self._signal_history_per_sample = {}

    def add(self, graph, exp_hash, sid, step, val):
        buf = self._signal_history_per_sample.setdefault(graph, {}).setdefault(
            exp_hash, {"sample_ids": [], "steps": [], "values": []})
        buf["sample_ids"].append(str(sid))
        buf["steps"].append(int(step))
        buf["values"].append(float(val))


class DeltaOutboxTests(unittest.TestCase):
    def setUp(self):
        ps.reset_outbox_state()
        self._orig_get_df = ledgers.get_dataframe
        self._orig_get_logger = ledgers.get_logger

    def tearDown(self):
        ledgers.get_dataframe = self._orig_get_df
        ledgers.get_logger = self._orig_get_logger
        ps.reset_outbox_state()

    # -- 1. df delta (driven by the outbox-dirty set) ----------------------
    def test_df_writes_emits_only_dirty_rows(self):
        df = pd.DataFrame({"sample_id": ["1", "2", "3"], "last_seen": [10, 20, 30]})
        fake = _FakeDFM(df)
        ledgers.get_dataframe = lambda: fake

        # Nothing marked dirty → nothing ships (no full-df scan).
        self.assertIsNone(ps.local_df_writes(), "no dirty sids → no delta")

        # Writers touched rows 1 and 3 → only those ship.
        fake.mark_dirty("1", "3")
        delta = ps.local_df_writes()
        self.assertEqual({r["sample_id"] for r in delta}, {"1", "3"})

        # Drained — next flush with nothing new ships nothing.
        self.assertIsNone(ps.local_df_writes(), "dirty set drained → no delta")

        # A write to row 2 → only row 2 ships, with its current value.
        fake._df.loc[fake._df["sample_id"] == "2", "last_seen"] = 25
        fake.mark_dirty("2")
        delta = ps.local_df_writes()
        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]["sample_id"], "2")
        self.assertEqual(delta[0]["last_seen"], 25)

    # -- 2. merge seed: no MAX regression ----------------------------------
    def test_merge_seed_prevents_max_regression(self):
        # rank-0 already holds last_seen=5 for sample A.
        df = pd.DataFrame({"sample_id": ["A"], "last_seen": [5]})
        fake = _FakeDFM(df)
        ledgers.get_dataframe = lambda: fake

        # A stale delta (last_seen=3) from another rank must NOT lower A.
        ps.merge_df_writes([[{"sample_id": "A", "last_seen": 3}]])
        self.assertTrue(fake.upserts, "merge should upsert")
        merged = fake.upserts[-1]
        self.assertEqual(int(merged.loc["A", "last_seen"]), 5,
                         "MAX seeded with existing value must not regress")

        # A higher delta (7) DOES raise it.
        ps.merge_df_writes([[{"sample_id": "A", "last_seen": 7}]])
        self.assertEqual(int(fake.upserts[-1].loc["A", "last_seen"]), 7)

    def test_merge_latest_picks_newest_delta(self):
        df = pd.DataFrame({"sample_id": ["A"], "note": ["old"]})
        fake = _FakeDFM(df)
        ledgers.get_dataframe = lambda: fake
        # object/string col → LATEST; seed is existing-first, delta is later → wins.
        ps.merge_df_writes([[{"sample_id": "A", "note": "new"}]])
        self.assertEqual(fake.upserts[-1].loc["A", "note"], "new")

    # -- 3. signal cursor --------------------------------------------------
    def test_signal_triples_cursor_emits_only_new(self):
        lg = _FakeLogger()
        ledgers.get_logger = lambda: lg
        lg.add("loss", "h", "1", 0, 0.5)
        lg.add("loss", "h", "2", 0, 0.6)

        first = ps.local_signal_triples()
        self.assertEqual(len(first["loss"]["h"]), 2, "first flush sends all triples")

        # No new entries → nothing emitted.
        self.assertEqual(ps.local_signal_triples(), {}, "no new triples → empty")

        # Append two → only those two ship.
        lg.add("loss", "h", "1", 1, 0.4)
        lg.add("loss", "h", "2", 1, 0.45)
        delta = ps.local_signal_triples()
        self.assertEqual(len(delta["loss"]["h"]), 2)
        self.assertEqual({t[1] for t in delta["loss"]["h"]}, {1}, "only step-1 triples")

    def test_signal_cursor_resends_after_buffer_shrinks(self):
        lg = _FakeLogger()
        ledgers.get_logger = lambda: lg
        lg.add("loss", "h", "1", 0, 0.5)
        lg.add("loss", "h", "2", 1, 0.6)
        ps.local_signal_triples()                      # cursor → 2
        # Simulate a restore that rebuilt the buffer shorter than the cursor.
        lg._signal_history_per_sample["loss"]["h"] = {
            "sample_ids": ["9"], "steps": [0], "values": [0.1]}
        delta = ps.local_signal_triples()
        self.assertEqual(len(delta["loss"]["h"]), 1,
                         "buffer shrank under cursor → resend from 0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
