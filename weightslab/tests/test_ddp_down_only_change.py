"""Unit tests for the DOWN_ONLY change gate that drives iterator invalidation.

After removing the (redundant) shm deny-list mirror, iterator invalidation on a
discard/tag is gated by _down_only_changed: a pandas before/after diff that
returns True only when a DOWN_ONLY (deny-list / tags) value actually changes.

The critical property is the DDP no-respawn guarantee: rank-N re-applies the SAME
reconciled deny-list snapshot every step, and that re-apply must NOT report a
change (else workers respawn every step and throughput collapses).
"""
import unittest

import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager


class DownOnlyChangeTests(unittest.TestCase):
    def _mgr_with(self, df):
        m = LedgeredDataFrameManager(
            enable_flushing_threads=False, enable_h5_persistence=False)
        m._df = df.copy()
        return m

    def _row(self, sid, **cols):
        d = pd.DataFrame([cols], index=[str(sid)])
        d.index.name = "sample_id"
        return d

    def test_new_discard_is_a_change(self):
        m = self._mgr_with(self._row("1", discarded=False))
        self.assertTrue(m._down_only_changed(self._row("1", discarded=True)))

    def test_reapplying_same_value_is_not_a_change(self):
        # The DDP no-respawn invariant: same snapshot re-applied → no change.
        m = self._mgr_with(self._row("1", discarded=True))
        self.assertFalse(m._down_only_changed(self._row("1", discarded=True)))

    def test_non_down_only_column_is_not_a_change(self):
        m = self._mgr_with(self._row("1", discarded=False, last_seen=5))
        # last_seen is an UP column, not DOWN_ONLY → must not trigger invalidation.
        self.assertFalse(m._down_only_changed(self._row("1", last_seen=9)))

    def test_user_tags_change_detected(self):
        m = self._mgr_with(self._row("1", user_tags=["a"]))
        self.assertTrue(m._down_only_changed(self._row("1", user_tags=["a", "b"])))
        # Same list re-applied → no change (DDP reconcile re-apply).
        m2 = self._mgr_with(self._row("1", user_tags=["a", "b"]))
        self.assertFalse(m2._down_only_changed(self._row("1", user_tags=["a", "b"])))

    def test_brand_new_down_only_column(self):
        m = self._mgr_with(self._row("1", last_seen=1))   # no 'discarded' col yet
        self.assertTrue(m._down_only_changed(self._row("1", discarded=True)))

    def test_undiscard_is_a_change(self):
        m = self._mgr_with(self._row("1", discarded=True))
        self.assertTrue(m._down_only_changed(self._row("1", discarded=False)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
