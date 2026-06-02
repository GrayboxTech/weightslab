"""Unit test for the DOWN-only shm deny-list capacity cap.

The shm vec is indexed directly by int(sample_id), so a single large/sparse uid
(e.g. an inode-based id ~1e8) would otherwise allocate a ~100MB bool array. The
cap skips the fast-path for over-cap ids and falls back to the sampler's pandas
deny-list check (the shm read site is the main-process sampler). This test proves
a huge id neither allocates a giant array nor breaks the small-id fast-path.
"""
import unittest

import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager


class ShmCapTests(unittest.TestCase):
    def _mgr(self):
        # No flush threads / no h5 — we only exercise the shm mirror.
        return LedgeredDataFrameManager(
            enable_flushing_threads=False, enable_h5_persistence=False)

    def test_small_id_uses_fastpath_huge_id_falls_back(self):
        mgr = self._mgr()
        big = mgr._shm_max_sid + 50          # comfortably over the cap
        df = pd.DataFrame(
            {"discarded": [True, True], "origin": ["train", "train"]},
            index=[5, big],
        )
        df.index.name = "sample_id"

        changed = mgr._propagate_to_shm(df, {"train"})
        self.assertTrue(changed, "the small-id write should register as changed")

        # Small id: fast-path hit.
        self.assertTrue(mgr.is_in_down_only_shm("train", "discarded", 5))
        # Huge id: skipped → read falls back to False (pandas deny-list covers it).
        self.assertFalse(mgr.is_in_down_only_shm("train", "discarded", big))

        # Crucially, the array was NOT sized to the huge id.
        arr = mgr._shm_down_only[("train", "discarded")]
        self.assertLess(len(arr), mgr._shm_max_sid,
                        "array must not be allocated to the huge sample_id")
        self.assertIn("train", mgr._shm_cap_warned, "over-cap id should warn once")

    def test_user_tags_list_column_not_mirrored(self):
        mgr = self._mgr()
        df = pd.DataFrame(
            {"discarded": [True], "user_tags": [["a", "b"]], "origin": ["train"]},
            index=[3],
        )
        df.index.name = "sample_id"
        mgr._propagate_to_shm(df, {"train"})
        # discarded (bool) is mirrored ...
        self.assertTrue(mgr.is_in_down_only_shm("train", "discarded", 3))
        # ... but user_tags (a list column) gets NO shm array — bool(list) would
        # be a meaningless bit nothing reads.
        self.assertNotIn(("train", "user_tags"), mgr._shm_down_only)

    def test_undiscard_clears_cell(self):
        mgr = self._mgr()
        df = pd.DataFrame({"discarded": [True], "origin": ["train"]}, index=[7])
        df.index.name = "sample_id"
        mgr._propagate_to_shm(df, {"train"})
        self.assertTrue(mgr.is_in_down_only_shm("train", "discarded", 7))
        # Flip back to False (restore a sample) — cell clears.
        df2 = pd.DataFrame({"discarded": [False], "origin": ["train"]}, index=[7])
        df2.index.name = "sample_id"
        changed = mgr._propagate_to_shm(df2, {"train"})
        self.assertTrue(changed)
        self.assertFalse(mgr.is_in_down_only_shm("train", "discarded", 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
