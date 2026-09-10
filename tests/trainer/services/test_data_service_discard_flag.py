"""The 'samples discard themselves in the studio while training' regression.

Reported for detection/segmentation runs: as the model worked through the
dataset, sample after sample greyed out in the UI, while the dataframe (checked
from the notebook) said nothing was discarded.

The chain, reproduced below:

1. the trainer touches a sample -> its rows go dirty;
2. `_fastUpdateInternals` syncs the columns the trainer mutates
   (``signals*``, ``last_seen``, ``discarded``, ``prediction``, ``target``)
   from the source into the view. It collapsed the source's per-annotation rows
   with ``duplicated(keep="last")``, i.e. it kept the LAST annotation row --
   whose *sample-level* columns are NaN, because the real values live on the
   canonical row (annotation_id == 0);
3. NaN therefore landed in the view's ``discarded``;
4. GetDataSamples reported that flag as ``"1" if bool(value) else "0"`` -- and
   ``bool(float("nan"))`` is True in Python.

So every sample the model had seen was served as discarded. Only for
annotation-expanded (detection/segmentation) ledgers, and only after training
touched the sample: exactly the reported shape.
"""

import unittest

import numpy as np
import pandas as pd

from weightslab.data.sample_stats import SampleStatsEx
from weightslab.trainer.services.data_service import (
    DataService,
    is_set_flag,
    set_flag_mask,
)


SID = SampleStatsEx.SAMPLE_ID.value
ANNOT = SampleStatsEx.INSTANCE_ID.value
DISCARDED = SampleStatsEx.DISCARDED.value


def _source_rows():
    """A segmentation ledger: sample-level values on annotation 0 only."""
    return pd.DataFrame(
        {
            DISCARDED: [False, np.nan, np.nan, True, np.nan],
            "last_seen": [11, 11, 11, 12, 12],
            "prediction": ["a", None, None, "b", None],
        },
        index=pd.MultiIndex.from_tuples(
            [("0", 0), ("0", 1), ("0", 2), ("1", 0), ("1", 1)],
            names=[SID, ANNOT],
        ),
    )


class _FakeManager:
    """Minimal dataframe manager: dirty tracking + source row lookup."""

    def __init__(self, source, dirty):
        self._df = source
        self._dirty = list(dirty)

    def take_view_dirty(self, limit=None):
        dirty, self._dirty = self._dirty, []
        return dirty

    def get_source_rows(self, sample_ids, columns=None):
        wanted = [str(s) for s in sample_ids]
        level = self._df.index.get_level_values(SID).astype(str)
        rows = self._df[level.isin(wanted)]
        return rows[columns] if columns else rows


class TestFastViewSyncKeepsSampleLevelValues(unittest.TestCase):
    def _service(self, source, dirty, view):
        service = DataService.__new__(DataService)
        service._all_datasets_df = view
        service._df_manager = _FakeManager(source, dirty)
        return service

    def _collapsed_view(self, source):
        """One row per sample, as the real view is built: annotation 0 wins."""
        base = source[source.index.get_level_values(ANNOT) == 0].droplevel(ANNOT)
        return base.copy()

    def test_a_seen_sample_keeps_its_discarded_flag(self):
        source = _source_rows()
        view = self._collapsed_view(source)
        service = self._service(source, ["0", "1"], view)

        self.assertTrue(service._fastUpdateInternals())

        # Sample 0 is NOT discarded and must stay that way; sample 1 is.
        self.assertIs(bool(view.loc["0", DISCARDED]), False)
        self.assertIs(bool(view.loc["1", DISCARDED]), True)
        # The give-away of the old behaviour: a NaN in a column that had a value.
        self.assertFalse(view[DISCARDED].isna().any(),
                         "sample-level flags were overwritten with an instance row's NaN")

    def test_the_columns_the_trainer_owns_still_sync(self):
        source = _source_rows()
        view = self._collapsed_view(source)
        source.loc[("0", 0), "last_seen"] = 99
        service = self._service(source, ["0"], view)

        self.assertTrue(service._fastUpdateInternals())
        self.assertEqual(view.loc["0", "last_seen"], 99)

    def test_falls_back_to_the_first_row_when_no_canonical_row_is_present(self):
        # A slice of instance rows only (no annotation 0) must still sync
        # something sane rather than raising or inventing a flag.
        source = _source_rows().drop(index=("0", 0))
        view = self._collapsed_view(_source_rows())
        service = self._service(source, ["0"], view)

        self.assertTrue(service._fastUpdateInternals())
        self.assertTrue(pd.isna(view.loc["0", DISCARDED]) or view.loc["0", DISCARDED] is False)


class TestDiscardedFlagIsNaNSafe(unittest.TestCase):
    """The second half: how a nullable flag becomes what the studio renders.

    is_set_flag / set_flag_mask are shared by the three places that read one:
    GetDataSamples' `discarded` rendering flag, the boolean ``tag:*`` columns in
    the metadata response, and the histogram's per-(origin, discarded) split.
    """

    @staticmethod
    def _served_flag(value):
        return "1" if is_set_flag(value) else "0"

    def test_missing_is_not_discarded(self):
        # bool(float("nan")) is True -- the whole bug in one line.
        self.assertEqual(self._served_flag(np.nan), "0")
        self.assertEqual(self._served_flag(None), "0")
        self.assertEqual(self._served_flag(pd.NA), "0")

    def test_real_values_still_come_through(self):
        self.assertEqual(self._served_flag(True), "1")
        self.assertEqual(self._served_flag(np.bool_(True)), "1")
        self.assertEqual(self._served_flag(1), "1")
        self.assertEqual(self._served_flag(False), "0")
        self.assertEqual(self._served_flag(0), "0")


    def test_a_string_flag_is_read_as_a_word_not_as_a_non_empty_string(self):
        # bool("False") is True, and a boolean column that has been through the
        # H5 store (categorical) can come back as these strings.
        self.assertEqual(self._served_flag("False"), "0")
        self.assertEqual(self._served_flag("false"), "0")
        self.assertEqual(self._served_flag("0"), "0")
        self.assertEqual(self._served_flag(""), "0")
        self.assertEqual(self._served_flag("True"), "1")
        self.assertEqual(self._served_flag("true"), "1")
        self.assertEqual(self._served_flag("1"), "1")


class TestSetFlagMask(unittest.TestCase):
    """The column-wide form, used for tags and the histogram split."""

    def test_a_sparse_tag_column_marks_only_the_tagged_samples(self):
        # A tag is set on a few samples; every other row is NaN. astype(bool)
        # turned those into True -- every sample wore every tag.
        column = pd.Series([True, np.nan, False, None, True], dtype=object)
        self.assertEqual(set_flag_mask(column).tolist(),
                         [True, False, False, False, True])

    def test_it_handles_a_categorical_column(self):
        # The H5 store optimises tag:* and discarded to categorical dtype.
        column = pd.Series([True, None, False], dtype=object).astype("category")
        self.assertEqual(set_flag_mask(column).tolist(), [True, False, False])

    def test_it_handles_a_float_column_of_zeros_and_nans(self):
        column = pd.Series([1.0, np.nan, 0.0])
        self.assertEqual(set_flag_mask(column).tolist(), [True, False, False])

    def test_an_absent_column_is_all_false(self):
        self.assertEqual(set_flag_mask(None).tolist(), [])


class TestFastViewSyncAddressing(unittest.TestCase):
    """How dirty source rows are matched to view rows.

    The view is indexed (origin, sample_id) precisely because one sample_id can
    appear under two origins. Looking positions up in that non-unique level
    raised InvalidIndexError, so the differential refresh failed on every call
    and quietly fell back to the full rebuild.
    """

    def _service(self, source, dirty, view):
        service = DataService.__new__(DataService)
        service._all_datasets_df = view
        service._df_manager = _FakeManager(source, dirty)
        return service

    def _source(self):
        return pd.DataFrame(
            {DISCARDED: [False, np.nan], "last_seen": [7, 7]},
            index=pd.MultiIndex.from_tuples([("5", 0), ("5", 1)], names=[SID, ANNOT]),
        )

    def test_one_sample_id_under_two_origins_does_not_raise(self):
        view = pd.DataFrame(
            {DISCARDED: [False, False], "last_seen": [1, 2]},
            index=pd.MultiIndex.from_tuples([("train_loader", "5"), ("test_loader", "5")],
                                            names=["origin", SID]),
        )
        service = self._service(self._source(), ["5"], view)

        self.assertTrue(service._fastUpdateInternals())
        # The source can only speak per sample_id, so both rows take its values.
        self.assertEqual(view["last_seen"].tolist(), [7, 7])
        self.assertFalse(view[DISCARDED].isna().any())

    def test_a_dirty_sample_the_view_does_not_hold_forces_a_rebuild(self):
        view = pd.DataFrame(
            {DISCARDED: [False], "last_seen": [1]},
            index=pd.MultiIndex.from_tuples([("train_loader", "5")], names=["origin", SID]),
        )
        source = pd.concat([self._source(), pd.DataFrame(
            {DISCARDED: [False], "last_seen": [3]},
            index=pd.MultiIndex.from_tuples([("99", 0)], names=[SID, ANNOT]))])
        service = self._service(source, ["5", "99"], view)

        # 99 is new -> structural change -> only the full rebuild can add it.
        self.assertFalse(service._fastUpdateInternals())

    def test_nothing_to_do_when_no_dirty_sample_is_in_the_view(self):
        view = pd.DataFrame(
            {DISCARDED: [False], "last_seen": [1]},
            index=pd.MultiIndex.from_tuples([("train_loader", "7")], names=["origin", SID]),
        )
        service = self._service(self._source(), ["5"], view)
        self.assertTrue(service._fastUpdateInternals())
        self.assertEqual(view["last_seen"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
