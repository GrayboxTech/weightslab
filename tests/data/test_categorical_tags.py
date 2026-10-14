import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager
from weightslab.data.h5_dataframe_store import H5DataFrameStore


class TestCategoricalTagRegistryManager(unittest.TestCase):
    """Registry behaviour on LedgeredDataFrameManager (no H5 persistence)."""

    def _mgr(self):
        return LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)

    def test_register_merge_and_replace(self):
        mgr = self._mgr()
        self.assertEqual(mgr.register_categorical_tag("weather", ["rainy", "sunny"]), ["rainy", "sunny"])
        # Merge keeps order, dedups
        self.assertEqual(mgr.register_categorical_tag("weather", ["sunny", "cloudy"]), ["rainy", "sunny", "cloudy"])
        # Replace wipes previous
        self.assertEqual(mgr.register_categorical_tag("weather", ["fog"], replace=True), ["fog"])
        self.assertTrue(mgr.is_categorical_tag("weather"))
        self.assertTrue(mgr.is_categorical_tag("tag:weather")) # prefix tolerated
        self.assertFalse(mgr.is_categorical_tag("does_not_exist"))

    def test_register_strips_prefix_and_cleans(self):
        mgr = self._mgr()
        # "tag:" prefix on the name is stripped; empty/None/nan categories dropped
        out = mgr.register_categorical_tag("tag:quality", ["high", "", None, "low", "nan"])
        self.assertEqual(out, ["high", "low"])
        self.assertIn("quality", mgr.get_categorical_tags())

    def test_auto_detect_from_string_tag_column(self):
        mgr = self._mgr()
        df = pd.DataFrame(
            {"sample_id": [1, 2, 3], "origin": "train", "tag:weather": ["rainy", "sunny", "rainy"]}
        ).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        reg = mgr.get_categorical_tags()
        self.assertIn("weather", reg)
        self.assertEqual(set(reg["weather"]), {"rainy", "sunny"})

    def test_boolean_tag_not_registered_as_categorical(self):
        mgr = self._mgr()
        df = pd.DataFrame(
            {"sample_id": [1, 2], "origin": "train", "tag:is_urban": [True, False]}
        ).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        self.assertNotIn("is_urban", mgr.get_categorical_tags())

    def test_optimize_applies_full_category_set(self):
        mgr = self._mgr()
        mgr.register_categorical_tag("weather", ["rainy", "sunny", "cloudy", "snow"])
        df = pd.DataFrame(
            {"sample_id": [1, 2], "origin": "train", "tag:weather": ["rainy", "sunny"]}
        ).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        view = mgr.get_df_view()
        col = view["tag:weather"]
        self.assertTrue(isinstance(col.dtype, pd.CategoricalDtype))
        # Full registered set present even though only 2 values appear in data
        self.assertEqual(set(col.dtype.categories), {"rainy", "sunny", "cloudy", "snow"})


class TestCategoricalTagH5RoundTrip(unittest.TestCase):
    def _store(self):
        d = tempfile.mkdtemp()
        return H5DataFrameStore(Path(d) / "data.h5")

    def test_registry_save_load(self):
        store = self._store()
        reg = {"weather": ["rainy", "sunny", "cloudy"], "quality": ["high", "low"]}
        store.save_tag_registry(reg)
        loaded = store.load_tag_registry()
        self.assertEqual(loaded, reg)

    def test_round_trip_preserves_unused_categories(self):
        store = self._store()
        store.save_tag_registry({"weather": ["rainy", "sunny", "cloudy", "snow"]})

        idx = pd.MultiIndex.from_arrays(
            [["a", "b", "c"], [0, 0, 0]], names=["sample_id", "annotation_id"]
        )
        df = pd.DataFrame(
            {
                "origin": ["train"] * 3,
                "tag:weather": ["rainy", "sunny", "rainy"], # only 2 of 4 used
                "tag:is_urban": [True, False, True], # boolean tag
            },
            index=idx,
        )
        store.upsert("train", df)
        back = store.load("train")

        weather = back["tag:weather"]
        self.assertTrue(isinstance(weather.dtype, pd.CategoricalDtype))
        self.assertEqual(set(weather.dtype.categories), {"rainy", "sunny", "cloudy", "snow"})
        values = back.set_index("sample_id")["tag:weather"].astype(str).to_dict()
        self.assertEqual(values, {"a": "rainy", "b": "sunny", "c": "rainy"})
        # Boolean tag survives independently
        self.assertIn("tag:is_urban", back.columns)

    def test_clear_value_becomes_unset(self):
        store = self._store()
        store.save_tag_registry({"weather": ["rainy", "sunny"]})
        idx = pd.MultiIndex.from_arrays([["a", "b"], [0, 0]], names=["sample_id", "annotation_id"])
        df = pd.DataFrame(
            {"origin": ["train"] * 2, "tag:weather": ["rainy", None]}, index=idx
        )
        store.upsert("train", df)
        back = store.load("train").set_index(["sample_id", "annotation_id"])
        # 'b' had no value → unset (NaN), 'a' keeps its category
        self.assertEqual(str(back.loc[("a", 0), "tag:weather"]), "rainy")
        self.assertTrue(pd.isna(back.loc[("b", 0), "tag:weather"]))


class TestBooleanTagsStillWork(unittest.TestCase):
    """Regression guards: the legacy boolean-tag path must be unaffected."""

    def _store(self):
        return H5DataFrameStore(Path(tempfile.mkdtemp()) / "data.h5")

    def test_boolean_tag_not_registered_categorical(self):
        mgr = LedgeredDataFrameManager(enable_flushing_threads=False, enable_h5_persistence=False)
        df = pd.DataFrame(
            {"sample_id": [1, 2, 3], "origin": "train", "tag:is_urban": [True, False, True]}
        ).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        self.assertEqual(mgr.get_categorical_tags(), {})

    def test_boolean_tag_and_discarded_survive_reupsert(self):
        # Regression: re-upserting must not stringify True/False into "True"/"False".
        store = self._store()
        idx = pd.MultiIndex.from_arrays([["a", "b"], [0, 0]], names=["sample_id", "annotation_id"])
        df = pd.DataFrame(
            {"origin": ["train"] * 2, "tag:flagged": [True, False], "discarded": [False, True]},
            index=idx,
        )
        store.upsert("train", df)
        store.upsert("train", df) # merge path
        back = store.load("train").set_index(["sample_id", "annotation_id"])
        self.assertTrue(bool(back.loc[("a", 0), "tag:flagged"]))
        self.assertFalse(bool(back.loc[("b", 0), "tag:flagged"]))
        self.assertFalse(bool(back.loc[("a", 0), "discarded"]))
        self.assertTrue(bool(back.loc[("b", 0), "discarded"]))

    def test_boolean_and_categorical_coexist(self):
        store = self._store()
        store.save_tag_registry({"weather": ["rainy", "sunny"]})
        idx = pd.MultiIndex.from_arrays([["a", "b"], [0, 0]], names=["sample_id", "annotation_id"])
        df = pd.DataFrame(
            {"origin": ["train"] * 2, "tag:weather": ["rainy", "sunny"], "tag:flagged": [True, False]},
            index=idx,
        )
        store.upsert("train", df)
        store.upsert("train", df)
        back = store.load("train").set_index(["sample_id", "annotation_id"])
        self.assertTrue(isinstance(back["tag:weather"].dtype, pd.CategoricalDtype))
        self.assertEqual(str(back.loc[("a", 0), "tag:weather"]), "rainy")
        self.assertTrue(bool(back.loc[("a", 0), "tag:flagged"]))
        self.assertFalse(bool(back.loc[("b", 0), "tag:flagged"]))


if __name__ == "__main__":
    unittest.main()
