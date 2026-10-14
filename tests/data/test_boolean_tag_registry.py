"""Boolean tags declared before any sample wears them.

The UI's painter lets a tag be created and then painted. Between those two
moments the tag exists nowhere in the data -- and the tag list the UI shows is
read back out of the dataframe's ``tag:<name>`` columns, so without a registry
the freshly created tag disappears on the next refresh. These tests cover the
declaration itself; ``DataService._get_unique_tags`` reporting it is covered in
tests/trainer/services/test_trainer_services_unit.py.
"""

import unittest

import pandas as pd

from weightslab.data.dataframe_manager import LedgeredDataFrameManager


class TestBooleanTagRegistry(unittest.TestCase):
    def _mgr(self):
        return LedgeredDataFrameManager(
            enable_flushing_threads=False, enable_h5_persistence=False
        )

    def _with_samples(self):
        mgr = self._mgr()
        df = pd.DataFrame(
            {"sample_id": [1, 2], "origin": "train", "tag:painted": [True, False]}
        ).set_index("sample_id")
        mgr.upsert_df(df, origin="train")
        return mgr

    def test_declares_tag_and_creates_its_column_unset(self):
        mgr = self._with_samples()
        self.assertTrue(mgr.register_boolean_tag("fresh"))
        self.assertIn("fresh", mgr.get_declared_boolean_tags())

        combined = mgr.get_combined_df()
        self.assertIn("tag:fresh", combined.columns)
        # Declared, not applied: no sample wears it.
        self.assertFalse(combined["tag:fresh"].any())

    def test_tolerates_the_tag_prefix_and_rejects_empty_names(self):
        mgr = self._with_samples()
        self.assertTrue(mgr.register_boolean_tag("tag:prefixed"))
        self.assertIn("prefixed", mgr.get_declared_boolean_tags())
        self.assertNotIn("tag:prefixed", mgr.get_declared_boolean_tags())

        self.assertFalse(mgr.register_boolean_tag("   "))
        self.assertFalse(mgr.register_boolean_tag("None"))

    def test_registering_twice_is_idempotent_and_keeps_painted_values(self):
        mgr = self._with_samples()
        self.assertTrue(mgr.register_boolean_tag("painted"))
        self.assertTrue(mgr.register_boolean_tag("painted"))
        self.assertEqual(
            [t for t in mgr.get_declared_boolean_tags() if t == "painted"], ["painted"]
        )
        # The existing column is left alone -- re-declaring a tag in use must not
        # wipe the samples that already carry it.
        self.assertTrue(mgr.get_combined_df()["tag:painted"].any())

    def test_never_shadows_a_categorical_tag(self):
        mgr = self._with_samples()
        mgr.register_categorical_tag("weather", ["rainy", "sunny"])
        self.assertFalse(mgr.register_boolean_tag("weather"))
        self.assertNotIn("weather", mgr.get_declared_boolean_tags())

    def test_declared_without_any_samples_yet(self):
        # No dataset registered: there is no row to hang a column on, but the tag
        # is still remembered so it shows up once data arrives.
        mgr = self._mgr()
        self.assertTrue(mgr.register_boolean_tag("early"))
        self.assertEqual(mgr.get_declared_boolean_tags(), ["early"])

    def test_unregister_forgets_the_tag(self):
        mgr = self._with_samples()
        mgr.register_boolean_tag("doomed")
        mgr.unregister_boolean_tag("tag:doomed")
        self.assertNotIn("doomed", mgr.get_declared_boolean_tags())


if __name__ == "__main__":
    unittest.main()
