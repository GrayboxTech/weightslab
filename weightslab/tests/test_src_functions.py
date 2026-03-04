import unittest
import numpy as np
import pandas as pd
import torch as th

import weightslab.src as src

from unittest.mock import MagicMock, patch

from weightslab.data.sample_stats import SampleStatsEx


class TestSrcTagAndDiscardFunctions(unittest.TestCase):
    def setUp(self):
        src.DATAFRAME_M = None

    def test_tag_samples_add_mode(self):
        df_manager = MagicMock()

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            ok = src.tag_samples([1, 2], "difficult", mode="add")

        self.assertTrue(ok)
        df_update = df_manager.upsert_df.call_args.args[0]
        kwargs = df_manager.upsert_df.call_args.kwargs

        self.assertEqual(list(df_update.index), [1, 2])
        self.assertIn("tag:difficult", df_update.columns)
        self.assertTrue(bool(df_update.loc[1, "tag:difficult"]))
        self.assertTrue(bool(df_update.loc[2, "tag:difficult"]))
        self.assertTrue(kwargs.get("force_flush"))

    def test_tag_samples_remove_mode(self):
        df_manager = MagicMock()

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            ok = src.tag_samples([3, 4], "outlier", mode="remove")

        self.assertTrue(ok)
        df_update = df_manager.upsert_df.call_args.args[0]
        self.assertFalse(bool(df_update.loc[3, "tag:outlier"]))
        self.assertFalse(bool(df_update.loc[4, "tag:outlier"]))

    def test_tag_samples_invalid_mode_returns_false(self):
        df_manager = MagicMock()

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            ok = src.tag_samples([1], "foo", mode="invalid")

        self.assertFalse(ok)
        df_manager.upsert_df.assert_not_called()

    def test_tag_samples_without_dataframe_manager_returns_false(self):
        with patch("weightslab.backend.ledgers.get_dataframe", return_value=None):
            ok = src.tag_samples([1], "foo", mode="add")

        self.assertFalse(ok)

    def test_discard_samples_updates_discard_column(self):
        df_manager = MagicMock()

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            ok = src.discard_samples([10, 11], discarded=True)

        self.assertTrue(ok)
        df_update = df_manager.upsert_df.call_args.args[0]

        self.assertIn(SampleStatsEx.DISCARDED.value, df_update.columns)
        self.assertTrue(bool(df_update.loc[10, SampleStatsEx.DISCARDED.value]))
        self.assertTrue(bool(df_update.loc[11, SampleStatsEx.DISCARDED.value]))

    def test_get_samples_by_tag_filters_true_values(self):
        df_manager = MagicMock()
        df_manager.get_df_view.return_value = pd.DataFrame(
            {
                "tag:difficult": [True, False, True],
                "other_col": [1, 2, 3],
            },
            index=[5, 6, 7],
        )

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            out = src.get_samples_by_tag("difficult", origin="train", limit=100)

        self.assertEqual(out, [5, 7])
        df_manager.get_df_view.assert_called_once_with("train", limit=100)

    def test_get_samples_by_tag_missing_tag_column_returns_empty(self):
        df_manager = MagicMock()
        df_manager.get_df_view.return_value = pd.DataFrame({"x": [1, 2]}, index=[1, 2])

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            out = src.get_samples_by_tag("missing", origin="train")

        self.assertEqual(out, [])

    def test_get_discarded_samples_filters_true_values(self):
        df_manager = MagicMock()
        df_manager.get_df_view.return_value = pd.DataFrame(
            {
                SampleStatsEx.DISCARDED.value: [True, False, True],
            },
            index=[100, 101, 102],
        )

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            out = src.get_discarded_samples(origin="eval", limit=10)

        self.assertEqual(out, [100, 102])
        df_manager.get_df_view.assert_called_once_with("eval", limit=10)


class TestSrcSaveSignals(unittest.TestCase):
    def setUp(self):
        src.DATAFRAME_M = None

    def test_save_signals_enqueues_expected_payload(self):
        df_manager = MagicMock()

        batch_ids = np.array(['10', '11'])
        signals = {"loss": th.tensor([1.0, 3.0], dtype=th.float32)}
        preds_raw = th.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=th.float32)
        targets = th.tensor([1, 0], dtype=th.int64)
        preds = th.tensor([1, 0], dtype=th.int64)

        with patch("weightslab.src.get_dataframe", return_value=df_manager), \
             patch("weightslab.src._get_step", return_value=7), \
             patch("weightslab.src._log_signal") as mock_log_signal:
            src.save_signals(
                batch_ids=batch_ids,
                signals=signals,
                preds_raw=preds_raw,
                targets=targets,
                preds=preds,
                log=True,
            )

        mock_log_signal.assert_called_once()
        df_manager.enqueue_batch.assert_called_once()

        kwargs = df_manager.enqueue_batch.call_args.kwargs

        np.testing.assert_array_equal(kwargs["sample_ids"], np.array(['10', '11']))
        self.assertEqual(kwargs["preds"].shape, (2, 1))
        self.assertEqual(kwargs["targets"].shape, (2, 1))
        self.assertEqual(kwargs["preds_raw"].shape, (2, 2))
        self.assertIn("signals//loss", kwargs["losses"])
        self.assertEqual(kwargs["step"], 7)


if __name__ == "__main__":
    unittest.main()
