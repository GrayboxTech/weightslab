import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch as th

import weightslab.src as src

from unittest.mock import MagicMock, patch

from weightslab.data.sample_stats import SampleStatsEx


class TestResolveConfiguredRootLogDir(unittest.TestCase):
    """root_log_dir resolution: explicit config > WEIGHTSLAB_ROOT_LOG_DIR > temp dir."""

    def setUp(self):
        self._env_prev = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR")

    def tearDown(self):
        if self._env_prev is None:
            os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        else:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._env_prev

    def test_explicit_config_value_wins_over_env(self):
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = "/env/dir"
        self.assertEqual(src._resolve_configured_root_log_dir("/explicit/dir"), "/explicit/dir")

    def test_env_used_when_config_absent_and_dir_exists(self):
        with tempfile.TemporaryDirectory() as real_dir:
            os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = real_dir
            self.assertEqual(src._resolve_configured_root_log_dir(None), real_dir)
            self.assertEqual(src._resolve_configured_root_log_dir(""), real_dir)

    def test_env_set_to_missing_dir_warns_and_falls_back_to_tempdir(self):
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = "/definitely/does/not/exist/abc123"
        with patch("weightslab.src.tempfile.mkdtemp", return_value="/tmp/generated") as mk, \
                self.assertLogs("weightslab.src", level="WARNING") as log_ctx:
            result = src._resolve_configured_root_log_dir(None)
        self.assertEqual(result, "/tmp/generated")
        mk.assert_called_once()
        self.assertTrue(any("does not exist" in msg for msg in log_ctx.output))

    def test_falls_back_to_tempdir_when_neither_set(self):
        os.environ.pop("WEIGHTSLAB_ROOT_LOG_DIR", None)
        with patch("weightslab.src.tempfile.mkdtemp", return_value="/tmp/generated") as mk:
            self.assertEqual(src._resolve_configured_root_log_dir(None), "/tmp/generated")
            mk.assert_called_once()


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

        # tag_samples must hand upsert a (sample_id, annotation_id) multi-index,
        # with the sample-level tag on the canonical row (annotation_id == 0).
        self.assertIsInstance(df_update.index, pd.MultiIndex)
        self.assertEqual(list(df_update.index.names), ["sample_id", "annotation_id"])
        self.assertEqual(list(df_update.index), [(1, 0), (2, 0)])
        self.assertIn("tag:difficult", df_update.columns)
        self.assertTrue(bool(df_update.loc[(1, 0), "tag:difficult"]))
        self.assertTrue(bool(df_update.loc[(2, 0), "tag:difficult"]))
        self.assertTrue(kwargs.get("force_flush"))

    def test_tag_samples_remove_mode(self):
        df_manager = MagicMock()

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            ok = src.tag_samples([3, 4], "outlier", mode="remove")

        self.assertTrue(ok)
        df_update = df_manager.upsert_df.call_args.args[0]
        self.assertIsInstance(df_update.index, pd.MultiIndex)
        self.assertFalse(bool(df_update.loc[(3, 0), "tag:outlier"]))
        self.assertFalse(bool(df_update.loc[(4, 0), "tag:outlier"]))

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

        self.assertIsInstance(df_update.index, pd.MultiIndex)
        self.assertIn(SampleStatsEx.DISCARDED.value, df_update.columns)
        self.assertTrue(bool(df_update.loc[(10, 0), SampleStatsEx.DISCARDED.value]))
        self.assertTrue(bool(df_update.loc[(11, 0), SampleStatsEx.DISCARDED.value]))

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
            out = src.get_samples_by_tag("difficult", origin="train_loader", limit=100)

        self.assertEqual(out, [5, 7])
        # `origin` filters the rows of the 'origin' column (the loader name); it is
        # not the column to select.
        df_manager.get_df_view.assert_called_once_with(
            column=SampleStatsEx.ORIGIN.value, value="train_loader", limit=100)

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
            out = src.get_discarded_samples(origin="val_loader", limit=10)

        self.assertEqual(out, [100, 102])
        df_manager.get_df_view.assert_called_once_with(
            column=SampleStatsEx.ORIGIN.value, value="val_loader", limit=10)

    def test_get_discarded_samples_without_origin_reads_every_split(self):
        df_manager = MagicMock()
        df_manager.get_df_view.return_value = pd.DataFrame(
            {SampleStatsEx.DISCARDED.value: [True, False]}, index=[1, 2])

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            out = src.get_discarded_samples()

        self.assertEqual(out, [1])
        # No origin and no limit: whole frame, and -1 (not None) so the manager's
        # `limit > 0` check cannot raise.
        df_manager.get_df_view.assert_called_once_with(limit=-1)

    def test_get_samples_by_tag_returns_sample_ids_from_a_multiindex(self):
        df_manager = MagicMock()
        index = pd.MultiIndex.from_tuples(
            [(5, 0), (5, 1), (6, 0)], names=["sample_id", "annotation_id"])
        df_manager.get_df_view.return_value = pd.DataFrame(
            {"tag:difficult": [True, True, False]}, index=index)

        with patch("weightslab.backend.ledgers.get_dataframe", return_value=df_manager):
            out = src.get_samples_by_tag("difficult")

        # Sample ids, de-duplicated — not (sample_id, annotation_id) tuples.
        self.assertEqual(out, [5])


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
        # Scalar-per-sample preds/targets must stay 1-D (B,) here -- an extra
        # trailing axis makes per-sample indexing downstream (enqueue_batch's
        # index_batch) yield a length-1 array like [6] instead of a true
        # scalar 6, which then round-trips through the dataframe as `[6]`.
        self.assertEqual(kwargs["preds"].shape, (2,))
        self.assertEqual(kwargs["targets"].shape, (2,))
        self.assertEqual(kwargs["preds_raw"].shape, (2, 2))
        self.assertIn("signals//loss", kwargs["losses"])
        self.assertEqual(kwargs["step"], 7)


class TestSrcGpuRelease(unittest.TestCase):
    def test_release_gpu_resources_does_not_touch_cuda_when_not_initialized(self):
        with patch("weightslab.src.list_models", return_value=[]), \
             patch("weightslab.src.get_optimizer", return_value=None), \
             patch("weightslab.src.th.cuda.is_initialized", return_value=False), \
             patch("weightslab.src.th.cuda.empty_cache") as mock_empty_cache, \
             patch("weightslab.src.th.cuda.ipc_collect") as mock_ipc_collect:
            src._release_gpu_resources()

        mock_empty_cache.assert_not_called()
        mock_ipc_collect.assert_not_called()

    def test_release_gpu_resources_cleans_cuda_when_initialized(self):
        with patch("weightslab.src.list_models", return_value=[]), \
             patch("weightslab.src.get_optimizer", return_value=None), \
             patch("weightslab.src.th.cuda.is_initialized", return_value=True), \
             patch("weightslab.src.th.cuda.empty_cache") as mock_empty_cache, \
             patch("weightslab.src.th.cuda.ipc_collect") as mock_ipc_collect:
            src._release_gpu_resources()

        mock_empty_cache.assert_called_once()
        mock_ipc_collect.assert_called_once()


class TestSrcStartTraining(unittest.TestCase):
    def test_start_training_forces_resume_for_model_only_workflows(self):
        with patch("weightslab.src.ledgers.list_dataloaders", return_value=[]), \
             patch("weightslab.src._warn_on_signal_cycles"), \
             patch("weightslab.src.pause_ctrl.resume") as mock_resume:
            src.start_training()

        mock_resume.assert_called_once_with(force=True)

    def test_start_training_retains_hash_guard_when_data_is_registered(self):
        with patch("weightslab.src.ledgers.list_dataloaders", return_value=["train"]), \
             patch("weightslab.src._warn_on_signal_cycles"), \
             patch("weightslab.src.pause_ctrl.resume") as mock_resume:
            src.start_training()

        mock_resume.assert_called_once_with(force=False)


if __name__ == "__main__":
    unittest.main()
