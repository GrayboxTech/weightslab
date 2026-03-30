import unittest
from unittest.mock import MagicMock
import threading

import pandas as pd

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.trainer.services.data_service import DataService
from weightslab.trainer.services.experiment_service import ExperimentService
from weightslab.trainer.trainer_services import ExperimentServiceServicer


class _MockContext:
    def add_callback(self, callback):
        return True


class _FakeCtx:
    def __init__(self, components=None):
        self.components = components or {}
        self.exp_name = None

    def ensure_components(self):
        return None


class _FakeDFManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_combined_df(self):
        return self.df.reset_index().copy()

    def get_df_view(self):
        return self.df.reset_index().copy()\

    def flush(self):
        return None

    def drop_column(self, column: str):
        if column in self.df.columns:
            return self.df.pop(column)
        return None

    def upsert_df(self, df_local: pd.DataFrame, origin: str = None, force_flush: bool = False):
        incoming = df_local.copy()
        if "sample_id" in incoming.columns:
            incoming = incoming.set_index("sample_id")

        if isinstance(incoming.index, pd.MultiIndex):
            rows = incoming.reset_index()
        else:
            rows = incoming.reset_index()
            if SampleStatsEx.ORIGIN.value not in rows.columns:
                rows[SampleStatsEx.ORIGIN.value] = origin or "unknown"

        for _, row in rows.iterrows():
            sid = str(row.get("sample_id", row.get(SampleStatsEx.SAMPLE_ID.value)))
            org = str(row.get(SampleStatsEx.ORIGIN.value, origin or "unknown"))
            key = (org, sid)

            if key not in self.df.index:
                self.df.loc[key, :] = pd.Series(dtype=object)

            for col in rows.columns:
                if col in {"sample_id", SampleStatsEx.ORIGIN.value}:
                    continue
                self.df.at[key, col] = row[col]

        self.df.index = pd.MultiIndex.from_tuples(
            [(str(i[0]), str(i[1])) for i in self.df.index.tolist()],
            names=[SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value],
        )


class TestGRPCWeightsStudioUserActions(unittest.TestCase):
    def _make_servicer(self):
        exp_service = MagicMock()
        exp_service.model_service = MagicMock()
        exp_service.data_service = MagicMock()
        return ExperimentServiceServicer(exp_service=exp_service), exp_service

    def test_tag_data_action_via_grpc(self):
        servicer, exp_service = self._make_servicer()
        exp_service.data_service.EditDataSample.return_value = pb2.DataEditsResponse(
            success=True,
            message="tag applied",
        )

        request = pb2.DataEditsRequest(
            stat_name="tags",
            string_value="focus",
            float_value=0,
            bool_value=False,
            type=pb2.EDIT_ACCUMULATE,
            samples_ids=["1", "2"],
            sample_origins=["test", "test"],
        )

        response = servicer.EditDataSample(request, _MockContext())

        self.assertTrue(response.success)
        self.assertIn("tag", response.message)
        exp_service.data_service.EditDataSample.assert_called_once()

    def test_discard_data_action_via_grpc(self):
        servicer, exp_service = self._make_servicer()
        exp_service.data_service.EditDataSample.return_value = pb2.DataEditsResponse(
            success=True,
            message="discard updated",
        )

        request = pb2.DataEditsRequest(
            stat_name="discarded",
            string_value="",
            float_value=0,
            bool_value=True,
            type=pb2.EDIT_OVERRIDE,
            samples_ids=["10", "11"],
            sample_origins=["test", "test"],
        )

        response = servicer.EditDataSample(request, _MockContext())

        self.assertTrue(response.success)
        self.assertIn("discard", response.message)
        exp_service.data_service.EditDataSample.assert_called_once()

    def test_sort_dataframe_action_via_grpc(self):
        servicer, exp_service = self._make_servicer()
        exp_service.data_service.ApplyDataQuery.return_value = pb2.DataQueryResponse(
            success=True,
            message="Applied view sort/slice",
            number_of_all_samples=100,
            number_of_samples_in_the_loop=95,
            number_of_discarded_samples=5,
        )

        request = pb2.DataQueryRequest(
            query="loss > 0.2 sortby index desc scope:view start:0 count:20",
            is_natural_language=False,
        )

        response = servicer.ApplyDataQuery(request, _MockContext())

        self.assertTrue(response.success)
        self.assertIn("sort", response.message.lower())
        self.assertEqual(response.number_of_all_samples, 100)

    def test_logger_plot_info_via_grpc(self):
        servicer, exp_service = self._make_servicer()
        exp_service.GetLatestLoggerData.return_value = pb2.GetLatestLoggerDataResponse(
            points=[
                pb2.LoggerDataPoint(
                    metric_name="train/loss",
                    model_age=3,
                    metric_value=0.42,
                    experiment_hash="abc",
                    timestamp=1,
                    sample_id="",
                )
            ]
        )

        request = pb2.GetLatestLoggerDataRequest(
            request_full_history=False,
            break_by_slices=False,
        )
        response = servicer.GetLatestLoggerData(request, _MockContext())

        self.assertEqual(len(response.points), 1)
        self.assertEqual(response.points[0].metric_name, "train/loss")
        self.assertEqual(response.points[0].model_age, 3)

    def test_break_by_slices_plot_from_tags_via_grpc(self):
        servicer, exp_service = self._make_servicer()
        exp_service.GetLatestLoggerData.return_value = pb2.GetLatestLoggerDataResponse(
            points=[
                pb2.LoggerDataPoint(
                    metric_name="test/loss",
                    model_age=8,
                    metric_value=0.11,
                    experiment_hash="exp-1",
                    timestamp=2,
                    sample_id="17",
                )
            ]
        )

        request = pb2.GetLatestLoggerDataRequest(
            request_full_history=False,
            break_by_slices=True,
            tags=["hard", "focus"],
            graph_name="test/loss",
        )
        response = servicer.GetLatestLoggerData(request, _MockContext())

        self.assertEqual(len(response.points), 1)
        self.assertEqual(response.points[0].sample_id, "17")
        self.assertEqual(response.points[0].metric_name, "test/loss")

    def test_agent_query_via_grpc(self):
        servicer, exp_service = self._make_servicer()
        exp_service.data_service.ApplyDataQuery.return_value = pb2.DataQueryResponse(
            success=True,
            message="Analysis Result: mean loss by class",
            agent_intent_type=pb2.INTENT_ANALYSIS,
            analysis_result="mean loss by class",
            number_of_all_samples=100,
            number_of_samples_in_the_loop=90,
            number_of_discarded_samples=10,
        )

        request = pb2.DataQueryRequest(
            query="What is the mean loss by class?",
            is_natural_language=True,
        )
        response = servicer.ApplyDataQuery(request, _MockContext())

        self.assertTrue(response.success)
        self.assertEqual(response.agent_intent_type, pb2.INTENT_ANALYSIS)
        self.assertIn("mean loss", response.analysis_result)


class TestGRPCWeightsStudioSDKState(unittest.TestCase):
    def _make_real_data_service(self):
        index = pd.MultiIndex.from_tuples(
            [("test", "1"), ("test", "2"), ("test", "3")],
            names=[SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value],
        )
        df = pd.DataFrame(
            {
                "loss": [0.2, 0.8, 0.5],
                SampleStatsEx.DISCARDED.value: [False, False, False],
            },
            index=index,
        )

        df_manager = _FakeDFManager(df)
        ctx = _FakeCtx(components={"df_manager": df_manager})

        ds = DataService.__new__(DataService)
        ds._ctx = ctx
        ds._lock = threading.RLock()
        ds._update_lock = threading.Lock()
        ds._df_manager = df_manager
        ds._all_datasets_df = df.copy()
        ds._compute_natural_sort = False
        ds._is_filtered = False
        ds._last_internals_update_time = 0
        ds._agent = MagicMock()
        ds._agent.is_ollama_available.return_value = True

        return ds, df_manager

    def _make_servicer_with_real_data_service(self, data_service):
        exp_service = MagicMock()
        exp_service.model_service = MagicMock()
        exp_service.data_service = data_service
        return ExperimentServiceServicer(exp_service=exp_service)

    def test_grpc_tag_action_updates_dataframe_and_response_counts(self):
        data_service, df_manager = self._make_real_data_service()
        servicer = self._make_servicer_with_real_data_service(data_service)

        request = pb2.DataEditsRequest(
            stat_name="tags",
            string_value="focus",
            float_value=0,
            bool_value=False,
            type=pb2.EDIT_ACCUMULATE,
            samples_ids=["1"],
            sample_origins=["test"],
        )
        edit_resp = servicer.EditDataSample(request, _MockContext())
        self.assertTrue(edit_resp.success)

        self.assertIn(f"{SampleStatsEx.TAG.value}:focus", data_service._all_datasets_df.columns)
        self.assertTrue(bool(data_service._all_datasets_df.loc[("test", "1"), f"{SampleStatsEx.TAG.value}:focus"]))
        self.assertTrue(bool(df_manager.df.loc[("test", "1"), f"{SampleStatsEx.TAG.value}:focus"]))

        query_resp = servicer.ApplyDataQuery(pb2.DataQueryRequest(query="", is_natural_language=False), _MockContext())
        self.assertTrue(query_resp.success)
        self.assertIn("focus", list(query_resp.unique_tags))

    def test_grpc_discard_action_updates_df_and_in_loop_counts(self):
        data_service, _ = self._make_real_data_service()
        servicer = self._make_servicer_with_real_data_service(data_service)

        edit_request = pb2.DataEditsRequest(
            stat_name=SampleStatsEx.DISCARDED.value,
            bool_value=True,
            float_value=0,
            string_value="",
            type=pb2.EDIT_OVERRIDE,
            samples_ids=["2"],
            sample_origins=["test"],
        )
        edit_resp = servicer.EditDataSample(edit_request, _MockContext())
        self.assertTrue(edit_resp.success)
        self.assertTrue(bool(data_service._all_datasets_df.loc[("test", "2"), SampleStatsEx.DISCARDED.value]))

        query_resp = servicer.ApplyDataQuery(pb2.DataQueryRequest(query="", is_natural_language=False), _MockContext())
        self.assertEqual(query_resp.number_of_all_samples, 3)
        self.assertEqual(query_resp.number_of_discarded_samples, 1)
        self.assertEqual(query_resp.number_of_samples_in_the_loop, 2)

    def test_grpc_sort_action_reorders_view_df(self):
        data_service, _ = self._make_real_data_service()
        servicer = self._make_servicer_with_real_data_service(data_service)

        request = pb2.DataQueryRequest(
            query="sortby loss desc",
            is_natural_language=False,
        )
        response = servicer.ApplyDataQuery(request, _MockContext())

        self.assertTrue(response.success)
        ordered_sample_ids = [idx[1] for idx in data_service._all_datasets_df.index.tolist()]
        self.assertEqual(ordered_sample_ids, ["2", "3", "1"])

    def test_grpc_agent_query_outputs_analysis_result(self):
        data_service, _ = self._make_real_data_service()
        data_service._agent.query.return_value = [
            {
                "function": "df.analyze",
                "params": {"code": "df['loss'].mean()"},
            }
        ]
        servicer = self._make_servicer_with_real_data_service(data_service)

        request = pb2.DataQueryRequest(
            query="compute average loss",
            is_natural_language=True,
        )
        response = servicer.ApplyDataQuery(request, _MockContext())

        self.assertTrue(response.success)
        self.assertEqual(response.agent_intent_type, pb2.INTENT_ANALYSIS)
        self.assertIn("0.5", response.analysis_result)

    def test_grpc_check_agent_health_uses_data_service(self):
        data_service, _ = self._make_real_data_service()
        servicer = self._make_servicer_with_real_data_service(data_service)

        response = servicer.CheckAgentHealth(pb2.Empty(), _MockContext())

        self.assertTrue(response.available)
        self.assertIn("available", response.message.lower())


class TestGRPCLoggerOutputIntegration(unittest.TestCase):
    def _make_exp_service_for_logger(self):
        signal_logger = MagicMock()
        df_manager = MagicMock()
        ctx = _FakeCtx(components={"signal_logger": signal_logger, "df_manager": df_manager})

        exp_service = ExperimentService.__new__(ExperimentService)
        exp_service._ctx = ctx
        exp_service.model_service = MagicMock()
        exp_service.data_service = MagicMock()
        return exp_service, signal_logger, df_manager

    def test_logger_plot_info_returns_expected_points(self):
        exp_service, signal_logger, _ = self._make_exp_service_for_logger()
        signal_logger.get_and_clear_queue.return_value = [
            {
                "metric_name": "train/loss",
                "model_age": 4,
                "metric_value": 0.33,
                "experiment_hash": "exp-h",
                "timestamp": 123,
            }
        ]
        servicer = ExperimentServiceServicer(exp_service=exp_service)

        request = pb2.GetLatestLoggerDataRequest(request_full_history=False, break_by_slices=False)
        response = servicer.GetLatestLoggerData(request, _MockContext())

        self.assertEqual(len(response.points), 1)
        self.assertEqual(response.points[0].metric_name, "train/loss")
        self.assertEqual(response.points[0].model_age, 4)

    def test_break_by_slices_from_tags_filters_expected_sample(self):
        exp_service, signal_logger, df_manager = self._make_exp_service_for_logger()

        df_manager.get_df_view.return_value = pd.DataFrame(
            {"tag:hard": [True, False]},
            index=[11, 12],
        )
        signal_logger.get_signal_history_per_sample.return_value = {
            "test/loss": {
                "exp-1": [
                    {"sample_id": "11", "model_age": 5, "metric_value": 0.2, "experiment_hash": "exp-1", "timestamp": 1},
                    {"sample_id": "12", "model_age": 5, "metric_value": 0.8, "experiment_hash": "exp-1", "timestamp": 1},
                ]
            }
        }

        servicer = ExperimentServiceServicer(exp_service=exp_service)
        request = pb2.GetLatestLoggerDataRequest(
            request_full_history=False,
            break_by_slices=True,
            tags=["hard"],
            graph_name="test/loss",
        )
        response = servicer.GetLatestLoggerData(request, _MockContext())

        self.assertEqual(len(response.points), 1)
        self.assertEqual(response.points[0].sample_id, "11")
        self.assertEqual(response.points[0].metric_name, "test/loss")


if __name__ == "__main__":
    unittest.main()
