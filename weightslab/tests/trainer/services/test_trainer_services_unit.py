import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from PIL import Image

import weightslab.proto.experiment_service_pb2 as pb2
from weightslab.trainer.services.experiment_service import ExperimentService
from weightslab.trainer.services.model_service import ModelService
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.trainer.services.data_service import (
    DataService,
    SampleEditType,
    create_data_stat,
    generate_thumbnail,
)


class _DummyCtx:
    def __init__(self, components=None, exp_name=None):
        self.components = components or {}
        self.exp_name = exp_name
        self.hyper_parameters = {}

    def ensure_components(self):
        return None


class TestExperimentServiceUnit(unittest.TestCase):
    def test_get_latest_logger_data_full_history_nested(self):
        signal_logger = MagicMock()
        signal_logger.get_signal_history.return_value = {
            "train/loss": {
                "exp-hash": {
                    0: [{"metric_name": "train/loss", "model_age": 0, "metric_value": 1.0, "experiment_hash": "exp-hash"}],
                    1: [{"metric_name": "train/loss", "model_age": 1, "metric_value": 0.8, "experiment_hash": "exp-hash"}],
                }
            }
        }
        ctx = _DummyCtx(components={"signal_logger": signal_logger})
        with patch("weightslab.trainer.services.experiment_service.DataService"):
            service = ExperimentService(ctx)

        request = pb2.GetLatestLoggerDataRequest(request_full_history=True, max_points=100, break_by_slices=False)
        response = service.GetLatestLoggerData(request, None)

        self.assertEqual(len(response.points), 2)
        self.assertEqual(response.points[0].metric_name, "train/loss")
        self.assertEqual(response.points[0].model_age, 0)
        self.assertEqual(response.points[1].model_age, 1)

    def test_get_latest_logger_data_queue_mode(self):
        signal_logger = MagicMock()
        signal_logger.get_and_clear_queue.return_value = [
            {"metric_name": "train/acc", "model_age": 2, "metric_value": 0.95, "experiment_hash": "exp"}
        ]
        ctx = _DummyCtx(components={"signal_logger": signal_logger})
        with patch("weightslab.trainer.services.experiment_service.DataService"):
            service = ExperimentService(ctx)

        request = pb2.GetLatestLoggerDataRequest(request_full_history=False, break_by_slices=False)
        response = service.GetLatestLoggerData(request, None)

        self.assertEqual(len(response.points), 1)
        self.assertEqual(response.points[0].metric_name, "train/acc")
        signal_logger.get_and_clear_queue.assert_called_once()

    def test_get_latest_logger_data_break_by_slices(self):
        signal_logger = MagicMock()
        signal_logger.get_signal_history_per_sample.return_value = {
            "test/loss": {
                "exp": [
                    {"sample_id": "11", "model_age": 3, "metric_value": 0.3, "experiment_hash": "exp"},
                    {"sample_id": "12", "model_age": 3, "metric_value": 0.6, "experiment_hash": "exp"},
                ]
            }
        }
        df_manager = MagicMock()
        df_manager.get_df_view.return_value = pd.DataFrame(
            {"tag:hard": [True, False]},
            index=[11, 12],
        )

        ctx = _DummyCtx(components={"signal_logger": signal_logger, "df_manager": df_manager})
        with patch("weightslab.trainer.services.experiment_service.DataService"):
            service = ExperimentService(ctx)

        request = pb2.GetLatestLoggerDataRequest(
            request_full_history=False,
            break_by_slices=True,
            tags=["hard"],
            graph_name="test/loss",
        )
        response = service.GetLatestLoggerData(request, None)

        self.assertEqual(len(response.points), 1)
        self.assertEqual(response.points[0].sample_id, "11")
        self.assertEqual(response.points[0].metric_name, "test/loss")

    def test_restore_checkpoint_missing_manager(self):
        ctx = _DummyCtx(components={"trainer": None, "hyperparams": {}})
        with patch("weightslab.trainer.services.experiment_service.DataService"):
            service = ExperimentService(ctx)

        with patch("weightslab.trainer.services.experiment_service.ledgers.get_checkpoint_manager", return_value=None):
            response = service.RestoreCheckpoint(pb2.RestoreCheckpointRequest(experiment_hash="abc"), None)

        self.assertFalse(response.success)

    def test_restore_checkpoint_weights_step_mode(self):
        trainer = MagicMock()
        checkpoint_manager = MagicMock()
        checkpoint_manager.load_state.return_value = True
        hp = {}

        ctx = _DummyCtx(
            components={
                "trainer": trainer,
                "hyperparams": hp,
                "checkpoint_manager": checkpoint_manager,
            }
        )
        with patch("weightslab.trainer.services.experiment_service.DataService"):
            service = ExperimentService(ctx)

        response = service.RestoreCheckpoint(
            pb2.RestoreCheckpointRequest(experiment_hash="abc@@weights_step=5"),
            None,
        )

        self.assertTrue(response.success)
        checkpoint_manager.load_state.assert_called_once()
        _, kwargs = checkpoint_manager.load_state.call_args
        self.assertEqual(kwargs.get("target_step"), 5)


class TestModelServiceUnit(unittest.TestCase):
    def test_get_weights_success(self):
        layer = MagicMock()
        layer.__class__.__name__ = "Linear"
        layer.in_neurons = 3
        layer.out_neurons = 2
        layer.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        model = MagicMock()
        model.get_layer_by_id.return_value = layer

        ctx = _DummyCtx(components={"model": model})
        service = ModelService(ctx)

        req = pb2.WeightsRequest()
        req.neuron_id.layer_id = 1
        req.neuron_id.neuron_id = 0

        resp = service.GetWeights(req, None)

        self.assertTrue(resp.success)
        self.assertEqual(resp.layer_type, "Linear")
        self.assertEqual(len(resp.weights), 3)

    def test_get_activations_returns_empty_for_last_layer(self):
        last_layer = MagicMock()
        last_layer.get_module_id.return_value = 7

        model = MagicMock()
        model.layers = [last_layer]

        ctx = _DummyCtx(components={"model": model})
        service = ModelService(ctx)

        req = pb2.ActivationRequest(layer_id=7, sample_id="1", origin="train")
        resp = service.GetActivations(req, None)

        self.assertEqual(resp.neurons_count, 0)

    def test_manipulate_weights_invalid_op_type(self):
        ctx = _DummyCtx(components={"model": None})
        service = ModelService(ctx)

        req = pb2.WeightsOperationRequest()
        req.weight_operation.op_type = 999

        resp = service.ManipulateWeights(req, None)

        self.assertFalse(resp.success)

    def test_get_samples_uses_process_sample_results(self):
        tracked_dataset = MagicMock()
        tracked_dataset._dataset_split = "train"
        loader = MagicMock()
        loader.tracked_dataset = tracked_dataset

        model = MagicMock()
        model.tasks = []
        model.task_type = "classification"
        model.num_classes = 10

        ctx = _DummyCtx(components={"main": loader, "model": model})
        service = ModelService(ctx)

        class _Req:
            origin = "train"
            sample_ids = ["1", "2"]

            def HasField(self, field_name):
                return False

        with patch("weightslab.backend.ledgers.get_dataloaders", return_value=["main"]), \
             patch("weightslab.trainer.services.model_service.process_sample") as mock_process:
            mock_process.side_effect = [
                ("1", b"img1", b"raw1", 0, b"", b""),
                ("2", b"img2", b"raw2", 1, b"", b""),
            ]
            resp = service.GetSamples(_Req(), None)

        self.assertEqual(len(resp.samples), 2)
        self.assertEqual(resp.samples[0].sample_id, "1")
        self.assertEqual(resp.samples[1].sample_id, "2")


class TestDataServiceHelpersUnit(unittest.TestCase):
    def test_create_data_stat_sets_defaults(self):
        stat = create_data_stat("foo", "scalar")
        self.assertEqual(stat.name, "foo")
        self.assertEqual(stat.type, "scalar")
        self.assertEqual(list(stat.shape), [])

    def test_generate_thumbnail_returns_bytes(self):
        img = Image.new("RGB", (256, 128), color=(20, 30, 40))
        thumb = generate_thumbnail(img)
        self.assertIsInstance(thumb, bytes)
        self.assertGreater(len(thumb), 0)

    def test_data_service_origin_filter_and_df_filter(self):
        service = DataService.__new__(DataService)

        class _Req:
            origins = ["train", "val"]

        origins = service._get_origin_filter(_Req())
        self.assertEqual(origins, ["train", "val"])

        df = pd.DataFrame(
            {"value": [1, 2, 3], "origin": ["train", "val", "test"]}
        )
        filtered = service._filter_df_by_origin(df, ["train", "val"])
        self.assertEqual(len(filtered), 2)

    def test_get_unique_tags_returns_sorted_names(self):
        service = DataService.__new__(DataService)
        service._all_datasets_df = pd.DataFrame(
            {
                f"{SampleStatsEx.TAG.value}:zeta": [True],
                f"{SampleStatsEx.TAG.value}:alpha": [False],
                "value": [1],
            }
        )

        tags = service._get_unique_tags()
        self.assertEqual(tags, ["alpha", "zeta"])

    def test_parse_tags_handles_mixed_separators(self):
        service = DataService.__new__(DataService)
        parsed = service._parse_tags(" hard , medium;easy ; ")
        self.assertEqual(parsed, {"hard", "medium", "easy"})

    def test_is_nan_value_handles_scalars_and_arrays(self):
        service = DataService.__new__(DataService)
        self.assertTrue(service._is_nan_value(float("nan")))
        self.assertFalse(service._is_nan_value([float("nan")]))
        self.assertFalse(service._is_nan_value(1.0))

    def test_calculate_tag_column_updates_remove_sets_false(self):
        service = DataService.__new__(DataService)
        index = pd.MultiIndex.from_tuples([("train", 10)], names=[SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value])
        service._all_datasets_df = pd.DataFrame(
            {f"{SampleStatsEx.TAG.value}:hard": [True]},
            index=index,
        )

        updates = service._calculate_tag_column_updates(
            sample_id=10,
            origin="train",
            new_tag_name="hard",
            edit_type=SampleEditType.EDIT_REMOVE,
        )

        self.assertEqual(updates, {f"{SampleStatsEx.TAG.value}:hard": False})

    def test_calculate_tag_column_updates_accumulate_toggles_existing(self):
        service = DataService.__new__(DataService)
        index = pd.MultiIndex.from_tuples([("train", 7)], names=[SampleStatsEx.ORIGIN.value, SampleStatsEx.SAMPLE_ID.value])
        service._all_datasets_df = pd.DataFrame(
            {f"{SampleStatsEx.TAG.value}:focus": [True]},
            index=index,
        )

        updates = service._calculate_tag_column_updates(
            sample_id=7,
            origin="train",
            new_tag_name="focus",
            edit_type=SampleEditType.EDIT_ACCUMULATE,
        )

        self.assertEqual(updates, {f"{SampleStatsEx.TAG.value}:focus": False})

    def test_parse_direct_query_builds_filter_and_view_sort(self):
        service = DataService.__new__(DataService)
        ops = service._parse_direct_query("loss > 0.5 sortby index desc scope:view start:2 count:3")

        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0]["function"], "df.query")
        self.assertEqual(ops[0]["params"]["expr"], "loss > 0.5")
        self.assertEqual(ops[1]["function"], "df.sort_view_slice")
        self.assertEqual(ops[1]["params"]["by"], ["index"])
        self.assertEqual(ops[1]["params"]["ascending"], [False])
        self.assertEqual(ops[1]["params"]["start"], 2)
        self.assertEqual(ops[1]["params"]["count"], 3)

    def test_apply_agent_operation_query_filters_dataframe(self):
        service = DataService.__new__(DataService)
        df = pd.DataFrame({"x": [1, 2, 3]})

        message = service._apply_agent_operation(df, "df.query", {"expr": "x > 1"})

        self.assertIn("Applied query", message)
        self.assertEqual(list(df["x"]), [2, 3])

    def test_apply_agent_operation_analyze_safety_violation(self):
        service = DataService.__new__(DataService)
        df = pd.DataFrame({"x": [1, 2]})

        result = service._apply_agent_operation(df, "df.analyze", {"code": "import os"})

        self.assertEqual(result, "Safety Violation")


if __name__ == "__main__":
    unittest.main()
