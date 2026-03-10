import unittest
from unittest.mock import MagicMock, patch

import weightslab.trainer.trainer_services as trainer_services


class TestExperimentServiceServicerDelegation(unittest.TestCase):
    def test_servicer_delegates_to_subservices(self):
        exp_service = MagicMock()
        exp_service.model_service = MagicMock()
        exp_service.data_service = MagicMock()

        servicer = trainer_services.ExperimentServiceServicer(exp_service=exp_service)
        req = object()
        ctx = object()

        servicer.GetSamples(req, ctx)
        servicer.GetWeights(req, ctx)
        servicer.GetActivations(req, ctx)
        servicer.ApplyDataQuery(req, ctx)
        servicer.GetDataSamples(req, ctx)
        servicer.EditDataSample(req, ctx)
        servicer.GetDataSplits(req, ctx)
        servicer.CheckAgentHealth(req, ctx)
        servicer.GetLatestLoggerData(req, ctx)
        servicer.ExperimentCommand(req, ctx)
        servicer.ManipulateWeights(req, ctx)
        servicer.RestoreCheckpoint(req, ctx)

        exp_service.model_service.GetSamples.assert_called_once_with(req, ctx)
        exp_service.model_service.GetWeights.assert_called_once_with(req, ctx)
        exp_service.model_service.GetActivations.assert_called_once_with(req, ctx)
        exp_service.data_service.ApplyDataQuery.assert_called_once_with(req, ctx)
        exp_service.data_service.GetDataSamples.assert_called_once_with(req, ctx)
        exp_service.data_service.EditDataSample.assert_called_once_with(req, ctx)
        exp_service.data_service.GetDataSplits.assert_called_once_with(req, ctx)
        exp_service.data_service.CheckAgentHealth.assert_called_once_with(req, ctx)
        exp_service.GetLatestLoggerData.assert_called_once_with(req, ctx)
        exp_service.ExperimentCommand.assert_called_once_with(req, ctx)
        exp_service.model_service.ManipulateWeights.assert_called_once_with(req, ctx)
        exp_service.RestoreCheckpoint.assert_called_once_with(req, ctx)


class TestGrpcServe(unittest.TestCase):
    def test_grpc_serve_starts_thread_and_server(self):
        fake_server = MagicMock()

        class _InstantThread:
            def __init__(self, target=None, daemon=None, name=None):
                self._target = target
                self.daemon = daemon
                self.name = name
                self.ident = 12345

            def start(self):
                if self._target is not None:
                    self._target()

        with patch("weightslab.trainer.trainer_services.grpc.server", return_value=fake_server), \
             patch("weightslab.trainer.trainer_services.pb2_grpc.add_ExperimentServiceServicer_to_server") as add_servicer, \
             patch("weightslab.trainer.trainer_services.Thread", _InstantThread), \
             patch("weightslab.trainer.trainer_services.ExperimentServiceServicer") as servicer_cls:
            trainer_services.grpc_serve(n_workers_grpc=2, grpc_host="127.0.0.1", grpc_port=50052)

        servicer_cls.assert_called_once()
        add_servicer.assert_called_once()
        fake_server.add_insecure_port.assert_called_once_with("127.0.0.1:50052")
        fake_server.start.assert_called_once()
        fake_server.wait_for_termination.assert_called_once()


if __name__ == "__main__":
    unittest.main()
