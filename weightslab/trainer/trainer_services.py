import io
import types
import time
import grpc
import torch
import logging
import traceback
import numpy as np
import pandas as pd  # <- needed for data service

import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from threading import Thread
from concurrent import futures

from weightslab.components.global_monitoring import weightslab_rlock, pause_controller
from weightslab.trainer.trainer_tools import *
from weightslab.trainer.trainer_tools import _get_input_tensor_for_sample
from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType

from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService

# Global logger
logger = logging.getLogger(__name__)


class ExperimentServiceServicer(pb2_grpc.ExperimentServiceServicer):
    """
    gRPC servicer for experiment-related services.

    This class now delegates to a domain-level ExperimentService instance
    which in turn uses smaller sub-services (model/data/etc.).
    """

    def __init__(self, exp_name: str = None, exp_service: ExperimentService | None = None):
        if exp_service is None:
            ctx = ExperimentContext(exp_name=exp_name)
            exp_service = ExperimentService(ctx=ctx)
        self._exp_service = exp_service

    # -------------------------------------------------------------------------
    # Training status stream
    # -------------------------------------------------------------------------
    def StreamStatus(self, request_iterator, context):
        logger.debug(f"ExperimentServiceServicer.StreamStatus({request_iterator})")
        # delegate to domain ExperimentService
        for status in self._exp_service.stream_status(request_iterator):
            yield status

    # -------------------------------------------------------------------------
    # Sample retrieval (images / segmentation / recon)
    # -------------------------------------------------------------------------
    def GetSamples(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetSamples({request})")
        return self._exp_service.model_service.get_samples(request)

    # -------------------------------------------------------------------------
    # Weights inspection
    # -------------------------------------------------------------------------
    def GetWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetWeights({request})")
        return self._exp_service.model_service.get_weights(request)

    # -------------------------------------------------------------------------
    # Activations
    # -------------------------------------------------------------------------
    def GetActivations(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetActivations({request})")
        return self._exp_service.model_service.get_activations(request)

    # -------------------------------------------------------------------------
    # Data service helpers + RPCs (for weights_studio UI)
    # -------------------------------------------------------------------------
    def ApplyDataQuery(self, request, context):
        return self._exp_service.data_service.ApplyDataQuery(request, context)

    def GetDataSamples(self, request, context):
        return self._exp_service.data_service.GetDataSamples(request, context)

    def EditDataSample(self, request, context):
        return self._exp_service.data_service.EditDataSample(request, context)

    # -------------------------------------------------------------------------
    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def ExperimentCommand(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ExperimentCommand({request})")
        return self._exp_service.ExperimentCommand(request, context)

    # -------------------------------------------------------------------------
    # Weight manipulation (architecture operations)
    # -------------------------------------------------------------------------
    def ManipulateWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ManipulateWeights({request})")
        return self._exp_service.model_service.ManipulateWeights(request, context)


# -----------------------------------------------------------------------------
# Serving gRPC communication
# -----------------------------------------------------------------------------
def grpc_serve(n_workers_grpc: int = 6, port_grpc: int = 50051, **_):
    """Configure trainer services such as gRPC server.

    Args:
        n_workers_grpc (int): Number of threads for the gRPC server.
        port_grpc (int): Port number for the gRPC server.
    """
    import weightslab.trainer.trainer_services as trainer
    from weightslab.trainer.trainer_tools import force_kill_all_python_processes

    def serving_thread_callback():
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=n_workers_grpc))
        servicer = trainer.ExperimentServiceServicer()
        pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f"[::]:{port_grpc}")
        try:
            server.start()
            logger.info("gRPC Server started on port %d. Press Ctrl+C to stop.", port_grpc)
            server.wait_for_termination()
        except KeyboardInterrupt:
            force_kill_all_python_processes()

    training_thread = Thread(target=serving_thread_callback)
    training_thread.start()


if __name__ == "__main__":
    grpc_serve()
