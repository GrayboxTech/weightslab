import os
import grpc
import logging
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

from threading import Thread
from concurrent import futures

from weightslab.trainer.trainer_tools import *

from weightslab.trainer.experiment_context import ExperimentContext
from weightslab.trainer.services.experiment_service import ExperimentService

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
            self._ctx = ctx
        self._exp_service = exp_service

    # -------------------------------------------------------------------------
    # Sample retrieval (images / segmentation / recon)
    # -------------------------------------------------------------------------
    def GetSamples(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetSamples({request})")
        return self._exp_service.model_service.GetSamples(request, context)

    # -------------------------------------------------------------------------
    # Weights inspection
    # -------------------------------------------------------------------------
    def GetWeights(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetWeights({request})")
        return self._exp_service.model_service.GetWeights(request, context)

    # -------------------------------------------------------------------------
    # Activations
    # -------------------------------------------------------------------------
    def GetActivations(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetActivations({request})")
        return self._exp_service.model_service.GetActivations(request, context)

    # -------------------------------------------------------------------------
    # Data service helpers + RPCs (for weights_studio UI)
    # -------------------------------------------------------------------------
    def ApplyDataQuery(self, request, context):
        logger.debug(f"ExperimentServiceServicer.ApplyDataQuery({request})")
        return self._exp_service.data_service.ApplyDataQuery(request, context)

    def GetDataSamples(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetDataSamples({request})")
        return self._exp_service.data_service.GetDataSamples(request, context)

    def EditDataSample(self, request, context):
        logger.debug(f"ExperimentServiceServicer.EditDataSample({request})")
        return self._exp_service.data_service.EditDataSample(request, context)

    def GetDataSplits(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetDataSplits({request})")
        return self._exp_service.data_service.GetDataSplits(request, context)

    def CheckAgentHealth(self, request, context):
        logger.debug(f"ExperimentServiceServicer.CheckAgentHealth({request})")
        return self._exp_service.data_service.CheckAgentHealth(request, context)

    # -------------------------------------------------------------------------
    # Logger data sync for WeightsStudio
    # -------------------------------------------------------------------------
    def GetLatestLoggerData(self, request, context):
        logger.debug(f"ExperimentServiceServicer.GetLatestLoggerData({request})")
        return self._exp_service.GetLatestLoggerData(request, context)

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

    # -------------------------------------------------------------------------
    # Checkpoint restore
    # -------------------------------------------------------------------------
    def RestoreCheckpoint(self, request, context):
        logger.debug(f"ExperimentServiceServicer.RestoreCheckpoint({request})")
        return self._exp_service.RestoreCheckpoint(request, context)


# -----------------------------------------------------------------------------
# Serving gRPC communication
# -----------------------------------------------------------------------------
def grpc_serve(n_workers_grpc: int = None, grpc_host: str = "[::]", grpc_port: int = 50051, **_):
    """Configure trainer services such as gRPC server.

    Args:
        n_workers_grpc (int): Number of threads for the gRPC server.
        port_grpc (int): Port number for the gRPC server.
    """
    import weightslab.trainer.trainer_services as trainer
    from weightslab.trainer.trainer_tools import force_kill_all_python_processes

    grpc_port = int(os.getenv("GRPC_BACKEND_PORT", grpc_port))

    def serving_thread_callback():
        server = grpc.server(
            futures.ThreadPoolExecutor(
                thread_name_prefix="WL-gRPC-Worker",
                max_workers=n_workers_grpc
            )
        )
        servicer = trainer.ExperimentServiceServicer()
        pb2_grpc.add_ExperimentServiceServicer_to_server(servicer, server)
        server.add_insecure_port(f'{grpc_host}:'+ str(grpc_port))  # guarantees IPv4 connectivity from containers.
        try:
            server.start()
            server.wait_for_termination()
        except KeyboardInterrupt:
            force_kill_all_python_processes()

    training_thread = Thread(
        target=serving_thread_callback,
        daemon=True,
        name="WL-gRPC_Server",
    )
    training_thread.start()
    logger.info("grpc_thread_started", extra={
        "thread_name": training_thread.name,
        "thread_id": training_thread.ident,
        "grpc_host": grpc_host,
        "grpc_port": grpc_port,
        "n_workers_grpc": n_workers_grpc
    })

if __name__ == "__main__":
    grpc_serve()
