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
    # Training & hyperparameter commands
    # -------------------------------------------------------------------------
    def StreamStatus(self, request_iterator, context):
        logger.debug(f"ExperimentServiceServicer.StreamStatus({request_iterator})")

        # Retry to ensure components are ready
        max_retries = 10
        for attempt in range(max_retries):
            self._ctx.ensure_components()
            components = self._ctx.components
            is_model_interfaced = components.get("model") is not None
            has_streaming_logger = components.get("signal_logger") is not None

            if is_model_interfaced and has_streaming_logger:
                logger.info(f"StreamStatus: Components ready on attempt {attempt + 1}")
                break

            if attempt < max_retries - 1:
                logger.debug(f"StreamStatus: Waiting for components (attempt {attempt + 1}/{max_retries})")
                import time
                time.sleep(0.5)

        if not is_model_interfaced or not has_streaming_logger:
            logger.warning(
                f"StreamStatus: Components not ready after {max_retries} attempts. "
                f"{{'is_model_interfaced': {is_model_interfaced}, 'has_streaming_logger': {has_streaming_logger}}}"
            )
            # Yield empty/placeholder status instead of closing stream
            import weightslab.proto.experiment_service_pb2 as pb2
            yield pb2.TrainingStatusEx(
                timestamp="N/A",
                experiment_name="N/A",
                model_age=0
            )
            return

        # stream status updates to client
        for status in self._exp_service.StreamingStatus(request_iterator):
            yield status

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
