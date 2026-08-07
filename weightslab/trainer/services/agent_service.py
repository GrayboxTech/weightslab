"""
services/agent_service.py
=========================
gRPC surface for AI-agent lifecycle management.

Responsibilities:
  - CheckAgentHealth : report whether any LLM provider is ready.
  - InitializeAgent : wire up a cloud provider from a user-supplied API key.

The actual ``DataManipulationAgent`` instance lives inside ``DataService``
because it requires the live dataframe context (schema, column index, etc.)
that ``DataService`` owns. ``AgentService`` receives a reference to
``DataService`` at construction time and delegates to its agent.

Wire-up (in ExperimentService):
    data_service = DataService(ctx)
    agent_service = AgentService(data_service)
"""

import logging

import weightslab.proto.experiment_service_pb2 as pb2

from weightslab.trainer.services.utils.tools import AGENT_PROVIDER_MAP, safe_grpc, truncate

logger = logging.getLogger(__name__)


class AgentService:
    """Handles agent health-check and cloud-provider initialization over gRPC."""

    def __init__(self, data_service):
        # We keep a reference to DataService rather than to the agent directly
        # because the agent may be re-created (e.g. after initialize_with_cloud_key).
        self._data_service = data_service

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _agent(self):
        """Convenience accessor for the agent living inside DataService."""
        return getattr(self._data_service, "_agent", None)

    def _is_available(self) -> bool:
        """Delegate availability check to DataService's internal helper."""
        try:
            return self._data_service._is_agent_available()
        except Exception as exc:
            logger.debug("AgentService._is_available error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # gRPC methods
    # ------------------------------------------------------------------

    @safe_grpc(lambda msg: pb2.AgentHealthResponse(available=False, message=msg))
    def CheckAgentHealth(self, request, context):
        """
        Check whether any LLM provider is ready to serve queries.

        Returns:
            AgentHealthResponse { available: bool, message: str }
              - available=True → "Ready to help you."
              - available=False → "Agent not configured. Type /init to set up."
        """
        available = self._is_available()
        message = (
            "Agent available. Ready to help you. Type /model to test another model or /reset to clear your API key and start over."
            if available
            else "Agent not configured. Type /init to set up."
        )
        logger.debug("CheckAgentHealth → available=%s", available)
        return pb2.AgentHealthResponse(available=available, message=message)

    @safe_grpc(lambda msg: pb2.InitializeAgentResponse(success=False, message=msg))
    def InitializeAgent(self, request, context):
        """
                Initialize the OpenCode agent backend.

                Supported providers (AgentProviderType enum):
                    PROVIDER_OPENCODE (1) — a local OpenCode server; api_key is
                        ignored, the credential lives in OpenCode's own config.
                PROVIDER_OPENROUTER (0) is a legacy enum value kept for wire
                compatibility with older frontends; requesting it is rejected
                below rather than removed from the .proto, so an old client
                sending it gets a clean error instead of a decode failure.

        Returns:
            InitializeAgentResponse { success: bool, message: str }
        """
        agent = self._agent
        if agent is None:
            return pb2.InitializeAgentResponse(
                success=False,
                message="Agent backend is not running.",
            )

        provider_name = AGENT_PROVIDER_MAP.get(request.provider)
        if provider_name != "opencode":
            return pb2.InitializeAgentResponse(
                success=False,
                message="Only OpenCode is supported as the agent backend.",
            )

        logger.debug(
            "InitializeAgent → provider=%s model=%s key=%s",
            provider_name,
            truncate(request.model or "", 64),
            truncate(request.api_key, 12) + "…",
        )

        success, message = agent.initialize_with_cloud_key(request.api_key, provider_name, request.model)
        return pb2.InitializeAgentResponse(success=success, message=message)

    @safe_grpc(lambda msg: pb2.ChangeAgentModelResponse(success=False, message=msg))
    def ChangeAgentModel(self, request, context):
        """
        Switch the active OpenCode model without re-entering credentials.

        Returns:
            ChangeAgentModelResponse { success: bool, message: str }
        """
        agent = self._agent
        if agent is None:
            return pb2.ChangeAgentModelResponse(
                success=False,
                message="Agent backend is not running.",
            )

        logger.debug("ChangeAgentModel → model=%s", truncate(request.model or "", 64))
        success, message = agent.change_model(request.model)
        return pb2.ChangeAgentModelResponse(success=success, message=message)

    @safe_grpc(lambda msg: pb2.GetAgentModelsResponse(success=False, models=[], message=msg))
    def GetAgentModels(self, request, context):
        """
        Return every provider/model the OpenCode server is authenticated for.

        Returns:
            GetAgentModelsResponse { success: bool, models: [str], message: str }
        """
        agent = self._agent
        if agent is None:
            return pb2.GetAgentModelsResponse(
                success=False,
                models=[],
                message="Agent backend is not running.",
            )

        ok, models, message = agent.get_available_models()
        return pb2.GetAgentModelsResponse(success=ok, models=models, message=message)

    @safe_grpc(lambda msg: pb2.ResetAgentResponse(success=False, message=msg))
    def ResetAgent(self, request, context):
        """Clear agent connection state and return to the uninitialized status."""
        agent = self._agent
        if agent is None:
            return pb2.ResetAgentResponse(
                success=False,
                message="Agent backend is not running.",
            )

        logger.debug("ResetAgent")
        success, message = agent.reset_connection()
        return pb2.ResetAgentResponse(success=success, message=message)

    @safe_grpc(lambda msg: pb2.ClearAgentHistoryResponse(success=False, message=msg))
    def ClearAgentHistory(self, request, context):
        """Wipe the agent's conversation history without touching the provider
        connection -- distinct from ResetAgent."""
        agent = self._agent
        if agent is None:
            return pb2.ClearAgentHistoryResponse(
                success=False,
                message="Agent backend is not running.",
            )

        logger.debug("ClearAgentHistory")
        success, message = agent.clear_history()
        return pb2.ClearAgentHistoryResponse(success=success, message=message)

    @safe_grpc(lambda msg: pb2.CompactAgentHistoryResponse(success=False, message=msg))
    def CompactAgentHistory(self, request, context):
        """Summarize the agent's conversation history via the active provider,
        replacing it with the summary."""
        agent = self._agent
        if agent is None:
            return pb2.CompactAgentHistoryResponse(
                success=False,
                message="Agent backend is not running.",
            )

        logger.debug("CompactAgentHistory")
        success, message = agent.compact_history()
        return pb2.CompactAgentHistoryResponse(success=success, message=message)

    @safe_grpc(lambda msg: pb2.GetAgentContextUsageResponse(success=False, message=msg))
    def GetAgentContextUsage(self, request, context):
        """Context-window usage breakdown for the active model (backs the
        /context command) -- see DataManipulationAgent.get_context_usage."""
        agent = self._agent
        if agent is None:
            return pb2.GetAgentContextUsageResponse(
                success=False,
                message="Agent backend is not running.",
            )

        logger.debug("GetAgentContextUsage")
        ok, usage, message = agent.get_context_usage()
        return pb2.GetAgentContextUsageResponse(
            success=ok,
            message=message,
            model=usage.get("model") or "",
            context_window=usage.get("context_window") or 0,
            input_tokens=usage.get("input_tokens") or 0,
            output_tokens=usage.get("output_tokens") or 0,
            reasoning_tokens=usage.get("reasoning_tokens") or 0,
            cache_read_tokens=usage.get("cache_read_tokens") or 0,
            cache_write_tokens=usage.get("cache_write_tokens") or 0,
        )
        return pb2.CompactAgentHistoryResponse(success=success, message=message)
