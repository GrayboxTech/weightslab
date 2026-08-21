"""
services/utils/tools.py
=======================
Shared utility helpers for the trainer service layer.

Keep this file free of heavy domain logic. It is the right place for:
  - Small, stateless helper functions used by two or more services.
  - Shared constants / lookup tables (e.g. provider maps).
  - Thin wrappers that reduce boilerplate inside service methods.
"""

import logging
from typing import TypeVar, Callable, Any


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent provider mapping
# ---------------------------------------------------------------------------

# Maps the AgentProviderType proto enum integer values to the internal
# provider names understood by DataManipulationAgent.initialize_with_cloud_key.
# OpenCode (1) is the only agent backend weightslab actually supports;
# PROVIDER_OPENROUTER (0) is kept here (rather than removed) purely for wire
# compatibility with older frontends, so a client that still sends it gets a
# clean "not supported" rejection from AgentService.InitializeAgent instead of
# an unrecognized-enum-value decode failure.
AGENT_PROVIDER_MAP: dict[int, str] = {
    0: "openrouter",
    1: "opencode",
}


# ---------------------------------------------------------------------------
# gRPC safe-call decorator
# ---------------------------------------------------------------------------

_R = TypeVar("_R")


def safe_grpc(default_factory: Callable[..., _R]) -> Callable:
    """
    Decorator that wraps a gRPC service method in a try/except block.

    On unhandled exception it logs the error and returns ``default_factory(str(exc))``
    so the caller always receives a well-formed response instead of a gRPC crash.

    Usage::

        @safe_grpc(lambda msg: pb2.AgentHealthResponse(available=False, message=msg))
        def CheckAgentHealth(self, request, context):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.error(
                    "[%s] Unhandled exception: %s",
                    fn.__qualname__,
                    exc,
                    exc_info=True,
                )
                return default_factory(f"Internal error: {exc}")
        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def truncate(value: Any, max_len: int = 120) -> str:
    """Return a string representation of *value* truncated to *max_len* chars.

    Useful for debug-log lines inside service methods where the full object
    repr would be too noisy.
    """
    s = str(value)
    return s if len(s) <= max_len else s[:max_len] + "…"
