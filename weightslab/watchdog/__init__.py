"""weightslab.watchdog — unified watchdog for locks and gRPC threads.

Public API
----------
WATCHDOG               : int   — custom log level (35, between WARNING and ERROR)
MonitoredRLock         : class — RLock with holder-thread tracking
raise_in_thread        : func  — deliver _WatchdogInterrupt to a thread by id
_WatchdogInterrupt     : class — BaseException raised in stuck threads
RpcWatchdogState       : class — tracks in-flight gRPC RPCs
RpcTimingAndWatchdogInterceptor : class — gRPC ServerInterceptor
GrpcServerManager      : class — controls gRPC server lifecycle / restarts
WeighlabsWatchdog      : class — unified watchdog (locks + gRPC)
"""

# Register WATCHDOG log level and logger.watchdog() method
from weightslab.watchdog.log_level import WATCHDOG  # noqa: F401

from weightslab.watchdog.lock_monitor import (  # noqa: F401
    MonitoredRLock,
    _WatchdogInterrupt,
    raise_in_thread,
)

from weightslab.watchdog.grpc_watchdog import (  # noqa: F401
    RpcWatchdogState,
    RpcTimingAndWatchdogInterceptor,
    GrpcServerManager,
)

from weightslab.watchdog.watchdog import WeighlabsWatchdog  # noqa: F401
