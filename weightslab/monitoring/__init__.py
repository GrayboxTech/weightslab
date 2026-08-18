"""weightslab.monitoring — automatic system resource monitoring.

Public API
----------
ResourceMonitor : class — background thread sampling CPU/memory/disk/network/GPU/process
load_resource_monitoring_config : func — resolve config from env vars + YAML
start_resource_monitor_from_config : func — start the process-wide singleton if enabled
get_resource_monitor : func — return the running singleton, if any
stop_resource_monitor : func — stop and clear the singleton
"""

from weightslab.monitoring.resource_monitor import ( # noqa: F401
    ResourceMonitor,
    load_resource_monitoring_config,
    start_resource_monitor_from_config,
    get_resource_monitor,
    stop_resource_monitor,
)
