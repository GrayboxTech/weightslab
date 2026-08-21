Resource Monitoring
====================

WeightsLab automatically tracks system and process resource usage — CPU,
memory, disk, network, and GPU — for the whole lifetime of a running
backend, and logs every value through the same signal pipeline used for
losses and metrics. The resulting curves appear in Weights Studio exactly
like any other signal, under graph names prefixed with ``resource/``.

This is enabled by default and requires no setup. It runs independently of
the training loop — metrics are sampled on a wall-clock interval, not tied
to training steps, so they keep updating even while training is paused or
between experiments.

What gets logged
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Category
     - Metrics
     - Signal names
   * - ``cpu``
     - System-wide CPU utilization (%)
     - ``resource/cpu/utilization_percent``
   * - ``memory``
     - System-wide memory utilization (%)
     - ``resource/memory/system_utilization_percent``
   * - ``disk``
     - Disk usage (%, GB) of ``disk_path``; cumulative bytes read/written (MB)
     - ``resource/disk/utilization_percent``, ``resource/disk/utilization_gb``,
       ``resource/disk/read_mb``, ``resource/disk/written_mb``
   * - ``network``
     - Cumulative bytes sent/received
     - ``resource/network/bytes_sent``, ``resource/network/bytes_received``
   * - ``process``
     - CPU %, thread count, RSS memory (MB/%), and system memory available (MB)
       for the WeightsLab backend process itself
     - ``resource/process/cpu_utilization_percent``,
       ``resource/process/cpu_threads_in_use``,
       ``resource/process/memory_in_use_mb``,
       ``resource/process/memory_in_use_percent``,
       ``resource/process/memory_available_mb``
   * - ``gpu``
     - Per-device memory/SM clock (MHz), memory used (bytes/%), temperature (°C)
     - ``resource/gpu/<index>/memory_clock_mhz``,
       ``resource/gpu/<index>/sm_clock_mhz``,
       ``resource/gpu/<index>/memory_allocated_bytes``,
       ``resource/gpu/<index>/memory_allocated_percent``,
       ``resource/gpu/<index>/temperature_celsius``

CPU/memory/disk/network/process metrics come from `psutil
<https://psutil.readthedocs.io/>`_. GPU metrics come from NVML (the
``pynvml`` import name, shipped by the ``nvidia-ml-py`` package) and are
per-device — multi-GPU machines get one full set of ``gpu`` signals per
device index. On a machine with no NVIDIA driver, the ``gpu`` category
degrades silently to a no-op; every other category is unaffected.

Sampling is wall-clock driven, but the x value each sample is logged against
is the **watched model's age** — the same axis your loss and metric curves
use. That is what lets a resource curve be read directly against a training
signal (or merged onto one chart with it), and it means resource curves
restart at 0 when training does instead of carrying on from wherever process
uptime had reached. One sample is kept per step, so a paused run — whose age
does not move — leaves the curve waiting rather than stacking points at the
same x. Before any model is registered, samples land at step 0.

Set ``WL_RESOURCE_MONITOR_STEP_SOURCE=seconds`` (or ``step_source: seconds``
in the YAML) for the older behaviour: elapsed seconds since the monitor
started, which counts process uptime and shares its axis with nothing else.
Worth it when you care about wall-clock behaviour, such as a leak developing
over hours.

Disabling monitoring
---------------------

To turn everything off, either:

.. code-block:: bash

   export WEIGHTSLAB_DISABLE_RESOURCE_MONITORING=1

or set ``enabled: false`` in ``resource_monitoring.yaml`` (see below) —
the YAML value wins if both are set.

Enabling only specific categories
----------------------------------

Two ways to restrict which categories are sampled:

- **Env var**, comma-separated category list (anything not listed is
  disabled):

  .. code-block:: bash

     export WL_RESOURCE_MONITOR_CATEGORIES=cpu,memory,gpu

- **YAML file** (``resource_monitoring.yaml``), per-category booleans —
  lets you leave everything on and disable just one or two:

  .. code-block:: yaml

     resource_monitoring:
       categories:
         disk: false
         network: false

Config file
------------

Create ``resource_monitoring.yaml`` at your repository root (next to
``agent_config.yaml``, if you use the agent) to control monitoring without
touching env vars:

.. code-block:: yaml

   resource_monitoring:
     enabled: true          # master switch
     interval_seconds: 15   # how often (seconds) to sample + log a batch of metrics
     disk_path: "/"         # filesystem path reported by the `disk` category
     step_source: model_age # x axis: model_age (default) or seconds
     categories:
       cpu: true
       memory: true
       disk: true
       network: true
       process: true
       gpu: true

Config lookup order
~~~~~~~~~~~~~~~~~~~~

1. ``<WL_RESOURCE_MONITOR_CONFIG_PATH>/.resource_monitoring.yaml`` /
   ``<WL_RESOURCE_MONITOR_CONFIG_PATH>/resource_monitoring.yaml`` (if
   ``WL_RESOURCE_MONITOR_CONFIG_PATH`` is set)
2. Repository-level ``resource_monitoring.yaml``
3. Package-level ``resource_monitoring.yaml``
4. Current working directory ``resource_monitoring.yaml``

Any key present in the YAML file overrides the corresponding env var (or
the built-in default); keys the file omits keep whatever the env var (or
default) already resolved to.

Environment variables
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Variable
     - Default
     - Description
   * - ``WEIGHTSLAB_DISABLE_RESOURCE_MONITORING``
     - ``0``
     - If set to ``1`` / ``true`` / ``yes`` / ``on``, disables resource
       monitoring entirely (no background thread is started).
   * - ``WL_RESOURCE_MONITOR_INTERVAL_SECONDS``
     - ``15``
     - How often (seconds) the monitor samples and logs a new batch of
       metrics. Clamped to a 1-second floor.
   * - ``WL_RESOURCE_MONITOR_CATEGORIES``
     - *(unset — all categories on)*
     - Comma-separated list of categories to enable
       (``cpu``, ``memory``, ``disk``, ``network``, ``process``, ``gpu``).
       When set, any category not listed is disabled.
   * - ``WL_RESOURCE_MONITOR_DISK_PATH``
     - OS root (``/`` or ``C:\``)
     - Filesystem path reported by the ``disk`` category's usage metrics.
   * - ``WL_RESOURCE_MONITOR_STEP_SOURCE``
     - ``model_age``
     - What the sampled values are plotted against. ``model_age`` logs each
       sample at the watched model's current age, so resource curves share the
       x axis of the loss/metric curves beside them and restart at 0 when
       training does; one sample is kept per step, so a paused run (whose age
       does not move) does not stack points at the same x. ``seconds`` restores
       the previous behaviour: elapsed seconds since the monitor started, which
       only ever counts process uptime. Before any model is registered, samples
       land at step 0.
   * - ``WL_RESOURCE_MONITOR_CONFIG_PATH``
     - *(empty)*
     - Optional directory override for ``resource_monitoring.yaml``. When
       set, WeightsLab first checks
       ``<WL_RESOURCE_MONITOR_CONFIG_PATH>/resource_monitoring.yaml`` before
       the built-in fallback paths.

Where it runs
---------------

The monitor is started once, alongside the watchdog, from
``grpc_serve()`` (``weightslab/trainer/trainer_services.py``) — so it covers
the whole backend server lifetime, not just active training. It is a single
daemon thread (``WL-ResourceMonitor``) and stops automatically when the
process exits.
