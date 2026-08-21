.. _studio-resource-monitoring:

Resource monitoring
===================

.. figure:: ../_static/screenshots/resource-signals.png
   :alt: Plots board filtered to the resource monitoring signals
   :width: 100%

WeightsLab samples CPU, memory, disk, network, GPU and process usage in the
background for the whole life of the backend, and logs every value through the
**same signal pipeline as your losses and metrics**. There is no separate
dashboard: the curves land in the plots board like any other signal, named with
a ``resource/`` prefix.

This is on by default and needs no setup.

Finding the signals
===================

Type ``resource/`` into the :ref:`plots board search <studio-plots>` to pull
every resource curve to the front of the board. Narrow it from there —
``resource/gpu`` for the accelerators, ``resource/process`` for the backend
process itself, or ``resource/gpu|resource/memory`` to compare both at once
(search is regex by default).

The signals, by category:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Category
     - Signals
   * - ``cpu``
     - ``resource/cpu/utilization_percent``
   * - ``memory``
     - ``resource/memory/system_utilization_percent``
   * - ``disk``
     - ``resource/disk/utilization_percent``, ``…/utilization_gb``,
       ``…/read_mb``, ``…/written_mb``
   * - ``network``
     - ``resource/network/bytes_sent``, ``…/bytes_received``
   * - ``process``
     - ``resource/process/cpu_utilization_percent``, ``…/cpu_threads_in_use``,
       ``…/memory_in_use_mb``, ``…/memory_in_use_percent``,
       ``…/memory_available_mb``
   * - ``gpu``
     - ``resource/gpu/<index>/memory_clock_mhz``, ``…/sm_clock_mhz``,
       ``…/memory_allocated_bytes``, ``…/memory_allocated_percent``,
       ``…/temperature_celsius`` — one full set **per device**

Reading them next to your own curves
====================================

Sampling runs on a wall-clock cadence, but each sample is logged against the
**model's age** — the same x axis your loss and metric curves use. That is what
makes these plots worth having in the same board rather than a separate one:

- :ref:`Merge <studio-plots>` a resource curve with a training signal
  (``resource/gpu/0/memory_allocated_percent <> train_loss``) and read them on
  one chart. A batch-size change that moved GPU memory and a loss that moved at
  the same step line up visually.
- Resource curves **restart at 0 when training does**, so they stay comparable
  across restarts instead of carrying on from wherever process uptime had
  reached.
- While training is paused the model's age doesn't move, so samples don't stack
  into a vertical smear at one x — the curve simply waits.

Set ``WL_RESOURCE_MONITOR_STEP_SOURCE=seconds`` to plot against elapsed seconds
since the monitor started instead. Useful when you care about wall-clock
behaviour (a memory leak over hours) rather than per-step behaviour, at the cost
of an axis no other plot shares.

.. note::

   Because the monitor is tied to the backend rather than the training loop, the
   curves keep updating while training is **paused**, and between experiments.
   A GPU that stays pinned after you hit Pause is visible here.

Configuring it
==============

Two ways, and the YAML wins where both are set.

**Environment variables**, before starting the backend:

.. code-block:: bash

   export WEIGHTSLAB_DISABLE_RESOURCE_MONITORING=1        # turn it all off
   export WL_RESOURCE_MONITOR_INTERVAL_SECONDS=30         # sample less often
   export WL_RESOURCE_MONITOR_CATEGORIES=cpu,memory,gpu   # allowlist; others off
   export WL_RESOURCE_MONITOR_DISK_PATH=/data             # which filesystem to report
   export WL_RESOURCE_MONITOR_STEP_SOURCE=seconds         # x axis: wall-clock instead

**A** ``resource_monitoring.yaml`` **file**, which is the better fit when you
want to keep everything on and disable one thing:

.. code-block:: yaml

   resource_monitoring:
     enabled: true
     interval_seconds: 15
     disk_path: "/"
     step_source: model_age
     categories:
       disk: false        # everything else stays on
       network: false

The env var takes a comma-separated **allowlist** — anything not named is off —
while the YAML takes **per-category booleans**, so reach for the file when you
only want to switch one category off.

Practical settings
==================

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Situation
     - What to change
   * - Shared or metered filesystem
     - Point ``disk_path`` at the volume your data actually lives on; the
       default reports the OS root, which is rarely the interesting one.
   * - Long runs, board feels crowded
     - Raise ``interval_seconds``. At the default of 15s an overnight run logs
       thousands of points per signal.
   * - No NVIDIA GPU
     - Nothing — the ``gpu`` category detects the missing driver and no-ops.
       Every other category is unaffected.
   * - Profiling a memory leak
     - ``step_source: seconds``, so the axis tracks wall-clock uptime rather
       than restarting with training.
   * - Container with restricted ``/proc``
     - Narrow ``WL_RESOURCE_MONITOR_CATEGORIES`` to what the container can
       actually read.

See :doc:`../resource_monitoring` for the full reference — the config lookup order,
every environment variable, and where the monitor thread runs.
