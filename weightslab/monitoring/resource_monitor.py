"""System resource monitoring — CPU / memory / disk / network / GPU / process.

Runs a single background daemon thread (started alongside the gRPC server in
``weightslab.trainer.trainer_services.grpc_serve``) that periodically samples
machine- and process-level resource usage and logs each value through the
same signal pipeline used for losses/metrics
(``weightslab.src._log_signal`` -> registered ``LoggerQueue.add_scalars``),
so the resulting curves show up in Weights Studio exactly like any other
signal.

Sampling is on a wall-clock cadence, but the x value each sample is logged
against is the MODEL'S AGE, so a resource curve lines up with the loss curves
beside it and restarts at 0 when training does. Logging elapsed seconds instead
(which is what this used to do) produced an axis that only ever counted the
process's uptime: restarting training from step 0 left the GPU-memory curve
carrying on from wherever it had got to, on a scale no other plot shared.
One sample per step is kept -- while training is paused (or between two steps
slower than the sampling interval) the age doesn't move, and repeating samples at
the same x would stack a vertical smear of points on top of each other rather
than extending the curve. Set ``WL_RESOURCE_MONITOR_STEP_SOURCE=seconds`` (or
``step_source: seconds`` in the YAML) to get the old elapsed-seconds axis back.

GPU metrics use NVML (via the ``pynvml`` import name, shipped by the
``nvidia-ml-py`` package) and are skipped gracefully when no NVIDIA driver
is present. Every other category is provided by ``psutil``.

See ``docs/resource_monitoring.rst`` for the user-facing config reference.
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import psutil
import yaml

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 15.0
DEFAULT_CATEGORIES = ("cpu", "memory", "gpu", "disk", "network", "process")

# What the sampled values are plotted against. "model_age" follows the training
# step (see the module docstring); "seconds" is the legacy elapsed-seconds axis.
STEP_SOURCES = ("model_age", "seconds")
DEFAULT_STEP_SOURCE = "model_age"


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_step_source(value, fallback: str) -> str:
    """Accept a step-source setting, ignoring anything unrecognised.

    A typo falls back rather than raising: this runs at server startup, and a
    misspelled axis is not worth refusing to boot over (it is logged instead).
    """
    if value is None:
        return fallback
    text = str(value).strip().lower()
    if text in STEP_SOURCES:
        return text
    logger.warning("[ResourceMonitor] Unknown step_source %r — using %r", value, fallback)
    return fallback


def load_resource_monitoring_config() -> dict:
    """Resolve resource-monitoring config with env-vars-then-YAML precedence
    (YAML overrides env vars if both set), mirroring the agent config loader
    in ``weightslab.trainer.services.agent.agent``.
    """
    enabled = not _bool_env("WEIGHTSLAB_DISABLE_RESOURCE_MONITORING", False)
    interval_seconds = float(os.environ.get("WL_RESOURCE_MONITOR_INTERVAL_SECONDS", DEFAULT_INTERVAL_SECONDS))
    disk_path = os.environ.get("WL_RESOURCE_MONITOR_DISK_PATH", os.sep)
    step_source = _coerce_step_source(
        os.environ.get("WL_RESOURCE_MONITOR_STEP_SOURCE"), DEFAULT_STEP_SOURCE)
    categories = {name: True for name in DEFAULT_CATEGORIES}

    env_categories = os.environ.get("WL_RESOURCE_MONITOR_CATEGORIES")
    if env_categories is not None:
        selected = {c.strip().lower() for c in env_categories.split(",") if c.strip()}
        categories = {name: (name in selected) for name in DEFAULT_CATEGORIES}

    repo_root = Path(__file__).resolve().parents[2] # weightslab/ root
    inner_pkg = Path(__file__).resolve().parents[1] # weightslab package dir

    config_dir = Path(os.environ.get("WL_RESOURCE_MONITOR_CONFIG_PATH", repo_root))
    config_paths = [
        config_dir / ".resource_monitoring.yaml",
        config_dir / "resource_monitoring.yaml",
        inner_pkg / "resource_monitoring.yaml",
        Path.cwd() / "resource_monitoring.yaml",
    ]
    for path in config_paths:
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
            if not cfg or "resource_monitoring" not in cfg:
                continue
            rm_cfg = cfg["resource_monitoring"] or {}

            enabled = bool(rm_cfg.get("enabled", enabled))
            interval_seconds = float(rm_cfg.get("interval_seconds", interval_seconds))
            disk_path = rm_cfg.get("disk_path", disk_path)
            step_source = _coerce_step_source(rm_cfg.get("step_source"), step_source)

            cats_cfg = rm_cfg.get("categories")
            if isinstance(cats_cfg, dict):
                categories = {
                    name: bool(cats_cfg.get(name, categories[name]))
                    for name in DEFAULT_CATEGORIES
                }

            logger.debug("[ResourceMonitor] Loaded config from %s", path)
        except Exception as e:
            logger.warning("[ResourceMonitor] Failed to load config %s: %s", path, e)
        break

    return {
        "enabled": enabled,
        "interval_seconds": interval_seconds,
        "categories": categories,
        "disk_path": disk_path,
        "step_source": step_source,
    }


class ResourceMonitor:
    """Background daemon thread sampling system resources on a fixed interval."""

    def __init__(
        self,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        categories: Optional[Dict[str, bool]] = None,
        disk_path: str = os.sep,
        step_source: str = DEFAULT_STEP_SOURCE,
    ) -> None:
        self._interval_seconds = max(float(interval_seconds), 1.0)
        self._categories = dict(categories) if categories else {name: True for name in DEFAULT_CATEGORIES}
        self._disk_path = disk_path or os.sep
        self._step_source = _coerce_step_source(step_source, DEFAULT_STEP_SOURCE)
        # Last x actually logged, so a step that hasn't moved is sampled once
        # instead of once per tick. None until the first sample.
        self._last_logged_step: Optional[int] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_monotonic: Optional[float] = None

        self._process = psutil.Process(os.getpid())
        self._gpu_available = False
        self._gpu_handles: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the resource-monitoring background thread."""
        self._start_monotonic = time.monotonic()

        # First call to cpu_percent() always returns a meaningless baseline
        # (0.0) since there's no prior sample to compare against — prime it
        # now so the first real tick already reports a useful value.
        try:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        if self._categories.get("gpu", True):
            self._init_gpu()

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="WL-ResourceMonitor", daemon=True)
        self._thread.start()
        logger.info(
            "[ResourceMonitor] Started (interval=%.1fs step_source=%s categories=%s)",
            self._interval_seconds,
            self._step_source,
            sorted(name for name, on in self._categories.items() if on),
        )

    def stop(self) -> None:
        """Stop the resource-monitoring background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 1.0)
        self._shutdown_gpu()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self._tick()
            except Exception:
                logger.exception("[ResourceMonitor] Unexpected error while sampling resources")

    def _model_age(self) -> Optional[int]:
        """The watched model's age, or None when there is no model to ask.

        The monitor starts with the gRPC server, which is typically before any
        model has been registered -- and the ledger proxy raises for a missing
        attribute rather than returning None, so every step of this lookup is
        guarded. None means "not knowable yet", not "zero".
        """
        try:
            from weightslab.backend.ledgers import get_model
            model = get_model()
        except Exception:
            return None
        if model is None:
            return None
        try:
            age = model.get_age()
        except Exception:
            return None
        try:
            return int(age)
        except (TypeError, ValueError):
            return None

    def _current_step(self) -> int:
        """The x value this tick's samples belong to.

        Before a model exists there is no age to follow, so the samples land at
        step 0 -- the pre-training baseline, and (thanks to the one-sample-per-step
        rule in _tick) exactly one point rather than a pile of them at the origin.
        """
        if self._step_source == "seconds":
            return int(round(time.monotonic() - (self._start_monotonic or 0.0)))
        age = self._model_age()
        return age if age is not None else 0

    def _tick(self) -> None:
        step = self._current_step()
        # Repeats at the same step are dropped rather than stacked on top of each
        # other (see the module docstring). Never applied to the seconds axis,
        # where the step advances on its own and a repeat would mean the clock
        # stood still.
        if self._step_source != "seconds":
            if step == self._last_logged_step:
                logger.debug("[ResourceMonitor] Step %d unchanged since the last sample — skipping", step)
                return
            self._last_logged_step = step
        metrics: Dict[str, float] = {}

        if self._categories.get("cpu", True):
            metrics.update(self._sample_cpu())
        if self._categories.get("memory", True):
            metrics.update(self._sample_memory())
        if self._categories.get("process", True):
            metrics.update(self._sample_process())
        if self._categories.get("disk", True):
            metrics.update(self._sample_disk())
        if self._categories.get("network", True):
            metrics.update(self._sample_network())
        if self._categories.get("gpu", True) and self._gpu_available:
            metrics.update(self._sample_gpu())

        for reg_name, value in metrics.items():
            self._log_metric(reg_name, value, step)

    # ------------------------------------------------------------------
    # Samplers (psutil) — one dict per category, graph name -> value
    # ------------------------------------------------------------------

    def _sample_cpu(self) -> Dict[str, float]:
        try:
            return {"resource/cpu/utilization_percent": float(psutil.cpu_percent(interval=None))}
        except Exception:
            return {}

    def _sample_memory(self) -> Dict[str, float]:
        try:
            vm = psutil.virtual_memory()
            return {"resource/memory/system_utilization_percent": float(vm.percent)}
        except Exception:
            return {}

    def _sample_process(self) -> Dict[str, float]:
        try:
            available_mb = float(psutil.virtual_memory().available) / (1024 ** 2)
            with self._process.oneshot():
                cpu_percent = self._process.cpu_percent(interval=None)
                num_threads = self._process.num_threads()
                mem_info = self._process.memory_info()
                mem_percent = self._process.memory_percent()
            return {
                "resource/process/cpu_utilization_percent": float(cpu_percent),
                "resource/process/cpu_threads_in_use": float(num_threads),
                "resource/process/memory_in_use_mb": float(mem_info.rss) / (1024 ** 2),
                "resource/process/memory_in_use_percent": float(mem_percent),
                "resource/process/memory_available_mb": available_mb,
            }
        except Exception:
            return {}

    def _sample_disk(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        try:
            usage = psutil.disk_usage(self._disk_path)
            metrics["resource/disk/utilization_percent"] = float(usage.percent)
            metrics["resource/disk/utilization_gb"] = float(usage.used) / (1024 ** 3)
        except Exception:
            pass
        try:
            io_counters = psutil.disk_io_counters()
            if io_counters is not None:
                metrics["resource/disk/read_mb"] = float(io_counters.read_bytes) / (1024 ** 2)
                metrics["resource/disk/written_mb"] = float(io_counters.write_bytes) / (1024 ** 2)
        except Exception:
            pass
        return metrics

    def _sample_network(self) -> Dict[str, float]:
        try:
            counters = psutil.net_io_counters()
            if counters is None:
                return {}
            return {
                "resource/network/bytes_sent": float(counters.bytes_sent),
                "resource/network/bytes_received": float(counters.bytes_recv),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # GPU / NVML
    # ------------------------------------------------------------------

    def _init_gpu(self) -> None:
        try:
            import pynvml
        except ImportError:
            logger.debug("[ResourceMonitor] pynvml not available; GPU metrics disabled.")
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._gpu_available = bool(self._gpu_handles)
            if self._gpu_available:
                logger.info("[ResourceMonitor] NVML initialized (%d GPU device(s)).", count)
            else:
                pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug("[ResourceMonitor] NVML unavailable, GPU metrics disabled: %s", e)
            self._gpu_available = False
            self._gpu_handles = []

    def _shutdown_gpu(self) -> None:
        if not self._gpu_available:
            return
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass
        finally:
            self._gpu_available = False
            self._gpu_handles = []

    def _sample_gpu(self) -> Dict[str, float]:
        import pynvml

        metrics: Dict[str, float] = {}
        for index, handle in enumerate(self._gpu_handles):
            prefix = f"resource/gpu/{index}"
            try:
                metrics[f"{prefix}/memory_clock_mhz"] = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                )
            except Exception:
                pass
            try:
                metrics[f"{prefix}/sm_clock_mhz"] = float(
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                )
            except Exception:
                pass
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"{prefix}/memory_allocated_bytes"] = float(mem_info.used)
                if mem_info.total:
                    metrics[f"{prefix}/memory_allocated_percent"] = (
                        float(mem_info.used) / float(mem_info.total) * 100.0
                    )
            except Exception:
                pass
            try:
                metrics[f"{prefix}/temperature_celsius"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
        return metrics

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    def _log_metric(self, reg_name: str, value: float, step: int) -> None:
        try:
            import weightslab.src as _src
        except Exception:
            return
        try:
            _src._log_signal(value, None, reg_name, step=step)
        except Exception:
            logger.debug("[ResourceMonitor] Failed to log metric %s", reg_name, exc_info=True)


# ----------------------------------------------------------------------
# Process-wide singleton — started once from grpc_serve()
# ----------------------------------------------------------------------

_singleton_lock = threading.Lock()
_singleton_instance: Optional[ResourceMonitor] = None


def start_resource_monitor_from_config() -> Optional[ResourceMonitor]:
    """Load config (env vars + optional YAML) and start the resource monitor
    singleton if enabled. Safe to call more than once — returns the existing
    instance (or None if disabled) on subsequent calls.
    """
    global _singleton_instance
    with _singleton_lock:
        if _singleton_instance is not None:
            return _singleton_instance

        config = load_resource_monitoring_config()
        if not config["enabled"]:
            logger.info("[ResourceMonitor] Disabled via WEIGHTSLAB_DISABLE_RESOURCE_MONITORING/config — not starting.")
            return None

        monitor = ResourceMonitor(
            interval_seconds=config["interval_seconds"],
            categories=config["categories"],
            disk_path=config["disk_path"],
            step_source=config.get("step_source", DEFAULT_STEP_SOURCE),
        )
        monitor.start()
        _singleton_instance = monitor
        return monitor


def get_resource_monitor() -> Optional[ResourceMonitor]:
    """Return the running resource monitor singleton, if any."""
    return _singleton_instance


def stop_resource_monitor() -> None:
    """Stop and clear the resource monitor singleton, if running."""
    global _singleton_instance
    with _singleton_lock:
        if _singleton_instance is not None:
            _singleton_instance.stop()
            _singleton_instance = None
