import torch as th
from copy import deepcopy

from weightslab.backend.ledgers import get_logger, register_logger, get_checkpoint_manager


class LoggerQueue:
    def __init__(self, register: bool = True) -> None:
        self.graph_names = set()
        self._current_step_buffer = {}
        self._last_step = None
        self._signal_history = {}  # Keep all signals in memory for persistence
        self._signal_history_per_sample = {}  # Keep all signals per sample in memory for persistence
        self._pending_queue = []  # Queue for new signals waiting to be sent to WeightsStudio
        self._buffered_step = None

        lg = None
        if register:
            try:
                lg = get_logger()
            except Exception:
                lg = None
            register_logger(self) if lg == None else None

        # Init checkpoint manager for experiment hash retrieval (if available)
        self.chkpt_manager = get_checkpoint_manager()

    # Clear history method (can be called by WeightsLabCallback at the start of a new experiment to reset state,
    # while preserving graph names which are derived from signals and may be needed for future signals after clearing history)
    def clear_signal_histories(self):
        """Clear signal histories."""
        # Note: We do not clear graph names here as they are derived from signals and may be needed for future signals after clearing history.
        self._signal_history.clear()
        self._signal_history_per_sample.clear()
        self._current_step_buffer.clear()
        self._buffered_step = None

    def _to_float(self, value):
        if isinstance(value, th.Tensor):
            value = value.item()
        return float(value)

    def _append_history_entry(self, graph_name, exp_hash, global_step, metric_value):
        signal_entry = {
            "experiment_name": graph_name,
            "model_age": global_step,
            "metric_name": graph_name,
            "metric_value": metric_value,
            "experiment_hash": exp_hash,
        }

        if graph_name not in self._signal_history:
            self._signal_history[graph_name] = {}
        if exp_hash not in self._signal_history[graph_name]:
            self._signal_history[graph_name][exp_hash] = {}
        if global_step not in self._signal_history[graph_name][exp_hash]:
            self._signal_history[graph_name][exp_hash][global_step] = []

        self._signal_history[graph_name][exp_hash][global_step].append(signal_entry)
        return signal_entry

    def _flush_current_step_buffer(self, add_to_queue: bool):
        if self._buffered_step is None or not self._current_step_buffer:
            return

        for (_, graph_name, exp_hash), payload in self._current_step_buffer.items():
            count = payload.get("count", 0)
            if count <= 0:
                continue
            metric_value = payload["sum"] / count
            signal_entry = self._append_history_entry(
                graph_name=graph_name,
                exp_hash=exp_hash,
                global_step=self._buffered_step,
                metric_value=metric_value,
            )
            if add_to_queue:
                self._pending_queue.append(signal_entry)

        self._current_step_buffer.clear()
        self._buffered_step = None

    # Main method for adding signals to the logger - this is called by the WeightsLabCallback and is responsible for updating
    # history and queueing signals for WeightsStudio
    def add_scalars(self, graph_name, signal, global_step, signal_per_sample, aggregate_by_step: bool = True):
        """Add a new signal to history.

        - Training/immediate mode (`aggregate_by_step=False`): append entry directly and queue immediately.
        - Test/per-sample mode (`aggregate_by_step=True`): aggregate values within the step,
          append one averaged entry when step changes, and queue only on step change.
        """
        self.graph_names.add(graph_name)
        self._last_step = global_step
        exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None

        if self._buffered_step is not None and global_step != self._buffered_step:
            self._flush_current_step_buffer(add_to_queue=True)

        if not aggregate_by_step and self._current_step_buffer:
            self._flush_current_step_buffer(add_to_queue=True)

        # Update per-sample signal history immediately
        if graph_name not in self._signal_history_per_sample:
            self._signal_history_per_sample[graph_name] = {}
        if exp_hash not in self._signal_history_per_sample[graph_name]:
            self._signal_history_per_sample[graph_name][exp_hash] = []

        # Save per-sample signals if provided (expected to be a dict of {sample_id: value})
        if signal_per_sample and isinstance(signal_per_sample, dict):
            for sid, value in signal_per_sample.items():
                self._signal_history_per_sample[graph_name][exp_hash].append(
                    {
                        "experiment_name": graph_name,
                        "sample_id": sid,
                        "model_age": global_step,
                        "metric_name": graph_name,
                        "metric_value": self._to_float(value),
                        "experiment_hash": exp_hash
                    }
                )

        metric_values = []
        if aggregate_by_step and signal_per_sample and isinstance(signal_per_sample, dict):
            for value in signal_per_sample.values():
                metric_values.append(self._to_float(value))
        else:
            for _, line_value in signal.items():
                metric_values.append(self._to_float(line_value))

        if aggregate_by_step:
            if metric_values:
                self._buffered_step = global_step
                buffer_key = (global_step, graph_name, exp_hash)
                if buffer_key not in self._current_step_buffer:
                    self._current_step_buffer[buffer_key] = {"sum": 0.0, "count": 0}
                self._current_step_buffer[buffer_key]["sum"] += sum(metric_values)
                self._current_step_buffer[buffer_key]["count"] += len(metric_values)
            return

        # Update averaged signal history immediately
        signal_entry = None

        # Only add to history if we have at least one valid metric value (otherwise we may end up with empty/invalid entries from signals that only contain per-sample values, which are stored separately in _signal_history_per_sample)
        if len(metric_values) > 0:
             signal_entry = self._append_history_entry(
                graph_name=graph_name,
                exp_hash=exp_hash,
                global_step=global_step,
                metric_value=sum(metric_values) / len(metric_values) if len(metric_values) > 1 else metric_values[0],
            )

        # Add signal to pending queue for live incremental update to WeightsStudio
        if signal_entry is not None:
            self._pending_queue.append(signal_entry)

    # Print methods for debugging/inspection of logger state (also return the printed data for potential programmatic use) -
    # these can be removed or replaced with proper accessors as needed
    def print_history(self):
        """Print all items in history."""
        for metric_name, experiments in self._signal_history.items():
            print(f"Metric: {metric_name}")
            for exp_hash, steps in experiments.items():
                print(f"  Experiment Hash: {exp_hash}")
                for step, signals in steps.items():
                    print(f"    Step: {step}")
                    for signal in signals:
                        print(f"      Signal: {signal}")
        return self._signal_history

    def print_history_per_sample(self):
        """Print all items in per-sample history."""
        for metric_name, samples in self._signal_history_per_sample.items():
            print(f"Metric: {metric_name}")
            for exp_hash, signals in samples.items():
                print(f"  Experiment Hash: {exp_hash}")
                for signal in signals:
                    print(f"    Sample ID: {signal['sample_id']}, Signal: {signal}")
        return self._signal_history_per_sample

    def print_buffer(self):
        """Print current step buffer contents."""
        print(f"Current step: {self._last_step}")
        print(f"Buffered metrics: {self._current_step_buffer}")
        return self._current_step_buffer

    # Accessor methods for retrieving logger state (e.g. for checkpoint saving or programmatic access)
    def get_graph_names(self):
        """
            Get list of all graph names encountered in signals.
            Returns:
                List of graph names.
        """
        return list(self.graph_names)

    def get_signal_history(self):
        """Retrieve all accumulated signals from memory."""
        self._flush_current_step_buffer(add_to_queue=False)
        return deepcopy(self._signal_history)

    def get_signal_history_per_sample(self):
        """Retrieve all accumulated per-sample signals from memory."""
        return deepcopy(self._signal_history_per_sample)

    def save_snapshot(self) -> dict:
        """Build a serializable snapshot of the logger state."""
        self._flush_current_step_buffer(add_to_queue=False)
        return {
            "graph_names": sorted(self.graph_names),
            "signal_history": self.get_signal_history(),
            "signal_history_per_sample": self.get_signal_history_per_sample(),
        }

    def get_and_clear_queue(self):
        """Get pending queue and clear it (for incremental updates to WeightsStudio)."""
        queue_copy = list(self._pending_queue)
        self._pending_queue.clear()
        return queue_copy

    # Logger saving/loading methods for checkpoint persistence (used in WeightsLabCallback)
    def load_signal_history(self, signals):
        """Load signal history into memory (supports legacy and nested formats)."""
        if not signals:
            return

        def _append_signal_entry(metric_name, exp_hash, step, signal_entry):
            if metric_name not in self._signal_history:
                self._signal_history[metric_name] = {}
            if exp_hash not in self._signal_history[metric_name]:
                self._signal_history[metric_name][exp_hash] = {}
            if step not in self._signal_history[metric_name][exp_hash]:
                self._signal_history[metric_name][exp_hash][step] = []
            self._signal_history[metric_name][exp_hash][step].append(signal_entry)

        if isinstance(signals, dict):
            for metric_name, experiments in signals.items():
                self.graph_names.add(metric_name)
                if not isinstance(experiments, dict):
                    continue
                for exp_hash, steps in experiments.items():
                    if not isinstance(steps, dict):
                        continue
                    for step_key, entries in steps.items():
                        step = step_key
                        if isinstance(step_key, str):
                            try:
                                step = int(step_key)
                            except Exception:
                                step = step_key

                        entries_list = entries if isinstance(entries, list) else [entries]
                        for entry in entries_list:
                            if not isinstance(entry, dict):
                                continue
                            signal_entry = dict(entry)
                            signal_entry.setdefault("metric_name", metric_name)
                            signal_entry.setdefault("experiment_name", metric_name)
                            signal_entry.setdefault("model_age", step)
                            signal_entry.setdefault("experiment_hash", exp_hash)
                            _append_signal_entry(metric_name, exp_hash, step, signal_entry)
            return

        if isinstance(signals, list):
            for signal in signals:
                if not isinstance(signal, dict):
                    continue
                metric_name = signal.get("metric_name") or signal.get("experiment_name")
                if not metric_name:
                    continue
                exp_hash = signal.get("experiment_hash")
                step = signal.get("model_age")
                signal_entry = dict(signal)
                signal_entry.setdefault("metric_name", metric_name)
                signal_entry.setdefault("experiment_name", metric_name)
                signal_entry.setdefault("model_age", step)
                signal_entry.setdefault("experiment_hash", exp_hash)
                self.graph_names.add(metric_name)
                _append_signal_entry(metric_name, exp_hash, step, signal_entry)

    def load_signal_history_per_sample(self, signals_per_sample):
        """Load per-sample history (supports legacy and nested formats)."""
        if not signals_per_sample:
            return

        for metric_name, samples_by_exp in signals_per_sample.items():
            self.graph_names.add(metric_name)
            if metric_name not in self._signal_history_per_sample:
                self._signal_history_per_sample[metric_name] = {}

            if not isinstance(samples_by_exp, dict):
                continue

            for exp_hash, entries in samples_by_exp.items():
                if isinstance(entries, list):
                    if exp_hash not in self._signal_history_per_sample[metric_name]:
                        self._signal_history_per_sample[metric_name][exp_hash] = []
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        normalized = dict(entry)
                        normalized.setdefault("metric_name", metric_name)
                        normalized.setdefault("experiment_name", metric_name)
                        normalized.setdefault("experiment_hash", exp_hash)
                        self._signal_history_per_sample[metric_name][exp_hash].append(normalized)
                elif isinstance(entries, dict):
                    if None not in self._signal_history_per_sample[metric_name]:
                        self._signal_history_per_sample[metric_name][None] = []
                    normalized = dict(entries)
                    normalized.setdefault("metric_name", metric_name)
                    normalized.setdefault("experiment_name", metric_name)
                    normalized.setdefault("sample_id", exp_hash)
                    normalized.setdefault("experiment_hash", None)
                    self._signal_history_per_sample[metric_name][None].append(normalized)

    def load_snapshot(self, snapshot: dict):
        """Restore logger state from a snapshot dict."""
        if not snapshot:
            return

        # Load graph names if available in snapshot (added in later versions)
        graph_names = snapshot.get("graph_names", [])
        self.graph_names.update(graph_names)

        # Load signal history if available in snapshot (added in later versions)
        signals = snapshot.get("signal_history", [])
        self.load_signal_history(signals)

        # Load per-sample signals if available in snapshot (added in later versions)
        signals_per_sample = snapshot.get("signal_history_per_sample", {})
        self.load_signal_history_per_sample(signals_per_sample)
