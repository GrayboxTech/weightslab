import torch as th

from weightslab.backend.ledgers import get_logger, register_logger, get_checkpoint_manager


class LoggerQueue:
    def __init__(self, register: bool = True) -> None:
        self.graph_names = set()
        self._current_step_buffer = {}  # {metric_name: [values]}
        self._last_step = None
        self._signal_history = []  # Keep all signals in memory for persistence
        self._signal_history_per_sample = {}  # Keep all signals per sample in memory for persistence
        self._pending_queue = []  # Queue for new signals waiting to be sent to WeightsStudio

        if register:
            try:
                lg = get_logger()
            except Exception:
                pass
            register_logger(self) if lg == None else None

        # Init checkpoint manager for experiment hash retrieval (if available)
        self.chkpt_manager = get_checkpoint_manager()

    def get_graph_names(self):
        """
            Get list of all graph names encountered in signals.
            Returns:
                List of graph names.
        """
        return list(self.graph_names)

    def _flush_step_buffer(self):
        """Backward-compatible no-op: history is now updated immediately in add_scalars."""
        self._current_step_buffer.clear()

    def add_scalars(self, graph_name, signal, global_step, signal_per_sample):
        """Add a new signal to the logger and push it immediately to history/queue."""
        self.graph_names.add(graph_name)
        self._last_step = global_step
        exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None

        # Update per-sample signal history immediately
        if graph_name not in self._signal_history_per_sample:
            self._signal_history_per_sample[graph_name] = {}
        if exp_hash not in self._signal_history_per_sample[graph_name]:
            self._signal_history_per_sample[graph_name][exp_hash] = []

        if signal_per_sample and isinstance(signal_per_sample, dict):
            for sid, value in signal_per_sample.items():
                self._signal_history_per_sample[graph_name][exp_hash].append(
                    {
                        "experiment_name": graph_name,
                        "sample_id": sid,
                        "model_age": global_step,
                        "metric_name": graph_name,
                        "metric_value": value.item() if isinstance(value, th.Tensor) else value,
                        "experiment_hash": exp_hash
                    }
                )

        # Update averaged signal history immediately
        metric_values = []
        for _, line_value in signal.items():
            metric_values.append(float(line_value.item() if isinstance(line_value, th.Tensor) else line_value))

        if len(metric_values) > 0:
            signal_entry = {
                "experiment_name": graph_name,
                "model_age": global_step,
                "metric_name": graph_name,
                "metric_value": sum(metric_values) / len(metric_values) if len(metric_values) > 1 else metric_values[0],
                "experiment_hash": exp_hash,
            }
            self._signal_history.append(signal_entry)
            self._pending_queue.append(signal_entry)

    def print_history(self):
        """Print all items in history."""
        for i, item in enumerate(self._signal_history):
            print(f"[{i}] {item}")
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

    def get_signal_history(self):
        """Retrieve all accumulated signals from memory."""
        return list(self._signal_history)

    def get_signal_history_per_sample(self):
        """Retrieve all accumulated per-sample signals from memory."""
        return self._signal_history_per_sample

    def get_and_clear_queue(self):
        """Get pending queue and clear it (for incremental updates to WeightsStudio)."""
        queue_copy = list(self._pending_queue)
        self._pending_queue.clear()
        return queue_copy

    def load_signal_history(self, signals):
        """Load a list of signals into history (used for checkpoint restore)."""
        if not signals:
            return
        for signal in signals:
            self._signal_history.append(signal)
            try:
                metric_name = signal.get("metric_name")
                if metric_name:
                    # Derive a graph name if encoded as 'graph:metric'
                    if ":" in metric_name:
                        graph, _ = metric_name.split(":", 1)
                        self.graph_names.add(graph)
            except Exception:
                continue

    def load_signal_history_per_sample(self, signals_per_sample):
        """Load a dict of per-sample signals into history (used for checkpoint restore)."""
        if not signals_per_sample:
            return
        for metric_name, samples in signals_per_sample.items():
            if metric_name not in self._signal_history_per_sample:
                self._signal_history_per_sample[metric_name] = {}
            for sid, signal in samples.items():
                self._signal_history_per_sample[metric_name][sid] = signal

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

    def clear_signal_histories(self):
        """Clear signal histories."""
        # Note: We do not clear graph names here as they are derived from signals and may be needed for future signals after clearing history.
        self._signal_history.clear()
        self._signal_history_per_sample.clear()
