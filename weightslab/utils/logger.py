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
        """Flush accumulated metrics for the previous step to history."""
        if self._current_step_buffer and self._last_step is not None:
            for metric_name, values in self._current_step_buffer.items():
                # Per sample signal
                if metric_name not in self._signal_history_per_sample:
                    self._signal_history_per_sample[metric_name] = {}
                for sid, value in values[0].items():
                    self._signal_history_per_sample[metric_name][sid] = {
                        "experiment_name": metric_name,
                        "model_age": self._last_step,
                        "metric_name": metric_name,
                        "metric_value": value.item() if isinstance(value, th.Tensor) else value,
                        "experiment_hash": 'Overview only'  # For now, we don't track per-sample signals by experiment hash in the checkpoint manager
                    }

                # Average signal for step 
                self._signal_history.append(
                    {
                        "experiment_name": metric_name,
                        "model_age": self._last_step,
                        "metric_name": metric_name,
                        "metric_value": sum(values[1]) / len(values[1]) if len(values[1]) > 1 else values[1][0],
                        "experiment_hash": self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None,
                    }
                )
                self._pending_queue.append(
                    self._signal_history[-1]
                )  # Add to pending queue for WeightsStudio gRPC updates
            self._current_step_buffer.clear()

    def add_scalars(self, graph_name, signal, global_step, signal_per_sample):
        """Add a new signal to the logger, buffering it by step and graph name."""
        self.graph_names.add(graph_name)

        # If step changed, flush the previous step's buffer
        if global_step != self._last_step:
            self._flush_step_buffer()
            self._last_step = global_step-1  # adjust for 0-based step indexing
        
        # Buffer the new signal for the current step
        if graph_name not in self._current_step_buffer:
            self._current_step_buffer[graph_name] = [
                signal_per_sample,
                []
            ]
        for _, line_value in signal.items():
            self._current_step_buffer[graph_name][1].append(
                float(line_value)
            )
        global_step -= 1  # adjust for 0-based step indexing

    def print_history(self):
        """Print all items in history."""
        for i, item in enumerate(self._signal_history):
            print(f"[{i}] {item}")
        return self._signal_history

    def print_buffer(self):
        """Print current step buffer contents."""
        print(f"Current step: {self._last_step}")
        print(f"Buffered metrics: {self._current_step_buffer}")
        return self._current_step_buffer

    def get_signal_history(self):
        """Retrieve all accumulated signals from memory."""
        return list(self._signal_history)

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

    def load_snapshot(self, snapshot: dict):
        """Restore logger state from a snapshot dict."""
        if not snapshot:
            return
        graph_names = snapshot.get("graph_names", [])
        self.graph_names.update(graph_names)
        signals = snapshot.get("signal_history", [])
        self.load_signal_history(signals)

    def clear_signal_history(self):
        """Clear signal history."""
        self._signal_history.clear()
