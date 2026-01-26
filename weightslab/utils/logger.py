import queue

from weightslab.backend.ledgers import get_logger, register_logger, get_checkpoint_manager


class LoggerQueue:
    def __init__(self, name: str = None, register: bool = True) -> None:
        self.queue = queue.Queue()  # Thread-safe queue for signals communication
        self.graph_names = set()
        self._current_step_buffer = {}  # {metric_name: [values]}
        self._last_step = None
        self._signal_history = []  # Keep all signals in memory for persistence

        if register:
            # # Initialize the proxy before setting the logger
            try:
                get_logger(name)
            except Exception:
                pass

            # Register the logger into the ledger. This will update any proxy in-place.
            register_logger(name, self)

        self.chkpt_manager = get_checkpoint_manager()

    def get_graph_names(self):
        return list(self.graph_names)

    def _flush_step_buffer(self):
        """Flush accumulated metrics for the previous step to queue."""
        if self._current_step_buffer and self._last_step is not None:
            for metric_name, values in self._current_step_buffer.items():
                signal = {
                    "experiment_name": metric_name,
                    "model_age": self._last_step,
                    "metric_name": metric_name,
                    "metric_value": sum(values) / len(values) if len(values) > 1 else values[0],
                    "experiment_hash": self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None,
                }
                self.queue.put(signal)
                self._signal_history.append(signal)
            self._current_step_buffer.clear()

    def add_scalars(self, graph_name, name_2_value, global_step: int):
        global_step -= 1  # adjust for 0-based step indexing
        self.graph_names.add(graph_name)

        # If step changed, flush the previous step's buffer
        if global_step != self._last_step:
            self._flush_step_buffer()
            self._last_step = global_step

        # Accumulate metrics for the current step
        for line_name, line_value in name_2_value.items():
            metric_key = f"{graph_name}:{line_name}"
            if metric_key not in self._current_step_buffer:
                self._current_step_buffer[metric_key] = []
            self._current_step_buffer[metric_key].append(float(line_value))

    def print_queue(self):
        """Print all items in queue without removing them."""
        items = list(self.queue.queue)
        for i, item in enumerate(items):
            print(f"[{i}] {item}")
        return items

    def print_buffer(self):
        """Print current step buffer contents."""
        print(f"Current step: {self._last_step}")
        print(f"Buffered metrics: {self._current_step_buffer}")
        return self._current_step_buffer

    def get_signal_history(self):
        """Retrieve all accumulated signals from memory."""
        return list(self._signal_history)

    def load_signal_history(self, signals):
        """Load a list of signals into history and queue (used for checkpoint restore)."""
        if not signals:
            return
        for signal in signals:
            self._signal_history.append(signal)
            try:
                self.queue.put(signal)  # Set the signals to be sent
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
