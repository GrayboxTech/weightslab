import torch as th
import time
from array import array as _array
from copy import deepcopy

from weightslab.backend.ledgers import get_logger, register_logger, get_checkpoint_manager


def _make_per_sample_buf():
    """Compact storage for per-sample signals: three typed C arrays.

    Uses array.array instead of a list of dicts to reduce memory by ~20-40x:
    - list of dicts:  ~400-600 bytes/entry (Python dict overhead + 6 string keys)
    - compact arrays: 12 bytes/entry (int32 + int32 + float32)

    Fields:
        sample_ids: signed int32 – dataset sample index
        steps:      signed int32 – global training step
        values:     float32       – signal value at that step for that sample
    """
    return {
        "sample_ids": _array('i'),  # int32, 4 bytes each
        "steps":      _array('i'),  # int32, 4 bytes each
        "values":     _array('f'),  # float32, 4 bytes each
    }


class LoggerQueue:
    def __init__(self, register: bool = True) -> None:
        self.graph_names = set()
        self._current_step_buffer = {}
        self._last_step = None
        self._signal_history = {}  # Keep all signals in memory for persistence
        self._signal_history_per_sample = {}  # Keep all signals per sample in memory for persistence
        self._pending_queue = []  # Queue for new signals waiting to be sent to WeightsStudio
        self._buffered_step = None

        # Evaluation mode state
        self._eval_mode_active: bool = False
        self._eval_mode_hash: str = ""
        self._eval_mode_split: str = ""
        self._eval_mode_tags: list[str] = []
        self._eval_accum: dict = {}  # {graph_name: [sum, count]}

        lg = None
        if register:
            try:
                lg = get_logger()
            except Exception:
                lg = None
            register_logger(self) if lg == None else None

        # Init checkpoint manager for experiment hash retrieval (if available)
        self.chkpt_manager = get_checkpoint_manager()

    def __len__(self):
        """Return logger length."""
        len_history = 0
        for k in self._signal_history:
            for exp_hash in self._signal_history[k]:
                l = len(self._signal_history[k][exp_hash])
                len_history = max(len_history, l)
        return len_history

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
            "timestamp": int(time.time()),
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

    # ------------------------------------------------------------------
    # Evaluation mode helpers
    # ------------------------------------------------------------------

    def get_next_evaluation_count(self, base_hash: str) -> int:
        """Return the next unused evaluation index for *base_hash*.

        Scans the current signal history for keys of the form
        ``<base_hash>_<integer>`` and returns max(found) + 1 (or 1 if none).
        """
        prefix = base_hash + "_"
        max_count = 0
        for gname in self._signal_history:
            for hash_key in self._signal_history[gname]:
                if isinstance(hash_key, str) and hash_key.startswith(prefix):
                    suffix = hash_key[len(prefix):]
                    try:
                        count = int(suffix)
                        if count > max_count:
                            max_count = count
                    except ValueError:
                        pass
        return max_count + 1

    def start_evaluation_mode(self, split_name: str, eval_hash: str, evaluation_tags=None) -> None:
        """Redirect subsequent add_scalars() calls into the evaluation buffer.

        While evaluation mode is active, signals are NOT added to the normal
        curve history.  Instead they accumulate in an internal buffer.
        ``stop_evaluation_mode()`` finalises the buffer into a single marker.

        Per-sample history *is* still updated (for Break-By-Slice on eval
        results), using *eval_hash* as the experiment key.

        Args:
            split_name: Human-readable split name (e.g. ``"train_loader"``).
            eval_hash:  Modified experiment hash (e.g. ``"abc123_1"``).
        """
        self._flush_current_step_buffer(add_to_queue=True)
        self._eval_mode_active = True
        self._eval_mode_hash = eval_hash
        self._eval_mode_split = split_name
        self._eval_mode_tags = list(evaluation_tags or [])
        self._eval_accum = {}

    def stop_evaluation_mode(self, model_age: int) -> dict:
        """Finalise evaluation mode and emit averaged markers.

        Computes the mean value for every graph name that was accumulated
        since ``start_evaluation_mode()``, writes each one into the signal
        history under *eval_hash* and into the pending queue, then resets
        evaluation-mode state.

        Args:
            model_age: Current model age (training step) at time of evaluation.

        Returns:
            Dict mapping graph_name → averaged value for all signals seen.
        """
        if not self._eval_mode_active:
            return {}

        self._eval_mode_active = False
        eval_hash = self._eval_mode_hash
        split_name = self._eval_mode_split
        evaluation_tags = list(self._eval_mode_tags)
        results = {}

        for graph_name, (total, count) in self._eval_accum.items():
            if count <= 0:
                continue
            avg = total / count
            results[graph_name] = avg
            self.graph_names.add(graph_name)

            # Store in signal history under eval_hash
            if graph_name not in self._signal_history:
                self._signal_history[graph_name] = {}
            if eval_hash not in self._signal_history[graph_name]:
                self._signal_history[graph_name][eval_hash] = {}
            if model_age not in self._signal_history[graph_name][eval_hash]:
                self._signal_history[graph_name][eval_hash][model_age] = []

            entry = {
                "experiment_name": graph_name,
                "model_age": model_age,
                "metric_name": graph_name,
                "metric_value": avg,
                "experiment_hash": eval_hash,
                "timestamp": int(time.time()),
                "is_evaluation_marker": True,
                "split_name": split_name,
                "evaluation_tags": evaluation_tags,
            }
            self._signal_history[graph_name][eval_hash][model_age].append(entry)
            self._pending_queue.append(entry)

        self._eval_accum = {}
        self._eval_mode_hash = ""
        self._eval_mode_split = ""
        self._eval_mode_tags = []
        return results

    def abort_evaluation_mode(self) -> None:
        """Abort evaluation mode and drop all in-progress evaluation data.

        This is used when an evaluation is canceled or timed out.
        It clears the accumulation buffer and removes any per-sample history
        that may have been written under the in-flight evaluation hash.
        """
        if not self._eval_mode_active:
            return

        eval_hash = self._eval_mode_hash
        self._eval_mode_active = False
        self._eval_accum = {}
        self._eval_mode_hash = ""
        self._eval_mode_split = ""
        self._eval_mode_tags = []

        if not eval_hash:
            return

        self.remove_evaluation_hash(eval_hash)

    def remove_evaluation_hash(self, eval_hash: str) -> None:
        """Remove all history/queue entries tied to a specific evaluation hash."""
        eval_hash = str(eval_hash or "").strip()
        if not eval_hash:
            return

        # Remove any marker/history entries tied to the evaluation hash.
        for graph_name in list(self._signal_history.keys()):
            try:
                self._signal_history[graph_name].pop(eval_hash, None)
            except Exception:
                pass

        # Remove per-sample traces recorded under the same hash.
        for graph_name in list(self._signal_history_per_sample.keys()):
            try:
                self._signal_history_per_sample[graph_name].pop(eval_hash, None)
            except Exception:
                pass

        # Drop queued points that reference this hash.
        self._pending_queue = [
            entry for entry in self._pending_queue
            if str(entry.get("experiment_hash", "")) != eval_hash
        ]

    # Main method for adding signals to the logger - this is called by the WeightsLabCallback and is responsible for updating
    # history and queueing signals for WeightsStudio
    def add_scalars(self, graph_name, signal, global_step, signal_per_sample, aggregate_by_step: bool = True):
        """Add a new signal to history.

        - Training/immediate mode (`aggregate_by_step=False`): append entry directly and queue immediately.
        - Test/per-sample mode (`aggregate_by_step=True`): aggregate values within the step,
          append one averaged entry when step changes, and queue only on step change.
        - Evaluation mode active: accumulate into internal buffer; per-sample history
          still gets written under the eval hash for Break-By-Slice support.
        """
        self.graph_names.add(graph_name)
        self._last_step = global_step

        # ----------------------------------------------------------------
        # Evaluation-mode interception
        # ----------------------------------------------------------------
        if self._eval_mode_active:
            # Collect scalar values to accumulate
            values: list = []
            if aggregate_by_step and signal_per_sample and isinstance(signal_per_sample, dict):
                values = [self._to_float(v) for v in signal_per_sample.values()]
            elif signal and isinstance(signal, dict):
                values = [self._to_float(v) for _, v in signal.items()]

            if values:
                if graph_name not in self._eval_accum:
                    self._eval_accum[graph_name] = [0.0, 0]
                self._eval_accum[graph_name][0] += sum(values)
                self._eval_accum[graph_name][1] += len(values)

            # Still store per-sample signals under eval_hash (for Break-By-Slice)
            if signal_per_sample and isinstance(signal_per_sample, dict):
                eval_hash = self._eval_mode_hash
                if graph_name not in self._signal_history_per_sample:
                    self._signal_history_per_sample[graph_name] = {}
                if eval_hash not in self._signal_history_per_sample[graph_name]:
                    self._signal_history_per_sample[graph_name][eval_hash] = _make_per_sample_buf()
                buf = self._signal_history_per_sample[graph_name][eval_hash]
                step_i = int(global_step)
                for sid, value in signal_per_sample.items():
                    buf["sample_ids"].append(int(sid))
                    buf["steps"].append(step_i)
                    buf["values"].append(self._to_float(value))

            return  # Do NOT add to normal history during evaluation mode
        # ----------------------------------------------------------------

        exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None

        if self._buffered_step is not None and global_step != self._buffered_step:
            self._flush_current_step_buffer(add_to_queue=True)

        if not aggregate_by_step and self._current_step_buffer:
            self._flush_current_step_buffer(add_to_queue=True)

        # Update per-sample signal history with compact array storage
        if signal_per_sample and isinstance(signal_per_sample, dict):
            if graph_name not in self._signal_history_per_sample:
                self._signal_history_per_sample[graph_name] = {}
            if exp_hash not in self._signal_history_per_sample[graph_name]:
                self._signal_history_per_sample[graph_name][exp_hash] = _make_per_sample_buf()

            buf = self._signal_history_per_sample[graph_name][exp_hash]
            step_i = int(global_step)
            for sid, value in signal_per_sample.items():
                buf["sample_ids"].append(int(sid))
                buf["steps"].append(step_i)
                buf["values"].append(self._to_float(value))

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

    # Print methods for debugging/inspection of logger state
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
        for metric_name, exps in self._signal_history_per_sample.items():
            print(f"Metric: {metric_name}")
            for exp_hash, buf in exps.items():
                print(f"  Experiment Hash: {exp_hash}")
                for sid, step, val in zip(buf["sample_ids"], buf["steps"], buf["values"]):
                    print(f"    Sample ID: {sid}, Step: {step}, Value: {val}")
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
        # self._flush_current_step_buffer(add_to_queue=False)  # History should already be up to date since we flush on step change and on add_scalars when not aggregating by step, but we can flush here as well to be safe before retrieving history for checkpoint saving
        return deepcopy(self._signal_history)

    def get_signal_history_per_sample(self):
        """Reconstruct per-sample history as list-of-dicts from compact array storage."""
        result = {}
        for graph_name, exps in self._signal_history_per_sample.items():
            result[graph_name] = {}
            for exp_hash, buf in exps.items():
                entries = []
                for sid, step, val in zip(buf["sample_ids"], buf["steps"], buf["values"]):
                    entries.append({
                        "experiment_name": graph_name,
                        "sample_id": sid,
                        "model_age": step,
                        "metric_name": graph_name,
                        "metric_value": float(val),
                        "experiment_hash": exp_hash,
                    })
                result[graph_name][exp_hash] = entries
        return result

    def query_per_sample(self, graph_name: str, sample_ids=None, exp_hash=None):
        """Efficiently query per-sample history for specific sample IDs.

        Returns a list of (sample_id, step, value) tuples, filtered by sample_ids
        and optionally by experiment hash. Much faster than get_signal_history_per_sample()
        for targeted queries (e.g., "show me only samples with label 8").

        Args:
            graph_name: Signal name (e.g., "loss", "accuracy").
            sample_ids: Collection of sample IDs to filter by. If None, returns all.
            exp_hash: Specific experiment hash to query. If None, queries all hashes.

        Returns:
            List of (sample_id, step, value) tuples.
        """
        if graph_name not in self._signal_history_per_sample:
            return []

        exps = self._signal_history_per_sample[graph_name]
        hashes = [exp_hash] if exp_hash is not None else list(exps.keys())
        sid_set = set(int(s) for s in sample_ids) if sample_ids is not None else None

        results = []
        for h in hashes:
            buf = exps.get(h)
            if buf is None:
                continue
            for sid, step, val in zip(buf["sample_ids"], buf["steps"], buf["values"]):
                if sid_set is None or sid in sid_set:
                    results.append((sid, step, float(val)))

        return results

    def save_snapshot(self) -> dict:
        """Build a serializable snapshot of the logger state."""
        self._flush_current_step_buffer(add_to_queue=False)

        # Compact serialization: store 3 parallel lists instead of list-of-dicts
        per_sample_compact = {}
        for graph_name, exps in self._signal_history_per_sample.items():
            per_sample_compact[graph_name] = {}
            for exp_hash, buf in exps.items():
                per_sample_compact[graph_name][exp_hash] = {
                    "_compact": True,
                    "sample_ids": list(buf["sample_ids"]),
                    "steps":      list(buf["steps"]),
                    "values":     list(buf["values"]),
                }

        return {
            "graph_names": sorted(self.graph_names),
            "signal_history": self.get_signal_history(),
            "signal_history_per_sample": per_sample_compact,
        }

    # ------------------------------------------------------------------
    # Convenience: list all evaluation-marker hashes in history
    # ------------------------------------------------------------------
    def get_evaluation_marker_hashes(self) -> list:
        """Return all experiment hashes that correspond to evaluation markers."""
        hashes = set()
        for gname in self._signal_history:
            for hash_key in self._signal_history[gname]:
                if isinstance(hash_key, str) and "_" in hash_key:
                    # Check that the suffix is a pure integer
                    suffix = hash_key.rsplit("_", 1)[-1]
                    try:
                        int(suffix)
                        hashes.add(hash_key)
                    except ValueError:
                        pass
        return sorted(hashes)

    def get_and_clear_queue(self):
        """Get pending queue and clear it (for incremental updates to WeightsStudio)."""
        queue_copy = list(self._pending_queue)
        self._pending_queue.clear()
        return queue_copy

    def set_point_note(self, metric_name: str, experiment_hash: str, model_age: int, note: str) -> bool:
        """Attach or clear a note for a specific signal point identified by metric/hash/step."""
        metric_name = str(metric_name or "")
        experiment_hash = str(experiment_hash or "")
        if not metric_name or not experiment_hash:
            return False

        normalized_step = int(model_age)
        cleaned_note = str(note or "").strip()
        updated = False

        entries = (
            self._signal_history.get(metric_name, {})
            .get(experiment_hash, {})
            .get(normalized_step, [])
        )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if cleaned_note:
                entry["point_note"] = cleaned_note
            else:
                entry.pop("point_note", None)
            updated = True

        for entry in self._pending_queue:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("metric_name", "")) != metric_name:
                continue
            if str(entry.get("experiment_hash", "")) != experiment_hash:
                continue
            try:
                if int(entry.get("model_age", -1)) != normalized_step:
                    continue
            except Exception:
                continue
            if cleaned_note:
                entry["point_note"] = cleaned_note
            else:
                entry.pop("point_note", None)

        return updated

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
                            signal_entry.setdefault("timestamp", int(time.time()))
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
                signal_entry.setdefault("timestamp", int(time.time()))
                self.graph_names.add(metric_name)
                _append_signal_entry(metric_name, exp_hash, step, signal_entry)

    def load_signal_history_per_sample(self, signals_per_sample):
        """Load per-sample history into compact array storage.

        Handles three formats:
          - New compact:  {graph_name: {exp_hash: {"_compact": True, "sample_ids": [...], "steps": [...], "values": [...]}}}
          - Legacy list:  {graph_name: {exp_hash: [{sample_id, model_age, metric_value, ...}, ...]}}
          - Legacy dict:  {graph_name: {sample_id_as_key: {model_age, metric_value, ...}}}  → stored under None key
        """
        if not signals_per_sample:
            return

        for metric_name, samples_by_exp in signals_per_sample.items():
            self.graph_names.add(metric_name)
            if metric_name not in self._signal_history_per_sample:
                self._signal_history_per_sample[metric_name] = {}

            if not isinstance(samples_by_exp, dict):
                continue

            for exp_hash, entries in samples_by_exp.items():
                # --- New compact format ---
                if isinstance(entries, dict) and entries.get("_compact"):
                    if exp_hash not in self._signal_history_per_sample[metric_name]:
                        self._signal_history_per_sample[metric_name][exp_hash] = _make_per_sample_buf()
                    buf = self._signal_history_per_sample[metric_name][exp_hash]
                    ids   = entries.get("sample_ids", [])
                    steps = entries.get("steps", [])
                    vals  = entries.get("values", [])
                    for s, t, v in zip(ids, steps, vals):
                        try:
                            buf["sample_ids"].append(int(s))
                            buf["steps"].append(int(t))
                            buf["values"].append(float(v))
                        except (TypeError, ValueError):
                            pass

                # --- Legacy list-of-dicts format ---
                elif isinstance(entries, list):
                    if exp_hash not in self._signal_history_per_sample[metric_name]:
                        self._signal_history_per_sample[metric_name][exp_hash] = _make_per_sample_buf()
                    buf = self._signal_history_per_sample[metric_name][exp_hash]
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        try:
                            buf["sample_ids"].append(int(entry.get("sample_id", -1)))
                            buf["steps"].append(int(entry.get("model_age", 0)))
                            buf["values"].append(float(entry.get("metric_value", 0.0)))
                        except (TypeError, ValueError):
                            pass

                # --- Legacy single-dict format (exp_hash key was actually the sample_id) ---
                elif isinstance(entries, dict):
                    null_key = None
                    if null_key not in self._signal_history_per_sample[metric_name]:
                        self._signal_history_per_sample[metric_name][null_key] = _make_per_sample_buf()
                    buf = self._signal_history_per_sample[metric_name][null_key]
                    try:
                        sid = int(exp_hash) if isinstance(exp_hash, (int, float)) else -1
                        buf["sample_ids"].append(sid)
                        buf["steps"].append(int(entries.get("model_age", 0)))
                        buf["values"].append(float(entries.get("metric_value", 0.0)))
                    except (TypeError, ValueError):
                        pass

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
