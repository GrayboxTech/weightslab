"""DuckDB-backed signal history logger.

``LoggerQueue`` is a thin interface that maps the logger's public methods onto
a DuckDB database holding three history tables:

* ``signals`` — aggregated training-curve points (one row per averaged
                      step entry / evaluation marker).
* ``per_sample`` — per-sample signal values ``(sample_id, step, value)``.
* ``per_instance`` — per-instance values ``(sample_id, annotation_id, step, value)``
                      for detection / segmentation.

Design notes
------------
* **Hot path is RAM, reads hit DuckDB.** ``add_scalars`` /
  ``add_instance_scalars`` only append to in-memory staging lists (O(1), no SQL).
  Rows are bulk-inserted into DuckDB lazily — right before any query, snapshot,
  delete or update — via a single vectorized ``INSERT ... SELECT``. This keeps
  per-step logging cheap while letting DuckDB do the heavy aggregation
  (``GROUP BY step`` over millions of rows) in native code — exactly what
  break-by-slices needs.
* **Transient runtime state stays in Python.** The live-streaming pending queue,
  the per-step aggregation buffer and the evaluation accumulator are small and
  short-lived, so they remain plain Python structures.
* **Persistence.** ``db_path`` defaults to ``":memory:"``. Pass a file path to
  back the history with an on-disk DuckDB file. Either way ``save_snapshot`` /
  ``load_snapshot`` round-trip the full history as a plain dict, so the
  checkpoint manager's snapshotting is unchanged.
* **Thread-safety.** A single DuckDB connection is guarded by an ``RLock``;
  staging appends and flushes take the same lock.
"""

import functools
import json
import threading
import time
from collections import defaultdict

import duckdb
import pandas as pd
import torch as th

from weightslab.backend.ledgers import get_logger, register_logger, get_checkpoint_manager


# Column order for each table's staging buffer / bulk insert.
_SIGNAL_COLS = [
    "metric_name", "experiment_hash", "step", "metric_value", "timestamp",
    "audit_mode", "is_evaluation_marker", "split_name", "evaluation_tags",
    "point_note", "seq",
]
_SAMPLE_COLS = ["metric_name", "experiment_hash", "sample_id", "step", "value", "seq"]
_INSTANCE_COLS = [
    "metric_name", "experiment_hash", "sample_id", "annotation_id", "step", "value", "seq",
]

# Auto-flush staged rows to DuckDB once the combined staging buffers exceed this
# many rows, to bound memory during long runs that never read history.
_STAGE_FLUSH_THRESHOLD = 50_000


class LoggerQueue:
    def __init__(self, register: bool = True, db_path: str = ":memory:") -> None:
        self.graph_names = set()
        self._current_step_buffer = {}
        self._last_step = None

        # Live-streaming queue of new points waiting to be sent to WeightsStudio.
        self._pending_queue = []
        self._buffered_step = None

        # Evaluation mode state (transient).
        self._eval_mode_active: bool = False
        self._eval_mode_hash: str = ""
        self._eval_mode_split: str = ""
        self._eval_mode_tags: list[str] = []
        self._eval_accum: dict = {} # {graph_name: [sum, count]}

        # DuckDB connection + write-staging buffers.
        self._lock = threading.RLock()
        self._db_path = db_path
        self._conn = duckdb.connect(database=db_path)
        self._stage_signals: list = []
        self._stage_sample: list = []
        self._stage_instance: list = []
        self._seq = 0

        # --- per-sample query cache ---------------------------------------
        # Many consumers in a single step read the SAME (signal, ids): e.g. 10
        # reactive signals all reading the loss, or a batched context re-reading
        # a history. Each read is a lock + _flush_stage + DuckDB scan, so N
        # identical reads cost Nx for one answer. Two memoized readers share one
        # per-signal version map:
        #   _qps_cache      -> query_per_sample        (full history)
        #   _qps_step_cache -> query_per_sample_at_step (values AT one step)
        # both keyed by (signal, ids, [step,] hash, version[signal]). Staging a
        # per-sample row bumps ITS version (see _stage_sample_row), so a cached
        # read is served until that signal changes, then recomputed. Per-signal
        # (not global) versioning is essential: persisting a derived signal must
        # NOT invalidate the loss every dependent is still reading this step.
        #
        # The caches are STEP-SCOPED: because the loader reshuffles ids and the
        # version bumps every step, a cache key never recurs across steps — so
        # cross-step entries are pure dead weight. _stage_sample_row clears both
        # caches when the training step advances, bounding them to one step's
        # worth of entries (the intra-step reuse — the actual win — is kept).
        self._qps_version: dict = defaultdict(int)
        self._qps_cache_step: int = -1
        self._qps_cache = functools.lru_cache(maxsize=2048)(self._query_per_sample_uncached)
        self._qps_step_cache = functools.lru_cache(maxsize=2048)(self._query_per_sample_at_step_uncached)

        self._ensure_tables()
        self._restore_runtime_state_from_db()

        lg = None
        if register:
            try:
                lg = get_logger()
            except Exception:
                lg = None
            register_logger(self) if lg == None else None

        # Init checkpoint manager for experiment hash retrieval (if available)
        self.chkpt_manager = get_checkpoint_manager()

    # ------------------------------------------------------------------
    # DuckDB plumbing
    # ------------------------------------------------------------------
    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    metric_name VARCHAR,
                    experiment_hash VARCHAR,
                    step INTEGER,
                    metric_value DOUBLE,
                    timestamp BIGINT,
                    audit_mode BOOLEAN,
                    is_evaluation_marker BOOLEAN,
                    split_name VARCHAR,
                    evaluation_tags VARCHAR,
                    point_note VARCHAR,
                    seq BIGINT
                );
                CREATE TABLE IF NOT EXISTS per_sample (
                    metric_name VARCHAR,
                    experiment_hash VARCHAR,
                    sample_id VARCHAR,
                    step INTEGER,
                    value REAL,
                    seq BIGINT
                );
                CREATE TABLE IF NOT EXISTS per_instance (
                    metric_name VARCHAR,
                    experiment_hash VARCHAR,
                    sample_id VARCHAR,
                    annotation_id INTEGER,
                    step INTEGER,
                    value REAL,
                    seq BIGINT
                );
                """
            )

    def _restore_runtime_state_from_db(self) -> None:
        """Repopulate seq counter and graph names from an existing (file) DB."""
        with self._lock:
            max_seq = self._conn.execute(
                """
                SELECT max(m) FROM (
                    SELECT max(seq) AS m FROM signals
                    UNION ALL SELECT max(seq) FROM per_sample
                    UNION ALL SELECT max(seq) FROM per_instance
                )
                """
            ).fetchone()[0]
            self._seq = (int(max_seq) + 1) if max_seq is not None else 0

            for tbl in ("signals", "per_sample", "per_instance"):
                for (name,) in self._conn.execute(
                    f"SELECT DISTINCT metric_name FROM {tbl}"
                ).fetchall():
                    if name is not None:
                        self.graph_names.add(name)

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def _maybe_autoflush(self) -> None:
        if (len(self._stage_signals) + len(self._stage_sample)
                + len(self._stage_instance)) >= _STAGE_FLUSH_THRESHOLD:
            self._flush_stage()

    def _flush_stage(self) -> None:
        """Bulk-insert all staged rows into DuckDB and clear the buffers.

        Uses ``register(pandas view) -> INSERT SELECT -> unregister``: this is
        DuckDB's fast vectorized bulk-insert path. (A row-wise ``executemany``
        from the staging tuples was measured ~6x SLOWER — DuckDB binds each row
        individually — so despite the register/unregister showing up in profiles,
        this stays the right approach for bulk.)"""
        with self._lock:
            if self._stage_signals:
                df = pd.DataFrame(self._stage_signals, columns=_SIGNAL_COLS)
                self._conn.register("_stg_sig", df)
                self._conn.execute("INSERT INTO signals SELECT * FROM _stg_sig")
                self._conn.unregister("_stg_sig")
                self._stage_signals = []
            if self._stage_sample:
                df = pd.DataFrame(self._stage_sample, columns=_SAMPLE_COLS)
                self._conn.register("_stg_ps", df)
                self._conn.execute("INSERT INTO per_sample SELECT * FROM _stg_ps")
                self._conn.unregister("_stg_ps")
                self._stage_sample = []
            if self._stage_instance:
                df = pd.DataFrame(self._stage_instance, columns=_INSTANCE_COLS)
                self._conn.register("_stg_pi", df)
                self._conn.execute("INSERT INTO per_instance SELECT * FROM _stg_pi")
                self._conn.unregister("_stg_pi")
                self._stage_instance = []

    def _stage_signal_row(self, graph_name, exp_hash, step, metric_value, timestamp,
                          audit_mode, is_marker, split_name, eval_tags, point_note):
        self._stage_signals.append((
            graph_name, exp_hash, int(step), float(metric_value), int(timestamp),
            bool(audit_mode), bool(is_marker), split_name or "",
            json.dumps(list(eval_tags or [])), point_note or "", self._next_seq(),
        ))
        self._maybe_autoflush()

    def _stage_sample_row(self, graph_name, exp_hash, sample_id, step, value):
        # Step advanced -> last step's cache entries can never be hit again
        # (ids reshuffle + version bump make every key unique per step). Drop
        # them so the cache stays bounded to the current step instead of
        # accumulating single-use entries.
        if int(step) > self._qps_cache_step:
            self._invalidate_qps_cache()
            self._qps_cache_step = int(step)
        self._stage_sample.append((
            graph_name, exp_hash, str(sample_id), int(step), float(value), self._next_seq(),
        ))
        # A new per-sample row for this signal invalidates its cached reads
        # WITHIN the step (an input written after a first read must be re-read).
        self._qps_version[graph_name] += 1
        self._maybe_autoflush()

    def _invalidate_qps_cache(self) -> None:
        """Drop all memoized per-sample query results (both the full-history and
        the at-step readers) and reset versions. Used on step advance and by the
        bulk delete/clear paths, where rows for many signals change at once and
        per-signal version bumps would be impractical."""
        self._qps_cache.cache_clear()
        self._qps_step_cache.cache_clear()
        self._qps_version.clear()

    def _stage_instance_row(self, graph_name, exp_hash, sample_id, annotation_id, step, value):
        self._stage_instance.append((
            graph_name, exp_hash, str(sample_id), int(annotation_id), int(step),
            float(value), self._next_seq(),
        ))
        self._maybe_autoflush()

    @staticmethod
    def _hash_filter(exp_hash, params, table_alias=""):
        """Append an experiment-hash WHERE fragment. ``None`` means 'all hashes'."""
        if exp_hash is None:
            return ""
        params.append(exp_hash)
        col = f"{table_alias}experiment_hash" if table_alias else "experiment_hash"
        return f" AND {col} = ?"

    def __len__(self):
        """Max number of distinct steps recorded for any (metric, hash) curve."""
        with self._lock:
            self._flush_stage()
            row = self._conn.execute(
                """
                SELECT max(cnt) FROM (
                    SELECT count(DISTINCT step) AS cnt
                    FROM signals GROUP BY metric_name, experiment_hash
                )
                """
            ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def clear_signal_histories(self):
        """Clear all signal histories (keeps graph names and runtime buffers reset)."""
        with self._lock:
            self._stage_signals = []
            self._stage_sample = []
            self._stage_instance = []
            self._conn.execute("DELETE FROM signals")
            self._conn.execute("DELETE FROM per_sample")
            self._conn.execute("DELETE FROM per_instance")
            self._current_step_buffer.clear()
            self._buffered_step = None
            self._invalidate_qps_cache()

    def _to_float(self, value):
        if isinstance(value, th.Tensor):
            value = value.item()
        return float(value)

    def _get_audit_mode(self):
        """Get current audit mode from model interface or hyperparams.

        Priority:
        1. Check model_interface.audit_mode (reflects actual model state: eval/train, tracking mode)
        2. Check hyperparams auditor_mode (fallback for legacy/hyperparams-based control)
        """
        try:
            from weightslab.backend.ledgers import get_model
            model = get_model()
            if model is not None and hasattr(model, 'audit_mode'):
                return bool(model.audit_mode)
        except Exception:
            pass

        try:
            from weightslab.backend.ledgers import get_hyperparams
            hp = get_hyperparams()
            if hp is not None:
                return bool(hp.get('auditor_mode', False))
        except Exception:
            pass
        return False

    def _append_history_entry(self, graph_name, exp_hash, global_step, metric_value,
                              audit_mode=None, is_marker=False, split_name="",
                              evaluation_tags=None):
        """Stage a signals row and return the live-queue entry dict."""
        if audit_mode is None:
            audit_mode = self._get_audit_mode()

        timestamp = int(time.time())
        signal_entry = {
            "model_age": global_step,
            "metric_name": graph_name,
            "metric_value": metric_value,
            "experiment_hash": exp_hash,
            "timestamp": timestamp,
            "audit_mode": audit_mode,
        }
        if is_marker:
            signal_entry["is_evaluation_marker"] = True
            signal_entry["split_name"] = split_name
            signal_entry["evaluation_tags"] = list(evaluation_tags or [])

        with self._lock:
            self._stage_signal_row(
                graph_name, exp_hash, global_step, metric_value, timestamp,
                bool(audit_mode), bool(is_marker), split_name,
                list(evaluation_tags or []), "",
            )
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

        Scans recorded experiment hashes for keys of the form
        ``<base_hash>_<integer>`` and returns max(found) + 1 (or 1 if none).
        """
        prefix = base_hash + "_"
        max_count = 0
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(
                "SELECT DISTINCT experiment_hash FROM signals "
                "WHERE experiment_hash LIKE ?",
                [prefix + "%"],
            ).fetchall()
        for (hash_key,) in rows:
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
        curve history. Instead they accumulate in an internal buffer.
        ``stop_evaluation_mode()`` finalises the buffer into a single marker.

        Per-sample history *is* still updated (for Break-By-Slice on eval
        results), using *eval_hash* as the experiment key.
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

        Returns:
            Dict mapping graph_name → averaged value for all signals seen.
        """
        if not self._eval_mode_active:
            return {}

        self._eval_mode_active = False
        eval_hash = self._eval_mode_hash
        split_name = self._eval_mode_split
        evaluation_tags = list(self._eval_mode_tags)
        audit_mode = self._get_audit_mode()
        results = {}

        for graph_name, (total, count) in self._eval_accum.items():
            if count <= 0:
                continue
            avg = total / count
            results[graph_name] = avg
            self.graph_names.add(graph_name)

            entry = self._append_history_entry(
                graph_name=graph_name,
                exp_hash=eval_hash,
                global_step=model_age,
                metric_value=avg,
                audit_mode=audit_mode,
                is_marker=True,
                split_name=split_name,
                evaluation_tags=evaluation_tags,
            )
            self._pending_queue.append(entry)

        self._eval_accum = {}
        self._eval_mode_hash = ""
        self._eval_mode_split = ""
        self._eval_mode_tags = []
        return results

    def abort_evaluation_mode(self) -> None:
        """Abort evaluation mode and drop all in-progress evaluation data."""
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

        with self._lock:
            self._flush_stage()
            self._conn.execute("DELETE FROM signals WHERE experiment_hash = ?", [eval_hash])
            self._conn.execute("DELETE FROM per_sample WHERE experiment_hash = ?", [eval_hash])
            self._invalidate_qps_cache()

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
        with self._lock:
            self.graph_names.add(graph_name)
            self._last_step = global_step

            # ------------------------------------------------------------
            # Evaluation-mode interception
            # ------------------------------------------------------------
            if self._eval_mode_active:
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
                    step_i = int(global_step)
                    for sid, value in signal_per_sample.items():
                        self._stage_sample_row(graph_name, eval_hash, sid, step_i, self._to_float(value))

                return # Do NOT add to normal history during evaluation mode
            # ------------------------------------------------------------

            exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None

            if self._buffered_step is not None and global_step != self._buffered_step:
                self._flush_current_step_buffer(add_to_queue=True)

            if not aggregate_by_step and self._current_step_buffer:
                self._flush_current_step_buffer(add_to_queue=True)

            # Update per-sample signal history
            if isinstance(signal_per_sample, dict) and len(signal_per_sample):
                step_i = int(global_step)
                for sid, value in signal_per_sample.items():
                    self._stage_sample_row(graph_name, exp_hash, sid, step_i, self._to_float(value))

            metric_values = []
            if isinstance(signal_per_sample, dict) and aggregate_by_step and len(signal_per_sample):
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

            # Update averaged signal history immediately. Only emit when we have at
            # least one valid metric value (signals carrying only per-sample data are
            # stored separately in per_sample).
            signal_entry = None
            if len(metric_values) > 0:
                signal_entry = self._append_history_entry(
                    graph_name=graph_name,
                    exp_hash=exp_hash,
                    global_step=global_step,
                    metric_value=sum(metric_values) / len(metric_values) if len(metric_values) > 1 else metric_values[0],
                )

            if signal_entry is not None:
                self._pending_queue.append(signal_entry)

    def ingest_per_sample(self, graph_name: str, exp_hash, triples) -> None:
        """Insert per-sample ``(sample_id, step, value)`` triples, de-duplicating
        on ``(sample_id, step)`` within ``(graph_name, exp_hash)``.

        Unlike ``add_scalars`` (which always appends), this is idempotent on the
        ``(sample_id, step)`` key: the first value wins and later duplicates are
        ignored. Useful for back-filling / importing history without creating
        repeated points.

        Args:
            graph_name: Signal name.
            exp_hash: Experiment hash (``None`` allowed).
            triples: Iterable of ``(sample_id, step, value)``.
        """
        triples = list(triples)
        if not triples:
            return

        with self._lock:
            self.graph_names.add(graph_name)
            self._flush_stage()

            # Existing (sample_id, step) keys for this (graph, hash).
            params = [graph_name]
            sql = "SELECT sample_id, step FROM per_sample WHERE metric_name = ?"
            sql += self._hash_filter(exp_hash, params)
            seen = {(str(s), int(t)) for s, t in self._conn.execute(sql, params).fetchall()}

            for sid, step, value in triples:
                key = (str(sid), int(step))
                if key in seen:
                    continue
                seen.add(key)
                self._stage_sample_row(graph_name, exp_hash, sid, step, self._to_float(value))

    # ------------------------------------------------------------------
    # Print helpers (debug)
    # ------------------------------------------------------------------
    def print_history(self):
        history = self.get_signal_history()
        for metric_name, experiments in history.items():
            print(f"Metric: {metric_name}")
            for exp_hash, steps in experiments.items():
                print(f" Experiment Hash: {exp_hash}")
                for step, signals in steps.items():
                    print(f" Step: {step}")
                    for signal in signals:
                        print(f" Signal: {signal}")
        return history

    def print_history_per_sample(self):
        history = self.get_signal_history_per_sample()
        for metric_name, exps in history.items():
            print(f"Metric: {metric_name}")
            for exp_hash, entries in exps.items():
                print(f" Experiment Hash: {exp_hash}")
                for e in entries:
                    print(f" Sample ID: {e['sample_id']}, Step: {e['model_age']}, Value: {e['metric_value']}")
        return history

    def print_buffer(self):
        print(f"Current step: {self._last_step}")
        print(f"Buffered metrics: {self._current_step_buffer}")
        return self._current_step_buffer

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_graph_names(self):
        """Get list of all graph names encountered in signals."""
        return list(self.graph_names)

    def list_sample_signal_names(self) -> list:
        """Distinct signal names that have per-sample history."""
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute("SELECT DISTINCT metric_name FROM per_sample").fetchall()
        return [r[0] for r in rows]

    def list_instance_signal_names(self) -> list:
        """Distinct signal names that have per-instance history."""
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute("SELECT DISTINCT metric_name FROM per_instance").fetchall()
        return [r[0] for r in rows]

    def get_signal_history(self):
        """Reconstruct aggregated history as ``{metric: {hash: {step: [entry, ...]}}}``."""
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(
                """
                SELECT metric_name, experiment_hash, step, metric_value, timestamp,
                       audit_mode, is_evaluation_marker, split_name, evaluation_tags, point_note
                FROM signals ORDER BY seq
                """
            ).fetchall()

        result: dict = {}
        for (metric, h, step, val, ts, audit, marker, split, tags, note) in rows:
            entry = {
                "model_age": step,
                "metric_name": metric,
                "metric_value": val,
                "experiment_hash": h,
                "timestamp": int(ts) if ts is not None else 0,
                "audit_mode": bool(audit),
                "is_evaluation_marker": bool(marker),
                "split_name": split or "",
                "evaluation_tags": json.loads(tags) if tags else [],
            }
            if note:
                entry["point_note"] = note
            result.setdefault(metric, {}).setdefault(h, {}).setdefault(step, []).append(entry)
        return result

    def get_current_signaL_history(self, graph_name: str, meta: bool = False):
        """Get current-hash aggregated history for a specific signal."""
        if graph_name not in self.graph_names:
            return {}

        exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None

        with self._lock:
            self._flush_stage()
            params = [graph_name]
            sql = "SELECT step, metric_value FROM signals WHERE metric_name = ?"
            sql += self._hash_filter(exp_hash, params)
            sql += " ORDER BY seq"
            rows = self._conn.execute(sql, params).fetchall()

        if meta:
            steps: dict = {}
            for step, val in rows:
                steps.setdefault(step, []).append({
                    "model_age": step, "metric_value": val,
                })
            return steps

        return [{"model_age": step, "metric_value": val} for step, val in rows]

    def get_signal_history_per_sample(self):
        """Per-sample history as ``{metric: {hash: [entry, ...]}}``."""
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(
                "SELECT metric_name, experiment_hash, sample_id, step, value "
                "FROM per_sample ORDER BY seq"
            ).fetchall()

        result: dict = {}
        for (metric, h, sid, step, val) in rows:
            result.setdefault(metric, {}).setdefault(h, []).append({
                "sample_id": sid,
                "model_age": step,
                "metric_name": metric,
                "metric_value": float(val),
                "experiment_hash": h,
            })
        return result

    def get_current_signaL_history_per_sample(self, graph_name: str, sample_ids: list = None, exp_hash: str = None):
        """Get current-hash per-sample history for a specific signal."""
        if graph_name not in self.graph_names:
            return {}

        exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager and exp_hash is None else exp_hash
        return self.query_per_sample(graph_name, sample_ids=sample_ids, exp_hash=exp_hash)

    def query_per_sample(self, graph_name: str, sample_ids=None, exp_hash=None):
        """Query per-sample history.

        Returns a list of ``(sample_id, step, value, experiment_hash)`` tuples,
        filtered by *sample_ids* and optionally *exp_hash* (``None`` = all hashes).

        Served from the per-signal query cache (see ``self._qps_cache``): the
        result is memoized until a new per-sample row for *graph_name* is staged,
        so repeated identical reads within a step cost one scan, not N. A fresh
        ``list`` copy is returned each call so callers may mutate it freely.
        """
        ids_key = tuple(str(s) for s in sample_ids) if sample_ids is not None else None
        cached = self._qps_cache(graph_name, ids_key, exp_hash, self._qps_version[graph_name])
        return list(cached)

    def _query_per_sample_uncached(self, graph_name, ids_key, exp_hash, _version):
        """Actual DuckDB read behind :meth:`query_per_sample`. ``_version`` is a
        cache-key discriminant only (unused in the body) — it changes when the
        signal is written, forcing a recompute. Returns a tuple so the cached
        value stays immutable across callers."""
        with self._lock:
            self._flush_stage()
            params = [graph_name]
            sql = "SELECT sample_id, step, value, experiment_hash FROM per_sample WHERE metric_name = ?"
            sql += self._hash_filter(exp_hash, params)
            if ids_key is not None:
                sql += " AND sample_id IN (SELECT UNNEST(?))"
                params.append(list(ids_key))
            sql += " ORDER BY seq"
            rows = self._conn.execute(sql, params).fetchall()

        return tuple((sid, int(step), float(val), h) for (sid, step, val, h) in rows)

    def query_per_sample_at_step(self, graph_name: str, sample_ids, step, exp_hash=None):
        """Per-sample values of *graph_name* at EXACTLY *step* — returns a list
        of ``(sample_id, value)``. Unlike :meth:`query_per_sample`, the result is
        O(batch), not O(history): the DuckDB ``WHERE step = ?`` filter is applied
        in-engine so only the current step's rows are materialized into Python.
        This keeps the reactive freshness gather flat as training accumulates
        history (it only ever needs the value at the firing step). Served from
        the step cache; a fresh ``list`` copy is returned each call.
        """
        ids_key = tuple(str(s) for s in sample_ids) if sample_ids is not None else None
        cached = self._qps_step_cache(graph_name, ids_key, int(step), exp_hash,
                                      self._qps_version[graph_name])
        return list(cached)

    def _query_per_sample_at_step_uncached(self, graph_name, ids_key, step, exp_hash, _version):
        """DuckDB read behind :meth:`query_per_sample_at_step`. ``_version`` is a
        cache-key discriminant only. Returns an immutable tuple.

        Fast path: the value at *step* was almost always just staged this step
        and is still in the in-memory staging buffer (not yet flushed to DuckDB).
        The reactive gather reads exactly this — the current step's value — so
        scanning the small staging list lets us skip the expensive
        flush -> register(pandas) -> INSERT -> unregister -> SELECT round-trip
        entirely (profiling showed that DuckDB dance is ~a third of per-step
        cost). We only fall through to DuckDB when some requested id is NOT in
        the buffer (a mid-step flush moved it, or it's an older step)."""
        step = int(step)
        with self._lock:
            if ids_key is not None:
                ids_set = set(ids_key)
                at = {}
                # Scan the staging buffer from the end: it's append-ordered by
                # seq (== non-decreasing step), so the target step's rows are
                # near the tail. Grab the latest value per id (first seen going
                # backwards wins) and stop as soon as all ids are found; break
                # out entirely once we pass below `step` (no earlier row can
                # match). Touches ~one batch of rows, not the whole 8k buffer.
                for row in reversed(self._stage_sample):
                    s = row[3]
                    if s < step:
                        break
                    if s == step and row[0] == graph_name \
                            and (exp_hash is None or row[1] == exp_hash):
                        sid = row[2]
                        if sid in ids_set and sid not in at:
                            at[sid] = row[4]
                            if len(at) == len(ids_set):
                                break
                if len(at) == len(ids_set):
                    return tuple((sid, float(val)) for sid, val in at.items())

            # Fallback: not fully in the staging buffer -> flush + query DuckDB.
            self._flush_stage()
            params = [graph_name, step]
            sql = "SELECT sample_id, value FROM per_sample WHERE metric_name = ? AND step = ?"
            sql += self._hash_filter(exp_hash, params)
            if ids_key is not None:
                sql += " AND sample_id IN (SELECT UNNEST(?))"
                params.append(list(ids_key))
            rows = self._conn.execute(sql, params).fetchall()

        return tuple((sid, float(val)) for (sid, val) in rows)

    def query_per_instance(
        self,
        graph_name: str,
        sample_id: str | None = None,
        annotation_id: int | None = None,
        exp_hash: str | None = None,
    ) -> list:
        """Query per-instance signal history.

        Returns a list of ``(sample_id, annotation_id, step, value, exp_hash)``
        tuples. Any of *sample_id*, *annotation_id*, *exp_hash* may be ``None``
        to return all values along that dimension.
        """
        with self._lock:
            self._flush_stage()
            params = [graph_name]
            sql = ("SELECT sample_id, annotation_id, step, value, experiment_hash "
                   "FROM per_instance WHERE metric_name = ?")
            sql += self._hash_filter(exp_hash, params)
            if sample_id is not None:
                sql += " AND sample_id = ?"
                params.append(str(sample_id))
            if annotation_id is not None:
                sql += " AND annotation_id = ?"
                params.append(int(annotation_id))
            sql += " ORDER BY seq"
            rows = self._conn.execute(sql, params).fetchall()

        return [(str(sid), int(aid), int(step), float(val), h)
                for (sid, aid, step, val, h) in rows]

    def aggregate_per_sample_by_step(
        self,
        graph_name: str,
        sample_ids=None,
        exp_hash: str | None = None,
    ) -> dict:
        """Return mean signal value per step, aggregated over matching samples.

        DuckDB performs the ``GROUP BY step`` average natively, which scales to
        millions of rows far better than a Python loop — this is the path used
        by break-by-slices.

        Returns:
            ``{exp_hash: [(step, mean_value), ...]}`` — one step-sorted series
            per hash.
        """
        with self._lock:
            self._flush_stage()
            params = [graph_name]
            sql = ("SELECT experiment_hash, step, avg(value) AS mean_value "
                   "FROM per_sample WHERE metric_name = ?")
            sql += self._hash_filter(exp_hash, params)
            if sample_ids is not None:
                sql += " AND sample_id IN (SELECT UNNEST(?))"
                params.append([str(s) for s in sample_ids])
            sql += " GROUP BY experiment_hash, step ORDER BY experiment_hash, step"
            rows = self._conn.execute(sql, params).fetchall()

        result: dict = {}
        for (h, step, mean_val) in rows:
            result.setdefault(h, []).append((int(step), float(mean_val)))
        return result

    def reduce_per_sample(
        self,
        graph_name: str,
        reduce: str = "min",
        sample_ids=None,
        exp_hash: str | None = None,
    ) -> dict:
        """Reduce each sample's signal HISTORY to a single value.

        Unlike ``aggregate_per_sample_by_step`` (which averages *across samples*
        per step), this groups ``per_sample`` rows BY sample_id and reduces over
        that sample's whole time series — the axis needed for questions like
        "which samples never had train_loss below 0.5" (``reduce='min'`` then
        compare ``>= 0.5``).

        Args:
            graph_name: The registered signal/metric name.
            reduce: One of ``min`` | ``max`` | ``mean``/``avg`` | ``count``.
            sample_ids: Optional iterable to restrict the query.
            exp_hash: ``None`` = all hashes; otherwise restrict to one.

        Returns:
            ``{sample_id (str): reduced_value (float)}``; empty if the metric is
            unknown or has no recorded history.
        """
        agg = {
            "min": "min(value)", "max": "max(value)",
            "mean": "avg(value)", "avg": "avg(value)", "count": "count(value)",
        }.get(str(reduce).lower())
        if agg is None:
            raise ValueError(f"Unsupported reduce '{reduce}'. Use min/max/mean/count.")

        with self._lock:
            self._flush_stage()
            params = [graph_name]
            sql = f"SELECT sample_id, {agg} AS v FROM per_sample WHERE metric_name = ?"
            sql += self._hash_filter(exp_hash, params)
            if sample_ids is not None:
                sql += " AND sample_id IN (SELECT UNNEST(?))"
                params.append([str(s) for s in sample_ids])
            sql += " GROUP BY sample_id"
            rows = self._conn.execute(sql, params).fetchall()

        return {str(sid): float(v) for (sid, v) in rows if v is not None}

    def resolve_graph_name(self, name: str) -> str | None:
        """Best-effort map a user-facing metric name to a stored graph name.

        The logger records signals under their registered name (e.g. ``train_loss``
        or ``train_mlt_loss/CE``), which rarely matches the dataframe's column
        spelling (``signals//train_loss/sample``). Resolve by exact match, then
        case-insensitive, then unambiguous substring either way; returns ``None``
        if nothing matches so callers can degrade gracefully.
        """
        if not name:
            return None
        if name in self.graph_names:
            return name
        low = str(name).lower()
        for g in self.graph_names:
            if g.lower() == low:
                return g
        candidates = [g for g in self.graph_names if low in g.lower() or g.lower() in low]
        if candidates:
            # Prefer the shortest (closest) match for determinism.
            return sorted(candidates, key=len)[0]
        return None

    def add_instance_scalars(
        self,
        graph_name: str,
        sample_ids,
        annotation_ids,
        values,
        global_step: int,
        exp_hash: str | None = None,
    ) -> None:
        """Record per-instance scalar values.

        Each element of *sample_ids*, *annotation_ids*, *values* corresponds to
        one detection / segmentation instance.

        Args:
            graph_name: Signal name (e.g. ``"confidence"``).
            sample_ids: Sequence of sample IDs, one per instance.
            annotation_ids: Sequence of annotation IDs (1-based), one per instance.
            values: Scalar values, one per instance (array-like or list).
            global_step: Current training step.
            exp_hash: Experiment hash. Resolved from the checkpoint manager if ``None``.
        """
        if exp_hash is None:
            exp_hash = (
                self.chkpt_manager.get_current_experiment_hash()
                if self.chkpt_manager
                else None
            )

        try:
            import numpy as _np
            vals = _np.asarray(values, dtype=_np.float32).ravel()
        except Exception:
            vals = [float(v) for v in values]

        with self._lock:
            step_i = int(global_step)
            for sid, aid, val in zip(sample_ids, annotation_ids, vals):
                self._stage_instance_row(graph_name, exp_hash, sid, aid, step_i, float(val))

    def get_signal_history_per_instance(self) -> dict:
        """Per-instance history as ``{metric: {hash: [entry, ...]}}``."""
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(
                "SELECT metric_name, experiment_hash, sample_id, annotation_id, step, value "
                "FROM per_instance ORDER BY seq"
            ).fetchall()

        result: dict = {}
        for (metric, h, sid, aid, step, val) in rows:
            result.setdefault(metric, {}).setdefault(h, []).append({
                "sample_id": str(sid),
                "annotation_id": int(aid),
                "model_age": int(step),
                "metric_name": metric,
                "metric_value": float(val),
                "experiment_hash": h,
            })
        return result

    def save_snapshot(self) -> dict:
        """Build a serializable snapshot of the logger state (compact format)."""
        self._flush_current_step_buffer(add_to_queue=False)

        per_sample_compact: dict = {}
        for graph_name, exps in self.get_signal_history_per_sample().items():
            per_sample_compact[graph_name] = {}
            for exp_hash, entries in exps.items():
                per_sample_compact[graph_name][exp_hash] = {
                    "_compact": True,
                    "sample_ids": [e["sample_id"] for e in entries],
                    "steps": [e["model_age"] for e in entries],
                    "values": [e["metric_value"] for e in entries],
                }

        per_instance_compact: dict = {}
        for graph_name, exps in self.get_signal_history_per_instance().items():
            per_instance_compact[graph_name] = {}
            for exp_hash, entries in exps.items():
                per_instance_compact[graph_name][exp_hash] = {
                    "_compact": True,
                    "sample_ids": [e["sample_id"] for e in entries],
                    "annotation_ids": [e["annotation_id"] for e in entries],
                    "steps": [e["model_age"] for e in entries],
                    "values": [e["metric_value"] for e in entries],
                }

        return {
            "graph_names": sorted(self.graph_names),
            "signal_history": self.get_signal_history(),
            "signal_history_per_sample": per_sample_compact,
            "signal_history_per_instance": per_instance_compact,
        }

    def get_evaluation_marker_hashes(self) -> list:
        """Return all experiment hashes of the form ``<base>_<int>`` in history."""
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(
                "SELECT DISTINCT experiment_hash FROM signals WHERE experiment_hash IS NOT NULL"
            ).fetchall()

        hashes = set()
        for (hash_key,) in rows:
            if isinstance(hash_key, str) and "_" in hash_key:
                suffix = hash_key.rsplit("_", 1)[-1]
                try:
                    int(suffix)
                    hashes.add(hash_key)
                except ValueError:
                    pass
        return sorted(hashes)

    def get_and_clear_queue(self):
        """Get pending queue and clear it (for incremental updates to WeightsStudio)."""
        with self._lock:
            queue_copy = list(self._pending_queue)
            self._pending_queue.clear()
        return queue_copy

    def set_point_note(self, metric_name: str, experiment_hash: str, model_age: int, note: str) -> bool:
        """Attach or clear a note for a signal point identified by metric/hash/step."""
        metric_name = str(metric_name or "")
        experiment_hash = str(experiment_hash or "")
        if not metric_name or not experiment_hash:
            return False

        normalized_step = int(model_age)
        cleaned_note = str(note or "").strip()

        with self._lock:
            self._flush_stage()
            matched = self._conn.execute(
                "SELECT count(*) FROM signals "
                "WHERE metric_name = ? AND experiment_hash = ? AND step = ?",
                [metric_name, experiment_hash, normalized_step],
            ).fetchone()[0]
            if matched:
                self._conn.execute(
                    "UPDATE signals SET point_note = ? "
                    "WHERE metric_name = ? AND experiment_hash = ? AND step = ?",
                    [cleaned_note, metric_name, experiment_hash, normalized_step],
                )

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

        return bool(matched)

    # ------------------------------------------------------------------
    # Snapshot loading (checkpoint persistence)
    # ------------------------------------------------------------------
    def load_signal_history(self, signals):
        """Load aggregated signal history (supports legacy list and nested dict)."""
        if not signals:
            return

        def _stage_entry(metric_name, exp_hash, step, entry):
            try:
                step_i = int(step)
            except (TypeError, ValueError):
                return
            with self._lock:
                self._stage_signal_row(
                    metric_name, exp_hash, step_i,
                    float(entry.get("metric_value", 0.0)),
                    int(entry.get("timestamp", int(time.time()))),
                    bool(entry.get("audit_mode", False)),
                    bool(entry.get("is_evaluation_marker", False)),
                    entry.get("split_name", ""),
                    entry.get("evaluation_tags", []),
                    entry.get("point_note", "") or "",
                )

        if isinstance(signals, dict):
            for metric_name, experiments in signals.items():
                self.graph_names.add(metric_name)
                if not isinstance(experiments, dict):
                    continue
                for exp_hash, steps in experiments.items():
                    if not isinstance(steps, dict):
                        continue
                    for step_key, entries in steps.items():
                        entries_list = entries if isinstance(entries, list) else [entries]
                        for entry in entries_list:
                            if isinstance(entry, dict):
                                _stage_entry(metric_name, exp_hash, step_key, entry)
            return

        if isinstance(signals, list):
            for signal in signals:
                if not isinstance(signal, dict):
                    continue
                metric_name = signal.get("metric_name")
                if not metric_name:
                    continue
                self.graph_names.add(metric_name)
                _stage_entry(
                    metric_name,
                    signal.get("experiment_hash"),
                    signal.get("model_age", 0),
                    signal,
                )

    def load_signal_history_per_sample(self, signals_per_sample):
        """Load per-sample history.

        Handles three formats:
          - Compact: {graph: {hash: {"_compact": True, "sample_ids": [...], "steps": [...], "values": [...]}}}
          - Legacy list: {graph: {hash: [{sample_id, model_age, metric_value, ...}, ...]}}
          - Legacy dict: {graph: {sample_id_as_key: {model_age, metric_value, ...}}} → stored under None hash
        """
        if not signals_per_sample:
            return

        for metric_name, samples_by_exp in signals_per_sample.items():
            self.graph_names.add(metric_name)
            if not isinstance(samples_by_exp, dict):
                continue

            for exp_hash, entries in samples_by_exp.items():
                # --- Compact format ---
                if isinstance(entries, dict) and entries.get("_compact"):
                    ids = entries.get("sample_ids", [])
                    steps = entries.get("steps", [])
                    vals = entries.get("values", [])
                    with self._lock:
                        for s, t, v in zip(ids, steps, vals):
                            try:
                                self._stage_sample_row(metric_name, exp_hash, s, int(t), float(v))
                            except (TypeError, ValueError):
                                pass

                # --- Legacy list-of-dicts ---
                elif isinstance(entries, list):
                    with self._lock:
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            try:
                                self._stage_sample_row(
                                    metric_name, exp_hash,
                                    entry.get("sample_id", -1),
                                    int(entry.get("model_age", 0)),
                                    float(entry.get("metric_value", 0.0)),
                                )
                            except (TypeError, ValueError):
                                pass

                # --- Legacy single-dict (exp_hash key was actually the sample_id) ---
                elif isinstance(entries, dict):
                    sid = str(exp_hash) if isinstance(exp_hash, (int, float)) else str(-1)
                    with self._lock:
                        try:
                            self._stage_sample_row(
                                metric_name, None, sid,
                                int(entries.get("model_age", 0)),
                                float(entries.get("metric_value", 0.0)),
                            )
                        except (TypeError, ValueError):
                            pass

    def load_signal_history_per_instance(self, signals_per_instance: dict) -> None:
        """Load per-instance history from a compact snapshot dict."""
        if not signals_per_instance:
            return
        for metric_name, exps in signals_per_instance.items():
            self.graph_names.add(metric_name)
            if not isinstance(exps, dict):
                continue
            for exp_hash, entries in exps.items():
                if not (isinstance(entries, dict) and entries.get("_compact")):
                    continue
                ids = entries.get("sample_ids", [])
                aids = entries.get("annotation_ids", [])
                steps = entries.get("steps", [])
                vals = entries.get("values", [])
                with self._lock:
                    for s, a, t, v in zip(ids, aids, steps, vals):
                        try:
                            self._stage_instance_row(metric_name, exp_hash, s, int(a), int(t), float(v))
                        except (TypeError, ValueError):
                            pass

    def load_snapshot(self, snapshot: dict):
        """Restore logger state from a snapshot dict."""
        if not snapshot:
            return

        self.graph_names.update(snapshot.get("graph_names", []))
        self.load_signal_history(snapshot.get("signal_history", []))
        self.load_signal_history_per_sample(snapshot.get("signal_history_per_sample", {}))
        self.load_signal_history_per_instance(snapshot.get("signal_history_per_instance", {}))
