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
import itertools
import json
import logging
import os
import threading
import time
from collections import defaultdict

import duckdb
import pandas as pd
import torch as th

from weightslab.backend.ledgers import get_logger, register_logger, get_checkpoint_manager

logger = logging.getLogger(__name__)


# Column order for each table's staging buffer / bulk insert.
_SIGNAL_COLS = [
    "metric_name", "experiment_hash", "step", "metric_value", "timestamp",
    "audit_mode", "is_evaluation_marker", "split_name", "evaluation_tags",
    "point_note", "outliers", "outlier_count", "sample_count",
    "trend_value", "trend_margin", "value_min", "value_max", "seq",
]
_SAMPLE_COLS = ["metric_name", "experiment_hash", "sample_id", "step", "value", "seq"]
_INSTANCE_COLS = [
    "metric_name", "experiment_hash", "sample_id", "annotation_id", "step", "value", "seq",
]

# Auto-flush staged rows to DuckDB once the combined staging buffers exceed this
# many rows, to bound memory during long runs that never read history.
_STAGE_FLUSH_THRESHOLD = 50_000

# How often the background flush thread wakes up (see LoggerQueue._flush_loop).
def _default_flush_interval_seconds() -> float:
    try:
        return float(os.environ.get("WL_LOGGER_FLUSH_INTERVAL_SECONDS", "2.0"))
    except (TypeError, ValueError):
        return 2.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _outliers_enabled() -> bool:
    return os.environ.get("WL_SIGNAL_OUTLIER_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# Hard cap on how many (sample_id, value) pairs one buffered step retains for
# outlier detection. Batches are small, but add_scalars can be called many times
# per step; this bounds memory on pathological loggers.
_MAX_BUFFERED_SAMPLES_PER_STEP = 4096

# --- Signal-history read caps ---------------------------------------------
# A plot is at most a few thousand pixels wide, so a curve drawn from more
# points than this is decimated away in the browser anyway -- after having been
# paid for in query time, wire bytes and JS heap. Reducing to this many points
# *inside DuckDB* is what keeps a 100M-row table readable at all: the aggregate
# streams, and only the reduced rows are ever turned into Python dicts.
_DEFAULT_MAX_POINTS_PER_CURVE = _env_int("WL_SIGNAL_MAX_POINTS_PER_CURVE", 1000)
# Never reduce a curve below first + last + one interior point, so a
# downsampled curve still reads as a curve rather than a straight segment.
_MIN_POINTS_PER_CURVE = 3
# Rows that must survive downsampling regardless of which bucket they land in
# (evaluation markers, annotated points, steps carrying outliers) are fetched by
# a separate filtered scan. That scan is bounded too -- a pathological run where
# every step carries an outlier would otherwise reintroduce the full-table read
# this whole path exists to avoid. Truncation is logged, never silent.
_MAX_SPECIAL_ROWS = _env_int("WL_SIGNAL_MAX_SPECIAL_ROWS", 20000)
# Chunk size for the uncapped (export/snapshot) path, which streams instead of
# calling fetchall() on the whole table.
_HISTORY_STREAM_CHUNK = _env_int("WL_SIGNAL_STREAM_CHUNK", 500_000)

# Column list shared by every signal-history read, in the order the row
# unpackers below expect.
_SIGNAL_READ_COLS = (
    "metric_name", "experiment_hash", "step", "metric_value", "timestamp",
    "audit_mode", "is_evaluation_marker", "split_name", "evaluation_tags",
    "point_note", "outliers", "outlier_count", "sample_count",
    "trend_value", "trend_margin", "value_min", "value_max",
)


class _TrendTracker:
    """Rolling trend of one signal's averaged curve, for outlier detection.

    Keeps an EMA of the per-step average plus an EMA of squared deviation (a
    rolling variance). A sample is "off-trend" when it sits further from the EMA
    than ``k`` rolling standard deviations.

    Two guards keep this from firing constantly:

    * ``min_steps`` — no flagging until the curve has enough history, so the
      steep warm-up of a fresh loss curve isn't one long outlier run.
    * ``rel_margin`` — the band never narrows below a fraction of |EMA|. On an
      almost-flat curve the rolling std collapses toward zero, and without this
      floor ordinary jitter would clear a 3-sigma test.

    Deviation is measured two-sided (by magnitude) so this works for signals
    where "bad" means low, e.g. accuracy, as well as loss-shaped ones.
    """

    __slots__ = ("ema", "ema_var", "steps", "alpha", "k", "min_steps", "rel_margin")

    def __init__(self) -> None:
        self.ema = None
        self.ema_var = 0.0
        self.steps = 0
        self.alpha = _env_float("WL_SIGNAL_OUTLIER_EMA_ALPHA", 0.05)
        self.k = _env_float("WL_SIGNAL_OUTLIER_K", 3.0)
        self.min_steps = _env_int("WL_SIGNAL_OUTLIER_MIN_STEPS", 10)
        self.rel_margin = _env_float("WL_SIGNAL_OUTLIER_REL_MARGIN", 0.5)

    def margin(self):
        """Half-width of the on-trend band, or ``None`` while still warming up."""
        if self.ema is None or self.steps < self.min_steps:
            return None
        std = self.ema_var ** 0.5
        return max(self.k * std, self.rel_margin * abs(self.ema))

    def observe(self, average: float) -> None:
        """Fold this step's average into the trend. Call once per emitted point."""
        if self.ema is None:
            self.ema = average
            self.steps = 1
            return
        deviation = average - self.ema
        self.ema += self.alpha * deviation
        self.ema_var = (1.0 - self.alpha) * self.ema_var + self.alpha * (deviation ** 2)
        self.steps += 1

    def find_outliers(self, samples):
        """Flag the off-trend members of one step's batch.

        Args:
            samples: Iterable of ``(sample_id, value)`` for this step.

        Returns:
            ``(top, total)`` where *top* is a list of ``{"sample_id", "value"}``
            dicts sorted by deviation (strongest first) and truncated to
            ``WL_SIGNAL_OUTLIER_TOP_N``, and *total* is how many samples were
            flagged before truncation. ``([], 0)`` while warming up.
        """
        margin = self.margin()
        if margin is None or not samples:
            return [], 0

        flagged = []
        for sample_id, value in samples:
            deviation = abs(value - self.ema)
            if deviation > margin:
                flagged.append((deviation, sample_id, value))

        if not flagged:
            return [], 0

        flagged.sort(key=lambda row: row[0], reverse=True)
        top_n = max(1, _env_int("WL_SIGNAL_OUTLIER_TOP_N", 5))
        top = [
            {"sample_id": str(sample_id), "value": float(value)}
            for _, sample_id, value in flagged[:top_n]
        ]
        return top, len(flagged)


class LoggerQueue:
    def __init__(self, register: bool = True, db_path: str = ":memory:") -> None:
        self.graph_names = set()
        self._current_step_buffer = {}
        self._last_step = None

        # Rolling trend per (graph_name, exp_hash), used to flag the samples in a
        # step's batch that sit off the curve (see _TrendTracker). In-memory only:
        # a resumed run re-warms from its first min_steps points rather than
        # inheriting a stale band.
        self._trend_trackers: dict = defaultdict(_TrendTracker)

        # Live-streaming queue of new points waiting to be sent to WeightsStudio.
        self._pending_queue = []
        self._buffered_step = None

        # Evaluation mode state (transient).
        self._eval_mode_active: bool = False
        self._eval_mode_hash: str = ""
        self._eval_mode_split: str = ""
        self._eval_mode_tags: list[str] = []
        self._eval_accum: dict = {} # {graph_name: [sum, count]}

        # Background flush + loss-shape autotag state. Every flag="loss" signal
        # is auto-classified by default (see _autotag_loss_shapes); these only
        # hold user overrides/opt-outs for specific signals (or all of them).
        self._loss_shape_overrides: dict = {}  # {signal_name: (tag_name, classifier)}
        self._loss_shape_disabled: set = set()
        self._loss_shape_all_disabled: bool = False
        # {(signal_name, exp_hash): {sample_id, ...}} — sample_ids with a new
        # per-sample write for that signal + hash since the last successful
        # tag pass (populated only by _stage_sample_row on an actual new
        # write). Lets _autotag_loss_shapes() reclassify just the samples
        # that actually changed instead of every sample ever logged under
        # that signal, and skip the pair entirely when nothing changed.
        # Deliberately separate from _qps_version: that one is also reset
        # wholesale by _invalidate_qps_cache() on every step boundary (it's a
        # read-cache key), which would otherwise make every signal look
        # "changed" on every step regardless of whether it actually got new
        # data (see _autotag_loss_shapes).
        self._loss_shape_dirty_samples: dict = defaultdict(set)
        self._flush_stop = threading.Event()
        self._flush_thread: threading.Thread | None = None

        # DuckDB connection + write-staging buffers.
        self._lock = threading.RLock()
        self._db_path = db_path
        self._conn = duckdb.connect(database=db_path)
        self._stage_signals: list = []
        self._stage_sample: list = []
        self._stage_instance: list = []
        self._seq = 0

        # Absolute paths of sibling DBs already merged in via merge_from_disk,
        # so repeated merge triggers (logger can be bound to disk from three
        # different call sites depending on init ordering) stay idempotent.
        self._merged_source_dbs: set = set()

        # Per-sample query cache. Many consumers read the SAME (signal, ids) in
        # one step (e.g. reactive signals all reading the loss); memoize so N
        # identical reads cost 1 scan. Keyed by (signal, ids, [step,] hash,
        # version[signal]); staging a row bumps that signal's version to
        # invalidate. Step-scoped: _stage_sample_row clears both caches when the
        # step advances (keys never recur across steps, so old entries are dead).
        # Cache size is env-configurable (WL_QUERY_CACHE_MAXSIZE, default 2048).
        _qps_maxsize = int(os.environ.get("WL_QUERY_CACHE_MAXSIZE", "2048"))
        self._qps_version: dict = defaultdict(int)
        self._qps_cache_step: int = -1
        self._qps_cache = functools.lru_cache(maxsize=_qps_maxsize)(self._query_per_sample_uncached)
        self._qps_step_cache = functools.lru_cache(maxsize=_qps_maxsize)(self._query_per_sample_at_step_uncached)

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

        # If no explicit db_path was given but a checkpoint manager already
        # exists, persist history to an on-disk DuckDB file under its loggers/
        # dir. (The reverse ordering — CM created after the logger — is handled
        # by CheckpointManager.__init__ calling set_db_path on the live logger.)
        if db_path == ":memory:":
            try:
                loggers_dir = getattr(self.chkpt_manager, "loggers_dir", None)
                if loggers_dir:
                    self.set_db_path(os.path.join(str(loggers_dir), "loggers.duckdb"))
            except Exception:
                pass

        # Checkpoint manager was created before this logger and already
        # resolved a multi-root parent directory — merge sibling roots' curves
        # in now (the reverse ordering is handled by
        # CheckpointManager._bind_logger_to_disk; merge_from_disk is
        # idempotent so both call sites firing is harmless).
        if getattr(self.chkpt_manager, "is_multi_root", False):
            try:
                self.chkpt_manager._merge_sibling_logger_histories()
            except Exception as exc:
                logger.warning(f"[LoggerQueue] Failed to merge multi-root sibling logger histories: {exc}")

        self._start_background_flush()

    # ------------------------------------------------------------------
    # Background flush + loss-shape autotag
    # ------------------------------------------------------------------
    def _start_background_flush(self) -> None:
        """Start the periodic flush/loss-shape thread (off the caller's thread).

        Runs for the life of the process (daemon) so neither the training loop
        nor the gRPC servicer has to remember to flush or re-tag loss shapes
        themselves — every flag="loss" signal is auto-classified with zero
        setup. See _autotag_loss_shapes for the tagging half."""
        if self._flush_thread is not None and self._flush_thread.is_alive():
            return
        self._flush_stop.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_loop, name="WL-Logger-Flush", daemon=True)
        self._flush_thread.start()

    def set_loss_shape_override(self, loss_signal: str, tag_name: str | None = None,
                                 classifier=None) -> None:
        """Override the tag name / classifier used when auto-tagging *loss_signal*.

        Every ``flag="loss"`` signal is already auto-classified with zero setup
        (see :meth:`_autotag_loss_shapes`); call this only to customize one
        specific signal (e.g. it isn't a decreasing loss, so the default
        classifier is wrong for it). Also re-enables that signal if it had been
        disabled via :meth:`disable_loss_shape_autotag`.
        """
        with self._lock:
            self._loss_shape_overrides[loss_signal] = (tag_name, classifier)
            self._loss_shape_disabled.discard(loss_signal)

    def disable_loss_shape_autotag(self, loss_signal: str | None = None) -> None:
        """Stop automatic loss-shape tagging for *loss_signal*, or for every
        signal (including ones registered later) if *loss_signal* is ``None``.
        The background flush itself keeps running either way."""
        with self._lock:
            if loss_signal is None:
                self._loss_shape_all_disabled = True
            else:
                self._loss_shape_disabled.add(loss_signal)
                self._loss_shape_overrides.pop(loss_signal, None)

    def _autotag_loss_shapes(self) -> None:
        """Re-classify, per sample, every auto-detected ``flag="loss"`` signal
        sample that got NEW per-sample data for the *current* experiment hash
        since its last tag pass.

        Scoped to ``chkpt_manager.get_current_experiment_hash()`` — a sample's
        shape is classified on its trajectory within the run that's actually
        active, not merged across every hash ever logged for that sample id
        (e.g. sibling roots merged in via merge_from_disk, or a resumed run
        under a new hash). Without a checkpoint manager (e.g. a standalone
        LoggerQueue in tests/notebooks) the hash resolves to ``None``, which
        ``write_signal_shapes`` treats as "no hash filter" — the only
        sensible behavior when there's no run boundary to scope to.

        A (signal, hash) pair is skipped entirely when ``_loss_shape_dirty_
        samples`` (populated only by ``_stage_sample_row`` on an actual new
        per-sample write for that signal + hash) has no sample_ids recorded
        for it since the last pass — no new points means the classification
        result (and thus every categorical tag written from it) would come
        out identical, so re-running the full read + classify + tag-write
        cycle on a bare 2s timer regardless of whether training even produced
        anything new was pure waste, and contended ``self._lock`` against
        unrelated reads (e.g. ``get_signal_history()`` for
        ``GetLatestLoggerData``) the whole time training was paused/idle too.
        And when it does run, only the dirty sample_ids are reclassified —
        each sample's shape is a function of its own trajectory alone, so a
        new point on sample A can't change sample B's label; re-running B's
        classifier every time some other sample in the same signal moves
        would be pure waste too. Failures are per-signal and never
        propagate — one bad classifier/signal can't stop the others.
        """
        exp_hash = self.chkpt_manager.get_current_experiment_hash() if self.chkpt_manager else None
        with self._lock:
            if self._loss_shape_all_disabled:
                return
            disabled = set(self._loss_shape_disabled)
            overrides = dict(self._loss_shape_overrides)
        try:
            # Lazy import: weightslab.src imports LoggerQueue at module load
            # time, so importing it back here at module scope would cycle.
            from weightslab.src import auto_loss_shape_signal_names, write_signal_shapes
        except Exception as exc:
            logger.debug(f"[LoggerQueue] loss-shape autotag: import failed: {exc}")
            return
        # Union, not just the auto-detected set: set_loss_shape_override() can
        # also opt in a signal that wasn't registered via flag="loss" (e.g. a
        # manually wl.save_signals()-logged one under a custom name).
        signal_names = set(auto_loss_shape_signal_names()) | set(overrides.keys())
        for signal_name in signal_names:
            if signal_name in disabled:
                continue
            dirty_key = (signal_name, exp_hash)
            with self._lock:
                dirty = self._loss_shape_dirty_samples.get(dirty_key)
                sample_ids = list(dirty) if dirty else None
            if not sample_ids:
                continue  # no new per-sample data logged since the last pass
            tag_name, classifier = overrides.get(signal_name, (None, None))
            try:
                write_signal_shapes(
                    signal_name, tag_name=tag_name, classifier=classifier,
                    exp_hash=exp_hash, sample_ids=sample_ids)
                with self._lock:
                    # Only drop the sample_ids snapshotted above — a write
                    # staged for this signal+hash *during* the call above
                    # (from a concurrent training thread) added a sample_id
                    # that was never read by write_signal_shapes, so it must
                    # stay dirty for the next pass rather than being dropped
                    # here as if it had already been classified.
                    still_dirty = self._loss_shape_dirty_samples.get(dirty_key)
                    if still_dirty is not None:
                        still_dirty.difference_update(sample_ids)
                        if not still_dirty:
                            self._loss_shape_dirty_samples.pop(dirty_key, None)
            except Exception as exc:
                logger.debug(
                    f"[LoggerQueue] loss-shape autotag failed for {signal_name!r}: {exc}")

    def _flush_loop(self) -> None:
        interval = _default_flush_interval_seconds()
        while not self._flush_stop.wait(interval):
            try:
                self.flush_to_disk()
            except Exception as exc:
                logger.debug(f"[LoggerQueue] background flush failed: {exc}")
            self._autotag_loss_shapes()

    def stop_background_flush(self) -> None:
        """Stop the background flush/loss-shape thread (e.g. at shutdown or in tests)."""
        self._flush_stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # DuckDB plumbing
    # ------------------------------------------------------------------
    @staticmethod
    def _schema_ddl(prefix: str = "") -> str:
        """Return the CREATE-TABLE DDL for all history tables.

        ``prefix`` lets the same schema be created inside an attached database
        (e.g. ``"ondisk."``) when migrating from in-memory to a file.
        """
        return f"""
            CREATE TABLE IF NOT EXISTS {prefix}signals (
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
                outliers VARCHAR,
                outlier_count INTEGER,
                sample_count INTEGER,
                trend_value DOUBLE,
                trend_margin DOUBLE,
                value_min DOUBLE,
                value_max DOUBLE,
                seq BIGINT
            );
            CREATE TABLE IF NOT EXISTS {prefix}per_sample (
                metric_name VARCHAR,
                experiment_hash VARCHAR,
                sample_id VARCHAR,
                step INTEGER,
                value REAL,
                seq BIGINT
            );
            CREATE TABLE IF NOT EXISTS {prefix}per_instance (
                metric_name VARCHAR,
                experiment_hash VARCHAR,
                sample_id VARCHAR,
                annotation_id INTEGER,
                step INTEGER,
                value REAL,
                seq BIGINT
            );
        """

    # Columns added to `signals` after the table's first release, as
    # (name, DDL type, default). A DB file written by an older weightslab
    # predates them, and CREATE TABLE IF NOT EXISTS won't retrofit them, so
    # _ensure_tables ALTERs them in on open. Appended columns land at the end of
    # the table, which is why every INSERT names its columns explicitly instead
    # of relying on staging-buffer order (see _flush_stage).
    _SIGNAL_MIGRATIONS = (
        ("outliers", "VARCHAR", "''"),
        ("outlier_count", "INTEGER", "0"),
        ("sample_count", "INTEGER", "0"),
        # NULL (not 0) so "no band recorded" stays distinguishable from a real
        # band centred on zero.
        ("trend_value", "DOUBLE", "NULL"),
        ("trend_margin", "DOUBLE", "NULL"),
        ("value_min", "DOUBLE", "NULL"),
        ("value_max", "DOUBLE", "NULL"),
    )

    # Indexes for the read paths that filter rather than scan. DuckDB is
    # columnar and leans on zone maps, so these matter less than on a row store
    # -- but the signal-history reads all filter by metric_name/experiment_hash
    # and range-scan `step`, and at hundreds of millions of rows an unindexed
    # equality filter on a low-cardinality string column is the difference
    # between a pruned read and a full scan.
    _INDEX_DDL = (
        ("idx_signals_curve", "signals", "(metric_name, experiment_hash, step)"),
        ("idx_signals_step", "signals", "(step)"),
        ("idx_per_sample_curve", "per_sample", "(metric_name, experiment_hash, step)"),
        ("idx_per_instance_curve", "per_instance", "(metric_name, experiment_hash, step)"),
    )

    def _ensure_indexes(self) -> None:
        """Kick off index creation in the background.

        On a fresh/small DB this is instant either way, but on an existing
        file with many runs' worth of history (the resume path, via
        ``set_db_path``) building 4 indexes over the full tables can take a
        long time -- and this used to run synchronously inside
        ``_ensure_tables()`` while holding ``self._lock``, which every read
        (including the one behind ``GetLatestLoggerData``) also needs. That
        made a large resume look like the server had hung: every RPC queued
        on the lock for the whole build instead of erroring or completing.

        Building on a dedicated cursor lets the main connection keep serving
        reads/writes (unindexed, same as before this feature existed) while
        the indexes come up in the background. An index that fails to build
        (older DuckDB, read-only db, a transient conflict with a concurrent
        write) must never stop logging, so failures here are swallowed.
        """
        def _build() -> None:
            try:
                cur = self._conn.cursor()
            except Exception as exc:
                logger.debug("Could not open index-build cursor: %s", exc)
                return
            for name, table, cols in self._INDEX_DDL:
                try:
                    cur.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table} {cols}")
                except Exception as exc:  # older DuckDB, read-only db, conflict, ...
                    logger.debug("Index %s on %s not created: %s", name, table, exc)

        threading.Thread(target=_build, daemon=True, name="wl-index-build").start()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.execute(self._schema_ddl())
            self._migrate_signal_columns()
        # Deliberately outside the lock -- see _ensure_indexes.
        self._ensure_indexes()

    def _migrate_signal_columns(self) -> None:
        """Add any post-release `signals` columns missing from an older DB file."""
        try:
            existing = {
                row[0] for row in self._conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'signals'"
                ).fetchall()
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not introspect signals columns: %s", exc)
            return

        for name, ddl_type, default in self._SIGNAL_MIGRATIONS:
            if name in existing:
                continue
            try:
                self._conn.execute(
                    f"ALTER TABLE signals ADD COLUMN {name} {ddl_type} DEFAULT {default}"
                )
                logger.info("Migrated signal history: added signals.%s", name)
            except Exception as exc:
                logger.warning("Failed to add signals.%s: %s", name, exc)

    _HISTORY_TABLES = ("signals", "per_sample", "per_instance")

    def set_db_path(self, db_path) -> None:
        """Persist signal history to an on-disk DuckDB file.

        Call this once, early in setup. If the file already exists (resume),
        its data is adopted as-is. If it does not, whatever little is currently
        in the in-memory DB is migrated into the new file. Either way the file
        becomes the live connection afterwards.

        The hot logging path is unaffected: ``add_scalars`` still stages to RAM
        and only bulk-flushes to DuckDB lazily. DuckDB serves reads from its
        in-memory buffer pool, so this adds durability, not per-read disk hits.
        """
        if not db_path or db_path == ":memory:":
            return
        db_path = str(db_path)

        with self._lock:
            if self._db_path == db_path:
                return

            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            file_preexists = os.path.exists(db_path)

            try:
                # Make sure staged rows are materialized before we migrate.
                self._flush_stage()

                if not file_preexists:
                    # Fresh file: copy whatever is in the in-memory DB into it.
                    # ATTACH doesn't accept bind parameters, so inline the path
                    # with SQL-escaped quotes.
                    escaped = db_path.replace("'", "''")
                    self._conn.execute(f"ATTACH '{escaped}' AS ondisk")
                    self._conn.execute(self._schema_ddl(prefix="ondisk."))
                    for tbl in self._HISTORY_TABLES:
                        self._conn.execute(
                            f"INSERT INTO ondisk.{tbl} SELECT * FROM {tbl}"
                        )
                    self._conn.execute("DETACH ondisk")

                # Adopt the on-disk file as the live connection. On resume this
                # is the source of truth; the fresh in-memory rows are ignored.
                self._conn.close()
                self._conn = duckdb.connect(database=db_path)
                self._db_path = db_path
                self._ensure_tables()
                self._invalidate_qps_cache()
                self._restore_runtime_state_from_db()
                logger.info(
                    f"[LoggerQueue] Signal history persisted on disk at {db_path} "
                    f"({'adopted existing' if file_preexists else 'new'} database)."
                )
            except Exception as exc:
                logger.warning(
                    f"[LoggerQueue] Failed to enable on-disk persistence at "
                    f"{db_path}: {exc}. Keeping in-memory history."
                )

    def flush_to_disk(self) -> None:
        """Flush staged rows and force a DuckDB checkpoint to the file.

        No-op for an in-memory database. Call at checkpoint time so history is
        durable even without a clean shutdown (DuckDB also replays its WAL on
        the next open, so this is belt-and-braces)."""
        with self._lock:
            try:
                self._flush_stage()
                if self._db_path != ":memory:":
                    self._conn.execute("CHECKPOINT")
            except Exception as exc:
                logger.warning(f"[LoggerQueue] flush_to_disk failed: {exc}")

    def merge_from_disk(self, other_db_path) -> bool:
        """Merge signal-history rows from another on-disk DuckDB file into
        this logger's live tables.

        Used to stitch training curves together from sibling experiment
        roots discovered under a shared parent directory (see
        CheckpointManager._merge_sibling_logger_histories). Purely additive:
        rows are namespaced by ``experiment_hash``, so independent hash
        chains from different roots never collide, and nothing already in
        this logger is touched. A no-op (returns False) if the path doesn't
        exist, is this same database, or the merge fails outright.
        """
        if not other_db_path:
            return False
        other_db_path = os.path.abspath(str(other_db_path))
        if not os.path.exists(other_db_path):
            return False
        if self._db_path not in (None, ":memory:") and other_db_path == os.path.abspath(self._db_path):
            return False

        with self._lock:
            if other_db_path in self._merged_source_dbs:
                # Already merged (this can be triggered from more than one
                # init-ordering call site) — skip to avoid duplicating rows.
                return False

            try:
                self._flush_stage()
                escaped = other_db_path.replace("'", "''")
                self._conn.execute(f"ATTACH '{escaped}' AS incoming (READ_ONLY)")
            except Exception as exc:
                logger.warning(f"[LoggerQueue] Failed to attach {other_db_path} for merge: {exc}")
                return False

            try:
                copied_any = False
                for tbl in self._HISTORY_TABLES:
                    try:
                        self._conn.execute(f"INSERT INTO {tbl} SELECT * FROM incoming.{tbl}")
                        copied_any = True
                    except Exception as exc:
                        # Table may be missing in an older-schema sibling DB;
                        # skip it, other tables still merge.
                        logger.debug(f"[LoggerQueue] Skipped merging table '{tbl}' from {other_db_path}: {exc}")
            finally:
                try:
                    self._conn.execute("DETACH incoming")
                except Exception:
                    pass

            if copied_any:
                self._merged_source_dbs.add(other_db_path)
                self._invalidate_qps_cache()
                self._restore_runtime_state_from_db()
                logger.info(f"[LoggerQueue] Merged signal history from {other_db_path}")
            return copied_any

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

        Uses register(pandas)->INSERT SELECT->unregister (DuckDB's fast bulk
        path). A row-wise executemany was measured ~6x slower — don't switch."""
        with self._lock:
            if self._stage_signals:
                df = pd.DataFrame(self._stage_signals, columns=_SIGNAL_COLS)
                self._conn.register("_stg_sig", df)
                # Column-explicit: migrated columns sit at the end of an older
                # table, so positional INSERT ... SELECT * would misalign.
                cols = ", ".join(_SIGNAL_COLS)
                self._conn.execute(
                    f"INSERT INTO signals ({cols}) SELECT {cols} FROM _stg_sig"
                )
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
                          audit_mode, is_marker, split_name, eval_tags, point_note,
                          outliers=None, outlier_count=0, sample_count=0,
                          trend_value=None, trend_margin=None,
                          value_min=None, value_max=None):
        self._stage_signals.append((
            graph_name, exp_hash, int(step), float(metric_value), int(timestamp),
            bool(audit_mode), bool(is_marker), split_name or "",
            json.dumps(list(eval_tags or [])), point_note or "",
            json.dumps(list(outliers)) if outliers else "",
            int(outlier_count), int(sample_count),
            None if trend_value is None else float(trend_value),
            None if trend_margin is None else float(trend_margin),
            None if value_min is None else float(value_min),
            None if value_max is None else float(value_max),
            self._next_seq(),
        ))
        self._maybe_autoflush()

    def _stage_sample_row(self, graph_name, exp_hash, sample_id, step, value):
        # New step -> last step's cache entries can't recur; drop them.
        if int(step) > self._qps_cache_step:
            self._invalidate_qps_cache()
            self._qps_cache_step = int(step)
        self._stage_sample.append(
            (
                graph_name, exp_hash, str(sample_id), int(step), float(value), self._next_seq(),
            )
        )
        self._qps_version[graph_name] += 1   # invalidate this signal's cached reads
        self._loss_shape_dirty_samples[(graph_name, exp_hash)].add(str(sample_id))
        self._maybe_autoflush()

    def _invalidate_qps_cache(self) -> None:
        """Drop both query caches + versions (step advance; bulk delete/clear).

        Deliberately leaves _loss_shape_dirty_samples alone: it tracks actual
        new per-sample writes per (signal, hash, sample_id), not the
        read-cache's step-scoped keys, so a step boundary elsewhere must not
        make _autotag_loss_shapes() think every sample of every signal changed.
        """
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
                              evaluation_tags=None, batch_samples=None):
        """Stage a signals row and return the live-queue entry dict.

        *batch_samples*, when given, is the ``(sample_id, value)`` batch this
        point's average came from. It is compared against the signal's rolling
        trend to flag off-trend samples, which ride along on the point so the UI
        can mark the spike and jump to the samples behind it.
        """
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

        outliers, outlier_count = [], 0
        trend_value, trend_margin = None, None
        value_min, value_max = None, None
        sample_count = len(batch_samples) if batch_samples else 0

        # Absolute extremes of the batch. These are the real lowest/highest
        # sample values, so the band the UI draws from them spikes out to an
        # outlier rather than averaging it down the way a std band would.
        if batch_samples:
            values = [value for _, value in batch_samples]
            value_min, value_max = min(values), max(values)

        if _outliers_enabled() and not is_marker:
            tracker = self._trend_trackers[(graph_name, exp_hash)]
            # Snapshot the band BEFORE folding this point in, so a spike is
            # measured against clean history instead of partly against itself.
            # This is the same band find_outliers uses, which is what lets the UI
            # draw the region a flagged sample fell outside of.
            margin = tracker.margin()
            if margin is not None:
                trend_value, trend_margin = tracker.ema, margin
            if batch_samples:
                outliers, outlier_count = tracker.find_outliers(batch_samples)
            tracker.observe(metric_value)
        if outliers:
            signal_entry["outliers"] = outliers
            signal_entry["outlier_count"] = outlier_count
        if sample_count:
            signal_entry["sample_count"] = sample_count
        if trend_value is not None:
            signal_entry["trend_value"] = trend_value
            signal_entry["trend_margin"] = trend_margin
        if value_min is not None:
            signal_entry["value_min"] = value_min
            signal_entry["value_max"] = value_max

        with self._lock:
            self._stage_signal_row(
                graph_name, exp_hash, global_step, metric_value, timestamp,
                bool(audit_mode), bool(is_marker), split_name,
                list(evaluation_tags or []), "",
                outliers=outliers, outlier_count=outlier_count,
                sample_count=sample_count,
                trend_value=trend_value, trend_margin=trend_margin,
                value_min=value_min, value_max=value_max,
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
                batch_samples=payload.get("samples"),
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

            # (sample_id, value) for this call's batch, so _append_history_entry
            # can attribute an off-trend point to the samples responsible.
            # Derived from signal_per_sample independently of which branch below
            # supplies metric_values: in immediate mode the emitted value comes
            # from `signal`, but the batch behind it is still signal_per_sample,
            # and outliers are judged against the curve's trend either way.
            batch_samples = [
                (str(sid), self._to_float(value))
                for sid, value in signal_per_sample.items()
            ] if isinstance(signal_per_sample, dict) and len(signal_per_sample) else []

            metric_values = []
            if isinstance(signal_per_sample, dict) and aggregate_by_step and len(signal_per_sample):
                metric_values = [value for _, value in batch_samples]
            else:
                for _, line_value in signal.items():
                    metric_values.append(self._to_float(line_value))

            if aggregate_by_step:
                if metric_values:
                    self._buffered_step = global_step
                    buffer_key = (global_step, graph_name, exp_hash)
                    if buffer_key not in self._current_step_buffer:
                        self._current_step_buffer[buffer_key] = {
                            "sum": 0.0, "count": 0, "samples": [],
                        }
                    payload = self._current_step_buffer[buffer_key]
                    payload["sum"] += sum(metric_values)
                    payload["count"] += len(metric_values)
                    if batch_samples:
                        headroom = _MAX_BUFFERED_SAMPLES_PER_STEP - len(payload["samples"])
                        if headroom > 0:
                            payload["samples"].extend(batch_samples[:headroom])
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
                    batch_samples=batch_samples or None,
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
        history = self.get_signal_history(max_points=None)
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

    @staticmethod
    def _decode_outliers(raw):
        """Parse a stored ``outliers`` JSON blob into a list of dicts.

        Rows written before the column existed read back as NULL/'' and legacy
        files could in principle hold junk, so a parse failure degrades to "no
        outliers" rather than breaking the whole history read.
        """
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            sample_id = str(item.get("sample_id", ""))
            if not sample_id:
                continue
            try:
                value = float(item.get("value", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            out.append({"sample_id": sample_id, "value": value})
        return out

    def get_step_outlier_sample_ids(self, metric_name: str, experiment_hash: str,
                                    model_age: int) -> list:
        """Sample ids flagged as off-trend for one point of one curve.

        Backs the plot's "Highlight step samples" action: the UI hands back the
        metric/hash/step it was right-clicked on and gets the ids to filter the
        data grid down to.
        """
        with self._lock:
            self._flush_stage()
            params = [metric_name, int(model_age)]
            sql = ("SELECT outliers FROM signals "
                   "WHERE metric_name = ? AND step = ?")
            if experiment_hash:
                sql += " AND experiment_hash = ?"
                params.append(experiment_hash)
            sql += " ORDER BY seq"
            rows = self._conn.execute(sql, params).fetchall()

        seen, ids = set(), []
        for (raw,) in rows:
            for item in self._decode_outliers(raw):
                sample_id = item["sample_id"]
                if sample_id and sample_id not in seen:
                    seen.add(sample_id)
                    ids.append(sample_id)
        return ids

    def get_step_sample_ids(self, metric_name: str, experiment_hash: str,
                            model_age: int, max_samples: int = 0) -> tuple:
        """Every sample id that contributed to one step of one signal.

        Backs the plot's "Highlight step samples" action, which shows the WHOLE
        batch behind a point rather than only the off-trend members of it.

        Args:
            metric_name: Signal name.
            experiment_hash: Restrict to one run; ``""``/``None`` means any.
            model_age: The step.
            max_samples: Cap on returned ids (0 = no cap).

        Returns:
            ``(ids, total_available)`` — *total_available* is the count before
            the cap, so a caller can say "showing 200 of 4096".
        """
        with self._lock:
            self._flush_stage()
            params = [metric_name, int(model_age)]
            sql = ("SELECT DISTINCT sample_id FROM per_sample "
                   "WHERE metric_name = ? AND step = ?")
            if experiment_hash:
                sql += " AND experiment_hash = ?"
                params.append(experiment_hash)
            rows = self._conn.execute(sql, params).fetchall()

        ids = [str(row[0]) for row in rows if row[0] is not None]
        # Numeric-aware ordering so "9" precedes "10"; falls back to plain text
        # for non-numeric ids.
        try:
            ids.sort(key=lambda value: (0, int(value)))
        except ValueError:
            ids.sort()
        total = len(ids)
        if max_samples and max_samples > 0:
            ids = ids[:max_samples]
        return ids, total

    def _scope_filters(self, metric_names=None, exp_hashes=None,
                       x_min=None, x_max=None, alias: str = ""):
        """Build the shared WHERE fragment + params for a scoped history read."""
        p = f"{alias}." if alias else ""
        sql, params = "", []
        if metric_names:
            names = list(dict.fromkeys(metric_names))
            sql += f" AND {p}metric_name IN ({','.join('?' * len(names))})"
            params.extend(names)
        if exp_hashes:
            hashes = list(dict.fromkeys(exp_hashes))
            sql += f" AND {p}experiment_hash IN ({','.join('?' * len(hashes))})"
            params.extend(hashes)
        if x_min is not None:
            sql += f" AND {p}step >= ?"
            params.append(int(x_min))
        if x_max is not None:
            sql += f" AND {p}step <= ?"
            params.append(int(x_max))
        return sql, params

    def get_signal_curve_index(self, metric_names=None, exp_hashes=None) -> dict:
        """Per-curve shape without any of its points.

        Returns ``{metric: {hash: {"first_step", "last_step", "count",
        "value_min", "value_max"}}}``. This is a single grouped aggregate --
        it lets the UI lay out axes, decide how many points to ask for, show
        which curves exist, and now also show each curve's own value range,
        all without reading a single data point.

        A hash-less signal (e.g. a global/resource metric logged with no
        experiment_hash) groups under a ``NULL`` key -- normalised to "N.A."
        here (the same sentinel the break-by-slices path already uses for
        this), since the caller assigns this straight into a protobuf string
        field and a bare ``None`` would raise there instead of just being an
        odd-looking curve.
        """
        where, params = self._scope_filters(metric_names, exp_hashes)
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(
                "SELECT metric_name, experiment_hash, MIN(step), MAX(step), COUNT(*), "
                "MIN(metric_value), MAX(metric_value) "
                f"FROM signals WHERE 1=1{where} GROUP BY 1, 2", params
            ).fetchall()
        out: dict = {}
        for metric, h, lo, hi, n, val_lo, val_hi in rows:
            out.setdefault(metric, {})[h if h is not None else "N.A."] = {
                "first_step": int(lo), "last_step": int(hi), "count": int(n),
                "value_min": float(val_lo), "value_max": float(val_hi),
            }
        return out

    def get_signal_history_downsampled(
        self, max_points: int | None = None, metric_names=None, exp_hashes=None,
        x_min=None, x_max=None, keep_special: bool = True,
    ) -> dict:
        """Aggregated history reduced to ~``max_points`` per curve *inside DuckDB*.

        Same ``{metric: {hash: {step: [entry, ...]}}}`` shape as
        :meth:`get_signal_history`, but the reduction happens in SQL, so the
        number of rows crossing into Python is bounded by the number of points
        actually drawable -- not by the table size. This is what makes a
        hundred-million-row history openable.

        The curve is split into ``max_points`` equal step-buckets and one
        representative (the bucket's earliest step) is emitted per bucket, via a
        streaming hash aggregate rather than a sort/window. Three further rules
        keep the reduced curve faithful:

        * the curve's true first and last steps are always emitted, so endpoints
          and the x-extent never move under downsampling;
        * evaluation markers, annotated points and steps carrying outliers are
          never dropped (``keep_special``) -- those are exactly the points a
          user zooms in to find;
        * ``max_points`` is clamped to at least ``_MIN_POINTS_PER_CURVE``.

        Args:
            max_points: target points per curve. ``None`` uses
                ``WL_SIGNAL_MAX_POINTS_PER_CURVE``.
            metric_names / exp_hashes: restrict to these curves. Passing the one
                signal being zoomed keeps a zoom refetch proportional to that
                plot, not to the whole dashboard.
            x_min / x_max: restrict to a step range -- the zoom path. Buckets
                are laid out across the *visible* range, so zooming in resolves
                real detail instead of restretching the same points.
            keep_special: emit marker/annotated/outlier rows regardless of
                bucketing.
        """
        n_buckets = max(int(max_points or _DEFAULT_MAX_POINTS_PER_CURVE),
                        _MIN_POINTS_PER_CURVE)
        where, params = self._scope_filters(metric_names, exp_hashes, x_min, x_max)
        cols = ", ".join(_SIGNAL_READ_COLS)
        # arg_min(col, step) picks each column from the bucket's earliest-step
        # row, so a representative is one real row rather than a blend. Rows
        # sharing a step within a bucket tie arbitrarily -- acceptable for a
        # decimated view, and the zoom path resolves them.
        picks = ", ".join(
            f"arg_min({c}, step) AS {c}" for c in _SIGNAL_READ_COLS
            if c not in ("metric_name", "experiment_hash", "step")
        )
        picks_max = ", ".join(
            f"arg_max({c}, step) AS {c}" for c in _SIGNAL_READ_COLS
            if c not in ("metric_name", "experiment_hash", "step")
        )
        sql = f"""
        WITH scoped AS (
            SELECT {cols} FROM signals WHERE 1=1{where}
        ),
        bounds AS (
            SELECT metric_name AS m, experiment_hash AS h,
                   MIN(step) AS lo, MAX(step) AS hi
            FROM scoped GROUP BY 1, 2
        ),
        tagged AS (
            SELECT s.*,
                   CASE WHEN b.hi <= b.lo THEN 0
                        -- step/lo/hi are all INTEGER (INT32) columns; the
                        -- subtraction fits fine, but multiplying that by
                        -- n_buckets can overflow INT32 on a long-running
                        -- experiment (e.g. step ~538k * a few thousand
                        -- buckets already exceeds it) well before the
                        -- outer CAST ever gets a chance to widen it. Cast
                        -- BEFORE the multiplication so DuckDB does the
                        -- whole computation in BIGINT instead.
                        -- DuckDB's `/` is float division even between two
                        -- integer operands (unlike Postgres/MySQL) -- it would
                        -- leave `bucket` a near-unique float per row instead of
                        -- an integer 0..n_buckets, so GROUP BY bucket below
                        -- would barely deduplicate anything. `//` is DuckDB's
                        -- floor-division operator; that's the one we need here.
                        ELSE (CAST(s.step - b.lo AS BIGINT) * {n_buckets}) // (b.hi - b.lo)
                   END AS bucket
            FROM scoped s
            JOIN bounds b
              ON s.metric_name = b.m AND s.experiment_hash IS NOT DISTINCT FROM b.h
        ),
        reps AS (
            SELECT metric_name, experiment_hash, MIN(step) AS step, {picks}
            FROM tagged GROUP BY metric_name, experiment_hash, bucket
        ),
        ends AS (
            -- One representative row per endpoint, not every raw row that
            -- happens to sit at the min/max step: a metric can log many rows
            -- at the same step (e.g. a per-step outlier snapshot), and a bare
            -- ``step = lo OR step = hi`` filter would pull all of them,
            -- silently defeating the downsampling above (in the extreme
            -- hi == lo case, every row matches and the "capped" query
            -- degenerates into a full-table read).
            SELECT metric_name, experiment_hash, MIN(step) AS step, {picks}
            FROM scoped GROUP BY metric_name, experiment_hash
            UNION ALL
            SELECT metric_name, experiment_hash, MAX(step) AS step, {picks_max}
            FROM scoped GROUP BY metric_name, experiment_hash
        )
        SELECT {cols} FROM reps
        UNION ALL
        SELECT {cols} FROM ends
        """
        special_sql = f"""
        SELECT {cols} FROM signals
        WHERE 1=1{where}
          AND (is_evaluation_marker
               OR (point_note IS NOT NULL AND point_note <> '')
               OR COALESCE(outlier_count, 0) > 0)
        LIMIT {_MAX_SPECIAL_ROWS + 1}
        """
        with self._lock:
            self._flush_stage()
            rows = self._conn.execute(sql, params).fetchall()
            special = (self._conn.execute(special_sql, params).fetchall()
                       if keep_special else [])
        if len(special) > _MAX_SPECIAL_ROWS:
            logger.warning(
                "Signal history: more than %d marker/annotated/outlier points "
                "matched; keeping the first %d. Narrow the step range or the "
                "signal set to see the rest.",
                _MAX_SPECIAL_ROWS, _MAX_SPECIAL_ROWS)
            special = special[:_MAX_SPECIAL_ROWS]

        # UNION ALL can repeat a row across the three branches; dedupe on the
        # identity the UI keys on. Bounded by the reduced row count, not by the
        # table size.
        seen: set = set()
        result: dict = {}
        for row in itertools.chain(rows, special):
            key = (row[0], row[1], row[2])
            if key in seen:
                continue
            seen.add(key)
            self._accumulate_history_row(result, row)
        for metric in result.values():
            for steps in metric.values():
                for entries in steps.values():
                    entries.sort(key=lambda e: e.get("timestamp", 0))
        return result

    def iter_signal_rows(self, metric_names=None, exp_hashes=None,
                         x_min=None, x_max=None, batch_size: int | None = None):
        """Stream raw ``signals`` rows as ``(graph_name, hash, step, value)``.

        The complete-fidelity read that does NOT build the nested history dict.
        ``get_signal_history(max_points=None)`` allocates a metadata dict per row
        before the caller sees anything, so exporting a large table costs several
        multiples of the table in RAM. This yields plain tuples in batches, so
        peak memory is one batch plus whatever the consumer keeps.

        Consumers that write incrementally (parquet row groups, CSV append) stay
        bounded end to end; one that appends everything to a list is still
        proportional to the table -- see :meth:`copy_signals_to_parquet` for the
        path that never brings the rows into Python at all.
        """
        n = int(batch_size or _HISTORY_STREAM_CHUNK)
        where, params = self._scope_filters(metric_names, exp_hashes, x_min, x_max)
        with self._lock:
            self._flush_stage()
            cur = self._conn.execute(
                "SELECT metric_name, experiment_hash, step, metric_value "
                f"FROM signals WHERE 1=1{where} "
                "ORDER BY metric_name, experiment_hash, step", params)
            while True:
                batch = cur.fetchmany(n)
                if not batch:
                    return
                yield from batch

    def copy_signals_to_parquet(self, path: str, metric_names=None,
                                exp_hashes=None, x_min=None, x_max=None) -> str:
        """Export the full signal history straight from DuckDB to parquet.

        Bounded regardless of table size: DuckDB streams the result to the file
        itself, so no row is ever a Python object. This is the only export path
        that is safe on a hundred-million-row history.
        """
        where, params = self._scope_filters(metric_names, exp_hashes, x_min, x_max)
        safe = str(path).replace("'", "''")
        with self._lock:
            self._flush_stage()
            self._conn.execute(
                "COPY (SELECT metric_name AS graph_name, experiment_hash, step, "
                f"metric_value FROM signals WHERE 1=1{where} "
                "ORDER BY metric_name, experiment_hash, step) "
                f"TO '{safe}' (FORMAT PARQUET)", params)
        return path

    def _accumulate_history_row(self, result: dict, row) -> None:
        """Turn one `signals` row into an entry dict under result[m][h][step]."""
        (metric, h, step, val, ts, audit, marker, split, tags, note,
         outliers, outlier_count, sample_count, trend_value, trend_margin,
         value_min, value_max) = row
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
        parsed_outliers = self._decode_outliers(outliers)
        if parsed_outliers:
            entry["outliers"] = parsed_outliers
            entry["outlier_count"] = int(outlier_count or 0)
        if sample_count:
            entry["sample_count"] = int(sample_count)
        if trend_value is not None and trend_margin is not None:
            entry["trend_value"] = float(trend_value)
            entry["trend_margin"] = float(trend_margin)
        if value_min is not None and value_max is not None:
            entry["value_min"] = float(value_min)
            entry["value_max"] = float(value_max)
        result.setdefault(metric, {}).setdefault(h, {}).setdefault(step, []).append(entry)

    def get_signal_history(self, max_points: int | None = -1, metric_names=None,
                           exp_hashes=None, x_min=None, x_max=None):
        """Aggregated history as ``{metric: {hash: {step: [entry, ...]}}}``.

        Downsampled by default. ``max_points`` semantics:

        * omitted / ``-1`` -- reduce to ``WL_SIGNAL_MAX_POINTS_PER_CURVE`` per
          curve. This is the UI path and the safe default: a full read of a
          large history is what exhausts memory.
        * an int -- reduce to that many points per curve.
        * ``None`` -- **every** row. Only for export/snapshot paths that need
          full fidelity. Streams in chunks rather than one ``fetchall()``, but
          the assembled dict still holds one entry per row, so on a very large
          table it is inherently proportional to the table.
        """
        if max_points is not None:
            return self.get_signal_history_downsampled(
                max_points=None if max_points == -1 else max_points,
                metric_names=metric_names, exp_hashes=exp_hashes,
                x_min=x_min, x_max=x_max)

        where, params = self._scope_filters(metric_names, exp_hashes, x_min, x_max)
        cols = ", ".join(_SIGNAL_READ_COLS)
        result: dict = {}
        with self._lock:
            self._flush_stage()
            n = self._conn.execute(
                f"SELECT COUNT(*) FROM signals WHERE 1=1{where}", params
            ).fetchone()[0]
            if n > _HISTORY_STREAM_CHUNK:
                logger.warning(
                    "get_signal_history(max_points=None) is materialising %s rows; "
                    "this is the uncapped export path. Pass max_points for the "
                    "UI/read path so the reduction happens in DuckDB.", f"{n:,}")
            # ORDER BY (metric, hash, step) rides the curve index; the previous
            # ORDER BY seq forced a sort of the whole table on every read.
            cur = self._conn.execute(
                f"SELECT {cols} FROM signals WHERE 1=1{where} "
                "ORDER BY metric_name, experiment_hash, step", params)
            while True:
                batch = cur.fetchmany(_HISTORY_STREAM_CHUNK)
                if not batch:
                    break
                for row in batch:
                    self._accumulate_history_row(result, row)
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

    def query_per_sample(self, graph_name: str, sample_ids=None, exp_hash=None,
                         max_points: int | None = None):
        """Query per-sample history.

        Returns a list of ``(sample_id, step, value, experiment_hash)`` tuples,
        filtered by *sample_ids* and optionally *exp_hash* (``None`` = all hashes).

        *sample_ids* already bounds this read to a handful of ids, but each id's
        own series can still be arbitrarily deep (one row per training step) --
        on a long run that is easily hundreds of millions of rows for a single
        sample. When *max_points* is given, each sample's series is reduced to
        ~*max_points* points *inside DuckDB* (same bucket-per-series decimation
        as :meth:`get_signal_history_downsampled`, just partitioned by
        ``sample_id`` instead of ``(metric_name, experiment_hash)``), so the
        number of rows crossing into Python is bounded by what a per-sample
        trajectory plot can draw, not by how long the run has trained.
        ``max_points=None`` (the default) returns every row, unchanged from
        before -- callers that already scope to a handful of steps rely on
        getting everything back.

        Cached (memoized until *graph_name* is next staged). Returns a fresh list.
        """
        ids_key = tuple(str(s) for s in sample_ids) if sample_ids is not None else None
        cached = self._qps_cache(graph_name, ids_key, exp_hash, max_points,
                                 self._qps_version[graph_name])
        return list(cached)

    def _query_per_sample_uncached(self, graph_name, ids_key, exp_hash, max_points,
                                   _version):
        """DuckDB read behind :meth:`query_per_sample`. ``_version`` is a cache
        key only (bumped on write -> recompute). Returns an immutable tuple."""
        with self._lock:
            self._flush_stage()
            params = [graph_name]
            where = " WHERE metric_name = ?"
            where += self._hash_filter(exp_hash, params)
            if ids_key is not None:
                where += " AND sample_id IN (SELECT UNNEST(?))"
                params.append(list(ids_key))

            if max_points is None:
                sql = ("SELECT sample_id, step, value, experiment_hash "
                       f"FROM per_sample{where} ORDER BY seq")
                rows = self._conn.execute(sql, params).fetchall()
            else:
                # Bucket each sample_id's own series independently -- a sample
                # trained for 10 steps and one trained for 10M steps both come
                # back around `n_buckets` points, instead of the deep one
                # dominating an overall LIMIT. arg_min(..., step) picks a real
                # row per bucket (first-by-step), and the endpoints union keeps
                # each series' true first/last step regardless of bucketing --
                # same shape of guarantee as get_signal_history_downsampled.
                n_buckets = max(int(max_points), _MIN_POINTS_PER_CURVE)
                sql = f"""
                WITH scoped AS (
                    SELECT sample_id, step, value, experiment_hash
                    FROM per_sample{where}
                ),
                bounds AS (
                    SELECT sample_id, MIN(step) AS lo, MAX(step) AS hi
                    FROM scoped GROUP BY sample_id
                ),
                tagged AS (
                    SELECT s.*,
                           -- Cast BEFORE multiplying, not after -- step/lo/hi
                           -- are INTEGER (INT32) columns, and (step - lo) *
                           -- n_buckets can overflow INT32 on a long run well
                           -- before an outer CAST would get a chance to widen
                           -- it (see get_signal_history_downsampled's own
                           -- version of this exact fix).
                           -- `/` is float division in DuckDB even for two
                           -- integer operands -- see the identical fix note in
                           -- get_signal_history_downsampled. `//` floor-divides.
                           CASE WHEN b.hi <= b.lo THEN 0
                                ELSE (CAST(s.step - b.lo AS BIGINT) * {n_buckets})
                                     // (b.hi - b.lo)
                           END AS bucket
                    FROM scoped s JOIN bounds b ON s.sample_id = b.sample_id
                ),
                reps AS (
                    SELECT sample_id,
                           arg_min(step, step) AS step,
                           arg_min(value, step) AS value,
                           arg_min(experiment_hash, step) AS experiment_hash
                    FROM tagged GROUP BY sample_id, bucket
                ),
                ends AS (
                    -- One representative row per endpoint, not every raw row
                    -- at the min/max step -- see get_signal_history_downsampled's
                    -- own version of this exact fix for why a bare
                    -- ``step = lo OR step = hi`` filter can blow up.
                    SELECT sample_id, arg_min(step, step) AS step,
                           arg_min(value, step) AS value,
                           arg_min(experiment_hash, step) AS experiment_hash
                    FROM scoped GROUP BY sample_id
                    UNION ALL
                    SELECT sample_id, arg_max(step, step) AS step,
                           arg_max(value, step) AS value,
                           arg_max(experiment_hash, step) AS experiment_hash
                    FROM scoped GROUP BY sample_id
                )
                SELECT DISTINCT sample_id, step, value, experiment_hash FROM (
                    SELECT sample_id, step, value, experiment_hash FROM reps
                    UNION ALL
                    SELECT sample_id, step, value, experiment_hash FROM ends
                )
                ORDER BY sample_id, step
                """
                rows = self._conn.execute(sql, params).fetchall()

        return tuple((sid, int(step), float(val), h) for (sid, step, val, h) in rows)

    def query_per_sample_at_step(self, graph_name: str, sample_ids, step, exp_hash=None):
        """``(sample_id, value)`` for *graph_name* at exactly *step* — O(batch),
        not O(history). Keeps the reactive gather flat as history grows. Cached."""
        ids_key = tuple(str(s) for s in sample_ids) if sample_ids is not None else None
        cached = self._qps_step_cache(graph_name, ids_key, int(step), exp_hash,
                                      self._qps_version[graph_name])
        return list(cached)

    def _query_per_sample_at_step_uncached(self, graph_name, ids_key, step, exp_hash, _version):
        """DuckDB read behind :meth:`query_per_sample_at_step` (``_version`` = cache key).

        Fast path: the current step's value is usually still in the in-memory
        staging buffer, so scan it and skip the flush->register->INSERT->SELECT
        round-trip. Fall through to DuckDB only if an id isn't staged."""
        step = int(step)
        with self._lock:
            if ids_key is not None:
                ids_set = set(ids_key)
                at = {}
                # Scan from the tail (append-ordered by step); first value per id
                # wins, stop when all found or once we drop below `step`.
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

        # register()'s pandas->DuckDB bulk insert (see _flush_stage) silently turns a
        # staged float('nan') into SQL NULL; float(None) raising here would bubble up
        # through GetStepSamples' broad except and blank out every OTHER sample's
        # value in the same batch, not just this row's.
        return tuple((sid, float(val) if val is not None else float("nan")) for (sid, val) in rows)

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
        exp_hashes=None,
    ) -> dict:
        """Return mean signal value per step, aggregated over matching samples.

        DuckDB performs the ``GROUP BY step`` average natively, which scales to
        millions of rows far better than a Python loop — this is the path used
        by break-by-slices.

        ``exp_hashes`` (plural) scopes the aggregate to specific runs -- e.g.
        break-by-slices for one curve only needs that curve's own hash, not
        every run's. Left ``None`` (the default), every hash in the table is
        aggregated and returned, same as before this parameter existed.

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
            hashes_sql, hashes_params = self._scope_filters(exp_hashes=exp_hashes)
            sql += hashes_sql
            params.extend(hashes_params)
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
        max_points: int | None = None,
    ) -> dict:
        """Reduce each sample's signal HISTORY to a single value.

        Unlike ``aggregate_per_sample_by_step`` (which averages *across samples*
        per step), this groups ``per_sample`` rows BY sample_id and reduces over
        that sample's whole time series — the axis needed for questions like
        "which samples never had train_loss below 0.5" (``reduce='min'`` then
        compare ``>= 0.5``).

        Args:
            graph_name: The registered signal/metric name.
            reduce: One of ``min`` | ``max`` | ``mean``/``avg`` | ``count``, or
                ``list``/``values``/``raw``/``history`` to return each sample's
                FULL time series (ordered by step) as a list instead of a scalar.
            sample_ids: Optional iterable to restrict the query.
            exp_hash: ``None`` = all hashes; otherwise restrict to one.
            max_points: List reduces only — cap each sample's returned series to
                at most this many points by keeping an evenly-spaced subset (first
                and last always kept). ``None`` (default) keeps the full history.

        Returns:
            ``{sample_id (str): reduced_value (float)}`` for a scalar reduce, or
            ``{sample_id (str): [value, ...]}`` (chronological) for a list reduce;
            empty if the metric is unknown or has no recorded history.
        """
        reduce_l = str(reduce).lower()
        is_list = reduce_l in ("list", "values", "raw", "history")
        agg = {
            "min": "min(value)", "max": "max(value)",
            "mean": "avg(value)", "avg": "avg(value)", "count": "count(value)",
        }.get(reduce_l)
        if is_list:
            # DuckDB collects the ordered time series into a list per sample.
            agg = "list(value ORDER BY step)"
        elif agg is None:
            raise ValueError(
                f"Unsupported reduce '{reduce}'. Use min/max/mean/count/list."
            )

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

        if is_list:
            return {
                str(sid): self._subsample_series(
                    [float(x) for x in (v or [])], max_points
                )
                for (sid, v) in rows if v is not None
            }
        return {str(sid): float(v) for (sid, v) in rows if v is not None}

    def top_k_samples_by_reduce(
        self,
        graph_name: str,
        reduce: str = "max",
        k: int = 5,
        sample_ids=None,
        exp_hash: str | None = None,
        descending: bool = True,
    ) -> list:
        """Top-``k`` samples by a reduction of their per-sample HISTORY,
        ranked entirely inside DuckDB (``GROUP BY sample_id ... ORDER BY
        ... LIMIT k``) — unlike ``reduce_per_sample`` (which returns EVERY
        sample's reduced value as a Python dict), this never materializes
        more than ``k`` rows in Python. Built for reporting/summary use over
        a dataset with millions of samples, where pulling a per-sample dict
        (let alone full history) would be a real memory/latency cost, and
        handing it to an LLM would be a real token cost.

        Args:
            reduce: ``min``/``max``/``mean`` (peak/trough/average over the
                sample's whole history), or ``spread`` (``max - min``, a
                simple instability proxy — how far a sample's loss has swung).
            descending: ``True`` for "worst k" on an ascending-is-better
                metric (e.g. highest peak loss); ``False`` for the other end.

        Returns:
            ``[{"sample_id": str, "value": float}, ...]``, length <= ``k``,
            ordered by ``value``; empty if the metric is unknown or has no
            recorded history.
        """
        agg = {
            "min": "min(value)", "max": "max(value)",
            "mean": "avg(value)", "avg": "avg(value)",
            "spread": "max(value) - min(value)",
        }.get(str(reduce).lower())
        if agg is None:
            raise ValueError(f"Unsupported reduce '{reduce}'. Use min/max/mean/spread.")

        with self._lock:
            self._flush_stage()
            params = [graph_name]
            sql = f"SELECT sample_id, {agg} AS reduced_value FROM per_sample WHERE metric_name = ?"
            sql += self._hash_filter(exp_hash, params)
            if sample_ids is not None:
                sql += " AND sample_id IN (SELECT UNNEST(?))"
                params.append([str(s) for s in sample_ids])
            sql += f" GROUP BY sample_id ORDER BY reduced_value {'DESC' if descending else 'ASC'} LIMIT ?"
            params.append(int(k))
            rows = self._conn.execute(sql, params).fetchall()

        return [{"sample_id": str(sid), "value": float(v)} for (sid, v) in rows if v is not None]

    @staticmethod
    def _subsample_series(values: list, max_points: int | None) -> list:
        """Downsample *values* to at most *max_points* evenly-spaced entries,
        always keeping the first and last (so the curve's shape/endpoints are
        preserved). Returns the list unchanged when no cap applies or it already
        fits."""
        n = len(values)
        if not max_points or max_points <= 0 or n <= max_points:
            return values
        if max_points == 1:
            return [values[-1]]
        step = (n - 1) / (max_points - 1)
        # Set-dedupe guards against rounding collisions when n is only slightly
        # above max_points; result is <= max_points points, endpoints included.
        keep = sorted({int(round(i * step)) for i in range(max_points)})
        return [values[i] for i in keep]

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
            # Snapshots restore state; they must be lossless, so they take the
            # uncapped path rather than the UI default.
            "signal_history": self.get_signal_history(max_points=None),
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
                    outliers=entry.get("outliers") or None,
                    outlier_count=int(entry.get("outlier_count", 0) or 0),
                    sample_count=int(entry.get("sample_count", 0) or 0),
                    trend_value=entry.get("trend_value"),
                    trend_margin=entry.get("trend_margin"),
                    value_min=entry.get("value_min"),
                    value_max=entry.get("value_max"),
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
