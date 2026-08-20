"""Scale/robustness tests for the signal-history read paths.

The shape being defended here is the one a real sweep produces: ~512 runs, each
logging between 3 and 25 signals, each signal running from step 0 to somewhere
between 10k and 1.2M steps, over a dataset of 100k-1M images. That is easily a
table with hundreds of millions of rows, and the property that matters is not
"the query returns the right numbers" but:

    what crosses out of DuckDB into Python must be bounded by what the UI can
    draw, never by what the table holds.

Every test below is written against that invariant. The suite runs at a modest
default size so it stays usable in CI, and scales up through env vars:

    WL_TEST_RUNS=512            runs (experiment hashes)
    WL_TEST_MIN_CURVES=3        signals per run (lower bound)
    WL_TEST_MAX_CURVES=25       signals per run (upper bound)
    WL_TEST_POINTS_PER_CURVE    logged points per curve (row depth)
    WL_TEST_SAMPLES=100000      distinct images in the per-sample table

Rows are synthesised *inside* DuckDB (a spec table cross-joined with range()),
never in a Python loop -- generating 100M rows row-by-row from Python would
take longer than the thing being tested.

Marked `scale`; deselect with `-m "not scale"`.
"""

import os
import random
import time

import pytest

from weightslab.backend.logger import LoggerQueue, _DEFAULT_MAX_POINTS_PER_CURVE

pytestmark = pytest.mark.scale


# ---------------------------------------------------------------------------
# Size knobs
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


N_RUNS = _env_int("WL_TEST_RUNS", 512)
MIN_CURVES = _env_int("WL_TEST_MIN_CURVES", 3)
MAX_CURVES = _env_int("WL_TEST_MAX_CURVES", 25)
# Row depth per curve. The DEFAULT keeps the whole fixture around a couple of
# million rows so the suite stays CI-friendly; the invariants under test are
# depth-independent, and test_output_is_bounded_by_budget_not_table_size proves
# that by measuring across two depths. Push this to ~20000 for a ~100M-row run.
POINTS_PER_CURVE = _env_int("WL_TEST_POINTS_PER_CURVE", 400)
N_SAMPLES = _env_int("WL_TEST_SAMPLES", 100_000)
# Real step extents, independent of how many points are logged along them:
# a curve spans 0 -> somewhere between 10k and 1.2M.
MIN_LAST_STEP = 10_000
MAX_LAST_STEP = 1_200_000

SEED = 20260819


# ---------------------------------------------------------------------------
# Fixture: a sweep-shaped history, generated inside DuckDB
# ---------------------------------------------------------------------------
def _build_history(lg: LoggerQueue, n_runs: int, points_per_curve: int) -> dict:
    """Populate `signals` with a sweep-shaped history. Returns ground truth.

    Ground truth is ``{(metric, run_hash): {"lo", "hi", "count"}}`` -- computed
    from the spec, not by reading back the table, so the assertions have an
    independent reference.
    """
    rng = random.Random(SEED)
    spec = []
    truth = {}
    # Special-row (marker/note/outlier) positions as fractions of the curve
    # depth rather than fixed step-index moduli: a fixed modulus (e.g. "every
    # 500th point") silently generates ZERO special rows whenever
    # points_per_curve is smaller than that modulus, which made
    # test_special_points_survive_decimation pass or fail depending on an
    # unrelated CI-speed knob rather than on real decimation behavior. Each
    # modulus is clamped to >= 2 so it always fires at least once even for a
    # tiny points_per_curve.
    marker_mod = max(2, points_per_curve // 8)
    note_mod = max(2, points_per_curve // 5)
    outlier_mod = max(2, points_per_curve // 3)
    for r in range(n_runs):
        run_hash = f"{r:08x}deadbeefcafe0000"[:24]
        for c in range(rng.randint(MIN_CURVES, MAX_CURVES)):
            metric = f"signal_{c:02d}"
            last_step = rng.randint(MIN_LAST_STEP, MAX_LAST_STEP)
            n_pts = points_per_curve
            # Steps are spread over the curve's real extent, so `step` values
            # look like a run that logged every Nth step rather than 0..N.
            stride = max(1, last_step // max(n_pts - 1, 1))
            spec.append((metric, run_hash, n_pts, stride, last_step))
            truth[(metric, run_hash)] = {
                "lo": 0, "hi": (n_pts - 1) * stride, "count": n_pts,
            }

    conn = lg._conn
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE curve_spec "
        "(metric VARCHAR, h VARCHAR, n_pts BIGINT, stride BIGINT, last_step BIGINT)")
    conn.executemany(
        "INSERT INTO curve_spec VALUES (?, ?, ?, ?, ?)", spec)

    # One set-based INSERT ... SELECT. DuckDB expands the cross product of the
    # spec with range(n_pts) natively; the equivalent Python loop for a 100M-row
    # fixture would dominate the test's runtime.
    conn.execute(
        f"""
        INSERT INTO signals (
            metric_name, experiment_hash, step, metric_value, timestamp,
            audit_mode, is_evaluation_marker, split_name, evaluation_tags,
            point_note, outliers, outlier_count, sample_count,
            trend_value, trend_margin, value_min, value_max, seq)
        SELECT
            cs.metric,
            cs.h,
            (t.i * cs.stride)::INTEGER                       AS step,
            -- a decaying curve with noise, so downsampling has real shape to
            -- preserve rather than a straight line
            (2.3 * exp(-3.0 * t.i / cs.n_pts)
                 + 0.05 * sin(t.i / 7.0))                    AS metric_value,
            1787000000 + t.i                                 AS timestamp,
            FALSE                                            AS audit_mode,
            (t.i > 0 AND t.i % {marker_mod} = 0)             AS is_evaluation_marker,
            CASE WHEN t.i % 2 = 0 THEN 'train' ELSE 'test' END AS split_name,
            '[]'                                             AS evaluation_tags,
            CASE WHEN t.i > 0 AND t.i % {note_mod} = 0
                 THEN 'note @' || t.i ELSE '' END            AS point_note,
            CASE WHEN t.i > 0 AND t.i % {outlier_mod} = 0
                 THEN '[["s1", 9.5]]' ELSE '' END            AS outliers,
            CASE WHEN t.i > 0 AND t.i % {outlier_mod} = 0 THEN 1 ELSE 0 END AS outlier_count,
            32                                               AS sample_count,
            NULL, NULL, NULL, NULL,
            t.i                                              AS seq
        FROM curve_spec cs, range(0, cs.n_pts) AS t(i)
        """
    )
    return truth


def _build_per_sample(lg: LoggerQueue, n_samples: int, n_steps: int = 50) -> None:
    """Per-sample rows for a 100k-1M image dataset, generated in DuckDB."""
    lg._conn.execute(
        """
        INSERT INTO per_sample (metric_name, experiment_hash, sample_id, step, value, seq)
        SELECT 'per_sample_loss',
               'run0',
               (s.i % ?)::VARCHAR,
               ((s.i / ?)::BIGINT * 100)::INTEGER,
               abs(sin(s.i / 13.0)) * 3.0,
               s.i
        FROM range(0, ?) AS s(i)
        """,
        [n_samples, n_samples, n_samples * n_steps],
    )


@pytest.fixture(scope="module")
def big_logger(tmp_path_factory):
    """A sweep-shaped history. Module-scoped: building it is the expensive part."""
    db = tmp_path_factory.mktemp("scale") / "loggers.duckdb"
    lg = LoggerQueue(register=False, db_path=str(db))
    lg.chkpt_manager = None
    t0 = time.monotonic()
    truth = _build_history(lg, N_RUNS, POINTS_PER_CURVE)
    _build_per_sample(lg, N_SAMPLES)
    build_s = time.monotonic() - t0
    n_rows = lg._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_ps = lg._conn.execute("SELECT COUNT(*) FROM per_sample").fetchone()[0]
    print(f"\n[scale fixture] {len(truth)} curves across {N_RUNS} runs | "
          f"{n_rows:,} signal rows | {n_ps:,} per-sample rows | built in {build_s:.1f}s")
    lg.truth = truth
    lg.n_signal_rows = n_rows
    yield lg
    try:
        lg.stop_background_flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The core invariant
# ---------------------------------------------------------------------------
def test_output_is_bounded_by_budget_not_table_size(tmp_path):
    """Doubling the table must NOT change how much comes back.

    This is the whole point of pushing the reduction into SQL. Two histories are
    built at different depths and read with the same budget; if the result grew
    with the table, the read path is still proportional to the DB.
    """
    sizes = []
    for depth in (200, 800):
        db = tmp_path / f"depth_{depth}.duckdb"
        lg = LoggerQueue(register=False, db_path=str(db))
        lg.chkpt_manager = None
        # A small run count keeps this test quick; depth is what varies.
        _build_history(lg, n_runs=8, points_per_curve=depth)
        total = lg._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        hist = lg.get_signal_history_downsampled(max_points=50)
        emitted = sum(len(entries)
                      for per_hash in hist.values()
                      for steps in per_hash.values()
                      for entries in steps.values())
        sizes.append((total, emitted))
        lg.stop_background_flush()

    (rows_small, out_small), (rows_big, out_big) = sizes
    assert rows_big >= rows_small * 3, "fixture did not actually grow"
    # 4x the rows must not produce meaningfully more output. Allow a little
    # slack: special rows (markers/notes/outliers) scale with depth by design.
    assert out_big <= out_small * 1.6, (
        f"output grew with table size: {rows_small:,} rows -> {out_small} points, "
        f"{rows_big:,} rows -> {out_big} points")


def test_every_curve_respects_max_points(big_logger):
    """No curve exceeds its budget, and none is reduced below 3 points."""
    budget = 100
    hist = big_logger.get_signal_history_downsampled(max_points=budget)
    assert hist, "no history returned"

    for metric, per_hash in hist.items():
        for h, steps in per_hash.items():
            n = sum(len(e) for e in steps.values())
            truth_count = big_logger.truth[(metric, h)]["count"]
            # Budget + the special rows that are deliberately exempt.
            assert n <= budget * 3, f"{metric}/{h}: {n} points for budget {budget}"
            assert n >= min(3, truth_count), f"{metric}/{h}: only {n} points"


def test_first_and_last_step_always_survive(big_logger):
    """Endpoints are never decimated away: the x-extent must not move."""
    hist = big_logger.get_signal_history_downsampled(max_points=10)
    for metric, per_hash in hist.items():
        for h, steps in per_hash.items():
            got = sorted(steps.keys())
            expected = big_logger.truth[(metric, h)]
            assert got[0] == expected["lo"], (
                f"{metric}/{h}: first step {got[0]} != {expected['lo']}")
            assert got[-1] == expected["hi"], (
                f"{metric}/{h}: last step {got[-1]} != {expected['hi']}")


def test_curve_index_matches_truth_and_reads_no_points(big_logger):
    """The index reports real extents/counts without returning data points."""
    idx = big_logger.get_signal_curve_index()
    n_curves = sum(len(v) for v in idx.values())
    assert n_curves == len(big_logger.truth)
    for metric, per_hash in idx.items():
        for h, info in per_hash.items():
            expected = big_logger.truth[(metric, h)]
            assert info["first_step"] == expected["lo"]
            assert info["last_step"] == expected["hi"]
            assert info["count"] == expected["count"]


# ---------------------------------------------------------------------------
# Zoom
# ---------------------------------------------------------------------------
def test_zoom_resolves_more_detail_than_the_overview(big_logger):
    """Zooming must ADD points in the visible window, not just restretch.

    This is the regression that server-side decimation would otherwise
    introduce: with a globally decimated curve, zooming into 1% of the range
    leaves ~1% of the points.
    """
    metric, h = next(iter(big_logger.truth))
    extent = big_logger.truth[(metric, h)]
    budget = 200
    lo, hi = extent["lo"], extent["hi"]
    span = hi - lo
    win_lo, win_hi = lo + span // 2, lo + span // 2 + max(1, span // 100)

    overview = big_logger.get_signal_history_downsampled(
        max_points=budget, metric_names=[metric], exp_hashes=[h])
    in_window_before = sum(
        len(entries) for step, entries in overview[metric][h].items()
        if win_lo <= step <= win_hi)

    zoomed = big_logger.get_signal_history_downsampled(
        max_points=budget, metric_names=[metric], exp_hashes=[h],
        x_min=win_lo, x_max=win_hi)
    in_window_after = sum(len(e) for e in zoomed[metric][h].values())

    assert in_window_after > in_window_before, (
        f"zoom returned no extra detail: {in_window_before} -> {in_window_after}")
    for step in zoomed[metric][h]:
        assert win_lo <= step <= win_hi, f"step {step} outside requested window"


def test_scoping_restricts_what_is_read(big_logger):
    """metric/hash scoping must actually narrow the result set."""
    metric, h = next(iter(big_logger.truth))
    scoped = big_logger.get_signal_history_downsampled(
        max_points=50, metric_names=[metric], exp_hashes=[h])
    assert list(scoped.keys()) == [metric]
    assert list(scoped[metric].keys()) == [h]

    all_metrics = big_logger.get_signal_history_downsampled(max_points=50)
    assert len(all_metrics) > 1, "fixture should span several signals"


def test_x_range_window_excludes_everything_outside(big_logger):
    metric, h = next(iter(big_logger.truth))
    hi = big_logger.truth[(metric, h)]["hi"]
    lo_w, hi_w = hi // 4, hi // 2
    hist = big_logger.get_signal_history_downsampled(
        max_points=100, metric_names=[metric], x_min=lo_w, x_max=hi_w)
    for per_hash in hist.values():
        for steps in per_hash.values():
            for step in steps:
                assert lo_w <= step <= hi_w


# ---------------------------------------------------------------------------
# Metadata fidelity -- "current data state (value + metadata)"
# ---------------------------------------------------------------------------
def test_special_points_survive_decimation(big_logger):
    """Markers, annotated points and outlier steps are what a user zooms to find.

    Uniform decimation would drop them at exactly the rate it drops everything
    else; they must be exempt.
    """
    metric, h = next(iter(big_logger.truth))
    hist = big_logger.get_signal_history_downsampled(
        max_points=10, metric_names=[metric], exp_hashes=[h])
    entries = [e for steps in hist[metric][h].values() for e in steps]

    assert any(e["is_evaluation_marker"] for e in entries), \
        "evaluation markers were decimated away"
    assert any(e.get("point_note") for e in entries), \
        "annotated points were decimated away"
    assert any(e.get("outlier_count") for e in entries), \
        "outlier-bearing steps were decimated away"


def test_entries_carry_full_metadata(big_logger):
    """Each rendered point must arrive with the metadata the UI draws with."""
    metric, h = next(iter(big_logger.truth))
    hist = big_logger.get_signal_history_downsampled(
        max_points=25, metric_names=[metric], exp_hashes=[h])
    for steps in hist[metric].values():
        for entries in steps.values():
            for e in entries:
                assert set(e) >= {
                    "model_age", "metric_name", "metric_value", "experiment_hash",
                    "timestamp", "audit_mode", "is_evaluation_marker",
                    "split_name", "evaluation_tags",
                }, f"missing metadata keys: {sorted(e)}"
                assert isinstance(e["metric_value"], float)
                assert e["experiment_hash"] == h
                assert e["metric_name"] == metric


def test_no_duplicate_steps_within_a_curve(big_logger):
    """The three-way UNION must not emit a step twice."""
    hist = big_logger.get_signal_history_downsampled(max_points=64)
    for metric, per_hash in hist.items():
        for h, steps in per_hash.items():
            for step, entries in steps.items():
                keys = [(e["model_age"], e["metric_value"]) for e in entries]
                assert len(keys) == len(set(keys)), \
                    f"{metric}/{h} step {step} duplicated"


# ---------------------------------------------------------------------------
# Per-sample paths (100k-1M images)
# ---------------------------------------------------------------------------
def test_per_sample_aggregate_is_grouped_in_sql(big_logger):
    """Aggregating across a 100k-image dataset returns per-step means, not rows."""
    t0 = time.monotonic()
    agg = big_logger.aggregate_per_sample_by_step("per_sample_loss")
    dt = time.monotonic() - t0
    assert agg, "no aggregate returned"
    # One entry per step, not per (sample, step): the whole point of grouping in
    # DuckDB rather than pulling rows into Python.
    n_ps = big_logger._conn.execute("SELECT COUNT(*) FROM per_sample").fetchone()[0]
    assert len(agg) < n_ps / 10, (
        f"aggregate returned {len(agg)} entries for {n_ps:,} rows -- "
        "looks like row-level data, not a grouped mean")
    print(f"\n[per-sample] aggregate over {n_ps:,} rows in {dt:.2f}s "
          f"-> {len(agg)} steps")


def test_top_k_ranks_without_materialising_the_dataset(big_logger):
    """Top-k over 100k+ images must come back as k rows."""
    k = 10
    t0 = time.monotonic()
    top = big_logger.top_k_samples_by_reduce("per_sample_loss", reduce="max", k=k)
    dt = time.monotonic() - t0
    assert len(top) <= k
    print(f"\n[per-sample] top-{k} in {dt:.2f}s")


def test_per_sample_query_scoped_to_requested_ids(big_logger):
    ids = [str(i) for i in range(50)]
    rows = big_logger.query_per_sample("per_sample_loss", sample_ids=ids)
    assert rows, "no per-sample rows returned"
    assert {r[0] for r in rows} <= set(ids)


# ---------------------------------------------------------------------------
# Frontend call simulation
# ---------------------------------------------------------------------------
def test_simulated_frontend_session_stays_bounded(big_logger):
    """Replay the sequence the UI actually issues, and bound every payload.

    1. curve index at init (axes + curve list, no points)
    2. one decimated read per visible plot
    3. a zoom on one plot

    Each step is asserted against a point budget rather than the table, which is
    what makes the dashboard openable on a history this size.
    """
    PLOT_BUDGET = 800           # ~2 points per CSS pixel on a wide plot
    VISIBLE_PLOTS = 6           # what fits on screen at once

    t0 = time.monotonic()
    index = big_logger.get_signal_curve_index()
    t_index = time.monotonic() - t0
    metrics = sorted(index)[:VISIBLE_PLOTS]
    assert metrics

    total_points = 0
    t0 = time.monotonic()
    for metric in metrics:
        hist = big_logger.get_signal_history_downsampled(
            max_points=PLOT_BUDGET, metric_names=[metric])
        per_metric = sum(len(e)
                         for steps in hist[metric].values()
                         for e in steps.values())
        total_points += per_metric
        n_curves = len(index[metric])
        assert per_metric <= PLOT_BUDGET * n_curves * 3, (
            f"{metric}: {per_metric} points for {n_curves} curves")
    t_initial = time.monotonic() - t0

    # Zoom into the middle 1% of one plot.
    metric = metrics[0]
    h = next(iter(index[metric]))
    hi = index[metric][h]["last_step"]
    t0 = time.monotonic()
    zoom = big_logger.get_signal_history_downsampled(
        max_points=PLOT_BUDGET, metric_names=[metric],
        x_min=hi // 2, x_max=hi // 2 + max(1, hi // 100))
    t_zoom = time.monotonic() - t0
    zoom_points = sum(len(e)
                      for per_hash in zoom.values()
                      for steps in per_hash.values()
                      for e in steps.values())

    print(f"\n[frontend sim] {big_logger.n_signal_rows:,} rows in DB | "
          f"index {t_index:.2f}s ({sum(len(v) for v in index.values())} curves) | "
          f"{VISIBLE_PLOTS} plots {t_initial:.2f}s ({total_points:,} pts) | "
          f"zoom {t_zoom:.2f}s ({zoom_points:,} pts)")

    # The dashboard must never pull a fraction of the table that grows with it.
    assert total_points < big_logger.n_signal_rows, \
        "initial render read as much as the table holds"


def test_history_read_latency_is_acceptable(big_logger):
    """A decimated read must stay interactive at sweep scale."""
    budget_s = float(os.environ.get("WL_TEST_MAX_QUERY_SECONDS", "20"))
    t0 = time.monotonic()
    big_logger.get_signal_history_downsampled(max_points=500)
    dt = time.monotonic() - t0
    print(f"\n[latency] full-dashboard decimated read: {dt:.2f}s "
          f"over {big_logger.n_signal_rows:,} rows")
    assert dt < budget_s, f"decimated read took {dt:.1f}s (budget {budget_s}s)"


def test_default_read_is_capped(tmp_path):
    """get_signal_history() with no arguments must NOT return the whole table.

    The old default did exactly that, which is what made a large history
    unopenable. Guards against a regression to the uncapped default.

    Needs its own fixture rather than ``big_logger``: the cap only bites when
    a curve is deeper than ``_DEFAULT_MAX_POINTS_PER_CURVE``, and
    ``big_logger`` is intentionally shallower than that so the module fixture
    stays CI-fast -- at that depth "default" and "uncapped" read back
    identical either way, which is what let a real regression here go
    unnoticed.
    """
    depth = _DEFAULT_MAX_POINTS_PER_CURVE * 5
    db = tmp_path / "capped.duckdb"
    lg = LoggerQueue(register=False, db_path=str(db))
    lg.chkpt_manager = None
    _build_history(lg, n_runs=2, points_per_curve=depth)
    n_rows = lg._conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    try:
        hist = lg.get_signal_history()
        emitted = sum(len(e)
                      for per_hash in hist.values()
                      for steps in per_hash.values()
                      for e in steps.values())
        assert emitted < n_rows, (
            f"default read returned {emitted:,} of {n_rows:,} rows "
            "-- the default is uncapped again")
    finally:
        lg.stop_background_flush()
