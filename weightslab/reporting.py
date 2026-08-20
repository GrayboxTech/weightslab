"""Experiment health report: signal plots + dataframe stats + logo, as one
self-contained HTML file. Built for the agent's "generate a report" action
(``DataService._agent_generate_experiment_report``) — this module has no
LLM/agent coupling of its own, it only turns already-fetched data (a live
``LoggerQueue`` and the sample dataframe) into a rendered report. The
narrative/conclusion text is written by the agent from ``collect_report_context``'s
output and handed back in via ``narrative=`` when rendering.

``generate_report`` ties the collect → narrate → render sequence together and
is the single code path behind every way a user can ask for a report (the
Studio button / chat action, ``wl.ai_report_generation``, and the CLI console's
``report`` command) — the LLM stays injected as a ``narrative_fn`` callable, so
this module still knows nothing about agents.
"""

import base64
import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from weightslab.data.sample_stats import SampleStatsEx

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_LOGO_PATH = _ASSETS_DIR / "logo.png"

# Brand colors (docs/_static/custom.css .wl-brand-w / .wl-brand-l) — reused
# here to color-code signal health so it reads the same way across the docs
# site and a generated report.
_COLOR_GOOD = "#2d9e3f"
_COLOR_WARN = "#d63333"
_COLOR_NEUTRAL = "#5f7393"

# weightslab.src.LOSS_SHAPES labels considered a healthy training pattern vs.
# a pattern worth flagging. Anything not in either bucket (e.g. too few
# points to classify) is rendered neutral.
_HEALTHY_SHAPES = {"monotonic", "plateaued"}
_CONCERNING_SHAPES = {"Flat_high", "high_variance", "U_Shape", "Spiked", "Forgotten"}

# Distinct per-run palette for the interactive multi-curve plots (see
# _multi_run_series) -- unrelated to _COLOR_GOOD/_COLOR_WARN/_COLOR_NEUTRAL,
# which classify a signal's *health*, not which run a curve belongs to.
# Cycled by index over hashes sorted deterministically, so the same run gets
# the same color across every signal card in one report, and regenerating
# the report from the same data reproduces the same assignment.
_RUN_PALETTE = (
    "#2f6fed", "#e0562f", "#2d9e3f", "#a12d9e", "#0aa3a3",
    "#c99a1e", "#d63333", "#5f7393", "#7a4fd6", "#3f8f6e",
    "#c2528a", "#4f7ad6",
)


def _color_for_run_index(index: int) -> str:
    return _RUN_PALETTE[index % len(_RUN_PALETTE)]


def _multi_run_series(full_history: dict, name: str, runs_map: dict) -> list:
    """Per-run breakdown of one signal, for the report's interactive
    multi-curve chart -- distinct from the single already-aggregated
    trajectory ``get_current_signaL_history`` returns (used for health
    classification / summary stats, unchanged). Each entry is one run's own
    curve: ``{hash, label, color, is_current, points: [[step, value], ...]}``.

    ``full_history`` is ``LoggerQueue.get_signal_history()``'s
    ``{metric: {hash: {step: [entry, ...]}}}`` shape -- fetched ONCE by the
    caller (collect_report_context) and reused across every signal, rather
    than one query per signal. ``runs_map`` (hash -> run info dict, from
    ``CheckpointManager.list_runs()``) supplies the human-readable
    experiment_name for the legend; a hash with no matching run (e.g.
    ``checkpoint_manager`` wasn't available) falls back to a shortened hash.

    A step with more than one logged entry (e.g. re-logged after a resume)
    keeps only the LAST entry's value -- same "last write wins" behavior as
    the rest of this module's step-keyed reads.
    """
    by_hash = full_history.get(name) if isinstance(full_history, dict) else None
    if not by_hash:
        return []

    series = []
    for index, h in enumerate(sorted(by_hash.keys(), key=str)):
        steps_map = by_hash[h] or {}
        points = sorted(
            (
                (step, entries[-1]["metric_value"])
                for step, entries in steps_map.items()
                if entries
            ),
            key=lambda pair: pair[0],
        )
        if not points:
            continue
        run_info = runs_map.get(h) or {}
        label = run_info.get("experiment_name") or (str(h)[:12] if h else "unknown")
        series.append({
            "hash": str(h),
            "label": str(label),
            "color": _color_for_run_index(index),
            "is_current": bool(run_info.get("is_current")),
            "points": [[float(s), float(v)] for s, v in points],
        })
    return series


def _import_matplotlib():
    """Soft import, mirroring notebook_service.py's ``_try_import_matplotlib`` —
    plotting is an optional extra (``pip install weightslab[reporting]``), not
    a hard dependency, so a minimal install never pulls it in unasked."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        logger.info("matplotlib unavailable for experiment report: %s", exc)
        return None


def _load_logo_base64() -> Optional[str]:
    try:
        data = _LOGO_PATH.read_bytes()
    except Exception as exc:
        logger.debug("experiment report: logo not found at %s: %s", _LOGO_PATH, exc)
        return None
    return base64.b64encode(data).decode("ascii")


def select_important_signals(logger_q, max_signals: Optional[int] = None) -> list:
    """Pick which registered signals to plot when the caller didn't ask for
    specific ones. ``max_signals=None`` (the default) means EVERY logged
    signal with enough history goes in the report -- no arbitrary top-N cut.
    When a cap is given, order matters: any signal whose name contains
    "loss" goes first (the single most universally load-bearing metric for
    "how is this training going"), then the rest ordered by how many
    aggregated points they have logged (a sparse/short-lived signal is less
    informative than one tracked across most of the run)."""
    try:
        names = list(logger_q.get_graph_names())
    except Exception:
        return []
    if not names:
        return []

    def _point_count(name: str) -> int:
        try:
            return len(logger_q.get_current_signaL_history(name))
        except Exception:
            return 0

    loss_first = sorted(
        (n for n in names if "loss" in n.lower()),
        key=_point_count, reverse=True,
    )
    rest = sorted(
        (n for n in names if "loss" not in n.lower()),
        key=_point_count, reverse=True,
    )
    ordered = loss_first + rest
    with_enough_history = [n for n in ordered if _point_count(n) >= 2]
    return with_enough_history if max_signals is None else with_enough_history[:max_signals]


def _health_label_and_color(values: list) -> tuple:
    """Classify a signal's trajectory via weightslab.src's built-in shape
    classifier and map the label to a health color. Imported lazily to avoid
    a hard circular import (weightslab.src imports quite a lot at module
    scope; reporting.py should stay importable standalone)."""
    from weightslab.src import classify_loss_shape

    try:
        label = classify_loss_shape(values)
    except Exception:
        label = None
    if label is None:
        return "not enough data", _COLOR_NEUTRAL
    if label in _HEALTHY_SHAPES:
        return label, _COLOR_GOOD
    if label in _CONCERNING_SHAPES:
        return label, _COLOR_WARN
    return label, _COLOR_NEUTRAL


def _render_signal_plot(plt, name: str, points: list, color: str) -> Optional[str]:
    """Render one signal's (step, value) trajectory to a base64 PNG. Returns
    None (never raises) on any rendering failure so one bad signal can't take
    down the whole report.

    Sized for how it's actually displayed: each card is one column of
    ``.wl-report-grid``'s ``repeat(auto-fit, minmax(340px, 1fr))`` (see
    render_report's CSS), so anything past ~340-500 logical px is wasted
    bytes, not visible sharpness. 520x200 @ 100dpi (down from the original
    896x364 @ 140dpi -- a ~3x pixel-area cut) still renders crisp at that
    display width while keeping the embedded-base64 HTML file well behaved
    even with EVERY logged signal included (see select_important_signals's
    max_signals=None default) rather than a top-6 cut.
    """
    try:
        steps = [p["model_age"] for p in points]
        values = [p["metric_value"] for p in points]
        fig, ax = plt.subplots(figsize=(5.2, 2.0), dpi=100)
        ax.plot(steps, values, color=color, linewidth=1.6)
        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("step", fontsize=9)
        ax.set_ylabel("value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("experiment report: failed to render plot for %s: %s", name, exc)
        return None


def _render_histogram_plot(plt, name: str, values: list, color: str) -> Optional[str]:
    """Render a value-distribution histogram for one column's per-sample
    values to a base64 PNG. Mirrors ``_render_signal_plot``'s sizing/failure
    handling (same figure size, same "never raise, return None" contract) so
    a Distributions card looks and behaves like a Signals card."""
    try:
        fig, ax = plt.subplots(figsize=(5.2, 2.0), dpi=100)
        n_bins = min(30, max(5, len(set(values))))
        ax.hist(values, bins=n_bins, color=color, alpha=0.85)
        ax.set_title(name, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("value", fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.25, axis="y")
        fig.tight_layout()

        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        logger.warning("experiment report: failed to render histogram for %s: %s", name, exc)
        return None


def _resolve_distribution_column(df: pd.DataFrame, requested: str) -> Optional[str]:
    """Best-effort match of a user-requested column/signal name (e.g.
    "train_loss") against the sample dataframe's actual columns, which may be
    nested (e.g. "signals//train_loss") -- the same "bare name maps to the
    real column" rule intent_prompt.py's COLUMN RESOLUTION RULES applies for
    the LLM, re-implemented defensively here since ``distributions`` action
    params reach this module directly, without the LLM's own schema-aware
    resolution in between.

    Preference order: exact match, then a column whose ``//``-suffix matches
    case-insensitively, then any column containing the requested name
    case-insensitively. Returns None if nothing matches."""
    if df is None or not requested:
        return None
    requested = str(requested)
    columns = [c for c in df.columns if isinstance(c, str)]
    if requested in columns:
        return requested
    lowered = requested.lower()
    for col in columns:
        if col.rsplit("//", 1)[-1].lower() == lowered:
            return col
    for col in columns:
        if lowered in col.lower():
            return col
    return None


def compute_distribution_entries(
    df: Optional[pd.DataFrame], distributions: Optional[list], plt=None,
) -> list:
    """Per-requested-column distribution summary + histogram, for the
    optional Distributions report section. Unlike ``signals`` (aggregated
    training curves pulled from the logger -- "how did the mean move over
    training"), these are value-distribution histograms over the CURRENT
    per-sample dataframe -- "how spread out is train_loss across samples
    right now". Opt-in only: an empty/``None`` ``distributions`` list (the
    default -- nobody asked for one) returns ``[]``, so a report with no
    distribution request looks exactly like it did before this existed.

    A requested name that can't be resolved to a column, or resolves to a
    column with no numeric values, still gets an entry (flagged
    unresolved/empty) so the report can say so explicitly rather than
    silently dropping what the user asked for -- see
    ``_distribution_card_html``.
    """
    if not distributions or df is None or df.empty:
        return []
    entries = []
    for requested in distributions:
        col = _resolve_distribution_column(df, requested)
        if col is None:
            entries.append({"name": str(requested), "resolved": False})
            continue
        try:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
        except Exception:
            values = pd.Series(dtype=float)
        if values.empty:
            entries.append({"name": str(requested), "column": col, "resolved": True, "n": 0})
            continue
        plot_b64 = (
            _render_histogram_plot(plt, col, values.tolist(), _COLOR_NEUTRAL)
            if plt is not None else None
        )
        entries.append({
            "name": str(requested),
            "column": col,
            "resolved": True,
            "n": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "plot_b64": plot_b64,
        })
    return entries


def compute_dataframe_stats(df: Optional[pd.DataFrame]) -> dict:
    """Summarize the sample dataframe: totals, discard rate, split sizes, and
    tag distributions. Every lookup is defensive (``.get``-style fallbacks)
    because which columns exist depends on the task type and what the
    training script has actually populated so far."""
    if df is None or df.empty:
        return {"total_samples": 0}

    stats: dict = {"total_samples": int(len(df))}

    discarded_col = SampleStatsEx.DISCARDED.value
    if discarded_col in df.columns:
        try:
            discarded = df[discarded_col].astype(bool)
            stats["discarded_count"] = int(discarded.sum())
            stats["discarded_pct"] = round(100.0 * discarded.mean(), 2)
        except Exception:
            pass

    origin_col = SampleStatsEx.ORIGIN.value
    if origin_col in df.columns:
        try:
            stats["splits"] = {
                str(k): int(v) for k, v in df[origin_col].value_counts().items()
            }
        except Exception:
            pass
    elif isinstance(df.index, pd.MultiIndex) and origin_col in (df.index.names or []):
        try:
            counts = df.index.get_level_values(origin_col).value_counts()
            stats["splits"] = {str(k): int(v) for k, v in counts.items()}
        except Exception:
            pass

    # Loss-shape classification tags get their own section (summarize_loss_shape_tags)
    # with health coloring + bounded concrete examples -- skip them here so they
    # aren't duplicated as a plain, uncolored, head(5)-truncated tag breakdown.
    tag_cols = [
        c for c in df.columns
        if isinstance(c, str) and c.startswith("tag:") and "shape" not in c.lower()
    ]
    tags: dict = {}
    for col in tag_cols:
        try:
            series = df[col]
            if series.dtype == bool:
                tags[col[4:]] = {"true_count": int(series.sum())}
            else:
                counts = series.value_counts(dropna=True).head(5)
                tags[col[4:]] = {str(k): int(v) for k, v in counts.items()}
        except Exception:
            continue
    if tags:
        stats["tags"] = tags

    return stats


def _sample_ids_for_mask(df: pd.DataFrame, mask: pd.Series, max_examples: int) -> list:
    """Up to ``max_examples`` sample_ids where ``mask`` is True. ``sample_id``
    may be a column or an index level (see the same handling in
    ``DataService._reduce_signal_history_series``) -- bounded output
    regardless of how many rows match."""
    sample_col = SampleStatsEx.SAMPLE_ID.value
    try:
        if sample_col in df.columns:
            ids = df.loc[mask, sample_col]
        elif isinstance(df.index, pd.MultiIndex) and sample_col in (df.index.names or []):
            ids = df.index.get_level_values(sample_col)[mask.to_numpy()]
        else:
            ids = df.index[mask.to_numpy()]
        return [str(x) for x in list(ids)[:max_examples]]
    except Exception:
        return []


def summarize_loss_shape_tags(df: Optional[pd.DataFrame], max_examples: int = 3) -> list:
    """Bounded summary of any loss-shape classification ALREADY COMPUTED on
    the dataframe (via ``write_signal_shapes``/``write_loss_shapes``/the
    background auto-tagger — see :doc:`logger`) — discovered by column name
    (``tag:*shape*``), never by re-running classification here.

    Output size is O(distinct labels, <= 7) + O(max_examples per concerning
    label) — NOT O(samples). Safe to hand to an LLM regardless of whether the
    dataset has a hundred samples or ten million: this never enumerates or
    returns every sample's row, only a per-label count and a small handful of
    concrete ids for whichever labels are concerning.
    """
    if df is None or df.empty:
        return []

    shape_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("tag:") and "shape" in c.lower()]
    results = []
    for col in shape_cols:
        try:
            series = df[col].dropna()
        except Exception:
            continue
        if series.empty:
            continue
        counts = series.value_counts()
        examples: dict = {}
        for label, count in counts.items():
            if str(label) in _CONCERNING_SHAPES:
                ids = _sample_ids_for_mask(df, series == label, max_examples)
                if ids:
                    examples[str(label)] = ids
        results.append({
            "tag": col[4:],
            "counts": {str(k): int(v) for k, v in counts.items()},
            "concerning_examples": examples,
        })
    return results


def find_signal_outliers(logger_q, signal_name: str, top_k: int = 5) -> dict:
    """Bounded (O(top_k), never O(samples)) per-sample outlier lookup for one
    signal: the samples with the highest logged peak, and the samples whose
    history swung the most (``max - min``, a cheap instability proxy). Both
    rankings happen inside DuckDB (``LoggerQueue.top_k_samples_by_reduce``) --
    nothing beyond the top few rows ever leaves the database, so this stays
    just as cheap whether the signal has a thousand or ten million samples'
    worth of per-sample history logged.
    """
    if not hasattr(logger_q, "top_k_samples_by_reduce"):
        return {}
    out: dict = {}
    try:
        peaks = logger_q.top_k_samples_by_reduce(signal_name, reduce="max", k=top_k, descending=True)
        if peaks:
            out["highest_peak"] = peaks
    except Exception as exc:
        logger.debug("experiment report: peak outlier lookup failed for %s: %s", signal_name, exc)
    try:
        unstable = logger_q.top_k_samples_by_reduce(signal_name, reduce="spread", k=top_k, descending=True)
        if unstable:
            out["most_unstable"] = unstable
    except Exception as exc:
        logger.debug("experiment report: instability outlier lookup failed for %s: %s", signal_name, exc)
    return out


def _resolve_runs_map(checkpoint_manager) -> dict:
    """``{hash: run_info}`` from ``CheckpointManager.list_runs()`` (see
    checkpoint_manager.py), or ``{}`` when unavailable/erroring -- the report
    degrades to hash-only labels rather than failing to generate."""
    if checkpoint_manager is None or not hasattr(checkpoint_manager, "list_runs"):
        return {}
    try:
        return {r["hash"]: r for r in checkpoint_manager.list_runs() if r.get("hash")}
    except Exception as exc:
        logger.debug("experiment report: could not list runs: %s", exc)
        return {}


def collect_report_context(
    root_log_dir,
    logger_q,
    df: Optional[pd.DataFrame],
    signals: Optional[list] = None,
    max_signals: Optional[int] = None,
    distributions: Optional[list] = None,
    checkpoint_manager=None,
) -> dict:
    """Gather everything a report needs EXCEPT the narrative: per-signal
    trajectories + health classification + plots, dataframe-level stats, and
    (opt-in) per-column value-distribution histograms. ``max_signals=None``
    (default) includes every logged signal with enough history -- pass an int
    to cap it (see select_important_signals). ``distributions`` names columns
    to additionally render as histograms (see compute_distribution_entries);
    omitted/empty means no Distributions section at all, e.g. "add a
    histogram of train_loss to the report" -> ``distributions=["train_loss"]``
    on a follow-up call.

    Split out from ``render_report`` so the caller (the agent) can hand the
    returned, LLM-friendly-sized ``context["signals"]``/``context["dataframe"]``
    summary to an LLM call for the narrative section, then pass the whole
    context (plus that narrative) to ``render_report`` — one data pass, one
    render pass, no wasted duplicate work.
    """
    plt = _import_matplotlib()
    resolved_signals = signals or select_important_signals(logger_q, max_signals=max_signals)

    runs_map = _resolve_runs_map(checkpoint_manager)
    try:
        full_history = logger_q.get_signal_history()
    except Exception as exc:
        logger.warning("experiment report: could not read multi-run history: %s", exc)
        full_history = {}

    signal_entries = []
    for name in resolved_signals:
        try:
            points = logger_q.get_current_signaL_history(name)
        except Exception as exc:
            logger.warning("experiment report: could not read history for %s: %s", name, exc)
            continue
        if len(points) < 2:
            continue
        values = [p["metric_value"] for p in points]
        label, color = _health_label_and_color(values)
        plot_b64 = _render_signal_plot(plt, name, points, color) if plt is not None else None
        # Per-sample view of the SAME signal, bounded to a handful of extreme
        # samples (find_signal_outliers) rather than every sample's history --
        # this is what lets the report speak to per-sample trends without its
        # size (or an LLM prompt built from it) scaling with dataset size.
        outliers = find_signal_outliers(logger_q, name, top_k=5)
        # Per-run breakdown for the report's interactive multi-curve chart --
        # additive alongside the aggregate stats above (first/last/min/max,
        # health classification, plot_b64), which stay computed from the
        # SAME aggregate they always were so nothing downstream (narrative
        # summary, existing tests) sees a behavior change.
        series = _multi_run_series(full_history, name, runs_map)
        signal_entries.append({
            "name": name,
            "label": label,
            "color": color,
            "n_points": len(points),
            "first_value": values[0],
            "last_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
            "plot_b64": plot_b64,
            "outliers": outliers,
            "series": series,
        })

    return {
        "root_log_dir": str(root_log_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signals": signal_entries,
        "distributions": compute_distribution_entries(df, distributions, plt),
        "dataframe": compute_dataframe_stats(df),
        "loss_shape_tags": summarize_loss_shape_tags(df),
        "plotting_available": plt is not None,
        "runs": list(runs_map.values()),
    }


def _fmt_num(x) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


_BLOCK_TOOLBAR_HTML = (
    '<div class="wl-block-toolbar">'
    '<button type="button" class="wl-block-btn" data-action="move-up" title="Move up">&#8593;</button>'
    '<button type="button" class="wl-block-btn" data-action="move-down" title="Move down">&#8595;</button>'
    '{expand_btn}'
    '<button type="button" class="wl-block-btn wl-block-btn-danger" data-action="remove" title="Remove from report">&times;</button>'
    '</div>'
)
_EXPAND_BTN_HTML = '<button type="button" class="wl-block-btn" data-action="expand" title="Expand &amp; zoom">&#10530;</button>'


def _signal_card_html(entry: dict, block_id: str) -> str:
    """One Signals card as an editable/reorderable/removable ``.wl-block``.

    When ``entry["series"]`` has data (see ``_multi_run_series``), the card
    embeds that per-run series as JSON for the page's script to render as an
    interactive, multi-curve, legend-carrying Chart.js line chart (one color
    per run, matching the live Studio app's per-run color scheme) -- the
    ``plot_b64`` matplotlib image still renders too, but only as the
    ``<noscript>`` fallback for a reader with JavaScript disabled or a report
    opened somewhere Chart.js failed to load. No JS/no data -> the old static
    image (or the textual summary) is what actually shows.
    """
    name = html.escape(str(entry["name"]))
    label = html.escape(str(entry["label"]))
    color = entry["color"]
    series = entry.get("series") or []
    has_interactive = bool(series)

    toolbar = _BLOCK_TOOLBAR_HTML.format(expand_btn=_EXPAND_BTN_HTML if has_interactive else "")
    body = (
        f'<div class="wl-block wl-plot-block" data-block-id="{block_id}">'
        f'{toolbar}'
        f'<div class="wl-report-card">'
        f'  <div class="wl-report-card-head">'
        f'    <span class="wl-report-signal-name" contenteditable="true" spellcheck="false">{name}</span>'
        f'    <span class="wl-report-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;">{label}</span>'
        f'  </div>'
    )
    if has_interactive:
        canvas_id = f"{block_id}-canvas"
        noscript_img = (
            f'<img class="wl-report-plot" src="data:image/png;base64,{entry["plot_b64"]}" alt="{name} trajectory" />'
            if entry.get("plot_b64") else ""
        )
        chart_payload = json.dumps({"name": str(entry["name"]), "series": series})
        body += (
            f'  <div class="wl-report-plot-wrap wl-chart-wrap">'
            f'<canvas class="wl-report-chart" id="{canvas_id}"></canvas>'
            f'<noscript>{noscript_img}</noscript>'
            f'</div>'
            f'<script type="application/json" class="wl-plot-data">{chart_payload}</script>'
        )
    elif entry.get("plot_b64"):
        body += (
            f'  <div class="wl-report-plot-wrap">'
            f'<img class="wl-report-plot" src="data:image/png;base64,{entry["plot_b64"]}" alt="{name} trajectory" />'
            f'</div>'
        )
    else:
        body += (
            '  <div class="wl-report-noplot">'
            f'Points: {entry["n_points"]} · first {_fmt_num(entry["first_value"])} '
            f'&rarr; last {_fmt_num(entry["last_value"])} '
            f'(min {_fmt_num(entry["min_value"])}, max {_fmt_num(entry["max_value"])})'
            '</div>'
        )
    body += _outliers_html(entry.get("outliers") or {})
    body += "</div></div>"
    return body


def _outliers_html(outliers: dict) -> str:
    """Per-signal outlier callout: a handful of concrete sample_ids (never
    the whole dataset) for whichever extremes exist. Empty string when there
    is nothing to show (e.g. no per-sample history logged for this signal)."""
    if not outliers:
        return ""
    parts = []
    if outliers.get("highest_peak"):
        items = ", ".join(
            f'{html.escape(o["sample_id"])} ({_fmt_num(o["value"])})' for o in outliers["highest_peak"][:5]
        )
        parts.append(f"<li>Highest peak — {items}</li>")
    if outliers.get("most_unstable"):
        items = ", ".join(
            f'{html.escape(o["sample_id"])} (&Delta;{_fmt_num(o["value"])})' for o in outliers["most_unstable"][:5]
        )
        parts.append(f"<li>Most unstable (max&minus;min) — {items}</li>")
    if not parts:
        return ""
    return (
        '<div class="wl-report-outliers">'
        '<span class="wl-report-subhead">Per-sample outliers</span>'
        f'<ul class="wl-report-list">{"".join(parts)}</ul>'
        '</div>'
    )


def _distribution_card_html(entry: dict, block_id: str) -> str:
    """One Distributions card: a histogram + n/mean/std/range, or an
    explanatory note in place of the plot when the requested name couldn't be
    resolved to a column or the column had no numeric values -- so an
    unresolvable request still shows up in the report saying so, instead of
    the whole Distributions section silently rendering one card short.

    A static image (no per-run breakdown makes sense for a value
    distribution), so "expand" here just opens it larger in the same modal
    the plot cards use (via ``data-expand-image``, not ``data-expand`` --
    see the report's embedded script), rather than a Chart.js re-render.
    """
    name = html.escape(str(entry["name"]))
    has_image = bool(entry.get("resolved") and entry.get("n") and entry.get("plot_b64"))
    toolbar = _BLOCK_TOOLBAR_HTML.format(expand_btn=_EXPAND_BTN_HTML if has_image else "")
    head = (
        f'<div class="wl-block wl-plot-block" data-block-id="{block_id}">'
        f'{toolbar}'
        '<div class="wl-report-card">'
        '  <div class="wl-report-card-head">'
        f'    <span class="wl-report-signal-name" contenteditable="true" spellcheck="false">{name}</span>'
        '  </div>'
    )
    if not entry.get("resolved"):
        return head + (
            f'  <div class="wl-report-noplot">No column matching &quot;{name}&quot; '
            'was found in the dataset.</div></div></div>'
        )
    if not entry.get("n"):
        return head + (
            '  <div class="wl-report-noplot">No numeric values logged for this column yet.</div></div></div>'
        )
    body = head
    if entry.get("plot_b64"):
        img_data_url = f'data:image/png;base64,{entry["plot_b64"]}'
        body += (
            f'  <div class="wl-report-plot-wrap">'
            f'<img class="wl-report-plot" data-expand-image="{img_data_url}" '
            f'data-expand-title="{name}" src="{img_data_url}" alt="{name} distribution" />'
            f'</div>'
        )
    body += (
        f'  <div class="wl-report-noplot">n={entry["n"]:,} &middot; mean {_fmt_num(entry["mean"])} '
        f'&middot; std {_fmt_num(entry["std"])} &middot; range [{_fmt_num(entry["min"])}, {_fmt_num(entry["max"])}]</div>'
    )
    body += "</div></div>"
    return body


def _distributions_section_html(distributions: list) -> str:
    """The optional Distributions section, e.g. "add a histogram of
    train_loss" (action_params={"distributions": ["train_loss"]}) -- omitted
    ENTIRELY (not even an empty placeholder) when nobody asked for one, so a
    report generated without that request renders exactly as it did before
    this feature existed."""
    if not distributions:
        return ""
    cards = "".join(
        _distribution_card_html(e, f"wl-dist-{i}") for i, e in enumerate(distributions)
    )
    return (
        '<div class="wl-report-section">'
        '<h2>Distributions</h2>'
        f'<div class="wl-report-grid">{cards}</div>'
        '</div>'
    )


def _loss_shape_section_html(loss_shape_tags: list) -> str:
    """Render any already-computed loss-shape classification tags found on
    the dataframe: a color-coded count per label, plus a handful of concrete
    sample_ids for whichever labels are concerning. Bounded by
    summarize_loss_shape_tags (O(labels) + O(examples)), not by sample count."""
    if not loss_shape_tags:
        return (
            '<p class="wl-report-muted">No loss-shape classification found on this dataset yet — '
            'see <code>wl.write_loss_shapes</code> / <code>wl.enable_loss_shape_autotag</code>.</p>'
        )
    blocks = []
    for entry in loss_shape_tags:
        tag = html.escape(entry["tag"])
        badges = []
        for label, count in entry["counts"].items():
            if label in _HEALTHY_SHAPES:
                color = _COLOR_GOOD
            elif label in _CONCERNING_SHAPES:
                color = _COLOR_WARN
            else:
                color = _COLOR_NEUTRAL
            badges.append(
                f'<span class="wl-report-badge" style="background:{color}22;color:{color};'
                f'border:1px solid {color}55;">{html.escape(str(label))}: {count:,}</span>'
            )
        block = f'<div class="wl-report-shape-block"><span class="wl-report-subhead">{tag}</span>' \
                f'<div class="wl-report-badge-row">{"".join(badges)}</div>'
        if entry.get("concerning_examples"):
            ex_rows = "".join(
                f"<li><b>{html.escape(label)}</b> — {', '.join(html.escape(i) for i in ids)}</li>"
                for label, ids in entry["concerning_examples"].items()
            )
            block += f'<ul class="wl-report-list">{ex_rows}</ul>'
        block += "</div>"
        blocks.append(block)
    return "".join(blocks)


def _dataframe_section_html(stats: dict) -> str:
    if not stats or not stats.get("total_samples"):
        return '<p class="wl-report-muted">No sample dataframe data available yet.</p>'

    rows = [f"<li><b>{stats['total_samples']:,}</b> total samples</li>"]
    if "discarded_count" in stats:
        rows.append(
            f"<li><b>{stats['discarded_count']:,}</b> discarded "
            f"({stats.get('discarded_pct', 0)}%)</li>"
        )
    if stats.get("splits"):
        parts = ", ".join(f"{html.escape(k)}: {v:,}" for k, v in stats["splits"].items())
        rows.append(f"<li>Splits — {parts}</li>")
    html_out = "<ul class=\"wl-report-list\">" + "".join(rows) + "</ul>"

    if stats.get("tags"):
        tag_rows = []
        for tag_name, counts in stats["tags"].items():
            parts = ", ".join(f"{html.escape(str(k))}: {v:,}" for k, v in counts.items())
            tag_rows.append(f"<li><b>{html.escape(tag_name)}</b> — {parts}</li>")
        html_out += "<p class=\"wl-report-subhead\">Tags</p><ul class=\"wl-report-list\">" + "".join(tag_rows) + "</ul>"
    return html_out


def _runs_section_html(runs: list) -> str:
    """Runs table: every run this experiment's manifest knows about (see
    ``CheckpointManager.list_runs``), each removable from the report via its
    own row ``&times;`` (client-side only, like every other block edit here).
    Omitted entirely when no ``checkpoint_manager`` was available to
    ``collect_report_context`` -- same "don't render a feature nobody asked
    for/nothing to show" convention as the Distributions section."""
    if not runs:
        return ""

    def _row(run: dict) -> str:
        run_hash = html.escape(str(run.get("hash", "")))
        name = html.escape(str(run.get("experiment_name") or run_hash[:12] or "unknown"))
        notes = html.escape(str(run.get("notes") or ""))
        created = html.escape(str(run.get("created") or ""))
        last_used = html.escape(str(run.get("last_used") or ""))
        current_badge = (
            f'<span class="wl-report-badge" style="background:{_COLOR_GOOD}22;color:{_COLOR_GOOD};'
            f'border:1px solid {_COLOR_GOOD}55;">current</span>'
            if run.get("is_current") else ""
        )
        return (
            f'<tr data-run-hash="{run_hash}">'
            f'<td><button type="button" class="wl-block-btn wl-block-btn-danger" '
            f'data-action="remove-row" title="Remove this run from the report">&times;</button></td>'
            f'<td>{name} {current_badge}</td>'
            f'<td class="wl-mono">{run_hash}</td>'
            f'<td>{notes}</td>'
            f'<td>{created}</td>'
            f'<td>{last_used}</td>'
            f'</tr>'
        )

    rows_html = "".join(_row(r) for r in runs)
    return (
        '<div class="wl-block wl-runs-block" data-block-id="wl-runs">'
        '<div class="wl-report-card">'
        '<table class="wl-report-runs-table">'
        '<thead><tr><th></th><th>Name</th><th>Hash</th><th>Notes</th><th>Created</th><th>Last used</th></tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        '</div>'
        '</div>'
    )


_CHARTJS_PATH = _ASSETS_DIR / "chart.umd.min.js"


def _chartjs_script_tag() -> str:
    """Inline ``<script>`` for the vendored Chart.js UMD build (MIT-licensed,
    same version weights_studio itself uses) -- embedded rather than loaded
    from a CDN so the report stays a genuinely self-contained file, openable
    offline exactly like its base64-embedded images. Empty string (report
    falls back to the ``<noscript>`` static images) if the asset is missing
    from this install for some reason."""
    try:
        source = _CHARTJS_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        logger.info("experiment report: Chart.js asset unavailable, falling back to static plots: %s", exc)
        return ""
    return f"<script>{source}</script>"


# Not passed through _HTML_TEMPLATE.format() as part of the template itself --
# inserted as one substituted VALUE (see render_report), so its own liberal
# use of { and } needs no doubling the way the template's other <script>
# blocks do.
_INTERACTIVE_JS = r"""
<script>
(function () {
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
    });
  }

  // ---- Generic block toolbar: move up/down, remove (any block type) ----
  function moveBlock(block, dir) {
    var sibling = dir === "up" ? block.previousElementSibling : block.nextElementSibling;
    if (!sibling || !sibling.classList || !sibling.classList.contains("wl-block")) return;
    if (dir === "up") block.parentNode.insertBefore(block, sibling);
    else block.parentNode.insertBefore(sibling, block);
  }
  function removeBlock(block) {
    if (!window.confirm("Remove this from the report?")) return;
    var canvas = block.querySelector("canvas");
    if (canvas && chartsById[canvas.id]) { chartsById[canvas.id].destroy(); delete chartsById[canvas.id]; }
    block.remove();
  }

  // ---- Chart.js rendering (one dataset per run -- real per-run color + legend) ----
  var chartsById = {};

  function buildChart(canvas, data) {
    var datasets = (data.series || []).map(function (s) {
      return {
        label: s.label,
        borderColor: s.color,
        backgroundColor: s.color,
        pointRadius: 0,
        pointHoverRadius: 3,
        borderWidth: s.is_current ? 2.4 : 1.4,
        data: (s.points || []).map(function (p) { return { x: p[0], y: p[1] }; }),
        parsing: false,
      };
    });
    var chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets: datasets },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        plugins: {
          legend: { display: datasets.length > 1, position: "bottom", labels: { boxWidth: 10, font: { size: 10 } } },
          tooltip: { enabled: true },
        },
        scales: {
          x: { type: "linear", title: { display: true, text: "step", font: { size: 10 } }, ticks: { font: { size: 9 } } },
          y: { title: { display: true, text: "value", font: { size: 10 } }, ticks: { font: { size: 9 } } },
        },
      },
    });
    chartsById[canvas.id] = chart;
    return chart;
  }

  // ---- Zoom: drag a rectangle over the canvas to zoom the step (x) range;
  // double-click resets. This is what "zoom in for the final PDF export"
  // means in this report -- the zoomed range is just the chart's current
  // state, which is exactly what window.print() rasterizes. ----
  function wireZoom(canvas, chart) {
    var dragging = false, startX = 0;
    var wrap = canvas.parentElement;
    var overlay = document.createElement("div");
    overlay.className = "wl-zoom-overlay";
    wrap.appendChild(overlay);

    canvas.addEventListener("mousedown", function (e) {
      dragging = true;
      startX = e.offsetX;
      overlay.style.display = "block";
      overlay.style.left = startX + "px";
      overlay.style.width = "0px";
    });
    canvas.addEventListener("mousemove", function (e) {
      if (!dragging) return;
      var x = e.offsetX;
      overlay.style.left = Math.min(startX, x) + "px";
      overlay.style.width = Math.abs(x - startX) + "px";
    });
    function endDrag(e) {
      if (!dragging) return;
      dragging = false;
      overlay.style.display = "none";
      var endX = e.offsetX;
      if (Math.abs(endX - startX) < 6) return; // a click, not a drag -- don't zoom
      var xScale = chart.scales.x;
      if (!xScale) return;
      var v1 = xScale.getValueForPixel(startX);
      var v2 = xScale.getValueForPixel(endX);
      chart.options.scales.x.min = Math.min(v1, v2);
      chart.options.scales.x.max = Math.max(v1, v2);
      chart.update();
    }
    canvas.addEventListener("mouseup", endDrag);
    canvas.addEventListener("mouseleave", function () { dragging = false; overlay.style.display = "none"; });
    canvas.addEventListener("dblclick", function () {
      delete chart.options.scales.x.min;
      delete chart.options.scales.x.max;
      chart.update();
    });
  }

  // ---- Expand modal: chart re-rendered bigger, same zoom interaction; or a
  // plain enlarged <img> for the static distribution histograms. ----
  var expandOverlay = null;
  function closeExpand() {
    if (!expandOverlay) return;
    var canvas = expandOverlay.querySelector("canvas");
    if (canvas && chartsById[canvas.id]) { chartsById[canvas.id].destroy(); delete chartsById[canvas.id]; }
    expandOverlay.remove();
    expandOverlay = null;
  }
  function openExpandShell(title) {
    closeExpand();
    expandOverlay = document.createElement("div");
    expandOverlay.className = "wl-expand-overlay";
    var panel = document.createElement("div");
    panel.className = "wl-expand-panel";
    var head = document.createElement("div");
    head.className = "wl-expand-head";
    head.innerHTML = "<span>" + escapeHtml(title || "") + "</span>"
      + '<button type="button" class="wl-block-btn" data-expand-reset title="Reset zoom">&#8635;</button>'
      + '<button type="button" class="wl-block-btn" data-expand-close title="Close (Esc)">&times;</button>';
    var body = document.createElement("div");
    body.className = "wl-expand-body";
    panel.appendChild(head);
    panel.appendChild(body);
    expandOverlay.appendChild(panel);
    document.body.appendChild(expandOverlay);
    head.querySelector("[data-expand-close]").addEventListener("click", closeExpand);
    expandOverlay.addEventListener("click", function (e) { if (e.target === expandOverlay) closeExpand(); });
    return { head: head, body: body };
  }
  function expandChart(data) {
    var shell = openExpandShell(data.name);
    var canvas = document.createElement("canvas");
    canvas.id = "wl-expand-canvas-" + Date.now();
    shell.body.appendChild(canvas);
    var chart = buildChart(canvas, data);
    wireZoom(canvas, chart);
    shell.head.querySelector("[data-expand-reset]").addEventListener("click", function () {
      delete chart.options.scales.x.min;
      delete chart.options.scales.x.max;
      chart.update();
    });
  }
  function expandImage(src, title) {
    var shell = openExpandShell(title);
    shell.head.querySelector("[data-expand-reset]").style.display = "none";
    var img = document.createElement("img");
    img.src = src;
    img.alt = title || "";
    shell.body.appendChild(img);
  }
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeExpand(); });

  // ---- Wire every embedded plot block ----
  function initPlots() {
    var hasChart = typeof Chart !== "undefined";
    document.querySelectorAll(".wl-plot-block").forEach(function (block) {
      var dataEl = block.querySelector(".wl-plot-data");
      var canvas = block.querySelector("canvas.wl-report-chart");
      if (!dataEl || !canvas || !hasChart) return;
      var data;
      try { data = JSON.parse(dataEl.textContent); } catch (e) { return; }
      var chart = buildChart(canvas, data);
      wireZoom(canvas, chart);
      var expandBtn = block.querySelector('[data-action="expand"]');
      if (expandBtn) expandBtn.addEventListener("click", function () { expandChart(data); });
    });
    // Distribution cards: static image, "expand" just opens it larger.
    document.querySelectorAll(".wl-plot-block [data-action=\"expand\"]").forEach(function (btn) {
      var block = btn.closest(".wl-plot-block");
      if (block.querySelector(".wl-plot-data")) return; // already handled above (chart block)
      var img = block.querySelector("img[data-expand-image]");
      if (!img) return;
      btn.addEventListener("click", function () {
        expandImage(img.getAttribute("data-expand-image"), img.getAttribute("data-expand-title"));
      });
    });
  }

  // ---- Delegated toolbar clicks: move/remove (every block), remove-row (runs table) ----
  function initToolbarDelegation() {
    document.addEventListener("click", function (e) {
      var moveBtn = e.target.closest('.wl-block-btn[data-action="move-up"], .wl-block-btn[data-action="move-down"]');
      if (moveBtn) {
        var block = moveBtn.closest(".wl-block");
        if (block) moveBlock(block, moveBtn.dataset.action === "move-up" ? "up" : "down");
        return;
      }
      var removeBtn = e.target.closest('.wl-block-btn[data-action="remove"]');
      if (removeBtn) {
        var rBlock = removeBtn.closest(".wl-block");
        if (rBlock) removeBlock(rBlock);
        return;
      }
      var rowBtn = e.target.closest('[data-action="remove-row"]');
      if (rowBtn) {
        var row = rowBtn.closest("tr");
        if (row && window.confirm("Remove this run from the report?")) row.remove();
      }
    });
  }

  // ---- "+ Title" / "+ Text" -- freeform, reorderable/removable blocks the
  // user can add anywhere in the #wl-freeform-blocks area. ----
  function makeTextBlock(kind, placeholder) {
    var block = document.createElement("div");
    block.className = "wl-block wl-text-block";
    block.innerHTML =
      '<div class="wl-block-toolbar">'
      + '<button type="button" class="wl-block-btn" data-action="move-up" title="Move up">&#8593;</button>'
      + '<button type="button" class="wl-block-btn" data-action="move-down" title="Move down">&#8595;</button>'
      + '<button type="button" class="wl-block-btn wl-block-btn-danger" data-action="remove" title="Remove">&times;</button>'
      + "</div>";
    var content = document.createElement(kind === "title" ? "h2" : "p");
    content.className = kind === "title" ? "wl-report-user-title" : "wl-report-user-text";
    content.contentEditable = "true";
    content.spellcheck = false;
    content.textContent = placeholder;
    content.addEventListener("focus", function onFocus() {
      if (content.textContent === placeholder) content.textContent = "";
      content.removeEventListener("focus", onFocus);
    });
    block.appendChild(content);
    return block;
  }
  function initAddButtons() {
    var host = document.getElementById("wl-freeform-blocks");
    var addTitleBtn = document.getElementById("wl-add-title-btn");
    var addTextBtn = document.getElementById("wl-add-text-btn");
    if (addTitleBtn) addTitleBtn.addEventListener("click", function () {
      var block = makeTextBlock("title", "New section title");
      host.appendChild(block);
      block.scrollIntoView({ behavior: "smooth", block: "center" });
      block.querySelector("[contenteditable]").focus();
    });
    if (addTextBtn) addTextBtn.addEventListener("click", function () {
      var block = makeTextBlock("text", "Click to write a note…");
      host.appendChild(block);
      block.scrollIntoView({ behavior: "smooth", block: "center" });
      block.querySelector("[contenteditable]").focus();
    });
  }

  function initPdfExport() {
    var btn = document.getElementById("wl-export-pdf-btn");
    if (btn) btn.addEventListener("click", function () { window.print(); });
  }

  function boot() {
    initPlots();
    initToolbarDelegation();
    initAddButtons();
    initPdfExport();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
</script>
"""


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<script>
  // Applied before first paint (head, synchronous) to avoid a flash of the
  // wrong theme on load -- restores an explicit choice from a previous visit
  // to THIS report file (localStorage is per-origin+path for a file:// URL,
  // so this is scoped to this exact report). No stored choice means "follow
  // prefers-color-scheme", handled entirely by the CSS below -- no attribute
  // is set in that case.
  (function () {{
    try {{
      var saved = localStorage.getItem("wl-report-theme");
      if (saved === "light" || saved === "dark") {{
        document.documentElement.setAttribute("data-theme", saved);
      }}
    }} catch (e) {{}}
  }})();
</script>
<style>
  :root {{
    --wl-good: {color_good};
    --wl-warn: {color_warn};
    --wl-neutral: {color_neutral};
    --wl-bg: #ffffff;
    --wl-bg-elevated: #f7f9fb;
    --wl-banner-a: #f7f9fb;
    --wl-banner-b: #eef3f8;
    --wl-fg: #1a1a1a;
    --wl-fg-muted: #5a6270;
    --wl-fg-subtle: #888888;
    --wl-border: #e5e5e5;
    --wl-card-bg: #ffffff;
  }}
  /* Follows the OS/browser theme when the reader hasn't picked one via the
     toggle button -- [data-theme] (set only by the toggle) always wins. */
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --wl-bg: #12161d;
      --wl-bg-elevated: #1a202b;
      --wl-banner-a: #1a202b;
      --wl-banner-b: #151a22;
      --wl-fg: #e8ecf2;
      --wl-fg-muted: #a7b0bf;
      --wl-fg-subtle: #838d9c;
      --wl-border: #2b3341;
      --wl-card-bg: #1a202b;
    }}
  }}
  :root[data-theme="dark"] {{
    --wl-bg: #12161d;
    --wl-bg-elevated: #1a202b;
    --wl-banner-a: #1a202b;
    --wl-banner-b: #151a22;
    --wl-fg: #e8ecf2;
    --wl-fg-muted: #a7b0bf;
    --wl-fg-subtle: #838d9c;
    --wl-border: #2b3341;
    --wl-card-bg: #1a202b;
  }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--wl-fg); background: var(--wl-bg); line-height: 1.5;
  }}
  .wl-report-content {{ padding: 0 clamp(16px, 5vw, 64px) 44px; }}
  .wl-report-banner {{
    display: flex; align-items: center; gap: 20px;
    padding: 28px clamp(16px, 5vw, 64px);
    margin-bottom: 32px;
    background: linear-gradient(135deg, var(--wl-banner-a) 0%, var(--wl-banner-b) 100%);
    border-bottom: 3px solid var(--wl-good);
  }}
  .wl-report-banner img {{ height: 46px; }}
  .wl-report-banner-text {{ display: flex; flex-direction: column; gap: 2px; }}
  .wl-report-banner-title {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em; }}
  .wl-report-banner-w {{ color: #d63333; }}
  .wl-report-banner-l {{ color: var(--wl-good); }}
  .wl-report-banner-subtitle {{ font-size: 0.95rem; color: var(--wl-fg-muted); font-weight: 500; }}
  .wl-report-theme-toggle {{
    margin-left: auto; border: 1px solid var(--wl-border); background: var(--wl-card-bg);
    color: var(--wl-fg); border-radius: 999px; width: 38px; height: 38px; font-size: 1.05rem;
    cursor: pointer; line-height: 1; flex: none;
  }}
  .wl-report-theme-toggle:hover {{ border-color: var(--wl-good); }}
  .wl-report-meta {{ color: var(--wl-fg-muted); font-size: 0.85rem; margin-bottom: 28px; }}
  .wl-report-section {{ margin: 32px 0; }}
  .wl-report-section h2 {{
    font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--wl-fg-muted); border-bottom: 1px solid var(--wl-border); padding-bottom: 6px;
  }}
  .wl-report-narrative {{
    background: var(--wl-bg-elevated); border-left: 3px solid var(--wl-good);
    padding: 16px 20px; border-radius: 6px; font-size: 0.98rem;
  }}
  .wl-report-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 18px;
  }}
  .wl-report-card {{
    border: 1px solid var(--wl-border); border-radius: 10px; padding: 14px 16px;
    background: var(--wl-card-bg);
  }}
  .wl-report-card-head {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
  }}
  .wl-report-signal-name {{ font-weight: 600; font-size: 0.92rem; }}
  .wl-report-badge {{
    font-size: 0.72rem; font-weight: 600; padding: 3px 9px; border-radius: 999px;
  }}
  .wl-report-badge-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 8px; }}
  /* Plots are rendered once, on a fixed white matplotlib canvas -- rather than
     render each one twice (light/dark figure), the image sits in an
     always-light thumbnail card so its own text/gridlines stay legible no
     matter which theme the surrounding page is in. */
  .wl-report-plot-wrap {{ background: #ffffff; border-radius: 6px; padding: 6px; }}
  .wl-report-plot {{ width: 100%; display: block; border-radius: 3px; }}
  .wl-report-noplot {{ color: var(--wl-fg-muted); font-size: 0.85rem; padding: 8px 0; }}
  .wl-report-outliers {{ margin-top: 10px; border-top: 1px dashed var(--wl-border); padding-top: 8px; }}
  .wl-report-shape-block {{ margin-bottom: 18px; }}
  .wl-report-list {{ padding-left: 1.2em; font-size: 0.92rem; }}
  .wl-report-subhead {{ font-weight: 600; margin: 12px 0 4px; font-size: 0.88rem; color: var(--wl-fg-muted); display: block; }}
  .wl-report-muted {{ color: var(--wl-fg-subtle); font-style: italic; }}
  .wl-report-footer {{ margin-top: 48px; color: var(--wl-fg-subtle); font-size: 0.78rem; text-align: center; }}

  /* --- Report editor: toolbar, blocks, charts, zoom, expand modal, runs table --- */
  .wl-editor-toolbar {{
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    padding: 8px clamp(16px, 5vw, 64px); margin-bottom: 24px;
    background: var(--wl-bg); border-bottom: 1px solid var(--wl-border);
  }}
  .wl-editor-toolbar-btn {{
    border: 1px solid var(--wl-border); background: var(--wl-card-bg); color: var(--wl-fg);
    border-radius: 6px; padding: 6px 12px; font-size: 0.82rem; cursor: pointer;
  }}
  .wl-editor-toolbar-btn:hover {{ border-color: var(--wl-good); }}
  .wl-editor-toolbar-hint {{ margin-left: auto; font-size: 0.76rem; color: var(--wl-fg-subtle); }}
  .wl-block {{ position: relative; }}
  .wl-block-toolbar {{
    position: absolute; top: 6px; right: 6px; z-index: 5;
    display: flex; gap: 2px; opacity: 0; transition: opacity 0.12s ease;
    background: var(--wl-card-bg); border: 1px solid var(--wl-border); border-radius: 6px; padding: 2px;
  }}
  .wl-block:hover > .wl-block-toolbar, .wl-block-toolbar:focus-within {{ opacity: 1; }}
  .wl-block-btn {{
    border: none; background: transparent; color: var(--wl-fg-muted); cursor: pointer;
    font-size: 0.85rem; line-height: 1; width: 24px; height: 24px; border-radius: 4px;
  }}
  .wl-block-btn:hover {{ background: var(--wl-bg-elevated); color: var(--wl-fg); }}
  .wl-block-btn-danger:hover {{ color: var(--wl-warn); }}
  .wl-chart-wrap {{ position: relative; height: 200px; background: var(--wl-card-bg) !important; }}
  .wl-chart-wrap canvas {{ cursor: crosshair; }}
  .wl-zoom-overlay {{
    display: none; position: absolute; top: 0; bottom: 0; pointer-events: none;
    background: color-mix(in srgb, var(--wl-good) 18%, transparent);
    border-left: 1px solid var(--wl-good); border-right: 1px solid var(--wl-good);
  }}
  .wl-report-runs-table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
  .wl-report-runs-table th, .wl-report-runs-table td {{
    padding: 6px 10px; border-bottom: 1px solid var(--wl-border); text-align: left;
  }}
  .wl-report-runs-table th {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--wl-fg-muted);
  }}
  .wl-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; }}
  .wl-report-user-title, .wl-report-user-text {{
    border: 1px dashed transparent; border-radius: 6px; padding: 6px 8px; margin: 0 0 12px;
    outline: none;
  }}
  .wl-report-user-title {{ font-size: 1.2rem; font-weight: 700; }}
  .wl-report-user-text {{ font-size: 0.95rem; }}
  .wl-report-user-title:focus, .wl-report-user-text:focus {{ border-color: var(--wl-border); background: var(--wl-bg-elevated); }}
  #wl-freeform-blocks:empty {{ display: none; }}
  .wl-expand-overlay {{
    position: fixed; inset: 0; z-index: 100; display: flex; align-items: center; justify-content: center;
    background: rgba(15, 23, 42, 0.55); padding: 24px;
  }}
  .wl-expand-panel {{
    width: min(1100px, 100%); height: min(680px, 90vh); display: flex; flex-direction: column;
    background: var(--wl-card-bg); border: 1px solid var(--wl-border); border-radius: 10px;
    box-shadow: 0 24px 60px rgba(15, 23, 42, 0.4); overflow: hidden;
  }}
  .wl-expand-head {{
    flex: 0 0 auto; display: flex; align-items: center; gap: 8px; padding: 10px 14px;
    border-bottom: 1px solid var(--wl-border); font-weight: 600;
  }}
  .wl-expand-head span {{ flex: 1 1 auto; }}
  .wl-expand-body {{ flex: 1 1 auto; min-height: 0; position: relative; padding: 12px; }}
  .wl-expand-body canvas {{ cursor: crosshair; }}
  .wl-expand-body img {{ max-width: 100%; max-height: 100%; display: block; margin: 0 auto; }}

  @media print {{
    .wl-editor-toolbar, .wl-block-toolbar, .wl-report-theme-toggle, .wl-expand-overlay {{ display: none !important; }}
    .wl-block {{ break-inside: avoid; }}
    .wl-report-user-title, .wl-report-user-text {{ border: none; }}
    body {{ background: #ffffff; color: #1a1a1a; }}
  }}
</style>
{chartjs_script}
</head>
<body>
  <div class="wl-report-banner">
    {logo_img}
    <div class="wl-report-banner-text">
      <span class="wl-report-banner-title"><span class="wl-report-banner-w">Weights</span><span class="wl-report-banner-l">Lab</span></span>
      <span class="wl-report-banner-subtitle">Experiment Report</span>
    </div>
    <button type="button" id="wl-theme-toggle" class="wl-report-theme-toggle" title="Toggle light/dark mode">&#9789;</button>
  </div>

  <div class="wl-editor-toolbar">
    <button type="button" class="wl-editor-toolbar-btn" id="wl-add-title-btn">+ Title</button>
    <button type="button" class="wl-editor-toolbar-btn" id="wl-add-text-btn">+ Text</button>
    <button type="button" class="wl-editor-toolbar-btn" id="wl-export-pdf-btn">Export to PDF</button>
    <span class="wl-editor-toolbar-hint">Hover a card for move / expand / remove &middot; edits stay in this browser tab until you export</span>
  </div>

  <div class="wl-report-content">
    <div class="wl-report-meta">
      Generated {generated_at} &middot; {root_log_dir}
    </div>

    <div class="wl-report-section">
      <h2>Analysis</h2>
      <div class="wl-report-narrative">{narrative}</div>
    </div>

    <div id="wl-freeform-blocks"></div>

    {runs_section_html}

    <div class="wl-report-section">
      <h2>Signals</h2>
      {signals_html}
    </div>

    {distributions_section_html}

    <div class="wl-report-section">
      <h2>Loss-Shape Classification</h2>
      {loss_shape_html}
    </div>

    <div class="wl-report-section">
      <h2>Dataset</h2>
      {dataframe_html}
    </div>

    <div class="wl-report-footer">WeightsLab &middot; generated automatically, review before sharing</div>
  </div>

  <script>
    (function () {{
      var btn = document.getElementById("wl-theme-toggle");
      if (!btn) return;
      function isDark() {{
        var explicit = document.documentElement.getAttribute("data-theme");
        if (explicit === "light") return false;
        if (explicit === "dark") return true;
        return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      }}
      function sync() {{
        var dark = isDark();
        btn.textContent = dark ? "\\u2600" : "\\u263D";
        btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
      }}
      btn.addEventListener("click", function () {{
        var next = isDark() ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", next);
        try {{ localStorage.setItem("wl-report-theme", next); }} catch (e) {{}}
        sync();
      }});
      sync();
    }})();
  </script>
{interactive_js}
</body>
</html>
"""


def render_report(context: dict, output_path, narrative: Optional[str] = None) -> str:
    """Render ``context`` (as returned by ``collect_report_context``) plus an
    optional ``narrative`` paragraph into a self-contained HTML file at
    ``output_path``. Returns the path written."""
    logo_b64 = _load_logo_base64()
    logo_img = (
        f'<img src="data:image/png;base64,{logo_b64}" alt="WeightsLab logo" />'
        if logo_b64 else ""
    )

    signals = context.get("signals") or []
    if signals:
        signals_html = '<div class="wl-report-grid">' + "".join(
            _signal_card_html(e, f"wl-signal-{i}") for i, e in enumerate(signals)
        ) + "</div>"
    else:
        msg = (
            "No signals with enough history to plot yet."
            if context.get("plotting_available", True)
            else "matplotlib is not installed — install with `pip install weightslab[reporting]` to render plots."
        )
        signals_html = f'<p class="wl-report-muted">{html.escape(msg)}</p>'

    narrative_html = html.escape(narrative).replace("\n", "<br/>") if narrative else (
        "<em>No narrative was generated for this report.</em>"
    )

    rendered = _HTML_TEMPLATE.format(
        title="WeightsLab Experiment Report",
        color_good=_COLOR_GOOD, color_warn=_COLOR_WARN, color_neutral=_COLOR_NEUTRAL,
        logo_img=logo_img,
        generated_at=html.escape(context.get("generated_at", "")),
        root_log_dir=html.escape(context.get("root_log_dir", "")),
        narrative=narrative_html,
        signals_html=signals_html,
        distributions_section_html=_distributions_section_html(context.get("distributions") or []),
        loss_shape_html=_loss_shape_section_html(context.get("loss_shape_tags") or []),
        dataframe_html=_dataframe_section_html(context.get("dataframe") or {}),
        runs_section_html=_runs_section_html(context.get("runs") or []),
        chartjs_script=_chartjs_script_tag(),
        interactive_js=_INTERACTIVE_JS,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return str(output_path)


def default_report_path(root_log_dir) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root_log_dir) / "reports" / f"experiment_report_{stamp}.html"


def list_reports(root_log_dir) -> list:
    """Every ``*.html`` report already written for this experiment, newest
    first (by mtime) -- the same directory and ordering the Studio report
    button's right-click dropdown uses (``weightslab/ui/server.py``'s
    ``_list_experiment_reports``), duplicated here rather than imported since
    that lives in the UI server module, not this LLM/agent-agnostic one."""
    reports_dir = Path(root_log_dir) / "reports"
    if not reports_dir.is_dir():
        return []
    paths = [p for p in reports_dir.iterdir() if p.is_file() and p.suffix == ".html"]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def latest_report_path(root_log_dir) -> Optional[Path]:
    """The most recently written report for this experiment, or ``None`` if
    none exists yet -- used to resolve "update the report"/"add X to the
    report" requests to the file they mean, see ``generate_report``'s
    ``update_existing``."""
    reports = list_reports(root_log_dir)
    return reports[0] if reports else None


def summarize_context_for_llm(context: dict) -> str:
    """The JSON summary of ``context`` that gets handed to an LLM for the
    narrative: everything except the base64 plot images (bytes an LLM can do
    nothing with, and by far the bulk of the context).

    Bounded by construction — see ``collect_report_context``: per-signal
    aggregates + O(top_k) outliers + a label→count histogram, never per-sample
    rows — so the prompt is the same size for a hundred samples or ten million.
    """
    return json.dumps({
        "signals": [
            {k: v for k, v in entry.items() if k not in ("plot_b64", "series")}
            for entry in context.get("signals", [])
        ],
        "distributions": [
            {k: v for k, v in entry.items() if k != "plot_b64"}
            for entry in context.get("distributions", [])
        ],
        "loss_shape_tags": context.get("loss_shape_tags", []),
        "dataframe": context.get("dataframe", {}),
    }, indent=2, default=str)


def generate_report(
    root_log_dir,
    logger_q,
    df: Optional[pd.DataFrame] = None,
    signals: Optional[list] = None,
    output_path=None,
    narrative_fn: Optional[Callable[[str], str]] = None,
    distributions: Optional[list] = None,
    update_existing: bool = False,
    checkpoint_manager=None,
) -> dict:
    """Collect → narrate → render, in one call. The single implementation
    behind every user-facing entry point (the Studio report button and the
    agent's ``generate_experiment_report`` action, ``wl.ai_report_generation``,
    and the CLI console's ``report`` command), so they cannot drift apart.

    ``narrative_fn`` is called with ``summarize_context_for_llm(context)`` and
    must return the Analysis prose — in practice
    ``DataManipulationAgent.generate_report_narrative``. It is kept as an
    injected callable so this module stays LLM-agnostic. Pass ``None`` to skip
    the analysis entirely; a ``narrative_fn`` that *raises* (no provider
    configured, request timed out, ...) degrades the same way — a report with
    no written analysis rather than no report at all.

    ``distributions`` names columns to render as value-distribution
    histograms in an extra Distributions section — opt-in, e.g. a follow-up
    "add a histogram of train_loss to the report" — see
    ``compute_distribution_entries``.

    ``update_existing`` (default ``False``, ignored if ``output_path`` is
    given explicitly) — "update the report"/"add X to the report" should
    overwrite the most recently written report for this experiment rather
    than create a new timestamped one; "generate a report" with no reference
    to an existing one should always create a fresh file. Resolved via
    ``latest_report_path``; when there is nothing to update yet (a fresh
    experiment, or its ``reports/`` directory was cleared), this falls back
    to creating a new report exactly as if ``update_existing`` were ``False``
    — there is nothing wrong with "update" on a first-ever report.

    ``checkpoint_manager`` (optional) supplies run metadata (experiment_name,
    notes, current-run flag — see ``CheckpointManager.list_runs``) for the
    report's Runs section and per-curve legend labels; omitted, curves and
    runs fall back to showing their raw hash.

    Returns ``{"path", "n_signals", "narrative", "narrative_error",
    "updated_existing"}`` — the last key tells the caller whether an existing
    file was overwritten (``True``) or a new one was created (``False``), so
    it can phrase its reply accordingly. Failures to gather data or write the
    file are NOT swallowed: they raise, because there is no report to hand
    back.
    """
    context = collect_report_context(
        root_log_dir, logger_q, df, signals=signals, distributions=distributions,
        checkpoint_manager=checkpoint_manager,
    )

    narrative = None
    narrative_error = None
    if narrative_fn is not None:
        try:
            narrative = narrative_fn(summarize_context_for_llm(context))
        except Exception as exc:
            narrative_error = str(exc)
            logger.warning("experiment report: narrative generation failed: %s", exc)

    resolved_output_path = output_path
    updated_existing = False
    if resolved_output_path is None and update_existing:
        existing = latest_report_path(root_log_dir)
        if existing is not None:
            resolved_output_path = existing
            updated_existing = True
    if resolved_output_path is None:
        resolved_output_path = default_report_path(root_log_dir)

    path = render_report(context, resolved_output_path, narrative=narrative)
    return {
        "path": path,
        "n_signals": len(context.get("signals") or []),
        "narrative": narrative,
        "narrative_error": narrative_error,
        "updated_existing": updated_existing,
    }
