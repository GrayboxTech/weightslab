"""Experiment health report: signal plots + dataframe stats + logo, as one
self-contained HTML file. Built for the agent's "generate a report" action
(``DataService._agent_generate_experiment_report``) — this module has no
LLM/agent coupling of its own, it only turns already-fetched data (a live
``LoggerQueue`` and the sample dataframe) into a rendered report. The
narrative/conclusion text is written by the agent from ``collect_report_context``'s
output and handed back in via ``narrative=`` when rendering.
"""

import base64
import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def collect_report_context(
    root_log_dir,
    logger_q,
    df: Optional[pd.DataFrame],
    signals: Optional[list] = None,
    max_signals: Optional[int] = None,
) -> dict:
    """Gather everything a report needs EXCEPT the narrative: per-signal
    trajectories + health classification + plots, and dataframe-level stats.
    ``max_signals=None`` (default) includes every logged signal with enough
    history -- pass an int to cap it (see select_important_signals).

    Split out from ``render_report`` so the caller (the agent) can hand the
    returned, LLM-friendly-sized ``context["signals"]``/``context["dataframe"]``
    summary to an LLM call for the narrative section, then pass the whole
    context (plus that narrative) to ``render_report`` — one data pass, one
    render pass, no wasted duplicate work.
    """
    plt = _import_matplotlib()
    resolved_signals = signals or select_important_signals(logger_q, max_signals=max_signals)

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
        })

    return {
        "root_log_dir": str(root_log_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signals": signal_entries,
        "dataframe": compute_dataframe_stats(df),
        "loss_shape_tags": summarize_loss_shape_tags(df),
        "plotting_available": plt is not None,
    }


def _fmt_num(x) -> str:
    try:
        return f"{float(x):.4g}"
    except (TypeError, ValueError):
        return str(x)


def _signal_card_html(entry: dict) -> str:
    name = html.escape(str(entry["name"]))
    label = html.escape(str(entry["label"]))
    color = entry["color"]
    body = (
        f'<div class="wl-report-card">'
        f'  <div class="wl-report-card-head">'
        f'    <span class="wl-report-signal-name">{name}</span>'
        f'    <span class="wl-report-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;">{label}</span>'
        f'  </div>'
    )
    if entry.get("plot_b64"):
        body += f'  <img class="wl-report-plot" src="data:image/png;base64,{entry["plot_b64"]}" alt="{name} trajectory" />'
    else:
        body += (
            '  <div class="wl-report-noplot">'
            f'Points: {entry["n_points"]} · first {_fmt_num(entry["first_value"])} '
            f'&rarr; last {_fmt_num(entry["last_value"])} '
            f'(min {_fmt_num(entry["min_value"])}, max {_fmt_num(entry["max_value"])})'
            '</div>'
        )
    body += _outliers_html(entry.get("outliers") or {})
    body += "</div>"
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


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  :root {{
    --wl-good: {color_good};
    --wl-warn: {color_warn};
    --wl-neutral: {color_neutral};
  }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1a1a1a; background: #ffffff; line-height: 1.5;
  }}
  .wl-report-content {{ padding: 0 clamp(16px, 5vw, 64px) 44px; }}
  .wl-report-banner {{
    display: flex; align-items: center; gap: 20px;
    padding: 28px clamp(16px, 5vw, 64px);
    margin-bottom: 32px;
    background: linear-gradient(135deg, #f7f9fb 0%, #eef3f8 100%);
    border-bottom: 3px solid var(--wl-good);
  }}
  .wl-report-banner img {{ height: 46px; }}
  .wl-report-banner-text {{ display: flex; flex-direction: column; gap: 2px; }}
  .wl-report-banner-title {{ font-size: 1.6rem; font-weight: 800; letter-spacing: -0.01em; }}
  .wl-report-banner-w {{ color: #d63333; }}
  .wl-report-banner-l {{ color: var(--wl-good); }}
  .wl-report-banner-subtitle {{ font-size: 0.95rem; color: #555; font-weight: 500; }}
  .wl-report-meta {{ color: #666; font-size: 0.85rem; margin-bottom: 28px; }}
  .wl-report-section {{ margin: 32px 0; }}
  .wl-report-section h2 {{
    font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.04em;
    color: #444; border-bottom: 1px solid #e5e5e5; padding-bottom: 6px;
  }}
  .wl-report-narrative {{
    background: #f7f9fb; border-left: 3px solid var(--wl-good);
    padding: 16px 20px; border-radius: 6px; font-size: 0.98rem;
  }}
  .wl-report-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 18px;
  }}
  .wl-report-card {{
    border: 1px solid #e5e5e5; border-radius: 10px; padding: 14px 16px;
    background: #fff;
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
  .wl-report-plot {{ width: 100%; display: block; border-radius: 6px; }}
  .wl-report-noplot {{ color: #666; font-size: 0.85rem; padding: 8px 0; }}
  .wl-report-outliers {{ margin-top: 10px; border-top: 1px dashed #e5e5e5; padding-top: 8px; }}
  .wl-report-shape-block {{ margin-bottom: 18px; }}
  .wl-report-list {{ padding-left: 1.2em; font-size: 0.92rem; }}
  .wl-report-subhead {{ font-weight: 600; margin: 12px 0 4px; font-size: 0.88rem; color: #444; display: block; }}
  .wl-report-muted {{ color: #888; font-style: italic; }}
  .wl-report-footer {{ margin-top: 48px; color: #999; font-size: 0.78rem; text-align: center; }}
</style>
</head>
<body>
  <div class="wl-report-banner">
    {logo_img}
    <div class="wl-report-banner-text">
      <span class="wl-report-banner-title"><span class="wl-report-banner-w">Weights</span><span class="wl-report-banner-l">Lab</span></span>
      <span class="wl-report-banner-subtitle">Experiment Report</span>
    </div>
  </div>

  <div class="wl-report-content">
    <div class="wl-report-meta">
      Generated {generated_at} &middot; {root_log_dir}
    </div>

    <div class="wl-report-section">
      <h2>Analysis</h2>
      <div class="wl-report-narrative">{narrative}</div>
    </div>

    <div class="wl-report-section">
      <h2>Signals</h2>
      {signals_html}
    </div>

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
            _signal_card_html(e) for e in signals
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
        loss_shape_html=_loss_shape_section_html(context.get("loss_shape_tags") or []),
        dataframe_html=_dataframe_section_html(context.get("dataframe") or {}),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return str(output_path)


def default_report_path(root_log_dir) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root_log_dir) / "reports" / f"experiment_report_{stamp}.html"
