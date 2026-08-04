Experiment Reports
===================

Ask the AI agent how your experiment is doing and it can produce a
self-contained HTML report: signal trajectory plots, an automatic health
classification per signal, dataset stats (sample counts, discard rate, tag
distribution), and a written analysis grounded in those exact numbers —
branded with the WeightsLab logo.

.. figure:: _static/logo-light.png
   :width: 160px
   :align: right
   :alt: WeightsLab logo

Generating a report
--------------------

This is an :doc:`agent <agent>` action — there is no separate CLI command or
Python function to call. Two equivalent ways to trigger it:

- **Chat / CLI**: ask for it in the chat bar (or via ``agent query`` on the CLI
  console; see :doc:`agent` for how to initialize a provider first):

  .. code-block:: text

     How is this experiment going? Generate a report.
     Create a report on training progress.
     Summarize the experiment.

  By default the report covers every signal with enough logged history (see
  `Signal selection`_ below). To report on specific signals instead:

  .. code-block:: text

     Generate an experiment report on train_loss and val_loss.

- **Weights Studio button**: the bar-chart icon immediately left of the
  notebook button in the connected app's header. Left-click generates a
  report (checking agent availability first — see below); right-click opens
  a dropdown of every report already on disk, newest first, click one to
  open it in a new browser tab.

Either path runs the exact same backend action; the button just sends the
same request the chat bar would, with a canned prompt. A colored status pill
tracks progress: amber while checking/generating, orange if the agent isn't
configured (the report still generates, just without a written analysis),
green on success, red on failure.

The reply names the file written under ``<root_log_dir>/reports/`` (a
timestamped ``experiment_report_<YYYYMMDD_HHMMSS>.html``) — open it in any
browser.

What's in the report
----------------------

- **Analysis** — a short, written summary of how the run is going, produced
  by the agent's own LLM. It is grounded *only* in the numbers described
  below (never raw per-step history), so it can comment on the data but
  cannot invent a signal, trend, or number that isn't actually there.
- **Signals** — one card per plotted signal: its trajectory (aggregated over
  training, from the experiment logger — see :doc:`logger`), and a health
  badge from the same :ref:`loss-shape classification <custom-signal-classifier>`
  vocabulary used elsewhere in WeightsLab:

  ==============  =========  ====================================================
  Label           Badge      Meaning
  ==============  =========  ====================================================
  monotonic       green      Steadily improving.
  plateaued       green      Improved, then leveled off.
  Flat_high       red        Never moved — likely stuck or unlearnable.
  high_variance   red        Noisy oscillation, no clear trend.
  U_Shape         red        Dipped, still moving — not settled yet.
  Forgotten       red        Regressed to a new, worse, flat level.
  Spiked          red        A transient jump that reverted.
  ==============  =========  ====================================================

- **Per-sample outliers** (within each signal's card) — the handful of
  samples with that signal's highest logged peak, and the handful whose
  history swung the most (``max - min``). Both are ranked *inside DuckDB*
  (``LoggerQueue.top_k_samples_by_reduce``) and only the top few ever leave
  the database — see `Why per-sample data doesn't blow up the report`_.
- **Loss-Shape Classification** — if per-sample loss-shape classification has
  already been computed for this experiment (:doc:`logger`'s
  ``wl.write_loss_shapes`` / the background auto-tagger), a count of samples
  per shape label across the *whole* dataset, plus a few example sample_ids
  for any concerning label. If nothing has been computed yet, the report says
  so — it never runs the classifier itself.
- **Dataset** — total sample count, discard count/rate, per-split counts
  (the ``origin`` column), and a breakdown of any ``tag:*`` columns present.

Why per-sample data doesn't blow up the report
--------------------------------------------------

A dataset can have millions of samples, but nothing in this report scales
with sample count — by construction, not by truncation-after-the-fact:

- **Outliers** come from ``LoggerQueue.top_k_samples_by_reduce``, which does
  the ``GROUP BY sample_id`` reduction *and* the ``ORDER BY ... LIMIT k``
  ranking in one DuckDB query. Only the top ``k`` (5) rows are ever pulled
  into Python — a per-sample Python dict of every sample's value is never
  built, let alone sent anywhere.
- **Loss-shape classification** is summarized as a label → count histogram
  (at most the 7 :data:`weightslab.src.LOSS_SHAPES` labels) plus up to 3
  example sample_ids per *concerning* label — never a per-sample dump.
- The same bounded summary — no plot images, no raw history — is exactly
  what gets handed to the LLM for the written analysis, so the prompt size
  (and cost) for a report is the same whether the experiment logged a
  hundred samples or ten million.

Signal selection
-------------------

With no explicit ``signals`` list, the agent includes **every** registered
signal that has at least 2 logged points — there is nothing to plot or
classify with fewer than that, so those are skipped. Ordering (not
filtering): any signal whose name contains "loss" comes first (ordered by how
many points it has logged), then the remaining signals by the same ordering.
Pass an explicit ``signals`` list (as in the chat example above) to report on
only specific ones instead.

Each plot is sized for how it's actually displayed (~520×200px at 100dpi,
tightly cropped) rather than a large print-quality image — this keeps the
HTML file reasonably sized even with every signal included, instead of a
handful of oversized plots dominating the page.

Requirements
--------------

Plotting needs matplotlib, an optional extra so a minimal install never pulls
it in unasked:

.. code-block:: bash

   pip install weightslab[reporting]

Without it, the report still renders — the health classification and
dataset stats sections are unaffected — but signal cards show a text summary
(first/last/min/max value) instead of a plot.

The written analysis needs a configured agent LLM provider (see
:doc:`agent`'s *Initializing the agent* section). If no provider is
available, the report is still generated with a note that no analysis was
written, rather than failing outright.

How it works (under the hood)
--------------------------------

1. The agent action reads the experiment logger's aggregated per-step history
   for each selected signal (``LoggerQueue.get_current_signaL_history``) and
   the live sample dataframe, and renders both to plots/stats via
   ``weightslab.reporting`` — a plain data-in-plots-and-stats-out module with
   no LLM involvement of its own.
2. That (plot-free) summary — signal names, health labels, value ranges,
   dataset stats — is handed to the agent's LLM in a single, focused call
   (``DataManipulationAgent.generate_report_narrative``) asking specifically
   for the Analysis section's prose.
3. The plots, stats, and narrative are assembled into one HTML file and
   written to disk. A failure in step 2 degrades to a report with no written
   analysis rather than no report at all.
