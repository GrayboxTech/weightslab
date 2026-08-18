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

Four ways to ask for one — all four run the **same** code path
(``weightslab.reporting.generate_report``: collect → narrate → render), so
they produce the same artifact:

- **Python** — :func:`ai_report_generation` (see :doc:`user_functions`), for a
  snapshot from your training script or a notebook:

  .. code-block:: python

     import weightslab as wl

     path = wl.ai_report_generation()                        # every signal, with analysis
     wl.ai_report_generation(signals=["train_loss"])         # specific signals
     wl.ai_report_generation(use_agent=False)                # skip the LLM call
     wl.ai_report_generation(output_path="reports/run.html") # choose the file
     wl.ai_report_generation(distributions=["train_loss"])   # + a histogram section

  It returns the path written.

- **CLI console** — the ``report`` command (see :ref:`cli-console`), from a
  terminal attached to a running experiment with ``weightslab cli``:

  .. code-block:: text

     report
     report train_loss val_loss
     report --output /tmp/run_42.html
     report --no-agent
     report --distributions train_loss,val_loss

  The reply gives the path, the number of signals included, and whether the
  written analysis made it in.

- **Chat** — ask for it in the chat bar (or via ``agent query`` on the CLI
  console; see :doc:`agent` for how to initialize a provider first):

  .. code-block:: text

     How is this experiment going? Generate a report.
     Create a report on training progress.
     Summarize the experiment.

  By default the report covers every signal with enough logged history (see
  `Signal selection`_ below). To report on specific signals instead:

  .. code-block:: text

     Generate an experiment report on train_loss and val_loss.

  To add a value-distribution histogram for a specific column (see
  `Distributions`_ below), including as a follow-up on a report you already
  generated:

  .. code-block:: text

     Generate an experiment report and include a histogram of train_loss.
     Add a distribution of val_loss to the report.

  This always goes through the SAME single backend action — the agent must
  never break "generate a report" into several separate analysis questions
  and hand-write its own summary; that would skip the plots/styling below
  entirely.

Updating a report vs. generating a new one
---------------------------------------------

Every path above always writes a *fresh*, separately timestamped file by
default. When you ask through chat, though, wording matters:

.. code-block:: text

   Generate a report.                                # always a NEW file
   Update the report with a histogram of val_loss.    # overwrites the last one
   Add a histogram of val_loss to the report.         # overwrites the last one
   Also include the confidence signal in it.          # overwrites the last one

"Generate"/"create"/"how is this going" (no reference to one already made)
always produces a new file. Wording that refers to an *existing* report
("update", "add X to **the** report", "also include Y in **it**") overwrites
the most recently generated report for this experiment instead — the agent's
reply says which happened ("updated"/"generated ... experiment report") and
still names the file. Asking to "update" when nothing has been generated yet
isn't an error: it just creates the first one, same as a plain "generate"
would.

A follow-up "add" is intentionally cumulative — asking to add a histogram of
``val_loss`` after already having one for ``train_loss`` keeps both in the
updated file, not just the newest one, as long as the request stays in the
same conversation. There's no server-side memory of a report's contents
behind this — the agent reasons about what to keep from what you (and it)
said earlier in the chat, so it works within one back-and-forth but doesn't
persist across separate sessions.

Python/CLI callers that want the same overwrite-in-place behavior can pass
the previous run's own path back in as ``output_path``
(:func:`ai_report_generation`) / ``--output`` (the CLI's ``report`` command)
— they already have direct control over the file, so there's no separate
"update" flag for them.

- **Weights Studio button**: the bar-chart icon immediately left of the
  notebook button in the connected app's header. Left-click generates a
  report (checking agent availability first — see below); right-click opens
  a dropdown of every report already on disk, newest first, click one to
  open it in a new browser tab.

The button just sends the same request the chat bar would, with a canned
prompt. A colored status pill tracks progress: amber while
checking/generating, orange if the agent isn't configured (the report still
generates, just without a written analysis), green on success, red on
failure.

Every path writes to ``<root_log_dir>/reports/`` (a timestamped
``experiment_report_<YYYYMMDD_HHMMSS>.html``) unless an explicit output path
is given — open it in any browser.

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
- **Distributions** *(optional — only when asked for)* — a value-distribution
  histogram plus n/mean/std/range for each column named via ``distributions``
  (see `Generating a report`_ above). Unlike a Signals card, this reads the
  *current* per-sample dataframe, not the aggregated training curve — so it
  answers "how spread out is train_loss across samples right now", not "how
  did it move over training". A name that doesn't resolve to a column, or
  resolves to one with no numeric values, still gets a card saying so rather
  than being silently dropped. Not present at all when nobody asked for one.
- **Loss-Shape Classification** — if per-sample loss-shape classification has
  already been computed for this experiment (:doc:`logger`'s
  ``wl.write_loss_shapes`` / the background auto-tagger), a count of samples
  per shape label across the *whole* dataset, plus a few example sample_ids
  for any concerning label. If nothing has been computed yet, the report says
  so — it never runs the classifier itself.
- **Dataset** — total sample count, discard count/rate, per-split counts
  (the ``origin`` column), and a breakdown of any ``tag:*`` columns present.

Light / dark mode
--------------------

The report follows the browser's ``prefers-color-scheme`` automatically, and
also has its own toggle button (top-right of the banner) for overriding that
— the choice is remembered (via ``localStorage``, scoped to that report file)
so reopening the same report keeps the theme you picked. Signal/distribution
plots are rendered once by matplotlib on a fixed white canvas, so they sit in
a small always-light thumbnail card in either theme — this keeps their own
text and gridlines legible instead of rendering (and shipping) two copies of
every plot.

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

Plotting uses matplotlib (installed as a core dependency of WeightsLab). If you are working in an environment where it isn't present, install it with:

.. code-block:: bash

   pip install matplotlib
Without it, the report still renders — the health classification and
dataset stats sections are unaffected — but signal cards show a text summary
(first/last/min/max value) instead of a plot.

The written analysis needs a configured agent LLM provider (see
:doc:`agent`'s *Initializing the agent* section). If no provider is
available, the report is still generated with a note that no analysis was
written, rather than failing outright. The same applies when
``wl.ai_report_generation`` is called from a script that isn't serving an
experiment: there is no live agent to ask, so the report comes out without
the Analysis section instead of erroring.

How it works (under the hood)
--------------------------------

``weightslab.reporting.generate_report`` is the one implementation behind all
four entry points above; the LLM reaches it as an injected ``narrative_fn``
callable, so ``weightslab/reporting.py`` itself stays a plain
data-in-plots-and-stats-out module with no agent coupling.

1. ``collect_report_context`` reads the experiment logger's aggregated
   per-step history for each selected signal
   (``LoggerQueue.get_current_signaL_history``) and the live sample
   dataframe, and renders both to plots/stats.
2. That (plot-free) summary — signal names, health labels, value ranges,
   dataset stats — is handed to the agent's LLM in a single, focused call
   (``DataManipulationAgent.generate_report_narrative``) asking specifically
   for the Analysis section's prose.
3. The plots, stats, and narrative are assembled into one HTML file and
   written to disk. A failure in step 2 degrades to a report with no written
   analysis rather than no report at all.
