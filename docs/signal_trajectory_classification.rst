Signal Trajectory Classification
=================================

Every per-sample signal you log — a loss, an accuracy, a custom metric — has
a **trajectory**: the ordered sequence of values one sample produced over
training. WeightsLab can turn that trajectory into a categorical label
("this sample's loss is plateaued", "this one was forgotten") automatically,
surface it as a filterable/sortable tag in Studio, and let you swap in your
own labeling rule in a few lines. This page is the conceptual walkthrough;
it links out to the full API reference and a runnable use case rather than
repeating either.

Prerequisite: wrapping a value into a signal
---------------------------------------------

Trajectory classification only has something to work with once a value is
being logged per-sample in the first place. There are two ways to get there:

- **Wrap an existing loss/metric object** — ``wl.watch_or_edit(criterion,
  flag="loss", signal_name="train/loss", per_sample=True, log=True)`` hooks
  the object's ``forward``/``compute`` method, so every call logs and
  persists a per-sample value with no extra code in your training loop. This
  is the fast path, and the one every "loss shape" auto-classification below
  assumes.
- **Define a signal from scratch** — the ``@wl.signal(name=..., subscribe_to=...,
  compute_every_n_steps=..., per_sample=True)`` decorator wraps any function
  of your own into a tracked, logged signal, optionally driven by (subscribed
  to) another signal's value.

Both mechanisms, every argument, and the difference between static and
dynamic signals are covered in full in :doc:`logger` (concept) and
:doc:`user_functions` (API reference, ``signal`` section) — start there if
you haven't wrapped a signal before. Everything below assumes you already
have a per-sample signal (most commonly one registered with ``flag="loss"``)
producing values over time.

The mental model
-----------------

.. code-block:: text

   per-sample value history  -->  classifier(values) -> label  -->  tag / column / filter

For one sample, a signal's trajectory is just ``list[float]`` — its values in
step order. A **classifier** is any function ``list[float] -> str | None``
that looks at that list and returns a label, or ``None`` if there isn't
enough history yet to call it. WeightsLab applies a classifier to every
sample's trajectory and writes the resulting labels as a categorical tag
column, so you end up filtering/sorting samples by *how their curve behaved*
rather than just their latest value.

The built-in classifier
------------------------

Every signal registered via ``wl.watch_or_edit(criterion, flag="loss", ...)``
is classified automatically, with zero setup, by :func:`wl.classify_loss_shape`
into one of seven shapes:

==============  ====================================================================
Label           Meaning
==============  ====================================================================
monotonic       Loss steadily decreasing — the model is learning the sample.
plateaued       Decreased then leveled off still-high — stuck / hard sample.
Flat_high       Never moved, stayed high — likely a mislabel or unlearnable.
high_variance   Noisy oscillation — model uncertain, often an ambiguous label.
U_Shape         Dipped, then is recovering/still moving — not settled yet.
Forgotten       Dipped, then permanently regressed to a new, worse, flat level.
Spiked          One-step jump that reverts — transient, not a lasting change.
==============  ====================================================================

The background logger flush thread (``WL_LOGGER_FLUSH_INTERVAL_SECONDS``,
default 2 seconds) discovers every ``flag="loss"`` signal on its own and
re-tags it as ``'<signal_name>_shape'`` each tick, once a sample has enough
points to classify — no call needed. This shows up in Studio immediately as
a ``tag:<signal>_shape`` column: filter on it in the Filter panel, or
right-click the column header in the List view and **Pin to left** to keep
it visible while scrolling through everything else.

Defining your own classifier
------------------------------

The built-in shapes assume a loss that should trend *down*. For anything
else — a reward that should trend *up*, a metric with its own vocabulary of
outcomes — register a custom classifier with :func:`wl.signal_classifier`:

.. code-block:: python

   import weightslab as wl

   @wl.signal_classifier(signal="reward_loss")   # bind to one signal
   def rising_is_good(values: list[float]) -> str | None:
       s = wl.trajectory_stats(values)           # built-in feature layer, see below
       if s is None:
           return None                            # not enough points yet
       return "improving" if s["drop_z"] < -2 else "stalled"

A classifier receives one sample's ordered value trajectory and returns a
label string, or ``None`` to leave that sample untagged for now. Labels are
**free-form** — the seven built-in shapes are only the built-in classifier's
own vocabulary; yours can return anything.

**Binding modes**

- ``@wl.signal_classifier(signal="loss_sample")`` — classify only that one
  signal.
- ``@wl.signal_classifier`` / ``@wl.signal_classifier()`` (no ``signal=``) —
  become the global default for every signal that doesn't have its own
  per-signal classifier registered.

**Resolution order**: per-signal registered classifier → global default →
built-in :func:`wl.classify_loss_shape`. This same order is used everywhere
a shape gets computed — the background auto-tagger, report-time
:func:`wl.write_signal_shapes`/:func:`wl.write_loss_shapes`, and the live
:func:`wl.enable_loss_shape_signal` curve — so registering a classifier once
is enough; you never pass ``classifier=`` through each call site yourself.
Call :func:`wl.resolve_signal_classifier(signal_name) <wl.resolve_signal_classifier>`
if you want to confirm which one is actually active for a given signal right
now.

Building on ``trajectory_stats``, rather than hand-rolling your own trend
detection, is the recommended starting point — it returns scale- and
noise-invariant z-scores (net drop, dip/rebound, biggest jump and how much of
it reverted, …) computed against *that trajectory's own* noise floor, so the
same threshold works whether the signal lives in the single digits or the
thousands. See :func:`wl.trajectory_stats` in :doc:`user_functions` for the
full key table.

Seeing the raw curve behind a label
--------------------------------------

A tag column tells you *what* a sample's trajectory was classified as; to
see *why*, right-click the signal in the left metadata panel or a List-view
column header and pick **Plot signal trajectory**. This calls the
``GetSignalTrajectory`` RPC on demand — only for the samples currently shown,
never as part of the regular metadata poll — and overlays each one's raw
per-step curve. It's a read-only visualization decoupled from classification
itself (which always happens on the write path, described above); use it to
eyeball a handful of ``Forgotten`` or ``high_variance`` samples and sanity
check that the label matches what the curve is actually doing.

Where to go next
------------------

- :doc:`logger` — the signal-wrapping concept (``watch_or_edit``, ``@wl.signal``).
- :doc:`user_functions` — full API reference for ``signal_classifier``,
  ``resolve_signal_classifier``, ``trajectory_stats``, ``classify_loss_shape``,
  ``write_signal_shapes``/``write_loss_shapes``, ``enable_loss_shape_signal``,
  and ``enable_loss_shape_autotag``/``disable_loss_shape_autotag``.
- :doc:`examples/usecases/loss_shape_classification` — a full runnable
  walkthrough, including the Studio filter-and-relabel workflow end to end.
