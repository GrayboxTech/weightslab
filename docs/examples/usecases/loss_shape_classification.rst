Loss-Shape Classification per Sample
=====================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--usecase">Usecase</span>
     <span class="wl-eg-tag">loss analysis</span>
     <span class="wl-eg-tag">signal</span>
     <span class="wl-eg-tag">categorical tag</span>
     <span class="wl-eg-tag">per-sample</span>
     <span class="wl-eg-tag">trajectory</span>
   </div>

**Example:** ``weightslab/examples/Usecases/wl-classification-signals_shape_classification``

This use case trains a small CNN on MNIST and adds one feature on top of the
plain per-sample logging: a **custom loss-shape classifier**. The *shape* of a
sample's loss trajectory over training tells you more than any single value —
steadily dropping (the model is learning it) versus stuck-high (a candidate
mislabel). WeightsLab tags each sample with that shape automatically; here we
**override** the built-in classifier with our own rule.

The built-in default and your labels
-------------------------------------

Out of the box, :func:`classify_loss_shape` sorts each trajectory into one of
six built-in shapes:

.. list-table::
   :header-rows: 1

   * - Shape
     - Meaning
   * - ``monotonic``
     - Loss steadily decreasing — model is learning this sample well
   * - ``plateaued``
     - Dropped then levelled off high — stuck, possibly a hard sample
   * - ``Flat_high``
     - Never moved — likely a mislabelled or unlearnable sample
   * - ``high_variance``
     - Noisy oscillation — ambiguous annotation
   * - ``U_Shape``
     - Model learned it then forgot — catastrophic interference
   * - ``Spiked``
     - Sudden jump — data pipeline or augmentation change

Those six labels are just the *built-in's* vocabulary. A custom classifier can
emit **any** labels. This example uses a binary ``monotonic`` /
``not_monotonic`` rule.

Base setup
----------

Loader, model, optimizer, and the watched loss are the usual per-sample
tracking (nothing loss-shape-specific here). The one signal that matters below
is the per-sample loss, whose name comes from ``config.yaml``
(``loss_signal_name: loss_sample``):

.. code-block:: python

   LOSS = cfg["loss_signal_name"]   # "loss_sample"

   crit = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name=LOSS, per_sample=True, log=True,
   )

The custom classifier — ``@wl.signal_classifier``
-------------------------------------------------

The classifier lives in ``utils/criterions.py``. It is a plain callable —
``trajectory (list[float]) -> label | None`` — registered with the
:func:`signal_classifier` decorator. Returning ``None`` leaves a sample untagged
(here, until it has enough history). It reuses :func:`trajectory_stats`, the
scale-invariant feature layer the built-in classifier is built on, so we read
the trend without re-deriving it:

.. code-block:: python

   import weightslab as wl

   MIN_POINTS = 5

   @wl.signal_classifier(signal="loss_sample")
   def monotonic_or_not(values):
       """"monotonic" when the loss dropped substantially start-to-end,
       else "not_monotonic". None until MIN_POINTS points exist."""
       s = wl.trajectory_stats(values)
       if s is None or s["n"] < MIN_POINTS:
           return None
       return "monotonic" if s["drop"] > 0.4 else "not_monotonic"

``@wl.signal_classifier(signal="loss_sample")`` binds this rule to the
``loss_sample`` signal only. (Use a bare ``@wl.signal_classifier`` — or
``@wl.signal_classifier()`` — to make it the **global default** for every signal
that has no per-signal classifier of its own.) The resolution order for any
signal is: per-signal registered → global registered → built-in
:func:`classify_loss_shape`.

When the loss signal name isn't known at import time (it comes from config),
bind it at runtime instead — this is what ``main.py`` calls:

.. code-block:: python

   def register_shape_classifier(loss_name):
       wl.signal_classifier(signal=loss_name)(monotonic_or_not)
       return monotonic_or_not

   # in main(), after watch_or_edit(..., flag="loss", signal_name=LOSS, ...):
   register_shape_classifier(LOSS)

Once registered, the classifier is consulted **everywhere shapes are
computed** — you don't wire up ``subscribe_to`` / history queries /
``set_categorical_tag`` yourself. The background auto-tagger applies it
automatically and fills a categorical ``tag:loss_shape`` column with our two
labels. The built-in six-way default is left untouched for every other signal.

Universal loss on the test split
---------------------------------

The watched criterion also runs over the test split each epoch (inside
``guard_testing_context``), so test samples accumulate a loss trajectory and get
a shape too — the classifier doesn't care which split a sample came from.

Reporting the tag
-----------------

At the end of the run, dump the categorical tag alongside the signals. Passing
``loss_shape_signal=LOSS`` runs the registered classifier once, synchronously,
so the ``tag:loss_shape`` column is guaranteed fresh in the dump:

.. code-block:: python

   path = wl.write_dataframe(
       OUT + "/report.csv", format="csv",
       columns=["signals", "tags"], loss_shape_signal=LOSS,
   )
   # report.csv now has a tag:loss_shape column of monotonic / not_monotonic

Workflow in the studio
----------------------

1. As samples accumulate ≥5 points, the ``loss_shape`` tag appears on each one,
   refreshed on every background tick.
2. Use the **Filter** panel to isolate ``not_monotonic`` samples — the ones the
   model is not learning cleanly — as relabelling candidates.
3. To eyeball *why*, right-click the ``loss_sample`` signal (in the left
   metadata panel or a List-view column header) and pick **Plot signal
   trajectory**. WeightsLab fetches each currently-shown sample's per-step
   trajectory for that signal on demand (via the ``GetSignalTrajectory`` RPC)
   and overlays the curves. This works for **any** signal — the name is resolved
   dynamically server-side, nothing is hardcoded to a "loss". Curves are
   downsampled to at most ``WL_SIGNAL_TRAJ_MAX_POINTS`` points (default 100), and
   samples with fewer than 3 recorded points are omitted rather than drawn as a
   misleading 1–2 point line.
4. Tag or discard the problem samples; the deny-aware sampler stops presenting
   them on the next training step.

.. tip::

   To run the example directly::

      cd weightslab/examples/Usecases/wl-classification-signals_shape_classification
      python main.py

   Knobs (epochs, output dir, signal name) live in ``config.yaml``; a few can be
   overridden via ``WL_STRESS_*`` environment variables for scripted runs.


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Usecases/wl-segmentation-loss-shapes-classification.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
