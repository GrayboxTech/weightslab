.. _studio-why-fail:

Why do these fail?
==================

.. note:: Beta

   New in v3. The analysis only reads the model; a repair changes it, and
   only when you accept it (never on its own in Automatic mode).

**Why do these fail?** takes samples the model gets wrong and works out why,
step by step, then proposes what to do: which data to collect or relabel, and
optionally a small repair of the weights, checked before it is applied. It
runs in the Agent Window's **Why fail?** tab, as a guided conversation that
asks before each step.

Starting an analysis
--------------------

In **Main**, choose **Why do these fail?** from:

- the data grid's **right-click** menu on a selection;
- a tag's right-click menu in the left panel (every sample with that tag);
- the painter's tag menu.

The tab also has its own field to start from a tag. One analysis runs at a
time; starting another stops the current one (a repair in progress is rolled
back).

Each step ends with a question and a default answer. Click an answer, or tick
**Automatic** (top right) to take every default and run to the end. **Stop**
ends the analysis at any point. The step bar at the top shows where it is.

The steps
---------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Step
     - What it does
   * - **Check**
     - Runs the model on the samples at the current weights and keeps the
       ones that fail now, shown as thumbnails with what went wrong. Samples
       that are fine are left out.
   * - **Look-alikes**
     - For each failure, the samples from train / val the model gets **right**
       that are most similar to it, in the model's own features (the input
       of its last layer), not in pixels. Failures farther from every
       training sample than 99% of training samples are flagged as unlike
       training.
   * - **Compare**
     - Compares failures and look-alikes layer by layer (Cohen's d per
       neuron, the layer's score is their RMS) and finds where they start to
       differ. Suspect neurons must differ more than when failures and
       look-alikes are shuffled together (the 95th percentile of 200
       shuffles), on their mean or their maximum.
   * - **Test**
     - Silences the suspects during an evaluation only (each replaced by its
       average on the look-alikes; the weights are untouched). If that fixes
       failures without breaking look-alikes, the suspects explain them; the
       set is then narrowed by halves to the smallest that still does.
   * - **Explain**
     - Shows the samples each confirmed neuron responds to most, and puts
       each failure in one category (below).
   * - **Fix**
     - Tags the samples for data work and, if you want, tries repairs of the
       model (below).

The **Open in Modeling (Compare)** link under the compare step opens the
failures (A) against their look-alikes (B) in the :doc:`modeling` tab, layer
by layer and neuron by neuron.

Categories
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Category
     - Meaning, and what helps
   * - **Spurious feature**
     - A few neurons explain the failure: the model reacts to something
       that should not matter (a background, a mark). Look at the samples
       those neurons respond to most; add data without that feature.
   * - **Mislabeled?**
     - Its nearest training samples carry the label the model predicted.
       Review the label before anything else.
   * - **Unlike training**
     - Nothing in train / val looks like it. Collect more samples like it.
   * - **Boundary / hard cases**
     - None of the above: it sits between classes or is simply hard. More
       data, or a bigger model.

What "failing" means per task
-----------------------------

The analysis recognises the task from ``task_type`` and from the shapes of the
model's output and targets:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Task
     - A sample fails when
     - Shown as
   * - Classification
     - it is predicted wrong
     - ``3 → 7 (92%)``
   * - Segmentation
     - its IoU (mean over the classes in the mask or the prediction) is low
     - wrong pixels in red
   * - Detection
     - the F1 of its boxes is low (a box is found at IoU 0.5 by a prediction
       at least 25% confident)
     - true boxes green, predicted boxes orange
   * - Language model
     - few of its next tokens are predicted right (labels equal to the inputs
       are shifted by one)
     - the text, and its most confident miss: ``"c" → "b"``
   * - Anything else
     - its loss, registered with ``flag="loss"``, is higher than most of
       train / val
     - ``loss 1.2 · higher than 85% of train / val``

For every task but classification, the check step asks where failing starts:
the task's default line, the worst quarter of train / val, or every selected
sample, each with how many samples it flags. A sample then counts as fixed (or
made worse) only for a change of at least 0.1, so a sample sitting on the line
does not flip on noise.

Tags
----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tag
     - Samples
   * - ``wopt_failures``
     - the samples that fail
   * - ``wopt_lookalikes``
     - their healthy look-alikes, the reference
   * - ``wopt_label_review``
     - failures whose label looks wrong
   * - ``wopt_unlike_training``
     - failures unlike anything in train / val
   * - ``wopt_neuron_top``
     - the samples the confirmed neurons respond to most
   * - ``wopt_fixed``
     - failures an applied repair fixed

Filter the grid by them to act on the data.

Repairs
-------

A repair fine-tunes the model from its current weights. Training is paused
while repairs run and stays paused afterwards; the weights are snapshotted
first and put back after each candidate.

The failures are split in two: half to train on, half **held out**, to see
whether a fix carries over to failures it never saw. Each candidate mixes 25%
failures and look-alikes with 75% normal training samples, with your
optimizer at half its learning rate and a small pull back toward the current
weights:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Candidate
     - What it trains
   * - **Control**
     - Normal training only, for the same number of steps: the reference.
       Without it, "better" may just mean "trained longer".
   * - **Fine-tune**
     - Every trainable weight, on the mix above.
   * - **Train only one layer**
     - The layer where failures and look-alikes start to differ; the rest
       is frozen. The smallest change.
   * - **Reset the neurons, then fine-tune**
     - The confirmed neurons are re-initialised, then relearn with the
       look-alikes.

The table then shows, for each candidate, the failures it fixed (trained and
held out), the metric on the check samples (the validation split, else the
test split without the analysed samples, else training samples outside the
failures and look-alikes) and the samples it made worse. A candidate passes
when:

- it fixes more held-out failures than the control;
- the metric on the check samples drops by at most 0.2 points;
- at most 0.5% of the check samples (or a quarter of its gains) go from right
  to wrong;
- no class drops by more than 1 point.

The best passing candidate is offered; the model changes only if you accept.
When none passes, the model stays as it was; by hand, you can still apply the
closest one anyway. Automatic mode never applies a candidate that fails the
checks.

.. tip::

   Read the control's row first. When the control fixes about as many
   failures as the best candidate, the model is mostly undertrained: keep
   training rather than repairing.

Big models and datasets
-----------------------

Every step is capped so it stays bounded: at most 256 failures (a seeded
subset), look-alikes searched among 4096 train / val samples, features
projected to 256 dimensions, a time budget per pass, about 120 s per repair
candidate and 1000 check samples. These defaults live in
``weightslab.components.failure_analysis.DEFAULTS``.

Hooks for your model
--------------------

Most models need nothing. When the analysis cannot tell the task or read your
outputs, it says so and names these hooks, all importable from
``weightslab.components.failure_analysis``:

.. code-block:: python

   from weightslab.components.failure_analysis import (
       set_failure_task,          # "classification", "segmentation", "detection",
                                  # "language_model" or "loss"
       set_failure_boxes_fn,      # detection: output -> one [N, 6] array per image,
                                  # [x1, y1, x2, y2, class, confidence]
       set_failure_loss_fn,       # (output, targets) -> one loss per sample
       set_failure_targets_fn,    # batch -> targets, for unusual batches
   )

   set_failure_task("detection")
   set_failure_boxes_fn(lambda output: decode_my_boxes(output))

A model can define the same as attributes or methods instead:
``wl_failure_task``, ``wl_failure_boxes(output)``,
``wl_failure_loss(output, targets)`` and ``wl_failure_targets(batch)``.

- **Detection** needs boxes per image: torchvision-style dicts and WeightsLab's
  ``[x1, y1, x2, y2, class, conf]`` rows are read as they are; a raw YOLO
  output needs ``set_failure_boxes_fn``, and an Ultralytics batch dict
  ``set_failure_targets_fn``. Repairs train with your loss.
- **Repairs** train with the task's own loss (cross-entropy for
  classification, segmentation and language models), else with the loss you
  registered with ``flag="loss"`` or ``set_failure_loss_fn``. Without one, the
  analysis and the tags still work; only the repair is skipped.
