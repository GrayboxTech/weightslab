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

**Example:** ``weightslab/examples/Usecases/wl-loss_shapes_classification_per_sample/``

This use case builds on :doc:`../pytorch/detection` (same Penn-Fudan dataset,
same model, same ``guard_training_context`` pattern) and adds one new feature:
a **dynamic subscribed signal** that watches each sample's loss *trajectory*
over training, classifies its shape, and tags the sample automatically.

The idea is that the *shape* of a loss curve over time tells you more than any
single value:

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

Base detection setup
--------------------

Loader, model, optimizer, and loss signals are identical to
:doc:`../pytorch/detection`. Refer to that page for the full walkthrough.
The heavy-experiment flags are also applied:

.. code-block:: python

   train_loader = wl.watch_or_edit(
       _train_dataset,
       flag="data", loader_name="train_loader",
       batch_size=8, shuffle=True, is_training=True,
       collate_fn=det_collate,
       array_autoload_arrays=False,
       array_return_proxies=True,
       array_use_cache=True,
       preload_labels=False,
   )

   train_sig = {
       "loss":         wl.watch_or_edit(PerSampleDetectionLoss(...),
                           flag="loss", name="train_loss/sample",
                           per_sample=True, log=True),
       "iou_sample":   wl.watch_or_edit(PerSampleIoU(...),
                           flag="metric", name="train_iou/sample",
                           per_sample=True, log=True),
       "iou_instance": wl.watch_or_edit(PerInstanceIoU(...),
                           flag="metric", name="train_iou/instance",
                           per_instance=True, log=True),
   }

Declaring the categorical tag
------------------------------

Before defining the subscribed signal, declare the categorical tag and its
allowed values so the studio shows all six choices in the filter panel even
before any sample has been classified:

.. code-block:: python

   LOSS_SHAPE_LABELS = [
       "monotonic", "plateaued", "Flat_high",
       "high_variance", "U_Shape", "Spiked",
   ]

   wl.register_categorical_tag("loss_shape", LOSS_SHAPE_LABELS)

This call must happen **after** ``wl.watch_or_edit(..., flag="data", ...)``
because the dataframe (which stores the tag column) is created at that point.

The subscribed signal
----------------------

.. code-block:: python

   checkpoint_manager = ledgers.get_checkpoint_manager()

   @wl.signal(
       name="train_loss_sample_shape_classifier",
       subscribe_to="train_loss/sample",
       compute_every_n_steps=25,
       log=False,
   )
   def classify_loss_shape(ctx: wl.SignalContext) -> int:
       history = wl.query_sample_history(
           ctx.sample_id,
           signal_name="train_loss/sample",
           exp_hash=checkpoint_manager.get_current_experiment_hash(),
       )
       series = sorted(
           ((step, val) for _, step, val, _ in history),
           key=lambda t: t[0],
       )
       values = [v for _, v in series]

       label = _classify_loss_shape(values)
       if label is None:
           return -1

       wl.set_categorical_tag([ctx.sample_id], "loss_shape", label)
       return LOSS_SHAPE_CODES[label]

Breaking down the decorator arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``subscribe_to="train_loss/sample"``
   The signal fires once per sample for which ``train_loss/sample`` was
   updated. ``ctx.sample_id`` identifies the exact sample; ``ctx.subscribed_value``
   holds the latest scalar value and ``ctx.subscribed_history`` (if
   ``include_history=True``) holds the full trajectory.

``compute_every_n_steps=25``
   The classifier runs every 25 optimisation steps, not every step. History
   classification is cheap but not free; 25 steps is a good balance between
   responsiveness and overhead.

``log=False``
   This signal has a side-effect (tagging the sample) and does not need an
   aggregate curve in the studio. The returned integer is still stored as a
   per-sample signal column so you can sort by ``loss_shape_code`` in the UI.

Inside the function
~~~~~~~~~~~~~~~~~~~~

``wl.query_sample_history`` returns the full list of ``(experiment_hash,
step, value, metadata)`` tuples recorded for a given sample and signal.
Filtering by ``exp_hash`` ensures only the current run's history is used.

``wl.set_categorical_tag([ctx.sample_id], "loss_shape", label)`` writes
the string label back to the sample's row in the ledger. The studio
immediately reflects the update — samples are colour-coded by loss shape
in the sample grid.

Shape classifier logic
~~~~~~~~~~~~~~~~~~~~~~~

The classifier uses scale-invariant thresholds (fractions of the trajectory's
own range):

.. code-block:: python

   def _classify_loss_shape(values):
       y = np.asarray(values, dtype=float)
       if y.size < 5:           # need at least 5 points to decide
           return None

       cv       = y.std() / (abs(y.mean()) + 1e-8)   # noisiness
       drop     = (y[0] - y[-1]) / (abs(y[0]) + 1e-8)# net improvement
       rebound  = (y[-1] - y.min()) / max(y.max() - y.min(), 1e-8)
       max_jump = np.diff(y).max() / max(y.max() - y.min(), 1e-8)
       tail_flat = (y[int(0.6*len(y)):].std() /
                    (abs(y[int(0.6*len(y)):].mean()) + 1e-8)) < 0.1

       if max_jump > 0.5:                return "Spiked"
       if cv > 0.5:                      return "high_variance"
       if 0.2*len(y) < np.argmin(y) < 0.8*len(y) and rebound > 0.3:
                                         return "U_Shape"
       if drop > 0.4:                    return "monotonic"
       if drop > 0.15 and tail_flat:     return "plateaued"
       return "Flat_high"

All thresholds are illustrative. Tune ``cv``, ``drop``, ``rebound``, and
``max_jump`` to match the scale and noise characteristics of your own task.

Workflow in the studio
-----------------------

1. After ~25 steps, the ``loss_shape`` tag column appears on each sample.
2. Use the **Filter** panel to show only ``Flat_high`` samples — candidates
   for relabelling.
3. Sort by ``train_loss_sample_shape_classifier`` to rank the most problematic
   samples.
4. Tag or discard them; the deny-aware sampler stops presenting them on the
   next training step.

.. tip::

   The base detection loop (Penn-Fudan dataset, downloaded automatically) is
   bundled with WeightsLab. The loss-shape classifier runs on top of it:

   .. code-block:: bash

      weightslab launch           # 1. deploy the studio
      weightslab start example --det # 2. start the detection demo

   To run the full loss-shape example (with the ``@wl.signal(subscribe_to=...)``
   classifier active), use the direct path above.


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/Usecases/wl-segmentation-loss-shapes-classification.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
