Model Signals on Fashion-MNIST
===============================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--usecase">Usecase</span>
     <span class="wl-eg-tag">model signals</span>
     <span class="wl-eg-tag">gradient norm</span>
     <span class="wl-eg-tag">activations</span>
     <span class="wl-eg-tag">per-layer</span>
     <span class="wl-eg-tag">training dynamics</span>
   </div>

**Example:** ``weightslab/examples/Usecases/wl-fashion-mnist-signals``

This use case trains a small CNN on Fashion-MNIST and adds one thing on top of
the plain per-sample logging: the run plots **its own training dynamics**. A
loss curve tells you whether the model is learning; these curves tell you
*where* in the model something went wrong.

Everything below comes from one argument.

The integration
---------------

.. code-block:: python

   model = wl.watch_or_edit(
       FashionCNN(),
       flag="model",
       device=device,
       track_model_signals=True,          # <- the whole feature
       model_signals_every_n_steps=1,
   )

No hooks to write, and **no call anywhere in the training loop** — the loop is
byte-for-byte the same as ``wl-classification``'s. Pass a list instead of
``True`` to narrow the set, e.g. ``track_model_signals=["grad_norm",
"activation_std"]``.

What gets plotted
-----------------

.. code-block:: text

   metrics/global/grad_norm                whole-model gradient L2 norm
   metrics/global/weights_norm             whole-model parameter L2 norm
   metrics/layer/<layer_id>/grad_norm      per-layer parameter gradients
   metrics/layer/<layer_id>/weights_norm   per-layer parameters
   metrics/layer/<layer_id>/activation_mean
   metrics/layer/<layer_id>/activation_std
   metrics/layer/<layer_id>/activation_max
   metrics/layer/<layer_id>/activation_min

Layers with parameters get all eight; parameter-free layers (``ReLU``,
``MaxPool2d``) get the four activation curves only. Containers and shape-only
ops (``Sequential``, ``Flatten``, ``Identity``, ``Dropout``) are skipped, since
their output statistics duplicate the layer before them.

For the model in this example — three conv blocks and a two-layer head — that
is 74 curves: 14 layers × 4 activation stats, 8 parameterized layers × 2 norms,
and the 2 global norms.

The layer legend
----------------

``metrics/layer/7/grad_norm`` says nothing on its own, so the example prints the
mapping at startup:

.. code-block:: text

    layer_id  module          shape
           1  Conv2d          (16, 1, 3, 3)
           2  BatchNorm2d     (16,)
           3  ReLU            -
           4  MaxPool2d       -
           5  Conv2d          (32, 16, 3, 3)
           6  BatchNorm2d     (32,)
           7  ReLU            -
           8  MaxPool2d       -
           9  Conv2d          (64, 32, 3, 3)
          10  BatchNorm2d     (64,)
          11  ReLU            -
          12  Flatten         -
          13  Linear          (128, 3136)
          14  ReLU            -
          15  Linear          (10, 128)

These are the same ids the model panel and every architecture op (freeze /
reset / operate) use — so a curve that looks wrong names the layer you then act
on, whether from the UI, the CLI, or the agent.

Note that every module in this example's model is a **named attribute** rather
than a member of an ``nn.Sequential``. That is deliberate: a Sequential block
resolves to one layer id, and therefore one curve, which defeats the purpose of
per-layer signals.

Reading the curves
------------------

Fashion-MNIST is small enough to make each failure mode legible:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - What you see
     - What it means
   * - ``grad_norm`` collapsing toward 0 in the **early** layers while the late
       ones stay healthy
     - Vanishing gradient. The run keeps "training" and stops learning. Act
       from the layer where it dies.
   * - ``grad_norm`` spiking by orders of magnitude
     - Exploding gradient. Compare against the loss curve to see which moved
       first.
   * - ``activation_std`` → 0 on a layer
     - That layer has gone constant (dead ReLUs, saturated BatchNorm). Still
       consuming compute, contributing nothing.
   * - ``activation_min`` pinned at exactly 0.0 across a whole ReLU
     - The same story from the other side — nothing is getting through.
   * - ``weights_norm`` climbing without bound while the loss flattens
     - The model is growing weights instead of learning structure. Add decay.

Cost, and how it is kept low
----------------------------

Three things keep the per-step overhead small enough to leave on by default:

- **Activations are reduced on-device** into 0-d tensors and held there. The
  whole step costs *one* host↔device sync no matter how many layers are
  tracked.
- **Gradients are captured by post-accumulate hooks**, so nothing walks the
  parameter list a second time — and nothing depends on where your loop calls
  ``optimizer.zero_grad()``.
- **``model_signals_every_n_steps``** samples every Nth step. On a large model,
  10–50 makes the cost negligible while the curves stay just as readable. Reach
  for this before dropping metrics.

Collection only happens inside ``guard_training_context``, so the evaluation
pass contributes nothing — a gradient or activation curve never contains values
the optimizer did not see. This holds even for eval loops that skip
``model.eval()`` or ``torch.no_grad()``.

Custom dynamics values
----------------------

``track_model_signals`` is a collector over ``wl.save_model_signals``, which is
the step-keyed write path in its own right. Use it directly for anything the
collector does not compute:

.. code-block:: python

   # gradient-to-weight ratio: how big a step is this, relative to the weights?
   wl.save_model_signals({
       "metrics/global/update_ratio": grad_norm / (weight_norm + 1e-12),
       "metrics/global/lr": optimizer.param_groups[0]["lr"],
   })

See :ref:`save_model_signals <model-signals>` for the full reference, and
:doc:`../../model_interaction` for how these fit alongside the rest of the
model surface.
