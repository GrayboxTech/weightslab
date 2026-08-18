Model Interaction
=================

Model interaction lets you introspect and control model behavior during training.

What you wrap
-------------

- Model object with ``wl.watch_or_edit(model, flag="model")``
- Optimizer with ``wl.watch_or_edit(optimizer, flag="optimizer")``
- Losses/metrics with ``wl.watch_or_edit(..., flag="loss"|"metric")``

Why it matters
--------------

- Observe training signals at batch/sample granularity.
- Watch the model's own training dynamics — gradients, weights, activations —
  per layer and per step (see `Training-dynamics signals`_).
- Keep a stable ledger/proxy handle across runtime updates.
- Enable dynamic controls without rewriting your loop architecture.

Minimal example
---------------

.. code-block:: python

   import weightslab as wl
   from torch import nn, optim

   model = wl.watch_or_edit(my_model, flag="model", device="cuda")
   optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=1e-3), flag="optimizer")

   train_loss = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss",
       name="train_loss/CE",
       per_sample=True,
       log=True,
   )

Training-dynamics signals
-------------------------

A loss curve tells you *whether* the model is learning. It does not tell you
**where** in the model something went wrong. Wrapping the model with
``track_model_signals=True`` adds that second view — one curve per layer, per
step, for the three quantities that explain most training failures:

.. code-block:: python

   model = wl.watch_or_edit(
       my_model,
       flag="model",
       device="cuda",
       track_model_signals=True,          # or a list, e.g. ["grad_norm"]
       model_signals_every_n_steps=1,     # raise to 10-50 on a large model
   )

That is the whole integration. Nothing is added to the training loop: gradients
are captured by post-accumulate hooks the moment they are final, activations by
forward hooks, and the set is flushed once per step just before
``optimizer.step()`` consumes it.

.. code-block:: text

   metrics/global/grad_norm                whole-model gradient L2 norm
   metrics/global/weights_norm             whole-model parameter L2 norm
   metrics/layer/<layer_id>/grad_norm      per-layer parameter gradients
   metrics/layer/<layer_id>/weights_norm   per-layer parameters
   metrics/layer/<layer_id>/activation_{mean,std,max,min}

``<layer_id>`` is the same module id the model panel and every architecture op
(freeze / reset / operate) use, so a curve that looks wrong names the layer you
then act on.

What each one catches:

- ``grad_norm`` collapsing toward 0 in the **early** layers while the late ones
  stay healthy is a vanishing gradient — the run keeps "training" and stops
  learning. Freeze or reinitialize from the layer where it dies.
- ``grad_norm`` spiking by orders of magnitude is the exploding case; compare
  against the loss curve to see which moved first.
- ``activation_std`` → 0 on a layer means that layer has gone constant (dead
  ReLUs, saturated BatchNorm). It is still consuming compute and contributing
  nothing.
- ``weights_norm`` climbing without bound while the loss flattens is the model
  growing weights instead of learning structure — time to add decay.

Collection only happens inside ``guard_training_context``, so an evaluation
pass can never contaminate these curves with values the optimizer never saw.
For a dynamics value of your own (a gradient-to-weight ratio, a custom norm),
write it directly with ``wl.save_model_signals({...})``.

Full reference: :ref:`track_model_signals <model-signals>`. Runnable example:
``examples/Usecases/wl-fashion-mnist-signals``.

Best practices
--------------

- Use explicit names for losses/metrics to keep logs readable.
- Prefer ``per_sample=True`` for losses when you need hard-example analysis.
- Keep model/device arguments explicit to avoid ambiguity in multi-device setups.
- Give each layer its own attribute (rather than burying it in an
  ``nn.Sequential``) if you want per-layer curves — a Sequential block resolves
  to a single layer id, and therefore a single curve.
- Raise ``model_signals_every_n_steps`` before dropping metrics: the activation
  forward hooks are the only per-step cost worth thinking about, and sampling
  every 10th step keeps the curves just as readable.
