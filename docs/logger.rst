Logger and Signals
==================

WeightsLab logger behavior is similar in spirit to TensorBoard: it tracks scalar evolution and per-sample context.

What gets logged
----------------

- Scalar signals (losses, metrics)
- Per-sample signal vectors
- Optional predictions/targets for deeper analysis

Start services
--------------

.. code-block:: python

   import weightslab as wl

   wl.serve(serving_cli=True, serving_grpc=True)

Custom signals
--------------

Use the signal decorator for static or dynamic signals.

.. code-block:: python

   import numpy as np
   import weightslab as wl

   @wl.signal(name="brightness")
   def brightness(ctx):
       image = ctx.image
       if image is None:
           return 0.0
       return float(np.mean(image))

   @wl.signal(name="weighted_train_loss", subscribe_to="train_loss/CE", compute_every_n_steps=10)
   def weighted_train_loss(ctx):
       if ctx.subscribed_value is None:
           return 0.0
       return 0.5 * float(ctx.subscribed_value)

Use in training loop
--------------------

.. code-block:: python

   wl.save_signals(
       signals={"train_loss/CE": loss},
       batch_ids=batch_ids,
       preds_raw=logits,
       targets=targets,
       step=step,
       log=True,
   )

Notes
-----

- Static signals can be batch-computed with ``wl.compute_signals(dataset_or_loader)``.
- Dynamic signals are triggered from subscribed metrics/signals.
- Call ``wl.keep_serving()`` at script end if you need services to remain available.
