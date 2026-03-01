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

Best practices
--------------

- Use explicit names for losses/metrics to keep logs readable.
- Prefer ``per_sample=True`` for losses when you need hard-example analysis.
- Keep model/device arguments explicit to avoid ambiguity in multi-device setups.
