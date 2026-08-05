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

Model graph and editing API
---------------------------

Model editing needs dependency information so a change to one layer can be
propagated to connected layers. Enable it when wrapping the model:

.. code-block:: python

   model = wl.watch_or_edit(
       my_model,
       flag="model",
       dummy_input=example_batch,
       compute_dependencies=True,
   )

   graph = model.get_model_graph()
   first_layer = model.get_layer_info(graph["layers"][0]["id"])

``get_model_graph()`` returns JSON-serializable primitives with a schema
version, model metadata, layers, and directed dependencies. Each dependency is
one of:

- ``SAME``: both layers expose the same neuron/channel dimension, such as a
  convolution followed by batch normalization.
- ``INCOMING``: the source output feeds the destination input, such as one
  linear layer feeding another.
- ``REC``: a recursive or skip-connection relationship that must be kept in
  sync during structural edits.

The structural graph omits per-neuron records by default to keep inspection
cheap. Use ``get_model_graph(include_neurons=True)`` or
``get_layer_info(layer_id)`` for learning-rate and frozen-state details.

The model exposes dependency-aware modifiers:

.. code-block:: python

   model.freeze_neurons(layer_id=0, neuron_indices=[0, 2])
   model.unfreeze_neurons(layer_id=0, neuron_indices=[0])
   model.reset_neurons(layer_id=0, neuron_indices=[1])
   model.perturb_neurons(layer_id=0, neuron_indices=[2], ratio=0.1)
   model.add_neurons(layer_id=0, count=2)
   model.prune_neurons(layer_id=0, neuron_indices=[3])

Omitting ``neuron_indices`` freezes, unfreezes, resets, or perturbs the whole
layer. Layer ids are stable only for the lifetime of the wrapped model; resolve
them from the graph instead of persisting them as checkpoint identifiers.

Best practices
--------------

- Use explicit names for losses/metrics to keep logs readable.
- Prefer ``per_sample=True`` for losses when you need hard-example analysis.
- Keep model/device arguments explicit to avoid ambiguity in multi-device setups.
- Pause training before structural edits so model and optimizer updates happen
  at a safe boundary.
