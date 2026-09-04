Model Interaction
=================

Model interaction focuses on wrapping a model, understanding its graph/dependency
structure, and applying architecture operations at runtime.

.. code-block:: python

   import weightslab as wl
   model = wl.watch_or_edit(my_model)


Model wrapping parameters (``flag="model"``)
--------------------------------------------

- Observe training signals at batch/sample granularity.
- Watch the model's own training dynamics — gradients, weights, activations —
  per layer and per step (see `Training-dynamics signals`_).
- Keep a stable ledger/proxy handle across runtime updates.
- Enable dynamic controls without rewriting your loop architecture.

Key ``wl.watch_or_edit(model, flag="model", ...)`` parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Behavior
   * - ``dummy_input``
     - ``None``
     - Example tensor/dict used to trace graph structure.
   * - ``device``
     - ``None``
     - Target device (for example ``"cpu"``, ``"cuda"``, ``"cuda:0"``).
   * - ``compute_dependencies``
     - ``False``
     - Computes layer dependency metadata used by advanced graph operations.
   * - ``use_onnx``
     - ``False``
     - Uses ONNX export path for graph/dependency analysis.
   * - ``opset_version``
     - ``17``
     - ONNX opset version when ``use_onnx=True``.
   * - ``print_graph``
     - ``False``
     - Prints traced graph output for debugging.
   * - ``forced_model_wrapping``
     - ``False``
     - Forces a fresh wrapper even when a checkpoint already contains one.
   * - ``log_model_signals``
     - ``True``
     - Logs ``model/grad_norm`` and ``model/parameters`` once per training step, so
       the model level produces experiment history on its own (see below).

For full API signatures, see :doc:`user_functions`.

With and without ``compute_dependencies``
-----------------------------------------

.. warning::

   Unstable: The dependency graph is still under development.
   It is not yet guaranteed to be correct for all models, and it may change in future releases.
   Please report any issues you encounter.

.. note::

   The dependency graph is used to determine which layers are affected by an
   architecture operation. If you do not need these operations, you can skip
   computing it for a faster wrapping.

Fast wrapping (no dependency graph):

.. code-block:: python

   import weightslab as wl

   model = wl.watch_or_edit(
       my_model,
       flag="model",
       device="cuda",
       compute_dependencies=False,
   )

Dependency-aware wrapping (TorchFX style):

.. code-block:: python

   model = wl.watch_or_edit(
       my_model,
       flag="model",
       device="cuda",
       dummy_input=example_batch,
       compute_dependencies=True,
       use_onnx=False,
   )

Dependency-aware wrapping (ONNX style):

.. code-block:: python

   model = wl.watch_or_edit(
       my_model,
       flag="model",
       device="cuda",
       dummy_input=example_batch,
       compute_dependencies=True,
       use_onnx=True,
       opset_version=17,
   )

Runtime architecture operations
-------------------------------

The wrapped model interface supports architecture operations through
``model.operate(...)``. Operation behavior:

- ``ADD``: expands layer capacity
- ``PRUNE``: removes selected neurons/channels
- ``FREEZE``: disables updates for selected neurons/channels
- ``RESET``: reinitializes selected weights

.. code-block:: python

   from weightslab.modules.neuron_ops import ArchitectureNeuronsOpType

   # Add one neuron to layer 0
   model.operate(
       0,
       neuron_indices=-1,
       op_type=ArchitectureNeuronsOpType.ADD,
   )

   # Prune selected neurons from layer 0
   model.operate(
       0,
       neuron_indices={1, 3, 5},
       op_type=ArchitectureNeuronsOpType.PRUNE,
   )

   # Freeze neurons in layer 0
   model.operate(
       0,
       neuron_indices={2, 4},
       op_type=ArchitectureNeuronsOpType.FREEZE,
   )

   # Reset selected neurons in layer 0
   model.operate(
       0,
       neuron_indices={2, 4},
       op_type=ArchitectureNeuronsOpType.RESET,
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
- Pause training before structural edits so model and optimizer updates happen
  at a safe boundary.
- Give each layer its own attribute (rather than burying it in an
  ``nn.Sequential``) if you want per-layer curves — a Sequential block resolves
  to a single layer id, and therefore a single curve.
- Raise ``model_signals_every_n_steps`` before dropping metrics: the activation
  forward hooks are the only per-step cost worth thinking about, and sampling
  every 10th step keeps the curves just as readable.

Standalone model-only integration (UI + CLI ready)
--------------------------------------------------

A complete, runnable MNIST script that wraps **only** the model and its
optimizer. Data loading is plain ``torch.utils.data``, the loss is a plain
``nn.CrossEntropyLoss``, and no hyperparameters are registered — the model level
alone drives the CLI and the studio.

**Bundled example:** ``weightslab/examples/PyTorch/wl-standalone-model/main.py``

.. code-block:: bash

   weightslab start example --model    # run it (MNIST downloads on first run)
   weightslab cli                      # attach a terminal, in another shell
   weightslab start                    # open Weights Studio, in a third shell

.. literalinclude:: ../weightslab/examples/PyTorch/wl-standalone-model/main.py
   :language: python
   :pyobject: main

The excerpt is the example's ``main()``; the file also contains the ``SmallCNN``
definition, the plain loader builder and the ``apply_architecture_op`` helper it
calls.

Signals the model level logs by itself
--------------------------------------

Every ``wl.guard_training_context`` step, the wrapped model logs its own signals
into the experiment history:

.. list-table::
   :header-rows: 1

   * - Signal
     - Meaning
   * - ``model/grad_norm``
     - Global L2 norm of the gradients that were just computed.
   * - ``model/parameters``
     - Trainable parameter count — it steps whenever an architecture operation
       resizes a layer.

Those are what make a model-only run non-empty in the studio, in
``report``, and in ``wl.write_history()``: nothing else writes to the logger until
a loss or metric is wrapped, which is the :doc:`logger` level. Pass
``log_model_signals=False`` to skip the per-step work.

Two details make the level self-sufficient:

- ``WEIGHTSLAB_ROOT_LOG_DIR`` gives the run an experiment directory without the
  config level. It is the same variable ``weightslab start [DIR]`` exports, so a
  model-only run lands where the UI looks.
- ``wl.serve()`` warns (rather than fails) about the missing serving config, and
  ``wl.start_training()`` can leave the paused state because only the levels you
  actually registered are waited on.

.. note::

   ``FREEZE`` and ``RESET`` keep layer shapes, so the loop above trains straight
   through them — that is why ``--op freeze`` is the example's default. ``ADD``
   and ``PRUNE`` do resize the layer (the printed parameter count proves it), but
   the autograd graph of an already-running loop still refers to the pre-op
   tensors, so the backward passes right after them are dropped by the guard.
   Apply shape-changing operations from Weights Studio / the agent, or restart the
   loop afterwards.

CLI and UI surfaces
-------------------

CLI:

- ``status``, ``list_models``, ``list_optimizers``, ``plot_model``
- ``pause`` / ``resume``
- ``agent query freeze layer ...`` / ``reset layer ...`` / ``unfreeze ...``

UI:

- model architecture and layer inspection
- model operations through controls/agent
- version/load interactions via experiment state

