Model Interaction
=================

Model interaction focuses on wrapping a model, understanding its graph/dependency
structure, and applying architecture operations at runtime.

Model wrapping parameters (``flag="model"``)
--------------------------------------------

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
``model.operate(...)``:

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

Operation behavior:

- ``ADD``: expands layer capacity
- ``PRUNE``: removes selected neurons/channels
- ``FREEZE``: disables updates for selected neurons/channels
- ``RESET``: reinitializes selected weights

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

Related automated tests (verified)
----------------------------------

Model wrapping, graph/dependency analysis, and operations are covered by:

- ``tests/general/test_four_way_standalone.py::TestModelLevelCli`` (the standalone
  above, driven through real CLI commands)
- ``tests/model/test_dependency_patterns.py`` (TorchFX and ONNX dependency paths)
- ``tests/model/test_model_with_ops.py`` (ADD/PRUNE/FREEZE/RESET behavior)
- ``tests/model/test_model_with_ops_unit.py`` (model ops utility behavior)
- ``tests/backend/test_model_interface_unit.py`` (model interface unit behavior)
