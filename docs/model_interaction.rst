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

.. code-block:: python

   import weightslab as wl
   import torch
   import torch.optim as optim

   model = wl.watch_or_edit(
       my_model,
       flag="model",
       device="cuda" if torch.cuda.is_available() else "cpu",
       dummy_input=example_batch,
       compute_dependencies=True,
       use_onnx=False,
   )
   optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=1e-3), flag="optimizer")

   # Start both integration surfaces
   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)

   # Minimal train/eval routing
   with wl.guard_training_context:
       preds = model(train_inputs)
   with wl.guard_testing_context:
       preds_eval = model(val_inputs)

   wl.keep_serving()

CLI and UI surfaces
-------------------

CLI:

- ``status``, ``list_models``, ``plot_model``
- ``pause`` / ``resume``
- ``agent query freeze layer ...`` / ``reset layer ...`` / ``unfreeze ...``

UI:

- model architecture and layer inspection
- model operations through controls/agent
- version/load interactions via experiment state

Related automated tests (verified)
----------------------------------

Model wrapping, graph/dependency analysis, and operations are covered by:

- ``tests/model/test_dependency_patterns.py`` (TorchFX and ONNX dependency paths)
- ``tests/model/test_model_with_ops.py`` (ADD/PRUNE/FREEZE/RESET behavior)
- ``tests/model/test_model_with_ops_unit.py`` (model ops utility behavior)
- ``tests/backend/test_model_interface_unit.py`` (model interface unit behavior)
