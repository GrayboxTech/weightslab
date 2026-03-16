Quickstart
==========

Install docs dependencies
-------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt

Build docs once
---------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Serve docs with live reload (localhost)
---------------------------------------

.. code-block:: bash

   sphinx-autobuild docs docs/_build/html --host 127.0.0.1 --port 8000

Then open: http://127.0.0.1:8000

WeightsLab CLI console (dev quick reference)
--------------------------------------------

Start from your training script:

.. code-block:: python

   import weightslab as wl

   wl.serve(serving_grpc=True, serving_cli=True)
   wl.keep_serving()

Standalone server/client:

.. code-block:: bash

   python -m weightslab.backend.cli serve --host localhost --port 60000
   python -m weightslab.backend.cli client --host localhost --port 60000

Most useful commands:

- ``help``: list all commands and examples.
- ``status``: show registered models/loaders/optimizers/hyperparams.
- ``pause`` / ``resume``: toggle training state.
- ``list_uids [--discarded] [--limit N]``: inspect sample IDs.
- ``discard <uid...>`` / ``undiscard <uid...>``: update sample usage.
- ``add_tag <uid> <tag>``: tag one sample.
- ``hp`` / ``hp <name>`` / ``set_hp ...``: inspect/update hyperparameters.

For full CLI behavior and detailed action semantics, see ``weights_studio``.

What to read next
-----------------

- ``usecases``: commented end-to-end PyTorch integration example.
- ``pytorch_lightning``: Lightning integration with single and multi-GPU setup.
- ``weights_studio``: UI setup and operations (Docker, ports, controls, actions).
