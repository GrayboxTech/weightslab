Experiment Agent Assistant
==========================

The WeightsLab agent translates natural-language requests into safe data/model
operations on your live experiment.

Where you can use it
--------------------

From Weights Studio (UI)
~~~~~~~~~~~~~~~~~~~~~~~~

- Use the chat bar in the Studio header
- Initialize provider with ``/init``
- Switch model with ``/model``
- Reset runtime state with ``/reset``

From CLI
~~~~~~~~

Use ``weightslab cli`` and run:

.. code-block:: text

   agent status
   agent init --api-key sk-or-... --model google/gemini-flash-latest
   agent query tag train samples with loss > 1.2 as hard_examples
   agent query freeze layer 3

From Python code
~~~~~~~~~~~~~~~~

Start services first so the agent backend is available:

.. code-block:: python

   import weightslab as wl

   # Start WeightsLab services
   wl.serve(serving_grpc=True, serving_cli=True)
   wl.keep_serving()

Prompt examples
---------------

Data exploration prompts:

- ``Sort by train loss, highest first``
- ``Show only validation samples with loss > 2``
- ``Tag samples with train loss > 1.5 as hard_examples``
- ``Discard samples tagged hard_examples``

Model prompts:

- ``Show me the complete model details``
- ``Which layers are currently frozen?``
- ``Freeze layer 3``
- ``Reset layer 3``
- ``Unfreeze layer 3``

Config/report prompts:

- ``Set batch size to 32``
- ``Increase learning rate by 10%``
- ``Generate an experiment report on train_loss and val_loss``

Safeguards
----------

The assistant enforces safe execution rules:

- No row deletion (drop/remove requests are mapped to ``discarded`` controls)
- No in-place overwrite of existing data columns
- Control-column writes only (``discarded`` and ``tag:*``)
- Read-only handling for analysis questions
- Clear failure messages when a request is ambiguous or unsupported

How it fits with core concepts
------------------------------

The agent is optional and sits on top of the four core levels:

- model interaction
- data exploration
- config management
- logger/signals

You can use those levels directly from SDK/CLI/UI even without the agent.
