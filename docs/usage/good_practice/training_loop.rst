.. _good-practice-open-ended-loop:

Training loop
=============



Write your training loop so it runs **until you stop it** — not for a
predefined number of steps. Use ``itertools.count()`` (or ``while True``), and
let the studio's Pause button, the CLI, or ``Ctrl+C`` decide when it ends:

.. code-block:: python

   import itertools

   for train_step in itertools.count():   # not: for step in range(n_steps)
       ...

.. warning::

   A ``range(training_steps_to_do)`` loop **ends the process** the moment the
   budget is spent. When the process exits, the gRPC backend goes with it: the
   studio drops to "no backend connected", the notebook's shared kernel dies,
   the agent loses the experiment, and the only way back is to restart
   WeightsLab and reload from a checkpoint. There is no "resume" button for a
   process that is no longer running.

Why this matters more here than in a normal training script: WeightsLab is
built around **staying in the experiment**. You watch the curves, spot a
signal going flat, sort the grid by loss, discard or retag the samples doing
the damage, freeze a layer, change the learning rate — and keep going, with
the same live objects and the same history. A step budget cuts that loop off
mid-thought, usually at the least convenient moment, because the number was
chosen before you knew what the run would look like.

.. note::

   ``training_steps_to_do`` is still a useful hyperparameter — it remains live
   and editable in the studio, and it drives the UI's own "run N more steps"
   control. Just don't use it as the bound of your ``for`` loop. It is a
   **target you can change while training**, not a ceiling on the process.

To stop cleanly, use whichever of these fits:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - To do this
     - Use
   * - Pause, keep the process alive
     - The studio's **Pause** button, or ``pause`` in the CLI console. Training
       stops; the backend, notebook kernel, and agent all stay up.
   * - Idle after the loop ends
     - ``wl.keep_serving()`` after the loop — keeps the process (and the whole
       studio session) alive so you can still inspect and export.
   * - Stop for real
     - ``Ctrl+C``, or ``wl.keep_serving(timeout=...)`` for an unattended run.

Every bundled example already follows this pattern — see
``weightslab/examples/PyTorch/wl-classification/main.py``, which iterates
``itertools.count()`` and finishes with ``wl.keep_serving()``.
