.. _embedded-notebook:

Embedded experiment notebook
============================

.. figure:: ../_static/screenshots/notebook-panel.png
   :alt: The embedded experiment notebook panel
   :width: 100%

Weights Studio has a Jupyter-like notebook panel built into the UI itself,
opened via the notebook button just left of the logo. Unlike a standalone
Jupyter server, it runs in a **shared in-process kernel inside the training
backend** — every cell sees the exact same live objects your training script
does (the tracked dataframe ``df``, the model, the checkpoint manager, and
the live hyperparameters dict), with no serialization or IPC in between.

.. note::

   This is a different feature from the **Local Jupyter Notebook** button on
   the "no backend connected" landing page, which launches a real, standalone
   ``jupyter notebook`` server process instead. Use the embedded panel below
   to interact with an experiment that's already training; use the
   landing-page button to bootstrap a brand-new experiment from a notebook
   before any backend exists at all.

How it works
-------------

- The button is disabled until a backend connects, then becomes clickable.
- The notebook document persists as ``notebook.ipynb`` under the experiment's
  ``root_log_dir``. Reopening the panel — even after restarting the UI,
  as long as it points at the same experiment — reloads the same cells,
  their source, and their last-run outputs.
- Every cell runs against the training process's ONE shared kernel: only one
  cell executes at a time. Clicking Run on a second cell while another is
  still running queues it rather than firing a second concurrent execution.
- Running a code cell streams its output live as it's produced: stdout/stderr
  (merged, in order), the value of the last expression, any ``matplotlib``
  figures rendered inline as images, and a full traceback on error.
- A run can be interrupted mid-flight with the stop button next to the cell.

Cell types
----------

Cells can be **code** or **markdown** — toggle a cell's type with the small
button in its gutter:

- **Code cells** execute against the shared kernel as described above.
- **Markdown cells** render to formatted HTML (headings, bold/italic, lists,
  links, blockquotes, fenced code) when run. Double-click the rendered view
  (or run the cell again) to drop back into the raw source for editing.

Asking the agent for code
--------------------------

A cell whose source starts with ``>`` is not executed as Python — it's sent
to the AI agent as a natural-language request for code:

.. code-block:: text

   > Compute the average training loss per sample over the last 100 steps,
   > with a progress bar.

The agent's proposed code replaces the cell's contents; review it, then run
it like any other cell. Press Enter at the end of a ``>`` line to continue
the same prompt onto a new line; press Enter again on an empty ``>`` line to
drop the marker and finish the prompt. Any plain code left in the same cell
below the ``>`` lines is sent to the agent as extra context, not executed.

If a cell's last run raised an error, an **"AI" debug button** appears on its
output — click it to send the code and traceback back to the agent and ask
for a fix, without retyping it as a ``>`` prompt yourself.

Example
-------

A typical first cell against a live experiment:

.. code-block:: python

   df.head()

Followed by, in a second cell:

.. code-block:: text

   > Plot a histogram of the per-sample loss for the current epoch,
   > highlighting samples tagged "hard_examples" in red.

Running that second cell doesn't execute anything yet — it fills the cell
with the agent's generated ``matplotlib`` code, which you then run to see
the plot rendered inline in the cell's output.

What a cell can actually do
----------------------------

Every cell shares the *same live objects* the training loop uses — but "same
objects" doesn't mean "same guardrails" for every one of them. Concretely:

- **Reading anything is unrestricted** — ``df``, ``model``, ``hp`` (the live
  hyperparameters dict), the checkpoint manager, and any importable module in
  the process are all fair game.
- **Tagging, discarding, and evaluation work directly** — ``wl.tag_samples(...)``,
  ``wl.discard_samples(...)``, ``wl.set_categorical_tag(...)``,
  ``wl.run_pending_evaluation(...)`` etc. mutate the live ledger/dataframe
  immediately, no different from calling them in your training script.
- **Editing ``hp`` in place changes training** — e.g. ``hp['lr'] = 0.001``
  genuinely takes effect the next time the training loop reads that key,
  since it's the exact same dict object, not a copy.
- **Setting ``hp['is_training'] = False`` does *not* pause training.** This is
  a real gap, not a design choice you're missing: the sync path that would
  drive the pause controller from that flag isn't wired up. Use the UI's
  play/pause button, or the chat agent, to actually pause/resume a run — a
  notebook cell can reach the same effect only by importing the pause
  controller directly (``from weightslab.components.global_monitoring import
  pause_controller; pause_controller.pause()``), which works but bypasses the
  namespace WeightsLab seeds for you.
- **Filesystem writes are sandboxed to the experiment directory.** A cell
  that opens a file for writing, or deletes/moves/renames one, is silently
  redirected under ``root_log_dir`` if the path it named was outside it.
  This is a best-effort guard against an accidental ``rm -rf`` or a stray
  absolute path in generated code — **not** a security boundary against a
  user deliberately trying to escape it (nothing stops importing ``os`` and
  working around it), and it only applies to files, not to any other
  capability listed above.

.. note::

   The notebook button (like the report-generation button) is disabled in
   sandbox mode.

Saving and running everything
------------------------------

- **Save**: writes the current notebook (source + outputs) to
  ``notebook.ipynb``. This also happens automatically right after any cell
  finishes running (a code execution or a markdown render), so you rarely
  need to click it by hand.
- **Run all**: runs every cell top to bottom, strictly one at a time (never
  concurrently), skipping ``>``-prefixed agent-prompt cells; saves once more
  when the whole pass finishes.
- **Rename**: renaming picks a non-colliding filename automatically if one
  already exists.

Window controls
----------------

The notebook opens as a floating, draggable window: drag its header to move
it, drag an edge or corner to resize, or use the maximize button to fill the
viewport and restore it back to its previous size and position.

Turning it off
---------------

Set ``ENABLE_NOTEBOOK=0`` before ``weightslab start`` to remove both the
button and the window entirely (dev server: ``VITE_ENABLE_NOTEBOOK`` — see
the *Frontend runtime feature toggles* table above).
