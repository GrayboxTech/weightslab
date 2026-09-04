Custom Evaluation Function
===========================

When the CLI's ``evaluate`` command (or the UI's evaluate action, or the
agent) runs an evaluation, something has to actually iterate the loader and
compute the numbers. By default WeightsLab builds that runner for you out of
whatever signals you've already registered as ``flag="loss"``/``flag="metric"``
— but decorating your own function with ``@wl.eval_fn`` replaces that default
with your real evaluation logic.

The built-in default
----------------------

If no ``@wl.eval_fn`` decorator is applied, WeightsLab uses a built-in
default. For every batch it:

- unpacks ``(inputs, targets, ids)`` from the batch using a heuristic
  (tuple/list/dict — see :doc:`user_functions` for the exact field-name
  precedence it tries for each);
- runs the registered model in eval mode, under ``torch.no_grad()``;
- calls every ``flag="loss"``/``flag="metric"`` signal you've registered via
  ``wl.watch_or_edit`` with the batch's predictions and targets, letting each
  one log itself exactly as it would during training.

This is enough for a straightforward classification/regression loop where
the watched losses and metrics are already the whole story. It stops being
enough the moment your eval pass needs custom unpacking, a different metric
than what you log during training, or any logic beyond "run the model,
call the watched losses" — that's what the decorator is for.

Defining your own
-------------------

.. code-block:: python

   import weightslab as wl

   @wl.eval_fn
   def eval_pass(loader):
       model.eval()
       with torch.no_grad():
           for batch in loader:
               inputs, targets = batch[:2]
               preds = model(inputs)
               criterion(preds, targets)   # a watch_or_edit-wrapped loss logs itself

The decorated function receives one argument — a *managed loader* that wraps
the requested split and handles cancellation, timeout, and progress
reporting for you, so you just iterate it like any other loader. Inside the
loop, write the same evaluation code you'd write for a normal test pass:
run the model, and call whatever losses/metrics you registered with
``wl.watch_or_edit(..., flag="loss")`` or ``flag="metric"`` — any
``add_scalars``-style call made during the run is captured into the
evaluation-mode buffer automatically, the same mechanism the default runner
uses. Only one ``@wl.eval_fn`` can be registered at a time; applying the
decorator again replaces whatever was registered before.

.. tip::

   ``SignalContext`` (passed to custom signal functions) is shared between
   ``@wl.signal`` and ``@wl.eval_fn`` — see :doc:`signal_trajectory_classification`
   for the signal-wrapping side of this same mechanism.

Triggering it
---------------

Nothing about the decorator changes how evaluation gets *triggered* — that's
still the CLI's ``evaluate``/``eval_status`` commands (see :doc:`logger`),
the UI's evaluate action, or the agent asking for one in natural language.
Registering ``@wl.eval_fn`` only changes what runs once triggered.

For training-loop integration without a UI/CLI trigger, :func:`wl.run_pending_evaluation`
and :func:`wl.trigger_pending_evaluation_async` both resolve the registered
``@wl.eval_fn`` (falling back to the built-in default) automatically — see
:doc:`user_functions` for their full signatures, including how to pass an
explicit ``eval_fn=`` for one-off calls without registering it globally.
