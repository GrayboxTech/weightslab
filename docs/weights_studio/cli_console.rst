WeightsLab CLI console
======================

The WeightsLab CLI console is a local developer REPL for inspecting and
controlling a running experiment through the global ledger.

Transport: local TCP text commands with JSON responses.

How to start it
===============

From your training script (recommended):

.. code-block:: python

  import weightslab as wl

  wl.serve(serving_grpc=True, serving_cli=True)
  wl.keep_serving()

Connect from a terminal::

  weightslab cli              # auto-discover port
  weightslab cli --port 60000 # or specify one

Console actions
===============

Full reference: :doc:`../user_commands`. Quick summary:

- Discovery/help: ``help``, ``status``, ``dump``, ``ledger_dump``.
- Training control: ``pause`` / ``resume``.
- Registry inspection: ``list_models``, ``list_optimizers``, ``list_loaders``,
  ``plot_model [model_name]``.
- Sample-level operations: ``list_uids``, ``discard``, ``undiscard``,
  ``add_tag``.
- Hyperparameters: ``hp``, ``set_hp``.
- Evaluation: ``evaluate``, ``eval_status``, ``cancel_eval``.
- Audit mode: ``audit [on|off]``.
- AI agent: ``agent`` / ``query`` / ``ask`` — see :doc:`../agent`.
- Experiment report: ``report`` — see :doc:`../experiment_reports`.
- Session control: ``exit`` / ``quit``, ``clear`` / ``cls``.
