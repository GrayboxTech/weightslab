.. _good-practice:

Good Practice
=============

Practical recommendations for running WeightsLab at scale — large datasets,
long experiments, and production-like setups.

.. toctree::
   :maxdepth: 1

   data_and_loaders
   training_loop
   signals

**Dataset and loaders** — keeping a large dataset off the critical path: the
``array_*`` loader flags, and implementing ``get_items`` so label scans don't
pay for an image decode.

**Training loop** — why the loop should run until you stop it, and what a fixed
step budget costs you.

**Signals and storage** — how much to send per step, and the three storage
modes to choose between.
