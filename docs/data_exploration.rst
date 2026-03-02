Data Exploration
================

Data exploration focuses on identifying useful, noisy, or problematic samples while training.

Core ideas
----------

- Wrap your dataset/dataloader so sample IDs are tracked.
- Annotate data with tags.
- Temporarily discard problematic samples.
- Query subsets for review or targeted retraining.

Minimal example
---------------

.. code-block:: python

   import weightslab as wl

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=16,
       shuffle=True,
       compute_hash=True,
   )

   wl.tag_samples([10, 42, 77], "hard_examples", mode="add")
   wl.discard_samples([5, 9], discarded=True)

   hard_ids = wl.get_samples_by_tag("hard_examples", origin="train")
   discarded_ids = wl.get_discarded_samples(origin="train")

Workflow pattern
----------------

.. mermaid::

   flowchart TD
     A[Run Training] --> B[Inspect Signals]
     B --> C{Sample Quality?}
     C -- Poor --> D[Tag / Discard]
     C -- Good --> E[Keep]
     D --> F[Retrain]
     E --> F

Recommendations
---------------

- Start with a small tag vocabulary (for example: ``hard_examples``, ``noisy_label``).
- Keep discard operations reversible by tracking them in your experiment notes.
- Re-evaluate discarded sets periodically after model improvements.
