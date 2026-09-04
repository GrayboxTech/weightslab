Data Exploration
================

Data exploration focuses on tracking sample IDs, tagging, discarding, filtering,
and exporting subsets while training.

Data wrapper parameters (``flag="data"``)
-----------------------------------------

Key ``wl.watch_or_edit(dataset, flag="data", ...)`` parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Behavior
   * - ``loader_name``
     - ``None``
     - Logical split name (for example ``train_loader``).
   * - ``batch_size`` / ``shuffle`` / ``num_workers``
     - ``1`` / ``False`` / ``0``
     - Loader runtime behavior.
   * - ``is_training``
     - ``False``
     - Enables deny-aware sampling logic for train split usage.
   * - ``compute_hash``
     - ``True``
     - Stable sample identifiers across runs.
   * - ``preload_labels`` / ``preload_metadata``
     - ``True`` / ``True``
     - Startup latency vs. runtime access trade-off.
   * - ``array_return_proxies`` / ``array_use_cache``
     - ``True`` / ``True``
     - Lazy array retrieval and cache behavior for large data.

For complete API details, see :doc:`user_functions` and :doc:`configuration`.

Core SDK calls
--------------

Typical data exploration functions:

- ``wl.tag_samples(sample_ids, tag, mode="add"|"remove")``
- ``wl.discard_samples(sample_ids, discarded=True|False)``
- ``wl.get_samples_by_tag(tag, origin=...)``
- ``wl.get_discarded_samples(origin=...)``
- ``wl.write_dataframe(path, format="json"|"csv", ...)``

``origin`` is the ``loader_name`` you registered (``"train_loader"``,
``"val_loader"``, ...); ``None`` searches every split.

Example:

.. code-block:: python

   import weightslab as wl

   wl.tag_samples([10, 42, 77], "hard_examples", mode="add")
   wl.discard_samples([5, 9], discarded=True)

   hard_ids = wl.get_samples_by_tag("hard_examples", origin="train_loader")
   discarded_ids = wl.get_discarded_samples(origin="train_loader")

   wl.write_dataframe(
       path="artifacts/hard_examples.csv",
       format="csv",
       sample_id=hard_ids,
       columns=["signals", "discarded", "tag:hard_examples"],
   )

Standalone data-only integration (UI + CLI ready)
-------------------------------------------------

A complete, runnable MNIST curation script with **no model at all**: it wraps the
two datasets, walks them once, tags a digit class, discards a slice, queries both
back, checks that discarded ids really leave the training batches, and exports the
result to CSV.

**Bundled example:** ``weightslab/examples/PyTorch/wl-standalone-data/main.py``

.. code-block:: bash

   weightslab start example --data     # run it (MNIST downloads on first run)
   weightslab cli                      # attach a terminal, in another shell
   weightslab start                    # open Weights Studio, in a third shell

.. literalinclude:: ../weightslab/examples/PyTorch/wl-standalone-data/main.py
   :language: python
   :pyobject: main

Notes on the wrapped loaders:

- A plain ``(image, label)`` dataset is enough. The wrapper injects the stable
  sample id, so the tracked loader yields ``(images, ids, targets)``.
- ``is_training=True`` turns on deny-aware sampling: a discarded id silently
  disappears from the batches, with no change to the loop.
- ``loader.wrapped_dataset`` is the tracking wrapper (``loader.dataset`` stays the
  raw dataset you passed in) and ``wrapped_dataset.unique_ids`` are the ids
  WeightsLab assigned.
- ``batch_size`` given here is authoritative; the hyperparameters config can
  override it later (``data.<loader_name>.batch_size``) but never has to exist.

CLI and UI surfaces
-------------------

CLI:

- ``list_loaders``
- ``list_uids [loader] [--discarded] [--limit N]`` — real sample ids, tags and
  discard state, read from the tracked sample dataframe
- ``discard <uid...>`` / ``undiscard <uid...>``
- ``add_tag <sample_id> <tag> ...``
- ``dump`` / ``ledger_dump``

UI:

- filter/sort the grid
- add/remove tags
- discard/restore rows
- inspect samples and metadata

Related automated tests (verified)
----------------------------------

Data tagging/discarding/query/export coverage:

- ``tests/general/test_four_way_standalone.py::TestDataLevelCli`` (the standalone
  above, driven through real CLI commands: ``list_uids``, ``add_tag``, ``discard``,
  ``undiscard``, ``dump``)
- ``tests/test_src_functions.py`` (tag/discard/get sample APIs)
- ``tests/gRPC/test_grpc_tag_operations.py`` (tag/discard flows through service layer)
- ``tests/trainer/services/test_data_service_sample_id_query.py`` (sample-id query correctness)
- ``tests/backend/test_write_dataframe.py`` (subset export behavior)
