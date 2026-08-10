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

- ``wl.tag_samples(sample_ids, tag_name, mode="add"|"remove")``
- ``wl.discard_samples(sample_ids, discarded=True|False)``
- ``wl.get_samples_by_tag(tag_name, origin=...)``
- ``wl.get_discarded_samples(origin=...)``
- ``wl.write_dataframe(path, format="json"|"csv", ...)``

Example:

.. code-block:: python

   import weightslab as wl

   wl.tag_samples([10, 42, 77], "hard_examples", mode="add")
   wl.discard_samples([5, 9], discarded=True)

   hard_ids = wl.get_samples_by_tag("hard_examples", origin="train")
   discarded_ids = wl.get_discarded_samples(origin="train")

   wl.write_dataframe(
       path="artifacts/hard_examples.csv",
       format="csv",
       sample_ids=hard_ids,
       columns=["signals", "discarded", "tag:hard_examples"],
   )

Standalone data-only integration (UI + CLI ready)
-------------------------------------------------

.. code-block:: python

   import weightslab as wl

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=16,
       shuffle=True,
       is_training=True,
       compute_hash=True,
       preload_labels=False,
   )

   val_loader = wl.watch_or_edit(
       val_dataset,
       flag="data",
       loader_name="val_loader",
       batch_size=16,
       shuffle=False,
   )

   # Start both integration surfaces
   wl.serve(serving_grpc=True, serving_cli=True)
   wl.start_training(timeout=3)
   wl.keep_serving()

CLI and UI surfaces
-------------------

CLI:

- ``list_uids [loader] [--discarded] [--limit N]``
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

- ``tests/test_src_functions.py`` (tag/discard/get sample APIs)
- ``tests/gRPC/test_grpc_tag_operations.py`` (tag/discard flows through service layer)
- ``tests/trainer/services/test_data_service_sample_id_query.py`` (sample-id query correctness)
- ``tests/backend/test_write_dataframe.py`` (subset export behavior)
