.. _good-practice:

Good Practice
=============

Practical recommendations for running WeightsLab at scale — large datasets,
long experiments, and production-like setups.

.. _good-practice-heavy-experiment:

i. Heavy-experiment loader flags
---------------------------------

When your dataset is large, dense annotation arrays (bounding boxes, masks,
point clouds) can easily overflow RAM if they are all preloaded at startup.
Use the three ``array_*`` flags together with ``preload_labels=False`` to defer
all array materialisation until it is actually needed:

.. code-block:: python

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=8,
       shuffle=True,
       is_training=True,
       # Don't store the dense arrays (predictions and ground truth) in RAM;
       # only keep their file paths with a small LRU cache for recent ones.
       array_autoload_arrays=False,
       array_return_proxies=True,
       array_use_cache=True,
       # Load labels on demand — don't scan every annotation at startup.
       preload_labels=False,
   )

What each flag does:

``array_autoload_arrays=False``
   Annotation arrays written by signal calls are **not** read back into
   RAM during startup or ledger queries. Only their storage paths are
   kept in the dataframe. Use this whenever per-sample arrays are
   larger than ~1 MB or the dataset has more than a few thousand samples.

``array_return_proxies=True``
   Reads of array columns return lazy ``ArrayProxy`` objects that load
   the underlying file only when ``.numpy()`` / ``.__array__()`` is called.
   Studio thumbnails and overlays trigger this load just-in-time, so only
   the arrays currently visible in the UI are in memory.

``array_use_cache=True``
   Recently accessed arrays are kept in a small LRU cache. This is crucial
   when the studio repeatedly accesses the same samples (e.g. a zoomed-in
   view of 20 images), avoiding redundant disk reads.

``preload_labels=False``
   Labels (bounding boxes, masks, class IDs) are read inside
   ``__getitem__`` on demand rather than scanned for all samples at init.
   Use this for datasets where label parsing is expensive or where not all
   samples will be visited in a single run.

.. note::

   For **light experiments** (small datasets, fast iteration), you can
   skip these flags. The default behaviour (preloaded, in-memory) is simpler
   and has lower per-batch latency.

.. _good-practice-get-items:

ii. Implementing ``get_items`` in your dataset class
-----------------------------------------------------

WeightsLab occasionally needs to access a sample's metadata or label without
loading the full image (e.g. computing class weights, building a histogram of
annotation counts). Implement ``get_items`` with optional loading of each
component so callers can request only what they need:

.. code-block:: python

   class MyDataset(Dataset):
       ...

       def get_items(self, idx,
                     include_metadata=False,
                     include_labels=False,
                     include_images=False):
           img_path  = self.images[idx]
           mask_path = self.masks[idx]
           uid = os.path.splitext(os.path.basename(img_path))[0]

           metadata = None
           if include_metadata:
               metadata = {
                   "img_path":  img_path,
                   "mask_path": mask_path,
               }

           img_t = None
           if include_images:
               img = Image.open(img_path).convert("RGB")
               img_t = self.image_transform(img)

           target = None
           if include_labels:
               target = self._load_boxes(mask_path)

           return img_t, uid, target, metadata

**Why this matters:** without ``get_items``, any WeightsLab utility that
scans annotations (class-weight computation, distribution analysis, label
preloading) is forced to run the full ``__getitem__`` pipeline — including
image decode, resize, and augmentation — even though it only needs the label.
On a large dataset this can cost minutes at startup.

.. warning::

   When ``include_images`` and ``include_labels`` are requested in separate
   ``get_items`` calls (as in the pattern below), any *random* augmentation
   (random crop, flip, etc.) must not be re-sampled independently on each
   call — otherwise the transform applied to the image and the transform
   applied to its annotations will diverge, silently misaligning boxes/masks
   with the image they describe. Derive the augmentation deterministically
   per sample (e.g. a seed keyed by ``uid``/``idx``), or sample it once and
   cache it — for instance in ``metadata`` — so every subsequent
   ``get_items`` call for that sample reuses the same transform instead of
   drawing a new random one.

Usage pattern:

.. code-block:: python

   # Compute class weights without loading any images
   for idx in range(len(train_dataset)):
       _, uid, target, _ = train_dataset.get_items(idx, include_labels=True)
       count_annotations(target)

   # Load a thumbnail for the studio without re-running augmentation
   img_t, uid, _, _ = train_dataset.get_items(idx, include_images=True)

The standard return order is ``(image, uid, target, metadata)``, mirroring
what the ``DataSampleTrackingWrapper`` yields from ``__iter__``.

iii. Signal storage mode — choosing what to send
-------------------------------------------------

Choose this based on storage budget and how often you need overlays during
training.

**Light mode** — train keeps only per-sample loss, eval keeps full data:

.. code-block:: python

   # Training step: store per-sample loss only
   train_loss = sig["loss"](outputs, targets, batch_ids=ids)

   # Evaluation step: store predictions + targets for overlay analysis
   eval_preds = decode_and_nms(outputs.detach())
   eval_loss = sig["loss"](outputs, targets, batch_ids=ids, preds=eval_preds, targets=targets)

Use this when you want lighter train-time writes but still need rich eval-time
inspection in Studio.

**Standard mode** — both train and eval store full data:

.. code-block:: python

   # Training step
   train_preds = decode_and_nms(outputs.detach())
   train_loss = sig["loss"](outputs, targets, batch_ids=ids, preds=train_preds, targets=targets)

   # Evaluation step
   eval_preds = decode_and_nms(outputs.detach())
   eval_loss = sig["loss"](outputs, targets, batch_ids=ids, preds=eval_preds, targets=targets)

``preds`` should be **processed** predictions (after NMS, argmax, etc.) rather
than raw model outputs, because the studio renders them directly as overlays.
The optional ``targets`` override is useful when the annotation fed to the loss
function differs from the one the studio should display (e.g. encoded anchors
vs. decoded boxes).

Summary table
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Mode
     - Call
     - Stores
     - Use when
   * - Light
     - train: ``sig(out, tgt, batch_ids=ids)`` / eval: ``sig(out, tgt, batch_ids=ids, preds=preds, targets=tgt)``
     - train: loss only / eval: loss + predictions + targets
     - Large or medium datasets, lower write cost during training
   * - Standard
     - train + eval: ``sig(out, tgt, batch_ids=ids, preds=preds, targets=tgt)``
     - train + eval: loss + predictions + targets
     - Smaller datasets, maximum observability on both phases
