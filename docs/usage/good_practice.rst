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

When the studio triggers **inference on the train set** (eval mode on a tagged
subset), it runs the forward pass and collects predictions. Decide upfront how
much data you want to store per step, since storing dense predictions for every
sample can become expensive.

**Light mode** — store everything (predictions + targets) per step:

.. code-block:: python

   loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids, preds=preds)

Use this for small datasets or during debugging. The studio can render the
prediction overlay and target overlay simultaneously.

**Standard mode** — store only the loss per step and per sample, omit predictions during training:

.. code-block:: python

   loss_per_sample = sig["loss"](outputs, targets, batch_ids=ids)

Predictions and targets are not written to disk. The studio still plots the
per-sample loss curve; overlays are only available when the studio explicitly
triggers an eval pass (which passes ``preds`` separately).

**Standard mode with eval-time predictions** — pass processed predictions and
optionally a different target tensor for eval:

.. code-block:: python

   # During eval (triggered by the studio or your own eval loop):
   preds_processed = decode_and_nms(outputs.detach())
   loss_per_sample = sig["loss"](
      outputs, targets,
      batch_ids=ids,
      preds=preds_processed,
      targets=targets
   )   # optional: override target if it differs from the one passed to the loss

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
     - ``sig(out, tgt, batch_ids=ids, preds=preds)``
     - loss + predictions + targets
     - Small dataset, debugging, overlay needed every step
   * - Standard
     - ``sig(out, tgt, batch_ids=ids)``
     - loss only
     - Large dataset, production, overlays on eval only
   * - Standard + eval
     - ``sig(out, tgt, batch_ids=ids, preds=preds, targets=tgt)``
     - loss + predictions + targets (eval step only)
     - Large dataset, overlay on studio-triggered eval passes
