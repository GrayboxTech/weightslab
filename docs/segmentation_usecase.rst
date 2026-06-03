Segmentation Use Case — Per-instance & Per-sample Signals (PyTorch)
==================================================================

This page walks through the segmentation integration from:

``weightslab/examples/PyTorch/ws-segmentation/main.py``

It builds on the classification :doc:`usecases` page and focuses on what is
**specific to segmentation**: a *list of per-instance masks* per sample, a custom
collate, and **custom user signals computed both per-sample and per-instance**
(Dice as a metric, BCE as a loss).

Goal
----

Use Weightslab to:

- track a U-Net segmentation model with tracked loaders,
- store **per-instance** targets (one mask per object/class) alongside the
  **per-sample** view,
- compute and log **custom Dice/BCE signals** at both granularities,
- drive the dashboard's *shape × ODD-slice* analysis from per-instance signals.

The multi-index data model
--------------------------

Segmentation samples are expanded into a ``(sample_id, annotation_id)``
multi-index:

- ``annotation_id == 0`` is the **canonical sample row** — it holds per-sample
  predictions/targets/signals plus sample-level metadata, origin and tags.
- ``annotation_id >= 1`` are the **instance rows** — one per object/class mask,
  holding only that instance's target and per-instance signals.

So a sample with N instance masks occupies ``N + 1`` rows. The studio collapses
this back to one row per sample for display: it keeps the ``instance_id 0`` row
as the main value and only falls back to aggregating the instance rows for cells
that are empty on the sample row.

1) A dataset that returns a list of instance masks
--------------------------------------------------

``utils/data.py`` returns ``(image, uid, instances, metadata)`` where
``instances`` is a **list** of per-instance mask tensors (pixel value = class id):

.. code-block:: python

   def get_items(self, idx, ...):
       ...
       mask_t_instances = []
       lbl_max = mask_t.max().item()
       for i in range(1, lbl_max + 1):
           m = torch.zeros_like(mask_t)
           m[mask_t == i] = i          # one mask per class id
           mask_t_instances.append(m)
       return img_t, uid, mask_t_instances, metadata

Why a list (not a single dense mask):

- A *list of array-like targets* is what tells Weightslab there are multiple
  instances → it creates instance rows ``1..N``. A single array/scalar target is
  treated as the sample's own target and stays on ``instance_id 0`` only.

2) A custom collate to batch variable-length instance lists
-----------------------------------------------------------

The default PyTorch collate cannot batch variable-length lists, so the example
provides ``seg_collate`` and passes it to the loader:

.. code-block:: python

   from utils.data import seg_collate

   train_loader = wl.watch_or_edit(
       _train_dataset, flag="data", loader_name="train_loader",
       batch_size=2, collate_fn=seg_collate, preload_labels=False,
   )

``seg_collate`` returns ``(images, ids, labels, metas)`` where ``labels`` is a
``list[B]`` and ``labels[s]`` is that sample's list of instance masks (empty
masks filtered out).

3) Custom Dice (metric) and BCE (loss) signals
----------------------------------------------

``utils/criterions.py`` defines four small ``nn.Module`` criterions. Each scores
every instance mask against the model's per-class probability map, then exposes
the value either aggregated per sample or flat per instance:

.. code-block:: python

   from utils.criterions import (
       PerSampleDice, PerInstanceDice,   # metric
       PerSampleBCE,  PerInstanceBCE,    # loss
   )

   def _make_seg_signals(split):
       return {
           "dice_sample":   wl.watch_or_edit(PerSampleDice(),   flag="metric",
                                name=f"{split}_dice/sample",   per_sample=True,  log=True),
           "dice_instance": wl.watch_or_edit(PerInstanceDice(), flag="metric",
                                name=f"{split}_dice/instance", per_instance=True, log=True),
           "bce_sample":    wl.watch_or_edit(PerSampleBCE(),    flag="loss",
                                name=f"{split}_bce/sample",    per_sample=True,  log=True),
           "bce_instance":  wl.watch_or_edit(PerInstanceBCE(),  flag="loss",
                                name=f"{split}_bce/instance",  per_instance=True, log=True),
       }

   train_sig = _make_seg_signals("train")
   test_sig  = _make_seg_signals("test")

Why two flavors:

- ``per_sample=True`` → the returned ``[B]`` vector is logged and written to the
  **sample row (instance_id 0)** via the per-sample path.
- ``per_instance=True`` → the returned **flat ``[total_instances]``** tensor is
  auto-saved at ``(sample_id, annotation_id)`` for ``annotation_id >= 1`` via
  :func:`wl.save_instance_signals`. Ordering is *sample-major* and must match the
  ``batch_idx`` you pass (see next step).

4) The training step: build ``batch_idx`` and route signals
-----------------------------------------------------------

The per-instance wrapper needs a ``batch_idx`` that maps each instance (in flat,
sample-major order) to its sample position; build it from the same instance
lists so ordering lines up:

.. code-block:: python

   def _instance_batch_idx(labels):
       return torch.tensor(
           [s for s, insts in enumerate(labels) for _ in insts],
           dtype=torch.long,
       )

   with guard_training_context:
       inputs, ids, labels, _ = next(loader)
       outputs = model(inputs)            # [B, C, H, W]
       batch_idx = _instance_batch_idx(labels)

       # Per-sample (→ IID 0) and per-instance (→ IID >= 1) signals.
       bce_sample  = sig["bce_sample"](outputs, labels, batch_ids=ids, preds=preds)
       dice_sample = sig["dice_sample"](outputs, labels, batch_ids=ids)
       sig["dice_instance"](outputs, labels, batch_ids=ids, batch_idx=batch_idx, targets=flat_masks)
       sig["bce_instance"](outputs, labels, batch_ids=ids, batch_idx=batch_idx)

       # Custom per-sample aggregate, saved on the sample row and used for backward.
       loss_per_sample = 0.5 * dice_sample + 0.5 * bce_sample
       wl.save_signals({"combined_bce_dice_per_sample": loss_per_sample}, ids)
       loss_per_sample.mean().backward()
       optimizer.step()

Important:

- Pass ``batch_ids=ids`` to every watched criterion so Weightslab can bind values
  to real samples (and apply discard masking).
- For per-instance criterions also pass ``batch_idx=...``; pass ``targets=`` (a
  **flat, sample-major** list of per-instance masks) to also persist the instance
  GT masks at ``annotation_id >= 1``.
- ``per_instance`` annotation ids are **1-based** (``instance_id 0`` is reserved
  for the sample row), assigned in the order instances appear per sample.

5) Custom static / dynamic signals (``@wl.signal``)
---------------------------------------------------

``utils/criterions.py`` also registers free-form signals via ``custom_signals()``
— a static signal computed from the image, and a dynamic signal that reacts to a
logged metric:

.. code-block:: python

   @wl.signal(name="blue_pixels")                       # STATIC: from ctx.image
   def compute_blue_pixels(ctx: wl.SignalContext) -> int:
       img = ctx.image
       ...
       return int(blue_mask.sum())

   @wl.signal(name="blue_weighted_loss",                # DYNAMIC: subscribes to a metric
              subscribe_to="train_mlt_loss/CE", compute_every_n_steps=1)
   def compute_blue_weighted_loss(ctx: wl.SignalContext) -> float:
       blue = ctx.dataframe.get_value(ctx.origin, ctx.sample_id, "signals_blue_pixels")
       return ctx.subscribed_value * (float(blue or 0) / (128 * 128))

   custom_signals()   # register before wl.serve()

See :doc:`user_functions` for the ``@wl.signal`` / ``SignalContext`` reference.

Where the arrays come from in the studio
----------------------------------------

When the UI requests a sample for a segmentation run:

- **Raw image** — read directly from the dataset file each time (never stored in
  the dataframe).
- **Prediction mask** — loaded lazily from the array store (``arrays.h5``) via a
  proxy, from whatever the per-sample path saved on ``instance_id 0``.
- **GT label** — taken from the sample row's ``target`` if present, otherwise
  reconstructed from the dataset file; the individual per-instance masks live on
  ``instance_id >= 1``.

Practical checklist
-------------------

- Return a **list of instance masks** per sample and wire ``collate_fn=seg_collate``.
- Wrap per-sample criterions with ``per_sample=True`` and per-instance ones with
  ``per_instance=True``.
- Build ``batch_idx`` from the same instance lists; pass it (and ``batch_ids``) to
  the per-instance criterions, plus a flat ``targets`` list to persist instance masks.
- Keep Dice as a ``metric`` and BCE as a ``loss``; aggregate them per sample for the
  backward pass with ``wl.save_signals``.
