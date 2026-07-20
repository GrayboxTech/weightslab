Segmentation — BDD100k (PyTorch)
=================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">segmentation</span>
     <span class="wl-eg-tag">semantic</span>
     <span class="wl-eg-tag">bdd100k</span>
     <span class="wl-eg-tag">masks</span>
     <span class="wl-eg-tag">dense prediction</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-segmentation/main.py``

**Task:** Per-instance semantic segmentation on BDD100k (6 classes) with a
small UNet.

Segmentation adds per-instance masks on top of the per-sample signals seen in
detection. The example also shows how to save custom per-sample class-level
signals via ``wl.save_signals``.

Integration walkthrough
-----------------------

1. Lazy loading with performance flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identical to the detection example — see :doc:`detection` section 1 for
rationale.

.. code-block:: python

   train_loader = wl.watch_or_edit(
       _train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=2,
       shuffle=True,
       is_training=True,
       collate_fn=seg_collate,
       array_autoload_arrays=False,
       array_return_proxies=True,
       array_use_cache=True,
       preload_labels=False,
   )

2. Mixed per-sample and per-instance signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   dice_sample    = wl.watch_or_edit(DiceLoss(),    flag="loss",
                       name="train_dice/sample",   per_sample=True,  log=True)
   dice_instance  = wl.watch_or_edit(DiceLoss(),    flag="loss",
                       name="train_dice/instance", per_instance=True, log=True)
   bce_sample     = wl.watch_or_edit(BCELoss(),     flag="loss",
                       name="train_bce/sample",    per_sample=True,  log=True)
   bce_instance   = wl.watch_or_edit(BCELoss(),     flag="loss",
                       name="train_bce/instance",  per_instance=True, log=True)

``per_instance=True`` creates a ``(sample_id, annotation_id)`` multi-index
in the ledger. For segmentation, each instance is a separate mask region;
the studio renders the per-instance IoU or Dice as a heat-map overlaid on
the predicted mask.

3. Combined loss with a single signal call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with guard_training_context:
       outputs = model(inputs)
       combined = bce_sample(outputs, targets, batch_ids=ids) \
                + dice_sample(outputs, targets, batch_ids=ids)
       combined.mean().backward()

Each signal call is independent — it stores its value and returns a
``(batch_size,)`` tensor you can add or reduce freely.

4. Custom per-sample class signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def _user_custom_signals(outputs, targets, ids, origin):
       pred_classes   = outputs.argmax(dim=1)          # (B, H, W)
       per_sample_tp  = compute_tp(pred_classes, targets)
       per_sample_fp  = compute_fp(pred_classes, targets)
       per_sample_fn  = compute_fn(pred_classes, targets)

       wl.save_signals(
           {
               "combined_bce_dice_per_sample": combined_loss,
               "pred_classes/per_sample":      pred_classes_encoded,
               "tp_per_sample":                per_sample_tp,
               "fp_per_sample":                per_sample_fp,
               "fn_per_sample":                per_sample_fn,
           },
           batch_ids=ids,
           origin=origin,
       )

``wl.save_signals`` accepts any dict of ``{signal_name: tensor_or_array}``
where each tensor has shape ``(batch_size,)``. The studio can plot any of
these signals over training steps and lets you sort samples by them.

.. tip::

   A self-contained bundled version runs without any dataset configuration:

   .. code-block:: bash

      weightslab start               # 1. deploy the studio
      weightslab start example --seg # 2. start the segmentation demo


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/ws-segmentation.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
