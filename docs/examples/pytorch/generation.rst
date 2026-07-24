Generation / Anomaly Detection — MVTec (PyTorch)
=================================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">anomaly detection</span>
     <span class="wl-eg-tag">generation</span>
     <span class="wl-eg-tag">unsupervised</span>
     <span class="wl-eg-tag">mvtec</span>
     <span class="wl-eg-tag">reconstruction</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-generation/main.py``

**Task:** Unsupervised anomaly detection on MVTec capsule images with a
multi-task UNet (classification head + reconstruction head + contrastive loss).

This example introduces two advanced patterns:

- ``compute_dependencies=False`` when wrapping a complex generative model.
- ``wl.save_group_signals`` for signals computed over **pairs** of samples.

Integration walkthrough
-----------------------

1. Multi-task model wrapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model = wl.watch_or_edit(
       _model,
       flag="model",
       device=device,
       compute_dependencies=False,
   )

``compute_dependencies=False`` skips the static dependency graph computation
for the wrapped model. Use this when the model has dynamic control flow,
multiple outputs, or cannot be traced by ``torch.fx`` — common in
encoder-decoder architectures.

2. Dataset with paired samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dataset yields two images per item (an anchor and a randomly paired image)
so the contrastive loss can be computed in-batch:

.. code-block:: python

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=parameters["batch_size"],
       shuffle=True,
       compute_hash=False,
   )
   # Loader yields: [img1, img2], [uid1, uid2], [label1, label2], metadata

The dataset's ``__getitem__`` returns a pair so each WeightsLab batch
contains two groups of sample IDs.

3. Per-sample signals for individual losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with guard_training_context:
       [img1, img2], [uid1, uid2], [label1, label2], _ = next(train_loader)
       cls_out, recon_out = model([img1, img2])

       cls_loss_per  = wl.save_signals({"train/cls_loss":   cls_loss_per_sample},
                                        batch_ids=uid1, origin="train_loader")
       recon_loss_per = wl.save_signals({"train/recon_loss": recon_loss_per_sample},
                                         batch_ids=uid1, origin="train_loader")

4. Group signals for contrastive (pair-level) losses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.save_group_signals(
       signals={"train/contrastive_loss": contrastive_per_pair},
       group_ids=list(zip(uid1, uid2)),
       origin="train_loader",
   )

``wl.save_group_signals`` stores signals that are defined over **groups** of
samples rather than individual ones. Each ``group_id`` is a tuple of sample
UIDs. The studio can display the contrastive loss for every pair and lets you
filter by pair distance to find the hardest negatives.

.. tip::

   This example is bundled with WeightsLab (uses a synthetic dataset):

   .. code-block:: bash

      weightslab start               # 1. deploy the studio
      weightslab start example --gen # 2. start the generation / anomaly demo


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/ws-generation.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
