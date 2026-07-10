Clustering — Face Recognition (PyTorch)
=========================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">clustering</span>
     <span class="wl-eg-tag">unsupervised</span>
     <span class="wl-eg-tag">embeddings</span>
     <span class="wl-eg-tag">face recognition</span>
     <span class="wl-eg-tag">metric learning</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-clustering/main.py``

**Task:** Metric learning with triplet loss on the Olivetti / LFW face dataset.
The goal is to train an embedding network so that embeddings from the same
person cluster together.

This example shows WeightsLab used in a **contrastive / metric-learning**
setting where there is no standard per-sample label — the signal of interest
is the embedding distance.

Integration walkthrough
-----------------------

1. Register hyperparameters and data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   wl.watch_or_edit(parameters, flag="hyperparameters",
                    defaults=parameters, poll_interval=1.0)

   train_loader = wl.watch_or_edit(
       train_dataset,
       flag="data",
       loader_name="train_loader",
       batch_size=parameters["batch_size"],
       shuffle=True,
       compute_hash=False,
       num_workers=parameters.get("num_workers", 0),
   )

The dataset yields image triplets ``(anchor, positive, negative)`` plus their
stable UIDs. WeightsLab records which triplets the model has seen and lets you
inspect the hardest negatives in the studio.

2. Guard contexts — same as classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train(loader, model, optimizer, device):
       with guard_training_context:
           images, uids, labels, _ = next(loader)
           embeddings = model(images.to(device))
           triplet_loss = compute_triplet_loss(embeddings, labels)
           triplet_loss.mean().backward()
           optimizer.step()

   def evaluate(loader, model, device):
       with guard_testing_context, torch.no_grad():
           for images, uids, labels, _ in loader:
               embeddings = model(images.to(device))
               ...

3. Saving embedding metrics per sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   metrics = FaceMetrics.compute_all_metrics(embeddings, labels, uids)
   wl.save_signals(
       {
           "train/intra_class_dist":  metrics["intra_class_dist"],
           "train/inter_class_dist":  metrics["inter_class_dist"],
           "train/silhouette_score":  metrics["silhouette_score"],
       },
       batch_ids=uids,
       origin="train_loader",
   )

Because there is no single "loss per sample" in metric learning, you compute
whichever distances are meaningful and push them through ``wl.save_signals``
directly. The studio plots these curves and lets you sort images by
silhouette score to identify consistently confused identities.

.. tip::

   This example is bundled with WeightsLab:

   .. code-block:: bash

      weightslab ui launch            # 1. deploy the studio
      weightslab start example --clus # 2. start the clustering demo


.. raw:: html

   <div style="text-align:right; margin-top:2rem;">
     <a href="https://colab.research.google.com/github/GrayboxTech/weightslab/blob/main/weightslab/examples/Notebooks/PyTorch/ws-clustering.ipynb" target="_blank" rel="noopener noreferrer">
       <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
     </a>
   </div>
