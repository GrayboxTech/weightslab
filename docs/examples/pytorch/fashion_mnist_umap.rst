Classification, Fashion-MNIST + UMAP (PyTorch)
===============================================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--pytorch">PyTorch</span>
     <span class="wl-eg-tag">classification</span>
     <span class="wl-eg-tag">fashion-mnist</span>
     <span class="wl-eg-tag">projection</span>
     <span class="wl-eg-tag">umap</span>
     <span class="wl-eg-tag">pca</span>
     <span class="wl-eg-tag">beta</span>
   </div>

**Example:** ``weightslab/examples/PyTorch/wl-fashion-mnist-umap/main.py``

**Task:** 10-class Fashion-MNIST classification with a small CNN, explored with
projections. The built-in parametric UMAP runs beside training; then you go back
to the weights of the first 1000 steps, discard the training samples that teach
the model least, lay what is left out with **a PCA you write yourself**, and
compare the two layouts in the :ref:`studio-projection-board` and in numbers.

It is the plain-PyTorch counterpart of the
:doc:`projection use case <../usecases/projection_fashion_mnist>`, driven in
stages so you can look at the board between them. Background and every option of
the projection: :doc:`../../projection`.

The stages
----------

1. Train with the live UMAP, keep a checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing in the loop is projection code. Wrapping the model attaches the UMAP
(it is on by default); the dict only tunes it:

.. code-block:: python

   model = wl.watch_or_edit(
       FashionCNN().to(device), flag="model", device=device,
       projection={"every_n_steps": 25, "graph_size": 1024},   # optional
   )

   train_to(1000)                                  # the UMAP fits alongside
   checkpoint_step = save_checkpoint()             # weights + encoder + per-sample state

The projection hooks the input of the last ``Linear`` (``fc2``): the 128-D
representation the classifier decides from, not the logits. The checkpoint holds
the weights, the UMAP encoder and every sample's loss at that step.

2. Reload that checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   manager = ledgers.get_checkpoint_manager()
   manager.load_state(manager.get_current_experiment_hash(), target_step=checkpoint_step)

Weights, optimizer, the UMAP encoder and each sample's loss and seen-count go
back to that step. Straight after stage 1 it changes nothing; it is the way back
to *these* weights if you train on and come back to try another cut.

3. Discard what teaches little
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   table = sample_table()                                  # one row per training sample
   easy = lowest_loss_ids(table, 0.3)                      # the 30% with the lowest loss
   wl.discard_samples(easy, discarded=True)                # the loaders skip them from now on

Samples with no loss yet are never picked: an unseen sample is not an easy one.
Expect the cut to come mostly from the clean classes (Trouser, Bag, the
footwear) and hardly at all from the tops the model still confuses.

4. A PCA of your own
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def pca_coordinates(features, dim=2):                   # plain torch, an SVD
       x = torch.as_tensor(features, dtype=torch.float32)
       x = x - x.mean(0)
       _, singular, axes = torch.linalg.svd(x, full_matrices=False)
       return (x @ axes[:dim].T).numpy()

   wl.project_dataset(model, train_loader, method=pca_coordinates,
                      prefix="pca2d_step1000")

``wl.project_dataset`` runs the loader through the model, collects the 128-D
features, hands them to **your function**, and stores what comes back as
``signals//pca2d_step1000_{x,y}``. The loader skips the samples you just
discarded, so the PCA covers what is left. The sweep always starts from the
loader's first batch (even though training was mid-epoch on it) and does not age
the model. Any algorithm returning ``(N, 2)`` or ``(N, 3)`` plugs in the same
way. The name carries the step, so snapshots stay apart in the picker.

5. Compare
~~~~~~~~~~

In the Studio, *Data Exploration → Projection*, open the picker and switch
between ``umap`` (3-D, learned live) and ``pca2d_step1000`` (2-D, drawn on the
z = 0 plane); lasso the same region in both and compare what the grid loads.

In the script, two more views of the same two layouts. A side-by-side picture is
written to ``<root_log_dir>/umap_vs_pca.png``. And ``knn_purity`` scores each
layout: for every point, the share of its *k* nearest neighbours **in the
picture** that have its class (1.0 = every neighbourhood is one class, about 0.1
= ten classes thrown together). The 128-D features are scored too, as the
reference both pictures summarize:

.. code-block:: text

   neighbours sharing a point's class (k=10), over 14000 kept samples:
     parametric UMAP (3-D) : 0.720
     your PCA (2-D)        : 0.545
     the 128-D features    : 0.858   (what both pictures summarize)

*(One CPU run of the defaults; yours will differ in the last digits. UMAP keeps
three dimensions and PCA two, so read the gap as a rough size, not a verdict.)*

6. Retrain on what is left
~~~~~~~~~~~~~~~~~~~~~~~~~~

The live UMAP **follows the model**: its encoder keeps fitting, so its layout
changes with the weights. The PCA does not: it is the snapshot you took, so to
see the PCA of the new model you project again under a new name
(``pca2d_step2000``). In the same run, 1000 more steps on the remaining samples
took the UMAP's score from 0.72 to 0.84 while a fresh PCA moved from 0.55 to
0.56: the representation got more class-separable, and only the layout that
learns shows it that clearly.

Every projection you compute is stored with the checkpoint of that moment (see
:doc:`../../projection`), so restoring an older checkpoint brings back the
projections that existed then.

Configuration
-------------

``config.yaml`` carries the tuning of the UMAP (``projection:``) and of the
stages (``exploration:``):

.. code-block:: yaml

   projection:
     every_n_steps: 25      # fit the UMAP encoder every 25 training steps
     graph_size: 1024       # samples in the kNN graph
   exploration:
     steps_initial: 1000    # train with the live UMAP, then keep a checkpoint
     discard_fraction: 0.3  # share of training samples to drop, lowest loss first
     steps_retrain: 1000    # train on what is left (0 = stop after the comparison)
     knn_k: 10              # neighbours counted by the class-purity score

Run it
------

.. code-block:: bash

   weightslab start                          # terminal 1: the studio, http://localhost:8080

   cd weightslab/examples/PyTorch/wl-fashion-mnist-umap
   python main.py --wait --keep-serving      # terminal 2
   python main.py --steps 200 100            # a quick pass: 200 initial + 100 retrain steps

``--wait`` stops for Enter after the comparison, so you can look at the board
before retraining; ``--keep-serving`` keeps the process (and the Studio
connection) up at the end. The defaults take a couple of minutes on a CPU. The
projection board is a **beta** feature.
