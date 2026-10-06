Projections on Fashion-MNIST
============================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--usecase">Usecase</span>
     <span class="wl-eg-tag">projection</span>
     <span class="wl-eg-tag">umap</span>
     <span class="wl-eg-tag">t-sne</span>
     <span class="wl-eg-tag">notebook</span>
     <span class="wl-eg-tag">beta</span>
   </div>

Runs on the same small CNN, showing the ways a projection reaches the
:ref:`studio-projection-board` (beta): the **built-in** live UMAP, and **your
own** algorithm, plugged in from a script or from the Studio notebook.
Background and every option: :doc:`../../projection`.

Built-in live UMAP, while it trains
-----------------------------------

**Example:** ``weightslab/examples/Usecases/wl-fashion-mnist-umap``

A plain training loop with nothing projection-specific in it. Wrapping the model
attaches the projection, because it is on by default; the dict only tunes it:

.. code-block:: python

   model = wl.watch_or_edit(
       FashionCNN().to(device), flag="model", device=device,
       projection={"every_n_steps": 25, "graph_size": 1024},   # optional
   )

Run ``python main.py``, press **Play** in the Studio, then open *Data
Exploration → Projection*. Until the first fit the board shows an orange *No
projection found* ribbon saying it is waiting; then one point per image appears,
coloured by class. The cloud starts as one tangle and pulls apart into clusters
as the representation forms — and the classes that stay mixed (Shirt, T-shirt,
Pullover, Coat) are the ones the model confuses. Lasso a mixed region to load
those images into the grid.

The projection reads the input of the last ``Linear`` (``fc2``): the 128-D
representation the classifier decides from, not the logits. Test images are
placed by the encoder but never fitted on.

Compare it with a PCA of your own
---------------------------------

**Example:** ``weightslab/examples/PyTorch/wl-fashion-mnist-umap``

Same CNN, driven in stages: train 1000 steps with the live UMAP, **reload** that
checkpoint, **discard** the 30 % of training samples with the lowest loss (the
ones the model already gets right), lay what is left out with a **2-D PCA you
write yourself**, then compare it with the UMAP in the board's picker and in
numbers (how often a point's nearest neighbours in the picture share its class).
A last stage retrains on what is left: the live UMAP follows the model, the PCA
stays the snapshot you took. The full walkthrough is
:doc:`../pytorch/fashion_mnist_umap`.

Converge, pause, your own t-SNE
-------------------------------

**Example:** ``weightslab/examples/Usecases/wl-fashion-mnist-custom-projection``

1. The built-in projection is **off** (``projection=False``): the projection in
   this run is yours.
2. The script evaluates every 250 steps and, once test accuracy stops improving
   (3 evaluations without +0.25 points), pauses itself:

   .. code-block:: python

      if plateau.update(test_accuracy):
          wl.pause_training()      # the next step waits for Play

3. It has written ``own-projection.ipynb`` into the run directory, so the
   Studio's notebook opens on it. Its cells collect the converged model's
   features, run an exact **t-SNE written from scratch in plain torch**, and
   hand the result over:

   .. code-block:: python

      wl.save_projection_coords(tsne_coords, batch_ids=ids, prefix="tsne")

4. *Data Exploration → Projection* now offers ``tsne`` — and ``pca``, from the
   notebook's last cell — in the picker.

Running a notebook cell pauses training first and leaves it paused (see
:ref:`notebook-pauses-training`), so even a cell run before convergence sees one
state of the model. Press **Play** to resume training when you are done.

Any algorithm plugs in the same way: anything that returns ``(N, 2)`` or
``(N, 3)`` coordinates for ``N`` sample ids, from ``sklearn``, ``openTSNE``,
``umap-learn`` or your own code. ``wl.project_dataset(model, loader,
method=TSNE(n_components=3))`` collects the features for you instead.
