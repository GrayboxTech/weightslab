Live 3-D Projection (Parametric UMAP)
=====================================

A small encoder is trained alongside your model, on the UMAP objective, mapping
the model's penultimate features to 3-D. Each sample gets coordinates, written
back as ordinary per-sample signals, and Weights Studio draws them as a
navigable cloud (see :ref:`studio-projection-board`).

It is **on by default and needs no code**. It never perturbs training: features
are detached at the hook and the encoder carries its own optimizer, so no
gradient from the projection can reach your model's weights.

Quickstart
-----------

Wrapping the model attaches it:

.. code-block:: python

   import weightslab as wl

   model = wl.watch_or_edit(MyNet(), flag="model")
   crit = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                           flag="loss", signal_name="loss_sample",
                           per_sample=True, log=True)

After the first few fits, ``signals//umap_x``, ``_y`` and ``_z`` appear as
per-sample columns — so sorting, filtering, histograms and export all work on
them like any other signal.

Turning it off::

   WEIGHTSLAB_PROJECTION=0 python train.py

``0``, ``false``, ``no``, ``off`` all disable it. Disabled means *removed*: no
hook, no encoder, no cost. Per model: ``wl.watch_or_edit(..., projection=False)``.

Configuration
--------------

======================================  =========  ==================================================
Variable                                Default    Meaning
======================================  =========  ==================================================
``WEIGHTSLAB_PROJECTION``               on         ``0``/``false``/``no``/``off`` removes the feature.
``WEIGHTSLAB_PROJECTION_EVERY``         ``50``     Fit + write every N training steps.
``WEIGHTSLAB_PROJECTION_DIM``           ``3``      Output dimensions (2 draws on the z = 0 plane).
``WEIGHTSLAB_PROJECTION_NEIGHBORS``     ``15``     UMAP ``n_neighbors``.
``WEIGHTSLAB_PROJECTION_GRAPH``         ``512``    Samples the kNN graph is built over (see below).
======================================  =========  ==================================================

The same knobs, plus the rest of the tracker's arguments, via the keyword:

.. code-block:: python

   model = wl.watch_or_edit(
       MyNet(), flag="model",
       projection={"layer": "backbone.avgpool", "every_n_steps": 20,
                   "n_neighbors": 30, "min_dist": 0.05},
   )

Why the graph is buffered
--------------------------

UMAP's graph is a **k-nearest-neighbour** graph, so it needs meaningfully more
samples than ``k``. Fitting on one training batch does not give it that: with a
batch of 16 and ``n_neighbors=15``, every sample is every other sample's
neighbour — the graph is complete, and there is no local structure left to
preserve. Measured on clustered features:

=========  ==========================  ===============================
Batch      Pairs that are edges        Within/between membership ratio
=========  ==========================  ===============================
16         100 %                       3 x
64         13 %                        112 x
512        4 %                         ~10^11 x
=========  ==========================  ===============================

So features are accumulated in a rolling buffer of the most recent
``WEIGHTSLAB_PROJECTION_GRAPH`` samples and the graph is built over that.
Graph quality then no longer depends on whatever batch size your training loop
happens to use. **If your projection shows one blob instead of clusters, this
is the first thing to check** — raise ``WEIGHTSLAB_PROJECTION_GRAPH``, or lower
``WEIGHTSLAB_PROJECTION_NEIGHBORS``.

Which layer gets hooked
------------------------

One rule: **the input of the last parameterised layer** — last ``nn.Linear``,
else the last conv, else the last leaf module's output. That layer is the
model's head, so what flows *into* it is the representation; what comes *out*
is class scores.

Reading the output instead fails silently: a segmentation head emits
``(B, num_classes)``, a well-shaped tensor UMAP will happily lay out, and you
would be looking at class scores believing they were the representation
(``fcn_resnet50`` gives ``(B, 512)`` features this way, ``(B, 21)`` logits the
other).

The activation is reduced to ``(B, F)`` along the right axis: a Linear's input
has features last (so ``(B, T, C)`` pools over tokens and keeps ``C``), a conv's
input is channels-first. Tuple returns and HuggingFace output dicts are
unwrapped automatically.

Override with ``projection={"layer": "<named_modules() name>"}``. Worth doing
when the model has multiple heads ("last Linear" means last *defined*, not last
executed) or when the auto-picked layer is not on the forward path — in which
case ``wl.projection.get_tracker().stats()`` reports ``fits: 0`` and a warning
is logged.

Verified across the bundled model zoo (MLPs, VGG, ResNet, U-Net/3+/3D,
TinyYOLO, FCN-ResNet50), torchvision, and transformers (GPT-2, BERT,
``BertForSequenceClassification``) — see
``tests/general/test_projection_baseline_models.py``.

Persistence
------------

The encoder is saved to ``<root_log_dir>/projection/<prefix>_encoder.pt`` —
its **own file**, deliberately not inside the model checkpoint, since it is not
part of your model and must not ride along into a deployable weights file.

It is written every 20 fits and restored automatically when the model is
wrapped, so a restarted run **continues the same layout** rather than
re-laying it out from a random initialisation. That matters: without it the
coordinates already in the dataframe would describe a different embedding than
the one overwriting them, leaving two layouts in one cloud.

``wl.projection.save_projection()`` / ``load_projection()`` are available
explicitly.

Re-projecting offline
----------------------

When a run ends and the live projection watched the wrong layer, you do not
have to retrain:

.. code-block:: python

   model = wl.watch_or_edit(MyNet(), flag="model")      # checkpoint restores
   loader = wl.watch_or_edit(train_ds, flag="data", loader_name="train_loader")

   wl.project_dataset(model, loader, layer="backbone.layer3",
                      prefix="umap_layer3", epochs=20)

No training step runs — the model is used in ``eval()`` under ``no_grad``, and
its weights and train/eval mode are left untouched. It also fits better than
the live path: every feature is collected first, so each epoch draws
neighbourhoods from the whole dataset rather than one batch.

Different ``prefix`` values coexist as separate columns, so several candidate
layers can be compared; the board's picker switches between them.

Resuming training afterwards: with a **different** prefix the offline result is
a frozen snapshot and the live projection carries on independently. With the
**same** prefix, the live projection *adopts* the offline fit by default
(``adopt=None``) — it re-hooks to that layer and keeps refining it — because
not adopting would let training overwrite those coordinates with a fresh
encoder's output, mixing two layouts. Force it with ``adopt=True``/``False``.

Progress is reported with ``tqdm`` (``features`` / ``fit`` / ``encode``);
``verbose=False`` silences it. Returns a dict with ``samples``,
``feature_dim``, ``columns``, ``final_loss`` and ``adopted_by_live`` — check
``feature_dim`` to confirm you hooked what you meant to.

Task coverage
--------------

The projection needs only a forward pass through the hooked layer and a
per-sample write carrying ``batch_ids`` — either a ``flag="loss"`` criterion or
a direct :func:`wl.save_signals`. Every supervised usecase provides one.

Detection/segmentation with ``per_instance=True`` is handled: the projection
receives the **sample**-level ids, not the per-annotation ones, so there is one
point per image. Sequence models pool over tokens, giving one point per
sequence.

Train and eval splits both appear. Eval batches are **placed but never fit on**
— held-out samples belong in the picture, but letting them shape the layout
would leak the eval set into it.

Limits
-------

- The layout is **not comparable across runs**, nor across an architecture
  edit: the encoder is rebuilt when the hooked layer's width changes
  (``stats()["rebuilds"]`` counts it) and the cloud re-lays-out. Compare
  structure, not coordinates.
- **Token pooling ignores the attention mask** — a forward hook cannot see it,
  so a padded language batch averages padding positions in. Harmless at uniform
  lengths; with varying lengths, hook your own pooling module instead.
- Coordinates exist only for samples seen since the projection attached.
- Batches under 4 samples are skipped, and ``n_neighbors`` is clamped to the
  graph size minus one.

See also
---------

- :ref:`studio-projection-board` — the Projection Board in Weights Studio.
- :doc:`signal_trajectory_classification` — the other "look at the shape, not
  the value" view, over time rather than over the representation.
- :doc:`configuration` — every environment variable in one table.
