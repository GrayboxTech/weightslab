Live 3-D Projection (Parametric UMAP)
=====================================

.. note::

   **Beta.** The projection and its board work end to end, but their options
   and on-disk format may still change between releases.

A small encoder is trained alongside your model, on the UMAP objective, mapping
the model's penultimate features to 3-D. Each sample gets coordinates, written
back as ordinary per-sample signals, and Weights Studio draws them as a
navigable cloud (see :ref:`studio-projection-board`).

It is **on by default and needs no code**. It never perturbs training: features
are detached at the hook and the encoder carries its own optimizer, so no
gradient from the projection can reach your model's weights.

You can also bring **your own** projection , t-SNE, ``umap-learn``, PCA,
anything , and explore it on the same board (see
:ref:`projection-bring-your-own`).

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
per-sample columns , so sorting, filtering, histograms and export all work on
them like any other signal.

Turning it off
---------------

For the whole process::

   WEIGHTSLAB_PROJECTION=0 python train.py

``0``, ``false``, ``no``, ``off`` (any case) all disable it. Disabled means
*removed*: no hook, no encoder, no cost.

For one model, from the model wrapper:

.. code-block:: python

   model = wl.watch_or_edit(MyNet(), flag="model", projection=False)

``projection=False`` also removes the hook of a model wrapped earlier in the
same process , there is one projection per process, and it follows the model
you wrapped last. The environment variable wins over everything: with
``WEIGHTSLAB_PROJECTION=0`` even ``projection=True`` or a ``projection={...}``
dict installs nothing.

Configuration
--------------

Environment variables (see also :doc:`configuration`):

======================================  =========  ==================================================
Variable                                Default    Meaning
======================================  =========  ==================================================
``WEIGHTSLAB_PROJECTION``               on         ``0``/``false``/``no``/``off`` removes the feature.
``WEIGHTSLAB_PROJECTION_EVERY``         ``50``     Fit + write every N training steps.
``WEIGHTSLAB_PROJECTION_DIM``           ``3``      Output dimensions (2 draws on the z = 0 plane).
``WEIGHTSLAB_PROJECTION_NEIGHBORS``     ``15``     UMAP ``n_neighbors``.
``WEIGHTSLAB_PROJECTION_GRAPH``         ``512``    Samples the kNN graph is built over (see below).
======================================  =========  ==================================================

Per model, through the ``projection`` keyword of
``wl.watch_or_edit(..., flag="model")``. A key set here takes precedence over
its environment variable:

=========================  ===========  =====================================================
Key                        Default      Meaning
=========================  ===========  =====================================================
``layer``                  auto         ``named_modules()`` name of the layer to hook (below).
``every_n_steps``          ``50``       As ``WEIGHTSLAB_PROJECTION_EVERY``.
``out_dim``                ``3``        As ``WEIGHTSLAB_PROJECTION_DIM``.
``n_neighbors``            ``15``       As ``WEIGHTSLAB_PROJECTION_NEIGHBORS``.
``graph_size``             ``512``      As ``WEIGHTSLAB_PROJECTION_GRAPH``.
``inner_steps``            ``4``        Encoder optimizer steps per fit.
``lr``                     ``1e-3``     Encoder learning rate (Adam).
``min_dist``               ``0.1``      UMAP ``min_dist``: how tightly points may pack.
``spread``                 ``1.0``      UMAP ``spread``: the scale of the embedding.
``repulsion_strength``     ``1.0``      Weight of the repulsive term.
``signal_prefix``          ``"umap"``   Column prefix: writes ``signals//<prefix>_{x,y,z}``.
=========================  ===========  =====================================================

.. code-block:: python

   model = wl.watch_or_edit(
       MyNet(), flag="model",
       projection={"layer": "backbone.avgpool", "every_n_steps": 20,
                   "n_neighbors": 30, "min_dist": 0.05},
   )

   # A string is shorthand for {"layer": ...}:
   model = wl.watch_or_edit(MyNet(), flag="model", projection="backbone.avgpool")

Why the graph is buffered
--------------------------

UMAP's graph is a **k-nearest-neighbour** graph, so it needs meaningfully more
samples than ``k``. Fitting on one training batch does not give it that: with a
batch of 16 and ``n_neighbors=15``, every sample is every other sample's
neighbour , the graph is complete, and there is no local structure left to
preserve. Measured on clustered features:

=========  ==========================  ===============================
Batch      Pairs that are edges        Within/between membership ratio
=========  ==========================  ===============================
16         100 %                       3 x
64         13 %                        112 x
512        4 %                         ~10^11 x
=========  ==========================  ===============================

So the features of **every** training batch go into a rolling buffer of the
most recent ``WEIGHTSLAB_PROJECTION_GRAPH`` samples, and each fit builds its
graph over that. Graph quality then no longer depends on whatever batch size
your training loop happens to use. **If your projection shows one blob instead
of clusters, this is the first thing to check** , raise
``WEIGHTSLAB_PROJECTION_GRAPH``, or lower ``WEIGHTSLAB_PROJECTION_NEIGHBORS``.

Between fits, each training batch is also **placed** with the encoder as it
stands , one small forward pass, no gradient , so every sample of an epoch gets
coordinates from a recent encoder, not only the batches that happen to land on
a fit step. Nothing is placed before the first fit, so the board never shows a
cloud drawn by an untrained encoder.

Both matter more than they sound. When only fit-step batches were buffered and
placed, the bundled ``wl-parametric-umap`` demo left a quarter of its samples
with no coordinates at all and separated its clusters 1–3x; with every batch
buffered and placed, all samples are drawn and the clusters separate ~25x.

Which layer gets hooked
------------------------

One rule: **the input of the last parameterised layer** , last ``nn.Linear``,
else the last conv, else the last leaf module's output. That layer is the
model's head, so what flows *into* it is the representation; what comes *out*
is class scores. Auxiliary heads (``aux_classifier``, ``aux_head``, ...) are
skipped.

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
executed) or when the auto-picked layer is not on the forward path , in which
case ``wl.projection.get_tracker().stats()`` reports ``fits: 0`` and a warning
is logged. An unknown name is not fatal: it is logged and the auto-pick is used.

Verified across the bundled model zoo (MLPs, VGG, ResNet, U-Net/3+/3D,
TinyYOLO, FCN-ResNet50), torchvision, and transformers (GPT-2, BERT,
``BertForSequenceClassification``) , see
``tests/general/test_projection_baseline_models.py``.

.. _projection-bring-your-own:

Bring your own projection
--------------------------

The board draws **any** per-sample ``signals//<prefix>_x`` / ``_y`` (and
optionally ``_z``) columns, whoever wrote them, and lists every prefix it finds
in its picker. So your own algorithm plugs in by writing coordinates under a
prefix of your choosing , beside the built-in ``umap``, or instead of it.

**You have the coordinates** , ``wl.save_projection_coords``:

.. code-block:: python

   from sklearn.manifold import TSNE

   # features: (N, F) array you collected; ids: the N sample ids (the ids the
   # WeightsLab loaders yield, in the same order)
   coords = TSNE(n_components=3).fit_transform(features)      # (N, 3)
   wl.save_projection_coords(coords, batch_ids=ids, prefix="tsne")

**WeightsLab collects the features, your algorithm lays them out**,
``wl.project_dataset(..., method=...)``. ``method`` is anything with a
scikit-learn style ``fit_transform`` (``TSNE``, ``PCA``, ``umap.UMAP``) or a
plain callable ``(N, F) ndarray -> (N, 2 | 3)``. The features come from the same
layer the live projection uses, or from ``layer=`` if you pass one:

.. code-block:: python

   import umap                                   # pip install umap-learn

   wl.project_dataset(model, eval_loader, method=umap.UMAP(n_components=3),
                      prefix="umap_learn")
   wl.project_dataset(model, eval_loader, method=TSNE(n_components=2),
                      layer="backbone.layer3")   # prefix defaults to "tsne"

Either call works whenever you like , after training, once per epoch, or at a
given step from inside your loop:

.. code-block:: python

   for x, ids, y in train_loader:
       with wl.guard_training_context:
           ...                                   # your training step
       if model.get_age() == 500:
           wl.project_dataset(model, eval_loader, method=TSNE(n_components=3),
                              prefix="tsne_step500", max_samples=20_000)

It runs synchronously, so training waits for it: t-SNE on tens of thousands of
samples takes minutes, which is what ``max_samples`` is for. The model is used
in ``eval()`` under ``no_grad`` and its train/eval mode is restored afterwards.

The sweep is a **whole pass, whatever training is doing**:

- A tracked loader your training loop is mid-epoch on is restarted first (the
  loop begins a fresh epoch afterwards), so ``train_loader`` is as good a choice
  as an evaluation loader. Discarded samples are skipped by the loader itself,
  as in training.
- It **does not age the model**: ``model.get_age()`` is the same before and
  after, and the model's tracking mode is put back.
- The returned dict names the samples placed (``sample_ids``), in the order of
  the feature rows your ``method`` received, so you can pair them with labels.

The rules:

- **Prefix** , letters, digits, ``_``, ``.``, ``-``, starting with a letter or
  digit. Writing the same prefix again replaces the coordinates of the samples
  you pass; samples you leave out keep theirs. Use a new prefix (such as
  ``tsne_step500``) to keep a snapshot.
- **Not** ``umap`` **while the built-in runs** , it rewrites its prefix on every
  fit and would overwrite yours sample by sample. Either name, or turn the
  built-in off with ``projection=False`` and use the name freely.
- **2-D or 3-D** , ``(N, 2)`` is drawn on the z = 0 plane. Anything else is
  refused with an error naming the shape. Rows of NaN mean "not placed" and are
  skipped by the board.
- **Not counted as seen** , writing coordinates leaves the samples'
  ``nb_seen`` / ``last_seen`` alone; placing a sample in a picture is not the
  model training on it.
- **Mistakes fail fast** , a bad prefix, the live prefix, or an unknown
  ``method`` string is refused *before* ``project_dataset`` sweeps the dataset.
- No encoder is saved or adopted for your own method; ``epochs``,
  ``fit_batch``, ``lr``, ``min_dist``, ``spread``, ``n_neighbors`` and ``adopt``
  apply to the built-in only.

When the built-in is off and only your projection exists, the board opens on it
directly; with both, ``umap`` comes first in the picker. The status line's fit
count belongs to the built-in, so it is shown only for its prefix. Until a
projection exists the board shows an orange *No projection found* ribbon with
the reason.

**From the Studio notebook.** The notebook runs in the training process, so a
cell can read the live model and call ``wl.save_projection_coords`` directly.
Running a cell pauses training first (see :ref:`notebook-pauses-training`), so
the features you project come from one state of the model. To pause at a
moment of your choosing from the training script itself , say, once the run
has converged , call ``wl.pause_training()``; training waits at its next step
until Play.

Three runnable examples:

- ``examples/Usecases/wl-fashion-mnist-umap`` , Fashion-MNIST with the built-in
  live UMAP running alongside training.
- ``examples/PyTorch/wl-fashion-mnist-umap`` , trains 1000 steps with the live
  UMAP, reloads that checkpoint, discards the lowest-loss training samples,
  projects what is left with a PCA you write yourself, and compares the two. See
  :doc:`examples/pytorch/fashion_mnist_umap`.
- ``examples/Usecases/wl-fashion-mnist-custom-projection`` , trains to
  convergence, pauses itself with ``wl.pause_training()``, and drops a notebook
  into the run directory that computes a t-SNE written from scratch and plugs it
  into the board.

Checkpoints and restarts
-------------------------

The encoder follows the model weights, so the cloud always describes the model
you actually have:

- **Every model checkpoint** saves the encoder beside it, in a ``projection/``
  subdirectory next to the weight files
  (``.../models/<hash>/projection/<checkpoint>.pt``). Never *inside* the
  checkpoint: the encoder is not part of your model and must not ride along into
  a deployable weights file. Never *beside* it either, where it would match the
  weight-checkpoint glob.
- A **run-level copy** is kept at ``<root_log_dir>/projection/<prefix>_encoder.pt``
  and refreshed every 20 fits and at every checkpoint, so a crash or a Ctrl-C
  does not cost the layout.

On a restore:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - What happens
     - Encoder you get
   * - Restart; the wrapper auto-loads the latest checkpoint
     - The encoder saved **with that checkpoint** , not the run-level copy,
       which may be from later steps.
   * - Restore a checkpoint from Studio or the agent
     - The encoder saved with it; buffered features are dropped.
   * - The checkpoint predates the first fit
     - None: the layout starts again from that point.
   * - The checkpoint restores a different architecture
     - Its encoder, re-hooked onto the same layer (by name) of the restored
       model.
   * - No checkpoint restored (fresh weights)
     - The run-level copy, so the layout continues.

The coordinates follow the same rule as every other signal. A restore from
Studio or the agent rewinds per-sample state to the checkpoint's step: samples
seen after it lose their coordinates , your own projections' too , and are
placed again as training revisits them, so the cloud thins and refills. A plain
restart keeps the stored coordinates until the next fits overwrite them.

``wl.projection.save_projection()`` / ``load_projection()`` save and load the
run-level copy explicitly.

When the projection cannot run
-------------------------------

The projection is instrumentation, so the contract is simple: **it never raises
into your training loop and never stops a run.** What you get instead:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Situation
     - Behaviour
   * - It cannot be installed (no hookable layer, an unknown option)
     - A warning; the model is wrapped normally, with no projection.
   * - The hooked layer is not on the forward path
     - A warning after 3 visits naming the fix (``projection={"layer": ...}``).
   * - A fit raises (out of memory, an unexpected shape)
     - A warning the first time; it retries at the next fit.
   * - 5 fits in a row fail
     - Turned off for the rest of the run, with a warning saying why
       (``stats()["disabled_reason"]``). Coordinates already written stay on
       the board.
   * - The model's features are NaN/inf (it diverged)
     - Those batches are skipped, warned once, so the encoder keeps its last
       good layout instead of turning NaN for good.
   * - The UMAP loss is NaN/inf
     - The step is not taken, so the encoder's weights stay finite; it counts
       as a failed fit.
   * - Coordinates cannot be written
     - A warning once; training is unaffected.
   * - The hook itself errors
     - Ignored; the forward pass is never affected.

The explicit calls are different: ``wl.save_projection_coords`` and
``wl.project_dataset`` are things you invoke yourself, so bad input raises there
(``ValueError`` / ``TypeError`` with what was expected).

Re-projecting offline
----------------------

When a run ends and the live projection watched the wrong layer, you do not
have to retrain:

.. code-block:: python

   model = wl.watch_or_edit(MyNet(), flag="model")      # checkpoint restores
   loader = wl.watch_or_edit(train_ds, flag="data", loader_name="train_loader")

   wl.project_dataset(model, loader, layer="backbone.layer3",
                      prefix="umap_layer3", epochs=20)

No training step runs , the model is used in ``eval()`` under ``no_grad``, and
its weights and train/eval mode are left untouched. It also fits better than
the live path: every feature is collected first, so each epoch draws
neighbourhoods from the whole dataset rather than one batch.

Different ``prefix`` values coexist as separate columns, so several candidate
layers can be compared; the board's picker switches between them. Pass
``method=`` to use your own algorithm instead (see
:ref:`projection-bring-your-own`).

Resuming training afterwards: with a **different** prefix the offline result is
a frozen snapshot and the live projection carries on independently. With the
**same** prefix, the live projection *adopts* the offline fit by default
(``adopt=None``) , it re-hooks to that layer and keeps refining it , because
not adopting would let training overwrite those coordinates with a fresh
encoder's output, mixing two layouts. Force it with ``adopt=True``/``False``.

Progress is reported with ``tqdm`` (``features`` / ``fit`` / ``encode``);
``verbose=False`` silences it. Returns a dict with ``samples``,
``feature_dim``, ``prefix``, ``columns``, ``method``, ``final_loss`` and
``adopted_by_live`` , check ``feature_dim`` to confirm you hooked what you
meant to.

Task coverage
--------------

The projection needs only a forward pass through the hooked layer and a
per-sample write carrying ``batch_ids`` , either a ``flag="loss"`` criterion or
a direct :func:`wl.save_signals`. Every supervised usecase provides one.

Detection/segmentation with ``per_instance=True`` is handled: the projection
receives the **sample**-level ids, not the per-annotation ones, so there is one
point per image. Sequence models pool over tokens, giving one point per
sequence.

Train and eval splits both appear. Eval batches are **placed but never fit on**
, held-out samples belong in the picture, but letting them shape the layout
would leak the eval set into it.

Debugging
----------

``wl.projection.get_tracker()`` is ``None`` when no projection is attached;
otherwise ``.stats()`` reports ``enabled``, ``layer``, ``feature_dim``,
``fits``, ``rebuilds``, ``buffered``, ``samples_written``, ``last_loss``,
``failures``, ``skipped_nonfinite``, ``disabled_reason`` and the ``columns`` it
writes. ``WEIGHTSLAB_LOG_LEVEL=DEBUG`` shows every retried fit.

Limits
-------

- The layout is **not comparable across runs**, nor across an architecture
  edit: the encoder is rebuilt when the hooked layer's width changes
  (``stats()["rebuilds"]`` counts it) and the cloud re-lays-out. Compare
  structure, not coordinates.
- **Token pooling ignores the attention mask** , a forward hook cannot see it,
  so a padded language batch averages padding positions in. Harmless at uniform
  lengths; with varying lengths, hook your own pooling module instead.
- Coordinates exist only for samples seen since the projection attached.
- A fit needs at least 4 buffered samples, and ``n_neighbors`` is clamped to the
  graph size minus one.
- The picker lists only prefixes a WeightsLab writer registered (the live
  projection, ``wl.project_dataset``, ``wl.save_projection_coords``), recorded
  in ``<root_log_dir>/projection/prefixes.json``. Columns you write yourself
  with ``wl.save_signals`` as ``tsne_x`` / ``tsne_y`` are *not* offered; use
  ``wl.save_projection_coords``. A run with no such file (written before it
  existed) still lists every ``_x``/``_y`` column pair.

See also
---------

- :ref:`studio-projection-board` , the Projection Board in Weights Studio.
- :doc:`signal_trajectory_classification` , the other "look at the shape, not
  the value" view, over time rather than over the representation.
- :doc:`configuration` , every environment variable in one table.
