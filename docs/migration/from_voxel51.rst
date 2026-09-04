.. _migration-voxel51:

From Voxel51 (FiftyOne)
========================

FiftyOne and WeightsLab overlap more than the other tools here — both put your
dataset in front of you and let you tag, filter and curate it. The difference
is *when*. FiftyOne is a workbench you visit between training runs;
WeightsLab is attached to the run that is happening now.

Migration notes
---------------

**There is no import step.** FiftyOne builds its own ``fo.Dataset`` — you
convert your data into it, and keep the two in sync afterwards. WeightsLab
wraps the ``torch.utils.data.Dataset`` you already have, in place:

.. code-block:: python

   # FiftyOne — build a parallel dataset
   dataset = fo.Dataset.from_dir(dataset_dir=..., dataset_type=fo.types.COCODetectionDataset)

   # WeightsLab — wrap the one you already train on
   train_loader = wl.watch_or_edit(train_dataset, flag="data",
                                   loader_name="train_loader", is_training=True)

Whatever your ``__getitem__`` returns is what the studio shows. Nothing is
copied, and there is no second source of truth to reconcile.

**Implement** ``get_items`` **on your dataset.** This is the one piece of real
work in the port. FiftyOne can read labels straight out of its own sample
documents; WeightsLab needs a way to read a sample's label or metadata
*without* running your full ``__getitem__`` pipeline — otherwise anything that
scans annotations pays for an image decode and augmentation per sample:

.. code-block:: python

   class MyDataset(Dataset):
       def get_items(self, idx, include_metadata=False,
                     include_labels=False, include_images=False):
           img_t = self.image_transform(Image.open(self.images[idx])) if include_images else None
           target = self._load_boxes(self.masks[idx]) if include_labels else None
           metadata = {"img_path": self.images[idx]} if include_metadata else None
           return img_t, self.uids[idx], target, metadata

See :ref:`good-practice-get-items` for the full pattern, including the
augmentation trap when images and labels are fetched in separate calls.

**Views become filters, and they act.** A FiftyOne view is a query producing a
read-only slice. The studio's :ref:`quick filters <studio-data-board>` do the
same job, but what you do next is different: discarding samples in the view
removes them from the **model's active set on the next step**, and tagging
them changes what your next evaluation covers. You are not preparing a list to
act on later — the action is the point.

**Predictions arrive continuously.** In FiftyOne you run inference, then load
predictions onto samples as a labelled field. In WeightsLab the predictions are
already flowing, because the loss that produced them is watched:

.. code-block:: python

   wl.save_signals(preds_raw=outputs, targets=targets, batch_ids=ids,
                   signals={"test_metric/Accuracy_per_sample": acc_per_sample},
                   preds=preds)      # processed: post-NMS / post-argmax

Pass ``preds`` **processed** — after NMS or argmax — because the studio renders
them directly as overlays.

**The brain methods have no equivalent.** ``compute_similarity``,
``compute_uniqueness``, ``compute_mistakenness`` and friends are FiftyOne
features with no WeightsLab counterpart. What WeightsLab gives you instead is
the per-sample loss trajectory over the whole run — a different, training-time
signal for "which samples are difficult". If you rely on the brain methods,
keep FiftyOne for that stage.

Replaced parts
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - FiftyOne
     - WeightsLab
   * - ``fo.Dataset.from_dir(...)``
     - ``wl.watch_or_edit(dataset, flag="data", loader_name=...)`` — no copy
   * - ``sample["ground_truth"]``
     - Your dataset's own ``get_items(..., include_labels=True)``
   * - ``sample["predictions"] = fo.Detections(...)``
     - ``wl.save_signals(..., preds=preds)`` during eval
   * - ``sample.tags.append("hard")``
     - ``wl.tag_samples(...)``, or tag from the UI's painter mode
   * - ``dataset.match(F("loss") > 2.0)``
     - Quick filters, or a plain-language agent query
   * - ``view.tag_samples(...)`` on a view
     - Select in the grid → right-click → tag
   * - Excluding samples from a view
     - ``wl.discard_samples(...)`` — actually removes them from training
   * - ``fo.launch_app(dataset)``
     - ``weightslab start <experiment_dir>``
   * - ``dataset.export(..., dataset_type=...)``
     - :func:`export_annotations` → CVAT, Label Studio, V7
       (see :doc:`../export`)
   * - ``fob.compute_uniqueness`` / ``compute_mistakenness``
     - *No equivalent* — per-sample loss trajectories serve a similar purpose
       at training time
   * - Dataset persistence / versioning
     - ``wl.write_dataframe()`` into the experiment directory

Updated examples
----------------

**Before** — curate in FiftyOne, then train:

.. code-block:: python
   :emphasize-lines: 4,10,13,15,16,19

   import fiftyone as fo
   import fiftyone.brain as fob

   dataset = fo.Dataset.from_dir(
       dataset_dir="data/",
       dataset_type=fo.types.ImageClassificationDirectoryTree)

   # ... run inference in a separate script, load predictions back on ...
   for sample, pred in zip(dataset, predictions):
       sample["predictions"] = fo.Classification(label=pred)
       sample.save()

   fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")

   view = dataset.sort_by("mistakenness", reverse=True)[:100]
   view.tag_samples("needs_review")

   session = fo.launch_app(view)
   # then: export the tags, edit the training script, retrain

**After** — curate while training:

.. code-block:: python
   :emphasize-lines: 4,5,9,10,13,21

   import itertools
   import weightslab as wl

   # The dataset you already have -- implement get_items on it (see above).
   train_loader = wl.watch_or_edit(
       train_dataset, flag="data", loader_name="train_loader",
       batch_size=16, shuffle=True, is_training=True)

   criterion = wl.watch_or_edit(
       nn.CrossEntropyLoss(reduction="none"),
       flag="loss", signal_name="train-loss-CE", log=True)

   wl.serve(serving_grpc=True)

   for step in itertools.count():
       images, targets, ids = next(train_iter)
       outputs = model(images)
       loss_per_sample = criterion(outputs, targets, batch_ids=ids)
       loss_per_sample.mean().backward(); optimizer.step(); optimizer.zero_grad()

   wl.keep_serving()

There is no separate curation script, no export/re-import round trip, and no
retrain step. You sort the grid by loss in the studio, tag or discard what you
find, and the next step already reflects it.

Expanded UI documentation
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - In the FiftyOne App you would…
     - In Weights Studio
   * - Browse the sample grid
     - The :ref:`data board <studio-data-board>` — grid or a sortable
       :ref:`list view <studio-data-board>`
   * - Build a view in the sidebar
     - :ref:`Quick filters <studio-data-board>`, no LLM involved; a banner
       reports what matched and ``@reset`` clears it
   * - Tag samples in a view
     - Selection + context menu, or **painter mode** — drag a tag straight
       across grid cells
   * - Toggle label fields on and off
     - Raw / ground-truth / prediction overlays, plus **diff** and **split**
       comparison modes in the detail modal
   * - Open the sample modal
     - The :ref:`detail modal <studio-detail-modal>` — with a 3D viewer for
       point clouds and frame stepping for clips
   * - Read a histogram in the sidebar
     - Right-click any metadata column → histogram
   * - Export tags and go retrain
     - Nothing to export — the edit already applied. Use
       :doc:`../export` only when you are sending data out for relabelling
