Detection — YOLO (Ultralytics)
===============================

.. raw:: html

   <div class="wl-eg-page-tags">
     <span class="wl-eg-badge wl-eg-badge--ultralytics">Ultralytics</span>
     <span class="wl-eg-tag">detection</span>
     <span class="wl-eg-tag">yolo</span>
     <span class="wl-eg-tag">object detection</span>
     <span class="wl-eg-tag">mAP</span>
   </div>

The full Ultralytics integration documentation is at :doc:`/ultralytics`.

This page summarises what ``WLAwareTrainer`` handles automatically and where
the example lives.

**Example:** ``weightslab/examples/Ultralytics/ws-detection/main.py``

What the example does
---------------------

One ``YOLO.train(trainer=WLAwareTrainer, ...)`` call gives you:

- Per-sample box / cls / dfl loss and live NMS overlay (train split).
- Per-sample IoU and post-NMS overlay (val split).
- Aggregate ``train/{box,cls,dfl}`` and ``val/{precision,recall,mAP50,...}``
  curves in the studio.
- Discard control: samples removed in the studio are silently excluded from
  future batches on both splits.

Integration in three lines
--------------------------

.. code-block:: python

   wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)
   wl.serve()
   YOLO("yolo11n.pt").train(trainer=WLAwareTrainer, data=..., workers=0, amp=False)
   wl.keep_serving()

Required kwargs: ``workers=0`` (UID counter lives in the parent process),
``amp=False`` (autocast doesn't see through the ``ModelInterface`` wrapper).

See :doc:`/ultralytics` for config walkthrough, platform notes, and the full
end-to-end sequence.
