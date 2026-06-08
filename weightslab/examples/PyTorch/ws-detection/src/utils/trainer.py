import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)

import logging
import torch

import weightslab as wl

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.cfg import get_cfg

from utils.criterions import (
    PerSampleDetectionLoss, PerSampleIoU,
    PerInstanceDetectionLoss, PerInstanceIoU,
    _decode_predictions
)

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


# ==================
# Trainer Definition
# ==================

class WLCompatileDetTrainer(DetectionTrainer):
    def __init__(self, *a, **kw):
        self.data_train_loader = kw.pop("train_loader")
        self.data_val_loader = kw.pop("val_loader")
        self.hparams = kw.pop("hparams", {})
        self.device = kw.get("device", torch.device("cpu"))
        super().__init__(*a, **kw)
        self._init_experiment_modules()

    @property
    def val_every(self):
        return self.hparams.get("eval_full_to_train_steps_ratio", 1)

    def _init_experiment_modules(self):
        super().setup_model()
        self.model = self.model.to(self.device)

        # Order matters: _setup_train builds the optimizer, which iterates model.modules()
        # with isinstance checks that fail once WL wraps the model. Run it BEFORE wrapping.
        self._init_training_loss()
        self._setup_train()
        self.optimizer = wl.watch_or_edit(self.optimizer, flag="optimizer")
        self.model = wl.watch_or_edit(
            self.model, flag="model", device=self.device, compute_dependencies=False)

    def _init_training_loss(self):
        if not hasattr(self.model, "args"):
            self.model.args = get_cfg()
        _LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]
        self.criterions = {"train": {}, "val": {}}
        self.criterions_per_instance = {"train": {}, "val": {}}
        self.iou = {}
        self.iou_per_instance = {}
        for split in ("train", "val"):
            # Per-sample metrics (aggregated)
            for t, n in _LOSS_PARTS:
                self.criterions[split][n] = wl.watch_or_edit(
                    PerSampleDetectionLoss(self.model, loss_type=t),
                    flag="loss", name=f"{split}/{n}_sample", per_sample=True, log=True,
                )
            self.iou[split] = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name=f"{split}/iou_sample", per_sample=True, log=True,
            )

            # Per-instance metrics (per annotation) — auto-saved by WL with annotation_id
            for t, n in _LOSS_PARTS:
                self.criterions_per_instance[split][n] = wl.watch_or_edit(
                    PerInstanceDetectionLoss(self.model, loss_type=t),
                    flag="loss", name=f"{split}/{n}_instance", per_instance=True, log=True,
                )
            self.iou_per_instance[split] = wl.watch_or_edit(
                PerInstanceIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name=f"{split}/iou_instance", per_instance=True, log=True,
            )

    def process_predictions(self, pred_raw, image, conf=0.25, cls_thresh=0.5):
        img_h, img_w = image[0].shape[-2:]

        # Process check for eval mode
        if isinstance(pred_raw, (tuple, list)):
            pred_raw = pred_raw[1]

        # Gen. bounding boxes
        pred_raw = torch.cat([pred_raw['boxes'], pred_raw['scores']], dim=1)
        # Convert to raw model predictions format [batch, 64+nc, 8400]
        preds_bboxes, preds_cls = _decode_predictions(
            pred_raw, img_h, img_w, conf=conf, iou_thres=cls_thresh)

        return preds_bboxes, preds_cls

    def train(self):
        cs, m = self.criterions["train"], self.iou["train"]
        cs_inst, m_inst = self.criterions_per_instance["train"], self.iou_per_instance["train"]

        while True:
            with wl.guard_training_context:
                self.optimizer.zero_grad()
                inputs = next(self.data_train_loader)
                image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']

                raw_preds = self.model(image.to(self.device))
                if not isinstance(raw_preds, dict):
                    raw_preds = raw_preds[1]  # For audit mode, we are in evaluation so model also output the bb
                preds_bboxes, preds_cls = self.process_predictions(raw_preds, image, conf=0.0001, cls_thresh=0.0001)
                imgsz = float(image.shape[-1])
                preds_by_batch = [
                    torch.cat([b.detach().cpu() / imgsz, c[:, 1:2].cpu(), c[:, 0:1].cpu()], dim=-1)
                    if b.numel() > 0 else torch.zeros((0, 6))
                    for b, c in zip(preds_bboxes, preds_cls)
                ]

                # Per-sample signals — used for backward pass
                per_sample = (
                    cs["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
                    + cs["clsf"](raw_preds, batch, batch_ids=batch_ids)
                    + cs["dfl"](raw_preds, batch, batch_ids=batch_ids)
                )
                m(raw_preds, batch, batch_ids=batch_ids)

                # Per-instance signals (per annotation) — auto-saved with annotation_id
                cs_inst["bbxs"](raw_preds, batch, batch_ids=batch_ids)
                cs_inst["clsf"](raw_preds, batch, batch_ids=batch_ids)
                cs_inst["dfl"](raw_preds, batch, batch_ids=batch_ids)
                m_inst(raw_preds, batch, batch_ids=batch_ids)

                loss = per_sample.mean()
                loss.backward()
                self.optimizer.step()

            if self.model.get_age() % self.val_every == 0:
                with wl.guard_testing_context:
                    self.do_validate(self.data_val_loader)

            # Write the history of these samples every x steps
            if self.model.get_age() % 100 == 0:
                wl.write_history(
                    # path=None,  # Use root_log_dir by default, filename generated from parameters md5 hash
                    type_of_history="instance",
                    graph_name=[
                        'train/clsf_instance',
                        'val/clsf_instance'
                    ],
                    # experiment_hash=None,  Default is 'last', i.e., current experiment hash
                    sample_id=['11', '29', '28', '27', '22'],
                    instance_id=[1, 2, 3]
                )
                # Dump the sample dataframe: all signals plus the loss_shape categorical tag,
                wl.write_dataframe(
                    columns=["signals", "tag:loss_shape"],
                    format='csv'
                    # sample_id=['0', '28']
                    # instance_id=[1, 2],
                )

    def do_validate(self, loader):
        cs, m = self.criterions["val"], self.iou["val"]
        cs_inst, m_inst = self.criterions_per_instance["val"], self.iou_per_instance["val"]
        for inputs in loader:
            image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']
            raw_preds = self.model(image.to(self.device))[1]
            preds_bboxes, preds_cls = self.process_predictions(raw_preds, image, conf=0.0001, cls_thresh=0.0001)
            imgsz = float(image.shape[-1])
            preds_by_batch = [
                torch.cat([b.detach().cpu() / imgsz, c[:, 1:2].cpu(), c[:, 0:1].cpu()], dim=-1)
                if b.numel() > 0 else torch.zeros((0, 6))
                for b, c in zip(preds_bboxes, preds_cls)
            ]

            # Per-sample metrics (aggregated per sample)
            cs["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
            cs["clsf"](raw_preds, batch, batch_ids=batch_ids)
            cs["dfl"](raw_preds, batch, batch_ids=batch_ids)
            m(raw_preds, batch, batch_ids=batch_ids)

            # Per-instance metrics (per annotation) — auto-saved with annotation_id
            cs_inst["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
            cs_inst["clsf"](raw_preds, batch, batch_ids=batch_ids)
            cs_inst["dfl"](raw_preds, batch, batch_ids=batch_ids)
            m_inst(raw_preds, batch, batch_ids=batch_ids)
