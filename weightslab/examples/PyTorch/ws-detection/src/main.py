"""WL integration for Ultralytics YOLO detection training."""
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
import tempfile

import torch
import yaml

import weightslab as wl
from weightslab.utils.logger import LoggerQueue
from weightslab.components.global_monitoring import pause_controller
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from utils.data import YOLODatasetWL, _wl_yolo_collate
from utils.criterions import PerSampleDetectionLoss, PerSampleIoU, _decode_predictions

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]


def _decode_preds_to_6col(raw_preds, image, conf, cls_thresh, device=None):
    if isinstance(raw_preds, (tuple, list)):
        raw_preds = raw_preds[1]
    img_h, img_w = image[0].shape[-2:]
    pred = torch.cat([raw_preds['boxes'], raw_preds['scores']], dim=1)
    preds_bboxes, preds_cls = _decode_predictions(pred, img_h, img_w, conf=conf, iou_thres=cls_thresh)
    imgsz = float(image.shape[-1])
    return [
        torch.cat([b.detach() / imgsz, c[:, 1:2], c[:, 0:1]], dim=-1).to(device) if b.numel() > 0
        else torch.zeros((0, 6), device=device)
        for b, c in zip(preds_bboxes, preds_cls)
    ]


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
        self.criterions = {"train": {}, "val": {}}
        self.iou = {}
        for split in ("train", "val"):
            for t, n in _LOSS_PARTS:
                self.criterions[split][n] = wl.watch_or_edit(
                    PerSampleDetectionLoss(self.model, loss_type=t),
                    flag="loss", name=f"{split}/{n}", per_sample=True, log=True,
                )
            self.iou[split] = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name=f"miou/{split}", per_sample=True, log=True,
            )

    def train(self):
        cs, m = self.criterions["train"], self.iou["train"]

        # `yield from loader` re-iterates each pass, re-invoking the sampler — so
        # shuffle=True re-shuffles each epoch instead of recycling order.
        def _infinite(loader):
            while True:
                yield from loader
        batches = _infinite(self.data_train_loader)

        while True:
            with wl.guard_training_context:
                self.optimizer.zero_grad()
                inputs = next(batches)
                image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']

                raw_preds = self.model(image.to(self.device))
                preds_by_batch = _decode_preds_to_6col(raw_preds, image, conf=0.1, cls_thresh=0.1)

                per_sample = (
                    cs["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
                    + cs["clsf"](raw_preds, batch, batch_ids=batch_ids)
                    + cs["dfl"](raw_preds, batch, batch_ids=batch_ids)
                )
                loss = per_sample.mean()
                m(raw_preds, batch, batch_ids=batch_ids)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            if self.model.get_age() % self.val_every == 0:
                with wl.guard_testing_context:
                    self.do_validate(self.data_val_loader)

    def do_validate(self, loader):
        cs, m = self.criterions["val"], self.iou["val"]
        for inputs in loader:
            image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']
            raw_preds = self.model(image.to(self.device))[1]
            preds_by_batch = _decode_preds_to_6col(
                raw_preds, image, conf=0.25, cls_thresh=0.5, device=self.device)

            cs["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
            cs["clsf"](raw_preds, batch, batch_ids=batch_ids)
            cs["dfl"](raw_preds, batch, batch_ids=batch_ids)
            m(raw_preds, batch, batch_ids=batch_ids)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parameters = yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    exp_name = parameters["experiment_name"]
    log_dir = parameters["root_log_dir"]
    device = parameters["device"]
    image_size = parameters.get("image_size")
    max_steps = parameters.get("training_steps_to_do")
    data_root = parameters["data_root"]
    model_name = parameters["model"]["name"]
    train_cfg = dict(parameters["data"]["train_loader"])
    test_cfg = dict(parameters["data"]["test_loader"])
    batch_size = train_cfg["batch_size"]
    serving_grpc = parameters.get("serving_grpc", True)
    serving_cli = parameters.get("serving_cli", False)

    wl.watch_or_edit(LoggerQueue(), flag="logger", name=exp_name, log_dir=log_dir)
    parameters = wl.watch_or_edit(parameters, flag="hyperparameters",
                                  name=exp_name, defaults=parameters, poll_interval=1.0)

    cfg = get_cfg()
    if image_size is not None:
        cfg.imgsz = image_size
    for k in ("mosaic", "mixup", "copy_paste",
              "hsv_h", "hsv_s", "hsv_v",
              "degrees", "translate", "scale", "shear", "perspective",
              "flipud", "fliplr", "erasing"):
        setattr(cfg, k, 0.0)
    cfg.auto_augment = None

    checked_data = check_det_dataset(data_root)

    def _build_dataset(split):
        is_train = split == "train"
        ds = YOLODataset(
            img_path=checked_data["train" if is_train else "val"],
            imgsz=cfg.imgsz, batch_size=batch_size,
            augment=is_train, hyp=cfg, rect=cfg.rect, cache=False,
            single_cls=cfg.single_cls or False,
            stride=32, pad=0.0 if is_train else 0.5,
            task=cfg.task, classes=cfg.classes, data=checked_data,
            fraction=cfg.fraction if is_train else 1.0,
        )
        ds.__class__ = YOLODatasetWL
        return ds

    def _build_loader(ds, split):
        c = train_cfg if split == "train" else test_cfg
        return wl.watch_or_edit(
            ds, flag="data", loader_name=f"{split}_loader",
            batch_size=c["batch_size"], shuffle=c["shuffle"],
            num_workers=0,
            drop_last=False, compute_hash=False,
            is_training=(split == "train"),
            collate_fn=_wl_yolo_collate,
            preload_labels=True, preload_metadata=True,
        )

    train_loader = _build_loader(_build_dataset("train"), "train")
    val_loader = _build_loader(_build_dataset("val"), "val")

    wl.serve(serving_grpc=serving_grpc, serving_cli=serving_cli)

    trainer = WLCompatileDetTrainer(
        overrides=dict(
            model=model_name,
            data=str(data_root),
            epochs=1000 if max_steps is None else max(1, max_steps // len(train_loader)),
            imgsz=image_size,
            batch=batch_size, resume=False,
            device=device,
            workers=0, cache=False, optimizer="SGD", lr0=0.001,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        hparams=parameters,
    )

    @wl.eval_fn
    def _wl_validate(loader):
        trainer.do_validate(loader)

    pause_controller.resume(force=True)
    trainer.train()
    wl.keep_serving()


if __name__ == "__main__":
    main()
