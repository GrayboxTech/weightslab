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

from weightslab.backend.logger import LoggerQueue

from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from utils.trainer import WLCompatileDetTrainer
from utils.data import YOLODatasetWL, _wl_yolo_collate as collate_fn

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


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
    val_cfg = dict(parameters["data"]["val_loader"])
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
        c = train_cfg if split == "train" else val_cfg
        return wl.watch_or_edit(
            ds, flag="data", loader_name=f"{split}_loader",
            batch_size=c["batch_size"], shuffle=c["shuffle"],
            num_workers=2,
            drop_last=False, compute_hash=False,
            is_training=(split == "train"),
            collate_fn=collate_fn,
            preload_labels=True,
            preload_metadata=True,
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
            workers=2, cache=False, optimizer="SGD", lr0=0.001,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        hparams=parameters,
    )

    # Wrap the custom function to be able to use WeightsStudio Evaluation Feature
    @wl.eval_fn
    def _wl_validate(loader):
        trainer.do_validate(loader)

    # ================
    # 7. Training Loop
    wl.start_training(timeout=None)  # This will block and keep the main thread alive while background services run. You can optionally set a timeout (in seconds) to automatically stop after a certain duration.
    trainer.train()
    wl.keep_serving()


if __name__ == "__main__":
    main()
