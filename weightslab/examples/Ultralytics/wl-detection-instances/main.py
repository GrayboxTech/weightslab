"""YOLO + WeightsLab, one projection point per BOX.

The per-image projection says which images look alike. This example projects
the OBJECTS: every labelled box and every confident prediction with no label
under it, train and validation on one map (``wl.project_instances``). Colour it
by class to see a ``truck`` box sitting inside the ``car`` cluster, or by
match to see where the model and the labels disagree.

Two ways in, both shown here:

* **automatic** -- the ``instance_projection`` section of config.yaml makes
  ``wl.WLAwareTrainer`` project every box at the end of every N epochs
  (prefix ``instances``);
* **explicit** -- after training, ``project_trainer_instances`` projects again,
  with PCA this time, under a second prefix (``instances_pca``): the board's
  picker switches between the two.

Then the process keeps serving: open the Studio, press the Projection button
in the Data header and pick ``instances``.
"""
import os

os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import logging
import tempfile
import warnings

import torch
import yaml

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)
try:
    from tables import NaturalNameWarning
    warnings.filterwarnings("ignore", category=NaturalNameWarning)
except Exception:
    pass

import weightslab as wl
from ultralytics import YOLO
from weightslab.integrations.ultralytics import project_trainer_instances

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    if cfg.get("device", "auto") == "auto":
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not cfg.get("root_log_dir"):
        cfg["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(cfg["root_log_dir"], exist_ok=True)
    wl.watch_or_edit(cfg, flag="hyperparameters", defaults=cfg, poll_interval=1.0)

    wl.serve(serving_grpc=cfg.get("serving_grpc", True),
             serving_cli=cfg.get("serving_cli", False))
    if cfg.get("is_training", False):
        # Otherwise training waits for Play in the Studio.
        wl.start_training()

    yolo = YOLO(cfg["model"]["name"])
    yolo.train(
        trainer=wl.WLAwareTrainer,      # runs the box projection every N epochs
        data=str(cfg["data_root"]),
        imgsz=cfg.get("image_size"),
        epochs=int(cfg.get("epochs") or 3),
        device=cfg["device"],
        project=cfg["root_log_dir"], name=cfg["experiment_name"],
        exist_ok=True,
        resume=False,
        cache=False,
        workers=0,
        optimizer="SGD",
        lr0=0.001,
        amp=False,
        # Augmentation off: a training image is one sample, boxes and all.
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.0, erasing=0.0,
        auto_augment=None,
    )

    # The explicit call: the same boxes, another layout, another prefix.
    stats = project_trainer_instances(yolo.trainer, prefix="instances_pca", method="pca")
    print(
        f"[instances] {stats['labels']} labelled boxes + {stats['predictions']} "
        f"unlabelled predictions over {stats['samples']} images; matches: {stats['matches']}",
        flush=True,
    )

    wl.keep_serving()


if __name__ == "__main__":
    main()
