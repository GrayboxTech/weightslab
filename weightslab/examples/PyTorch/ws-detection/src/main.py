"""Full WL integration with Ultralytics YOLO detection training.

Includes:
- WL parameter watching and UI editing
- Data loader tracking with UIDs and deny-list support
- Training/valing guards for signal logging
- Per-sample loss and metrics tracking
- Model, optimizer, and loss function wrapping
"""
import os
# Disable WL's preview-cache prewarm. Must be set BEFORE `import weightslab` so the
# DataService constructor (site-packages/weightslab/trainer/services/data_service.py:279)
# sees it. With this off, the studio generates 64x64 thumbnails on-demand instead of
# pre-warming all 1190 samples (~2.5 min of startup at imgsz=1024).
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")

# Silence pytables NaturalNameWarning: WL writes column names like
# 'signals.defaults.saturation_dtype' (dotted) which pytables flags as not valid
# Python identifiers for natural-name access. WL uses getattr-style access so the
# warning is noise. Filtered here, before `import weightslab` triggers the first H5 write.
import warnings
from tables import NaturalNameWarning
warnings.filterwarnings("ignore", category=NaturalNameWarning)
# Pandas 3.x deprecation: WL's h5_dataframe_store writes string 'nan' arrays into
# typed columns, which spams a multi-KB FutureWarning every flush. Targeted by
# message so unrelated FutureWarnings still surface.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated.*",
)

import time
import logging
import tempfile
from xml.parsers.expat import model

import torch
import yaml
import numpy as np

import weightslab as wl

from weightslab.utils.logger import LoggerQueue
from weightslab.components.global_monitoring import pause_controller
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
# from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss

from utils.data import YOLODatasetWL, _wl_yolo_collate
from utils.criterions import (
    PerSampleDetectionLoss,
    PerSampleIoU,
    AccumulatedConsecutiveAbsWeightDiff,
    _decode_predictions,
)


logging.basicConfig(level=logging.ERROR)

# Quiet noisy WL informational loggers. dataframe_manager dumps a line per
# batch (`Enqueued N records ...`) plus per-minute flush ticks — drowns out
# real signal. Override after WL has set up its own loggers (via the import
# above) so this stays sticky.
for _wl_logger in (
    "weightslab.data.dataframe_manager",
):
    logging.getLogger(_wl_logger).setLevel(logging.WARNING)


class WLCompatileDetTrainer(DetectionTrainer):
    """DetectionTrainer with WL integration for signals, parameters, and guards."""

    def __init__(self, *a, **kw):
        # Loaders
        self.data_train_loader = kw.pop("train_loader", None)
        self.data_val_loader = kw.pop("val_loader", None)

        # Custom criterion
        self.train_criterion = kw.pop("train_criterion", None)
        self.val_criterion = kw.pop("val_criterion", None)

        # Metrics
        self.train_metric = kw.pop("train_metric", None)
        self.val_metric = kw.pop("val_metric", None)

        # Experiment parameters
        self.val_every = kw.pop("val_every", 1)
        self.device = kw.get("device", torch.device("cpu"))

        # Init after removing parameters
        super().__init__(*a, **kw)

        # Init experiment modules (model, criterions)
        self._init_experiment_modules()

    def _init_experiment_modules(self):
        """
            Initialize experiment modules (i.e., model and criterion, optimizer already include in model).
        """
        # WL-wrapped model will be set in setup_model
        super().setup_model()
        self.model = self.model.to(self.device)

        # Init criterions (need to be done before model wrapping for DetectionLoss module from ultralytics)
        self._init_training_loss()

        # Order matters: ultralytics' build_optimizer iterates model.modules()
        # with isinstance(m, nn.Conv2d/Linear/BatchNorm). After WL wraps the
        # model those type checks fail and the optimizer ends up with 0 param
        # groups → no learning. Run _setup_train BEFORE wrapping.
        self._setup_train()

        # Bare-model param refs BEFORE the wrap. The wrap's WL-checkpoint auto-load
        # swaps every Parameter tensor (a fresh model is built from the saved
        # state_dict and registered into the ledger). Without re-binding, the
        # optimizer keeps referencing the bare model's now-orphan tensors:
        # optimizer.step() moves them, but the wrapped model's forward never reads
        # them → weight_diff stays 0 and the model never actually learns.
        _bare_params = list(self.model.parameters())

        # Wrap the model (triggers WL checkpoint auto-load).
        self.model = wl.watch_or_edit(
            self.model,
            flag="model",
            device=self.device,
            compute_dependencies=False
        )

        # Re-bind optimizer's param_groups to the post-wrap (checkpoint-loaded)
        # params, position-matched (architecture is unchanged). Migrate per-tensor
        # optimizer state (momentum etc.) so resume keeps its warmup history.
        _wrapped_params = list(self.model.parameters())
        assert len(_bare_params) == len(_wrapped_params), (
            f"Param count drifted across wrap "
            f"({len(_bare_params)} → {len(_wrapped_params)}); rebind unsafe")
        _remap = {id(b): w for b, w in zip(_bare_params, _wrapped_params)}
        for group in self.optimizer.param_groups:
            new_params = []
            for p in group["params"]:
                w = _remap.get(id(p), p)
                if w is not p and p in self.optimizer.state:
                    self.optimizer.state[w] = self.optimizer.state.pop(p)
                new_params.append(w)
            group["params"] = new_params

        # Tracks Σ|p_now - p_prev| across trainable params per step. Built from
        # self.model AFTER the optimizer rebind above, so the snapshot lives on
        # the same Parameter tensors the optimizer updates.
        self.weight_diff_monitor = wl.watch_or_edit(
            AccumulatedConsecutiveAbsWeightDiff(self.model),
            flag="metric", name="weight_diff_per_step",
            per_sample=False, log=True,
        )

    def _init_training_loss(self):
        """
            Initialize exp. criterions.
        """
        # Use the model's built-in criterion if not provided
        if self.train_criterion is None:
            if not hasattr(self.model, 'args'):
                self.model.args = get_cfg()  # Fallback if model doesn't have args
            # # Wrap loss criterion
            self.train_criterion_boxes = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=0),
                flag="loss",
                name="train/bbxs",
                per_sample=True,
                log=True,
            )
            self.train_criterion_cls = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=1),
                flag="loss",
                name="train/clsf",
                per_sample=True,
                log=True,
            )
            self.train_criterion_dfl = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=2),
                flag="loss",
                name="train/dfl",
                per_sample=True,
                log=True,
            )
        if self.train_metric is None:
            # # Wrap IoU metric
            self.train_metric = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric",
                name="miou/train",
                per_sample=True,
                log=True,
            )
        if self.val_criterion is None:
            # # Wrap loss criterion
            self.val_criterion_boxes = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=0),
                flag="loss",
                name="val/bbxs",
                per_sample=True,
                log=True,
            )
            self.val_criterion_cls = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=1),
                flag="loss",
                name="val/clsf",
                per_sample=True,
                log=True,
            )
            self.val_criterion_dfl = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=2),
                flag="loss",
                name="val/dfl",
                per_sample=True,
                log=True,
            )
        if self.val_metric is None:
            # # Wrap IoU metric
            self.val_metric = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric",
                name="miou/val",
                per_sample=True,
                log=True,
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
        """
            Override the default step loop to add WL guards and per-sample IoU tracking.
            This is a simplified loop for demonstration.
        """
        loss = 0.0
        loader = self.data_train_loader

        while True:
            with wl.guard_training_context:
                self.optimizer.zero_grad()  # Zero gradients at the start of the step

                # --- One training batch ---
                inputs = next(loader)

                image = inputs[0].float()
                batch_ids = inputs[1]  # uids

                batch = inputs[3]['batch']

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

                # Compute training loss with per-sample tracking. Pass preds as the bare
                # list (framework's index_batch slices by batch index); row[PREDICTION]
                # ends up as the (N_i, 6) tensor.
                # Each criterion already returns a stacked per-sample tensor (see
                # PerSampleDetectionLoss.forward); the old torch.stack(list(...))
                # wraps were no-op rebuilds. Sum per-sample, then reduce once.
                per_sample = (
                    self.train_criterion_boxes(
                        raw_preds,
                        batch,
                        batch_ids=batch_ids,
                        preds={'bboxes': preds_by_batch}
                    )
                    + self.train_criterion_cls(raw_preds, batch, batch_ids=batch_ids)
                    + self.train_criterion_dfl(raw_preds, batch, batch_ids=batch_ids)
                )
                loss = per_sample.mean()

                # Drives WL signal logging (return value unused — studio reads it).
                self.train_metric(raw_preds, batch, batch_ids=batch_ids)

                # Learning
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # Drives the weight_diff_per_step WL signal. 0 across many steps
                # would mean the optimizer isn't moving the model's params.
                self.weight_diff_monitor()

            # --- Validate on full val set ---
            step = self.model.get_age()
            if step % self.val_every == 0:
                with wl.guard_testing_context:
                    self.validate()

    def validate(self):
        """Instance method that delegates to standalone validate function."""
        validate(self.data_val_loader)

@wl.eval_fn
def validate(loader):
    """
    Standalone validation function that uses parameters from the ledger.

    This function retrieves the model, device, and metric/loss functions from the
    WeightsLab ledger, making it suitable for independent evaluation workflows.

    Args:
        loader: Data loader providing batches of validation data.
    """
    # Get components from ledger
    model = wl.ledger.get_model()
    device = model.device
    # Names must match what was registered in _init_training_loss() — these were renamed
    # to short forms (e.g. 'val_detection_loss/bboxes' → 'val/bbxs').
    val_criterion_boxes = wl.ledger.get_loss(name="val/bbxs")
    val_criterion_cls = wl.ledger.get_loss(name="val/clsf")
    val_criterion_dfl = wl.ledger.get_loss(name="val/dfl")
    val_metric = wl.ledger.get_metric(name="miou/val")

    if model is None or device is None:
        raise RuntimeError("Model or device not found in ledger. Ensure they are registered with wl.watch_or_edit()")

    # --- One validation batch ---
    l_loader = len(loader)
    for step, inputs in enumerate(loader):
        # Process inputs
        image = inputs[0].float()
        batch_ids = inputs[1]  # uids

        # Process dataset labels (already in ultralytics flat format)
        batch = inputs[3]['batch']

        # Eval mode model in ultralytics returns (pred, dict of preds) - train mode returns dict of preds
        raw_preds = model(image.to(device))[1]

        # Process outputs to generate predictions as BB
        # Helper function to process predictions
        def process_predictions(pred_raw, image, conf=0.001, cls_thresh=0.001):
            img_h, img_w = image[0].shape[-2:]
            # if isinstance(pred_raw, (tuple, list)):
            #     pred_raw = pred_raw[1]
            pred_raw = torch.cat([pred_raw['boxes'], pred_raw['scores']], dim=1)
            preds_bboxes, preds_cls = _decode_predictions(pred_raw, img_h, img_w, conf=conf, iou_thres=cls_thresh)
            return preds_bboxes, preds_cls

        preds_bboxes, preds_cls = process_predictions(raw_preds, image)

        # Convert predictions to [N, 6] format ([x1, y1, x2, y2, class_id, score])
        batch_size = image.shape[0]
        imgsz = float(image.shape[-1])
        preds_by_batch = [
            torch.cat([b.detach() / imgsz, c[:, 1:2], c[:, 0:1]], dim=-1) if b.numel() > 0 else torch.zeros((0, 6), device=device)
            for b, c in zip(preds_bboxes, preds_cls)
        ]

        # Same pattern as train: criterions already return stacked per-sample
        # tensors, sum them, reduce once. (None-guards dropped — these are
        # registered in _init_training_loss; if they aren't, fail loudly.)
        per_sample = (
            val_criterion_boxes(raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
            + val_criterion_cls(raw_preds, batch, batch_ids=batch_ids)
            + val_criterion_dfl(raw_preds, batch, batch_ids=batch_ids)
        )
        loss = per_sample.mean()

        # Drives WL signal logging (return value unused — studio reads it).
        if val_metric is not None:
            val_metric(raw_preds, batch, batch_ids=batch_ids)
        print(f'\tLoss value during validation is {loss} at step {step}/{l_loader}.')

def main():
    start_time = time.time()

    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "det_YOLO_usecase")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", None)  # None = infinite
    parameters.setdefault("eval_full_to_train_steps_ratio", 5)
    parameters.setdefault("experiment_dump_to_train_steps_ratio", 5)
    parameters.setdefault("num_classes", 2)
    parameters.setdefault("image_size", None)
    parameters.setdefault("data_root")
    parameters.setdefault("class_names", ["class_0", "class_1"])  # Default class names
    parameters.setdefault("compute_natural_sort", True)
    parameters.setdefault("is_training", False)
    parameters.setdefault("serving_grpc", True)
    parameters.setdefault("serving_cli", False)
    parameters.setdefault("tqdm_display", True)

    # Nested data and model configs with defaults
    if "data" not in parameters:
        parameters["data"] = {}
    parameters["data"].setdefault("train_loader", {})
    parameters["data"].setdefault("test_loader", {})
    parameters["data"]["train_loader"].setdefault("batch_size", 4)
    parameters["data"]["train_loader"].setdefault("shuffle", True)
    parameters["data"]["test_loader"].setdefault("batch_size", 4)
    parameters["data"]["test_loader"].setdefault("shuffle", False)

    if "model" not in parameters:
        parameters["model"] = {}
    if isinstance(parameters["model"], str):
        # If model is just a string (name), convert to dict
        parameters["model"] = {"name": parameters["model"]}
    parameters["model"].setdefault("name", "yolo11s.pt")
    parameters["model"].setdefault("lr", 0.01)
    parameters["model"].setdefault("momentum", 0.9)

    parameters.setdefault("checkpoint_manager", {})
    parameters["checkpoint_manager"].setdefault("load_config", True)
    parameters["checkpoint_manager"].setdefault("load_data", True)
    parameters["checkpoint_manager"].setdefault("load_model", True)

    # Extract key parameters
    exp_name = parameters["experiment_name"]
    image_size = parameters["image_size"]
    max_steps = parameters["training_steps_to_do"]
    eval_every = parameters["eval_full_to_train_steps_ratio"]

    # Data loader configs
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})
    batch_size = train_cfg.get("batch_size", 4)

    # Model config
    model_name = parameters["model"]["name"]

    # Data root
    data_root = parameters.get("data_root")
    if data_root is None:
        data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), './data/data.yaml')

    # --- 2) Device selection ---
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    # --- 3) Logging directory ---
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]

    # --- 4) Register logger + hyperparameters ---
    logger = LoggerQueue()
    wl.watch_or_edit(logger, flag="logger", name=exp_name, log_dir=log_dir)
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # --- 5) Build YOLO config with augmentation disabled for consistency ---
    cfg = get_cfg()
    if image_size is not None:
        cfg.imgsz = image_size
    for k in ("mosaic", "mixup", "copy_paste",
              "hsv_h", "hsv_s", "hsv_v",
              "degrees", "translate", "scale", "shear", "perspective",
              "flipud", "fliplr", "erasing"):
        setattr(cfg, k, 0.0)
    cfg.auto_augment = None

    # --- 6) Build YOLO datasets ---
    checked_data = check_det_dataset(data_root)
    train_yolo_dataset = YOLODataset(
            img_path=checked_data["train"],
            imgsz=cfg.imgsz,
            batch_size=batch_size,
            augment=True,  # augmentation
            hyp=cfg,
            rect=cfg.rect,  # rectangular batches
            cache=False,
            single_cls=cfg.single_cls or False,
            stride=32,
            pad=0.0,
            task=cfg.task,
            classes=cfg.classes,
            data=checked_data,
            fraction=cfg.fraction,
    )
    train_yolo_dataset.__class__ = YOLODatasetWL  # Monkey-patch to our WL-compatible dataset for loader tracking
    val_yolo_dataset = YOLODataset(
            img_path=checked_data["val"],
            imgsz=cfg.imgsz,
            batch_size=batch_size,
            augment=False,  # No augmentation for validation
            hyp=cfg,
            rect=cfg.rect ,  # rectangular batches
            cache=False,
            single_cls=cfg.single_cls or False,
            stride=32,
            pad=0.5,
            task=cfg.task,
            classes=cfg.classes,
            data=checked_data,
            fraction=1.0,
    )
    val_yolo_dataset.__class__ = YOLODatasetWL  # Monkey-patch to our WL-compatible dataset for loader tracking

    # --- 7) Create WL-watched loaders (num_workers=0 to avoid fork issues after wl.serve()) ---
    train_loader = wl.watch_or_edit(
        train_yolo_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", batch_size),
        shuffle=train_cfg.get("shuffle", True),
        num_workers=0,
        drop_last=False,
        compute_hash=False,
        is_training=True,
        collate_fn=_wl_yolo_collate,
        preload_labels=True,
        preload_metadata=True,
    )
    val_loader = wl.watch_or_edit(
        val_yolo_dataset,
        flag="data",
        loader_name="val_loader",
        batch_size=test_cfg.get("batch_size", batch_size),
        shuffle=test_cfg.get("shuffle", False),
        num_workers=0,
        drop_last=False,
        compute_hash=False,
        is_training=False,
        collate_fn=_wl_yolo_collate,
        preload_labels=True,
        preload_metadata=True,
    )

    # --- 8) Initialize WL's gRPC server ---
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", False),
    )

    # --- 9) Train with WL-compatible trainer ---
    trainer = WLCompatileDetTrainer(
        overrides=dict(
            model=model_name,
            data=str(data_root),
            epochs=1000 if max_steps is None else max(1, max_steps // len(train_loader)),  # Convert steps to epochs
            imgsz=image_size,
            batch=batch_size,
            resume=False,
            device=device,
            workers=0,  # Single process
            cache=False,
            optimizer="SGD",
            lr0=0.001,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        val_every=eval_every,
    )

    # Start training
    trainer.train()

    wl.keep_serving()


if __name__ == "__main__":
    main()
