"""Full WL integration with Ultralytics YOLO detection training.

Includes:
- WL parameter watching and UI editing
- Data loader tracking with UIDs and deny-list support
- Training/valing guards for signal logging
- Per-sample loss and metrics tracking
- Model, optimizer, and loss function wrapping
"""
import os
import time
import logging
import tempfile
from xml.parsers.expat import model

import torch
import yaml
import numpy as np

import weightslab as wl

from weightslab.utils.logger import LoggerQueue
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
# from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss

from utils.data import YOLODatasetWL, _wl_yolo_collate
from utils.criterions import PerSampleDetectionLoss, PerSampleIoU, _decode_predictions


logging.basicConfig(level=logging.ERROR)


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

        # Finally wrap the model
        self.model = wl.watch_or_edit(
            self.model,
            flag="model",
            device=self.device,
            compute_dependencies=False
        )
        self._setup_train()  # init model, optimizer, scheduler in Ultralytics env.

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
                name="train_detection_loss/bboxes",
                per_sample=True,
                log=True,
            )
            self.train_criterion_cls = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=1),
                flag="loss",
                name="train_detection_loss/cls",
                per_sample=True,
                log=True,
            )
            self.train_criterion_dfl = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=2),
                flag="loss",
                name="train_detection_loss/dfl",
                per_sample=True,
                log=True,
            )
        if self.train_metric is None:
            # # Wrap IoU metric
            self.train_metric = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric",
                name="train_per_sample_iou",
                per_sample=True,
                log=True,
            )
        if self.val_criterion is None:
            # # Wrap loss criterion
            self.val_criterion_boxes = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=0),
                flag="loss",
                name="val_detection_loss/bboxes",
                per_sample=True,
                log=True,
            )
            self.val_criterion_cls = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=1),
                flag="loss",
                name="val_detection_loss/cls",
                per_sample=True,
                log=True,
            )
            self.val_criterion_dfl = wl.watch_or_edit(
                PerSampleDetectionLoss(self.model, loss_type=2),
                flag="loss",
                name="val_detection_loss/dfl",
                per_sample=True,
                log=True,
            )
        if self.val_metric is None:
            # # Wrap IoU metric
            self.val_metric = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric",
                name="val_per_sample_iou",
                per_sample=True,
                log=True,
            )

    def process_predictions(self, pred_raw, image, cls_thresh=0.5):
        img_h, img_w = image[0].shape[-2:]

        # Process check for eval mode
        if isinstance(pred_raw, (tuple, list)):
            pred_raw = pred_raw[1]

        # Gen. bounding boxes
        pred_raw = torch.cat([pred_raw['boxes'], pred_raw['scores']], dim=1)  # Convert to raw model predictions format [batch, 64+nc, 8400]
        preds_nms = _decode_predictions(pred_raw, img_h, img_w, conf=0.25, iou_thres=cls_thresh)

        return preds_nms

    def train(self):
        """
            Override the default step loop to add WL guards and per-sample IoU tracking.
            This is a simplified loop for demonstration.
        """
        loss = 0.0

        while True:
            with wl.guard_training_context:
                self.optimizer.zero_grad()  # Zero gradients at the start of the step

                # --- One training batch ---
                inputs = next(iter(self.data_train_loader))

                # Process inputs
                image = inputs[0].float()
                batch_ids = inputs[1]  # uids

                # Process dataset labels (already in ultralytics flat format)
                batch = inputs[3]['batch']
                targets = batch['bboxes']  # (N, 4)

                # Inference
                raw_preds = self.model(image.to(self.device))

                # Process outputs to generate predictions as BB (bs, x, 5 | 6)
                preds = self.process_predictions(raw_preds, image, cls_thresh=0.5)

                # Split preds and targets by batch index
                batch_size = image.shape[0]
                preds_by_batch = []
                targets_by_batch = []
                for i in range(batch_size):
                    mask = batch['batch_idx'].view(-1) == i
                    p = preds[mask] if isinstance(preds, torch.Tensor) and preds.shape[0] > 0 else torch.zeros((0, 4), device=self.device)
                    t = targets[mask] if isinstance(targets, torch.Tensor) and targets.shape[0] > 0 else torch.zeros((0, 4), device=self.device)
                    preds_by_batch.append(p)
                    targets_by_batch.append(t)

                # Compute training loss with per-sample tracking
                # # Bboxes
                per_sample_losses_bboxes = self.train_criterion_boxes(
                    raw_preds,
                    batch,
                    batch_ids=batch_ids,
                    preds={'bboxes': preds_by_batch},
                    targets={'bboxes': targets_by_batch}
                )
                per_sample_losses_bboxes = torch.stack(list(per_sample_losses_bboxes)) if per_sample_losses_bboxes else torch.zeros(batch_size, device=self.device)

                # # Cls
                per_sample_losses_cls = self.train_criterion_cls(raw_preds, batch, batch_ids=batch_ids)
                per_sample_losses_cls = torch.stack(list(per_sample_losses_cls)) if per_sample_losses_cls else torch.zeros(batch_size, device=self.device)

                # # Dfl
                per_sample_losses_dfl = self.train_criterion_dfl(raw_preds, batch, batch_ids=batch_ids)
                per_sample_losses_dfl = torch.stack(list(per_sample_losses_dfl)) if per_sample_losses_dfl else torch.zeros(batch_size, device=self.device)

                # # Compute final loss
                loss = (per_sample_losses_bboxes + per_sample_losses_cls + per_sample_losses_dfl).mean()

                # Compute training metric
                ious = self.train_metric(raw_preds, batch, batch_ids=batch_ids)
                iou_mean = ious.mean().item() if ious is not None and ious.numel() > 0 else 0.0

                # Verbose
                step = self.model.get_age()
                print(f"Step {step} — loss: {loss.item():.4f} (bboxes: {per_sample_losses_bboxes.mean().item():.4f}; cls: {per_sample_losses_cls.mean().item():.4f}; dfl: {per_sample_losses_dfl.mean().item():.4f}) - iou: {iou_mean:.4f}")

                # Learning
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

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
    val_criterion_boxes = wl.ledger.get_loss(name="val_detection_loss/bboxes")
    val_criterion_cls = wl.ledger.get_loss(name="val_detection_loss/cls")
    val_criterion_dfl = wl.ledger.get_loss(name="val_detection_loss/dfl")
    val_metric = wl.ledger.get_metric(name="val_per_sample_iou")

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
        targets = batch['bboxes']  # (N, 4)

        # Inference
        raw_preds = model(image.to(device))[1]  # Eval mode model in ultralytics returns (pred, dict of preds) - train mode returns dict of preds

        # Process outputs to generate predictions as BB
        # Helper function to process predictions
        def process_predictions(pred_raw, image, cls_thresh=0.5):
            img_h, img_w = image[0].shape[-2:]
            if isinstance(pred_raw, (tuple, list)):
                pred_raw = pred_raw[1]
            pred_raw = torch.cat([pred_raw['boxes'], pred_raw['scores']], dim=1)
            preds_nms = _decode_predictions(pred_raw, img_h, img_w, conf=0.25, iou_thres=cls_thresh)
            return preds_nms

        preds = process_predictions(raw_preds, image, cls_thresh=0.5)

        # Split preds and targets by batch index
        batch_size = image.shape[0]
        preds_by_batch = []
        targets_by_batch = []
        for i in range(batch_size):
            mask = batch['batch_idx'].view(-1) == i
            p = preds[mask] if isinstance(preds, torch.Tensor) and preds.shape[0] > 0 else torch.zeros((0, 4), device=device)
            t = targets[mask] if isinstance(targets, torch.Tensor) and targets.shape[0] > 0 else torch.zeros((0, 4), device=device)
            preds_by_batch.append(p)
            targets_by_batch.append(t)

        # Compute validation loss with per-sample tracking
        per_sample_losses_bboxes = val_criterion_boxes(
            raw_preds,
            batch,
            batch_ids=batch_ids,
            preds={'bboxes': preds_by_batch},
            targets={'bboxes': targets_by_batch}
        ) if val_criterion_boxes is not None else None
        per_sample_losses_bboxes = torch.stack(list(per_sample_losses_bboxes)) if per_sample_losses_bboxes else torch.zeros(batch_size, device=device)

        per_sample_losses_cls = val_criterion_cls(raw_preds, batch, batch_ids=batch_ids) if val_criterion_cls is not None else None
        per_sample_losses_cls = torch.stack(list(per_sample_losses_cls)) if per_sample_losses_cls else torch.zeros(batch_size, device=device)

        per_sample_losses_dfl = val_criterion_dfl(raw_preds, batch, batch_ids=batch_ids) if val_criterion_dfl is not None else None
        per_sample_losses_dfl = torch.stack(list(per_sample_losses_dfl)) if per_sample_losses_dfl else torch.zeros(batch_size, device=device)

        # Compute final loss
        loss = (per_sample_losses_bboxes + per_sample_losses_cls + per_sample_losses_dfl).mean()

        # Compute validation metric
        ious = val_metric(raw_preds, batch, batch_ids=batch_ids) if val_metric is not None else None
        iou_mean = ious.mean().item() if ious is not None and ious.numel() > 0 else 0.0

        # Verbose
        print(f"Validation step {step+1}/{l_loader} — loss: {loss.item():.4f} (bboxes: {per_sample_losses_bboxes.mean().item():.4f}; cls: {per_sample_losses_cls.mean().item():.4f}; dfl: {per_sample_losses_dfl.mean().item():.4f}) - iou: {iou_mean:.4f}")


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
    parameters.setdefault("data_root", "./data/data.yaml")
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
    num_classes = int(parameters["num_classes"])
    image_size = parameters["image_size"]
    max_steps = parameters["training_steps_to_do"]
    eval_every = parameters["eval_full_to_train_steps_ratio"]

    # Data loader configs
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})
    batch_size = train_cfg.get("batch_size", 4)

    # Model config
    model_name = parameters["model"]["name"]
    model_lr = float(parameters["model"]["lr"])
    model_momentum = float(parameters["model"]["momentum"])

    # Data root
    data_root = parameters.get("data_root", "./data/data.yaml")

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
        preload_labels=False,
        preload_metadata=False
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
        preload_labels=False,
        preload_metadata=False,
    )

    # --- 8) Initialize WL's gRPC server ---
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", False),
    )

    print("=" * 60)
    print("🚀 STARTING YOLO DETECTION TRAINING")
    print(f"📈 Training steps: {max_steps if max_steps else 'infinite'}")
    print(f"📊 Evaluation every {eval_every} steps")
    print(f"📊 Batch size: {batch_size}")
    print(f"🖼️ Image size: {image_size}")
    print(f"💾 Logs will be saved to: {log_dir}")
    print(f"📂 Data: {data_root}")
    print("=" * 60 + "\n")

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
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        val_every=eval_every,
    )

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive for WL UI exploration
    wl.keep_serving()


if __name__ == "__main__":
    main()
