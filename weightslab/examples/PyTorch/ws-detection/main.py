import os
import time
import tempfile
import logging

import tqdm
import torch
import yaml

from torch import nn
from torchvision import ops, transforms
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

import weightslab as wl

from weightslab.utils.logger import LoggerQueue as Logger
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)
from weightslab.baseline_models.pytorch.models import Yolov11


SUBSET = 100  # Use n images for quick testing


class GIoULoss(nn.Module):
    """Generalized IoU loss with box assignment and batching support.

    Handles mismatched numbers of predicted vs ground truth boxes by:
    1. Computing pairwise IoU between all predicted and ground truth boxes
    2. Greedily matching boxes with highest IoU
    3. Computing GIoU loss only on matched pairs

    Supports both single-image and batched inputs:
    - Single image: pred_boxes [M, 4], target_boxes [N, 4] -> scalar loss
    - Batched: pred_boxes [B, M, 4], target_boxes [B, N, 4] -> per-batch losses [B]

    Forward signature:
        forward(pred_boxes: Tensor[M, 4] or [B, M, 4],
                target_boxes: Tensor[N, 4] or [B, N, 4]) -> scalar or [B] loss
    """

    def __init__(self, reduction: str = "mean", iou_threshold: float = 0.1):
        super().__init__()
        self.reduction = reduction
        self.reduction = 'none' if reduction not in ['none', 'mean', 'sum'] else reduction
        self.iou_threshold = iou_threshold

    def _match_boxes_greedy(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
        """Greedy box matching using pairwise IoU (single image).

        Returns:
            matched_pred: [K, 4] matched predicted boxes
            matched_target: [K, 4] matched ground truth boxes
        """
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes, target_boxes

        # Compute pairwise IoU [M, N]
        iou_matrix = ops.box_iou(pred_boxes, target_boxes)

        matched_pred = []
        matched_target = []
        used_targets = set()

        # Greedy matching: for each prediction, find best unmatched ground truth
        for m_idx in range(pred_boxes.size(0)):
            iou_row = iou_matrix[m_idx]  # [N]

            # Find best match among unused ground truths
            for n_idx in torch.argsort(iou_row, descending=True):
                n_idx = n_idx.item()
                if n_idx not in used_targets and iou_row[n_idx] > self.iou_threshold:
                    matched_pred.append(pred_boxes[m_idx])
                    matched_target.append(target_boxes[n_idx])
                    used_targets.add(n_idx)
                    break

        if not matched_pred:
            # No matches found, return empty tensors
            return pred_boxes[:0], target_boxes[:0]

        return torch.stack(matched_pred), torch.stack(matched_target)

    def _compute_single_batch_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute GIoU loss for a single image [M, 4] and [N, 4]."""
        if target_boxes.numel() == 0 or pred_boxes.numel() == 0:
            return pred_boxes.new_tensor(0.0)

        # Match boxes if dimensions don't align
        if pred_boxes.size(0) != target_boxes.size(0):
            pred_boxes, target_boxes = self._match_boxes_greedy(pred_boxes, target_boxes)

        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.new_tensor(0.0)

        return ops.generalized_box_iou_loss(pred_boxes, target_boxes, reduction=self.reduction)

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Compute GIoU loss for single or batched inputs.

        Args:
            pred_boxes: [M, 4] or [B, M, 4] in xyxy format
            target_boxes: [N, 4] or [B, N, 4] in xyxy format

        Returns:
            scalar loss or [B] per-batch losses
        """
        # Handle batched inputs [B, M, 4] and [B, N, 4]
        if pred_boxes.dim() == 3 and target_boxes.dim() == 3:
            batch_size = pred_boxes.size(0)
            batch_losses = []

            for b in range(batch_size):
                loss_b = self._compute_single_batch_loss(
                    pred_boxes[b],
                    target_boxes[b]
                )
                batch_losses.append(loss_b.mean())

            # Stack batch losses and apply reduction
            batch_losses = torch.stack(batch_losses)
            if self.reduction == "mean":
                return batch_losses.mean()
            elif self.reduction == "sum":
                return batch_losses.sum()

            return batch_losses

        # Handle single image inputs [M, 4] and [N, 4]
        elif pred_boxes.dim() == 2 and target_boxes.dim() == 2:
            loss = self._compute_single_batch_loss(pred_boxes, target_boxes)[None]
            return loss
        else:
            raise ValueError(
                f"pred_boxes and target_boxes must have dims 2 (single) or 3 (batched). "
                f"Got pred_boxes.dim()={pred_boxes.dim()}, target_boxes.dim()={target_boxes.dim()}"
            )

    def compute(self):
        """Placeholder for compatibility."""
        pass


class COCOBBoxSegmentationDataset(Dataset):
    """COCO test dataset that turns bounding boxes into segmentation masks.

    Returns ``(image_tensor, mask_tensor, target_dict)`` where ``mask_tensor`` is
    an HxW class map built by rasterizing each bbox as a filled rectangle using
    its class id (optionally remapped via ``class_map``).

    Intended for evaluation/inference-only workflows where models predict masks
    from box-derived ground truth.

    Args:
        images_dir: Directory containing images.
        annotations_file: COCO annotations JSON (use the *val/test* split).
        image_transform: Optional transform applied to the PIL image.
        mask_transform: Optional transform applied to the mask (e.g., Resize
                        with nearest interpolation). If None, mask is converted
                        to a LongTensor via ``ToTensor``.
        class_map: Optional dict mapping category_id -> contiguous class id.
    """

    def __init__(
        self,
        images_dir,
        annotations_file,
        image_transform=None,
        mask_transform=None,
        class_map=None,
        max_samples=None,
    ):
        super().__init__()
        try:
            from pycocotools.coco import COCO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "pycocotools is required for COCOBBoxSegmentationDataset. Install with: pip install pycocotools"
            ) from exc

        self.images_dir = Path(images_dir)
        self.coco = COCO(annotations_file)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.class_map = class_map or {}
        subset_limit = max_samples if max_samples is not None else SUBSET
        self.image_ids = list(self.coco.imgs.keys())[:subset_limit]

    def __len__(self):
        return len(self.image_ids)

    def _build_mask(self, anns, height, width):
        mask = torch.zeros((height, width), dtype=torch.int64)
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cls = self.class_map.get(ann["category_id"], ann["category_id"])
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = cls
        return mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = self.images_dir / img_info["file_name"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.image_transform:
            image_t = self.image_transform(image)
        else:
            image_t = transforms.ToTensor()(image)

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Build mask from bboxes
        h, w = img_info["height"], img_info["width"]
        mask = self._build_mask(anns, h, w)[None, ...]  # Add channel dim

        if self.mask_transform:
            transformed_gt = self.mask_transform(mask)
        else:
            # ToTensor would cast to float; keep long for class ids
            transformed_gt = mask

        # Scale boxes to match the transformed mask/image size
        out_h, out_w = transformed_gt.shape[-2:]
        scale_x = out_w / float(w)
        scale_y = out_h / float(h)
        boxes = torch.tensor([
                [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]]
                for ann in anns
        ], dtype=torch.float32) if anns else torch.zeros((0, 4), dtype=torch.float32)
        if boxes.numel():
            scales = torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=boxes.dtype)
            boxes = boxes * scales
        labels = torch.tensor([
                self.class_map.get(ann["category_id"], ann["category_id"])
                for ann in anns
        ], dtype=torch.int64) if anns else torch.zeros((0,), dtype=torch.int64)
        image_id = torch.tensor([image_id], dtype=torch.int64)
        # Outputs is stackable tensors only
        return image_t, image_id, transformed_gt, boxes, labels


# Set up logging
logging.basicConfig(level=logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "debug"

# Suppress PIL warnings and set to INFO level
logging.getLogger("PIL").setLevel(logging.INFO)


def train(loader, model, optimizer, criterion_mlt=None, device='cpu', train_loader_len=1, tqdm_display=False):
    """Iterate over the train loader to compute a detection loss on trainset.

    Note: This loop uses the model's high-level predict API (non-differentiable)
    to obtain boxes and compute a GIoU-based loss for monitoring. It does not
    perform optimization and serves primarily to iterate over the trainset.
    """
    losses = 0.0

    it = tqdm.tqdm(loader, desc="Training (iter)") if tqdm_display else loader
    with guard_training_context:
        for data in it:
            inputs = data[0].to(device)
            ids = data[1].to(device)
            tbbxs = data[3].to(device)  # No batches

            # Infer
            optimizer.zero_grad()
            outputs = model.predict(inputs)[0]

            # Process predicted boxes
            pbbxs = torch.cat(
                [
                    outputs.boxes.cls[..., None],
                    outputs.boxes.xyxy
                ],
                dim=1
            )[None]

            if criterion_mlt is not None:
                loss_batch = criterion_mlt(
                    pbbxs[..., :-1],  # Remove class for loss
                    tbbxs,
                    batch_ids=ids,
                    preds=pbbxs,  # Full preds for data store logging, bboxs with cls
                )
                losses += torch.mean(loss_batch)

            # Training step
            losses.backward()
            optimizer.step()

    loss = float((losses / max(1, train_loader_len)).detach().cpu().item()) if criterion_mlt is not None else None
    return loss


def test(loader, model, criterion_mlt=None, metric_mlt=None, device='cpu', test_loader_len=1):
    """Full evaluation pass over the val loader."""
    losses = 0.0
    metric_mlt.reset() if metric_mlt is not None else None

    with guard_testing_context, torch.no_grad():
        for data in loader:
            inputs = data[0].to(device)
            ids = data[1].to(device)
            tseg = data[2].to(device)
            tbbxs = data[3].to(device)  # No batches

            # Predict boxes
            outputs = model.predict(inputs)[0]
            pbbxs = torch.cat([outputs.boxes.cls[..., None], outputs.boxes.xyxy], dim=1)[None]
            if criterion_mlt is not None:
                loss_batch = criterion_mlt(
                    pbbxs[..., :-1],  # Remove class for loss
                    tbbxs,
                    batch_ids=ids,
                    preds=pbbxs,  # Full preds for data store logging, bboxs with cls
                )
                losses += torch.mean(loss_batch)

            # ========== Build segmentation from boxes ==========
            # ===================================================
            # Build a class map from detected boxes (0 = background)
            H, W = tseg.shape[-2:]  # target mask spatial size
            class_map = torch.zeros((H, W), dtype=torch.int64, device=inputs.device)

            # Unpack detections; outputs.boxes.xyxy is [N,4], outputs.boxes.cls is [N]
            boxes_xyxy = outputs.boxes.xyxy
            classes = outputs.boxes.cls.int()
            conf = outputs.boxes.conf if hasattr(outputs.boxes, "conf") else None

            # Optional: filter by confidence
            keep = (conf >= 0.25) if conf is not None else torch.ones_like(classes, dtype=torch.bool)
            boxes_xyxy = boxes_xyxy[keep]
            classes = classes[keep]

            # Rasterize boxes into the class map
            for box, cls in zip(boxes_xyxy, classes):
                x1, y1, x2, y2 = box.round().int().tolist()
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(W, x2); y2 = min(H, y2)
                if x2 > x1 and y2 > y1:
                    class_map[y1:y2, x1:x2] = cls + 1  # offset by 1 to keep 0 as background

            # Now class_map is your predicted segmentation; compare/log against tseg
            pseg = class_map[None, ...]  # add channel if your metric expects [1, H, W]
            metric_mlt.update(pseg, tseg) if metric_mlt is not None else None

    loss = float((losses / test_loader_len).detach().cpu().item()) if criterion_mlt is not None else None
    metric = float(metric_mlt.compute().detach().cpu().item() * 100.0) if metric_mlt is not None else None
    return loss, metric


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "coco_detection")
    parameters.setdefault("device", "auto")
    parameters.setdefault("number_of_workers", 4)

    # Parameters
    exp_name = parameters.get("experiment_name", "coco_detection")

    # --- 2) Device selection ---
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = parameters["device"]

    # --- 3) Logging directory ---
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    log_dir = parameters["root_log_dir"]
    verbose = parameters.get("verbose", True)
    tqdm_display = parameters.get("tqdm_display", False)

    # --- 4) Register logger + hyperparameters ---
    logger = Logger()
    wl.watch_or_edit(logger, flag="logger", log_dir=log_dir)

    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        defaults=parameters,
        poll_interval=1.0,
    )

    # --- 5) Data ---
    img_size = parameters.get("img_size", 128)

    # Create transforms to resize images and masks to match model input shape
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    data_cfg = parameters.get("data", {})
    test_cfg = data_cfg.get("test_loader", {})
    _test_dataset = COCOBBoxSegmentationDataset(
        test_cfg["images_dir"],
        test_cfg["annotations_file"],
        image_transform=image_transform,
        mask_transform=mask_transform,
        class_map=None,
        max_samples=SUBSET // 2,
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 2),
        shuffle=test_cfg.get("shuffle", False),
        compute_hash=True,
        is_training=False
    )

    # --- Define criterion and metric ---
    train_criterion = wl.watch_or_edit(
        GIoULoss(reduction=None),
        flag="criterion",
        name="train_criterion/GIoU",
    )
    test_criterion = wl.watch_or_edit(
        GIoULoss(reduction=None),
        flag="criterion",
        name="test_criterion/GIoU",
    )

    # --- 6) Model, optimizer, losses, metric ---
    model = Yolov11(img_size=parameters.get("img_size", 128)).to(device)
    model = wl.watch_or_edit(
        model,
        flag="model",
        device=device,
        use_onnx=True  # Torch fx doesn't support Ultralytics models well yet. Use ONNX instead for dep. generation.
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parameters.get("learning_rate", 1e-3),
        weight_decay=parameters.get("weight_decay", 1e-4),
    )
    optimizer = wl.watch_or_edit(
        optimizer,
        flag="optimizer",
    )
    
    # --- Compute class weights to handle class imbalance ---
    print("\n" + "=" * 60)
    print("Computing class weights to address class imbalance...")
    print("=" * 60)

    # --- 7) Start WeightsLab services ---
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", True),
    )

    print("=" * 60)
    print("🚀 STARTING COCO DETECTION TRAIN/TEST")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    # ==============================
    # 7. Optional Trainset iteration
    train_cfg = data_cfg.get("train_loader", {})
    train_loss = None
    train_loader = test_loader
    train_loader_len = len(train_loader)
    train_loss = train(
        train_loader,
        model,
        optimizer=optimizer,
        criterion_mlt=train_criterion,
        device=device,
        train_loader_len=train_loader_len,
        tqdm_display=tqdm_display,
    )
    if train_loss is not None:
        print(f"Train loss (GIoU-based, no backprop): {train_loss:.4f}")

    # ===============
    # 8. Testing Loop
    test_loader_len = len(test_loader)  # Store length before wrapping with tqdm
    test_loader = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display else test_loader
    test_loss, test_metric = None, None
    start_time = time.time()
    test_loss, test_metric = test(test_loader, model, test_criterion, device=device, test_loader_len=test_loader_len)
    print("\n" + "=" * 60)
    print(f"✅ Testing completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
