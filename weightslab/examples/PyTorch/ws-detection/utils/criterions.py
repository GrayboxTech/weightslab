import torch as th
import torch.nn as nn
import weightslab as wl

from copy import deepcopy
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import RANK, colorstr


def _decode_predictions(pred, img_h, img_w, conf=0.25, iou_thres=0.5):
    """Decode model predictions to NMS-filtered bounding boxes.

    Args:
        pred: Raw model predictions [batch, 64+nc, 8400]
        img_h: Image height
        img_w: Image width
        conf: Confidence threshold for NMS
        iou_thres: IoU threshold for NMS

    Returns:
        preds_nms: List of NMS-filtered predictions per batch item
        gt_xyxy: Ground truth boxes in xyxy format (if applicable)
    """
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.tal import make_anchors, dist2bbox

    pred = pred.detach().cpu()
    pred = pred.permute(0, 2, 1)  # [batch, 8400, 64+nc]

    nc = pred.shape[-1] - 64
    pred_boxes = pred[..., :64]  # DFL distributions
    pred_cls = pred[..., 64:]    # class logits

    # Generate anchors
    strides = [8, 16, 32]
    anchors, _ = make_anchors(
        [th.zeros(1, 1, img_h // s, img_w // s) for s in strides],
        strides, grid_cell_offset=0.5
    )

    # Decode DFL to bbox distances
    pred_boxes = pred_boxes.view(*pred_boxes.shape[:2], 4, 16)
    pred_boxes = pred_boxes.softmax(-1) @ th.arange(16).float()
    pred_boxes = dist2bbox(pred_boxes, anchors.unsqueeze(0), xywh=False)  # xyxy in pixels

    # Apply NMS
    pred_scores = pred_cls.sigmoid()
    pred_combined = th.cat([pred_boxes, pred_scores], dim=-1)
    preds_nms = non_max_suppression(pred_combined, conf_thres=conf, iou_thres=iou_thres, max_det=300)

    return preds_nms


class PerSampleIoU(nn.Module):
    """IoU metric as a criterion class for per-sample tracking.

    Computes IoU between predictions and ground truth boxes with NMS and decoding.
    Integrates with weightslab for per-sample loss tracking.
    """

    def __init__(self, conf: float = 0.25, iou_thres: float = 0.5):
        super().__init__()
        self.conf = conf
        self.iou_thres = iou_thres

    def forward(self, pred, batch):
        """
        Compute per-sample IoU.

        Args:
            pred: Model predictions
            img: Input images
            batch_idx: Batch indices for boxes
            boxes_norm: Ground truth boxes in normalized xywh format
            batch_ids: WeightsLab batch IDs for per-sample tracking
            **kwargs: Additional arguments (preds, labels) from weightslab

        Returns:
            Tensor of per-sample IoU values
        """
        from ultralytics.utils.nms import box_iou
        from ultralytics.utils.ops import xywh2xyxy

        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        elif isinstance(pred, dict) and 'boxes' in pred and 'scores' in pred:
            pred = th.cat([pred['boxes'], pred['scores']], dim=1)

        img, batch_idx, boxes_norm = batch['img'], batch['batch_idx'], batch['bboxes']

        img_h, img_w = img.shape[-2:]
        preds_nms = _decode_predictions(pred, img_h, img_w, self.conf, self.iou_thres)

        # Convert GT boxes to xyxy
        scale = th.tensor([img_w, img_h, img_w, img_h], dtype=boxes_norm.dtype)
        gt_xyxy = xywh2xyxy(boxes_norm.detach().cpu()) * scale

        bs = img.shape[0]
        out_ious = th.full((bs,), float("nan"))

        for i in range(bs):
            gi = (batch_idx == i).nonzero(as_tuple=True)[0]
            if preds_nms[i].shape[0] > 0 and gi.numel() > 0:
                m = box_iou(gt_xyxy[gi], preds_nms[i][:, :4])
                out_ious[i] = m.max(dim=1).values.mean()

        out_ious = th.nan_to_num(out_ious, nan=0.0)

        return out_ious


class PerSampleDetectionLoss(nn.Module):
    """Per-sample detection loss wrapping Ultralytics DetectionLoss.

    Returns per-sample losses to enable fine-grained tracking via weightslab.
    """

    def __init__(self, model, reduction: str = "none", loss_type=None):
        super().__init__()
        from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss
        self.base_loss = DetectionLoss(model)
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, batch):
        """
        Compute per-sample detection loss.

        Args:
            pred: Model predictions
            targets: Ground truth targets dict with 'bboxes', 'cls', 'batch_idx'
            batch_ids: WeightsLab batch IDs for per-sample tracking
            **kwargs: Additional arguments from weightslab

        Returns:
            Tensor of per-sample losses [num_samples, num_loss_components]
        """
        loss = self.base_loss(pred, batch)
        if self.loss_type is None:
            return loss
        else:
            loss = loss[0]
        if loss.ndim == 1:
            loss = loss[None]
        if isinstance(loss, th.Tensor) and loss.dim() >= 1:
            per_sample_loss = loss
        else:
            per_sample_loss = th.tensor([loss.item()] * len(batch.get('batch_idx', [1])))

        return per_sample_loss[:, self.loss_type] if self.loss_type is not None and isinstance(self.loss_type, int) else per_sample_loss


class GIoULoss(nn.Module):
    """Generalized IoU loss with box assignment and batching support.

    Handles mismatched numbers of predicted vs ground truth boxes by:
    1. Computing pairwise IoU between all predicted and ground truth boxes
    2. Greedily matching boxes with highest IoU
    3. Computing GIoU loss only on matched pairs

    Supports both single-image and batched inputs:
    - Single image: pred_boxes [M, 4], target_boxes [N, 4] -> scalar loss
    - Batched: pred_boxes [B, M, 4], target_boxes [B, N, 4] -> per-batch losses [B]
    """

    def __init__(self, reduction: str = "mean", iou_threshold: float = 0.1):
        super().__init__()
        self.reduction = 'none' if reduction not in ['none', 'mean', 'sum'] else reduction
        self.iou_threshold = iou_threshold

    def _match_boxes_greedy(self, pred_boxes: th.Tensor, target_boxes: th.Tensor):
        """Greedy box matching using pairwise IoU (single image)."""
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes, target_boxes

        iou_matrix = th.ops.box_iou(pred_boxes, target_boxes)
        matched_pred = []
        matched_target = []
        used_targets = set()

        for m_idx in range(pred_boxes.size(0)):
            iou_row = iou_matrix[m_idx]
            for n_idx in th.argsort(iou_row, descending=True):
                n_idx = n_idx.item()
                if n_idx not in used_targets and iou_row[n_idx] > self.iou_threshold:
                    matched_pred.append(pred_boxes[m_idx])
                    matched_target.append(target_boxes[n_idx])
                    used_targets.add(n_idx)
                    break

        if not matched_pred:
            return pred_boxes[:0], target_boxes[:0]

        return th.stack(matched_pred), th.stack(matched_target)

    def _compute_single_batch_loss(self, pred_boxes: th.Tensor, target_boxes: th.Tensor) -> th.Tensor:
        """Compute GIoU loss for a single image [M, 4] and [N, 4]."""
        if target_boxes.numel() == 0 or pred_boxes.numel() == 0:
            return pred_boxes.new_tensor(0.0)

        if pred_boxes.size(0) != target_boxes.size(0):
            pred_boxes, target_boxes = self._match_boxes_greedy(pred_boxes, target_boxes)

        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.new_tensor(0.0)

        return th.ops.generalized_box_iou_loss(pred_boxes, target_boxes, reduction=self.reduction)

    def forward(self, pred_boxes: th.Tensor, target_boxes: th.Tensor) -> th.Tensor:
        """
        Compute GIoU loss for single or batched inputs.

        Args:
            pred_boxes: [M, 4] or [B, M, 4] in xyxy format
            target_boxes: [N, 4] or [B, N, 4] in xyxy format
            batch_ids: WeightsLab batch IDs for per-sample tracking
            **kwargs: Additional arguments from weightslab

        Returns:
            scalar loss or [B] per-batch losses
        """
        if pred_boxes.dim() == 3 and target_boxes.dim() == 3:
            batch_size = pred_boxes.size(0)
            batch_losses = []

            for b in range(batch_size):
                loss_b = self._compute_single_batch_loss(pred_boxes[b], target_boxes[b])
                batch_losses.append(loss_b.mean())
            batch_losses = th.stack(batch_losses)

            if self.reduction == "mean":
                return batch_losses.mean()
            elif self.reduction == "sum":
                return batch_losses.sum()
            return batch_losses

        elif pred_boxes.dim() == 2 and target_boxes.dim() == 2:
            loss = self._compute_single_batch_loss(pred_boxes, target_boxes)
            return loss[None]
        else:
            raise ValueError(
                f"pred_boxes and target_boxes must have dims 2 or 3. "
                f"Got pred_boxes.dim()={pred_boxes.dim()}, target_boxes.dim()={target_boxes.dim()}"
            )
