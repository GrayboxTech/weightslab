import torch as th
import torch.nn as nn
import weightslab as wl

from copy import deepcopy
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import RANK, colorstr


class AccumulatedConsecutiveAbsWeightDiff(nn.Module):
    """Σ|p_now - p_prev| across trainable params, refreshed each call.

    Diagnostic metric: if this stays at 0, the optimizer isn't actually stepping
    (e.g. param groups are empty because the model was WL-wrapped before
    build_optimizer). Call AFTER optimizer.step(). First call returns 0 because
    there's no prior snapshot to diff against.
    """

    def __init__(self, model: nn.Module, only_trainable: bool = True):
        super().__init__()
        self._params = [p for p in model.parameters() if (p.requires_grad or not only_trainable)]
        self._snapshot = None
        self._last_value = th.tensor(0.0)

    @th.no_grad()
    def forward(self) -> th.Tensor:
        if self._snapshot is None:
            self._snapshot = [p.detach().clone() for p in self._params]
            return self._last_value
        total = 0.0
        for p, prev in zip(self._params, self._snapshot):
            total += (p.detach() - prev).abs().sum().item()
        self._snapshot = [p.detach().clone() for p in self._params]
        self._last_value = th.tensor(float(total))
        return self._last_value


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

    pred_boxes = pred[..., :64]  # DFL distributions
    pred_cls = pred[..., 64:]    # class logits

    # make_anchors returns anchor points in feature-grid coords + a stride_tensor
    # that maps them to pixel coords. We need both — the previous version dropped
    # stride_tensor and produced boxes in anchor-grid units (off by 8/16/32×).
    strides = [8, 16, 32]
    anchors, stride_tensor = make_anchors(
        [th.zeros(1, 1, img_h // s, img_w // s) for s in strides],
        strides, grid_cell_offset=0.5
    )

    # Decode DFL to bbox distances, then scale by stride to pixel coords.
    # ultralytics' non_max_suppression unconditionally calls xywh2xyxy on its
    # input, so we MUST emit center-xywh here — emitting xyxy makes NMS
    # double-convert and silently drop boxes.
    pred_boxes = pred_boxes.view(*pred_boxes.shape[:2], 4, 16)
    pred_boxes = pred_boxes.softmax(-1) @ th.arange(16).float()
    pred_boxes = dist2bbox(pred_boxes, anchors.unsqueeze(0), xywh=True) * stride_tensor

    # NMS expects channels-second [B, 4+nc, N]; without the permute the
    # conf-threshold check reads the wrong axis and drops everything.
    pred_scores = pred_cls.sigmoid()
    pred_combined = th.cat([pred_boxes, pred_scores], dim=-1).permute(0, 2, 1)
    preds_nms = non_max_suppression(
        pred_combined, conf_thres=conf, iou_thres=iou_thres, max_det=300)

    # NMS output per sample: (N_i, 6) = [x1, y1, x2, y2, conf, cls] in pixels.
    return [i[:, :-2] if i.ndim > 1 else i for i in preds_nms], [i[:, -2:] if i.ndim > 1 else i for i in preds_nms]


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
        preds_nms, _ = _decode_predictions(pred, img_h, img_w, self.conf, self.iou_thres)

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

    Computes actual per-sample losses by computing loss for each sample separately.
    """

    def __init__(self, model, reduction: str = "none", loss_type=None):
        super().__init__()
        from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss
        self.base_loss = DetectionLoss(model)
        self.reduction = reduction
        self.loss_type = loss_type
        self.model = model

    def forward(self, pred, batch):
        """
        Compute per-sample detection loss by computing once on full batch, then splitting by sample.

        This avoids per-sample normalization issues (dividing by very small n_positive_anchors per sample).
        Loss is distributed proportionally to each sample's box count.

        Args:
            pred: Model predictions
            batch: Ground truth targets dict with 'bboxes', 'cls', 'batch_idx', etc.
            **kwargs: Additional arguments from weightslab

        Returns:
            Tensor of per-sample losses [num_unique_samples]
        """
        # Get batch indices
        batch_idx = batch.get('batch_idx', None)

        # Get the actual batch size from predictions
        if isinstance(pred, dict):
            first_key = next(iter(pred.keys()))
            pred_batch_size = pred[first_key].shape[0]
            device = pred[first_key].device
        elif isinstance(pred, (tuple, list)):
            pred_batch_size = pred[0].shape[0]
            device = pred[0].device
        else:
            pred_batch_size = pred.shape[0]
            device = pred.device

        # Compute loss on full batch
        loss_output = self.base_loss(pred, batch)
        if isinstance(loss_output, (tuple, list)):
            loss_components = loss_output[0]  # [box_loss, cls_loss, dfl_loss]
        else:
            loss_components = loss_output

        # Select the loss component (box=0, cls=1, dfl=2), or sum all if not specified
        if loss_components.numel() > 1:
            batch_loss = loss_components[self.loss_type] if self.loss_type is not None else loss_components.sum()
        else:
            batch_loss = loss_components.squeeze()

        # If no batch_idx or empty, return batch loss as single sample
        if batch_idx is None or batch_idx.numel() == 0:
            return batch_loss.unsqueeze(0)

        # Split loss by sample proportionally based on box count
        batch_idx_flat = batch_idx.flatten().long()
        total_boxes = batch_idx_flat.numel()

        per_sample_losses = []
        for sample_idx in range(pred_batch_size):
            sample_mask = batch_idx_flat == sample_idx
            n_boxes_in_sample = sample_mask.sum().item()

            if n_boxes_in_sample == 0:
                # No boxes for this sample - contribute zero loss
                per_sample_losses.append(th.tensor(0.0, device=device, dtype=batch_loss.dtype))
            else:
                # Distribute batch loss proportionally by box count
                # This preserves gradients and avoids normalization by very small numbers
                loss_weight = n_boxes_in_sample / total_boxes
                per_sample_losses.append(batch_loss * loss_weight)

        return th.stack(per_sample_losses)


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
