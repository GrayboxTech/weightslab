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


@wl.signal(name="detection/dice_score", compute_every_n_steps=10)
def detection_dice_signal(pred, batch, conf: float = 0.25, iou_thres: float = 0.5, **kwargs):
    """
    Compute Dice score (F1 score) for detection predictions.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|) = 2 * IoU / (1 + IoU)

    This metric combines precision and recall for bounding box predictions.
    Computed as a function of the IoU between predicted and ground truth boxes.

    Args:
        pred: Model predictions
        batch: Batch dict with 'img', 'batch_idx', 'bboxes'
        conf: Confidence threshold for NMS (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.5)
        **kwargs: Additional arguments from WeightsLab

    Returns:
        Mean Dice score (F1) value for logging
    """
    from ultralytics.utils.nms import box_iou
    from ultralytics.utils.ops import xywh2xyxy

    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    elif isinstance(pred, dict) and 'boxes' in pred and 'scores' in pred:
        pred = th.cat([pred['boxes'], pred['scores']], dim=1)

    img, batch_idx, boxes_norm = batch['img'], batch['batch_idx'], batch['bboxes']

    img_h, img_w = img.shape[-2:]
    preds_nms, _ = _decode_predictions(pred, img_h, img_w, conf, iou_thres)

    # Convert GT boxes to xyxy
    scale = th.tensor([img_w, img_h, img_w, img_h], dtype=boxes_norm.dtype)
    gt_xyxy = xywh2xyxy(boxes_norm.detach().cpu()) * scale

    bs = img.shape[0]
    dice_scores = th.full((bs,), float("nan"))

    for i in range(bs):
        gi = (batch_idx == i).nonzero(as_tuple=True)[0]
        pred_boxes = preds_nms[i][:, :4]

        if pred_boxes.shape[0] > 0 and gi.numel() > 0:
            # Compute IoU matrix between GT and predictions
            iou_matrix = box_iou(gt_xyxy[gi], pred_boxes)
            # Get max IoU for each GT box
            max_ious = iou_matrix.max(dim=1).values

            if max_ious.numel() > 0:
                # Convert IoU to Dice score: Dice = 2*IoU / (1 + IoU)
                # This is the F1 score based on IoU
                dice = 2.0 * max_ious / (1.0 + max_ious)
                dice_scores[i] = dice.mean()

    dice_scores = th.nan_to_num(dice_scores, nan=0.0)

    return dice_scores.mean().item() if dice_scores.numel() > 0 else 0.0


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
        Compute actual per-sample detection loss by computing loss for each sample separately.

        Args:
            pred: Model predictions
            batch: Ground truth targets dict with 'bboxes', 'cls', 'batch_idx', etc.
            **kwargs: Additional arguments from weightslab

        Returns:
            Tensor of per-sample losses [num_unique_samples]
        """
        # Get batch indices
        batch_idx = batch.get('batch_idx', None)

        if batch_idx is None or batch_idx.numel() == 0:
            # No batch indices, compute once
            loss_output = self.base_loss(pred, batch)
            loss = loss_output[0] if isinstance(loss_output, (tuple, list)) else loss_output
            # v8DetectionLoss returns the 3-vector [box, cls, dfl]. Select this
            # criterion's component, or sum if loss_type was not set.
            if loss.numel() > 1:
                loss = loss[self.loss_type] if self.loss_type is not None else loss.sum()
            return loss.unsqueeze(0)

        # Get the actual batch size from predictions (real batch size)
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

        # Prepare batch_idx: flatten, convert to long
        batch_idx_flat = batch_idx.flatten().long()

        # Build per-sample losses in a list to maintain gradient flow
        sample_losses_list = []

        # Split batch by sample and compute loss for each
        for sample_idx in range(pred_batch_size):
            # Get mask for boxes belonging to this sample
            sample_box_mask = batch_idx_flat == sample_idx

            if not sample_box_mask.any():
                # No ground truth for this sample, create zero loss as scalar
                sample_losses_list.append(th.tensor(0.0, device=device, dtype=th.float32))
                continue

            # Create batch dict for this sample's boxes
            sample_batch = {}
            for key, value in batch.items():
                if isinstance(value, th.Tensor) and value.shape[0] == len(batch_idx_flat):
                    # This tensor has one entry per box, filter by mask
                    if key == 'batch_idx':
                        # sample_pred is sliced to batch dim 1, so all boxes here belong to image 0
                        sample_batch[key] = th.zeros_like(value[sample_box_mask])
                    else:
                        sample_batch[key] = value[sample_box_mask]
                else:
                    sample_batch[key] = value

            # Extract predictions for this sample
            if isinstance(pred, dict):
                sample_pred = {k: v[sample_idx:sample_idx+1] if isinstance(v, th.Tensor) else v for k, v in pred.items()}
            elif isinstance(pred, (tuple, list)):
                sample_pred = tuple(p[sample_idx:sample_idx+1] if isinstance(p, th.Tensor) else p for p in pred)
            else:
                sample_pred = pred[sample_idx:sample_idx+1]

            # Compute loss for this sample
            try:
                sample_loss_output = self.base_loss(sample_pred, sample_batch)
                if isinstance(sample_loss_output, (tuple, list)):
                    sample_loss = sample_loss_output[0]
                else:
                    sample_loss = sample_loss_output
                # v8DetectionLoss returns the 3-vector [box, cls, dfl]. Pick this
                # criterion's component (0=box, 1=cls, 2=dfl); .mean() would
                # produce (box+cls+dfl)/3 for ALL three criterions, inflating
                # the effective gradient 3× when their outputs are summed.
                if sample_loss.numel() > 1:
                    if self.loss_type is not None:
                        sample_loss = sample_loss[self.loss_type]
                    else:
                        sample_loss = sample_loss.sum()
                elif sample_loss.ndim > 0:
                    sample_loss = sample_loss.squeeze()
                sample_losses_list.append(sample_loss)
            except Exception:
                sample_losses_list.append(th.tensor(0.0, device=device, dtype=th.float32))

        # Stack losses - preserves gradient flow
        per_sample_losses = th.stack(sample_losses_list)

        return per_sample_losses


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
