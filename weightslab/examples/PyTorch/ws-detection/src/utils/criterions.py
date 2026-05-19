import torch as th
import torch.nn as nn

from ultralytics.utils.loss import v8DetectionLoss as DetectionLoss
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.tal import make_anchors, dist2bbox
from ultralytics.utils.nms import box_iou
from ultralytics.utils.ops import xywh2xyxy

def _decode_predictions(pred, img_h, img_w, conf=0.25, iou_thres=0.5):
    """Decode raw model preds [batch, 64+nc, 8400] → per-sample (boxes, [score, cls]) after NMS."""


    pred = pred.detach().cpu()
    pred = pred.permute(0, 2, 1)  # [batch, 8400, 64+nc]
    pred_boxes = pred[..., :64]   # DFL distributions
    pred_cls = pred[..., 64:]     # class logits

    # make_anchors returns grid-coord points + stride_tensor to scale to pixels.
    # Dropping stride_tensor → boxes off by 8/16/32×.
    strides = [8, 16, 32]
    anchors, stride_tensor = make_anchors(
        [th.zeros(1, 1, img_h // s, img_w // s) for s in strides],
        strides, grid_cell_offset=0.5,
    )

    # Emit xywh (not xyxy): non_max_suppression unconditionally xywh2xyxys its input.
    pred_boxes = pred_boxes.view(*pred_boxes.shape[:2], 4, 16)
    pred_boxes = pred_boxes.softmax(-1) @ th.arange(16).float()
    pred_boxes = dist2bbox(pred_boxes, anchors.unsqueeze(0), xywh=True) * stride_tensor

    # NMS expects [B, 4+nc, N].
    pred_scores = pred_cls.sigmoid()
    pred_combined = th.cat([pred_boxes, pred_scores], dim=-1).permute(0, 2, 1)
    preds_nms = non_max_suppression(
        pred_combined, conf_thres=conf, iou_thres=iou_thres, max_det=300)

    # Per sample: (N_i, 6) = [x1, y1, x2, y2, conf, cls] in pixels.
    return [i[:, :-2] if i.ndim > 1 else i for i in preds_nms], [i[:, -2:] if i.ndim > 1 else i for i in preds_nms]


class PerSampleIoU(nn.Module):
    """Per-sample IoU between GT and NMS-filtered predictions."""

    def __init__(self, conf: float = 0.25, iou_thres: float = 0.5):
        super().__init__()
        self.conf = conf
        self.iou_thres = iou_thres

    def forward(self, pred, batch):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        elif isinstance(pred, dict) and 'boxes' in pred and 'scores' in pred:
            pred = th.cat([pred['boxes'], pred['scores']], dim=1)

        img, batch_idx, boxes_norm = batch['img'], batch['batch_idx'], batch['bboxes']
        img_h, img_w = img.shape[-2:]
        preds_nms, _ = _decode_predictions(pred, img_h, img_w, self.conf, self.iou_thres)

        scale = th.tensor([img_w, img_h, img_w, img_h], dtype=boxes_norm.dtype)
        gt_xyxy = xywh2xyxy(boxes_norm.detach().cpu()) * scale

        out = th.full((img.shape[0],), float("nan"))
        for i in range(img.shape[0]):
            gi = (batch_idx == i).nonzero(as_tuple=True)[0]
            if preds_nms[i].shape[0] > 0 and gi.numel() > 0:
                out[i] = box_iou(gt_xyxy[gi], preds_nms[i][:, :4]).max(dim=1).values.mean()
        return th.nan_to_num(out, nan=0.0)


class PerSampleDetectionLoss(nn.Module):
    """Per-sample wrapper around v8DetectionLoss; loss_type picks [0=box, 1=cls, 2=dfl] or None=sum."""

    def __init__(self, model, loss_type=None):
        super().__init__()
        
        self.base_loss = DetectionLoss(model)
        self.loss_type = loss_type
        self.model = model

    def _pick(self, loss):
        # v8DetectionLoss returns [box, cls, dfl]. Pick this criterion's component; .mean() would
        # give (box+cls+dfl)/3 across all three → 3× gradient inflation when summed.
        if loss.numel() > 1:
            return loss[self.loss_type] if self.loss_type is not None else loss.sum()
        return loss.squeeze() if loss.ndim > 0 else loss

    def forward(self, pred, batch):
        batch_idx = batch.get('batch_idx', None)

        if batch_idx is None or batch_idx.numel() == 0:
            out = self.base_loss(pred, batch)
            loss = out[0] if isinstance(out, (tuple, list)) else out
            return self._pick(loss).unsqueeze(0)

        # pred can be dict / tuple / tensor — derive batch size and device once.
        if isinstance(pred, dict):
            first = next(iter(pred.values()))
            pred_bs, device = first.shape[0], first.device
        elif isinstance(pred, (tuple, list)):
            pred_bs, device = pred[0].shape[0], pred[0].device
        else:
            pred_bs, device = pred.shape[0], pred.device

        batch_idx_flat = batch_idx.flatten().long()
        sample_losses = []

        for s in range(pred_bs):
            box_mask = batch_idx_flat == s

            # sample_pred is sliced to batch dim 1 → all boxes belong to image 0.
            sample_batch = {
                k: (th.zeros_like(v[box_mask]) if k == 'batch_idx' else v[box_mask])
                if isinstance(v, th.Tensor) and v.shape[0] == len(batch_idx_flat) else v
                for k, v in batch.items()
            }

            if isinstance(pred, dict):
                sample_pred = {k: v[s:s+1] if isinstance(v, th.Tensor) else v for k, v in pred.items()}
            elif isinstance(pred, (tuple, list)):
                sample_pred = tuple(p[s:s+1] if isinstance(p, th.Tensor) else p for p in pred)
            else:
                sample_pred = pred[s:s+1]

            out = self.base_loss(sample_pred, sample_batch)
            sample_loss = out[0] if isinstance(out, (tuple, list)) else out
            sample_losses.append(self._pick(sample_loss))

        return th.stack(sample_losses)
