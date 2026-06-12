# =============================================================================
# Small single-shot grid detector on a pretrained MobileNetV3-Small backbone
# =============================================================================
# A frozen/fine-tuned ImageNet-pretrained backbone extracts features; a tiny
# detection head turns them into an S x S grid where each cell predicts ONE box:
# (objectness, tx, ty, tw, th, class_logits...).
#
# Encoding (all coordinates normalized to the [0, 1] image frame):
#   * objectness = sigmoid(t_obj)          -> P(box present in this cell)
#   * cx = (col + sigmoid(tx)) / S         -> box center, x
#   * cy = (row + sigmoid(ty)) / S         -> box center, y
#   * w  = sigmoid(tw)                     -> box width  (fraction of image)
#   * h  = sigmoid(th)                     -> box height (fraction of image)
#   * class = softmax(class_logits)
#
# Raw forward output keeps logits (loss applies the activations); `decode`
# turns logits into xyxy boxes for metrics and UI rendering.
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def decode_grid(outputs, grid_size):
    """Decode raw grid logits -> per-cell xyxy boxes, objectness and class probs.

    Shared by the model (``SmallDetector.decode``) and the criterions so the
    encoding lives in exactly one place.

    Args:
        outputs:   [B, S, S, 5 + num_classes] raw logits.
        grid_size: S.

    Returns:
        boxes:     [B, S, S, 4]  xyxy in [0, 1]
        obj:       [B, S, S]     objectness probability
        cls_probs: [B, S, S, num_classes] class probabilities
    """
    B, S, _, _ = outputs.shape
    device = outputs.device

    obj = torch.sigmoid(outputs[..., 0])
    tx = torch.sigmoid(outputs[..., 1])
    ty = torch.sigmoid(outputs[..., 2])
    w = torch.sigmoid(outputs[..., 3])
    h = torch.sigmoid(outputs[..., 4])
    cls_probs = torch.softmax(outputs[..., 5:], dim=-1)

    # Cell-origin grid (col=x, row=y).
    cols = torch.arange(S, device=device).view(1, 1, S).expand(B, S, S)
    rows = torch.arange(S, device=device).view(1, S, 1).expand(B, S, S)

    cx = (cols + tx) / S
    cy = (rows + ty) / S
    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return boxes, obj, cls_probs


class SmallDetector(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        image_size=256,
        grid_size=8,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()
        # For WeightsLab
        self.task_type = "detection"
        self.num_classes = num_classes
        self.class_names = ["person"][:num_classes]
        self.grid_size = grid_size
        self.image_size = image_size
        self.input_shape = (1, in_channels, image_size, image_size)

        # Channels per cell: objectness(1) + box(4) + class logits(num_classes)
        self.preds_per_cell = 5 + num_classes

        # --- Pretrained backbone (ImageNet) ---
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.backbone = backbone.features  # [B, 576, H/32, W/32]
        backbone_out_ch = 576

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- Detection head (lightweight, trained from scratch) ---
        self.neck = nn.Sequential(
            nn.Conv2d(backbone_out_ch, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(128, self.preds_per_cell, 1)

    def train(self, mode=True):
        """Keep a frozen backbone in eval mode (so its BatchNorm stats stay fixed)."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        """Returns raw logits [B, S, S, 5 + num_classes]."""
        if self.freeze_backbone:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)

        feat = self.neck(feat)
        out = self.head(feat)            # [B, preds_per_cell, S', S']

        # Resize feature grid to the configured grid_size.
        if out.shape[-1] != self.grid_size or out.shape[-2] != self.grid_size:
            out = nn.functional.adaptive_avg_pool2d(out, self.grid_size)
        # [B, C, S, S] -> [B, S, S, C]
        return out.permute(0, 2, 3, 1).contiguous()

    def decode(self, outputs):
        """Decode raw logits -> per-cell xyxy boxes, objectness, class probs."""
        return decode_grid(outputs, self.grid_size)
