import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W) or (B, C, H, W) one-hot
        probs = F.softmax(logits, dim=1)

        if probs.shape != targets.shape:
            # convert target to one-hot
            num_classes = probs.shape[1]
            targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        edge_probs = self._gradient_magnitude(probs)
        edge_targets = self._gradient_magnitude(targets)
        return F.l1_loss(edge_probs, edge_targets)

    def _gradient_magnitude(self, img):
        dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])  # (B, C, H-1, W)
        dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])  # (B, C, H, W-1)

        # Crop to (B, C, H-1, W-1) to make both tensors match in shape
        dx = dx[:, :, :-1, :]  # (B, C, H-1, W-1)
        dy = dy[:, :, :, :-1]  # (B, C, H-1, W-1)

        return torch.sqrt(dx**2 + dy**2 + 1e-8)
