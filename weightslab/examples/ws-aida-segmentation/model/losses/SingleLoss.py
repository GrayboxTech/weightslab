import torch.nn as nn

from .BoundaryLoss import BoundaryLoss
from .FocalLoss import FocalLossSeg
from .TotalVariationLoss import TotalVariationLoss
from .TverskyLoss import TverskyLoss


class SingleLoss(nn.Module):
    """
    This file defines a cross entropy loss for 2D images
    """

    def __init__(self, hyp):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        """
        super().__init__()

        alpha1, gamma1, alpha3, gamma3 = (
            hyp["alpha1"],
            hyp["gamma1"],
            hyp["alpha3"],
            hyp["gamma3"],
        )

        self.seg_tver = TverskyLoss(
            mode="multiclass",
            alpha=alpha1,
            beta=1 - alpha1,
            gamma=gamma1,
            from_logits=True,
        )
        self.seg_focal = FocalLossSeg(mode="multiclass", alpha=alpha3, gamma=gamma3)
        self.seg_total_variation = TotalVariationLoss()
        self.seg_boundary = BoundaryLoss()

    def forward(self, outputs, targets):
        seg = targets.to(outputs.device)

        tversky_loss = self.seg_tver(outputs, seg)
        focal_loss = self.seg_focal(outputs, seg)
        boundary_loss = self.seg_boundary(outputs, seg)
        total_variation_loss = self.seg_total_variation(outputs)

        loss = focal_loss + tversky_loss
        return (
            focal_loss.item(),
            tversky_loss.item(),
            total_variation_loss.item(),
            boundary_loss.item(),
            loss,
        )
