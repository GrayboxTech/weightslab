import torch
import torch.nn as nn
import torch.nn.functional as F


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss.
    Encourages spatial smoothness in the output image.
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the total variation loss.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Computed total variation loss.
        """
        x = F.softmax(x, dim=1)

        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return h_tv + w_tv
