import torch.nn as nn


class DilatedConv(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=d,
            groups=groups,
        )

    def forward(self, input_tensor):
        """
        :param input_tensor: input feature map
        :return: transformed feature map
        """
        output = self.conv(input_tensor)
        return output
