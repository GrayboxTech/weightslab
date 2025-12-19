import torch
import torch.nn as nn

from model.common import ConvBatchnormRelu, UpSimpleBlock


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False, kernel_size=3):
        super(UpConvBlock, self).__init__()
        self.last = last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        if not last:
            self.conv1 = ConvBatchnormRelu(
                out_channels + sub_dim, out_channels, kernel_size
            )
        self.conv2 = ConvBatchnormRelu(out_channels, out_channels)

    def forward(self, x, ori_img=None):
        x = self.up_conv(x)
        if not self.last:
            x = torch.cat([x, ori_img], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x
