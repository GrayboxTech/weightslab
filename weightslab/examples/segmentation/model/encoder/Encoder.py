import torch
import torch.nn as nn

from model.common import ConvBatchnormRelu

from .AvgDownsampler import AvgDownsampler
from .DepthwiseESP import DepthwiseESP
from .StrideESP import StrideESP


class Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(
        self,
        in_channels: int,
        level_0_ch: int,
        level_1_ch: int,
        level_2_ch: int,
        level_3_ch: int,
        level_4_ch: int,
        out_channels: int,
        p: int,
        q: int,
    ):
        super().__init__()
        self.level1 = ConvBatchnormRelu(in_channels, level_0_ch, stride=2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)

        self.b1 = ConvBatchnormRelu(level_0_ch + in_channels, level_1_ch)
        self.level2_0 = StrideESP(level_1_ch, level_2_ch)

        self.level2 = nn.ModuleList()
        for _ in range(0, p):
            self.level2.append(DepthwiseESP(level_2_ch, level_2_ch))
        self.b2 = ConvBatchnormRelu(level_3_ch + in_channels, level_3_ch + in_channels)

        self.level3_0 = StrideESP(level_3_ch + in_channels, level_3_ch)
        self.level3 = nn.ModuleList()
        for _ in range(0, q):
            self.level3.append(DepthwiseESP(level_3_ch, level_3_ch))
        self.b3 = ConvBatchnormRelu(level_4_ch, out_channels)

    def forward(self, input_tensor):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input_tensor)
        inp1 = self.sample1(input_tensor)
        inp2 = self.sample2(input_tensor)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)

        return out_encoder, inp1, inp2
