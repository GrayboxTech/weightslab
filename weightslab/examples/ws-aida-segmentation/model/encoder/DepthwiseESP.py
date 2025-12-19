import torch
import torch.nn as nn

from model.common import BatchnormRelu, DepthwiseSeparableConv


class DepthwiseESP(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = max(int(nOut / 5), 1)
        n1 = max(nOut - 4 * n, 1)
        self.c1 = DepthwiseSeparableConv(nIn, n, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BatchnormRelu(nOut)
        self.add = add

    def forward(self, input_tensor):
        """
        :param input_tensor: input feature map
        :return: transformed feature map
        """
        # reduce
        output1 = self.c1(input_tensor)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input_tensor + combine
        output = self.bn(combine)
        return output
