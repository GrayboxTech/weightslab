import torch.nn as nn


class BatchnormRelu(nn.Module):
    """
    This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.nOut = nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input_tensor):
        """
        :param input_tensor: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input_tensor)
        output = self.act(output)
        return output
