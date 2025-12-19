import torch.nn as nn


class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input_tensor):
        output = self.deconv(input_tensor)
        output = self.bn(output)
        output = self.act(output)
        return output
