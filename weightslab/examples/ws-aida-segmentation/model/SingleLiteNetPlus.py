import torch.nn as nn

from .caam import CAAM
from .common import ConvBatchnormRelu
from .decoder import UpConvBlock
from .encoder.Encoder import Encoder


class SingleLiteNetPlus(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, encoder_hp, caam_hp, decoder_hp):

        super().__init__()
        self.encoder = Encoder(
            in_channels=encoder_hp.in_channels,
            level_0_ch=encoder_hp.level_0_ch,
            level_1_ch=encoder_hp.level_1_ch,
            level_2_ch=encoder_hp.level_2_ch,
            level_3_ch=encoder_hp.level_3_ch,
            level_4_ch=encoder_hp.level_4_ch,
            out_channels=encoder_hp.out_channels,
            p=encoder_hp.p,
            q=encoder_hp.q,
        )

        self.caam = CAAM(
            feat_in=caam_hp.in_channels,
            num_classes=caam_hp.num_classes,
            bin_size=(2, 4),
            norm_layer=nn.BatchNorm2d,
        )
        self.conv_caam = ConvBatchnormRelu(caam_hp.in_channels, caam_hp.out_channels)

        self.up_1 = UpConvBlock(
            decoder_hp.in_channels, decoder_hp.level_0_ch
        )  # out: Hx4, Wx4
        self.up_2 = UpConvBlock(
            decoder_hp.level_0_ch, decoder_hp.level_1_ch
        )  # out: Hx2, Wx2

        self.out = UpConvBlock(
            in_channels=decoder_hp.level_1_ch,
            out_channels=decoder_hp.out_channels,
            last=True,
        )

    def forward(self, input_tensor):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        out_encoder, inp1, inp2 = self.encoder(input_tensor)

        out_caam = self.caam(out_encoder)
        out_caam = self.conv_caam(out_caam)

        out = self.up_1(out_caam, inp2)
        out = self.up_2(out, inp1)
        out = self.out(out)

        return out
