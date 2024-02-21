import segmentation_models_pytorch as smp
from torch import nn

from models.duq import DUQHead


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert (
            in_channels == 3
        ), "Currently only RGB images are supported (due to pretrained weights)"

        decoder_channels = [256, 128, 64, 32, 16]

        self.unet = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_depth=5,
            encoder_weights="imagenet",
            decoder_channels=decoder_channels,
            classes=out_channels,
            activation=None,
        )

        self.unet.segmentation_head = DUQHead(
            in_channels=decoder_channels[-1],
            num_classes=out_channels,
        )

    def forward(self, img):
        x = self.unet(img)
        return x
