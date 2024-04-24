from typing import List
import segmentation_models_pytorch as smp
from torch import nn


class UNet(nn.Module):
    """Small wrapper around smp.UNet

    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        image_channels: int,
        label_channels: int,
        encoder_depth: int = 5,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        activation=None,
    ):
        super().__init__()
        assert (
            image_channels == 3
        ), "Currently only RGB images are supported (due to pretrained weights)"

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            classes=label_channels,
            activation=activation,
        )

    def forward(self, input):
        x = self.unet(input)
        return {"out": x}
