from typing import List
import torchseg as smp
from torch import nn
from torchvision.transforms import v2 as transforms


class UNet(nn.Module):
    """Standard UNet based on"""

    def __init__(
        self,
        image_channels: int,
        label_channels: int,
        encoder_depth: int = 5,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        encoder_name="mobilenetv2_100",
        encoder_weights="imagenet",
        activation=nn.Identity(),
    ):
        super().__init__()
        assert (
            image_channels == 3
        ), "Currently only RGB images are supported (due to pretrained weights)"

        # Hardcode the imagenet normalization function
        # I see no value in making this adaptable
        self._normalize = (
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        )

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            classes=label_channels,
            activation=activation,
        )

    def forward(self, input):
        input = self._normalize(input)
        x = self.unet(input)
        return {"out": x}
