from typing import Any, List, Mapping, Optional, Union
import torch
import torchseg
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
        encoder_state_dict: Optional[Union[str, dict]] = None,
        activation=nn.Identity(),
    ):
        super().__init__()
        assert (
            image_channels == 3
        ), "Currently only RGB images are supported (due to pretrained weights)"

        # Hardcode the imagenet normalization function
        # I see no value in making this adaptable
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.unet = torchseg.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            classes=label_channels,
            activation=activation,
        )

        if encoder_state_dict is not None:
            if isinstance(encoder_state_dict, str):
                encoder_state_dict = torch.load(encoder_state_dict)
            self._load_encoder(encoder_state_dict)

    def _load_encoder(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = False,
        assign: bool = False,
    ):
        state_dict = {
            k: v for k, v in state_dict.items() if "encoder." in k
        }
        prefix = "encoder."
        encoder_state_dict = {
            k.lstrip(prefix): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        return self.unet.encoder.load_state_dict(
            encoder_state_dict, strict, assign
        )

    def forward(self, input):
        input = self._normalize(input)
        x = self.unet(input)
        return {"out": x}
