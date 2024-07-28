import os
from typing import List, Literal, Optional
import torchseg
from torch import nn
from torchvision.transforms import v2 as transforms

from . import utils


class TorchSegModel(nn.Module):
    """Wrapper around torchseg for better support in our framework"""

    ARCH_TO_INIT = {"unet": torchseg.Unet, "fpn": torchseg.FPN}

    def __init__(
        self,
        architecture: Literal["fpn", "unet"],
        label_channels: int,
        image_channels: int = 3,
        encoder_depth: int = 5,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        encoder_name="mobilenetv2_100",
        encoder_weights="imagenet",
        encoder_freeze: bool = False,
        encoder_state_dict: Optional[str] = None,
        activation=nn.Identity(),
    ):
        super().__init__()
        assert image_channels == 3
        encoder_weights = utils.parse_enc_weights(encoder_weights)

        if encoder_weights is None and encoder_freeze is True:
            raise RuntimeError(
                "Frozen encoder with random weights is just stupid"
            )

        # Torchseg determines to use pretrained weights based on
        # the following internal check:
        # 'encoder_weights is not None'
        # Thus both 'vae' and 'imagenet' will load the imagenet weights'
        self.model = self.ARCH_TO_INIT[architecture](
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            classes=label_channels,
            activation=activation,
        )

        if (
            encoder_state_dict is None
            and encoder_weights is not None
            and encoder_weights.startswith("vae")
        ):
            encoder_state_dict = utils.determine_state_dict_path(
                encoder_name, encoder_weights
            )

        if encoder_state_dict is not None:
            encoder_state_dict = utils.load_state_dict(
                encoder_state_dict
            )
            encoder_state_dict = utils.extract_encoder(
                encoder_state_dict
            )
            self.model.encoder.load_state_dict(encoder_state_dict)

        if encoder_freeze:
            self.model.encoder = utils.freeze_model(self.model.encoder)

        self.name = self.model.name

    def forward(self, input):
        x = self.model(input)
        return {"out": x}
