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
        image_channels: int,
        label_channels: int,
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
        assert encoder_weights.lower() in ["none", "vae", "imagenet"]

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Hydra does not support 'null' to None conversion in sweep
        if encoder_weights.lower() == "none":
            encoder_weights = None

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
            and encoder_weights.lower() == "vae"
        ):
            # Load the vae state-dict
            # TODO dynamically determine the correct one based on encoder-name
            # eg: {vae}-{\beta}-{encoder-name}.pt
            encoder_state_dict = "models/pretrained-b10-vae.pt"

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
        input = self._normalize(input)
        x = self.model(input)
        return {"out": x}
