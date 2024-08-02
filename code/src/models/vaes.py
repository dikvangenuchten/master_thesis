from typing import Callable, Dict, List, Optional, Union

from torch import nn
from torch.nn import functional as F

import torch
from torchseg.base import (
    SegmentationHead,
    initialization,
)
from torchseg.encoders import get_encoder

from models.variational_u_net import VariationalConv2d
from . import utils


def get_or_default(list_, idx, default=None):
    if idx < len(list_):
        return list_[idx]
    return default


class SkipLayer(nn.Module):
    SUPPORTED_TYPES = ["none", "skip", "proj", "var"]

    def __init__(self, type, channels):
        super().__init__()
        type = type.lower()
        assert (
            type in self.SUPPORTED_TYPES
        ), f"{type} must be one of: {self.SUPPORTED_TYPES}"

        def wrapper(fn):
            """Wraps a function into our framework for vae"""

            def _inner(x):
                return {"out": fn(x)}

            return _inner

        if type == "none":
            self._fn = wrapper(lambda x: x * 0)
        elif type == "skip":
            self._fn = wrapper(nn.Identity())
        elif type == "proj":
            self._layer = nn.Conv2d(channels, channels, 3, 1, "same")
            self._fn = wrapper(self._layer)
        elif type == "var":
            self._fn = VariationalConv2d(
                channels, channels, 3, 1, "same"
            )
        else:
            raise RuntimeError("Unreachable")

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return self._fn(x)


class VAES(nn.Module):
    """Copied from torchseg.Unet and modified to insert
    Variational powers!
    """

    def __init__(
        self,
        label_channels: int,
        image_channels: int = 3,
        encoder_depth: int = 5,
        decoder_channels: list[int] = (256, 128, 64, 32, 16),
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        encoder_freeze: bool = False,
        skip_connections: List[str] = ["skip"] * 5,
        encoder_state_dict: Optional[Union[str, dict]] = None,
        activation: Callable = nn.Identity(),
    ):
        super().__init__()
        encoder_weights = utils.parse_enc_weights(encoder_weights)

        if encoder_weights is None and encoder_freeze is True:
            raise RuntimeError(
                "Frozen encoder with random weights is just stupid"
            )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=image_channels,
            indices=None,
            depth=encoder_depth,
            output_stride=None,
            weights=encoder_weights,
        )

        # remove first skip with same spatial resolution
        encoder_channels = self.encoder.out_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        skip_layers = self.create_skip_layers(
            skip_connections, encoder_channels
        )
        self.skip_layers = nn.ModuleList(skip_layers)

        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=label_channels,
            activation=activation,
            kernel_size=3,
            upsampling=1,
        )

        # This initializes weights with specific distributions
        initialization.initialize_decoder(self.decoder)
        initialization.initialize_head(self.segmentation_head)

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
            self.encoder.load_state_dict(encoder_state_dict)

        if encoder_freeze:
            self.encoder = utils.freeze_model(self.encoder)

        self.name = f"vae-{encoder_name}"

    @staticmethod
    def create_skip_layers(skip_connections, encoder_channels):
        skip_layers = []
        for channels, type in zip(encoder_channels, skip_connections):
            skip_layers.append(SkipLayer(type, channels))
        return skip_layers

    def forward(self, x):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        features = self.encoder(x)

        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        # project the features
        projected_features = []
        priors = []
        posteriors = []
        for layer, skip in zip(self.skip_layers, features):
            state = layer(skip)
            projected_features.append(state["out"])
            priors.extend(state.get("priors", []))
            posteriors.extend(state.get("posteriors", []))

        decoded = self.decoder(*projected_features)
        masks = self.segmentation_head(decoded)
        return {
            "out": masks,
            "priors": priors,
            "posteriors": posteriors,
        }


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            err = f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."  # noqa: E501
            raise ValueError(err)

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels,
                head_channels,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            use_batchnorm=use_batchnorm, attention_type=attention_type
        )
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, out_channels
            )
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = get_or_default(skips, i, None)
            x = decoder_block(x, skip)

        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.attention1 = modules.Attention(
        # attention_type, in_channels=in_channels + skip_channels
        # )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.attention2 = modules.Attention(
        # attention_type, in_channels=out_channels
        # )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # Removed attention as it is not used in UNet
            # x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # Removed attention as it is not used in UNet
        # x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super().__init__(conv, bn, relu)
