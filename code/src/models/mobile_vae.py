import itertools
from typing import List, Optional
from torch import nn
import torchseg

from .modules import SampleConvLayer, DecoderBlock


class MobileVAE(nn.Module):
    def __init__(
        self,
        label_channels: int,
        encoder_name: str = "mobilenetv2_100",
        encoder_depth: int = 5,
        image_channels: int = 3,
        decoder_channels: Optional[List[int]] = None,
        encoder_weights: str = "imagenet",
        **kwargs,
    ) -> None:
        super().__init__()

        self._encoder = torchseg.encoders.get_encoder(
            encoder_name,
            depth=encoder_depth,
            weights=encoder_weights,
            in_channels=image_channels,
        )
        # First one is always the raw image
        encoder_out_channels = self._encoder.out_channels

        if decoder_channels is None:
            decoder_channels = encoder_out_channels[::-1]
            decoder_channels[-1] = label_channels
        assert (
            len(decoder_channels) == len(encoder_out_channels)
        ), "The length of decoder_channels must be equal to the encoder_depth"

        self._sample_layers = nn.ModuleList(
            [
                SampleConvLayer(in_channels, out_channels)
                for in_channels, out_channels in zip(
                    encoder_out_channels, decoder_channels
                )
            ]
        )

        expansions = [
            int(b / a)
            for a, b in itertools.pairwise(self._encoder.reductions)
        ]

        self._decoder = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=in_channels,
                    latent_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    bottleneck_ratio=1.0,
                    expansion=expansion,
                )
                for in_channels, skip_channels, out_channels, expansion in zip(
                    decoder_channels[:-1],
                    reversed(encoder_out_channels[:-1]),
                    decoder_channels[1:],
                    expansions,
                    strict=True,
                )
            ]
        )

        self._kwargs = kwargs

    def forward(self, x):
        # First one is alwasy the raw image
        skip_connections = self._encoder(x)[::-1]

        out = skip_connections[0]
        for layer, skip in zip(
            self._decoder, skip_connections[1:], strict=True
        ):
            out = layer(out, skip)

        return out
