import itertools
from typing import Dict, List, Optional
from torch import nn, distributions
import torch

import torchseg

from .modules import SampleConvLayer, DecoderBlock


class MidBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sample_layer = SampleConvLayer(in_channels, out_channels)
        
        
    def forward(self, input) -> Dict[str, torch.Tensor]:
        dist = self._sample_layer(input)
        return {
            "out": dist.rsample() if self.training else dist.mean,
            "priors": [distributions.Normal(torch.zeros_like(dist.mean), torch.ones_like(dist.stddev))],
            "posteriors": [dist],
        }

class MobileVAE(nn.Module):
    def __init__(
        self,
        label_channels: int,
        encoder_name: str = "mobilenetv2_100",
        encoder_depth: int = 5,
        image_channels: int = 3,
        decoder_channels: Optional[List[int]] = None,
        encoder_weights: str = "imagenet",
        skip_connections: List[bool] = [True, True, True, True, True],
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

        expansions = [
            int(b / a)
            for a, b in itertools.pairwise(self._encoder.reductions)
        ]
        
        self._mid_block = MidBlock(in_channels=decoder_channels[0])

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
        self._skip_connections = skip_connections

        self._kwargs = kwargs

    def forward(self, x):
        # First one is alwasy the raw image
        skip_connections = self._encoder(x)[::-1]

        out = self._mid_block(skip_connections[0])
        for layer, skip_data, skip in zip(
            self._decoder,
            skip_connections[1:],
            self._skip_connections,
            strict=True,
        ):
            out = layer(out, skip_data if skip else None)

        return out
