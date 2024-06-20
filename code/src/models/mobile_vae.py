import itertools
from typing import Any, Dict, List, Mapping, Optional
from torch import nn, distributions
import torch
from torchvision.transforms import v2 as transforms

import torchseg

from .modules import (
    SampleConvLayer,
    VariationalDecoderBlock,
    UnpoolLayer,
    ResBlock,
)


class UpscaleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
    ) -> None:
        super().__init__()
        if expansion > 1 and in_channels == out_channels:
            self._unpool = nn.Identity()
        else:
            self._unpool = UnpoolLayer(
                in_channels, in_channels, expansion
            )
        self._block1 = ResBlock(in_channels, out_channels)
        self._block2 = ResBlock(out_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._unpool(input["out"])
        x = self._block1(x)
        input["out"] = self._block2(x)
        return input


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sample_layer = SampleConvLayer(in_channels, out_channels)

    def forward(self, input) -> Dict[str, torch.Tensor]:
        dist = self._sample_layer(input)
        return {
            "out": dist.mean if self.training else dist.mean,
            "priors": [
                distributions.Normal(
                    torch.zeros_like(dist.mean),
                    torch.ones_like(dist.stddev),
                )
            ],
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
        variational_connections: List[bool] = [
            True,
            True,
            True,
            True,
            True,
        ],
        activation: nn.Module = nn.Identity(),
        state_dict: Optional[dict] = None,
        load_encoder: bool = True,
        load_mid_block: bool = True,
        load_decoder: bool = True,
    ) -> None:
        super().__init__()

        # Hardcode the imagenet normalization function
        # I see no value in making this adaptable
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self._skip_connections = skip_connections[:encoder_depth]
        self._variational_connections = variational_connections[
            :encoder_depth
        ]
        self._activation = activation

        self._encoder = self._create_encoder(
            encoder_name=encoder_name,
            depth=encoder_depth,
            weights=encoder_weights,
            in_channels=image_channels,
        )
        self._mid_block = self._create_mid_block(
            self._encoder.out_channels[-1],
        )
        self._decoder = self._create_decoder(
            encoder_out_channels=self._encoder.out_channels,
            encoder_reductions=self._encoder.reductions,
            label_channels=label_channels,
            decoder_channels=decoder_channels,
        )

        if state_dict is not None:
            self.load_state_dict(
                state_dict,
                load_encoder=load_encoder,
                load_mid_block=load_mid_block,
                load_decoder=load_decoder,
            )

    @staticmethod
    def _create_encoder(
        encoder_name: str,
        depth: int,
        weights: str,
        in_channels: int,
    ) -> torch.nn.Module:
        return torchseg.encoders.get_encoder(
            encoder_name,
            depth=depth,
            weights=weights,
            in_channels=in_channels,
        )

    @staticmethod
    def _create_mid_block(in_channels):
        return MidBlock(in_channels=in_channels)

    @staticmethod
    def _create_decoder(
        encoder_out_channels: List[int],
        encoder_reductions: List[int],
        label_channels: int,
        decoder_channels: Optional[List[int]] = None,
    ) -> nn.ModuleList:
        expansions = [
            int(b / a) for a, b in itertools.pairwise(encoder_reductions)
        ]

        if decoder_channels is None:
            decoder_channels = encoder_out_channels[::-1]
            decoder_channels[-1] = label_channels
        assert (
            len(decoder_channels) == len(encoder_out_channels)
        ), "The length of decoder_channels must be equal to the encoder_depth"

        return nn.ModuleList(
            [
                VariationalDecoderBlock(
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

    def state_dict(self, *args, **kwargs):
        return {
            "encoder": self._encoder.state_dict(*args, **kwargs),
            "mid_block": self._mid_block.state_dict(*args, **kwargs),
            "decoder": self._decoder.state_dict(*args, **kwargs),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
        load_encoder: bool = True,
        load_mid_block: bool = True,
        load_decoder: bool = True,
    ):
        if load_encoder:
            self._encoder.load_state_dict(
                state_dict["encoder"], strict=strict, assign=assign
            )
        if load_mid_block:
            self._mid_block.load_state_dict(
                state_dict["mid_block"], strict=strict, assign=assign
            )
        if load_decoder:
            self._decoder.load_state_dict(
                state_dict["decoder"], strict=strict, assign=assign
            )

    def forward(self, x):
        x = self._normalize(x)
        skip_connections = self.encode(x)
        return self.decode(skip_connections)

    def encode(self, x) -> List[torch.Tensor]:
        # Reverse the order so we can iterate from bottom up
        skip_connections = self._encoder(x)[::-1]
        skip_connections[0] = self._mid_block(skip_connections[0])
        return skip_connections

    def decode(self, skip_connections):
        out = skip_connections[0]
        for layer, skip_data, skip, var in zip(
            self._decoder,
            skip_connections[1:],
            self._skip_connections,
            self._variational_connections,
            strict=True,
        ):
            out = layer(out, skip_data if skip else None, var=var)

        out["out"] = self._activation(out["out"])

        return out
