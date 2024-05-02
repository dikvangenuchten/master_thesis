from typing import Dict, Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


from .sampleconv import SampleConvLayer
from .resblock import Conv2dBN, ResBlock


class UnpoolLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, expansion: int
    ) -> None:
        super().__init__()
        self._conv = Conv2dBN(
            in_channels,
            out_channels,
            1,
            1,
            activation=nn.ReLU(inplace=True),
        )
        self._expansion = expansion

    def forward(self, x) -> torch.Tensor:
        x = self._conv(x)
        x = F.interpolate(x, scale_factor=self._expansion)
        return x


class DecoderBlock(nn.Module):
    """DecoderBlock in the VAE

    HParams:
        inC: The amount of input channels
        latC: The amount of input channels of the skip connection
        outC: The amount of output channels
        expansion: The upscaling of the input

        latC: The amount of internal latent channels

    Inputs:
        x ([B, inC, H, W]):
        x_skip (B, skipC, H * expansion, W * expansion):

    Output: Tensor[B, outC, H * expansion, W * expansion]
    """

    @classmethod
    def make_block(
        cls,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bottleneck_ratio: float,
        expansion: int,
        latent_channels: int = 32,  # Always 32 in Efficient VDVAE
    ) -> "DecoderBlock":
        return cls(
            in_channels=in_channels,
            skip_channels=skip_channels,
            latent_channels=latent_channels,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
            expansion=expansion,
        )

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        latent_channels: int,
        out_channels: int,
        bottleneck_ratio: float,
        expansion: int,
    ) -> None:
        super().__init__()
        self._params = {
            "in_channels": in_channels,
            "skip_channels": skip_channels,
            "latent_channels": latent_channels,
            "out_channels": out_channels,
            "bottleneck_ratio": bottleneck_ratio,
            "expansion": expansion,
        }
        self._unpool = UnpoolLayer(
            in_channels, skip_channels, expansion
        )

        self._prior_net = ResBlock(
            in_channels=skip_channels, out_channels=skip_channels
        )
        self._prior_layer = SampleConvLayer(
            skip_channels, latent_channels
        )

        self._posterior_net = ResBlock(
            in_channels=skip_channels + skip_channels,
            out_channels=skip_channels,
        )
        self._posterior_layer = SampleConvLayer(
            skip_channels, latent_channels
        )

        self._z_projection = nn.Sequential(
            nn.Conv2d(latent_channels, skip_channels, 1, 1),
            nn.ReLU(inplace=True),
        )

        self._out_resblock = ResBlock(
            in_channels=skip_channels,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
        )

    def __call__(
        self,
        input: Dict[str, Tensor] | Tensor,
        x_skip: Tensor = None,
    ) -> torch.Any:
        if isinstance(input, Tensor):
            return super().__call__({"out": input}, x_skip)
        return super().__call__(input, x_skip)

    def forward(
        self,
        input: Dict[str, Tensor],
        x_skip: Optional[Tensor] = None,
    ) -> Tensor:
        x = self._unpool(input["out"])
        # Prior net is a residual block
        residual = self._prior_net(x)
        prior = self._prior_layer(residual)
        out = {
            "priors": [*input.get("priors", []), prior],
        }

        if x_skip is not None:
            # This is only the case if no skip connection is present
            # As this model is not for generating novel images/segmentations
            post = self._posterior_net(torch.cat((x, x_skip), dim=1))
            posterior = self._posterior_layer(post)
            dist = posterior
            out["posteriors"] = [
                *input.get("posteriors", []),
                posterior,
            ]
        else:
            dist = prior

        if self.training:
            z = dist.rsample()
        else:
            z = dist.loc

        z_proj = self._z_projection(z)
        out["out"] = self._out_resblock(residual + z_proj)

        return out

    def posterior(
        self, x: Tensor, x_skip: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return (x, x)


class DownSampleBlock(nn.Module):
    """Downsamples the input

    HParams:
        inC (int): The number of input channels
        outC (int): The number of output channels
        df (int): The downsample factor

    Input:
        x Tensor[B, inC, W, H]

    Output:
        x Tensor[B, outC, W/df, H/df]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_factor: int,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )
        self._activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)
        return self._activation(x)


class EncoderBlock(nn.Module):
    """The initial encoder block is based on the Efficient-VDVAE paper

    It consist of a ResBlock and a downsample block

        |
        | [B, C_in, W, H]
        v
    ---------
    | ResBl |
    ---------
        | [B, C_skip, W, H]
        L-> Skip connection
        v
    ---------
    | DownS |
    ---------
        | [B, C_out, W/d, H/d]
        v
        out
    """

    @classmethod
    def make_block(
        cls,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: float,
        downsample_factor: int,
    ):
        return cls(
            in_channels,
            out_channels,
            out_channels,
            bottleneck_ratio,
            downsample_factor,
        )

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        bottleneck_ratio: float,
        downsample_factor: int,
    ) -> None:
        super().__init__()

        self._resblock = ResBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            bottleneck_ratio=bottleneck_ratio,
        )

        self._skip_projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=skip_channels,
            kernel_size=1,
            stride=1,
        )

        self._downblock = DownSampleBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            downsample_factor=downsample_factor,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._resblock(x)
        skip = self._skip_projection(x)
        out = self._downblock(x)
        return out, skip
