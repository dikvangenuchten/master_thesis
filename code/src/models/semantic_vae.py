from collections import defaultdict
from itertools import zip_longest

from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions


class ResBlock(nn.Module):
    """Residual Convolutional Block

    HParams:
        inC: The number of input channels
        outC: The number of output channels

    Input:
        in Tensor[B, inC, H, W]

    Output:
        out Tensor[B, outC, H, W]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_ratio: float = 1.0,
        activation: nn.Module = F.silu,
    ) -> None:
        super().__init__()

        bottle_filters = max(int(in_channels * bottleneck_ratio), 1)
        self._residual = in_channels == out_channels
        self._activation = activation

        self._layers = nn.ModuleList(
            [
                # in_filters, out_filters, kernel, stride
                nn.Conv2d(in_channels, bottle_filters, 1, 1),
                nn.Conv2d(
                    bottle_filters, bottle_filters, 3, 1, padding="same"
                ),
                nn.Conv2d(
                    bottle_filters, bottle_filters, 3, 1, padding="same"
                ),
                nn.Conv2d(bottle_filters, out_channels, 1, 1),
            ]
        )

    def forward(self, x) -> torch.Tensor:
        identity = x
        for layer in self._layers:
            x = layer(x)
            x = self._activation(x)
        if self._residual:
            return x + identity
        return x


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
        self._activation = nn.LeakyReLU(0.1)

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


class UnpoolLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, expansion: int
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1
        )
        self._expansion = expansion

    def forward(self, x) -> torch.Tensor:
        x = self._conv(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=self._expansion)
        return x


class SampleConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            # Mean and (log) variance
            out_channels=out_channels * 2,
            kernel_size=1,
        )
        # Recommended value by Efficient-VDVAE
        self._softplus = nn.Softplus(beta=torch.log(torch.tensor(2.0)))

    def __call__(
        self, *args: torch.Any, **kwds: torch.Any
    ) -> distributions.Distribution:
        """Only added for type-hinting"""
        return super().__call__(*args, **kwds)

    def forward(
        self, x, distribution=True
    ) -> distributions.Distribution:
        x = self._conv(x)
        mean, std = torch.chunk(x, chunks=2, dim=1)
        std = self._softplus(std)
        if distribution:
            return distributions.Normal(mean, std)
        return mean, std


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
    ) -> "DecoderBlock":
        return cls(
            in_channels=in_channels,
            skip_channels=skip_channels,
            latent_channels=32,  # Always 32 in Efficient VDVAE
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
            in_channels, latent_channels, expansion
        )

        self._prior_net = ResBlock(
            in_channels=latent_channels, out_channels=latent_channels
        )
        self._prior_layer = SampleConvLayer(latent_channels)

        self._posterior_net = ResBlock(
            in_channels=latent_channels + skip_channels,
            out_channels=latent_channels,
        )
        self._posterior_layer = SampleConvLayer(latent_channels)

        self._out_resblock = ResBlock(
            in_channels=latent_channels,
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
        )

    def __call__(
        self,
        input: Dict[str, torch.Tensor] | torch.Tensor,
        x_skip: torch.Tensor = None,
    ) -> torch.Any:
        if isinstance(input, torch.Tensor):
            return super().__call__({"out": input}, x_skip)
        return super().__call__(input, x_skip)

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        x_skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            out["posteriors"] = [*input.get("posteriors", []), posterior]
        else:
            dist = prior

        if not self.training:
            z = dist.scale
        else:
            z = dist.sample()

        out["out"] = self._out_resblock(residual + z)

        return out

    def posterior(
        self, x: torch.Tensor, x_skip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (x, x)

    def _sample(self, mu: torch.Tensor, std: torch.Tensor):
        # TODO during bootstrapping we should also sample
        if not self.training:
            return mu
        else:
            # Sample during training
            eps = torch.empty_like(mu).normal_(0.0, 1.0)
            return mu + std * eps


class SemanticVAE(nn.Module):
    """Semantic VAE

    The Semantic VAE uses a VAE as pretrained model.
    The output is generated purely from the latent space.

    During training the following options are available:
     - Complete Finetune (No freezing, no reguralizing using x-decoder)
     - Reguralize the encoder using previous decoder
     - Freeze the encoder (x-decoder is thus not necesarry)
    """

    def __init__(
        self,
        image_channels: int,
        label_channels: int,
        layer_depths: List[int],
        reductions: List[int],
        bottlenecks: List[float],
    ) -> None:
        super().__init__()

        image_layers = [image_channels, *layer_depths]
        self._image_encoder_layers = nn.ModuleList(
            [
                EncoderBlock.make_block(
                    in_channels=image_layers[i],
                    out_channels=image_layers[i + 1],
                    bottleneck_ratio=bottlenecks[i],
                    downsample_factor=reductions[i],
                )
                for i in range(len(image_layers) - 1)
            ]
        )

        self._image_decoder_layers = nn.ModuleList(
            [
                DecoderBlock.make_block(
                    in_channels=image_layers[-i - 1],
                    skip_channels=image_layers[-i - 1],
                    out_channels=image_layers[-i - 2],
                    bottleneck_ratio=bottlenecks[-1 - i],
                    expansion=reductions[-i - 1],
                )
                for i in range(len(image_layers) - 1)
            ]
        )

        label_layers = [label_channels, *layer_depths]
        self._label_decoder_layers = nn.ModuleList(
            [
                DecoderBlock.make_block(
                    in_channels=label_layers[-i - 1],
                    skip_channels=label_layers[-i - 1],
                    out_channels=label_layers[-i - 2],
                    bottleneck_ratio=bottlenecks[-1 - i],
                    expansion=reductions[-i - 1],
                )
                for i in range(len(label_layers) - 1)
            ]
        )

    @classmethod
    def default(
        cls,
        image_channels: int = 3,
        label_channels: int = 3,
    ):
        layer_depths: List[int] = ([8, 16, 32, 64],)
        reductions: List[int] = [2, 2, 2, 2]
        return cls(
            image_channels, label_channels, layer_depths, reductions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_space, x_skips = self.encode_image(x)
        mask = self.decode_label(latent_space, x_skips)
        return mask

    def encode_image(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode the image to the latent space

        Args:
            x (torch.Tensor): The image to be encoded

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Tuple containing:
                latent_space (torch.Tensor): The 'deepest' latent space
                latent_spaces (List[torch.Tensor]): All latent spaces
        """
        skips = []
        h = x
        for layer in self._image_encoder_layers:
            h, skip = layer(h)
            skips.append(skip)
        return h, skips

    def encode_label(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_image(
        self,
        z: torch.Tensor,
        x_skip: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x_skip = [] if x_skip is None else x_skip
        for layer, x_skip in zip_longest(
            self._image_decoder_layers, reversed(x_skip)
        ):
            z = layer(z, x_skip)
        return z

    def decode_label(
        self,
        z: torch.Tensor,
        x_skip: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        x_skip = [] if x_skip is None else x_skip
        for layer, x_skip_ in zip_longest(
            self._label_decoder_layers, reversed(x_skip)
        ):
            z = layer(z, x_skip_)
        return z

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_image(x)
        # TODO: Take mean
        mean = z
        return self.decode_image(mean)
