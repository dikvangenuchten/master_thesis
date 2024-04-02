from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: nn.Module,
        activation: nn.Module = F.relu,
    ) -> None:
        super().__init__()
        self._conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=1,
        )
        self._bn1 = nn.BatchNorm2d(out_channels)
        self._conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
        )
        self._bn2 = nn.BatchNorm2d(out_channels)
        self._downsample = downsample
        self._activation = activation

    def forward(self, x) -> torch.Tensor:
        identity = self._downsample(x)

        out = self._conv1(x)
        out = self._bn1(out)
        out = self._activation(out)

        out = self._conv2(out)
        out = self._bn2(out)
        out = self._activation(out)

        out = out + identity

        return self._activation(out)

    @classmethod
    def make_block(cls, in_channels: int, out_channels: int, downsample_factor: int):
        """Create a block based on the required dimensions + strides"""

        # Using conv is more flexible then using pooling and (theoratically) can become a AvgPool
        downsample = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(downsample_factor, downsample_factor),
            stride=downsample_factor,
        )

        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=downsample_factor,
            downsample=downsample,
        )


class EncoderBlock(nn.Module):
    """The initial encoder block is based on the Efficient-VDVAE paper"""

    def __init__(self, channels: List[int], downsample_factor: int = 2) -> None:
        super().__init__()
        self.layers = []
        for i in range(len(channels) - 1):
            self.layers.append(
                ResBlock.make_block(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    # Only downsample the last layer
                    downsample_factor=downsample_factor
                    if i == (len(channels) - 2)
                    else 1,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UnpoolLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int) -> None:
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self._expansion = expansion

    def forward(self, x) -> torch.Tensor:
        x = self._conv(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=self._expansion)
        return x


class SampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
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

    def forward(self, x):
        x = self._conv(x)
        mean, std = torch.chunk(x, chunks=2, dim=1)
        return mean, std


class DecoderBlock(nn.Module):
    @classmethod
    def make_block(
        cls,
        in_channels: int,
        skip_channels: int,
        latent_channels: int,
        out_channels: int,
        expansion: int,
    ) -> "DecoderBlock":
        return cls(
            in_channels=in_channels,
            skip_channels=skip_channels,
            latent_channels=latent_channels,
            out_channels=out_channels,
            expansion=expansion,
        )

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        latent_channels: int,
        out_channels: int,
        expansion: int,
    ) -> None:
        super().__init__()
        self._unpool = UnpoolLayer(in_channels, latent_channels, expansion)

        self._prior_net = ResBlock.make_block(
            latent_channels, latent_channels, downsample_factor=1
        )
        self._prior_layer = SampleConvLayer(latent_channels)

        self._posterior_net = ResBlock.make_block(
            latent_channels + skip_channels,
            latent_channels,
            downsample_factor=1,
        )
        self._posterior_layer = SampleConvLayer(latent_channels)

        self._out_resblock = ResBlock.make_block(
            latent_channels, out_channels, downsample_factor=1
        )

    def forward(self, x, x_skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self._unpool(x)
        # Prior net is a residual block
        residual = self._prior_net(x)
        prior_stats = self._prior_layer(residual)

        if x_skip is not None:
            # This is only the case if no skip connection is present
            # As this model is not for generating novel images/segmentations
            post = self._posterior_net(torch.cat((x, x_skip), dim=1))
            posterior_stats = self._posterior_layer(post)
            z = self._sample(*posterior_stats)
        else:
            z = self._sample(*prior_stats)

        out = residual + z

        return self._out_resblock(out)

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
    ) -> None:
        super().__init__()

        image_layers = [image_channels, *layer_depths]
        self._image_encoder = nn.Sequential(
            *[
                EncoderBlock([image_layers[i], image_layers[i + 1]], reductions[i])
                for i in range(len(image_layers) - 1)
            ]
        )

        self._image_decoder = nn.Sequential(
            *[
                DecoderBlock(
                    in_channels=image_layers[-i - 1],
                    skip_channels=image_layers[-i - 2],
                    latent_channels=image_layers[-i - 2],
                    out_channels=image_layers[-i - 2],
                    expansion=reductions[-i],
                )
                for i in range(len(image_layers) - 1)
            ]
        )

        label_layers = [label_channels, *layer_depths]
        self._label_decoder = nn.Sequential(
            *[
                DecoderBlock(
                    in_channels=label_layers[-i - 1],
                    skip_channels=label_layers[-i - 2],
                    latent_channels=label_layers[-i - 2],
                    out_channels=label_layers[-i - 2],
                    expansion=reductions[-i],
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
        return cls(image_channels, label_channels, layer_depths, reductions)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self._image_encoder(x)

    def encode_label(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        return self._image_decoder(z)

    def decode_label(self, z: torch.Tensor) -> torch.Tensor:
        return self._label_decoder(z)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_image(x)
        # TODO: Take mean
        mean = z
        return self.decode_image(mean)
