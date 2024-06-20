from typing import Any, Dict, List
import torch
from torchseg.base import (
    modules,
    SegmentationModel,
    SegmentationHead as SegmentationHead_,
)
from torchseg.encoders import get_encoder
from torchseg.decoders.unet.decoder import DecoderBlock
from torch import Tensor, nn, distributions
from torchvision.transforms import v2 as transforms


class MetadataModule(nn.Module):
    """Small wrapper around nn.Module

    This ensures that the input is always an dictionary,
    with 'out' being the output of the previous layer.
    This allows for metadata to be added, which will be
    passed verbatim (unless modified). Mostly it is used
    to pass extra information to loss functions.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, state, *args: Any, **kwds: Any) -> Any:
        if isinstance(state, Tensor):
            return super().__call__({"out": state}, *args, **kwds)
        return super().__call__(state, *args, **kwds)


class Conv2d(MetadataModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, state):
        state["out"] = self._conv(state["out"])
        return state


class VariationalConv2d(MetadataModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    ) -> None:
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        # Recommended value by Efficient-VDVAE
        self._softplus = nn.Softplus(beta=torch.log(torch.tensor(2.0)))
        self._eps = torch.as_tensor(1e-5)
        self.always_sample = False

    def forward(self, state):
        x = self._conv(state["out"])
        mean, std = torch.chunk(x, chunks=2, dim=1)
        std = self._softplus(std) + self._eps
        posterior = distributions.Normal(mean, std)
        prior = distributions.Normal(
            torch.zeros_like(posterior.mean),
            torch.ones_like(posterior.stddev),
        )
        state.setdefault("priors", []).append(prior)
        state.setdefault("posteriors", []).append(posterior)
        state["out"] = (
            posterior.rsample()
            if self.training or self.always_sample
            else posterior.mean
        )
        return state


class VariationalConv2dReLU(MetadataModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        variational: bool = True,
    ):
        super().__init__()
        if variational:
            self._conv = VariationalConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not (use_batchnorm),
            )
        else:
            self._conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=not (use_batchnorm),
            )
        self._relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            self._bn = nn.BatchNorm2d(out_channels)
        else:
            self._bn = nn.Identity()

    def forward(self, state: Dict[str, Tensor]):
        state = self._conv(state)
        x = self._relu(state["out"])
        state["out"] = self._bn(x)
        return state


class VariationalCenterBlock(MetadataModule):
    """Variational variant of torchseg.decoders.CenterBlock"""

    def __init__(
        self,
        in_channels,
        out_channels,
        use_batchnorm,
        variational: bool,
    ):
        super().__init__()
        self._conv1 = modules.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        if variational:
            self._conv2 = VariationalConv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )
        else:
            self._conv2 = modules.Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=use_batchnorm,
            )

    def forward(self, state):
        state = self._conv1(state["out"])
        state["out"] = self._conv2(state)
        return state


class VariationalDecoderBlock(MetadataModule):
    """Variational variant of the decoder block

    Adds an extra
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
        variational: bool = True,
    ):
        super().__init__()
        self._skip_projection = VariationalConv2dReLU(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=1,
            variational=variational,
        )
        self._decoder_block = DecoderBlock(
            in_channels=in_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            use_batchnorm=use_batchnorm,
            attention_type=attention_type,
        )

    def forward(self, state, skip=None):
        x = state["out"]
        if skip is not None:
            skip_state = self._skip_projection(skip)
            state.setdefault("priors", []).extend(
                skip_state.get("priors", [])
            )
            state.setdefault("posteriors", []).extend(
                skip_state.get("posteriors", [])
            )
        state["out"] = self._decoder_block(x, skip)
        return state


class VariationalUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        center_variational: bool = True,
        skip_connections: List[bool] = [True] * 5,
        variational_skip_connections: List[bool] = [True] * 5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            err = f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."  # noqa: E501
            raise ValueError(err)
        if n_blocks != len(skip_connections):
            err = f"Model depth is {n_blocks}, but you provide `skip_connections` for {len(skip_connections)} blocks."  # noqa: E501
            raise ValueError(err)
        if n_blocks != len(variational_skip_connections):
            err = f"Model depth is {n_blocks}, but you provide `variational` for {len(variational_skip_connections)} blocks."  # noqa: E501
            raise ValueError(err)

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = VariationalCenterBlock(
                head_channels,
                head_channels,
                use_batchnorm=use_batchnorm,
                variational=center_variational,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            use_batchnorm=use_batchnorm, attention_type=attention_type
        )
        blocks = [
            VariationalDecoderBlock(
                in_ch,
                skip_ch,
                out_ch,
                variational=var,
                **kwargs,
            )
            for in_ch, skip_ch, out_ch, var in zip(
                in_channels,
                skip_channels,
                out_channels,
                variational_skip_connections,
            )
        ]
        self.blocks = nn.ModuleList(blocks)
        self._skip_connections = skip_connections

    def forward(self, *features):
        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            if len(skips) > i:
                skip = skips[i]
                if not self._skip_connections[i]:
                    skip = skip * 0
            else:
                skip = None
            x = decoder_block(x, skip)
        return x


class SegmentationHead(SegmentationHead_):
    """Wrap Segmentation head to work with dict"""

    def forward(self, state: Dict[str, Tensor]):
        state["out"] = super().forward(state["out"])
        return state


class VariationalUNet(SegmentationModel):
    """Variational UNet (Also Hierachical Unet)"""

    def __init__(
        self,
        image_channels: int,
        label_channels: int,
        encoder_depth: int = 5,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        skip_connections: List[bool] = [True] * 5,
        variational_skip_connections: List[bool] = [True] * 5,
        encoder_name="mobilenetv2_100",
        encoder_weights="imagenet",
        activation=nn.Identity(),
        encoder_params: dict = {},
    ):
        super().__init__()
        if encoder_weights is None or encoder_weights.lower() == "none":
            encoder_weights = None

        assert (
            image_channels == 3
        ), "Currently only RGB images are supported (due to pretrained weights)"

        # Hardcode the imagenet normalization function
        # I see no value in making this adaptable
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.encoder = get_encoder(
            encoder_name,
            in_channels=image_channels,
            indices=None,
            depth=encoder_depth,
            output_stride=None,
            weights=encoder_weights,
            **encoder_params,
        )

        self.decoder = VariationalUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=True,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=None,
            skip_connections=skip_connections,
            variational_skip_connections=variational_skip_connections,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=label_channels,
            activation=activation,
            kernel_size=3,
            upsampling=1,
        )

        self.classification_head = None

        self.name = f"variational-u-{encoder_name}"
        self.initialize()
