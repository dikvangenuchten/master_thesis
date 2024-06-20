from itertools import zip_longest

from typing import List, Optional, Tuple
import torch
from torch import nn

from .modules import VariationalDecoderBlock, EncoderBlock


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
        encoder_channels: List[int] | str,
        reductions: List[int] | str,
        bottlenecks: List[float] | str,
        decoder_channels: List[int] | str | None = None,
        input_shape: Tuple[int, int] = (128, 128),
    ) -> None:
        super().__init__()

        # Note: This is not 'safe', but I only run configs I create
        allowed_characters = list("[](),.*+ ") + list(
            str(i) for i in range(10)
        )
        if isinstance(encoder_channels, str):
            assert all(c in allowed_characters for c in encoder_channels)
            encoder_channels = eval(encoder_channels)
        if isinstance(reductions, str):
            assert all(c in allowed_characters for c in reductions)
            reductions = eval(reductions)
        if isinstance(bottlenecks, str):
            assert all(c in allowed_characters for c in bottlenecks)
            bottlenecks = eval(bottlenecks)

        assert (
            len(encoder_channels) == len(reductions) == len(bottlenecks)
        ), f"`layer_depths`, `reductions` and, `bottlenecks` should all be the same length, but are: {len(encoder_channels)=}, {len(reductions)=}, {len(bottlenecks)=}"

        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels))

        self._image_channels = image_channels
        self._label_channels = label_channels
        self._encoder_channels = encoder_channels
        self._decoder_channels = decoder_channels
        self._reductions = reductions
        self._bottlenecks = bottlenecks
        tot_reduction = torch.prod(torch.tensor(reductions))
        assert (
            input_shape[0] % tot_reduction == 0
        ), f"Input shape {(input_shape[0])} must be a multiple of {tot_reduction}"
        assert (
            input_shape[1] % tot_reduction == 0
        ), f"Input shape {(input_shape[1])} must be a multiple of {tot_reduction}"
        self._latent_shape = (
            int(input_shape[0] / tot_reduction),
            int(input_shape[1] / tot_reduction),
        )

        img_encoder_channels = [image_channels, *encoder_channels]
        self._image_encoder_layers = self._create_encoder(
            reductions, bottlenecks, img_encoder_channels
        )

        self._image_decoder_layers = self._create_decoder(
            reductions, bottlenecks, img_encoder_channels
        )

        label_layers = [*decoder_channels, label_channels]
        self._label_decoder_layers = self._create_decoder(
            reductions, bottlenecks, list(reversed(label_layers))
        )

        # TODO: Determine if we want a seperate one for image and label
        self._learnable_latent = nn.Parameter(
            torch.empty(
                size=(1, encoder_channels[-1], *self._latent_shape)
            ),
            requires_grad=True,
        )
        nn.init.kaiming_uniform_(
            self._learnable_latent, nonlinearity="linear"
        )

    @staticmethod
    def _create_encoder(reductions, bottlenecks, layers):
        return nn.ModuleList(
            [
                EncoderBlock.make_block(
                    in_channels=layers[i],
                    out_channels=layers[i + 1],
                    bottleneck_ratio=bottlenecks[i],
                    downsample_factor=reductions[i],
                )
                for i in range(len(layers) - 1)
            ]
        )

    @staticmethod
    def _create_decoder(reductions, bottlenecks, layers):
        return nn.ModuleList(
            [
                VariationalDecoderBlock.make_block(
                    in_channels=layers[-i - 1],
                    skip_channels=layers[-i - 1],
                    out_channels=layers[-i - 2],
                    bottleneck_ratio=bottlenecks[-1 - i],
                    expansion=reductions[-i - 1],
                )
                for i in range(len(layers) - 1)
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
        x_skips = self.encode_image(x)
        mask = self.decode_label(x_skips)
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
        return skips

    def encode_label(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_image(
        self,
        x_skip: Optional[List[torch.Tensor]] = None,
        z_dim: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if x_skip is not None:
            bs = x_skip[0].size(0)
        elif z_dim is not None:
            x_skip = []
            bs, _, _ = z_dim
        else:
            bs = 1
        z = self._learnable_latent.expand(bs, -1, -1, -1)

        for layer, x_skip in zip_longest(
            self._image_decoder_layers, reversed(x_skip)
        ):
            z = layer(z, x_skip)
        return z

    def decode_label(
        self,
        x_skip: Optional[List[torch.Tensor]] = None,
        z_dim: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if x_skip is not None:
            bs = x_skip[0].size(0)
        elif z_dim is not None:
            x_skip = []
            bs, _, _ = z_dim
        else:
            bs = 1
        z = self._learnable_latent.expand(bs, -1, -1, -1)

        for layer, x_skip_ in zip_longest(
            self._label_decoder_layers, reversed(x_skip)
        ):
            z = layer(z, x_skip_)
        z["probs"] = z["out"].softmax(1)
        return z

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = self.encode_image(x)
        return self.decode_image(x_skip)
