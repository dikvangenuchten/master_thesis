import torch
from torch import nn
from diffusers import AutoencoderKL

from models.duq import DUQHead


class VAE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # For now hardcode the backbone model
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        model = AutoencoderKL.from_single_file(url)

        self._encoder = model.encoder
        self._encoder.gradient_checkpointing = True
        self._decoder = model.decoder
        self._decoder.gradient_checkpointing = True

        # Replace the decoder head
        self._decoder.conv_out = DUQHead(
            in_channels=self._decoder.conv_norm_out.num_channels,
            num_classes=out_channels,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A full forward pass, from image: x to output: y

        Internally calls the encode and decode back to back.

        Args:
            x (torch.Tensor): An input image

        Returns:
            torch.Tensor: The Panoptic Segmentation output
        """
        z = self.encode(x)
        return self.decode(z)
