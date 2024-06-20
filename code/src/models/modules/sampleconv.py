from typing import Optional
import torch
from torch import nn, distributions


class SampleConvLayer(nn.Module):
    """A convulational sample layer.

    Creates a torch.distributions.Normal() parameterized by a learnable conv layer.
    """

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
        self._eps = torch.as_tensor(1e-5)

    def __call__(
        self, *args: torch.Any, **kwds: torch.Any
    ) -> distributions.Distribution:
        """Only added for type-hinting"""
        return super().__call__(*args, **kwds)

    def forward(self, x, distribution=True) -> distributions.Distribution:
        x = self._conv(x)
        mean, std = torch.chunk(x, chunks=2, dim=1)
        std = self._softplus(std) + self._eps
        if distribution:
            return distributions.Normal(mean, std)
        return mean, std
