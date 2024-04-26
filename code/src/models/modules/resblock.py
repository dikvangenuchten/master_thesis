from typing import Optional, Callable

from torch import nn, Tensor


class Conv2dBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: Optional[Callable] = None,
    ):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        bn = nn.BatchNorm2d(out_channels)
        if activation is None:
            activation = nn.Identity()
        activation = activation
        super().__init__(conv, bn, activation)


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
        activation: nn.Module = nn.ReLU(inplace=True),
    ) -> None:
        super().__init__()

        bottle_filters = max(int(in_channels * bottleneck_ratio), 1)
        self._residual = in_channels == out_channels

        self._conv_bn_1 = Conv2dBN(
            in_channels, bottle_filters, 1, 1, activation
        )
        self._conv_bn_2 = Conv2dBN(
            bottle_filters, bottle_filters, 1, 1, activation
        )
        self._conv_bn_3 = Conv2dBN(
            bottle_filters, out_channels, 1, 1, None
        )

    def forward(self, x) -> Tensor:
        identity = x
        x = self._conv_bn_1(x)
        x = self._conv_bn_2(x)
        x = self._conv_bn_3(x)

        if self._residual:
            return x + identity
        return x
