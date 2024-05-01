import torch
from torch import nn


class BinarySegmentationModel(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        activation=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._conv1 = nn.Conv2d(
            in_channels, out_channels, 5, padding="same"
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self._conv1(img)
        return {"out": x}
