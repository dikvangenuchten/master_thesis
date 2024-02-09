import torch
from torch import nn


class BinarySegmentationModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._conv1 = nn.Conv2d(3, 1, 5, padding="same")
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self._conv1(img)
        
