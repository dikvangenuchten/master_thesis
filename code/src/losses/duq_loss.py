import torch
from torch import nn
import torch.nn.functional as F


class DUQLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(y_pred, y_true)
