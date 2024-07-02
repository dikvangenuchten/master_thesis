from typing import Optional
from torch import nn
import torch
import math


class AnnealingWeightedLoss(nn.Module):
    def __init__(
        self,
        loss_fn: nn.Module,
        start_value: float,
        end_value: Optional[float],
        max_step: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        if end_value is None or end_value == start_value:
            self.forward = loss_fn
            end_value = start_value
            return

        self._eta_min = start_value
        self._half_range = 0.5 * (end_value - start_value)
        self._max_step = max_step
        self._loss_fn = loss_fn

    def forward(self, batch, model_out):
        return self._calc_weight(batch.get("step", 0)) * self._loss_fn(
            batch, model_out
        )

    def _calc_weight(self, step) -> torch.Tensor:
        period = math.pi * step / self._max_step
        return self._eta_min + self._half_range * (1 + math.cos(period))
