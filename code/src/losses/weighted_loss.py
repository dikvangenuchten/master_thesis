from typing import Dict, List

import torch
from torch import nn


class WeightedLoss(nn.Module):
    """Combines multiple losses into one.

    Each loss is multiplied by a weight, before being summed.

    Args:
        losses (List[nn.Module]):   A list of loss functions.
        weights (List[float]):      A list of weights.
    """

    def __init__(self, losses: List[nn.Module], weights: List[float]):
        assert len(losses) == len(
            weights
        ), "Ensure weights and losses are of equal length."
        self.weights = weights
        self.losses = losses

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss_sum = 0
        for w, fn in zip(self.weights, self.losses):
            loss = fn(model_out, batch)
            loss_sum += w * loss
        return loss
