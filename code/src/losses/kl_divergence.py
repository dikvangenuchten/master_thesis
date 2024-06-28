from typing import Dict
import torch
from torch import nn


def calculate_std_loss(p: torch.Tensor, q: torch.Tensor):
    q_std = q[1]
    p_std = p[1]
    term1 = (p[0] - q[0]) / q_std
    term2 = p_std / q_std
    loss = (
        0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    )
    return loss


class HierarchicalKLDivergenceLoss(nn.Module):
    """Calculates the Hierachical KLDivergence loss."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Small hack to ensure accelerate works automagically
        self.register_buffer("_zero", torch.zeros(1), persistent=False)

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # TODO warn once when unused
        posteriors = model_out.get("posteriors", [])
        priors = model_out.get("priors", [])

        loss = torch.zeros_like(self._zero)
        for posterior, prior in zip(posteriors, priors):
            if posterior is None or prior is None:
                continue
            _loss = (
                torch.distributions.kl_divergence(posterior, prior)
                .flatten(1)
                .sum(1)
            )
            loss += _loss.mean()
        return loss
