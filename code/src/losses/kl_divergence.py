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


class KLDivergence(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        model_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        posteriors = model_out["posteriors"]
        priors = model_out["priors"]

        for posterior, prior in zip(posteriors, priors):
            loss = nn.functional.kl_div()
            p_mean, p_std = posterior.chunk(2, dim=1)
            q_mean, q_std = prior.chunk(2, dim=1)
