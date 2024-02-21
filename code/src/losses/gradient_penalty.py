import torch
from torch import nn


class GradientPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # TODO check if `inputs` requires grad
        y_pred_sum = y_pred
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=inputs,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        return ((grad_norm - 1) ** 2).mean()
