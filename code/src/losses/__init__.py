from typing import Callable, Optional, Dict, List

import torch
from torch import nn

from .duq_loss import DUQLoss
from .gradient_penalty import GradientPenalty
from .kl_divergence import HierarchicalKLDivergence


class WrappedLoss(nn.Module):
    """Wraps a loss function with a 'nicer' interface.

    Args:
        loss_fn (nn.Module): The loss function to be wrapped.
        keys (Dict[str, str]): A mapping of state keys to loss_fn kwargs
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        keys: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self._loss_fn = loss_fn
        if keys is None:
            keys = {"out": "input", "target": "target"}
        self._keys = keys

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ):
        # TODO: Determine if it is better to have
        #       this 'state' be the interface.
        state = {**batch, **model_out}
        kwargs = {v: state[k] for k, v in self._keys.items()}
        return self._loss_fn(**kwargs)


class WeightedLoss(nn.Module):
    """Weighted Loss
    Loss is multiplied by the given weigth

    Args:
        losses (nn.Module): The loss function.
        weights (float):    The weight.
    """

    def __init__(self, loss_fn: nn.Module, weight: float):
        super().__init__()
        self._weight = weight
        self._loss_fn = loss_fn

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._weight * self._loss_fn(*args, **kwargs)


class SummedLoss(nn.Module):
    """Sums the losses

    Currently it is not possible to track the losses individually.
    """

    def __init__(
        self, losses: List[nn.Module], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._losses = nn.ModuleList(losses)
        self._handles = []

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return sum(fn(*args, **kwargs) for fn in self._losses)

    def add_log_callback(self, fn: Callable[[str, torch.Tensor], None]):
        """Adds a logging callback

        Args:
            fn (Callable[[str, torch.Tensor]]): The callback function
                Signature should be: fn(loss_name: str, loss: torch.Tensor)
                Return value is ignored.
        """

        def _hook_logger(module, args, output):
            # TODO: Not all modules have `__name__` set.
            fn(module.__class__.__name__, output)

        self._handles.extend(fn.register_forward_hook(_hook_logger) for fn in self._losses)


__all__ = [
    "DUQLoss",
    "GradientPenalty",
    "HierarchicalKLDivergence",
    "WeightedLoss",
    "WrappedLoss",
    "SummedLoss",
]
