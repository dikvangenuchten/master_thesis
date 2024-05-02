from typing import Callable, Optional, Dict

import torch
from torch import nn

from .duq_loss import DUQLoss
from .gradient_penalty import GradientPenalty
from .kl_divergence import HierarchicalKLDivergenceLoss


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

    def add_log_callback(self, fn: Callable[[str, torch.Tensor], None]):
        """Adds a logging callback

        Args:
            fn (Callable[[str, torch.Tensor]]): The callback function
                Signature should be: fn(loss_name: str, loss: torch.Tensor)
                Return value is ignored.
        """
        self._loss_fn.register_forward_hook(_create_log_hook(fn))


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

    def add_log_callback(self, fn: Callable[[str, torch.Tensor], None]):
        """Adds a logging callback

        Args:
            fn (Callable[[str, torch.Tensor]]): The callback function
                Signature should be: fn(loss_name: str, loss: torch.Tensor)
                Return value is ignored.
        """
        if hasattr(self._loss_fn, "add_log_callback"):
            return self._loss_fn.add_log_callback(fn)

        self._loss_fn.register_forward_hook(_create_log_hook(fn))


class SummedLoss(nn.Module):
    """Sums the losses

    To log the individual losses use the `add_log_callback`.
    """

    def __init__(
        self, losses: Dict[str, nn.Module], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._names = list(losses.keys())
        self._losses = nn.ModuleList(list(losses.values()))
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
        for loss_fn in self._losses:
            if hasattr(loss_fn, "add_log_callback"):
                self._handles.append(loss_fn.add_log_callback(fn))
            else:
                self._handles.append(
                    loss_fn.register_forward_hook(_create_log_hook(fn))
                )


def _create_log_hook(fn):
    def _hook_logger(module: nn.Module, args, output: torch.Tensor):
        prefix = "train" if output.requires_grad else "val"
        fn(prefix + module.__class__.__name__, output)

    return _hook_logger


__all__ = [
    "DUQLoss",
    "GradientPenalty",
    "HierarchicalKLDivergenceLoss",
    "WeightedLoss",
    "WrappedLoss",
    "SummedLoss",
]
