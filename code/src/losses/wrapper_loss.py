from typing import Dict, Optional

import torch
from torch import nn


class WrappedLoss(nn.Module):
    """Wraps a loss function with a 'nicer' interface.

    Args:
        loss_fn (nn.Module): The loss function to be wrapped.
        keys (Dict[str, str]): A mapping of state keys to loss_fn kwargs
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        keys: Optional[Dict[str, str]],
    ):
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
