from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor

LogSignature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


class BaseMetric(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def update(self, x: Tensor, y_true: Tensor, y_pred: Tensor, loss: Tensor):
        """Add a batch of data to the metric"""

    @abstractmethod
    def compute(self) -> Tensor:
        """Compute the metric"""

    @abstractmethod
    def reset(self):
        """Reset this metric for a new epoch"""
