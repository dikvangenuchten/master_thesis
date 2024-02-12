from abc import ABC, abstractmethod
from typing import Dict, Callable

from torch import Tensor


LogSignature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


class BaseMetric(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def add_batch(self, x: Tensor, y_true: Tensor, y_pred: Tensor, loss: Tensor):
        """Add a batch of data to the metric"""

    @abstractmethod
    def get_log_dict(self) -> Dict[str, Tensor]:
        """Retreive the metric such that it can be logged"""

    @abstractmethod
    def reset(self):
        """Reset this metric for a new epoch"""
