from abc import ABC, abstractmethod
from typing import Callable, Dict

from torch import Tensor

from models import ModelOutput

LogSignature = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


class StepData:
    """Standardize Step Data for easier logging"""

    def __init__(
        self,
        batch: Dict[str, Tensor],
        model_out: ModelOutput,
        loss: Tensor,
    ) -> None:
        self.batch = batch
        self.model_out = model_out
        self.loss = loss


class BaseMetric(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def update(self, step_data: StepData):
        """Add a batch of data to the metric"""

    @abstractmethod
    def compute(self) -> Tensor:
        """Compute the metric"""

    @abstractmethod
    def reset(self):
        """Reset this metric for a new epoch"""
