from typing import Dict
from torch import Tensor
from metrics.base_metric import BaseMetric, LogSignature


class AverageMetric(BaseMetric):
    """Calculates the average of the"""

    def __init__(self, name: str, fn: LogSignature):
        super().__init__(name)
        self._fn = fn
        self._sum = 0
        self._count = 0

    def add_batch(self, x: Tensor, y_true: Tensor, y_pred: Tensor, loss: Tensor):
        self._count += x.shape[0]
        self._sum += self._fn(x, y_true, y_pred, loss).sum()

    def get_log_dict(self) -> Dict[str, Tensor]:
        return {self.name: self._sum / self._count}

    def reset(self):
        self._sum = 0
        self._count = 0
