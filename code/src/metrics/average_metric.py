from torch import Tensor

from metrics.base_metric import BaseMetric, LogSignature, StepData


class AverageMetric(BaseMetric):
    """Calculates the average of the"""

    def __init__(self, name: str, fn: LogSignature):
        super().__init__(name)
        self._fn = fn
        self._sum = 0
        self._count = 0

    def update(self, step_data: StepData):
        self._count += step_data.batch["input"].shape[0]
        self._sum += self._fn(step_data).sum()

    def compute(self) -> Tensor:
        return {self.name: self._sum / self._count}

    def reset(self):
        self._sum = 0
        self._count = 0
