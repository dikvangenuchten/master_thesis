from torch import Tensor

from metrics.base_metric import BaseMetric, LogSignature, StepData


class AverageMetric(BaseMetric):
    """Calculates the average of the"""

    def __init__(self, name: str, fn: LogSignature):
        super().__init__(name)
        self._fn = fn
        self._sum = 0
        self._sum_squared = 0
        self._count = 0

    def update(self, step_data: StepData):
        self._count += step_data.batch["input"].shape[0]
        self._sum += self._fn(step_data).sum()
        self._sum_squared += self._fn(step_data).pow(2).sum()

    def compute(self) -> Tensor:
        mean = self._sum / self._count
        var = (mean.pow(2) - (self._sum_squared / self._count)).abs()
        return {self.name: mean, f"{self.name}-var": var}

    def reset(self):
        self._sum = 0
        self._sum_squared = 0
        self._count = 0
