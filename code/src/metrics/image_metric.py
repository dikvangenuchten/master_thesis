from typing import Dict

import torchvision
import wandb

from metrics.base_metric import BaseMetric, StepData


class ImageMetric(BaseMetric):
    """Saves the first batch of input and output as images"""

    def __init__(
        self,
        name: str,
        limit: int = 32,
    ) -> None:
        super().__init__(name)
        self._first_batch = None
        self._limit = limit
        self._input_label = "input"
        self._output_label = "out"

    def update(self, step_data: StepData):
        if self._first_batch is None:
            input = step_data.batch[self._input_label]
            output = step_data.model_out["out"]

            if input.shape[0] > self._limit:
                input, output = (
                    input[: self._limit],
                    output[: self._limit],
                )

            self._first_batch = (
                input.detach().to(device="cpu", non_blocking=True),
                output.detach().to(device="cpu", non_blocking=True),
            )

    def compute(self) -> Dict[str, wandb.Image]:
        if self._first_batch is None:
            return {}

        images = [
            wandb.Image(torchvision.utils.make_grid([img, recon]))
            for img, recon in zip(*self._first_batch)
        ]
        return {self.name: images}

    def reset(self):
        self._first_batch = None
