import numpy as np
from torch import Tensor

import torch
import wandb
from metrics.base_metric import BaseMetric, StepData


class MaskMetric(BaseMetric):
    """Saves the first batch of images with segmentation masks"""

    def __init__(
        self, name: str, class_labels: dict, limit: int = 32
    ) -> None:
        super().__init__(name)
        self._first_batch = None
        self._limit = limit
        self.class_labels = class_labels
        self._image_label = "input"
        self._target_label = "target"

    def update(self, step_data: StepData):
        if self._first_batch is None:
            x = step_data.batch[self._image_label]
            y_true = step_data.batch[self._target_label]
            y_pred = step_data.model_out["out"]

            if x.shape[0] > self._limit:
                x, y_true, y_pred = (
                    x[: self._limit],
                    y_true[: self._limit],
                    y_pred[: self._limit],
                )
            # Convert from BCHW to BHWC
            self._first_batch = (
                np.transpose(x.detach().cpu().numpy(), (0, 2, 3, 1)),
                y_true.detach().cpu().numpy(),
                np.transpose(
                    y_pred.detach().cpu().numpy(), (0, 2, 3, 1)
                ),
            )

    def compute(self) -> Tensor:
        images = []
        for img, gt_mask, pr_mask in zip(*self._first_batch):
            images.append(
                wandb.Image(
                    img,
                    masks={
                        "predictions": {
                            "mask_data": pr_mask.argmax(-1),
                            "class_labels": self.class_labels,
                        },
                        "ground_truth": {
                            "mask_data": gt_mask,
                            "class_labels": self.class_labels,
                        },
                    },
                )
            )
        return {self.name: images}

    def reset(self):
        self._first_batch = None
