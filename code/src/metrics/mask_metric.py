import numpy as np
from torch import Tensor

import wandb
from metrics.base_metric import BaseMetric


class MaskMetric(BaseMetric):
    """Saves the first batch of images with segmentation masks"""

    def __init__(self, name: str, class_labels: dict, limit: int = 32) -> None:
        super().__init__(name)
        self._first_batch = None
        self._limit = limit
        self.class_labels = class_labels

    def update(self, x: Tensor, y_true: Tensor, y_pred: Tensor, loss: Tensor):
        if self._first_batch is None:
            if x.shape[0] > self._limit:
                x, y_true, y_pred = (
                    x[: self._limit],
                    y_true[: self._limit],
                    y_pred[: self._limit],
                )
            # Convert from BCHW to BHWC
            self._first_batch = (
                np.transpose(x.detach().cpu().numpy(), (0, 2, 3, 1)),
                np.transpose(y_true.detach().cpu().numpy(), (0, 2, 3, 1)),
                np.transpose(y_pred.detach().cpu().numpy(), (0, 2, 3, 1)),
            )
        return super().update(x, y_true, y_pred, loss)

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
                            "mask_data": gt_mask[..., 0],
                            "class_labels": self.class_labels,
                        },
                    },
                )
            )
        return {self.name: images}

    def reset(self):
        self._first_batch = None
