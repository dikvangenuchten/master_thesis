from typing import Dict

import numpy as np
from torch import Tensor
from metrics.base_metric import BaseMetric

import wandb


class MaskMetric(BaseMetric):
    """Saves the first batch of images with segmentation masks"""

    def __init__(self, name: str, class_labels: dict, limit: int = 32) -> None:
        super().__init__(name)
        self._first_batch = None
        self._limit = limit

    def add_batch(self, x: Tensor, y_true: Tensor, y_pred: Tensor, loss: Tensor):
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
        return super().add_batch(x, y_true, y_pred, loss)

    def get_log_dict(self) -> Dict[str, Tensor]:
        images = []
        for img, gt_mask, pr_mask in zip(*self._first_batch):
            images.append(
                wandb.Image(
                    img,
                    masks={
                        "predictions": {"mask_data": pr_mask[0] > 0.5},
                        "ground_truth": {"mask_data": gt_mask[0]},
                    },
                )
            )
        return {self.name: images}

    def reset(self):
        return super().reset()
