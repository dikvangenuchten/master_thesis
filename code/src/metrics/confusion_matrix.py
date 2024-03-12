from typing import Optional, Dict
from torch import Tensor
import torch
from torch.nn import functional as F
from torchmetrics.classification import MultilabelConfusionMatrix

from metrics.base_metric import BaseMetric, StepData


class ConfusionMetrics(BaseMetric):
    """Computes various metrics from a confusion matrix.

    In the future it might be more efficient to compute
    these metrics in the trackers backend/frontend,
    instead of during the run itself.

    Currently supported metrics are:
        - True negatives
        - False positives
        - False negatives
        - True positives
        - SQ
        - RQ
        - PQ
        - Precision
        - Recall
        - Accuracy
    """

    def __init__(
        self,
        name: str,
        num_labels: int,
        threshold: float = 0.5,
        logits: bool = True,
        ignore_index: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(name)
        self.num_classes = num_labels
        self._logits = logits
        self.ignore_index = ignore_index
        self._confusion_matrix = MultilabelConfusionMatrix(
            num_labels, threshold, ignore_index, normalize=None
        ).to(device=device)

    def update(self, step_data: StepData):
        y_true = step_data.batch["target"]
        y_pred = step_data.model_out.out
        if y_true.shape != y_pred.shape:
            mask = y_true != self.ignore_index
            y_true = mask * F.one_hot(
                mask * y_true, num_classes=self.num_classes
            ).swapaxes(1, -1).squeeze(-1)
        self._confusion_matrix.to(device=y_pred.device)
        self._confusion_matrix.update(y_pred, y_true)

    def compute(self) -> Dict[str, Tensor]:
        """Compute the various metrics from the confusion matrix.

        - :math:`C_{0, 0}`: True negatives
        - :math:`C_{0, 1}`: False positives
        - :math:`C_{1, 0}`: False negatives
        - :math:`C_{1, 1}`: True positives
        """
        cm = self._confusion_matrix.compute()
        tn = cm[:, 0, 0]
        fp = cm[:, 0, 1]
        fn = cm[:, 1, 0]
        tp = cm[:, 1, 1]

        SQ = tp / (fp + tp + fn)
        RQ = tp / (tp + 0.5 * (fp + fn))
        PQ = SQ * RQ

        return {
            "True negatives": cm[:, 0, 0],
            "False positives": cm[:, 0, 1],
            "False negatives": cm[:, 1, 0],
            "True positives": cm[:, 1, 1],
            "SQ": SQ,
            "RQ": RQ,
            "PQ": PQ,
            "Precision": _safe_div(tp, (tp + fp)),
            "Recall": _safe_div(tp, (tp + fn)),
            "Accuracy": _safe_div(tp + tn, (tp + tn + fp + fn)),
        }

    def reset(self):
        return self._confusion_matrix.reset()


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor, default: float=0) -> torch.Tensor:
    """y = numerator / denominator, but sets y[numerator==0] to `default`(=0) regardless of denominator.

    Args:
        numerator (torch.Tensor): The numerator of the division
        denominator (torch.Tensor): The denominator of the division
        default float: The value to use when numerator==0

    Returns:
        torch.Tensor: Resulting Tensor
    """
    y = numerator / numerator
    y[numerator==0] = default
    return y