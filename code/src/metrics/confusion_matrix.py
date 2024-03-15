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
        prefix: Optional[str] = None,
    ):
        super().__init__(name)
        prefix = "" if prefix is None else prefix
        if prefix != "" and not prefix.endswith(" "):
            prefix += " "
        self.num_classes = num_labels
        self._logits = logits
        self.ignore_index = ignore_index
        self._confusion_matrix = MultilabelConfusionMatrix(
            num_labels, threshold, ignore_index, normalize=None
        ).to(device=device)
        self._prefix = prefix

    def update(self, step_data: StepData):
        y_true = step_data.batch["target"]
        y_pred = step_data.model_out.out
        if y_true.shape != y_pred.shape:
            mask = y_true != self.ignore_index
            y_true = mask * F.one_hot(
                mask * y_true, num_classes=self.num_classes
            ).permute(0, -1, 1, 2).squeeze(-1)
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
        tn = cm[:, 0, 0].sum()
        fp = cm[:, 0, 1].sum()
        fn = cm[:, 1, 0].sum()
        tp = cm[:, 1, 1].sum()

        SQ = _safe_div(tp, (fp + tp + fn))
        RQ = _safe_div(tp, (tp + 0.5 * (fp + fn)))
        PQ = SQ * RQ

        return {
            f"{self._prefix}True negatives": tn,
            f"{self._prefix}False positives": fp,
            f"{self._prefix}False negatives": fn,
            f"{self._prefix}True positives": tp,
            f"{self._prefix}SQ": SQ,
            f"{self._prefix}RQ": RQ,
            f"{self._prefix}PQ": PQ,
            f"{self._prefix}Precision": _safe_div(tp, (tp + fp)),
            f"{self._prefix}Recall": _safe_div(tp, (tp + fn)),
            f"{self._prefix}Accuracy": _safe_div(tp + tn, (tp + tn + fp + fn)),
        }

    def reset(self):
        return self._confusion_matrix.reset()


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """y = numerator / denominator, but sets y[numerator==0] to 0 regardless of denominator.

    Args:
        numerator (torch.Tensor): The numerator of the division
        denominator (torch.Tensor): The denominator of the division

    Returns:
        torch.Tensor: Resulting Tensor
    """
    y = numerator / denominator
    y[numerator == 0] = 0.0
    return y
