from typing import List, Optional, Dict
from torch import Tensor
import torch
from torchmetrics.classification import MulticlassConfusionMatrix

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
        device: Optional[torch.device] = "cuda",
        prefix: Optional[str] = None,
        include: Optional[List[str]] = None
    ):
        super().__init__(name)
        prefix = "" if prefix is None else prefix
        if prefix != "" and not prefix.endswith(" "):
            prefix += " "
        self.num_classes = num_labels
        self._logits = logits
        self.ignore_index = ignore_index
        self._confusion_matrix = MulticlassConfusionMatrix(
            num_classes=num_labels,
            ignore_index=ignore_index,
            normalize=None,
            validate_args=False,
        ).to(device=device if torch.cuda.is_available else None)
        self._prefix = prefix
        self._include = include if include is not None else []

    def update(self, step_data: StepData):
        y_true = step_data.batch["target"]
        if "probs" in step_data.model_out:
            y_pred = step_data.model_out["probs"]
        elif "out" in step_data.model_out:
            y_pred = step_data.model_out["out"]
        else:
            raise RuntimeError(
                "The model output did not contain the keys 'probs' or 'out' which are required"
            )

        self._confusion_matrix.update(y_pred, y_true)

    def compute(self) -> Dict[str, Tensor]:
        """Compute the various metrics from the confusion matrix.

        - :math:`C_{0, 0}`: True negatives
        - :math:`C_{0, 1}`: False positives
        - :math:`C_{1, 0}`: False negatives
        - :math:`C_{1, 1}`: True positives
        """
        cm = self._confusion_matrix.compute()
        tp = (cm.diag()).sum()
        fp = (cm.sum(0) - cm.diag()).sum()
        fn = (cm.sum(1) - cm.diag()).sum()
        # True negatives are a bit useless in this context
        # tn = (cm.sum() - (tp + fp + fn)).sum()

        RQ = _safe_div(tp, (tp + 0.5 * (fp + fn)))
        # This definition of PQ/SQ is not entirely correct
        # As a segment is a bit arbitrary in the case of semantic segmentation
        # And no matching is taking place, we cannot calculate the IoU properly
        # Hence, the IoU is calculated as if each pixel is a segment
        # This results in `TP` having an IoU of 1, and 'FP' and 'FN' an IoU of 0
        # SQ = _safe_div(1, (fp + tp + fn))
        # PQ = SQ * RQ

        out = {
            f"{self._prefix}False positives": fp,
            f"{self._prefix}True positives": tp,
            f"{self._prefix}False negatives": fn,
            # f"{self._prefix}True negatives": tn,
            f"{self._prefix}RQ": RQ,
            # f"{self._prefix}SQ": SQ,
            # f"{self._prefix}PQ": PQ,
            f"{self._prefix}Recall": _safe_div(tp, (tp + fn)),
            f"{self._prefix}Precision": _safe_div(tp, (tp + fp)),
            f"{self._prefix}Jaccard Index": _safe_div(
                tp, (tp + fn + fp)
            ),
        }
        if len(self._include) > 0:
            if len(self._include) == 1:
                return out[self._include[0]]
            return {k: out[k] for k in self._include}
        return out

    def reset(self):
        return self._confusion_matrix.reset()


def _safe_div(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
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
