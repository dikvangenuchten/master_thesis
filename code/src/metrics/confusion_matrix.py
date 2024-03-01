from typing import Optional, Dict
from torch import Tensor
from torchmetrics.classification import MultilabelConfusionMatrix

from metrics.base_metric import BaseMetric


class ConfusionMetrics(BaseMetric):
    """Computes various metrics from a confusion matrix.
    
    Currently supported metrics are:
        -
        
    """
    def __init__(
        self,
        num_labels: int,
        threshold: float,
        logits: bool = True,
        ignore_index: Optional[int] = None,
    ):
        self._confusion_matrix = MultilabelConfusionMatrix(
            num_labels, threshold, ignore_index, normalize=None
        )

    def update(self, x: Tensor, y_true: Tensor, y_pred: Tensor, loss: Tensor):
        self._confusion_matrix.update(
            y_pred, y_true
        )

    def compute(self) -> Dict[str, Tensor]:
        """
        
        - :math:`C_{0, 0}`: True negatives
        - :math:`C_{0, 1}`: False positives
        - :math:`C_{1, 0}`: False negatives
        - :math:`C_{1, 1}`: True positives
        """
        cm = self._confusion_matrix.compute()
        tn = cm[:, 0, 0]
        fp =  cm[:, 0, 1]
        fn =  cm[:, 1, 0]
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
            "Precision": tp / (tp + fp),
            "Recall": tp / (tp + fn),
            "Accuracy": tp + tn / (tp + tn + fp + fn)   
        }