import pytest
import torch

from metrics.base_metric import StepData
from metrics.confusion_matrix import ConfusionMetrics

def test_confustion_matrix_metric():
    
    ground_truth = torch.tensor([
        0,1,2,
        0,1,2,
        # 0,1,2,
    ])
    
    prediction = torch.nn.functional.one_hot(torch.tensor([
        0,1,2,
        1,2,1,
        # 2,0,1
    ]))
    
    # We have:
    # 3 TP
    # 3 FP (2 * '1', 1 * '2')
    # 3 FN (1 * '0')
    
    cm = ConfusionMetrics(name="test", num_labels=3, logits=False, device="cpu")
    cm.update(
        StepData(batch={"target": ground_truth}, model_out={"probs": prediction}, loss=None)
    )
    
    out = cm.compute()
    
    assert out["True positives"] == 3
    assert out["False negatives"] == 3
    assert out["False positives"] == 3
    assert out["Jaccard Index"] == 3 / 9

def test_confusion_matrix_random():
    bs = 16
    depth = 2
    ground_truth = torch.randint(0, 3, size=[bs, depth])
    prediction = torch.randint(0, 3, size=[bs, depth])
    
    tp = (ground_truth == prediction).sum()
    fn = (bs * depth) - tp
    fp = (bs * depth) - tp

    cm = ConfusionMetrics(name="test", num_labels=3, logits=False, device="cpu")
    cm.update(
        StepData(batch={"target": ground_truth}, model_out={"probs": prediction}, loss=None)
    )
    
    out = cm.compute()
    
    assert out["True positives"] == tp
    assert out["False negatives"] == fn
    assert out["False positives"] == fp
    assert out["Jaccard Index"] == tp / (tp + fn + fp)