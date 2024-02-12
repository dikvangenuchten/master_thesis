import pytest
import torch

from metrics.average_metric import AverageMetric


def loss_fn():
    def _loss_extractor(_x, _y_true, _y_pred, loss):
        return loss

    return _loss_extractor


@pytest.mark.parametrize("val", [0, 1, 2])
def test_average_of_single_sample(val):
    bs = 16
    average_metric = AverageMetric("AverageTestConstant", loss_fn())
    average_metric.add_batch(
        x=torch.rand((bs, 1)),
        y_true=torch.rand((bs, 1)),
        y_pred=torch.rand((bs, 1)),
        loss=torch.ones((bs, 1)) * val,
    )
    log_dict = average_metric.get_log_dict()
    assert log_dict[average_metric.name] == val, "Average was not calculated correctly"


def test_reset():
    bs = 16
    average_metric = AverageMetric("AverageTestConstant", loss_fn())
    average_metric.add_batch(
        x=torch.rand((bs, 1)),
        y_true=torch.rand((bs, 1)),
        y_pred=torch.rand((bs, 1)),
        loss=torch.ones((bs, 1)) * 1,
    )
    log_dict = average_metric.get_log_dict()
    assert log_dict[average_metric.name] == 1, "Average was not calculated correctly"
    average_metric.reset()
    average_metric.add_batch(
        x=torch.rand((bs, 1)),
        y_true=torch.rand((bs, 1)),
        y_pred=torch.rand((bs, 1)),
        loss=torch.ones((bs, 1)) * 2,
    )
    log_dict = average_metric.get_log_dict()
    assert (
        log_dict[average_metric.name] == 2
    ), "Average was not calculated correctly after reset"
