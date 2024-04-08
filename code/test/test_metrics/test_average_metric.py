import pytest
import torch

from metrics.average_metric import AverageMetric
from metrics.base_metric import StepData
from models import ModelOutput


@pytest.fixture(params=[1, 2, 16])
def batch_size(request):
    yield request.param


def loss_fn():
    def _loss_extractor(step_data: StepData):
        return step_data.loss

    return _loss_extractor


@pytest.mark.parametrize("val", [0, 1, 2])
def test_average_of_single_sample(batch_size: int, val: int):
    average_metric = AverageMetric("AverageTestConstant", loss_fn())
    sd = StepData(
        {"input": torch.rand(batch_size)},
        ModelOutput(),
        torch.ones((batch_size, 1)) * val,
    )
    average_metric.update(sd)
    log_dict = average_metric.compute()
    assert (
        log_dict[average_metric.name] == val
    ), "Average was not calculated correctly"


def test_reset(batch_size: int):
    average_metric = AverageMetric("AverageTestConstant", loss_fn())
    sd = StepData(
        {"input": torch.rand(batch_size)},
        ModelOutput(),
        torch.ones((batch_size, 1)),
    )
    average_metric.update(sd)
    log_dict = average_metric.compute()
    assert (
        log_dict[average_metric.name] == 1
    ), "Average was not calculated correctly"
    average_metric.reset()
    sd = StepData(
        {"input": torch.rand(batch_size)},
        ModelOutput(),
        torch.ones((batch_size, 1)) * 2,
    )
    average_metric.update(sd)
    log_dict = average_metric.compute()
    assert (
        log_dict[average_metric.name] == 2
    ), "Average was not calculated correctly after reset"
