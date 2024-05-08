import pytest
from losses import AnnealingWeightedLoss


@pytest.mark.parametrize(
    "start_value,end_value,period", [(0, 1, 10), (1, 0, 10)]
)
def test_adheres_to_min_max_values(start_value, end_value, period):
    def loss_fn(*args, **kwargs):
        return 1

    weighted_loss = AnnealingWeightedLoss(
        loss_fn,
        start_value=start_value,
        end_value=end_value,
        max_step=period,
    )
    assert weighted_loss({"step": period}, ...) == start_value
    assert weighted_loss({"step": 0}, ...) == end_value

    assert min(
        weighted_loss({"step": i}, ...) for i in range(period * 4)
    ) == min(start_value, end_value)
    assert max(
        weighted_loss({"step": i}, ...) for i in range(period * 4)
    ) == max(start_value, end_value)
