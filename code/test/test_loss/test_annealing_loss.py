import pytest
from losses import AnnealingWeightedLoss



@pytest.mark.parametrize("min_w,max_w,period",
                         [(0, 1, 10)])
def test_adheres_to_min_max_values(min_w, max_w, period):
    def loss_fn(*args, **kwargs):
        return 1

    weighted_loss = AnnealingWeightedLoss(
        loss_fn, eta_min=min_w, eta_max=max_w, max_step=period
    )
    assert weighted_loss({"step": period}, ...) == min_w
    assert weighted_loss({"step": 0}, ...) == max_w
    
    assert min(weighted_loss({"step": i}, ...) for i in range(period * 2)) == min_w
    assert max(weighted_loss({"step": i}, ...) for i in range(period * 2)) == max_w
