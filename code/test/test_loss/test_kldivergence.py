import pytest
import torch
from torch import distributions, nn

from losses import HierarchicalKLDivergenceLoss


def test_loss_is_minimal_for_correct():
    loss_fn = HierarchicalKLDivergenceLoss()

    # Loss should be minimal when prior == posterior
    shape = 16, 8, 2, 2
    model_out = {
        "priors": [
            distributions.Normal(
                torch.zeros(*shape), torch.ones(*shape)
            )
        ],
        "posteriors": [
            distributions.Normal(
                torch.zeros(*shape), torch.ones(*shape)
            )
        ],
    }

    loss = loss_fn(model_out, ...)
    assert float(loss) == 0


def test_loss_is_bigger_if_incorrect():
    loss_fn = HierarchicalKLDivergenceLoss()

    # Loss should be minimal when prior == posterior
    shape = 16, 8, 2, 2
    target_dist = distributions.Normal(
        torch.zeros(*shape), torch.ones(*shape)
    )
    non_target_dist = distributions.Normal(
        torch.ones(*shape), 2 * torch.ones(*shape)
    )
    correct = {
        "priors": [target_dist],
        "posteriors": [target_dist],
    }
    incorrect = {
        "priors": [target_dist],
        "posteriors": [non_target_dist],
    }

    assert loss_fn(correct, ...) < loss_fn(incorrect, ...)


@pytest.mark.parametrize("target_mu,target_var", [(0, 1), (1, 0.1)])
def test_very_simple_model(device, target_mu, target_var):
    batch_size = 32
    shape = torch.tensor((1, 1, 1), device=device)
    hidden = shape.prod()
    model = nn.Sequential(
        nn.Linear(1, hidden),
        nn.Linear(hidden, hidden * 2),
    ).to(device=device)

    loss_fn = HierarchicalKLDivergenceLoss()

    optim = torch.optim.Adam(model.parameters())
    input_ = torch.ones([batch_size, 1], device=device)
    target_dist = distributions.Normal(
        target_mu * torch.ones(batch_size, *shape, device=device),
        target_var * torch.ones(batch_size, *shape, device=device),
    )

    losses = []
    init_mu, init_var_logits = model(input_).chunk(2, dim=1)
    init_var = nn.functional.softplus(init_var_logits)

    for i in range(10):
        optim.zero_grad()
        out = model(input_)
        mu, var_logits = out.chunk(2, dim=1)
        var = nn.functional.softplus(var_logits)
        cur_dist = distributions.Normal(
            mu.view(-1, *shape), var.view(-1, *shape)
        )
        loss = loss_fn(
            {"priors": [target_dist], "posteriors": [cur_dist]}, ...
        )
        loss.backward()
        optim.step()
        losses.append(loss.detach())
    # Although not guaranteed to always be the case, for this simple problem loss should always be decreasing.
    assert (
        (torch.tensor(losses)[:-1] - torch.tensor(losses)[1:]) > 0
    ).all()

    assert (
        (init_mu - target_mu).abs() > (mu - target_mu).abs()
    ).all(), "Mean diverged"
    assert (
        (init_var - target_var).abs() > (var - target_var).abs()
    ).all(), "Variance diverged"
