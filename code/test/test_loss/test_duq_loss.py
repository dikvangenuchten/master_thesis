import pytest
import torch

from losses.duq_loss import DUQLoss


@pytest.fixture()
def loss_fn():
    return DUQLoss()


def test_gradient_zero_for_correct(loss_fn):
    logits = torch.tensor(
        [
            [-100.0, 100.0],
        ],
        requires_grad=True,
    )
    y_pred = logits.softmax(-1)
    target = torch.tensor(
        [
            [
                0.0,
                1.0,
            ],
        ]
    )

    loss = loss_fn(y_pred, target)
    loss.backward()

    assert (
        logits.grad.sum() == 0
    ), "y_pred == target => This should result in zero grad"
    assert (
        loss.sum().item() == 0
    ), "For DUQ (BCE), loss value should be 0 for correct answers"


def test_gradient_non_zero_for_incorrect(loss_fn):
    y_pred = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
        ]
    )

    loss = loss_fn(y_pred, target)
    loss.backward()
    assert (
        -y_pred.grad[:, 0].item() > 0
    ), "idx: 0 should become bigger, but is not <0"
    assert (
        -y_pred.grad[:, 1].item() < 0
    ), "idx: 1 should become smaller, but is not >0"
