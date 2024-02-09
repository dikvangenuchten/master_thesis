import pytest
from torch import nn
from torch.utils import data

from trainer import Trainer


@pytest.fixture
def trainer(bs_model: nn.Module):
    bs_dataloader = ...

    return Trainer(bs_dataloader, bs_model, nn.MSELoss)


def test_trainer_single_epoch(test_image_batch, trainer):
    """Train a model on a single image."""
    pre = trainer.model(test_image_batch)
    trainer.epoch()
    post = trainer.model(test_image_batch)
    assert not (pre == post).all()
