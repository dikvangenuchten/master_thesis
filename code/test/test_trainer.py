import pytest
from torch import nn, optim
from torch.utils import data

from trainer import Trainer
from data import SegmentationToyDataset


@pytest.fixture
def trainer(bs_model: nn.Module):
    dataset = SegmentationToyDataset(limit=1)
    bs_dataloader = dataset.to_loader(batch_size=1)

    return Trainer(
        bs_dataloader,
        bs_model,
        nn.MSELoss(),
        optim.Adam(bs_model.parameters()),
    )


def test_trainer_single_epoch(test_image_batch, trainer):
    """Train a model on a single image."""
    input = test_image_batch.to(trainer.device)
    pre = trainer.model(input)
    pre_loss = trainer.epoch()
    post = trainer.model(input)
    post_loss = trainer.epoch()
    assert not (pre == post).all(), "The model was not updated"
    assert pre_loss > post_loss, "Loss increased for simple example training"


def test_trainer_eval_epoch(test_image_batch, trainer):
    input = test_image_batch.to(trainer.device)
    pre = trainer.model(input)
    pre_loss = trainer.eval_epoch()
    post = trainer.model(input)
    assert (pre == post).all(), "Evaluation should not modify model"
