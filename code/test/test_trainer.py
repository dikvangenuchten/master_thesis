import pytest
from torch import nn, optim
from torch.utils import data

from trainer import Trainer
from data import SegmentationToyDataset

@pytest.fixture
def trainer(bs_model: nn.Module):
    dataset = SegmentationToyDataset(limit=1)
    bs_dataloader = dataset.to_loader(batch_size=1)

    return Trainer(bs_dataloader, bs_model, nn.MSELoss(), optim.Adam(bs_model.parameters()))


def test_trainer_single_epoch(test_image_batch, trainer):
    """Train a model on a single image."""
    # TODO verify the model is better after epoch
    pre = trainer.model(test_image_batch)
    trainer.epoch()
    post = trainer.model(test_image_batch)
    assert not (pre == post).all()
