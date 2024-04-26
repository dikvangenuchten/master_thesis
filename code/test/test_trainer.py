import pytest
import torch
from torch import nn, optim

from datasets.toy_data import SegmentationToyDataset
from trainer import Trainer


@pytest.fixture(scope="module")
def trainer(bs_model: nn.Module):
    dataset = SegmentationToyDataset(limit=1, base_path="test/data")
    bs_dataloader = dataset.to_loader(batch_size=1)

    return Trainer(
        bs_dataloader,
        bs_model,
        nn.MSELoss(),
        optim.Adam(bs_model.parameters()),
        # Default log_with is ["wandb"], but that needs to
        # be a singleton, which causes trouble in testing
        log_with=[],
    )


def test_trainer_single_epoch(image_batch, trainer):
    """Train a model on a single image."""
    input = image_batch.to(trainer.device)
    pre = trainer.model(input)
    pre_loss = trainer.epoch()
    for _ in range(4):
        trainer.epoch()
    post = trainer.model(input)
    post_loss = trainer.epoch()
    assert not (pre == post).all(), "The model was not updated"
    assert (
        pre_loss > post_loss
    ), "Loss increased for simple example training"


def test_trainer_eval_epoch(image_batch, trainer):
    input = image_batch.to(trainer.device)
    pre = trainer.model(input)
    _ = trainer.eval_epoch()
    post = trainer.model(input)
    assert (pre == post).all(), "Evaluation should not modify model"


@pytest.mark.xfail(reason="Saving the dataloader breaks somehow.")
def test_save_load(
    trainer: Trainer,
    tmp_path: str,
    image_batch: torch.Tensor,
):
    epoch = 0
    path = tmp_path / f"trainer-ckpt-{epoch}"
    trainer.save(path)

    loaded = Trainer.load(path)
    x = image_batch.to(device=trainer.device)

    assert torch.allclose(trainer.model(x), loaded.model(x))
