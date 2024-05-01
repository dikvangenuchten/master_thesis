import pytest
import torch
from torch import nn, optim, utils

import losses
from models import MobileVAE
from trainer import Trainer


@pytest.fixture(scope="module")
def trainer(dataset, device):
    dataloader = utils.data.DataLoader(dataset, batch_size=16)

    loss_fn = losses.WrappedLoss(
        nn.CrossEntropyLoss(
            weight=torch.tensor(dataset.class_weights, device=device),
            ignore_index=133,
        )
    )

    model = MobileVAE(
        len(dataset.class_map),
        encoder_depth = 1
    )

    return Trainer(
        dataloader,
        model,
        loss_fn,
        optim.Adam(model.parameters()),
        # Default log_with is ["wandb"], but that needs to
        # be a singleton, which causes trouble in testing
        log_with=[],
    )


def test_trainer_single_epoch(image_batch, trainer):
    """Train a model on a single image."""
    input = image_batch.to(trainer.device)
    pre = trainer.model.eval()(input)["out"]
    pre2 = trainer.model.eval()(input)["out"]
    assert (pre == pre2).all(), "For this test to work model should be determenistic in eval mode"
    pre_loss = trainer.epoch()
    for _ in range(1):
        trainer.epoch()
    post = trainer.model.eval()(input)["out"]
    post_loss = trainer.epoch()
    assert not (pre == post).all(), "The model was not updated"
    assert (
        pre_loss > post_loss
    ), "Loss increased for simple example training"


def test_trainer_eval_epoch(image_batch, trainer):
    input = image_batch.to(trainer.device)
    pre = trainer.model.eval()(input)["out"]
    pre2 = trainer.model.eval()(input)["out"]
    assert (pre == pre2).all(), "For this test to work model should be determenistic in eval mode"
    trainer.model.train() # This ensures eval must be called in trainer eval epoch
    _ = trainer.eval_epoch()
    post = trainer.model(input)["out"]
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
