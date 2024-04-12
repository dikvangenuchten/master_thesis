import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchinfo import summary

import utils  # noqa

import losses
from models.semantic_vae import SemanticVAE
import datasets
from trainer import Trainer


def train(
    model: SemanticVAE,
    train_dataset: datasets.CoCoDataset,
    val_dataset: datasets.CoCoDataset,
):
    optimizer = torch.optim.Adamax(
        model.parameters(),
        # Taken from Efficient_VDVAE
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=750000, eta_min=2e-4
    )

    # TODO: Add the semantic loss
    # TODO: Create a weighted loss combiner
    loss_fn = losses.SummedLoss(
        losses=[
            losses.WeightedLoss(losses.KLDivergence(), 0.1),
            losses.WeightedLoss(
                losses.WrappedLoss(
                    nn.CrossEntropyLoss(
                        weight=torch.tensor(
                            train_dataset.class_weights, device="cuda"
                        ),
                        ignore_index=133,
                    )
                ),
                1,
            ),
        ]
    )

    trainer = Trainer(
        DataLoader(train_dataset, batch_size=64),
        model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=schedule,
        eval_dataloader=DataLoader(val_dataset, batch_size=64),
        config={},  # TODO ensure all values are hparams
        train_metrics=[],
        eval_metrics=[],
    )

    trainer.steps(1000)


if __name__ == "__main__":
    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]

    data_transforms = transforms.Compose(
        [transforms.Resize((64, 64)), *image_net_transforms]
    )

    train_dataset = datasets.CoCoDataset(
        "train",
        transform=data_transforms,
        output_structure={"input": "img", "target": "semantic_mask"},
    )
    val_dataset = datasets.CoCoDataset(
        "val",
        transform=data_transforms,
        output_structure={"input": "img", "target": "semantic_mask"},
    )

    model = SemanticVAE(
        3,
        len(train_dataset.class_map),
        [64, 64, 128, 128, 128, 512, 512],
        [2, 1, 2, 1, 2, 1, 1],
        [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
    )

    summary(model.to("cuda"), (1, 3, 256, 256))
    train(model, train_dataset, val_dataset)
