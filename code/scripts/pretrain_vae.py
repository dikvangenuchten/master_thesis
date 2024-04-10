import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

import utils  # noqa

import losses
from models.semantic_vae import SemanticVAE
from datasets.coco import CoCoDataset
from trainer import Trainer


def train(
    model: SemanticVAE,
    train_dataset: CoCoDataset,
    val_dataset: CoCoDataset,
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

    loss_fn = losses.KLDivergence()

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

    train_dataset = CoCoDataset(
        "train",
        transform=data_transforms,
        output_structure={"input": "img", "target": "semantic_mask"},
    )
    val_dataset = CoCoDataset(
        "val",
        transform=data_transforms,
        output_structure={"input": "img", "target": "semantic_mask"},
    )

    model = SemanticVAE(
        3,
        len(train_dataset.class_map),
        [8, 16, 64],
        [2, 2, 2],
        [1.0, 0.5, 0.5]
    )

    train(model, train_dataset, val_dataset)
