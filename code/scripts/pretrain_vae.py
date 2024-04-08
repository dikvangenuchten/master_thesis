import torch
from models.semantic_vae import SemanticVAE
from datasets.coco import CoCoDataset
from trainer import Trainer


def train(
    model: SemanticVAE,
    train_dataset: CoCoDataset,
    val_dataset: CoCoDataset,
):
    optimizer = torch.optim.Adamax(
        # Taken from Efficient_VDVAE
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=750000, eta_min=2e-4
    )

    Trainer(
        train_dataset,
        model,
        loss_fn=...,
        optimizer=optimizer,
        scheduler=schedule,
        eval_dataloader=val_dataset,
        config={},  # TODO ensure all values are hparams
        train_metrics=[],
        eval_metrics=[],
    )

    pass


if __name__ == "__main__":
    train_dataset = CoCoDataset(
        "train", output_structure={"input": "img", "target": "img"}
    )
    val_dataset = CoCoDataset(
        "val", output_structure={"input": "img", "target": "img"}
    )

    model = SemanticVAE(3, None, [8, 16, 64], [2, 3, 3])

    train(model, train_dataset, val_dataset)
