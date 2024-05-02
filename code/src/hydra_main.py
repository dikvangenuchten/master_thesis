import os
import hydra
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from omegaconf import DictConfig, OmegaConf

import metrics
from trainer import Trainer


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    loss_fn = hydra.utils.instantiate(cfg.loss)

    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]

    input_shape = cfg.general.input_shape
    data_transforms = transforms.Compose(
        [transforms.Resize(input_shape), *image_net_transforms]
    )

    # Load datasets
    dataset_factory = hydra.utils.instantiate(
        cfg.dataset, _partial_=True
    )
    train_dataset = dataset_factory(
        split="train", transform=data_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.general.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", 4)),
        pin_memory=True,
    )
    val_dataset = dataset_factory(
        split="val", transform=data_transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.general.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", 4)),
        pin_memory=True,
    )

    ignore_index = num_classes = len(train_dataset.class_map)

    model = hydra.utils.instantiate(
        cfg.model,
        label_channels=num_classes,
    )

    optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )

    train_metrics = [
        metrics.AverageMetric(
            "TrainAverageLoss", lambda step_data: step_data.loss
        ),
        metrics.MaskMetric("TrainMaskMetric", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics",
            num_classes,
            ignore_index=ignore_index,
            prefix="Train",
        ),
    ]

    eval_metrics = [
        metrics.AverageMetric(
            "EvalAverageLoss", lambda step_data: step_data.loss
        ),
        metrics.MaskMetric("EvalMaskMetric", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics",
            num_classes,
            ignore_index=ignore_index,
            prefix="Eval",
        ),
    ]

    trainer = Trainer(
        train_dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_dataloader=val_loader,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        import torchinfo

        torchinfo.summary(
            model, input_size=(3, *input_shape), batch_dim=0
        )
    except Exception as e:
        print(f"Could not generate torchinfo.summary because: {e}")

    trainer.steps(cfg.general.num_steps)


if __name__ == "__main__":
    main()