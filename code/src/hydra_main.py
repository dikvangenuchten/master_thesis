import os
import hydra
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from omegaconf import DictConfig, OmegaConf

import metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    image_net_transforms = [
        # Rescale to [0, 1]
        transforms.ToDtype(torch.float32, scale=True),
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
    cfg.general.class_weights = train_dataset.class_weights
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

    num_classes = len(train_dataset.class_map)

    loss_fn = hydra.utils.instantiate(cfg.loss)

    model = hydra.utils.instantiate(
        cfg.model,
        label_channels=num_classes,
    )

    optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )

    if cfg.scheduler is not None:
        scheduler = hydra.utils.instantiate(
            cfg.scheduler, optimizer=optimizer
        )
    else:
        scheduler = None

    train_metrics = [
        metrics.AverageMetric(
            "TrainAverageLoss", lambda step_data: step_data.loss
        ),
        # metrics.ImageMetric("TrainReconstruction"),
        metrics.MaskMetric("TrainMask", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics",
            num_classes,
            ignore_index=train_dataset.ignore_index,
            prefix="Train",
        ),
    ]

    eval_metrics = [
        metrics.AverageMetric(
            "EvalAverageLoss", lambda step_data: step_data.loss
        ),
        # metrics.ImageMetric("EvalReconstruction"),
        metrics.MaskMetric("EvalMask", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics",
            num_classes,
            ignore_index=train_dataset.ignore_index,
            prefix="Eval",
        ),
    ]

    # Required to be able to use config as kwarg.
    trainer_factory = hydra.utils.instantiate(
        cfg.trainer, _partial_=True
    )
    trainer = trainer_factory(
        train_dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
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
    trainer.save(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )


if __name__ == "__main__":
    main()
