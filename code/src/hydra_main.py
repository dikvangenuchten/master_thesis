import logging
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
    if os.environ.get("DATA_DIR", None) is None:
        os.environ["DATA_DIR"] = cfg.paths.datasets

    input_shape = cfg.input_shape

    # These transforms need to happen before the batching
    pre_data_transforms = transforms.Compose(
        [
            transforms.Resize(input_shape),
        ]
    )
    # These transforms can be batched (on gpu)
    post_data_transforms = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(),
            transforms.GaussianBlur(5),
        ]
    )

    # Load datasets
    log = logging.getLogger(__name__)
    log.warning(cfg.dataset.dataset_root)
    dataset_factory = hydra.utils.instantiate(
        cfg.dataset, _partial_=True
    )
    train_dataset = dataset_factory(
        split="train", transform=pre_data_transforms
    )
    cfg.class_weights = train_dataset.class_weights
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", os.cpu_count() * 2)),
        pin_memory=True,
    )
    val_dataset = dataset_factory(
        split="val", transform=pre_data_transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", os.cpu_count() * 2)),
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
    ]

    eval_metrics = [
        metrics.AverageMetric(
            "EvalAverageLoss", lambda step_data: step_data.loss
        ),
        # metrics.ImageMetric("EvalReconstruction"),
    ]

    if cfg.dataset.output_structure.target == "img":
        train_metrics.append(metrics.ImageMetric("TrainReconstruction"))
        eval_metrics.append(metrics.ImageMetric("EvalReconstruction"))
    else:
        train_metrics.extend(
            [
                metrics.MaskMetric(
                    "TrainMask", train_dataset.class_map
                ),
                metrics.ConfusionMetrics(
                    "ConfusionMetrics",
                    num_classes,
                    ignore_index=train_dataset.ignore_index,
                    prefix="Train",
                ),
            ]
        )
        eval_metrics.extend(
            [
                metrics.MaskMetric("EvalMask", train_dataset.class_map),
                metrics.ConfusionMetrics(
                    "ConfusionMetrics",
                    num_classes,
                    ignore_index=train_dataset.ignore_index,
                    prefix="Eval",
                ),
            ]
        )

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
        data_transforms=post_data_transforms,
    )

    try:
        import torchinfo

        torchinfo.summary(
            model, input_size=(3, *input_shape), batch_dim=0
        )
    except Exception as e:
        print(f"Could not generate torchinfo.summary because: {e}")

    trainer.steps(cfg.num_steps)
    trainer.save(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    # TODO Return an tuple containing the


if __name__ == "__main__":
    main()
