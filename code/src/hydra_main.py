from contextlib import contextmanager
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

    log = logging.getLogger(__name__)
    log.warning(cfg.dataset.dataset_root)

    if os.environ.get("DATA_DIR", None) is None:
        os.environ["DATA_DIR"] = cfg.paths.datasets

    # These transforms need to happen before the batching
    pre_data_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.input_shape),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
    # These transforms can be batched (on gpu)
    post_data_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(),
            transforms.GaussianBlur(5),
        ]
    )

    with create_trainer(
        cfg, pre_data_transforms, post_data_transforms
    ) as trainer:
        try_print_summary(cfg.input_shape, trainer.model)

        last_loss = trainer.steps(cfg.num_steps)
        hydra_config = hydra.core.hydra_config.HydraConfig.get()
        trainer.save(hydra_config.runtime.output_dir)

        # Run the final evaluation
        if cfg.get("eval_metric") is not None:
            eval_metric = hydra.utils.instantiate(cfg.eval_metric)
            score = trainer.full_eval(eval_metric)
            return score
        return last_loss


def try_print_summary(input_shape, model):
    try:
        import torchinfo

        torchinfo.summary(
            model, input_size=(3, *input_shape), batch_dim=0
        )
    except Exception as e:
        print(f"Could not generate torchinfo.summary because: {e}")


@contextmanager
def create_trainer(cfg, pre_data_transforms, post_data_transforms):
    # Load datasets
    # This needs to happen first as some settings are infered based on the dataset
    cfg, train_loader, val_loader = create_dataloaders(
        cfg, pre_data_transforms
    )

    # Create model and loss function
    model = hydra.utils.instantiate(
        cfg.model,
        label_channels=cfg.num_classes,
    )
    loss_fn = hydra.utils.instantiate(cfg.loss)

    optimizer, scheduler = create_optimizer(cfg, model)
    train_metrics, eval_metrics = create_metrics(cfg)

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

    yield trainer
    # Clean up trackers
    trainer.end_training()


def create_optimizer(cfg, model):
    optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )

    if cfg.scheduler is not None:
        scheduler = hydra.utils.instantiate(
            cfg.scheduler, optimizer=optimizer
        )
    else:
        scheduler = None
    return optimizer, scheduler


def create_metrics(cfg):
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
        train_metrics.extend(
            [metrics.ImageMetric("TrainReconstruction")]
        )
        eval_metrics.extend([metrics.ImageMetric("EvalReconstruction")])
    else:
        train_metrics.extend(
            [
                metrics.MaskMetric("TrainMask", dict(cfg.class_map)),
                metrics.ConfusionMetrics(
                    "ConfusionMetrics",
                    num_labels=cfg.num_classes,
                    ignore_index=cfg.ignore_index,
                    prefix="Train",
                ),
            ]
        )
        eval_metrics.extend(
            [
                metrics.MaskMetric("EvalMask", dict(cfg.class_map)),
                metrics.ConfusionMetrics(
                    "ConfusionMetrics",
                    num_labels=cfg.num_classes,
                    ignore_index=cfg.ignore_index,
                    prefix="Eval",
                ),
            ]
        )

    return train_metrics, eval_metrics


def create_dataloaders(cfg, pre_data_transforms):
    dataset_factory = hydra.utils.instantiate(
        cfg.dataset, _partial_=True
    )
    train_dataset = dataset_factory(
        split="train", transform=pre_data_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", os.cpu_count())),
        pin_memory=True,
    )
    val_dataset = dataset_factory(
        split="val", transform=pre_data_transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ.get("SLURM_NTASKS", os.cpu_count())),
        pin_memory=True,
    )

    # Set some 'global' information
    cfg.class_weights = train_dataset.class_weights
    cfg.class_map = train_dataset.class_map
    cfg.num_classes = len(cfg.class_map)
    cfg.ignore_index = train_dataset.ignore_index

    return cfg, train_loader, val_loader


if __name__ == "__main__":
    main()
