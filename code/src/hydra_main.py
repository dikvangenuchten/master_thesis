from contextlib import contextmanager
import logging
import os
import hydra
import torch
from torchvision.transforms import v2 as transforms

from omegaconf import DictConfig, OmegaConf

import losses
import metrics
import datasets


def uint8_to_long(batch):
    batch["target"] = batch["target"].to(dtype=torch.long)
    return batch


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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomGrayscale(),
        ]
    )
    extra = (
        [uint8_to_long]
        if cfg.dataset.output_structure.target == "semantic_mask"
        else []
    )
    # These transforms can be batched (on gpu)
    post_data_transforms = transforms.Compose(
        [
            transforms.GaussianBlur(5, (0.01, 2.0)),
            *extra,
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
            if cfg.dataset.output_structure.target == "img":
                l1_loss = losses.WrappedLoss(
                    torch.nn.L1Loss(),
                    keys={"out": "input", "input": "target"},
                )
                eval_metric = metrics.AverageMetric(
                    "L1-Loss",
                    lambda step_data: l1_loss(
                        step_data.model_out, step_data.batch
                    ),
                )
            else:
                eval_metric = metrics.ConfusionMetrics(
                    "ConfusionMetrics",
                    num_labels=cfg.num_classes,
                    ignore_index=cfg.ignore_index,
                    include=["Jaccard Index"],
                )
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
    cfg, train_loader, val_loader = datasets.create_dataloaders(
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


if __name__ == "__main__":
    main()
