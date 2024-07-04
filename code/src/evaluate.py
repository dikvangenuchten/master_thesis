"""Run an evaluation for a given model

Creates some extra outputs
    - Visualization of latent-space (if present)
    - Visualization of feature-maps
"""
import logging
import os
import hydra
from accelerate import Accelerator
import torch
from torchvision.transforms import v2 as transforms

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import datasets
import losses
import metrics
from metrics.base_metric import StepData
from utils.visualize_feature_maps import (
    visualize_filters,
    visualize_posteriors,
    visualize_encoder_features,
)


def uint8_to_long(batch):
    batch["target"] = batch["target"].to(dtype=torch.long)
    return batch


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # TODO make sure these are somewhat unique
    output_dir = "evaluation-graphs/vae"
    os.makedirs(output_dir, exist_ok=True)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    log = logging.getLogger(__name__)
    log.warning(cfg.dataset.dataset_root)

    if os.environ.get("DATA_DIR", None) is None:
        os.environ["DATA_DIR"] = cfg.paths.datasets

    data_transforms = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToDtype(torch.float32, scale=True),
            uint8_to_long,
        ]
    )

    cfg, _, val_dataloader = datasets.create_dataloaders(
        cfg, data_transforms
    )
    model = hydra.utils.instantiate(
        cfg.model,
        label_channels=cfg.num_classes,
    )
    model.eval()

    loss_fn = hydra.utils.instantiate(cfg.loss)

    if cfg.dataset.output_structure.target == "img":
        l1_loss = losses.WrappedLoss(
            torch.nn.L1Loss,
            keys={"out": "input", "input": "target"},
        )
        metric = metrics.AverageMetric(
            "L1-Loss",
            lambda step_data: l1_loss(
                step_data.model_out, step_data.batch
            ),
        )
    else:
        metric = metrics.ConfusionMetrics(
            "ConfusionMetrics",
            num_labels=cfg.num_classes,
            ignore_index=cfg.ignore_index,
            include=["Jaccard Index"],
        )

    accel = Accelerator()
    model, val_dataloader, metric, loss_fn = accel.prepare(
        model, val_dataloader, metric, loss_fn
    )

    visualize_filters(model, 30)

    with torch.no_grad():
        for i, batch in enumerate(
            tqdm(val_dataloader, desc="Evaluating")
        ):
            if i == 0:
                _posterior_visualizations = visualize_posteriors(
                    model, batch, f"{output_dir}/posteriors/"
                )
                _encoder_feature_visualizations = (
                    visualize_encoder_features(
                        model, batch, f"{output_dir}/features/"
                    )
                )
            model_out = model(batch["input"])
            loss = loss_fn(model_out, batch)
            metric.update(
                StepData(batch=batch, model_out=model_out, loss=loss)
            )
        result = metric.compute()
    log.info(f"Evaluation result: {result}")


if __name__ == "__main__":
    main()
