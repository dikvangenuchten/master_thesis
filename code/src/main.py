from typing import Callable
import segmentation_models_pytorch as smp
import torch
from dotenv import load_dotenv
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from tqdm import trange


import metrics
from datasets.coco import CoCoDataset
from datasets.fiftyone import FiftyOneDataset
from models import ModelOutput, UNet, VAE
from trainer import Trainer

DATA_ROOT = "/datasets/"


def main():
    print("Main")
    load_dotenv()
    batch_size = 64

    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    data_transforms = transforms.Compose(
        [transforms.Resize((64, 64)), *image_net_transforms]
    )

    train_dataset = FiftyOneDataset(
        split="train",
        transform=data_transforms,
        # latents=True,
        output_structure={"input": "latent", "image": "img", "target": "semantic_mask"},
    )
    val_dataset = FiftyOneDataset(
        split="validation",
        transform=data_transforms,
        # latents=True,
        output_structure={"input": "latent", "image": "img", "target": "semantic_mask"},
    )

    # train_dataset = OxfordSpeciesDataset(
    #   root=DATA_ROOT, mode="train", transform=data_transforms
    # )
    # val_dataset = OxfordSpeciesDataset(
    #   root=DATA_ROOT, mode="valid", transform=data_transforms
    # )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1)

    num_classes = len(train_dataset.class_map)
    ignore_index = (
        train_dataset.ignore_index if hasattr(train_dataset, "ignore_index") else None
    )
    model = VAE(3, num_classes)
    mode = "binary" if num_classes == 2 else "multiclass"

    def wrapped_loss(
        loss_fn, from_logits: bool
    ) -> Callable[[ModelOutput, torch.Tensor], torch.Tensor]:
        def _inner(model_out: ModelOutput, target: torch.Tensor) -> torch.Tensor:
            if from_logits and model_out.logits is not None:
                return loss_fn(model_out.logits, target)
            return loss_fn(model_out.out, target)

        return _inner

    loss_fn = wrapped_loss(
        smp.losses.DiceLoss(mode=mode, from_logits=False, ignore_index=ignore_index),
        from_logits=False,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_metrics = [
        metrics.AverageMetric("TrainAverageLoss", lambda step_data: step_data.loss),
        metrics.MaskMetric("TrainMaskMetric", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics", num_classes, ignore_index=ignore_index, prefix="Train"
        ),
    ]

    eval_metrics = [
        metrics.AverageMetric("EvalAverageLoss", lambda step_data: step_data.loss),
        metrics.MaskMetric("EvalMaskMetric", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics", num_classes, ignore_index=ignore_index, prefix="Eval"
        ),
    ]

    trainer = Trainer(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        eval_dataloader=eval_dataloader,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )

    trainer.steps(1_000_000, eval_every_n_steps=100)


if __name__ == "__main__":
    main()

## Steps:
# Define a model based on (V)AE -- Done
# Create a cache creation script -- Done
# Create a training script -- Testing
# Create a inference script
# Also with infinite sampling


### TODO LIST
# 1. Make experimental design specific for new idea
#       Be explicit in what is "cached"
#       Be explicit about what metrics


# Add github repo of group
