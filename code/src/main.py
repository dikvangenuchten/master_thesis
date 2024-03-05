import segmentation_models_pytorch as smp
import torch
from dotenv import load_dotenv
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from tqdm import trange


import metrics
from datasets.coco import CoCoDataset
from models.u_net import UNet
from trainer import Trainer

DATA_ROOT = "/datasets/"


def main():
    print("Main")
    load_dotenv()
    batch_size = 8

    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    data_transforms = transforms.Compose(
        [transforms.Resize((128, 128)), *image_net_transforms]
    )

    train_dataset = CoCoDataset(split="train", transform=data_transforms)
    val_dataset = CoCoDataset(split="val", transform=data_transforms)

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
    model = UNet(3, num_classes)
    mode = "binary" if num_classes == 2 else "multiclass"
    loss_fn = smp.losses.DiceLoss(
        mode=mode, from_logits=False, ignore_index=ignore_index
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    run_metrics = [
        metrics.AverageMetric("AverageLoss", lambda x, y_t, y_p, loss: loss),
        metrics.MaskMetric("MaskMetric", train_dataset.class_map),
        metrics.ConfusionMetrics(
            "ConfusionMetrics", num_classes, ignore_index=ignore_index
        ),
    ]

    trainer = Trainer(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        eval_dataloader=eval_dataloader,
        metrics=run_metrics,
    )

    for epoch in (pbar := trange(0, 100)):
        loss = trainer.epoch(epoch)
        eval_loss = trainer.eval_epoch(epoch)
        pbar.set_description(f"train_avg_loss: {loss}, eval_avg_loss: {eval_loss}")


if __name__ == "__main__":
    main()

## Steps:
# Define a model based on (V)AE
# Create a cache creation script
# Create a training script
# Create a inference script
#


### TODO LIST
# 1. Make experimental design specific for new idea
#       Be explicit in what is "cached"
#       Be explicit about what metrics
