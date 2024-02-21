import segmentation_models_pytorch as smp
import torch
from dotenv import load_dotenv
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from tqdm import trange

import metrics
from datasets.oxford_pet import (
    OxfordSpeciesDataset,
)
from models.u_net import UNet
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
        [transforms.Resize((128, 128)), *image_net_transforms]
    )

    train_dataset = OxfordSpeciesDataset(
        root=DATA_ROOT, mode="train", transform=data_transforms
    )
    val_dataset = OxfordSpeciesDataset(
        root=DATA_ROOT, mode="valid", transform=data_transforms
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    num_classes = len(train_dataset.class_map)
    model = UNet(3, num_classes)
    mode = "binary" if num_classes == 2 else "multiclass"
    loss_fn = smp.losses.DiceLoss(mode=mode, from_logits=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    run_metrics = [
        metrics.AverageMetric("AverageLoss", lambda x, y_t, y_p, loss: loss),
        metrics.MaskMetric("MaskMetric", {1: "blob"}),
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
    pass


if __name__ == "__main__":
    main()
