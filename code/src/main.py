from tqdm import tqdm, trange
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from dotenv import load_dotenv
import segmentation_models_pytorch as smp


from datasets.oxford_pet import OxfordPetDataset
from datasets.toy_data import SegmentationToyDataset, OneColorBackground
from models.binary_segmentation_model import BinarySegmentationModel
from models.u_net import UNet
from trainer import Trainer
import metrics

DATA_ROOT = "/datasets/"

def main():
    print("Main")
    load_dotenv()
    batch_size = 64
    
    data_transforms = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    train_dataset = OxfordPetDataset(root=DATA_ROOT, mode="train", transform=data_transforms)
    val_dataset = OxfordPetDataset(root=DATA_ROOT, mode="valid", transform=data_transforms)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    model = UNet(3, 1)
    loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)
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
