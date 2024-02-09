from tqdm import tqdm, trange
from torch import optim, nn
from torchvision.transforms import v2 as transforms

from data import SegmentationToyDataset
from model import BinarySegmentationModel
from trainer import Trainer

def main():
    dataset = SegmentationToyDataset()
    dataloader = dataset.to_loader()

    model = BinarySegmentationModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    trainer = Trainer(
        dataloader, model, loss_fn, optimizer
    )

    for epoch in (pbar := trange(0, 100)):
        loss = trainer.epoch()
        pbar.set_description(f"avg_loss: {loss}")



if __name__ == "__main__":
    main()
