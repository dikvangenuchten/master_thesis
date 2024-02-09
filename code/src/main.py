from tqdm import tqdm, trange
from torch import optim, nn
from torchvision.transforms import v2 as transforms

from data import SegmentationToyDataset
from model import BinarySegmentationModel

def main():
    dataset = SegmentationToyDataset()
    dataloader = dataset.to_loader()

    model = BinarySegmentationModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    pbar = trange(0, 100)
    for epoch in pbar:
        loss_sum = 0
        loss_count = 0
        for batch_idx, (img, target) in enumerate(tqdm(dataloader, leave=False)):
            pred = model(img)
            loss = loss_fn(pred, target)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Keep track of average loss
            loss_d = loss.detach()
            loss_sum += loss_d.sum()
            loss_count += img.shape[0]
        pbar.set_description(f"avg_loss: {loss_sum/loss_count}")



if __name__ == "__main__":
    main()
