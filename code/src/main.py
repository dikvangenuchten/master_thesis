from tqdm import tqdm, trange
from torch import optim, nn

from dotenv import load_dotenv


from data import SegmentationToyDataset
from model import BinarySegmentationModel
from trainer import Trainer
import metrics


def main():
    load_dotenv()

    train_dataset = SegmentationToyDataset(split="train", limit=-1)
    train_dataloader = train_dataset.to_loader(batch_size=256)

    eval_dataset = SegmentationToyDataset(split="val", limit=-1)
    eval_dataloader = eval_dataset.to_loader(batch_size=256)

    model = BinarySegmentationModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    run_metrics = [metrics.MaskMetric("MaskMetric", {1: "blob"})]

    trainer = Trainer(
        train_dataloader,
        model,
        loss_fn,
        optimizer,
        eval_dataloader=eval_dataloader,
        metrics=run_metrics,
    )

    for epoch in (pbar := trange(0, 100)):
        loss = trainer.epoch()
        eval_loss = trainer.eval_epoch()
        pbar.set_description(f"train_avg_loss: {loss}, eval_avg_loss: {eval_loss}")


if __name__ == "__main__":
    main()
