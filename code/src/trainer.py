from torch import nn, optim
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        dataloader: nn.Module,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer
    ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def epoch(self):
        loss_sum = 0
        loss_count = 0
        for batch_idx, (img, target) in enumerate(tqdm(self.dataloader, leave=False)):
            pred = self.model(img)
            loss = self.loss_fn(pred, target)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Keep track of average loss
            loss_d = loss.detach()
            loss_sum += loss_d.sum()
            loss_count += img.shape[0]

        return loss_sum / loss_count