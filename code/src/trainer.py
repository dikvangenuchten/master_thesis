from torch import nn


class Trainer:
    def __init__(
        self,
        dataloader: nn.Module,
        model: nn.Module,
        loss_fn: nn.Module,
    ) -> None:
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn

    def epoch(self):
        pass
