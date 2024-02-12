from typing import Optional
import logging

import torch
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm
from accelerate import Accelerator


class Trainer:
    def __init__(
        self,
        train_dataloader: data.DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        eval_dataloader: Optional[data.DataLoader] = None,
    ) -> None:
        self._accelerator = Accelerator(log_with="wandb")

        if scheduler is None:
            # If no scheduler is given create a 'constant' scheduler
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        if eval_dataloader is None:
            logging.warning("No eval data provided, Using train data as eval data")
            eval_dataloader = train_dataloader

        model, optimizer, train_dataloader, eval_dataloader, scheduler = (
            self._accelerator.prepare(
                model, optimizer, train_dataloader, eval_dataloader, scheduler
            )
        )

        self.train_dataloader = train_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_dataloader = eval_dataloader

    @property
    def device(self):
        return self._accelerator.device

    def epoch(self) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0
        for batch_idx, (img, target) in enumerate(
            tqdm(self.train_dataloader, leave=False)
        ):
            self.optimizer.zero_grad()

            pred = self.model(img)
            loss = self.loss_fn(pred, target)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Keep track of average loss
            loss_d = loss.detach()
            loss_sum += loss_d.sum()
            loss_count += img.shape[0]

        self.scheduler.step()
        return loss_sum / loss_count

    def eval_epoch(self) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0

        with torch.no_grad():
            for batch_idx, (img, target) in enumerate(
                tqdm(self.train_dataloader, leave=False)
            ):
                pred = self.model(img)
                loss = self.loss_fn(pred, target)

                # Keep track of average loss
                loss_d = loss.detach()
                loss_sum += loss_d.sum()
                loss_count += img.shape[0]

        return loss_sum / loss_count
