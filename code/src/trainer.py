import logging
from typing import List, Optional

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm

from metrics.base_metric import BaseMetric
from losses.gradient_penalty import GradientPenalty


class Runner:
    def __init__(
        self,
        dataloader: data.DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
    ) -> None:
        pass


class Trainer:
    def __init__(
        self,
        train_dataloader: data.DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        eval_dataloader: Optional[data.DataLoader] = None,
        config: dict = {},
        metrics: List[BaseMetric] = [],
    ) -> None:
        self._accelerator = Accelerator(log_with="wandb")
        self._accelerator.init_trackers(project_name="MasterThesis", config=config)

        if scheduler is None:
            # If no scheduler is given create a 'constant' scheduler
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        if eval_dataloader is None:
            logging.warning("No eval data provided, Using train data as eval data")
            eval_dataloader = train_dataloader

        (
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
        ) = self._accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
        )

        self.train_dataloader = train_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_dataloader = eval_dataloader
        self.metrics = metrics
        self._gradient_penalty = GradientPenalty()

    @property
    def device(self):
        return self._accelerator.device

    def _metrics_add_batch(self, *args, **kwargs):
        for metric in self.metrics:
            metric.update(*args, **kwargs)

    def _log_and_reset_metrics(self, prefix: str = "", epoch: Optional[int] = None):
        log_dict = {}
        for metric in self.metrics:
            log_dict.update(**metric.compute())
            metric.reset()

        log_dict = {f"{prefix}_{k}": v for k, v in log_dict.items()}
        self._accelerator.log(log_dict, step=epoch)

    def epoch(self, epoch: Optional[int] = None) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0
        self.model.train()

        pbar = tqdm(self.train_dataloader, leave=False, desc="training")
        for batch_idx, (img, target) in enumerate(pbar):
            self.optimizer.zero_grad()

            img.requires_grad_(True)  # Is required for the gp

            output = self.model(img)
            loss = self.loss_fn(output, target)

            # Gradient penalty
            gp = self._gradient_penalty(img, output)
            loss += gp

            # Backpropagation
            self._accelerator.backward(loss)
            self.optimizer.step()

            # Keep track of average loss
            loss_d = loss.detach()
            loss_sum += loss_d.sum()
            loss_count += img.shape[0]
            pbar.set_description(f"Training: loss={(loss_sum / loss_count).item():.4f}")

            self._metrics_add_batch(img, target, output.softmax(1), loss_d)

        self.scheduler.step()
        self._log_and_reset_metrics("train", epoch)

        return (loss_sum / loss_count).item()

    def eval_epoch(self, epoch: Optional[int] = None) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (img, target) in enumerate(
                tqdm(self.eval_dataloader, leave=False, desc="evaluation")
            ):
                logits = self.model(img)
                loss = self.loss_fn(logits, target)

                # Keep track of average loss
                loss_d = loss.detach()
                loss_sum += loss_d.sum()
                loss_count += img.shape[0]

                self._metrics_add_batch(img, target, logits.softmax(1), loss_d)

        self._log_and_reset_metrics("eval", epoch)
        return (loss_sum / loss_count).item()
