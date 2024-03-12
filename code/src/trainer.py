import logging
from typing import Dict, List, Optional

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from metrics.base_metric import BaseMetric, StepData
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

    def _metrics_add_batch(self, step_data: StepData):
        for metric in self.metrics:
            metric.update(step_data)

    def _log_and_reset_metrics(self, prefix: str = "", epoch: Optional[int] = None):
        log_dict = {}
        for metric in self.metrics:
            log_dict.update(**metric.compute())
            metric.reset()

        log_dict = {f"{prefix}_{k}": v for k, v in log_dict.items()}
        self._accelerator.log(log_dict, step=epoch)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.optimizer.zero_grad()
        # Forward pass
        input = batch["input"]
        input.requires_grad_(True)
        model_out = self.model(input)
        # Calculate Loss
        target = batch["target"]
        loss = self.loss_fn(model_out, target)
        # Backward
        self._accelerator.backward(loss)
        self.optimizer.step()
        step_data = StepData(batch, model_out, loss)
        self._metrics_add_batch(step_data)
        return loss.detach()

    @torch.no_grad
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_out = self.model(batch["input"])
        loss = self.loss_fn(model_out, batch["target"])
        step_data = StepData(batch, model_out, loss)
        self._metrics_add_batch(step_data)
        return loss.detach()

    def steps(
        self,
        training_steps: int,
        eval_every_n_steps: int = 100,
        log_every_n_steps: Optional[int] = None,
    ):
        """Run `n` training steps

        Args:
            training_steps (int): The total number of training steps to run
            eval_every_n_steps (int, optional): The amount of training steps for every eval step. Defaults to 100.
            log_every_n_steps (Optional[int], optional): _description_. Defaults to None.
        """
        log_every_n_steps = (
            eval_every_n_steps if log_every_n_steps is None else log_every_n_steps
        )
        iter_train = _Repeater(self.train_dataloader)
        iter_eval = _Repeater(self.eval_dataloader)
        for i in trange(training_steps):
            self.train_step(next(iter_train))
            if i % eval_every_n_steps == 0:
                self.eval_step(next(iter_eval))
            if i % log_every_n_steps == 0:
                self._log_and_reset_metrics()

    def epoch(self, epoch: Optional[int] = None) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0
        self.model.train()

        stop = False
        pbar = tqdm(self.train_dataloader, leave=False, desc="training")
        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            input = batch["input"]
            input.requires_grad_(True)
            target = batch["target"]

            model_out = self.model(input)

            loss = self.loss_fn(model_out, target)

            # Gradient penalty
            # Gradient penalty is broken for VAE
            # gp = self._gradient_penalty(input, output)
            # loss += gp

            # Backpropagation
            self._accelerator.backward(loss)
            self.optimizer.step()

            # Keep track of average loss
            loss_d = loss.detach()
            loss_sum += loss_d.sum()
            loss_count += input.shape[0]
            pbar.set_description(f"Training: loss={(loss_sum / loss_count).item():.4f}")

            step_data = StepData(batch, model_out, loss)
            self._metrics_add_batch(step_data)
            if stop:
                # Manual break using debugger
                break

        self.scheduler.step()
        self._log_and_reset_metrics("train", epoch)

        return (loss_sum / loss_count).item()

    def eval_epoch(self, epoch: Optional[int] = None) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(self.eval_dataloader, leave=False, desc="evaluation")
            ):
                input = batch["input"]
                input.requires_grad_(True)
                target = batch["target"]

                model_out = self.model(input)

                loss = self.loss_fn(model_out, target)

                # Keep track of average loss
                loss_d = loss.detach()
                loss_sum += loss_d.sum()
                loss_count += input.shape[0]

                step_data = StepData(batch, model_out, loss)
                self._metrics_add_batch(step_data)

        self._log_and_reset_metrics("eval", epoch)
        return (loss_sum / loss_count).item()


class _Repeater:
    def __init__(self, dataloader: data.DataLoader) -> None:
        self.count = 0
        self._dataloader = dataloader
        self._iterable = iter(self._dataloader)

    def __iter__(self):
        return self

    def __next__(self) -> Dict:
        try:
            return next(self._iterable)
        except StopIteration:
            self.count += 1
            self._iterable = iter(self._dataloader)
            return next(self._iterable)
