import datetime
import os
import itertools
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


class RunningMean:
    def __init__(self, alpha=0.99, start: Optional[torch.Tensor] = None) -> None:
        self._alpha = alpha
        self._val = start
        self.last = None

    @torch.no_grad
    def add(self, val: torch.Tensor) -> torch.Tensor:
        self.last = val
        if self._val is None:
            self._val = val
        self._val = self._alpha * self._val + (1 - self._alpha) * val
        return self._val

    @property
    def val(self) -> torch.Tensor:
        return self._val

class Constant:
    def __init__(self, val) -> None:
        self._val = val

    def __call__(self, *args, **kwargs):
        return self._val


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
        train_metrics: List[BaseMetric] = None,
        eval_metrics: List[BaseMetric] = None,
        log_with: List[str] = ["wandb"]
    ) -> None:
        train_metrics = [] if train_metrics is None else train_metrics
        eval_metrics = [] if eval_metrics is None else eval_metrics

        self._accelerator = Accelerator(log_with=log_with)
        self._accelerator.init_trackers(project_name="MasterThesis", config=config)

        if scheduler is None:
            # If no scheduler is given create a 'constant' scheduler
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, Constant(1.0))

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
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self._gradient_penalty = GradientPenalty(),

        # Save based on start time of run
        self._ckpt_dir = os.path.join("ckpts", datetime.datetime.now().strftime("%Y/%m/%d-%H:%M"))

    @property
    def device(self):
        return self._accelerator.device

    def _log_and_reset_metrics(self, step: Optional[int] = None):
        log_dict = {}
        for metric in itertools.chain(self.train_metrics, self.eval_metrics):
            log_dict.update(**metric.compute())
            metric.reset()

        self._accelerator.log(log_dict, step=step, log_kwargs={"wandb": {"commit": True}})

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.train()
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
        [metric.update(step_data) for metric in self.train_metrics]
        return loss.detach()

    @torch.no_grad
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        model_out = self.model(batch["input"])
        loss = self.loss_fn(model_out, batch["target"])
        step_data = StepData(batch, model_out, loss)
        [metric.update(step_data) for metric in self.eval_metrics]
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

        train_loss = RunningMean()
        eval_loss = RunningMean()

        for step in (pbar := trange(training_steps)):
            loss = self.train_step(next(iter_train))
            train_loss.add(loss)
            if step % eval_every_n_steps == 0:
                e_loss = self.eval_step(next(iter_eval))
                eval_loss.add(e_loss)
            if step % log_every_n_steps == 0:
                self.save(os.path.join(self._ckpt_dir, f"step-{step}"))
                self._log_and_reset_metrics(step=step)
            pbar.set_description(
                f"train_loss: {train_loss.val:.5f} (last: {train_loss.last:.5f}) "
                f"eval_loss: {eval_loss.val:.5f} (last: {eval_loss.last:.5f})"
            )


    def save(self, dir) -> None:
        os.makedirs(dir, exist_ok=True)

        torch.save(self.train_dataloader, os.path.join(dir, "train_dataloader.pt"))
        torch.save(self.model, os.path.join(dir, "model.pt"))
        torch.save(self.loss_fn, os.path.join(dir, "loss_fn.pt"))
        torch.save(self.optimizer, os.path.join(dir, "optimizer.pt"))
        torch.save(self.scheduler, os.path.join(dir, "scheduler.pt"))
        torch.save(self.eval_dataloader, os.path.join(dir, "eval_dataloader.pt"))
        torch.save(self.train_metrics, os.path.join(dir, "train_metrics.pt"))
        torch.save(self.eval_metrics, os.path.join(dir, "eval_metrics.pt"))
        self._accelerator.save_state(dir)


    @classmethod
    def load(cls: "Trainer", dir) -> "Trainer":
        train_dataloader = torch.load(os.path.join(dir, "train_dataloader.pt"))
        model = torch.load(os.path.join(dir, "model.pt"))
        loss_fn = torch.load(os.path.join(dir, "loss_fn.pt"))
        optimizer = torch.load(os.path.join(dir, "optimizer.pt"))
        scheduler = torch.load(os.path.join(dir, "scheduler.pt"))
        eval_dataloader = torch.load(os.path.join(dir, "eval_dataloader.pt"))
        train_metrics = torch.load(os.path.join(dir, "train_metrics.pt"))
        eval_metrics = torch.load(os.path.join(dir, "eval_metrics.pt"))

        config = {}
        log_with = []
        
        trainer =  cls(
            train_dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            eval_dataloader=eval_dataloader,
            config=config,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            log_with=log_with
        )
        trainer._accelerator.load_state(dir)
        return trainer


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
            [metric.update(step_data) for metric in self.train_metrics]
            if stop:
                # Manual break using debugger
                break

        self.scheduler.step()
        self._log_and_reset_metrics(epoch)

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
                [metric.update(step_data) for metric in self.eval_metrics]

        self._log_and_reset_metrics(epoch)
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
