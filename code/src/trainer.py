import datetime
import os
import itertools
import logging
from typing import Any, Dict, List, Optional

import torch
from torchvision.transforms import v2 as transforms
from accelerate import Accelerator
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from metrics.base_metric import BaseMetric, StepData
from losses import SummedLoss
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
    def __init__(
        self, alpha=0.99, start: Optional[torch.Tensor] = None
    ) -> None:
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
        log_with: List[str] = ["wandb"],
        clip_norm: float = 1e10,
        init_kwargs: Dict[str, Any] = {},
        data_transforms: Any = transforms.Identity(),
    ) -> None:
        train_metrics = [] if train_metrics is None else train_metrics
        eval_metrics = [] if eval_metrics is None else eval_metrics

        self._accelerator = Accelerator(
            log_with=log_with,
            # mixed_precision="fp16"
        )
        logging.info("Created accelerator")
        self._accelerator.init_trackers(
            project_name="MasterThesis",
            config=config,
            init_kwargs=init_kwargs,
        )
        logging.info("Initialised trackers")

        self._clip_norm = clip_norm

        if scheduler is None:
            # If no scheduler is given create a 'constant' scheduler
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, Constant(1.0)
            )

        if eval_dataloader is None:
            logging.warning(
                "No eval data provided, Using train data as eval data"
            )
            eval_dataloader = train_dataloader

            # Save based on start time of run
        self._ckpt_dir = os.path.join(
            config.get("paths", {}).get("checkpoints", "ckpts"),
            datetime.datetime.now().strftime("%Y/%m/%d/%H:%M:%S.%f"),
        )
        logging.info(f"Model ckpts will be saved in: {self._ckpt_dir}")

        if log_with is not None and "wandb" in log_with:
            import wandb

            wandb.watch(
                models=model,
                criterion=loss_fn,
                log="all",
                log_freq=100,
                log_graph=True,
            )
            wandb.log({"ckpt_dir": self._ckpt_dir}, commit=False)

        (
            model,
            loss_fn,
            optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            data_transforms,
        ) = self._accelerator.prepare(
            model,
            loss_fn,
            optimizer,
            train_dataloader,
            eval_dataloader,
            scheduler,
            data_transforms,
        )
        logging.info("Moved data to the GPU")

        self.train_dataloader = train_dataloader
        self.model = model
        self.loss_fn = loss_fn
        if isinstance(self.loss_fn, SummedLoss):
            self.loss_fn.add_log_callback(
                lambda name, val: self._accelerator.log(
                    {name: val}, log_kwargs={"wandb": {"commit": False}}
                )
            )

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.eval_dataloader = eval_dataloader
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.data_transforms = data_transforms
        self._gradient_penalty = GradientPenalty()

    @property
    def device(self):
        return self._accelerator.device

    def end_training(self):
        return self._accelerator.end_training()

    def _log_and_reset_metrics(self, step: Optional[int] = None):
        log_dict = {}
        for metric in itertools.chain(
            self.train_metrics, self.eval_metrics
        ):
            log_dict.update(**metric.compute())
            metric.reset()

        self._accelerator.log(
            log_dict, step=step, log_kwargs={"wandb": {"commit": True}}
        )

    def train_step(
        self, **batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        # Forward pass
        batch = self.data_transforms(batch)
        model_out = self.model(batch["input"])
        # Calculate Loss
        loss = self.loss_fn(model_out, batch)
        # Backward
        self._accelerator.backward(loss)
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self._clip_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self._accelerator.log(
            {"lr": self.scheduler.get_last_lr()[0]},
            log_kwargs={"wandb": {"commit": False}},
        )
        step_data = StepData(batch, model_out, loss)
        [metric.update(step_data) for metric in self.train_metrics]
        return loss.detach()

    @torch.no_grad
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        batch = self.data_transforms(batch)
        model_out = self.model(batch["input"])
        loss = self.loss_fn(model_out, batch)
        step_data = StepData(batch, model_out, loss)
        [metric.update(step_data) for metric in self.eval_metrics]
        return loss.detach()

    @torch.no_grad
    def full_eval(self, metric: BaseMetric) -> torch.Tensor:
        self.model.eval()
        metric = self._accelerator.prepare(metric)
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = self.data_transforms(batch)
            model_out = self.model(batch["input"])
            loss = self.loss_fn(model_out, batch)
            metric.update(
                StepData(batch=batch, model_out=model_out, loss=loss)
            )
        result = metric.compute()
        print(f"result: {result}")
        return result

    def steps(
        self,
        training_steps: int,
        eval_every_n_steps: int = 100,
        log_every_n_steps: Optional[int] = None,
        num_eval_steps: int = 10,
    ):
        """Run `n` training steps

        Args:
            training_steps (int): The total number of training steps to run
            eval_every_n_steps (int, optional): The amount of training steps for every eval step. Defaults to 100.
            log_every_n_steps (Optional[int], optional): _description_. Defaults to None.
        """
        log_every_n_steps = (
            eval_every_n_steps
            if log_every_n_steps is None
            else log_every_n_steps
        )
        iter_train = _Repeater(self.train_dataloader)
        iter_eval = _Repeater(self.eval_dataloader)

        train_loss = RunningMean()
        eval_loss = RunningMean()

        for step in (pbar := trange(training_steps, smoothing=0.01)):
            loss = self.train_step(**next(iter_train), step=step)
            train_loss.add(loss)
            if step % eval_every_n_steps == 0:
                e_loss = sum(
                    self.eval_step(next(iter_eval))
                    for i in range(num_eval_steps)
                )
                eval_loss.add(e_loss / num_eval_steps)
            if step % log_every_n_steps == 0:
                self.save(os.path.join(self._ckpt_dir, f"{step % 5}"))
                self._log_and_reset_metrics(step=step)
            pbar.set_description(
                f"train_loss: {train_loss.val:.5f} (last: {train_loss.last:.5f}) "
                f"eval_loss: {eval_loss.val:.5f} (last: {eval_loss.last:.5f})"
            )
        return eval_loss.val

    def save(self, dir) -> None:
        os.makedirs(dir, exist_ok=True)
        # Currently only saving model to save storage space
        torch.save(
            self.model.state_dict(), os.path.join(dir, "model.pt")
        )
        # torch.save(
        #     self.train_dataloader,
        #     os.path.join(dir, "train_dataloader.pt"),
        # )
        # torch.save(
        #     self.loss_fn.state_dict(), os.path.join(dir, "loss_fn.pt")
        # )
        # torch.save(
        #     self.optimizer.state_dict(),
        #     os.path.join(dir, "optimizer.pt"),
        # )
        # torch.save(
        #     self.scheduler.state_dict(),
        #     os.path.join(dir, "scheduler.pt"),
        # )
        # torch.save(
        # self.eval_dataloader,
        # os.path.join(dir, "eval_dataloader.pt"),
        # )
        # torch.save(
        #     self.train_metrics, os.path.join(dir, "train_metrics.pt")
        # )
        # torch.save(
        #     self.eval_metrics, os.path.join(dir, "eval_metrics.pt")
        # )
        # self._accelerator.save_state(dir)

    @classmethod
    def load(cls: "Trainer", dir) -> "Trainer":
        train_dataloader = torch.load(
            os.path.join(dir, "train_dataloader.pt")
        )
        model = torch.load(os.path.join(dir, "model.pt"))
        loss_fn = torch.load(os.path.join(dir, "loss_fn.pt"))
        optimizer = torch.load(os.path.join(dir, "optimizer.pt"))
        scheduler = torch.load(os.path.join(dir, "scheduler.pt"))
        eval_dataloader = torch.load(
            os.path.join(dir, "eval_dataloader.pt")
        )
        train_metrics = torch.load(
            os.path.join(dir, "train_metrics.pt")
        )
        eval_metrics = torch.load(os.path.join(dir, "eval_metrics.pt"))

        config = {}
        log_with = []

        trainer = cls(
            train_dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            eval_dataloader=eval_dataloader,
            config=config,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            log_with=log_with,
        )
        # trainer._accelerator.load_state(dir)
        return trainer

    def epoch(self, epoch: Optional[int] = None) -> torch.Tensor:
        loss_sum = 0
        loss_count = 0
        epoch = epoch if epoch is not None else 0
        step_offset = epoch * len(self.train_dataloader)
        self.model.train()

        stop = False
        pbar = tqdm(self.train_dataloader, leave=False, desc="training")
        for batch_idx, batch in enumerate(pbar):
            loss = self.train_step(
                **batch, step=batch_idx + step_offset
            )

            # Keep track of average loss
            loss_d = loss.detach()
            loss_sum += loss_d.sum()
            loss_count += batch["input"].shape[0]
            pbar.set_description(
                f"Training: loss={(loss_sum / loss_count).item():.4f}"
            )

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
                tqdm(
                    self.eval_dataloader, leave=False, desc="evaluation"
                )
            ):
                loss = self.eval_step(batch)

                # Keep track of average loss
                loss_sum += loss.detach().sum()
                loss_count += batch["input"].shape[0]

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
