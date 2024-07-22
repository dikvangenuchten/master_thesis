"""Minimal working example to debug wandb on the cluster
"""

import logging
from accelerate import Accelerator
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_entrypoint(cfg: DictConfig) -> None:
    logging.warning("Started in logger")
    log_with = "wandb"

    accelerator = Accelerator(
        log_with=log_with,
    )
    logging.info("Created accelerator")
    accelerator.init_trackers(
        project_name="MasterThesis-debug",
    )
    logging.info("Initialized trackers")


if __name__ == "__main__":
    hydra_entrypoint()
