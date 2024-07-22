"""Minimal working example to debug wandb on the cluster
"""

import logging
logging.info("Imported: logging")
from accelerate import Accelerator
logging.info("Imported: Accelerator")
import hydra
logging.info("Imported: hydra")
from omegaconf import DictConfig
logging.info("Imported: DictConfig")


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
    logging.warning("Reached main")
    hydra_entrypoint()
