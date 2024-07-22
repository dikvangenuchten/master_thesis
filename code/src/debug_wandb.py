"""Minimal working example to debug wandb on the cluster
"""

import logging
logging.info("Imported: logging")
import hydra
logging.info("Imported: hydra")
from omegaconf import DictConfig
logging.info("Imported: DictConfig")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_entrypoint(cfg: DictConfig) -> None:
    logging.warning("Started in entrypoint")
    import wandb
    logging.warning("Imported wandb")
    wandb.init(project_name="MasterThesis-debug")
    logging.info("Initialized wandb")


if __name__ == "__main__":
    logging.warning("Reached main")
    hydra_entrypoint()
