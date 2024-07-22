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
    try:
        logging.warning("Started in entrypoint")
        import wandb
        logging.warning("Imported wandb")
        wandb.sdk.wandb_init._set_logger(logging.getLogger())
        logging.warning("set wandb logger")
        wandb.init("MasterThesis-debug")
        logging.warning("Initialized wandb")
    except Exception as e:
        logging.exception("Got exception", exc_info=e)


if __name__ == "__main__":
    logging.warning("Reached main")
    hydra_entrypoint()
