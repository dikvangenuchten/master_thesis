"""Entrypoint for multirun

This is a minimal import such it loads quicker on the cluster
"""

import logging
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        from hydra_main import main
        return main(cfg)
    except Exception as exception:
        # Hydra catches and ignores any raised exceptions
        logging.error(exception)
        raise

if __name__ == "__main__":
    main()