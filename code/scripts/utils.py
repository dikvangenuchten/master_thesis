import sys
from pathlib import Path
from typing import Dict, Optional

import torchinfo
import hydra


def fix_path():
    # Nasty hack to add src directory
    sys.path.insert(1, str(Path(__file__).parents[1] / "src"))
    # I am sorry


fix_path()  # noqa


def instantiate_dict(config, *args, **kwargs) -> object:
    return hydra.utils.instantiate(config, *args, **kwargs)


def get_model_summary(model, input_shape=(3, 128, 128)):
    return torchinfo.summary(model, (1, *input_shape))


def get_model_from_config(name: str, params: Dict):
    with hydra.initialize(
        version_base=None,
        config_path="src/conf/models",
    ):
        cfg = hydra.compose(config_name=name)
