import logging
import torch


def load_state_dict(path: str):
    """Loads a model from the `models` dir in this repo

    Args:
        path (str): A path starting with: `models/...`
    """
    try:
        return torch.load(path)
    except FileNotFoundError:
        logging.warning(
            f"Could not find state dict in: {path}. Trying parent directory"
        )
        state_dict = torch.load("../" + path)
        logging.warning("Found state_dict in parent directory")
        return state_dict


def extract_encoder(state_dict: dict, prefix: str = "encoder."):
    """Extracts the encoder weights from

    Args:
        state_dict (dict): A state_dict from the VAES
        prefix (str, optional): _description_. Defaults to "encoder.".

    Returns:
        dict: The state dict of the encoder
    """
    state_dict = {k: v for k, v in state_dict.items() if prefix in k}
    encoder_state_dict = {
        k.lstrip(prefix): v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    return encoder_state_dict


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    for layer in model.parameters():
        layer.requires_grad = False
    return model
