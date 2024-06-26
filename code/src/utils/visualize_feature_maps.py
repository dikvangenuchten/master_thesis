import math
import os
from typing import List
import torch
from torchvision import utils
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def _visualize_3d_tensor(
    input_img, tensor: torch.Tensor, title: str
) -> plt.Figure:
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(np.asarray(F.to_pil_image(input_img)))
    axs[0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    num_channels = tensor.shape[0]
    nrow = math.ceil(math.sqrt(num_channels))
    grid = utils.make_grid(tensor.unsqueeze(1), nrow=nrow).cpu()
    axs[1].imshow(np.asarray(F.to_pil_image(grid)))
    axs[1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.suptitle(title)
    return fig


def _get_stat_string(tensor) -> str:
    min = float(tensor.min())
    max = float(tensor.max())
    mean = float(tensor.mean())
    var = float(tensor.var())
    return f"{min=:.2f}, {max=:.2f}, {mean=:.2f}, {var=:.2f}"


@torch.no_grad()
def visualize_posteriors(model, batch, dir: str) -> List[plt.Figure]:
    """Uses the first image of the batch to visualize the posteriors of the network"""
    out = model(batch["input"][0:1])
    posteriors = out["posteriors"]
    all_figs = []
    os.makedirs(dir, exist_ok=True)
    for i, posterior in enumerate(
        tqdm.tqdm(
            posteriors, leave=False, desc="Visualizing posteriors"
        )
    ):
        # Create the visualization of the mean of the posterior
        shape = posterior.loc.shape
        tensor = posterior.mean[0]
        title = "\n".join(
            [
                f"Mean of var-layer {i}: {tuple(shape[1:3])}",
                _get_stat_string(tensor),
            ]
        )
        mean_activations = _visualize_3d_tensor(
            batch["input"][0], tensor, title
        )
        mean_activations.savefig(
            f"{dir}/{title.splitlines()[0]}.png", bbox_inches="tight"
        )
        all_figs.append(mean_activations)

        # Create the visualization of the variance of the posterior
        tensor = posterior.scale[0]
        title = "\n".join(
            [
                f"Std of var-layer {i}: {tuple(shape[1:3])}",
                _get_stat_string(tensor),
            ]
        )
        std_activations = _visualize_3d_tensor(
            batch["input"][0], tensor, title
        )
        std_activations.savefig(
            f"{dir}/{title.splitlines()[0]}.png", bbox_inches="tight"
        )
        all_figs.append(std_activations)
    return all_figs


@torch.no_grad()
def visualize_encoder_features(model, batch, dir: str):
    input = model.prepare_input(batch["input"])
    features = model.encoder(input)
    os.makedirs(dir, exist_ok=True)
    all_figs = []
    for i, feature in enumerate(features):
        shape = feature.shape
        tensor = feature[0]
        title = "\n".join(
            [
                f"Feature activations {i}: {tuple(shape[1:3])}",
                _get_stat_string(tensor),
            ]
        )
        mean_activations = _visualize_3d_tensor(
            batch["input"][0], tensor, title
        )
        mean_activations.savefig(
            f"{dir}/{title.splitlines()[0]}.png", bbox_inches="tight"
        )
        all_figs.append(mean_activations)
    return all_figs
