import copy
import math
import os
from typing import List
import torch
from accelerate import Accelerator
import torch.random
from torchvision import utils
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)
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
    if hasattr(model, "prepare_input"):
        input = model.prepare_input(batch["input"])
    else:
        input = batch["input"]
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


def visualize_filters(
    model: torch.nn.Module, steps=1000, lr=1e-1, device="cuda"
):
    """Visualize the filters by optimizing the input, such that the filter output"""
    model = copy.deepcopy(model).to(device)

    def simple_forward(x):
        # nonlocal model
        features = model.encoder.model(x)
        features = [x] + features
        return features

    model.encoder.forward = simple_forward

    for layer in range(4):
        fe = create_feature_extractor(
            model, {f"encoder.model.layer{layer + 1}": "out"}
        )

        fe = fe.eval()
        out = fe(torch.rand((1, 3, 256, 256), device=device))["out"]

        canvas_history = []
        num_filters = out.shape[1]

        for filter in range(num_filters):
            canvas = torch.rand(
                (1, 3, 128, 128), device=device, requires_grad=True
            )
            optimizer = torch.optim.Adam(
                [canvas], lr=lr, weight_decay=1e-6
            )
            for i in (pbar := tqdm.trange(steps)):
                optimizer.zero_grad()
                out = fe(canvas)["out"][:, filter]
                loss = -torch.mean(out)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"{i} {loss} {canvas[0, 0, 0, 0]}")
            canvas_history.append(canvas.detach().cpu().numpy())

        utils.save_image(torch.tensor(canvas_history).squeeze(), f"filter_l{layer}.png", normalize=True)
    model


def visualize_gradient_per_class(model, batch, dir: str):
    if hasattr(model, "prepare_input"):
        input = model.prepare_input(batch["input"])
    else:
        input = batch["input"]
    features = model.encoder(input)

    os.makedirs(dir, exist_ok=True)
    all_figs = []
