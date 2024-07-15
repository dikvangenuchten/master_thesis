import copy
import math
import os
from typing import List
import torch
from torch import nn

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
    model: torch.nn.Module,
    dir: str,
    steps=10000,
    lr=1e-1,
    device="cuda",
    layers=list(range(4)),
):
    """Visualize the filters by optimizing the input, such that the filter output"""

    for layer in layers:
        fe = make_feature_extractor(model, layer, device=device)
        out = fe(torch.rand((1, 3, 128, 128), device=device))["out"]

        canvas_history = []
        num_filters = out.shape[1]
        rand_input = torch.rand((num_filters, 3, 32, 32), device=device)
        out = fe(rand_input)["out"]

        for filter in tqdm.trange(
            num_filters, desc=f"Processing layer: {layer + 1}"
        ):
            canvas = torch.rand(
                (1, 3, 128, 128), device=device, requires_grad=True
            )
            optimizer = torch.optim.Adam(
                [canvas], lr=lr, weight_decay=1e-6
            )
            for i in (pbar := tqdm.trange(steps, leave=False)):
                optimizer.zero_grad()
                out = fe(canvas)["out"][:, filter]
                loss = -torch.mean(out)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"{i} {loss} {canvas[0, 0, 0, 0]}")
            canvas_history.append(canvas.mean(0).detach().cpu().numpy())

        utils.save_image(
            torch.tensor(canvas_history).squeeze(),
            f"{dir}/filter_l{layer}.png",
            normalize=True,
            nrow=32,
        )


def visualize_filters_batched(
    model: torch.nn.Module,
    dir: str,
    steps=1000,
    lr=1e-0,
    device="cuda",
    layers=list(range(4)),
):
    for layer in layers:
        fe = make_feature_extractor(model, layer, device=device)
        canvas = visualize_layer(fe, device=device, lr=lr, steps=steps)
        utils.save_image(
            canvas.squeeze().clone().detach(),
            f"{dir}/filter_l{layer}.png",
            normalize=True,
            scale_each=True,
            nrow=32,
        )
        pass


def visualize_layer(
    fe: nn.Module, lr: float, steps: int, device="cuda"
) -> torch.Tensor:
    out = fe(torch.rand((1, 3, 128, 128), device=device))["out"]
    num_filters = out.shape[1]

    canvas = torch.tensor(
        np.random.normal(0.5, 0.1, (num_filters, 3, 32, 32)),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    # optimizer = torch.optim.Adam([canvas], lr=lr, weight_decay=1e-5)
    for i in (pbar := tqdm.trange(steps, leave=False)):
        canvas, loss = gradient_ascent_step(fe, canvas, lr=lr)
        # optimizer.zero_grad()
        # out = fe(canvas)["out"]
        # Ensure each output matches exactly one filter
        # eye = torch.eye(num_filters, device=device).reshape(
        # num_filters, num_filters, 1, 1
        # )
        # out_ = eye * out
        # loss = -torch.mean(out_, (1, 2, 3)).sum()
        # loss.backward()
        # optimizer.step()
        with torch.no_grad():
            # canvas.clamp_(-2, 2)
            mean = float(canvas.numpy(force=True).mean())
            min = float(canvas.numpy(force=True).min())
            std = float(canvas.numpy(force=True).std())
        pbar.set_description(
            f"{i} {loss=:.2f} {canvas[0, 0, 0, 0]} {mean=:.2f} {min=:.2f} {std=:.2f}"
        )
    return canvas.detach()


def gradient_ascent_step(fe, canvas, lr):
    out = fe(canvas)["out"]
    # Ensure each output matches exactly one filter
    eye = torch.eye(canvas.shape[0], device=canvas.device).reshape(
        canvas.shape[0], canvas.shape[0], 1, 1
    )
    out_ = eye * out
    loss = torch.mean(out_, (1, 2, 3)).sum()
    loss.backward()
    grad = nn.functional.normalize(canvas.grad, p=2)
    with torch.no_grad():
        canvas += grad * lr
        # canvas.clamp_(0, 1)
        canvas.grad.zero_()
    return canvas, loss.detach()


def make_feature_extractor(model, layer: int, device):
    model = copy.deepcopy(model).to(device)

    fe = create_feature_extractor(
        model.encoder.model, {f"layer{layer + 1}": "out"}
    )

    fe = fe.eval()
    return fe


def visualize_gradient_per_class(model, batch, dir: str):
    if hasattr(model, "prepare_input"):
        input = model.prepare_input(batch["input"])
    else:
        input = batch["input"]
    model.encoder(input)

    os.makedirs(dir, exist_ok=True)
