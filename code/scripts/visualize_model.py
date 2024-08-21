from torchviz import make_dot

import torch
from torch import nn
from models import VarUNet, UNet


def load_unet_model() -> nn.Module:
    model = UNet(3, 3, encoder_name="resnet18")
    return model


def load_custom_model() -> nn.Module:
    model = VarUNet(3, 3, encoder_name="resnet18")
    return model


def visualize(model: nn.Module, name: str):
    X = torch.ones(1, 3, 64, 64)
    y = model(X)["out"]
    dot = make_dot(y.mean(), params=dict(model.named_parameters()))
    dot.render(f"{name}_viz", format="png")
    state_dict = model.state_dict()
    with open(f"{name}_layers", mode="w") as f:
        f.write("\n".join(sorted(list(state_dict.keys()))))


def main():
    unet = load_unet_model()
    visualize(unet, "unet")

    custom = load_custom_model()
    visualize(custom, "custom")


if __name__ == "__main__":
    main()
