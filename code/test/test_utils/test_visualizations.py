from models import VariationalUNet
from utils.visualize_feature_maps import (
    visualize_filters,
    visualize_filters_batched,
)


def test_visualize_filters(device, tmp_path):
    model = VariationalUNet(
        3,
        3,
        encoder_weights="imagenet",
        encoder_name="resnet50",
        skip_connections=[False] * 5,
        variational_skip_connections=[False] * 5,
    )

    visualize_filters(model, tmp_path, steps=1, device=device)


def test_visualize_filters_batch(tmp_path, device):
    model = VariationalUNet(
        3,
        3,
        encoder_weights="imagenet",
        encoder_name="resnet50",
        skip_connections=[False] * 5,
        variational_skip_connections=[False] * 5,
    )

    visualize_filters_batched(model, dir=tmp_path, steps=1, device=device)
