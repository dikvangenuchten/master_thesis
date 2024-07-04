from models import VariationalUNet
from utils.visualize_feature_maps import visualize_filters


def test_visualize_filters():
    model = VariationalUNet(
        3,
        3,
        encoder_weights="imagenet",
        encoder_name="resnet50",
        skip_connections=[False] * 5,
        variational_skip_connections=[False] * 5,
    )

    visualize_filters(model, steps=1)
