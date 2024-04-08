import os

import pytest


@pytest.fixture(scope="session")
def test_image(path="test/data/example.jpg"):
    from torchvision.tv_tensors import Image
    from PIL import Image as PImage

    return Image(PImage.open(path))


@pytest.fixture(scope="session")
def test_image_batch(test_image):
    import torch

    return torch.reshape(test_image, [1, *test_image.shape]).to(
        torch.float32
    )


@pytest.fixture(params=[True, False])
def true_or_false(request):
    return request.param


@pytest.fixture(scope="module")
def bs_model():
    from models.binary_segmentation_model import BinarySegmentationModel

    return BinarySegmentationModel()


@pytest.fixture(autouse=True)
def set_wandb_to_offline():
    os.environ["WANDB_MODE"] = "offline"
