import os
import pytest
import torch

from torchvision.tv_tensors import Image
from PIL import Image as PImage

from model import BinarySegmentationModel


@pytest.fixture
def test_image(path="test/test_data/example.jpg"):
    return Image(PImage.open(path))


@pytest.fixture
def test_image_batch(test_image):
    return torch.reshape(test_image, [1, *test_image.shape]).to(torch.float32)


@pytest.fixture
def bs_model():
    return BinarySegmentationModel()


@pytest.fixture(autouse=True)
def set_wandb_to_offline():
    os.environ["WANDB_MODE"] = "offline"
