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
    return torch.reshape(test_image, [1, *test_image.shape])


@pytest.fixture
def bs_model():
    return BinarySegmentationModel()