import os

import pytest


@pytest.fixture(scope="session")
def image(path="test/data/example.jpg"):
    from torchvision.tv_tensors import Image
    from PIL import Image as PImage

    return Image(PImage.open(path))


@pytest.fixture(scope="session")
def image_batch(image):
    import torch

    return torch.reshape(image, [1, *image.shape]).to(torch.float32)


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


@pytest.fixture(scope="session")
def dataset(image):
    import torch
    from torchvision.transforms import v2 as transforms
    from datasets import CoCoDataset

    image_net_transforms = [
        # Rescale to [0, 1], then normalize using mean and std of ImageNet1K DS
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]

    data_transforms = transforms.Compose(
        [transforms.Resize((64, 64)), *image_net_transforms]
    )

    ds = CoCoDataset(
        split="val",
        output_structure={"input": "img", "target": "semantic_mask"},
        dataset_root="test/data",
        transform=data_transforms,
    )

    return ds


@pytest.fixture(scope="session")
def dataloader(dataset):
    from torch.utils.data.dataloader import DataLoader

    return DataLoader(dataset, batch_size=4)


@pytest.fixture()
def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
