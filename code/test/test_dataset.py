import torch
from torchvision.tv_tensors import Image

from datasets.toy_data import (
    SegmentationToyDataset,
    ToySegmentationTransform,
)
from datasets.coco import CoCoDataset


def test_toy_dataset_initialization():
    dataset = SegmentationToyDataset(base_path="test/data/")
    assert dataset is not None


def test_toy_transform(image: Image):
    transform = ToySegmentationTransform()
    img, mask = transform(image)
    assert (
        img.shape[1:] == mask.shape[1:]
    ), "Mask does not have the same W,H dimensions"
    assert (
        torch.where(mask == 0, img, 0)
        == torch.where(mask == 0, image, 0)
    ).all(), "The image was modified outside of the mask"
    assert (
        torch.where(mask == 1, img[0:1], 255) == 255
    ).all(), "The red blob was not present"

    # For visualization uncommenct below:
    # import os
    # from torchvision.transforms.v2 import ToPILImage
    # pillify = ToPILImage()
    # path = "dbg"
    # os.makedirs(path, exist_ok=True)
    # pillify(img).save(os.path.join(path, "img.png"))
    # pillify(mask + 0).save(os.path.join(path, "msk.png"))


def test_coco_dataset():
    ds = CoCoDataset(
        output_structure={"img": "img", "target": "semantic_mask"},
        dataset_root="test/data",
    )
    # TODO add instance mask
    batch = ds[0]
    img = batch["img"]
    semantic_mask = batch["target"]

    assert (
        img.shape[1:] == semantic_mask.shape
    ), "Semantic and image shape do not match."


def test_caching(tmp_path):
    # This test is terrible, but stepping through the code shows it works
    cache_dir = tmp_path / "cache"
    ds = CoCoDataset(
        output_structure={"img": "img", "target": "semantic_mask"},
        dataset_root="test/data",
        cache_dir=cache_dir,
    )
    batch_1 = ds[0]
    batch_2 = ds[0]
    assert torch.equal(batch_1["img"], batch_2["img"])
    assert torch.equal(batch_1["target"], batch_2["target"])
