import time
import pytest
import torch
from torchvision.tv_tensors import Image

from datasets.toy_data import (
    SegmentationToyDataset,
    ToySegmentationTransform,
)
from datasets.coco import CoCoDataset
import timeit


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


@pytest.mark.usefixtures("instant_sleep")
def test_caching(monkeypatch, tmp_path):
    sleep_time = 1
    old_open = open
    slow_calls = [0]  # Use a list to pass by reference

    def slow_open(file, *args, _hidden=slow_calls, **kwargs):
        # Pretend to be a slow disk
        # With the 'instant_sleep' fixture this manipulates the
        # builtin time function, and does not actually sleep
        if not file.startswith(str(tmp_path)):
            _hidden[0] += 1
            time.sleep(sleep_time)
        return old_open(file, *args, **kwargs)

    # This will redirect all calls from open to `slow_open`
    monkeypatch.setitem(__builtins__, "open", slow_open)

    cache_dir = tmp_path / "cache"
    ds = CoCoDataset(
        output_structure={"img": "img", "target": "semantic_mask"},
        dataset_root="test/data",
        cache_dir=cache_dir,
    )
    start_calls = slow_calls[0]
    start = time.time()
    batch_1 = ds[0]
    mid_calls = slow_calls[0]
    mid = time.time()
    batch_2 = ds[0]
    end = time.time()
    end_calls = slow_calls[0]

    assert torch.equal(batch_1["img"], batch_2["img"])
    assert torch.equal(batch_1["target"], batch_2["target"])

    assert (mid - start) > sleep_time
    assert (
        start_calls < mid_calls
    ), "During generating the batch, no calls to open were made."
    assert (mid - start) > (end - mid)
    assert (
        mid_calls == end_calls
    ), "A call to open was made after loading the batch for the second time"


@pytest.mark.usefixtures("instant_sleep")
def test_in_memory_caching(monkeypatch):
    sleep_time = 1
    old_open = open

    def slow_open(*args, **kwargs):
        # Pretend to be a slow disk
        # With the 'instant_sleep' fixture this manipulates the
        # builtin time function, and does not actually sleep
        time.sleep(sleep_time)
        return old_open(*args, **kwargs)

    # This will redirect all calls from open to `slow_open`
    monkeypatch.setitem(__builtins__, "open", slow_open)

    cache_dir = "in_memory"
    ds = CoCoDataset(
        output_structure={"img": "img", "target": "semantic_mask"},
        dataset_root="test/data",
        cache_dir=cache_dir,
    )
    start = time.time()
    batch_1 = ds[0]
    mid = time.time()
    batch_2 = ds[0]
    end = time.time()
    batch_3 = ds[0]

    assert torch.equal(batch_1["img"], batch_2["img"])
    assert torch.equal(batch_1["target"], batch_2["target"])
    assert torch.equal(batch_3["img"], batch_2["img"])
    assert torch.equal(batch_3["target"], batch_2["target"])

    # In memory should be much faster then not in memory
    assert (mid - start) > sleep_time
    assert (mid - start) > (end - mid)


def test_percentage():
    ds_full = CoCoDataset(
        split="train",
        output_structure={"img": "img", "target": "semantic_mask"},
        dataset_root="test/data",
    )

    ds_partial = CoCoDataset(
        split="train",
        output_structure={"img": "img", "target": "semantic_mask"},
        dataset_root="test/data",
        percentage=1 / 16,  # This should result in a length 1 dataset
    )

    assert len(ds_full) > len(ds_partial)
    assert len(ds_full) == 16
    assert len(ds_partial) == 1

    full_data = list(iter(ds_full))
    partial_data = list(iter(ds_partial))

    assert len(full_data) > len(partial_data)
    assert len(full_data) == 16
    assert len(partial_data) == 1
