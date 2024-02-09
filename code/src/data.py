import os
from typing import Any, Optional
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.tv_tensors import Mask


def binary_segmentation_dataloader():

    pass


class SegmentationToyDataset(data.Dataset):
    """A toy dataset for image segmentation"""

    def __init__(self, background_dataset: Optional[data.Dataset] = None, limit: int=100):
        if background_dataset is None:
            background_dataset = self.default_background_dataset()
        self._background_dataset = datasets.wrap_dataset_for_transforms_v2(
            background_dataset, # target_keys=["masks", "labels"]
        )
        self._limit = limit
        
        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                ToySegmentationTransform(),
                transforms.ToDtype(torch.float32),
            ]
        )

    def __getitem__(self, index) -> Any:
        img, _ = self._background_dataset[index]

        img, mask = self.transforms(img)
        mask = mask.to(dtype=torch.float32) / 255
        return img, mask

    def __len__(self) -> int:
        return min(self._limit, len(self._background_dataset))

    @staticmethod
    def default_background_dataset():
        base_path = "/datasets/coco/"
        split = "val"
        return datasets.CocoDetection(
            os.path.join(base_path, f"{split}2017/"),            
            os.path.join(base_path, f"annotations/instances_{split}2017.json"),
        )

    def to_loader(self, *args, **kwargs) -> data.DataLoader:
        return data.DataLoader(self, *args, **kwargs)


class ToySegmentationTransform(nn.Module):
    """Augment an image with a blob ontop of it.

    Returns the modified image and the mask of the blob.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x_size = 10
        self._y_size = 10

    def forward(self, img):
        mask = Mask(torch.zeros_like(img[0:1]))

        _, w, h = mask.shape
        x = torch.randint(self._x_size, w - self._x_size, [1])
        y = torch.randint(self._y_size, h - self._y_size, [1])

        img[
            0,
            x - self._x_size : x + self._x_size,
            y - self._y_size : y + self._y_size
        ] = 255
        mask[
            0,
            x - self._x_size : x + self._x_size,
            y - self._y_size : y + self._y_size
        ] = 255

        return img, mask


if __name__ == "__main__":
    SegmentationToyDataset()
